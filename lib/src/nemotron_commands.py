#!/usr/bin/env python3
"""Manage the optional local Nemotron 3.5 streaming backend."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Iterable, Optional

MODEL_ID = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4"
MODEL_REVISION = "8364d9e2dd9da23789b480bdbba9e423717e42ee"
ORT_GENAI_VERSION = "0.14.1"
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SAMPLES = 8960
BACKEND_NAME = "nemotron-streaming"
SCHEMA_URL = (
    "https://raw.githubusercontent.com/goodroot/hyprwhspr/main/"
    "share/nemotron-config.schema.json"
)


def _xdg_path(env_name: str, fallback: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser() if value else fallback


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_file() -> Path:
    base = _xdg_path("XDG_CONFIG_HOME", Path.home() / ".config")
    return base / "hyprwhspr" / "config.json"


def venv_dir() -> Path:
    base = _xdg_path(
        "XDG_DATA_HOME", Path.home() / ".local" / "share"
    )
    return base / "hyprwhspr" / "venv"


def venv_python() -> Path:
    return venv_dir() / "bin" / "python"


def _run(
    command: Iterable[object],
    *,
    check: bool = True,
    capture_output: bool = False,
):
    return subprocess.run(
        [str(part) for part in command],
        check=check,
        text=True,
        capture_output=capture_output,
    )


def resolve_device(device: str) -> str:
    normalized = str(device).lower()
    if normalized in {"cpu", "cuda"}:
        return normalized
    if normalized != "auto":
        raise ValueError("device must be auto, cpu, or cuda")

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return "cpu"
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return "cpu"
    return (
        "cuda"
        if result.returncode == 0 and "GPU" in result.stdout
        else "cpu"
    )


def selected_runtime_package(device: str) -> str:
    suffix = "-cuda" if resolve_device(device) == "cuda" else ""
    return f"onnxruntime-genai{suffix}=={ORT_GENAI_VERSION}"


def _installed_version(python: Path, package: str) -> Optional[str]:
    result = _run(
        [python, "-m", "pip", "show", package],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip() or None
    return None


def _restore_runtime(
    python: Path,
    selected_package: str,
    selected_version: Optional[str],
    opposite_package: str,
    opposite_version: Optional[str],
) -> None:
    pip = [python, "-m", "pip"]
    _run([*pip, "uninstall", "-y", selected_package], check=False)
    if selected_version:
        _run(
            [*pip, "install", f"{selected_package}=={selected_version}"],
            check=False,
        )
    elif opposite_version:
        _run(
            [*pip, "install", f"{opposite_package}=={opposite_version}"],
            check=False,
        )


def ensure_venv() -> Path:
    python = venv_python()
    if not python.is_file():
        raise RuntimeError(
            "hyprwhspr virtual environment is missing. Run the normal "
            "'hyprwhspr setup' first, then run this command again."
        )
    return python


def install_dependencies(
    device: str,
) -> tuple[str, str, Optional[str], str, Optional[str]]:
    python = ensure_venv()
    resolved = resolve_device(device)
    selected = selected_runtime_package(resolved)
    selected_name = selected.split("==", 1)[0]
    opposite = (
        "onnxruntime-genai-cuda"
        if resolved == "cpu"
        else "onnxruntime-genai"
    )
    selected_version = _installed_version(python, selected_name)
    opposite_version = _installed_version(python, opposite)
    requirements = repo_root() / "requirements-nemotron.txt"
    if not requirements.is_file():
        raise RuntimeError(f"missing optional requirements file: {requirements}")

    pip = [python, "-m", "pip"]
    _run([*pip, "uninstall", "-y", opposite], check=False)
    try:
        _run([*pip, "install", selected, "-r", requirements])
    except Exception:
        _restore_runtime(
            python, selected_name, selected_version,
            opposite, opposite_version,
        )
        raise
    return (
        resolved, selected_name, selected_version,
        opposite, opposite_version,
    )


def _snapshot_script(model_id: str, revision: str, download: bool) -> str:
    return (
        "from huggingface_hub import snapshot_download; "
        "print(snapshot_download("
        f"repo_id={model_id!r}, revision={revision!r}, "
        f"local_files_only={not download!r}))"
    )


def resolve_model_path(
    *,
    download: bool,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
) -> Path:
    local_path = Path(str(model_id)).expanduser()
    if local_path.is_dir():
        return local_path

    result = _run(
        [ensure_venv(), "-c", _snapshot_script(model_id, revision, download)],
        capture_output=True,
    )
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("model snapshot command returned no path")
    path = Path(lines[-1])
    if not path.is_dir():
        raise RuntimeError(f"model snapshot was not resolved: {path}")
    return path


def validate_model_config(path: Path) -> dict:
    config_path = path / "genai_config.json"
    if not config_path.is_file():
        raise ValueError(f"missing {config_path}")
    model = json.loads(config_path.read_text(encoding="utf-8")).get("model") or {}

    model_type = model.get("type")
    if model_type not in (None, "nemotron_speech"):
        raise ValueError(
            f"unexpected model type {model_type!r}; expected 'nemotron_speech'"
        )
    sample_rate = int(model.get("sample_rate", 0))
    chunk_samples = int(model.get("chunk_samples", 0))
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"unexpected sample_rate={sample_rate}; expected {EXPECTED_SAMPLE_RATE}"
        )
    if chunk_samples != EXPECTED_CHUNK_SAMPLES:
        raise ValueError(
            f"unexpected chunk_samples={chunk_samples}; "
            f"expected {EXPECTED_CHUNK_SAMPLES} (560 ms)"
        )
    return model


def _runtime_validation_script(model_path: Path, device: str) -> str:
    return (
        "import onnxruntime_genai as og; "
        f"config=og.Config({str(model_path)!r}); "
        "config.clear_providers(); "
        + (
            "config.append_provider('cuda'); "
            if device == "cuda"
            else ""
        )
        + "model=og.Model(config); print('ok')"
    )


def validate_runtime_load(model_path: Path, device: str) -> None:
    resolved = resolve_device(device)
    result = _run(
        [
            ensure_venv(),
            "-c",
            _runtime_validation_script(model_path, resolved),
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"ONNX Runtime could not load Nemotron on {resolved}: "
            f"{detail or 'unknown error'}"
        )


def update_config(
    path: Path,
    *,
    device: str,
    language: Optional[str],
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
) -> Optional[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    backup = None
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        backup = path.with_name(
            f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.copy2(path, backup)

    data["$schema"] = SCHEMA_URL
    data.update(
        {
            "transcription_backend": BACKEND_NAME,
            "nemotron_streaming_model": model_id,
            "nemotron_streaming_revision": revision,
            "nemotron_streaming_device": device,
            "nemotron_streaming_buffer_max_seconds": 3.0,
            "nemotron_streaming_finalize_timeout": 5.0,
            "nemotron_streaming_use_vad": False,
        }
    )
    if language is not None:
        data["language"] = language

    if data.get("websocket_provider") == "nemotron-local":
        data.pop("websocket_provider", None)
        data.pop("websocket_model", None)
        data.pop("realtime_mode", None)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return backup


def _read_config() -> dict:
    path = config_file()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _configured_model() -> tuple[str, str]:
    config = _read_config()
    return (
        str(config.get("nemotron_streaming_model", MODEL_ID)),
        str(config.get("nemotron_streaming_revision", MODEL_REVISION)),
    )


def _service_active() -> bool:
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False
    result = subprocess.run(
        [systemctl, "--user", "is-active", "hyprwhspr.service"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _restart_service() -> bool:
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False
    return subprocess.run(
        [systemctl, "--user", "restart", "hyprwhspr.service"],
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0


def dependency_status() -> tuple[bool, str]:
    python = venv_python()
    if not python.is_file():
        return False, "hyprwhspr venv missing"
    result = subprocess.run(
        [
            python,
            "-c",
            "import onnxruntime_genai, huggingface_hub, soxr, numpy; "
            "print(getattr(onnxruntime_genai, '__version__', 'installed'))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or "Nemotron dependencies missing"
    return True, result.stdout.strip()


def command_setup(args) -> int:
    runtime_change = None
    activated = False
    try:
        runtime_change = install_dependencies(args.device)
        (
            device, selected_name, selected_version,
            opposite, opposite_version,
        ) = runtime_change
        print(f"Runtime installed for: {device}")
        model_path = resolve_model_path(download=True)
        validate_model_config(model_path)
        validate_runtime_load(model_path, device)
        print(f"Model cached and runtime-validated: {model_path}")

        backup = update_config(
            config_file(), device=device, language=args.language
        )
        activated = True
        print(f"Configuration updated: {config_file()}")
        if backup:
            print(f"Backup: {backup}")
        if not args.no_restart:
            print(
                "hyprwhspr.service restarted"
                if _restart_service()
                else "Service was not restarted; start it when ready."
            )
        return 0
    except Exception as exc:
        if runtime_change is not None and not activated:
            (
                device, selected_name, selected_version,
                opposite, opposite_version,
            ) = runtime_change
            _restore_runtime(
                ensure_venv(), selected_name, selected_version,
                opposite, opposite_version,
            )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def command_download(_args) -> int:
    try:
        model_id, revision = _configured_model()
        path = resolve_model_path(
            download=True, model_id=model_id, revision=revision
        )
        validate_model_config(path)
        print(path)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _model_status() -> tuple[bool, str]:
    try:
        model_id, revision = _configured_model()
        path = resolve_model_path(
            download=False, model_id=model_id, revision=revision
        )
        validate_model_config(path)
        return True, str(path)
    except Exception as exc:
        return False, str(exc)


def command_status(_args) -> int:
    config = _read_config()
    configured = config.get("transcription_backend") == BACKEND_NAME
    deps_ok, deps_note = dependency_status()
    model_ok, model_note = _model_status()
    print(f"Configured: {'yes' if configured else 'no'}")
    print(f"Dependencies: {'ok' if deps_ok else 'missing'} ({deps_note})")
    print(
        f"Model: {'cached' if model_ok else 'missing or invalid'} "
        f"({model_note})"
    )
    print(f"Service: {'active' if _service_active() else 'inactive'}")
    return 0 if configured and deps_ok and model_ok else 1


def command_validate(_args) -> int:
    deps_ok, deps_note = dependency_status()
    if not deps_ok:
        print(f"ERROR: {deps_note}", file=sys.stderr)
        return 1
    model_ok, model_note = _model_status()
    if not model_ok:
        print(f"ERROR: {model_note}", file=sys.stderr)
        return 1
    config = _read_config()
    device = str(config.get("nemotron_streaming_device", "cpu"))
    try:
        validate_runtime_load(Path(model_note), device)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Dependencies: ok ({deps_note})")
    print(f"Model: {model_note}")
    print(f"Runtime: load ok ({resolve_device(device)})")
    print("Streaming chunk: 560 ms")
    return 0


def _load_pcm16_wav(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if width != 2:
        raise ValueError("benchmark accepts 16-bit PCM WAV files")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.float32)
    if not audio.size:
        raise ValueError("benchmark WAV contains no audio")
    return audio, sample_rate


def _benchmark_worker(args) -> int:
    sys.path.insert(0, str(repo_root() / "lib" / "src"))
    from backends.nemotron_streaming_backend import NemotronStreamingBackend

    audio, sample_rate = _load_pcm16_wav(args.wav)
    model_id, revision = _configured_model()

    class Config:
        values = {
            "language": args.language,
            "recording_mode": "toggle",
            "nemotron_streaming_model": model_id,
            "nemotron_streaming_revision": revision,
            "nemotron_streaming_device": args.device,
            "nemotron_streaming_buffer_max_seconds": max(
                3.0, len(audio) / sample_rate + 1.0
            ),
            "nemotron_streaming_finalize_timeout": args.timeout,
            "nemotron_streaming_use_vad": args.use_vad,
        }

        def get_setting(self, key, default=None):
            return self.values.get(key, default)

    class Manager:
        def __init__(self):
            self.config = Config()
            self.temp_dir = None
            self.ready = False
            self.current_model = None
            self._last_use_time = 0.0
            self._streaming_partial_callback = None
            self._realtime_partial_callback = None

    backend = NemotronStreamingBackend(Manager())
    if not backend.initialize():
        return 1
    try:
        started = time.perf_counter()
        text = backend.transcribe(audio, sample_rate, args.language)
        elapsed = time.perf_counter() - started
        duration = len(audio) / sample_rate
        print(text)
        print(
            f"audio={duration:.2f}s wall={elapsed:.2f}s "
            f"batch_rtf={elapsed / duration:.3f}"
        )
        return 0 if text else 1
    finally:
        backend.cleanup()


def command_benchmark(args) -> int:
    try:
        python = ensure_venv()
        if Path(sys.executable).resolve() != python.resolve():
            command = [
                python,
                Path(__file__).resolve(),
                "_benchmark-worker",
                args.wav,
                "--device",
                args.device,
                "--timeout",
                args.timeout,
            ]
            if args.language:
                command.extend(["--language", args.language])
            if args.use_vad:
                command.append("--use-vad")
            return subprocess.run(
                [str(part) for part in command], check=False
            ).returncode
        return _benchmark_worker(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyprwhspr nemotron",
        description="Manage the optional local Nemotron 3.5 streaming backend",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="install, download, and configure")
    setup.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    setup.add_argument(
        "--language",
        default=None,
        help="Language/locale to store; omitted preserves the current setting",
    )
    setup.add_argument("--no-restart", action="store_true")
    setup.set_defaults(handler=command_setup)

    download = subparsers.add_parser("download", help="download the configured model")
    download.set_defaults(handler=command_download)

    status = subparsers.add_parser("status", help="show configuration and cache status")
    status.set_defaults(handler=command_status)

    validate = subparsers.add_parser(
        "validate", help="validate dependencies and model metadata"
    )
    validate.set_defaults(handler=command_validate)

    benchmark = subparsers.add_parser(
        "benchmark", help="transcribe a 16-bit PCM WAV"
    )
    benchmark.add_argument("wav", type=Path)
    benchmark.add_argument(
        "--device", choices=["cpu", "cuda", "auto"], default="auto"
    )
    benchmark.add_argument("--language", default=None)
    benchmark.add_argument("--timeout", type=float, default=30.0)
    benchmark.add_argument("--use-vad", action="store_true")
    benchmark.set_defaults(handler=command_benchmark)

    worker = subparsers.add_parser("_benchmark-worker", help=argparse.SUPPRESS)
    worker.add_argument("wav", type=Path)
    worker.add_argument("--device", default="auto")
    worker.add_argument("--language", default=None)
    worker.add_argument("--timeout", type=float, default=30.0)
    worker.add_argument("--use-vad", action="store_true")
    worker.set_defaults(handler=_benchmark_worker)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
