#!/usr/bin/env python3
"""Install, configure, validate, and benchmark local Nemotron streaming."""

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
ORT_GENAI_VERSION = "0.14.0"
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SAMPLES = 8960
LOCAL_PROVIDER = "nemotron-local"


def _xdg_path(env_name: str, fallback: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser() if value else fallback


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_file() -> Path:
    base = _xdg_path("XDG_CONFIG_HOME", Path.home() / ".config")
    return base / "hyprwhspr" / "config.json"


def venv_dir() -> Path:
    base = _xdg_path("XDG_DATA_HOME", Path.home() / ".local" / "share")
    return base / "hyprwhspr" / "venv"


def venv_python() -> Path:
    return venv_dir() / "bin" / "python"


def _run(command: Iterable[str], *, check: bool = True,
         capture_output: bool = False, env: Optional[dict] = None):
    return subprocess.run(
        [str(part) for part in command], check=check, text=True,
        capture_output=capture_output, env=env,
    )


def resolve_device(device: str) -> str:
    normalized = device.lower()
    if normalized in {"cpu", "cuda"}:
        return normalized
    if normalized != "auto":
        raise ValueError("device must be auto, cpu, or cuda")
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return "cpu"
    result = subprocess.run(
        [nvidia_smi, "-L"], check=False, capture_output=True,
        text=True, timeout=3,
    )
    return "cuda" if result.returncode == 0 and "GPU" in result.stdout else "cpu"


def selected_runtime_package(device: str) -> str:
    if resolve_device(device) == "cuda":
        return f"onnxruntime-genai-cuda=={ORT_GENAI_VERSION}"
    return f"onnxruntime-genai=={ORT_GENAI_VERSION}"


def ensure_existing_venv() -> Path:
    python = venv_python()
    if not python.is_file():
        raise RuntimeError(
            "hyprwhspr virtual environment is missing. Run the normal "
            "'hyprwhspr setup' first, then run this command again."
        )
    return python


def install_dependencies(device: str) -> str:
    python = ensure_existing_venv()
    resolved = resolve_device(device)
    runtime = selected_runtime_package(resolved)
    pip = [python, "-m", "pip"]
    _run([*pip, "install", "--upgrade", "pip"])
    opposite = "onnxruntime-genai-cuda" if resolved == "cpu" else "onnxruntime-genai"
    _run([*pip, "uninstall", "-y", opposite], check=False)
    _run([
        *pip, "install", runtime, "huggingface-hub>=0.34,<2",
        "soxr>=0.5,<1", "numpy>=1.26",
    ])
    return resolved


def _model_path_script(download: bool) -> str:
    local_flag = "False" if download else "True"
    return (
        "from huggingface_hub import snapshot_download; "
        f"print(snapshot_download(repo_id={MODEL_ID!r}, revision={MODEL_REVISION!r}, "
        f"local_files_only={local_flag}))"
    )


def resolve_model_path(*, download: bool) -> Path:
    result = _run(
        [ensure_existing_venv(), "-c", _model_path_script(download)],
        capture_output=True,
    )
    path = Path(result.stdout.strip().splitlines()[-1])
    if not path.is_dir():
        raise RuntimeError(f"model snapshot was not resolved: {path}")
    return path


def validate_model_config(path: Path) -> dict:
    model_config_path = path / "genai_config.json"
    if not model_config_path.is_file():
        raise ValueError(f"missing {model_config_path}")
    model = json.loads(model_config_path.read_text(encoding="utf-8")).get("model") or {}
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


def update_config(path: Path, *, device: str,
                  language: Optional[str]) -> Optional[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    backup = None
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        shutil.copy2(path, backup)
    data.update({
        "transcription_backend": "realtime-ws",
        "websocket_provider": LOCAL_PROVIDER,
        "websocket_model": MODEL_ID,
        "realtime_mode": "transcribe",
        "nemotron_streaming_model": MODEL_ID,
        "nemotron_streaming_revision": MODEL_REVISION,
        "nemotron_streaming_device": device,
        "nemotron_streaming_buffer_max_seconds": 3.0,
        "nemotron_streaming_finalize_timeout": 5.0,
        "nemotron_streaming_use_vad": False,
    })
    if language:
        data["language"] = language
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return backup


def _service(action: str, *, check: bool = False) -> bool:
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False
    result = subprocess.run(
        [systemctl, "--user", action, "hyprwhspr.service"],
        check=False, capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(message or f"systemctl {action} failed")
    return result.returncode == 0


def dependency_status() -> tuple[bool, str]:
    python = venv_python()
    if not python.is_file():
        return False, "hyprwhspr venv missing"
    code = (
        "import onnxruntime_genai, huggingface_hub, soxr, numpy; "
        "print(getattr(onnxruntime_genai, '__version__', 'installed'))"
    )
    result = subprocess.run(
        [python, "-c", code], check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or "Nemotron dependencies missing"
    return True, result.stdout.strip()


def command_setup(args) -> int:
    try:
        device = install_dependencies(args.device)
        print(f"Runtime installed for: {device}")
        if not args.skip_download:
            path = resolve_model_path(download=True)
            validate_model_config(path)
            print(f"Model cached: {path}")
        backup = update_config(config_file(), device=device, language=args.language)
        print(f"Configuration updated: {config_file()}")
        if backup:
            print(f"Backup: {backup}")
        if not args.no_restart:
            if _service("restart"):
                print("hyprwhspr.service restarted")
            else:
                print("Service was not restarted; start it when ready.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def command_download(_args) -> int:
    try:
        path = resolve_model_path(download=True)
        validate_model_config(path)
        print(path)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _read_config() -> dict:
    path = config_file()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def command_status(_args) -> int:
    config = _read_config()
    configured = (
        config.get("transcription_backend") == "realtime-ws"
        and config.get("websocket_provider") == LOCAL_PROVIDER
    )
    deps_ok, deps_note = dependency_status()
    print(f"Configured: {'yes' if configured else 'no'}")
    print(f"Dependencies: {'ok' if deps_ok else 'missing'} ({deps_note})")
    try:
        model_path = resolve_model_path(download=False)
        validate_model_config(model_path)
        print(f"Model: cached ({model_path})")
        model_ok = True
    except Exception as exc:
        print(f"Model: missing or invalid ({exc})")
        model_ok = False
    print(f"Service: {'active' if _service('is-active') else 'inactive'}")
    return 0 if configured and deps_ok and model_ok else 1


def command_validate(_args) -> int:
    ok, note = dependency_status()
    if not ok:
        print(f"ERROR: {note}", file=sys.stderr)
        return 1
    try:
        path = resolve_model_path(download=False)
        model = validate_model_config(path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    duration_ms = model["chunk_samples"] / model["sample_rate"] * 1000
    print(f"Dependencies: ok ({note})")
    print(f"Model: {path}")
    print(f"Streaming chunk: {duration_ms:.0f} ms")
    return 0


def _benchmark_worker(args) -> int:
    sys.path.insert(0, str(repo_root() / "lib" / "src"))
    import numpy as np
    from backends.nemotron_streaming_backend import NemotronStreamingBackend

    with wave.open(str(args.wav), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if width != 2:
        raise ValueError("benchmark currently accepts 16-bit PCM WAV files")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.float32)

    class Config:
        values = {
            "language": args.language,
            "recording_mode": "toggle",
            "nemotron_streaming_model": MODEL_ID,
            "nemotron_streaming_revision": MODEL_REVISION,
            "nemotron_streaming_device": args.device,
            "nemotron_streaming_buffer_max_seconds": max(
                3.0, len(audio) / sample_rate + 1.0
            ),
            "nemotron_streaming_finalize_timeout": args.timeout,
            "nemotron_streaming_use_vad": False,
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
            self._realtime_partial_callback = None

    backend = NemotronStreamingBackend(Manager())
    if not backend.initialize():
        return 1
    started = time.perf_counter()
    text = backend.transcribe(audio, sample_rate, args.language)
    elapsed = time.perf_counter() - started
    duration = len(audio) / sample_rate
    print(text)
    print(f"audio={duration:.2f}s elapsed={elapsed:.2f}s RTF={elapsed / duration:.3f}")
    return 0 if text else 1


def command_benchmark(args) -> int:
    python = ensure_existing_venv()
    if Path(sys.executable).resolve() != python.resolve():
        command = [
            python, str(Path(__file__).resolve()), "_benchmark-worker",
            str(args.wav), "--device", args.device,
            "--language", args.language, "--timeout", str(args.timeout),
        ]
        return subprocess.run(command, check=False).returncode
    try:
        return _benchmark_worker(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyprwhspr-nemotron",
        description="Manage the optional local Nemotron 3.5 streaming provider",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser("setup", help="install, download, and configure")
    setup.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    setup.add_argument("--language", default="pl-PL")
    setup.add_argument("--skip-download", action="store_true")
    setup.add_argument("--no-restart", action="store_true")
    setup.set_defaults(handler=command_setup)
    download = subparsers.add_parser("download", help="download the pinned model")
    download.set_defaults(handler=command_download)
    status = subparsers.add_parser("status", help="show configuration and cache status")
    status.set_defaults(handler=command_status)
    validate = subparsers.add_parser("validate", help="validate dependencies and model metadata")
    validate.set_defaults(handler=command_validate)
    benchmark = subparsers.add_parser("benchmark", help="transcribe a 16-bit PCM WAV")
    benchmark.add_argument("wav", type=Path)
    benchmark.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    benchmark.add_argument("--language", default="pl-PL")
    benchmark.add_argument("--timeout", type=float, default=30.0)
    benchmark.set_defaults(handler=command_benchmark)
    for action, help_text in (
        ("unload", "stop the service and free model memory"),
        ("reload", "start the service and load the model"),
        ("restart", "restart the service"),
    ):
        command = subparsers.add_parser(action, help=help_text)
        command.set_defaults(handler=lambda _args, action=action: 0 if _service(
            {"unload": "stop", "reload": "start", "restart": "restart"}[action],
            check=True,
        ) else 1)
    worker = subparsers.add_parser("_benchmark-worker", help=argparse.SUPPRESS)
    worker.add_argument("wav", type=Path)
    worker.add_argument("--device", default="cpu")
    worker.add_argument("--language", default="pl-PL")
    worker.add_argument("--timeout", type=float, default=30.0)
    worker.set_defaults(handler=_benchmark_worker)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
