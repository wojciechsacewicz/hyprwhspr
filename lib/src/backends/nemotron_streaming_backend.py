"""Local cache-aware Nemotron 3.5 streaming ASR backend."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

try:
    from ..dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package("numpy")

from .base import TranscriptionBackend
from .nemotron_streaming_session import _NemotronSession

DEFAULT_MODEL = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4"
DEFAULT_REVISION = "8364d9e2dd9da23789b480bdbba9e423717e42ee"


class _StreamingCallback:
    """Small callback passed to AudioCapture; inference stays off its thread."""

    def __init__(self, backend: "NemotronStreamingBackend"):
        self._backend = backend

    def set_input_sample_rate(self, sample_rate: int) -> None:
        self._backend.set_input_sample_rate(sample_rate)

    def __call__(self, audio_chunk: np.ndarray) -> None:
        self._backend.append_audio(audio_chunk)


class NemotronStreamingBackend(TranscriptionBackend):
    """Local ONNX Runtime GenAI backend for Nemotron 3.5 streaming ASR."""

    name = "nemotron-streaming"
    is_local = True
    supports_streaming_capture = True
    supports_partial_transcripts = True
    requires_streaming_capture = True
    streaming_description = "local Nemotron streaming model"
    reinit_on_resume = True

    def __init__(self, manager):
        super().__init__(manager)
        self._og = None
        self._model = None
        self._model_path: Optional[Path] = None
        self._model_sample_rate = 16000
        self._chunk_samples = 8960

        self._session_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._active_session: Optional[_NemotronSession] = None
        self._generation = 0
        self._input_sample_rate = 16000
        self._language_override: Optional[str] = None
        self._partial_callback: Optional[Callable[[str], None]] = None
        self._streaming_callback = _StreamingCallback(self)

    def initialize(self) -> bool:
        try:
            import onnxruntime_genai as og
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            print(
                "ERROR: Nemotron streaming dependencies are missing. "
                "Run: hyprwhspr nemotron setup",
                flush=True,
            )
            print(f"ERROR: {exc}", flush=True)
            self.ready = False
            return False

        model_ref = self.config.get_setting(
            "nemotron_streaming_model", DEFAULT_MODEL
        )
        revision = self.config.get_setting(
            "nemotron_streaming_revision", DEFAULT_REVISION
        )
        model_path = Path(str(model_ref)).expanduser()

        try:
            if not model_path.is_dir():
                # Runtime startup must never begin a multi-hundred-megabyte
                # download. Setup owns network access and populates the HF cache.
                model_path = Path(
                    snapshot_download(
                        repo_id=str(model_ref),
                        revision=str(revision),
                        local_files_only=True,
                    )
                )

            config_path = model_path / "genai_config.json"
            model_config = json.loads(
                config_path.read_text(encoding="utf-8")
            )["model"]
            self._model_sample_rate = int(model_config["sample_rate"])
            self._chunk_samples = int(model_config["chunk_samples"])

            if self._model_sample_rate != 16000:
                raise ValueError(
                    f"unsupported model sample rate: {self._model_sample_rate}"
                )
            if self._chunk_samples <= 0:
                raise ValueError("model chunk_samples must be positive")

            config = og.Config(str(model_path))
            device = str(
                self.config.get_setting(
                    "nemotron_streaming_device", "cpu"
                )
            ).lower()
            if device == "cuda":
                config.clear_providers()
                config.append_provider("cuda")
            elif device == "cpu":
                config.clear_providers()
            elif device != "auto":
                raise ValueError(
                    "nemotron_streaming_device must be auto, cpu, or cuda"
                )

            print(
                "[BACKEND] Loading Nemotron 3.5 streaming "
                f"(device={device}, "
                f"chunk={self._chunk_samples / self._model_sample_rate * 1000:.0f}ms)",
                flush=True,
            )
            self._og = og
            self._model = og.Model(config)
            self._partial_callback = getattr(
                self._manager,
                "_streaming_partial_callback",
                getattr(
                    self._manager,
                    "_realtime_partial_callback",
                    self._partial_callback,
                ),
            )
            self._model_path = model_path
            self.current_model = str(model_ref)
            self.ready = True
            print("[BACKEND] Nemotron 3.5 streaming ready", flush=True)
            return True
        except Exception as exc:
            print(
                "ERROR: Failed to load the cached Nemotron streaming model: "
                f"{exc}",
                flush=True,
            )
            print(
                "Run 'hyprwhspr nemotron setup' to install or repair it.",
                flush=True,
            )
            self._og = None
            self._model = None
            self._model_path = None
            self.current_model = None
            self.ready = False
            return False

    def get_streaming_callback(self) -> Optional[Callable]:
        if self._model is None:
            return None
        if not self.cancel_stream(timeout=2.0):
            print(
                "[NEMOTRON] Previous streaming worker did not stop; "
                "refusing to start a concurrent session",
                flush=True,
            )
            return None
        with self._session_lock:
            self._active_session = self._new_session(
                publish_partials=True
            )
        return self._streaming_callback

    def apply_partial_callback(
        self, callback: Optional[Callable[[str], None]]
    ) -> None:
        self._partial_callback = callback
        if callback is None:
            return
        try:
            callback("")
        except Exception:
            pass

    def close(self) -> None:
        """Cancel only the active utterance; keep the model resident."""
        self.cancel_stream(timeout=2.0)

    def update_language(self, language: Optional[str]) -> None:
        self._language_override = language

    def set_input_sample_rate(self, sample_rate: int) -> None:
        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            return
        if sample_rate <= 0:
            return
        self._input_sample_rate = sample_rate
        with self._session_lock:
            session = self._active_session
        if session is not None:
            session.set_input_sample_rate(sample_rate)

    def append_audio(self, audio_chunk: np.ndarray) -> None:
        with self._session_lock:
            session = self._active_session
        if session is None:
            return
        if session.enqueue(audio_chunk):
            return
        error = session.consume_unlogged_error()
        if error:
            print(f"[NEMOTRON] {error}; cancelling utterance", flush=True)
            self._clear_partial()

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language_override: Optional[str] = None,
    ) -> str:
        if self._model is None:
            print("[NEMOTRON] Model not loaded", flush=True)
            return ""

        with self._session_lock:
            session = self._active_session
            if session is not None:
                self._active_session = None
                if (
                    self.config.get_setting("recording_mode", "toggle")
                    == "continuous"
                ):
                    # Keep capture live. The bounded queue makes overload explicit
                    # if finalization cannot keep up with the next utterance.
                    self._active_session = self._new_session(
                        publish_partials=True
                    )

        timeout = float(
            self.config.get_setting(
                "nemotron_streaming_finalize_timeout", 5.0
            )
        )
        if session is not None:
            result = session.finish(timeout)
            self._last_use_time = time.monotonic()
            return result

        # Batch fallback is used by long-form, file tests, and retry paths. Give
        # it enough queue capacity for the supplied recording instead of the
        # small realtime backlog limit.
        audio_seconds = len(audio_data) / max(1, int(sample_rate))
        batch = self._new_session(
            input_sample_rate=sample_rate,
            language=language_override,
            publish_partials=False,
            buffer_max_seconds=max(audio_seconds + 1.0, 3.0),
        )
        if not batch.enqueue(audio_data):
            batch.cancel()
            return ""
        result = batch.finish(timeout)
        self._last_use_time = time.monotonic()
        return result

    def cancel_stream(self, timeout: float = 1.0) -> bool:
        with self._session_lock:
            session = self._active_session
            self._active_session = None
        stopped = True
        if session is not None:
            stopped = session.cancel(timeout=timeout)
            if not stopped:
                print(
                    "[NEMOTRON] Streaming worker did not stop within "
                    f"{timeout:.1f}s",
                    flush=True,
                )
        self._clear_partial()
        return stopped

    def _new_session(
        self,
        input_sample_rate: Optional[int] = None,
        language: Optional[str] = None,
        publish_partials: bool = True,
        buffer_max_seconds: Optional[float] = None,
    ) -> _NemotronSession:
        self._generation += 1
        selected_language = (
            language
            if language is not None
            else self._language_override
            if self._language_override is not None
            else self.config.get_setting("language", None)
        )
        return _NemotronSession(
            self,
            generation_id=self._generation,
            input_sample_rate=(
                input_sample_rate or self._input_sample_rate
            ),
            language=selected_language,
            publish_partials=publish_partials,
            buffer_max_seconds=buffer_max_seconds,
        )

    def _publish_partial(
        self, session: _NemotronSession, text: str
    ) -> None:
        with self._session_lock:
            if self._active_session is not session:
                return
            callback = self._partial_callback
        if callback is None:
            return
        try:
            callback(text)
        except Exception as exc:
            print(
                f"[NEMOTRON] Partial-preview callback failed: {exc}",
                flush=True,
            )

    def _clear_partial(self) -> None:
        callback = self._partial_callback
        if callback is None:
            return
        try:
            callback("")
        except Exception:
            pass

    def reinitialize(self) -> bool:
        if not self.unload():
            return False
        return self.initialize()

    def unload(self) -> bool:
        if not self.cancel_stream(timeout=5.0):
            print(
                "[NEMOTRON] Refusing to release the model while an inference "
                "worker is still active",
                flush=True,
            )
            return False
        self._model = None
        self._og = None
        self._model_path = None
        self.current_model = None
        self.ready = False
        return True

    def cleanup(self) -> None:
        self.unload()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
