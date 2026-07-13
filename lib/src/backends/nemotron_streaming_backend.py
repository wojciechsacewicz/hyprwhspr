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
from .nemotron_languages import language_id_for
from .nemotron_streaming_session import _NemotronSession

DEFAULT_MODEL = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4"
DEFAULT_REVISION = "8364d9e2dd9da23789b480bdbba9e423717e42ee"


class _StreamingCallback:
    """Tiny callback passed to AudioCapture; inference stays off its thread."""

    def __init__(self, backend: "NemotronStreamingBackend"):
        self._backend = backend

    def set_input_sample_rate(self, sample_rate: int) -> None:
        self._backend.set_input_sample_rate(sample_rate)

    def __call__(self, audio_chunk: np.ndarray) -> None:
        self._backend.append_audio(audio_chunk)


class NemotronStreamingBackend(TranscriptionBackend):
    """Local ONNX Runtime GenAI backend for Nemotron 3.5 streaming ASR."""

    # The dispatcher registers this local provider under the existing realtime
    # backend key, so WhisperManager and main.py can reuse the proven stream
    # lifecycle without adding provider-specific branches.
    name = "realtime-ws"
    supports_streaming_capture = True
    supports_partial_transcripts = True
    requires_streaming_capture = True
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
                "Run: hyprwhspr-nemotron setup",
                flush=True,
            )
            print(f"ERROR: {exc}", flush=True)
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
                model_path = Path(
                    snapshot_download(repo_id=model_ref, revision=revision)
                )

            config_path = model_path / "genai_config.json"
            model_config = json.loads(config_path.read_text(encoding="utf-8"))["model"]
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
                self.config.get_setting("nemotron_streaming_device", "cpu")
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
                f"[BACKEND] Loading Nemotron 3.5 streaming "
                f"(device={device}, chunk={self._chunk_samples / self._model_sample_rate * 1000:.0f}ms)",
                flush=True,
            )
            self._og = og
            self._model = og.Model(config)
            self._partial_callback = getattr(
                self._manager, "_realtime_partial_callback", self._partial_callback
            )
            self._model_path = model_path
            self.current_model = str(model_ref)
            self.ready = True
            print("[BACKEND] Nemotron 3.5 streaming ready", flush=True)
            return True
        except Exception as exc:
            print(f"ERROR: Failed to load Nemotron streaming model: {exc}", flush=True)
            self._og = None
            self._model = None
            self._model_path = None
            return False

    def get_streaming_callback(self) -> Optional[Callable]:
        if self._model is None:
            return None
        self.cancel_stream()
        with self._session_lock:
            self._active_session = self._new_session(publish_partials=True)
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
        """Cancel only the active utterance; keep the local model resident."""
        self.cancel_stream()

    def update_language(self, language: Optional[str]) -> None:
        # main.py applies the override before asking for a streaming callback.
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
        if not session.enqueue(audio_chunk):
            print("[NEMOTRON] Audio chunk rejected; cancelling session", flush=True)

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
                if self.config.get_setting("recording_mode", "toggle") == "continuous":
                    # AudioCapture remains open in continuous mode. Swap sessions
                    # before finalizing so incoming callback chunks have no gap.
                    self._active_session = self._new_session(publish_partials=True)

        timeout = float(
            self.config.get_setting("nemotron_streaming_finalize_timeout", 5.0)
        )
        if session is not None:
            result = session.finish(timeout)
            self._last_use_time = time.monotonic()
            return result

        # Batch fallback keeps long-form, file tests, and retry paths compatible.
        batch = self._new_session(
            input_sample_rate=sample_rate,
            language=language_override,
            publish_partials=False,
        )
        if not batch.enqueue(audio_data):
            batch.cancel()
            return ""
        result = batch.finish(timeout)
        self._last_use_time = time.monotonic()
        return result

    def cancel_stream(self) -> None:
        with self._session_lock:
            session = self._active_session
            self._active_session = None
        if session is not None:
            session.cancel()
        self._clear_partial()

    def _new_session(
        self,
        input_sample_rate: Optional[int] = None,
        language: Optional[str] = None,
        publish_partials: bool = True,
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
            input_sample_rate=input_sample_rate or self._input_sample_rate,
            language=selected_language,
            publish_partials=publish_partials,
        )

    def _publish_partial(self, session: _NemotronSession, text: str) -> None:
        with self._session_lock:
            if self._active_session is not session:
                return
            callback = self._partial_callback
        if callback is None:
            return
        try:
            callback(text)
        except Exception as exc:
            print(f"[NEMOTRON] Partial-preview callback failed: {exc}", flush=True)

    def _clear_partial(self) -> None:
        callback = self._partial_callback
        if callback is None:
            return
        try:
            callback("")
        except Exception:
            pass

    def reinitialize(self) -> bool:
        self.unload()
        return self.initialize()

    def unload(self) -> None:
        self.cancel_stream()
        self._model = None
        self._og = None
        self._model_path = None

    def cleanup(self) -> None:
        self.unload()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
