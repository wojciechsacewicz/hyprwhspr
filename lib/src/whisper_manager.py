"""Facade over the transcription backend classes."""

import threading
import time
from typing import Callable, Optional

try:
    from .dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package("numpy")

try:
    from .config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from .backend_utils import normalize_backend
except ImportError:
    from backend_utils import normalize_backend

try:
    from .backend_installer import PYWHISPERCPP_MODELS_DIR
except ImportError:
    from backend_installer import PYWHISPERCPP_MODELS_DIR

try:
    from .backends import BACKENDS, PywhispercppBackend
except ImportError:
    from backends import BACKENDS, PywhispercppBackend


class WhisperManager:
    """Manage transcription through one active backend instance."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.current_model = None
        self.temp_dir = None

        # Kept under the old name as an alias because main.py and existing
        # realtime providers use it. The capability is transport-independent.
        self._streaming_partial_callback = None
        self._realtime_partial_callback = None

        self._model_lock = threading.Lock()
        self.ready = False
        self._last_use_time = 0.0
        self._model_manually_unloaded = False
        self._backend = None

    def initialize(self) -> bool:
        try:
            self.temp_dir = self.config.get_temp_directory()
            backend = self._current_backend_name()
            backend_cls = BACKENDS.get(backend, PywhispercppBackend)
            self._backend = backend_cls(self)
            initialized = self._backend.initialize()
            if initialized and self._streaming_partial_callback is not None:
                self._backend.apply_partial_callback(
                    self._streaming_partial_callback
                )
            return initialized
        except Exception as exc:
            print(f"ERROR: Failed to initialize Whisper manager: {exc}")
            return False

    def get_realtime_streaming_callback(self) -> Optional[Callable]:
        """Return a capture callback for any streaming-capable backend.

        The historical method name is retained so main.py and external callers
        remain compatible while local streaming backends no longer need to
        impersonate ``realtime-ws``.
        """
        if (
            self._backend is not None
            and (
                self._backend.supports_streaming_capture
                or self._backend.name == "realtime-ws"
            )
        ):
            return self._backend.get_streaming_callback()
        return None

    def set_realtime_partial_callback(
        self, callback: Optional[Callable[[str], None]]
    ) -> None:
        """Set partial transcript preview callback for streaming backends."""
        self._streaming_partial_callback = callback
        self._realtime_partial_callback = callback
        if (
            self._backend is not None
            and (
                self._backend.supports_partial_transcripts
                or self._backend.name == "realtime-ws"
            )
        ):
            self._backend.apply_partial_callback(callback)

    def _current_backend_name(self) -> str:
        return normalize_backend(
            self.config.get_setting(
                "transcription_backend", "pywhispercpp"
            )
        )

    def update_realtime_language(self, language: Optional[str]) -> None:
        """Apply a language override to a loaded streaming backend."""
        if (
            self._backend is not None
            and (
                self._backend.supports_streaming_capture
                or self._backend.name == "realtime-ws"
            )
            and self._backend.is_loaded
        ):
            self._backend.update_language(language)

    def close_realtime_connection(self, reason: str = "") -> None:
        """Close the active streaming session without assuming its transport."""
        if (
            self._backend is not None
            and (
                self._backend.supports_streaming_capture
                or self._backend.name == "realtime-ws"
            )
            and self._backend.is_loaded
        ):
            note = f" ({reason})" if reason else ""
            description = getattr(
                self._backend,
                "streaming_description",
                "streaming backend",
            )
            print(f"[CLEANUP] Closing {description}{note}", flush=True)
            self._backend.close()

    def reinitialize_after_resume(
        self, only_if_idle: bool = False
    ) -> bool:
        if self._backend is None or not self._backend.reinit_on_resume:
            return True

        if only_if_idle:
            if not (self._backend.is_local and self._backend.is_loaded):
                return True
            idle = time.monotonic() - self._last_use_time
            if not (idle > 1800 and self._last_use_time > 0):
                return True
            print(
                f"[RECOVERY] Reinitializing {self._current_backend_name()} "
                "model after audio recovery (suspend/resume detected)",
                flush=True,
            )

        with self._model_lock:
            return self._backend.reinitialize()

    def is_ready(self) -> bool:
        return self.ready

    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language_override: Optional[str] = None,
    ) -> str:
        if not self.ready:
            raise RuntimeError("Whisper manager not initialized")

        if audio_data is None:
            print("No audio data provided to transcribe", flush=True)
            return ""
        if len(audio_data) == 0:
            print("Empty audio data provided to transcribe", flush=True)
            return ""

        try:
            if not isinstance(audio_data, np.ndarray):
                print(
                    f"Invalid audio data type: {type(audio_data)}, "
                    "expected numpy.ndarray",
                    flush=True,
                )
                return ""
            if audio_data.ndim != 1:
                print(
                    f"Invalid audio data shape: {audio_data.shape}, "
                    "expected 1D array",
                    flush=True,
                )
                if audio_data.ndim == 2 and audio_data.shape[1] == 1:
                    audio_data = audio_data.flatten()
                else:
                    return ""
            if audio_data.dtype != np.float32:
                print(
                    f"Converting audio data from {audio_data.dtype} to float32",
                    flush=True,
                )
                audio_data = audio_data.astype(np.float32)
            if not audio_data.flags["C_CONTIGUOUS"]:
                audio_data = np.ascontiguousarray(
                    audio_data, dtype=np.float32
                )
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                print(
                    "Audio data contains NaN or inf values - invalid",
                    flush=True,
                )
                return ""
            if np.all(audio_data == 0.0):
                print(
                    "Audio data is all zeros (silence) - skipping transcription",
                    flush=True,
                )
                return ""
            min_samples = int(sample_rate * 0.1)
            if len(audio_data) < min_samples:
                print(
                    f"Audio too short: {len(audio_data)} samples "
                    f"(minimum {min_samples})",
                    flush=True,
                )
                return ""
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 1e-6:
                print(
                    f"Audio level too low (RMS: {rms:.2e}) - likely invalid",
                    flush=True,
                )
                return ""
        except Exception as exc:
            print(f"[ERROR] Audio data validation failed: {exc}", flush=True)
            import traceback

            traceback.print_exc()
            return ""

        backend_name = self._current_backend_name()
        if backend_name == "rest-api":
            if self._backend is None:
                self._backend = BACKENDS["rest-api"](self)
            return self._backend.transcribe(
                audio_data,
                sample_rate,
                language_override=language_override,
            )

        if self._backend is None:
            print("[ERROR] No transcription backend initialized", flush=True)
            return ""

        # Remote streaming transports own their connection concurrency. Local
        # streaming models still use the model lock so unload/reinitialize cannot
        # release the runtime while finalization is in progress.
        requires_streaming = (
            self._backend.requires_streaming_capture
            or self._backend.name == "realtime-ws"
        )
        if requires_streaming and not self._backend.is_local:
            return self._backend.transcribe(
                audio_data,
                sample_rate,
                language_override=language_override,
            )

        with self._model_lock:
            if not self._ensure_backend_fresh_locked(backend_name):
                return ""
            return self._backend.transcribe(
                audio_data,
                sample_rate,
                language_override=language_override,
            )

    def _ensure_backend_fresh_locked(self, backend: str) -> bool:
        time_since_last_use = time.monotonic() - self._last_use_time
        if not (time_since_last_use > 1800 and self._last_use_time > 0):
            return True
        if self._backend is None or not self._backend.reinit_on_idle:
            return True

        print(
            f"[MODEL] Long idle detected - reinitializing {backend} model "
            "(suspend/resume likely)",
            flush=True,
        )
        if not self._backend.reinitialize():
            print(
                "[MODEL] Reinitialization failed, transcription may fail",
                flush=True,
            )
            return False
        return True

    def cleanup(self) -> None:
        if self._backend is not None:
            self._backend.cleanup()

    def unload_model(self) -> bool:
        if self._backend is None:
            print("[MODEL] No backend initialized", flush=True)
            return False
        if not self._backend.is_local:
            print(
                "[MODEL] Unload not applicable for non-local backend",
                flush=True,
            )
            return False

        with self._model_lock:
            try:
                result = self._backend.unload()
                if result is False or self._backend.is_loaded:
                    print(
                        "[MODEL] ERROR: Backend did not release its model",
                        flush=True,
                    )
                    return False

                import gc

                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("[MODEL] CUDA cache cleared", flush=True)
                except ImportError:
                    pass

                self.ready = False
                self._model_manually_unloaded = True
                print(
                    "[MODEL] Model unloaded from memory — resources freed",
                    flush=True,
                )
                return True
            except Exception as exc:
                print(
                    f"[MODEL] ERROR: Failed to unload model: {exc}",
                    flush=True,
                )
                return False

    def reload_model(self) -> bool:
        print("[MODEL] Reloading model...", flush=True)
        result = self.initialize()
        if result:
            with self._model_lock:
                self._model_manually_unloaded = False
            print("[MODEL] Model reloaded successfully", flush=True)
        else:
            print("[MODEL] ERROR: Failed to reload model", flush=True)
        return result

    def set_threads(self, num_threads: int) -> bool:
        set_threads = getattr(self._backend, "set_threads", None)
        if set_threads is None:
            print("ERROR: Backend does not support changing threads")
            return False
        with self._model_lock:
            return set_threads(int(num_threads))

    def set_model(self, model_name: str) -> bool:
        set_model = getattr(self._backend, "set_model", None)
        if set_model is None:
            print("ERROR: Active backend does not support changing models")
            return False
        with self._model_lock:
            return set_model(model_name)

    def get_current_model(self) -> str:
        return self.current_model or ""

    def get_available_models(self) -> list:
        models_dir = PYWHISPERCPP_MODELS_DIR
        available_models = []
        for model in ["tiny", "base", "small", "medium", "large"]:
            model_files = [
                models_dir / f"ggml-{model}.bin",
                models_dir / f"ggml-{model}.en.bin",
            ]
            for model_file in model_files:
                if model_file.exists():
                    model_name = (
                        f"{model}.en"
                        if model_file.name.endswith(".en.bin")
                        else model
                    )
                    if model_name not in available_models:
                        available_models.append(model_name)
                    break
        return sorted(available_models)

    def get_backend_info(self) -> str:
        backend = self._current_backend_name()
        if backend == "rest-api":
            endpoint_url = self.config.get_setting(
                "rest_endpoint_url", "not configured"
            )
            return f"REST API ({endpoint_url})"
        if backend not in BACKENDS:
            backend = "pywhispercpp"
        if backend == "pywhispercpp":
            return (
                "pywhispercpp (in-process, "
                f"model: {self.current_model})"
            )
        return (
            f"{backend} (model: {self.current_model})"
            if self.current_model
            else backend
        )
