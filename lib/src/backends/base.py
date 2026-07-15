"""Base class for transcription backends."""

import sys
import wave
from io import BytesIO
from typing import Callable, Optional

try:
    import numpy as np
except (ImportError, ModuleNotFoundError) as exc:
    print("ERROR: python-numpy is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {exc}", file=sys.stderr)
    sys.exit(1)


class TranscriptionBackend:
    """Base class for transcription backends owned by ``WhisperManager``."""

    name = ""
    is_local = True
    reinit_on_idle = False
    reinit_on_resume = False

    # Streaming capabilities are independent from transport. A local model may
    # consume capture chunks just like a WebSocket backend without pretending to
    # be a remote connection.
    supports_streaming_capture = False
    supports_partial_transcripts = False
    requires_streaming_capture = False
    streaming_description = "streaming backend"

    def __init__(self, manager):
        self._manager = manager

    @property
    def config(self):
        return self._manager.config

    @property
    def temp_dir(self):
        return self._manager.temp_dir

    @property
    def ready(self):
        return self._manager.ready

    @ready.setter
    def ready(self, value):
        self._manager.ready = value

    @property
    def current_model(self):
        return self._manager.current_model

    @current_model.setter
    def current_model(self, value):
        self._manager.current_model = value

    @property
    def _last_use_time(self):
        return self._manager._last_use_time

    @_last_use_time.setter
    def _last_use_time(self, value):
        self._manager._last_use_time = value

    def initialize(self) -> bool:
        raise NotImplementedError

    def transcribe(
        self,
        audio_data: "np.ndarray",
        sample_rate: int = 16000,
        language_override: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def get_streaming_callback(self) -> Optional[Callable]:
        return None

    def apply_partial_callback(
        self, callback: Optional[Callable[[str], None]]
    ) -> None:
        del callback

    def update_language(self, language: Optional[str]) -> None:
        del language

    def close(self) -> None:
        """Close an active stream/session without necessarily unloading a model."""

    def reinitialize(self) -> bool:
        return True

    def unload(self) -> None:
        """Release a loaded local model."""

    def cleanup(self) -> None:
        """Release resources at shutdown."""

    @property
    def is_loaded(self) -> bool:
        return False

    def _resample_audio(
        self,
        audio_data: "np.ndarray",
        source_rate: int,
        target_rate: int,
    ) -> "np.ndarray":
        if source_rate == target_rate:
            return audio_data
        try:
            from math import gcd
            from scipy import signal

            divisor = gcd(int(source_rate), int(target_rate))
            resampled = signal.resample_poly(
                audio_data,
                up=int(target_rate) // divisor,
                down=int(source_rate) // divisor,
            )
            return resampled.astype(np.float32, copy=False)
        except Exception as exc:
            print(
                f"[WARN] Failed to resample audio {source_rate}Hz -> "
                f"{target_rate}Hz: {exc}",
                flush=True,
            )
            return audio_data

    def _numpy_to_wav_bytes(
        self, audio_data: "np.ndarray", sample_rate: int = 16000
    ) -> bytes:
        try:
            if audio_data.ndim != 1:
                raise ValueError(
                    f"Expected mono audio array, got shape {audio_data.shape}"
                )
            if audio_data.dtype == np.float32:
                audio_clipped = np.clip(audio_data, -1.0, 1.0)
                audio_int16 = (audio_clipped * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)

            wav_buffer = BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            return wav_buffer.getvalue()
        except Exception as exc:
            print(f"ERROR: Failed to convert audio to WAV: {exc}")
            raise
