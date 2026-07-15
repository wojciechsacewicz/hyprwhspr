import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends.base import TranscriptionBackend
from whisper_manager import WhisperManager


class Config:
    def __init__(self, backend="nemotron-streaming"):
        self.values = {"transcription_backend": backend}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)

    def get_temp_directory(self):
        return Path("/tmp")


class LocalStreamingBackend(TranscriptionBackend):
    name = "nemotron-streaming"
    is_local = True
    supports_streaming_capture = True
    supports_partial_transcripts = True
    requires_streaming_capture = True

    def __init__(self, manager):
        super().__init__(manager)
        self.callback = object()
        self.partial = None
        self.loaded = True

    def initialize(self):
        self.ready = True
        return True

    def get_streaming_callback(self):
        return self.callback

    def apply_partial_callback(self, callback):
        self.partial = callback

    def transcribe(self, *_args, **_kwargs):
        return "ok"

    def unload(self):
        self.loaded = False
        return True

    @property
    def is_loaded(self):
        return self.loaded


class LegacyRealtimeBackend(LocalStreamingBackend):
    name = "realtime-ws"
    is_local = False
    supports_streaming_capture = False
    supports_partial_transcripts = False
    requires_streaming_capture = False


class WhisperManagerCapabilityTests(unittest.TestCase):
    def test_local_streaming_backend_uses_generic_capabilities(self):
        manager = WhisperManager(Config())
        manager._backend = LocalStreamingBackend(manager)
        manager.ready = True
        self.assertIs(
            manager.get_realtime_streaming_callback(),
            manager._backend.callback,
        )
        callback = lambda text: text
        manager.set_realtime_partial_callback(callback)
        self.assertIs(manager._backend.partial, callback)
        self.assertTrue(manager.unload_model())

    def test_legacy_realtime_backend_remains_compatible(self):
        manager = WhisperManager(Config("realtime-ws"))
        manager._backend = LegacyRealtimeBackend(manager)
        manager.ready = True
        self.assertIs(
            manager.get_realtime_streaming_callback(),
            manager._backend.callback,
        )
        callback = lambda text: text
        manager.set_realtime_partial_callback(callback)
        self.assertIs(manager._backend.partial, callback)
        self.assertFalse(manager.unload_model())


if __name__ == "__main__":
    unittest.main()
