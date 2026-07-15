import os
import sys
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends.nemotron_streaming_backend import NemotronStreamingBackend


@unittest.skipUnless(
    os.environ.get("HYPRWHSPR_RUN_MODEL_TESTS") == "1",
    "set HYPRWHSPR_RUN_MODEL_TESTS=1 and provide a real speech fixture",
)
class NemotronModelIntegrationTests(unittest.TestCase):
    def test_real_model_streams_configured_speech_fixture(self):
        audio_path = os.environ.get("HYPRWHSPR_NEMOTRON_TEST_AUDIO")
        expected = os.environ.get("HYPRWHSPR_NEMOTRON_EXPECTED_TEXT")
        if not audio_path or not expected:
            self.fail(
                "set HYPRWHSPR_NEMOTRON_TEST_AUDIO and "
                "HYPRWHSPR_NEMOTRON_EXPECTED_TEXT"
            )

        path = Path(audio_path)
        with wave.open(str(path), "rb") as source:
            self.assertEqual(source.getsampwidth(), 2)
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            frames = source.readframes(source.getnframes())
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.float32)
        self.assertGreater(len(audio), 0)

        class Config:
            values = {
                "nemotron_streaming_device": os.environ.get(
                    "HYPRWHSPR_NEMOTRON_TEST_DEVICE", "cpu"
                ),
                "language": os.environ.get(
                    "HYPRWHSPR_NEMOTRON_TEST_LANGUAGE", "pl-PL"
                ),
                "recording_mode": "toggle",
                "nemotron_streaming_buffer_max_seconds": max(
                    3.0, len(audio) / sample_rate + 1.0
                ),
                "nemotron_streaming_finalize_timeout": 30.0,
                "nemotron_streaming_use_vad": False,
            }

            def get_setting(self, key, default=None):
                return self.values.get(key, default)

        class Manager:
            config = Config()
            temp_dir = None
            ready = False
            current_model = None
            _last_use_time = 0.0
            _streaming_partial_callback = None
            _realtime_partial_callback = None

        backend = NemotronStreamingBackend(Manager())
        partials = []
        backend.apply_partial_callback(partials.append)
        self.assertTrue(backend.initialize())
        try:
            callback = backend.get_streaming_callback()
            self.assertIsNotNone(callback)
            callback.set_input_sample_rate(sample_rate)
            for offset in range(0, len(audio), 1024):
                callback(audio[offset:offset + 1024])

            transcript = backend.transcribe(
                audio, sample_rate, Config.values["language"]
            ).lower()
            for word in expected.lower().split():
                self.assertIn(word, transcript)
            self.assertFalse(backend._sessions)
            self.assertTrue(
                any(text.strip() for text in partials) or bool(transcript)
            )
        finally:
            backend.cleanup()


if __name__ == "__main__":
    unittest.main()
