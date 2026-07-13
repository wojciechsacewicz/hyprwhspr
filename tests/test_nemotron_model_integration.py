import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends.nemotron_streaming_backend import NemotronStreamingBackend


@unittest.skipUnless(
    os.environ.get("HYPRWHSPR_RUN_MODEL_TESTS") == "1",
    "set HYPRWHSPR_RUN_MODEL_TESTS=1 to download and initialize the real model",
)
class NemotronModelIntegrationTests(unittest.TestCase):
    def test_real_model_initializes_and_flushes_one_chunk(self):
        class Config:
            values = {
                "nemotron_streaming_device": "cpu",
                "language": "pl-PL",
                "recording_mode": "toggle",
                "nemotron_streaming_buffer_max_seconds": 3.0,
                "nemotron_streaming_finalize_timeout": 20.0,
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
            _realtime_partial_callback = None

        backend = NemotronStreamingBackend(Manager())
        self.assertTrue(backend.initialize())
        try:
            session = backend._new_session(
                input_sample_rate=16000,
                language="pl-PL",
                publish_partials=False,
            )
            self.assertTrue(session.enqueue(np.zeros(8960, dtype=np.float32)))
            self.assertEqual(session.finish(20.0), "")
        finally:
            backend.cleanup()


if __name__ == "__main__":
    unittest.main()
