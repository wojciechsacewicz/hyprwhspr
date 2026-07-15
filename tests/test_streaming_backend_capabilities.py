import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends import BACKENDS
from backends.nemotron_streaming_backend import NemotronStreamingBackend


class StreamingBackendCapabilityTests(unittest.TestCase):
    def test_nemotron_is_registered_under_its_own_backend_name(self):
        self.assertIs(
            BACKENDS["nemotron-streaming"], NemotronStreamingBackend
        )
        self.assertNotEqual(NemotronStreamingBackend.name, "realtime-ws")

    def test_realtime_cloud_backend_remains_registered(self):
        self.assertIn("realtime-ws", BACKENDS)
        self.assertIsNot(
            BACKENDS["realtime-ws"], NemotronStreamingBackend
        )


if __name__ == "__main__":
    unittest.main()
