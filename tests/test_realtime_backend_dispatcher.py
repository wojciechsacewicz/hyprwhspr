import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends import realtime_backend_dispatcher as dispatcher


class Config:
    def __init__(self, provider):
        self.provider = provider

    def get_setting(self, key, default=None):
        if key == "websocket_provider":
            return self.provider
        return default


class Manager:
    def __init__(self, provider):
        self.config = Config(provider)


class FakeLocal:
    def __init__(self, manager):
        self.manager = manager


class FakeCloud:
    def __init__(self, manager):
        self.manager = manager


class RealtimeBackendDispatcherTests(unittest.TestCase):
    def test_local_provider_selects_nemotron(self):
        with mock.patch.object(dispatcher, "NemotronStreamingBackend", FakeLocal):
            result = dispatcher.RealtimeBackendDispatcher(Manager("nemotron-local"))
        self.assertIsInstance(result, FakeLocal)

    def test_other_provider_keeps_existing_websocket_backend(self):
        with mock.patch.object(dispatcher, "RealtimeWsBackend", FakeCloud):
            result = dispatcher.RealtimeBackendDispatcher(Manager("elevenlabs"))
        self.assertIsInstance(result, FakeCloud)

    def test_registry_key_remains_realtime_ws(self):
        self.assertEqual(dispatcher.RealtimeBackendDispatcher.name, "realtime-ws")


if __name__ == "__main__":
    unittest.main()
