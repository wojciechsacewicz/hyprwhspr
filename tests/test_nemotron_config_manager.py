import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from src import config_manager


class NemotronConfigManagerTests(unittest.TestCase):
    def _manager_for(self, cfg_dir: Path):
        return mock.patch.multiple(
            config_manager,
            CONFIG_DIR=cfg_dir,
            CONFIG_FILE=cfg_dir / "config.json",
        )

    def test_legacy_websocket_config_migrates_to_local_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp) / "hyprwhspr"
            cfg_dir.mkdir(parents=True)
            config_path = cfg_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "$schema": config_manager.ConfigManager.SCHEMA_URL,
                        "transcription_backend": "realtime-ws",
                        "websocket_provider": "nemotron-local",
                        "websocket_model": "legacy-export",
                        "realtime_mode": "transcribe",
                        "language": "pl-PL",
                    }
                ),
                encoding="utf-8",
            )

            with self._manager_for(cfg_dir):
                manager = config_manager.ConfigManager(verbose=False)

            self.assertEqual(
                manager.get_setting("transcription_backend"),
                "nemotron-streaming",
            )
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                saved["$schema"],
                config_manager.ConfigManager.NEMOTRON_SCHEMA_URL,
            )
            self.assertNotIn("websocket_provider", saved)
            self.assertNotIn("websocket_model", saved)
            self.assertNotIn("realtime_mode", saved)

    def test_nemotron_config_keeps_extended_schema_after_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp) / "hyprwhspr"
            with self._manager_for(cfg_dir):
                manager = config_manager.ConfigManager(verbose=False)
                manager.set_setting(
                    "transcription_backend", "nemotron-streaming"
                )
                manager.set_setting("language", "pl-PL")
                self.assertTrue(manager.save_config())

            saved = json.loads(
                (cfg_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                saved["$schema"],
                config_manager.ConfigManager.NEMOTRON_SCHEMA_URL,
            )
            self.assertEqual(
                saved["transcription_backend"], "nemotron-streaming"
            )


if __name__ == "__main__":
    unittest.main()
