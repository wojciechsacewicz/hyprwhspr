import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import nemotron_commands as commands


class NemotronCommandTests(unittest.TestCase):
    def test_cpu_and_cuda_runtime_packages_are_mutually_selected(self):
        self.assertEqual(
            commands.selected_runtime_package("cpu"),
            "onnxruntime-genai==0.14.1",
        )
        self.assertEqual(
            commands.selected_runtime_package("cuda"),
            "onnxruntime-genai-cuda==0.14.1",
        )

    def test_config_update_preserves_values_and_existing_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "$schema": commands.SCHEMA_URL,
                        "primary_shortcut": "SUPER+F8",
                        "language": "de-DE",
                        "websocket_provider": "nemotron-local",
                        "websocket_model": "legacy",
                    }
                ),
                encoding="utf-8",
            )
            backup = commands.update_config(
                path, device="cpu", language=None
            )
            saved = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(saved["primary_shortcut"], "SUPER+F8")
            self.assertEqual(saved["language"], "de-DE")
            self.assertEqual(
                saved["transcription_backend"], "nemotron-streaming"
            )
            self.assertNotIn("websocket_provider", saved)
            self.assertNotIn("websocket_model", saved)
            self.assertTrue(backup and backup.exists())

    def test_explicit_language_updates_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            commands.update_config(
                path, device="cpu", language="pl-PL"
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["language"], "pl-PL")

    def test_model_metadata_accepts_pinned_560ms_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "genai_config.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "nemotron_speech",
                            "sample_rate": 16000,
                            "chunk_samples": 8960,
                        }
                    }
                ),
                encoding="utf-8",
            )
            model = commands.validate_model_config(path)
            self.assertEqual(model["chunk_samples"], 8960)

    def test_model_metadata_rejects_incompatible_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "genai_config.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "nemotron_speech",
                            "sample_rate": 16000,
                            "chunk_samples": 5120,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "chunk_samples"):
                commands.validate_model_config(path)

    def test_auto_device_falls_back_to_cpu_without_nvidia_smi(self):
        with mock.patch.object(commands.shutil, "which", return_value=None):
            self.assertEqual(commands.resolve_device("auto"), "cpu")

    def test_setup_language_defaults_to_preserve_existing_value(self):
        args = commands.build_parser().parse_args(["setup"])
        self.assertIsNone(args.language)
        self.assertEqual(args.device, "auto")


if __name__ == "__main__":
    unittest.main()
