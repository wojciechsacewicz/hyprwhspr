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
                        "$schema": "https://raw.githubusercontent.com/goodroot/hyprwhspr/main/share/config.schema.json",
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

            self.assertEqual(saved["$schema"], commands.SCHEMA_URL)
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

    def test_missing_base_runtime_is_rejected(self):
        with mock.patch.object(
            commands, "venv_python", return_value=Path("/missing/venv/python")
        ):
            with self.assertRaisesRegex(RuntimeError, "hyprwhspr setup"):
                commands.ensure_venv()

    def test_runtime_validation_script_selects_requested_provider(self):
        path = Path("/tmp/model")
        cpu_script = commands._runtime_validation_script(path, "cpu")
        cuda_script = commands._runtime_validation_script(path, "cuda")
        self.assertIn("config.clear_providers()", cpu_script)
        self.assertNotIn("append_provider", cpu_script)
        self.assertIn("config.append_provider('cuda')", cuda_script)

    def test_failed_runtime_switch_restores_previous_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "requirements-nemotron.txt").write_text(
                "numpy>=1.26\n", encoding="utf-8"
            )
            calls = []

            def fake_run(command, **kwargs):
                parts = [str(part) for part in command]
                calls.append(parts)
                if ("install" in parts
                        and "onnxruntime-genai-cuda==0.14.1" in parts):
                    raise RuntimeError("install failed")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(commands, "ensure_venv", return_value=Path("/venv/python")), \
                    mock.patch.object(commands, "repo_root", return_value=root), \
                    mock.patch.object(commands, "_installed_version", return_value="0.14.0"), \
                    mock.patch.object(commands, "_run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "install failed"):
                    commands.install_dependencies("cuda")

            self.assertTrue(any(
                "onnxruntime-genai-cuda" in call and "uninstall" in call
                for call in calls
            ))
            self.assertTrue(any(
                "onnxruntime-genai==0.14.0" in call and "install" in call
                for call in calls
            ))

    def test_setup_rolls_runtime_back_before_config_activation(self):
        args = commands.build_parser().parse_args(["setup", "--no-restart"])
        runtime_change = (
            "cuda",
            "onnxruntime-genai-cuda",
            "onnxruntime-genai",
            "0.14.0",
        )
        with mock.patch.object(
            commands, "install_dependencies", return_value=runtime_change
        ), mock.patch.object(
            commands, "resolve_model_path", side_effect=RuntimeError("download failed")
        ), mock.patch.object(
            commands, "ensure_venv", return_value=Path("/venv/python")
        ), mock.patch.object(commands, "_restore_runtime") as restore:
            self.assertEqual(commands.command_setup(args), 1)
        restore.assert_called_once_with(
            Path("/venv/python"),
            "onnxruntime-genai-cuda",
            "onnxruntime-genai",
            "0.14.0",
        )


if __name__ == "__main__":
    unittest.main()
