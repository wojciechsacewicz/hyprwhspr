import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NemotronCliRoutingTests(unittest.TestCase):
    def test_launcher_routes_nemotron_to_system_cli(self):
        launcher = (ROOT / "bin" / "hyprwhspr").read_text(
            encoding="utf-8"
        )
        self.assertIn("|nemotron)", launcher)
        self.assertIn(
            'exec "$CLI_PYTHON" "$LIB_DIR/cli.py" "$@"', launcher
        )

    def test_main_cli_registers_nemotron_namespace(self):
        cli = (ROOT / "lib" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("from nemotron_commands import main as nemotron_main", cli)
        self.assertIn("subparsers.add_parser(\n        'nemotron'", cli)
        self.assertIn("nemotron_main(", cli)


if __name__ == "__main__":
    unittest.main()
