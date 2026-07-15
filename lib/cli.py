#!/usr/bin/env python3
"""hyprwhspr command-line interface."""

import argparse
import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from src.output_control import OutputController, VerbosityLevel
except ImportError:
    from output_control import OutputController, VerbosityLevel

from cli_commands import (
    backend_repair_command,
    backend_reset_command,
    config_command,
    keyboard_command,
    mic_osd_command,
    model_command,
    noctalia_command,
    omarchy_command,
    record_capture_command,
    record_command,
    setup_command,
    state_reset_command,
    state_show_command,
    state_validate_command,
    status_command,
    systemd_command,
    test_command,
    uninstall_command,
    validate_command,
    waybar_command,
)
from nemotron_commands import main as nemotron_main


def _get_version():
    import subprocess

    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=7"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        prog="hyprwhspr",
        description="hyprwhspr - ferocious speech-to-text for Linux",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_get_version()}"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only show errors"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show debug output"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress indicators",
    )
    parser.add_argument("--log-file", type=str, metavar="PATH")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    setup_parser = subparsers.add_parser("setup", help="Full initial setup")
    setup_parser.add_argument(
        "--python", dest="python_path", metavar="PATH"
    )
    setup_subparsers = setup_parser.add_subparsers(
        dest="setup_action", help="Setup actions"
    )
    auto_parser = setup_subparsers.add_parser(
        "auto", help="Automated setup"
    )
    auto_parser.add_argument(
        "--backend",
        choices=["nvidia", "vulkan", "cpu", "onnx-asr"],
        help="Backend to install (default: auto-detect GPU)",
    )
    auto_parser.add_argument("--model")
    auto_parser.add_argument("--no-waybar", action="store_true")
    auto_parser.add_argument("--no-mic-osd", action="store_true")
    auto_parser.add_argument("--no-systemd", action="store_true")
    auto_parser.add_argument("--hypr-bindings", action="store_true")
    auto_parser.add_argument(
        "--python", dest="python_path", metavar="PATH"
    )

    install_parser = subparsers.add_parser(
        "install", help="Installation management"
    )
    install_subparsers = install_parser.add_subparsers(
        dest="install_action", help="Install actions"
    )
    install_subparsers.add_parser("auto", help=argparse.SUPPRESS)

    config_parser = subparsers.add_parser(
        "config", help="Configuration management"
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_action", help="Config actions"
    )
    config_subparsers.add_parser("init", help="Create default config")
    config_show_parser = config_subparsers.add_parser(
        "show", help="Display current config"
    )
    config_show_parser.add_argument(
        "--all", action="store_true", dest="show_all"
    )
    config_subparsers.add_parser("edit", help="Open config in editor")
    config_subparsers.add_parser(
        "secondary-shortcut",
        help="Configure secondary shortcut and language",
    )
    config_subparsers.add_parser(
        "focused-window", help="Show focused-window identifiers"
    )

    waybar_parser = subparsers.add_parser(
        "waybar", help="Waybar integration"
    )
    waybar_subparsers = waybar_parser.add_subparsers(
        dest="waybar_action", help="Waybar actions"
    )
    waybar_subparsers.add_parser("install")
    waybar_subparsers.add_parser("remove")
    waybar_subparsers.add_parser("status")

    noctalia_parser = subparsers.add_parser(
        "noctalia", help="Noctalia shell integration"
    )
    noctalia_subparsers = noctalia_parser.add_subparsers(
        dest="noctalia_action", help="Noctalia actions"
    )
    noctalia_subparsers.add_parser("install")
    noctalia_subparsers.add_parser("remove")
    noctalia_subparsers.add_parser("status")

    mic_osd_parser = subparsers.add_parser(
        "mic-osd", help="Microphone visualization overlay"
    )
    mic_osd_subparsers = mic_osd_parser.add_subparsers(
        dest="mic_osd_action", help="Mic-OSD actions"
    )
    mic_osd_subparsers.add_parser("enable")
    mic_osd_subparsers.add_parser("disable")
    mic_osd_subparsers.add_parser("status")

    systemd_parser = subparsers.add_parser(
        "systemd", help="Systemd service management"
    )
    systemd_subparsers = systemd_parser.add_subparsers(
        dest="systemd_action", help="Systemd actions"
    )
    for action in ("install", "enable", "disable", "status", "restart"):
        systemd_subparsers.add_parser(action)

    model_parser = subparsers.add_parser(
        "model", help="Model management"
    )
    model_subparsers = model_parser.add_subparsers(
        dest="model_action", help="Model actions"
    )
    model_download_parser = model_subparsers.add_parser("download")
    model_download_parser.add_argument("name", nargs="?", default="base")
    for action in ("list", "status", "unload", "reload"):
        model_subparsers.add_parser(action)

    subparsers.add_parser("status", help="Overall status check")
    subparsers.add_parser("validate", help="Validate installation")

    test_parser = subparsers.add_parser(
        "test", help="Test microphone and backend connectivity"
    )
    test_parser.add_argument("--live", action="store_true")
    test_parser.add_argument("--mic-only", action="store_true")

    keyboard_parser = subparsers.add_parser(
        "keyboard", help="Keyboard device management"
    )
    keyboard_subparsers = keyboard_parser.add_subparsers(
        dest="keyboard_action", help="Keyboard actions"
    )
    for action in ("list", "test", "configure", "detect"):
        keyboard_subparsers.add_parser(action)

    record_parser = subparsers.add_parser(
        "record", help="Control recording"
    )
    record_subparsers = record_parser.add_subparsers(
        dest="record_action", help="Recording actions"
    )
    for action in ("start", "toggle"):
        action_parser = record_subparsers.add_parser(action)
        action_parser.add_argument("--lang", dest="language")
    record_subparsers.add_parser("stop")
    record_subparsers.add_parser("cancel")
    capture_parser = record_subparsers.add_parser("capture")
    capture_parser.add_argument("--lang", dest="language")
    record_subparsers.add_parser("status")

    backend_parser = subparsers.add_parser(
        "backend", help="Backend management"
    )
    backend_subparsers = backend_parser.add_subparsers(
        dest="backend_action", help="Backend actions"
    )
    backend_subparsers.add_parser("repair")
    backend_subparsers.add_parser("reset")

    # Optional local Nemotron management lives in the main CLI namespace while
    # keeping its parser isolated from the already-large setup implementation.
    nemotron_parser = subparsers.add_parser(
        "nemotron",
        help="Install and inspect local Nemotron streaming",
        add_help=False,
    )
    nemotron_parser.add_argument(
        "nemotron_args", nargs=argparse.REMAINDER
    )

    state_parser = subparsers.add_parser(
        "state", help="State management"
    )
    state_subparsers = state_parser.add_subparsers(
        dest="state_action", help="State actions"
    )
    state_subparsers.add_parser("show")
    state_subparsers.add_parser("validate")
    state_reset_parser = state_subparsers.add_parser("reset")
    state_reset_parser.add_argument("--all", action="store_true")

    uninstall_parser = subparsers.add_parser(
        "uninstall", help="Remove hyprwhspr and user data"
    )
    uninstall_parser.add_argument("--keep-models", action="store_true")
    uninstall_parser.add_argument(
        "--remove-permissions", action="store_true"
    )
    uninstall_parser.add_argument(
        "--skip-permissions", action="store_true"
    )
    uninstall_parser.add_argument("--yes", action="store_true")

    args = parser.parse_args()

    if args.quiet:
        OutputController.set_verbosity(VerbosityLevel.QUIET)
    elif args.debug:
        OutputController.set_verbosity(VerbosityLevel.DEBUG)
    elif args.verbose:
        OutputController.set_verbosity(VerbosityLevel.VERBOSE)
    else:
        OutputController.set_verbosity(VerbosityLevel.NORMAL)
    if args.no_progress:
        OutputController.set_progress_enabled(False)
    if args.log_file:
        OutputController.set_log_file(Path(args.log_file))

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "setup":
            if getattr(args, "setup_action", None) == "auto":
                if not omarchy_command(args):
                    sys.exit(1)
            elif getattr(args, "setup_action", None):
                setup_parser.print_help()
                sys.exit(1)
            else:
                setup_command(
                    python_path=getattr(args, "python_path", None)
                )
        elif args.command == "install":
            if not args.install_action:
                install_parser.print_help()
                sys.exit(1)
            if args.install_action == "auto" and not omarchy_command(args):
                sys.exit(1)
        elif args.command == "config":
            if not args.config_action:
                config_parser.print_help()
                sys.exit(1)
            config_command(
                args.config_action,
                show_all=getattr(args, "show_all", False),
            )
        elif args.command == "waybar":
            if not args.waybar_action:
                waybar_parser.print_help()
                sys.exit(1)
            waybar_command(args.waybar_action)
        elif args.command == "noctalia":
            if not args.noctalia_action:
                noctalia_parser.print_help()
                sys.exit(1)
            noctalia_command(args.noctalia_action)
        elif args.command == "mic-osd":
            if not args.mic_osd_action:
                mic_osd_parser.print_help()
                sys.exit(1)
            mic_osd_command(args.mic_osd_action)
        elif args.command == "systemd":
            if not args.systemd_action:
                systemd_parser.print_help()
                sys.exit(1)
            systemd_command(args.systemd_action)
        elif args.command == "model":
            if not args.model_action:
                model_parser.print_help()
                sys.exit(1)
            model_command(
                args.model_action, getattr(args, "name", "base")
            )
        elif args.command == "nemotron":
            nemotron_args = getattr(args, "nemotron_args", None) or ["--help"]
            raise SystemExit(nemotron_main(nemotron_args))
        elif args.command == "status":
            status_command()
        elif args.command == "validate":
            validate_command()
        elif args.command == "test":
            test_command(
                live=getattr(args, "live", False),
                mic_only=getattr(args, "mic_only", False),
            )
        elif args.command == "keyboard":
            if not args.keyboard_action:
                keyboard_parser.print_help()
                sys.exit(1)
            keyboard_command(args.keyboard_action)
        elif args.command == "backend":
            if not args.backend_action:
                backend_parser.print_help()
                sys.exit(1)
            if args.backend_action == "repair":
                backend_repair_command()
            elif args.backend_action == "reset":
                backend_reset_command()
        elif args.command == "state":
            if not args.state_action:
                state_parser.print_help()
                sys.exit(1)
            if args.state_action == "show":
                state_show_command()
            elif args.state_action == "validate":
                state_validate_command()
            elif args.state_action == "reset":
                state_reset_command(getattr(args, "all", False))
        elif args.command == "record":
            if not args.record_action:
                record_parser.print_help()
                sys.exit(1)
            if args.record_action == "capture":
                record_capture_command(
                    language=getattr(args, "language", None)
                )
            else:
                record_command(
                    args.record_action,
                    language=getattr(args, "language", None),
                )
        elif args.command == "uninstall":
            uninstall_command(
                keep_models=getattr(args, "keep_models", False),
                remove_permissions=getattr(
                    args, "remove_permissions", False
                ),
                skip_permissions=getattr(
                    args, "skip_permissions", False
                ),
                yes=getattr(args, "yes", False),
            )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
