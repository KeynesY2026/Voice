import argparse
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path

from voice_reader.config import OUTPUT_DIR
from voice_reader.console import configure_utf8_output
from voice_reader.engine import (
    EMOTION_CATEGORIES,
    EmotionSettings,
    SpeechEngine,
    make_emotion_settings,
)
from voice_reader.profiles import ProfileStore
from voice_reader.language import LANGUAGE_CHOICES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-reader")
    commands = parser.add_subparsers(dest="command", required=True)
    speak = commands.add_parser("speak", help="Generate speech without translating the input")
    inputs = speak.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="Text to speak")
    inputs.add_argument("--text-file", type=Path, help="UTF-8 text file to speak")
    inputs.add_argument("--stdin", action="store_true", help="Read text from standard input")
    speak.add_argument("--output", type=Path, default=OUTPUT_DIR / "speech.wav")
    speak.add_argument(
        "--emotion",
        choices=("auto", *EMOTION_CATEGORIES),
        default="auto",
        help="Automatic emotion detection or a manual emotion category (requires --advanced)",
    )
    speak.add_argument("--strength", type=float, default=0.4)
    speak.add_argument("--language", choices=[code for _, code in LANGUAGE_CHOICES], default="auto")
    speak.add_argument("--advanced", action="store_true",
                       help="Enable explicit pauses, pronunciation annotations and emotion controls")
    speak.add_argument("--profile", help="Saved voice profile ID (omit for the default recording)")
    speak.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    speak.add_argument("--runtime", choices=("auto", "standard", "low-vram"), default="auto",
                       help="Automatic small-VRAM optimization, original runtime, or resident CUDA core")
    profiles = commands.add_parser("profiles", help="Manage local voice-cloning profiles")
    actions = profiles.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create", help="Import a 3-30 second authorized recording")
    create.add_argument("--name", required=True)
    create.add_argument("--audio", type=Path, required=True)
    actions.add_parser("list", help="List profile IDs and names")
    rename = actions.add_parser("rename", help="Rename a saved profile")
    rename.add_argument("profile_id")
    rename.add_argument("--name", required=True)
    delete = actions.add_parser("delete", help="Permanently remove a saved profile")
    delete.add_argument("profile_id")
    delete.add_argument("--yes", action="store_true", help="Confirm permanent deletion")
    return parser


def _manage_profiles(args: argparse.Namespace) -> None:
    store = ProfileStore()
    if args.action == "list":
        profiles = store.list_profiles()
        if not profiles:
            print("No profiles. Use 'profiles create' to import a recording.")
        for profile in profiles:
            print(f"{profile.id}\t{profile.name}")
    elif args.action == "create":
        profile = store.create(args.name, args.audio)
        print(f"{profile.id}\t{profile.name}")
    elif args.action == "rename":
        profile = store.rename(args.profile_id, args.name)
        print(f"{profile.id}\t{profile.name}")
    elif args.action == "delete":
        if not args.yes:
            raise ValueError("Deletion requires explicit confirmation: add --yes.")
        store.delete(args.profile_id)
        print(f"Deleted profile {args.profile_id}.")


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_output()
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "profiles":
            _manage_profiles(args)
            return 0
        if not args.advanced and (args.emotion != "auto" or args.strength != 0.4):
            raise ValueError("Emotion overrides require --advanced; fast mode uses the reference expression.")
        if args.text_file is not None:
            text = args.text_file.read_text(encoding="utf-8")
        elif args.stdin:
            text = sys.stdin.read()
        else:
            text = args.text
        category = args.emotion if args.emotion != "auto" else ""
        emotion: EmotionSettings | None = make_emotion_settings(
            "auto" if args.emotion == "auto" else "manual",
            category,
            args.strength,
        ) if args.advanced else None
        speech = SpeechEngine(device=args.device, runtime=args.runtime)
        with redirect_stdout(sys.stderr):
            print(speech.device_info.describe(), file=sys.stderr)
            output = speech.synthesize(
                text, args.output, emotion, profile_id=args.profile, language=args.language,
                advanced=args.advanced,
            )
            print(speech.runtime_description, file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(output)
    return 0
