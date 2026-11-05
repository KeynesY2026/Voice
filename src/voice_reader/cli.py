import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from voice_reader.config import OUTPUT_DIR
from voice_reader.engine import (
    EMOTION_CATEGORIES,
    EmotionSettings,
    SpeechEngine,
    make_emotion_settings,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-reader")
    commands = parser.add_subparsers(dest="command", required=True)
    speak = commands.add_parser("speak", help="Generate English speech")
    inputs = speak.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="Text to speak")
    inputs.add_argument("--text-file", type=Path, help="UTF-8 text file to speak")
    inputs.add_argument("--stdin", action="store_true", help="Read text from standard input")
    speak.add_argument("--output", type=Path, default=OUTPUT_DIR / "speech.wav")
    speak.add_argument(
        "--emotion",
        choices=("auto", *EMOTION_CATEGORIES),
        default="auto",
        help="Automatic emotion detection or a manual emotion category",
    )
    speak.add_argument("--strength", type=float, default=0.6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.text_file is not None:
            text = args.text_file.read_text(encoding="utf-8")
        elif args.stdin:
            text = sys.stdin.read()
        else:
            text = args.text
        category = args.emotion if args.emotion != "auto" else ""
        emotion: EmotionSettings = make_emotion_settings(
            "auto" if args.emotion == "auto" else "manual",
            category,
            args.strength,
        )
        output = SpeechEngine().synthesize(text, args.output, emotion)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(output)
    return 0
