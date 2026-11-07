import math
import secrets
from pathlib import Path

from voice_reader.config import PRIVATE_DATA_DIR
from voice_reader.engine import EMOTION_CATEGORIES
from voice_reader.language import LANGUAGE_CHOICES

DEFAULT_PREFERENCES = dict(
    text="", profile="builtin-male", mode="auto", category="calm",
    strength=0.4, language="auto", device="auto", advanced=False,
)


def load_browser_secret(path: Path = PRIVATE_DATA_DIR / "browser-state.key") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            with path.open("x", encoding="ascii") as output:
                output.write(secrets.token_hex(32))
        except FileExistsError:
            pass
    value = path.read_text(encoding="ascii")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("Invalid browser-state encryption key; restore or remove the local key file.")
    return value


def restore_preferences(saved: object, profiles: set[str], devices: set[str]) -> tuple[dict, str]:
    restored = dict(DEFAULT_PREFERENCES)
    if saved is None:
        return restored, ""
    if not isinstance(saved, dict):
        return restored, "Saved browser settings were invalid and have been reset."
    choices = dict(
        profile=profiles, device=devices, mode={"auto", "manual"},
        category=set(EMOTION_CATEGORIES), language={code for _, code in LANGUAGE_CHOICES},
    )
    invalid = []
    for field, value in saved.items():
        if field not in restored:
            continue
        if field == "text":
            valid = isinstance(value, str)
        elif field == "advanced":
            valid = isinstance(value, bool)
        elif field == "strength":
            valid = (isinstance(value, (float, int)) and not isinstance(value, bool)
                     and math.isfinite(value) and 0 <= value <= 1)
        else:
            valid = isinstance(value, str) and value in choices[field]
        if valid:
            restored[field] = value
        else:
            invalid.append(field)
    return restored, (
        f"Unavailable or invalid saved settings reset: {', '.join(invalid)}." if invalid else ""
    )
