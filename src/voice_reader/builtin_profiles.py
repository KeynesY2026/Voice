from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class BuiltinProfile:
    name: str
    reference_path: Path


_ASSET_DIR = Path(__file__).parent / "assets"

BUILTIN_PROFILES: Mapping[str, BuiltinProfile] = MappingProxyType(
    {
        "builtin-male": BuiltinProfile(
            name="Male",
            reference_path=_ASSET_DIR / "builtin_male.wave",
        ),
        "builtin-female": BuiltinProfile(
            name="Female",
            reference_path=_ASSET_DIR / "builtin_female.wave",
        ),
    }
)

__all__ = ["BuiltinProfile", "BUILTIN_PROFILES"]
