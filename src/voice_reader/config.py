from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_DATA_DIR = PROJECT_ROOT / "private-data"
DEFAULT_REFERENCE = PRIVATE_DATA_DIR / "reference.m4a"
CACHE_DIR = PRIVATE_DATA_DIR / "cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PRIVATE_DATA_DIR / "outputs"
