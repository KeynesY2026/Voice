import hashlib
import subprocess
from pathlib import Path

import imageio_ffmpeg


def prepare_reference(source: Path, cache_dir: Path) -> Path:
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Reference audio not found: {source}")

    digest = hashlib.sha256()
    with source.open("rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
            digest.update(chunk)

    cache_dir = Path(cache_dir)
    output = cache_dir / f"{source.stem}-{digest.hexdigest()[:16]}.wav"
    if output.is_file() and output.stat().st_size:
        return output

    cache_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        output.unlink(missing_ok=True)
        details = result.stderr.strip() or "no error details provided"
        raise RuntimeError(f"FFmpeg failed to convert reference audio: {details}")
    if not output.is_file() or not output.stat().st_size:
        output.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg completed without creating a non-empty reference WAV.")
    return output
