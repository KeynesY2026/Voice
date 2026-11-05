import hashlib
import subprocess
import wave
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
    _convert_audio(source, output)
    return output


def _convert_audio(source: Path, output: Path, max_duration: int | None = None) -> None:
    limit = ["-t", str(max_duration)] if max_duration is not None else []
    try:
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-i",
                str(source),
                *limit,
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
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        output.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg conversion timed out after 120 seconds.") from error
    if result.returncode:
        output.unlink(missing_ok=True)
        details = result.stderr.strip() or "no error details provided"
        raise RuntimeError(f"FFmpeg failed to convert reference audio: {details}")
    if not output.is_file() or not output.stat().st_size:
        output.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg completed without creating a non-empty reference WAV.")


def prepare_profile_reference(source: Path, output: Path) -> float:
    try:
        _convert_audio(source, output, max_duration=31)
        with wave.open(str(output), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        if not 3 <= duration <= 30:
            raise ValueError(
                "Profile audio must be between 3 and 30 seconds. "
                "Select and apply a crop in the web waveform editor, or supply a 3-30 second recording. "
                "No audio was trimmed automatically."
            )
        return duration
    except (wave.Error, EOFError) as error:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to read profile WAV: {error}") from error
    except (OSError, RuntimeError, ValueError):
        output.unlink(missing_ok=True)
        raise
