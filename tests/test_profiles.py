import json
import wave
from pathlib import Path

import pytest


def write_wav(path: Path, seconds: float = 3) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\0\0" * round(24000 * seconds))
    return path


def test_profile_persists_after_source_removal_and_rename(tmp_path):
    from voice_reader.profiles import ProfileStore

    store = ProfileStore(tmp_path / "profiles")
    source = write_wav(tmp_path / "input.wav")
    profile = store.create(" Narrator ", source)
    source.unlink()
    reloaded = ProfileStore(store.root).get(profile.id)
    assert reloaded.name == "Narrator"
    with wave.open(str(reloaded.reference_path)) as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 72000
    renamed = store.rename(profile.id, "New name")
    assert renamed.id == profile.id
    assert store.list_profiles()[0].name == "New name"
    metadata = json.loads((store.root / profile.id / "profile.json").read_text())
    assert metadata["source_sha256"]
    assert str(tmp_path) not in json.dumps(metadata)


def test_duplicate_names_and_empty_names_are_rejected(tmp_path):
    from voice_reader.profiles import ProfileStore

    store = ProfileStore(tmp_path / "profiles")
    source = write_wav(tmp_path / "input.wav")
    first = store.create("Voice", source)
    second = store.create("Other", source)
    for name in (" voice ", "VOICE"):
        with pytest.raises(ValueError, match="already exists"):
            store.create(name, source)
        with pytest.raises(ValueError, match="already exists"):
            store.rename(second.id, name)
    with pytest.raises(ValueError, match="empty"):
        store.create("  ", source)
    assert store.rename(first.id, "VOICE").name == "VOICE"


@pytest.mark.parametrize("seconds", [2.99, 30.01, 45])
def test_invalid_duration_leaves_no_profile(tmp_path, seconds):
    from voice_reader.profiles import ProfileStore

    store = ProfileStore(tmp_path / "profiles")
    source = write_wav(tmp_path / "input.wav", seconds)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="3.*30"):
        store.create("Invalid", source)
    assert store.list_profiles() == []
    assert list(store.root.iterdir()) == []
    assert source.read_bytes() == original


def test_thirty_seconds_is_accepted(tmp_path):
    from voice_reader.profiles import ProfileStore

    profile = ProfileStore(tmp_path / "profiles").create(
        "Boundary", write_wav(tmp_path / "input.wav", 30)
    )
    assert profile.duration_seconds == 30


@pytest.mark.parametrize("extension", [".mp3", ".m4a", ".flac", ".WAV"])
def test_supported_formats_are_decoded(tmp_path, extension):
    import subprocess

    import imageio_ffmpeg

    from voice_reader.profiles import ProfileStore

    wav = write_wav(tmp_path / "input.wav")
    source = tmp_path / ("converted" + extension)
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(wav), str(source)],
        check=True, capture_output=True,
    )
    profile = ProfileStore(tmp_path / "profiles").create("Decoded", source)
    assert profile.reference_path.is_file()
    assert 3 <= profile.duration_seconds <= 30


def test_invalid_sources_do_not_publish_profiles(tmp_path):
    from voice_reader.profiles import ProfileStore

    store = ProfileStore(tmp_path / "profiles")
    for filename, data in [("bad.wav", b"not audio"), ("empty.wav", b""), ("wrong.txt", b"abc")]:
        source = tmp_path / filename
        source.write_bytes(data)
        with pytest.raises((ValueError, RuntimeError)):
            store.create(filename, source)
    with pytest.raises(FileNotFoundError):
        store.create("Missing", tmp_path / "missing.wav")
    assert store.list_profiles() == []


def test_size_limit_rejects_before_conversion(tmp_path):
    from voice_reader.profiles import ProfileStore

    source = tmp_path / "large.wav"
    with source.open("wb") as audio:
        audio.truncate(100 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="100 MiB"):
        ProfileStore(tmp_path / "profiles").create("Large", source)


def test_exact_size_limit_is_accepted(tmp_path):
    from voice_reader.profiles import ProfileStore

    source = write_wav(tmp_path / "boundary.wav")
    with source.open("r+b") as audio:
        audio.truncate(100 * 1024 * 1024)
    profile = ProfileStore(tmp_path / "profiles").create("Boundary", source)
    assert profile.duration_seconds == 3


def test_delete_removes_only_selected_profile(tmp_path):
    from voice_reader.profiles import ProfileStore

    store = ProfileStore(tmp_path / "profiles")
    source = write_wav(tmp_path / "input.wav")
    first = store.create("First", source)
    second = store.create("Second", source)
    store.delete(first.id)
    assert [item.id for item in store.list_profiles()] == [second.id]
    assert source.is_file()
    with pytest.raises(FileNotFoundError):
        store.get(first.id)
    for invalid in ("..", "../outside", str(tmp_path), ""):
        with pytest.raises(ValueError, match="ID"):
            store.delete(invalid)


@pytest.mark.parametrize("corruption", ["json", "version", "id", "missing_wav", "truncated_wav"])
def test_corrupt_profile_is_reported(tmp_path, corruption):
    from voice_reader.profiles import ProfileStore

    store = ProfileStore(tmp_path / "profiles")
    profile = store.create("Voice", write_wav(tmp_path / "input.wav"))
    metadata_path = store.root / profile.id / "profile.json"
    if corruption == "missing_wav":
        profile.reference_path.unlink()
    elif corruption == "truncated_wav":
        with profile.reference_path.open("r+b") as audio:
            audio.truncate(100)
    elif corruption == "json":
        metadata_path.write_text("{", encoding="utf-8")
    else:
        metadata = json.loads(metadata_path.read_text())
        metadata["schema_version" if corruption == "version" else "id"] = 999
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises((ValueError, OSError), match="[Pp]rofile|[Rr]eference"):
        store.list_profiles()


def test_metadata_write_failure_cleans_staging(tmp_path, monkeypatch):
    from voice_reader.profiles import ProfileStore

    store = ProfileStore(tmp_path / "profiles")
    source = write_wav(tmp_path / "input.wav")
    original_write = Path.write_text

    def fail_metadata(path, *args, **kwargs):
        if path.name.startswith("profile.json"):
            raise OSError("disk full")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata)
    with pytest.raises(OSError, match="disk full"):
        store.create("Voice", source)
    assert list(store.root.iterdir()) == []
