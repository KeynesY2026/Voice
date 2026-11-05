from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from voice_reader.config import PRIVATE_DATA_DIR
from voice_reader.reference import prepare_profile_reference

MAX_AUDIO_BYTES = 100 * 1024 * 1024
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac")


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    name: str
    created_at: str
    source_extension: str
    source_sha256: str
    sample_rate: int
    duration_seconds: float
    reference_path: Path
    schema_version: int = 1


class ProfileStore:
    def __init__(self, root: Path = PRIVATE_DATA_DIR / "profiles") -> None:
        self.root = Path(root).resolve()
        self._lock = threading.RLock()

    def _directory(self, profile_id: str) -> Path:
        try:
            if str(UUID(profile_id)) != profile_id:
                raise ValueError
        except (ValueError, AttributeError) as error:
            raise ValueError("Invalid profile ID.") from error
        directory = self.root / profile_id
        if directory.resolve() != directory:
            raise ValueError("Profile directory must not be a symbolic link or junction.")
        return directory

    def _name(self, name: str, exclude: str | None = None) -> str:
        name = name.strip()
        if not name:
            raise ValueError("Profile name must not be empty.")
        if any(p.id != exclude and p.name.casefold() == name.casefold() for p in self.list_profiles()):
            raise ValueError(f"A profile named '{name}' already exists.")
        return name

    def list_profiles(self) -> list[VoiceProfile]:
        with self._lock:
            if not self.root.exists():
                return []
            profiles = [
                self.get(path.name) for path in self.root.iterdir()
                if not path.name.startswith(".pending-")
            ]
            return sorted(profiles, key=lambda profile: (profile.name.casefold(), profile.id))

    def get(self, profile_id: str) -> VoiceProfile:
        with self._lock:
            directory = self._directory(profile_id)
            metadata = directory / "profile.json"
            if not directory.is_dir():
                raise FileNotFoundError(f"Profile not found: {profile_id}")
            try:
                if metadata.resolve() != metadata:
                    raise ValueError("Linked metadata is not allowed.")
                data = json.loads(metadata.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("id") != profile_id:
                    raise ValueError("Invalid schema or ID.")
                for field in ("name", "created_at", "source_extension", "source_sha256"):
                    if not isinstance(data.get(field), str) or not data[field].strip():
                        raise ValueError(f"Invalid {field}.")
                if data["source_extension"] not in AUDIO_EXTENSIONS:
                    raise ValueError("Invalid source extension.")
                if len(data["source_sha256"]) != 64 or any(
                    char not in "0123456789abcdef" for char in data["source_sha256"]
                ):
                    raise ValueError("Invalid source digest.")
                datetime.fromisoformat(data["created_at"])
                reference = directory / "reference.wav"
                source = directory / ("source" + data["source_extension"])
                for path in (reference, source):
                    if path.resolve() != path or not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f"Missing or linked audio: {path.name}")
                with wave.open(str(reference), "rb") as audio:
                    duration = audio.getnframes() / audio.getframerate()
                    if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 24000):
                        raise ValueError("Invalid reference WAV format.")
                    if not 3 <= duration <= 30:
                        raise ValueError("Invalid reference duration.")
                    if len(audio.readframes(audio.getnframes())) != audio.getnframes() * 2:
                        raise ValueError("Truncated reference WAV.")
                if not 3 <= duration <= 30 or data.get("duration_seconds") != duration or data.get("sample_rate") != 24000:
                    raise ValueError("Invalid reference duration or sample rate.")
                return VoiceProfile(
                    id=profile_id, name=data["name"], created_at=data["created_at"],
                    source_extension=data["source_extension"], source_sha256=data["source_sha256"],
                    sample_rate=24000, duration_seconds=duration, reference_path=reference,
                )
            except (OSError, ValueError, wave.Error, EOFError) as error:
                raise ValueError(f"Profile {profile_id} is invalid: {error}") from error

    @staticmethod
    def _write_metadata(profile: VoiceProfile, directory: Path) -> None:
        data = asdict(profile)
        del data["reference_path"]
        temporary = directory / "profile.json.tmp"
        try:
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(directory / "profile.json")
        finally:
            temporary.unlink(missing_ok=True)

    def create(self, name: str, source: Path) -> VoiceProfile:
        with self._lock:
            name = self._name(name)
            source = Path(source)
            if not source.is_file():
                raise FileNotFoundError(f"Reference audio not found: {source}")
            extension = source.suffix.lower()
            if extension not in AUDIO_EXTENSIONS:
                raise ValueError("Profile audio must be WAV, MP3, M4A, or FLAC.")
            if not 0 < source.stat().st_size <= MAX_AUDIO_BYTES:
                raise ValueError("Profile audio must be non-empty and at most 100 MiB.")
            self.root.mkdir(parents=True, exist_ok=True)
            profile_id = str(uuid4())
            directory = self._directory(profile_id)
            staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
            try:
                copied = staging / ("source" + extension)
                digest = hashlib.sha256()
                size = 0
                with source.open("rb") as reader, copied.open("wb") as writer:
                    for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                        size += len(chunk)
                        if size > MAX_AUDIO_BYTES:
                            raise ValueError("Profile audio must be at most 100 MiB.")
                        digest.update(chunk)
                        writer.write(chunk)
                duration = prepare_profile_reference(copied, staging / "reference.wav")
                profile = VoiceProfile(
                    profile_id, name, datetime.now(timezone.utc).isoformat(), extension,
                    digest.hexdigest(), 24000, duration, directory / "reference.wav",
                )
                self._write_metadata(profile, staging)
                staging.rename(directory)
                return profile
            finally:
                if staging.exists():
                    shutil.rmtree(staging)

    def rename(self, profile_id: str, name: str) -> VoiceProfile:
        from dataclasses import replace

        with self._lock:
            profile = self.get(profile_id)
            updated = replace(profile, name=self._name(name, exclude=profile_id))
            self._write_metadata(updated, self._directory(profile_id))
            return updated

    def delete(self, profile_id: str) -> None:
        with self._lock:
            self.get(profile_id)
            shutil.rmtree(self._directory(profile_id))
