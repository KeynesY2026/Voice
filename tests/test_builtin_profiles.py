import hashlib
import json
import wave
from pathlib import Path

from voice_reader.builtin_profiles import BUILTIN_PROFILES


ASSET_DIR = Path(__file__).parents[1] / "src" / "voice_reader" / "assets"
PROVENANCE_PATH = ASSET_DIR / "builtin_profiles.json"


def test_builtin_profiles_have_fixed_ids_and_names():
    assert set(BUILTIN_PROFILES) == {"builtin-male", "builtin-female"}
    assert BUILTIN_PROFILES["builtin-male"].name == "Male"
    assert BUILTIN_PROFILES["builtin-female"].name == "Female"
    for profile in BUILTIN_PROFILES.values():
        assert profile.reference_path.is_file()


def test_builtin_reference_audio_is_mono_pcm16_24khz_and_short():
    for profile in BUILTIN_PROFILES.values():
        with wave.open(str(profile.reference_path), "rb") as audio:
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getframerate() == 24000
            assert 3 <= audio.getnframes() / audio.getframerate() <= 15
            assert len(audio.readframes(audio.getnframes())) == audio.getnframes() * 2


def test_builtin_audio_digest_and_provenance_are_recorded():
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert provenance["license"] == "CMU ARCTIC voice-specific COPYING"
    license_text = (ASSET_DIR / provenance["license_file"]).read_text(encoding="utf-8")
    assert "Carnegie Mellon University" in license_text
    assert "Permission to use, copy, modify" in license_text
    assert set(provenance["profiles"]) == set(BUILTIN_PROFILES)

    for profile_id, profile in BUILTIN_PROFILES.items():
        source = provenance["profiles"][profile_id]
        assert source["source_corpus"] == "CMU ARCTIC"
        assert source["source_license"] == "CMU ARCTIC voice-specific COPYING"
        assert source["source_speaker"] == {
            "builtin-male": "bdl",
            "builtin-female": "slt",
        }[profile_id]
        assert source["speaker_description"] == {
            "builtin-male": "US English male",
            "builtin-female": "US English female",
        }[profile_id]
        assert source["source_url"].startswith(
            "http://festvox.org/cmu_arctic/cmu_arctic/"
        )
        assert source["source_url"].endswith("/wav/arctic_a0003.wav")
        assert source["source_license_url"].endswith(
            f"cmu_us_{source['source_speaker']}_arctic/COPYING"
        )
        assert len(source["source_sha256"]) == 64
        assert source["resampling"] == "16 kHz mono PCM16 to 24 kHz mono PCM16"
        digest = hashlib.sha256(profile.reference_path.read_bytes()).hexdigest()
        assert digest == source["sha256"]
