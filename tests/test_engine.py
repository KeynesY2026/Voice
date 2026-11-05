from pathlib import Path

import pytest

from voice_reader import engine
from voice_reader.device import DeviceInfo
from voice_reader.engine import EmotionSettings


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    monkeypatch.setattr(
        engine, "resolve_device",
        lambda mode: DeviceInfo(mode, "cpu", "CPU", "Test CPU"),
        raising=False,
    )


def test_auto_synthesis_loads_model_once_and_uses_text_emotion(tmp_path, monkeypatch):
    instances = []
    calls = []

    class FakeIndexTTS:
        def __init__(self, **kwargs):
            instances.append(kwargs)

        def infer(self, **kwargs):
            calls.append(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"wav")

    monkeypatch.setattr(engine, "_load_index_tts", lambda: FakeIndexTTS)
    monkeypatch.setattr(engine, "prepare_reference", lambda source, cache: tmp_path / "speaker.wav")
    speech = engine.SpeechEngine(
        reference_path=tmp_path / "voice.m4a",
        checkpoint_dir=tmp_path / "checkpoints",
        cache_dir=tmp_path / "cache",
    )

    first = speech.synthesize("Hello there.", tmp_path / "first.wav", EmotionSettings())
    second = speech.synthesize("How are you?", tmp_path / "second.wav", EmotionSettings())

    assert first.read_bytes() == b"wav"
    assert second.read_bytes() == b"wav"
    assert len(instances) == 1
    assert instances[0]["use_qwen_emo"] is True
    assert calls[0]["use_emo_text"] is True
    assert calls[0]["emo_alpha"] == 0.6
    assert calls[0]["lang"] == "EN"


def test_manual_synthesis_passes_selected_emotion_vector_and_strength(tmp_path, monkeypatch):
    calls = []

    class FakeIndexTTS:
        def __init__(self, **kwargs):
            pass

        def infer(self, **kwargs):
            calls.append(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"wav")

    monkeypatch.setattr(engine, "_load_index_tts", lambda: FakeIndexTTS)
    monkeypatch.setattr(engine, "prepare_reference", lambda source, cache: tmp_path / "speaker.wav")
    speech = engine.SpeechEngine(
        reference_path=tmp_path / "voice.m4a",
        checkpoint_dir=tmp_path / "checkpoints",
        cache_dir=tmp_path / "cache",
    )

    speech.synthesize(
        "That is wonderful.",
        tmp_path / "manual.wav",
        EmotionSettings(mode="manual", vector=(0, 0, 0, 0, 0, 0, 1, 0), strength=0.4),
    )

    assert calls[0]["emo_vector"] == [0, 0, 0, 0, 0, 0, 1, 0]
    assert calls[0]["emo_alpha"] == 0.4
    assert "use_emo_text" not in calls[0]


@pytest.mark.parametrize(
    "settings",
    [
        EmotionSettings(mode="unknown"),
        EmotionSettings(mode="manual"),
        EmotionSettings(mode="manual", vector=(1, 0)),
        EmotionSettings(mode="auto", vector=(1, 0, 0, 0, 0, 0, 0, 0)),
        EmotionSettings(strength=1.1),
    ],
)
def test_invalid_emotion_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        settings.validate()


def test_invalid_text_and_output_path_are_rejected(tmp_path):
    speech = engine.SpeechEngine(
        reference_path=tmp_path / "voice.m4a",
        checkpoint_dir=tmp_path / "checkpoints",
        cache_dir=tmp_path / "cache",
    )

    with pytest.raises(ValueError, match="Text must not be empty"):
        speech.synthesize("  ", tmp_path / "empty.wav", EmotionSettings())

    with pytest.raises(ValueError, match="Output path must end in .wav"):
        speech.synthesize("Hello.", tmp_path / "output.mp3", EmotionSettings())


def test_inference_errors_are_not_hidden(tmp_path, monkeypatch):
    class FakeIndexTTS:
        def __init__(self, **kwargs):
            pass

        def infer(self, **kwargs):
            raise RuntimeError("model failed")

    monkeypatch.setattr(engine, "_load_index_tts", lambda: FakeIndexTTS)
    monkeypatch.setattr(engine, "prepare_reference", lambda source, cache: tmp_path / "speaker.wav")
    speech = engine.SpeechEngine(
        reference_path=tmp_path / "voice.m4a",
        checkpoint_dir=tmp_path / "checkpoints",
        cache_dir=tmp_path / "cache",
    )

    with pytest.raises(RuntimeError, match="model failed"):
        speech.synthesize("Hello.", tmp_path / "output.wav", EmotionSettings())


def test_profiles_switch_without_reloading_model(tmp_path, monkeypatch):
    import wave

    from voice_reader.profiles import ProfileStore

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 72000)
    store = ProfileStore(tmp_path / "profiles")
    first = store.create("First", source)
    second = store.create("Second", source)
    calls = []
    instances = []

    class Model:
        def __init__(self, **kwargs):
            instances.append(kwargs)

        def infer(self, **kwargs):
            calls.append(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"wav")

    monkeypatch.setattr(engine, "_load_index_tts", lambda: Model)
    speech = engine.SpeechEngine(profile_store=store, device="cpu")
    for profile in (first, second, first):
        speech.synthesize("Hello.", tmp_path / "output.wav", profile_id=profile.id)
    assert len(instances) == 1
    assert instances[0]["device"] == "cpu"
    assert instances[0]["use_cuda_kernel"] is False
    assert instances[0]["use_bf16"] is False
    assert [call["spk_audio_prompt"] for call in calls] == [
        str(first.reference_path), str(second.reference_path), str(first.reference_path)
    ]


def test_invalid_profile_preserves_existing_output_and_does_not_load_model(tmp_path, monkeypatch):
    from voice_reader.profiles import ProfileStore

    def forbidden():
        pytest.fail("A bad profile must be rejected before model loading")

    monkeypatch.setattr(engine, "_load_index_tts", forbidden)
    output = tmp_path / "output.wav"
    output.write_bytes(b"previous")
    speech = engine.SpeechEngine(profile_store=ProfileStore(tmp_path / "profiles"))
    with pytest.raises(ValueError, match="ID"):
        speech.synthesize("Hello.", output, profile_id="missing")
    assert output.read_bytes() == b"previous"


@pytest.mark.parametrize("output_name", ["reference.wav", "source.wav"])
def test_output_cannot_overwrite_saved_profile_audio(tmp_path, monkeypatch, output_name):
    import wave

    from voice_reader.profiles import ProfileStore

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 72000)
    store = ProfileStore(tmp_path / "profiles")
    profile = store.create("Voice", source)
    output = store.root / profile.id / output_name
    before = output.read_bytes()

    class Model:
        def __init__(self, **kwargs):
            pass

        def infer(self, **kwargs):
            Path(kwargs["output_path"]).write_bytes(b"replaced")

    monkeypatch.setattr(engine, "_load_index_tts", lambda: Model)
    speech = engine.SpeechEngine(profile_store=store)
    with pytest.raises(ValueError, match="profile"):
        speech.synthesize("Hello.", output, profile_id=profile.id)
    assert output.read_bytes() == before


def test_output_cannot_overwrite_default_reference(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.write_bytes(b"original recording")
    monkeypatch.setattr(engine, "prepare_reference", lambda source, cache: tmp_path / "cached.wav")

    class Model:
        def __init__(self, **kwargs):
            pass

        def infer(self, **kwargs):
            Path(kwargs["output_path"]).write_bytes(b"replaced")

    monkeypatch.setattr(engine, "_load_index_tts", lambda: Model)
    speech = engine.SpeechEngine(reference_path=source)
    with pytest.raises(ValueError, match="reference"):
        speech.synthesize("Hello.", source)
    assert source.read_bytes() == b"original recording"


def test_explicit_gpu_failure_is_not_retried_on_cpu(tmp_path, monkeypatch):
    loaded = []

    class Model:
        def __init__(self, **kwargs):
            loaded.append(kwargs["device"])
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(engine, "_load_index_tts", lambda: Model)
    monkeypatch.setattr(engine, "prepare_reference", lambda source, cache: tmp_path / "reference.wav")
    monkeypatch.setattr(
        engine, "resolve_device",
        lambda mode: DeviceInfo(mode, "cuda:0", "GPU", "Explicit CUDA"),
        raising=False,
    )
    speech = engine.SpeechEngine(device="cuda")
    with pytest.raises(RuntimeError, match="out of memory"):
        speech.synthesize("Hello.", tmp_path / "output.wav")
    assert loaded == ["cuda:0"]


def test_concurrent_requests_load_once_and_release_lock_after_error(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    instances = []
    entered = threading.Event()
    release = threading.Event()
    calls = []

    class Model:
        def __init__(self, **kwargs):
            instances.append(kwargs)

        def infer(self, **kwargs):
            calls.append(kwargs["text"])
            if kwargs["text"] == "First":
                entered.set()
                assert release.wait(5)
                raise RuntimeError("first request failed")
            Path(kwargs["output_path"]).write_bytes(b"wav")

    monkeypatch.setattr(engine, "_load_index_tts", lambda: Model)
    monkeypatch.setattr(engine, "prepare_reference", lambda source, cache: tmp_path / "ref.wav")
    speech = engine.SpeechEngine(device="cpu")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(speech.synthesize, "First", tmp_path / "first.wav")
        assert entered.wait(5)
        second = pool.submit(speech.synthesize, "Second", tmp_path / "second.wav")
        release.set()
        with pytest.raises(RuntimeError, match="first request failed"):
            first.result(timeout=5)
        assert second.result(timeout=5).read_bytes() == b"wav"
    assert calls == ["First", "Second"]
    assert len(instances) == 1
