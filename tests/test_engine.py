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

    first = speech.synthesize("Hello there.", tmp_path / "first.wav", EmotionSettings(), advanced=True)
    second = speech.synthesize("How are you?", tmp_path / "second.wav", EmotionSettings(), advanced=True)

    assert first.read_bytes() == b"wav"
    assert second.read_bytes() == b"wav"
    assert len(instances) == 1
    assert instances[0]["use_qwen_emo"] is True
    assert calls[0]["use_emo_text"] is True
    assert calls[0]["emo_alpha"] == 0.4
    assert calls[0]["lang"] == "EN"


def test_language_is_resolved_per_explicit_block(tmp_path, monkeypatch):
    import wave

    calls = []

    class Model:
        def __init__(self, **kwargs):
            pass

        def infer(self, **kwargs):
            calls.append(kwargs)
            with wave.open(kwargs["output_path"], "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0\0" * 240)

    monkeypatch.setattr(engine, "_load_index_tts", lambda: Model)
    monkeypatch.setattr(engine, "prepare_reference", lambda *args: tmp_path / "reference.wav")
    speech = engine.SpeechEngine()
    speech.synthesize("Hello there.\n这是中文。", tmp_path / "auto.wav", language="auto", advanced=True)
    assert [call["lang"] for call in calls] == ["EN", "ZH"]
    calls.clear()
    speech.synthesize("Hello，这是混合文本。", tmp_path / "forced.wav", language="en")
    assert calls[0]["text"] == "Hello，这是混合文本。"
    assert calls[0]["lang"] == "EN"


def test_emotion_strength_defaults_to_point_four():
    assert EmotionSettings().strength == 0.4


@pytest.mark.parametrize("emotion", [
    EmotionSettings(),
    EmotionSettings(mode="manual", vector=(1, 0, 0, 0, 0, 0, 0, 0), strength=0.8),
])
def test_default_fast_mode_ignores_explicit_breaks_annotations_and_emotion(
    tmp_path, monkeypatch, emotion, capsys,
):
    calls = []

    class Model:
        def __init__(self, **kwargs):
            pass

        def infer(self, **kwargs):
            calls.append(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"wav")

    monkeypatch.setattr(engine, "_load_index_tts", lambda: Model)
    monkeypatch.setattr(engine, "prepare_reference", lambda *args: tmp_path / "reference.wav")
    speech = engine.SpeechEngine()
    speech.synthesize("Read <minute|M IH1 . N AH0 T>.   Next.\nLast.", tmp_path / "fast.wav", emotion)
    assert len(calls) == 1
    assert calls[0]["text"] == "Read minute. Next. Last."
    assert not calls[0].get("use_emo_text", False)
    assert calls[0].get("emo_vector") is None
    assert "Fast mode" in capsys.readouterr().out


def test_advanced_mode_preserves_pronunciation_annotation(tmp_path, monkeypatch):
    calls = []

    class Model:
        def __init__(self, **kwargs):
            pass

        def infer(self, **kwargs):
            calls.append(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"wav")

    monkeypatch.setattr(engine, "_load_index_tts", lambda: Model)
    monkeypatch.setattr(engine, "prepare_reference", lambda *args: tmp_path / "reference.wav")
    engine.SpeechEngine().synthesize(
        "A <minute|M IH1 . N AH0 T>.", tmp_path / "advanced.wav", advanced=True,
    )
    assert calls[0]["text"] == "A <minute|M IH1 . N AH0 T>."
    assert calls[0]["use_emo_text"] is True


def test_built_in_profiles_use_bundled_audio_without_private_reference(monkeypatch):
    from voice_reader.builtin_profiles import BUILTIN_PROFILES

    def unexpected(*args):
        raise AssertionError("Built-in voices must not prepare the private reference.")

    monkeypatch.setattr(engine, "prepare_reference", unexpected)
    speech = engine.SpeechEngine()
    for profile_id, profile in BUILTIN_PROFILES.items():
        assert speech._get_reference_audio(profile_id) == profile.reference_path


def test_runtime_selection_reaches_loaded_model_and_status(tmp_path, monkeypatch):
    class Model:
        def __init__(self, *, runtime, **kwargs):
            self.runtime = runtime

        def infer(self, **kwargs):
            Path(kwargs["output_path"]).write_bytes(b"wav")

    monkeypatch.setattr(engine, "_load_index_tts", lambda: Model)
    monkeypatch.setattr(engine, "prepare_reference", lambda *args: tmp_path / "reference.wav")
    speech = engine.SpeechEngine(runtime="standard")
    speech.synthesize("Hello.", tmp_path / "speech.wav")
    assert "standard" in speech.runtime_description


def test_invalid_runtime_is_rejected_before_loading():
    with pytest.raises(ValueError, match="runtime"):
        engine.SpeechEngine(runtime="invalid")


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
        advanced=True,
    )

    assert calls[0]["emo_vector"] == [0, 0, 0, 0, 0, 0, 1, 0]
    assert calls[0]["emo_alpha"] == 0.4
    assert "use_emo_text" not in calls[0]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("separator,pause_frames", [
    ("\r\n \r\n", 12000), ("\n", 12000), ("\r", 12000),
    ("  ", 4800), ("   ", 7200), ("    ", 9600), ("     ", 12000),
    ("      ", 14400), ("  \n  ", 12000), ("\n  ", 12000),
    ("      \n  ", 14400),
])
def test_explicit_breaks_generate_independently_with_exact_pause(
    tmp_path, monkeypatch, device, automatic, separator, pause_frames,
):
    import wave

    calls = []
    instances = []

    class Model:
        def __init__(self, **kwargs):
            instances.append(kwargs)

        def infer(self, **kwargs):
            calls.append(kwargs)
            value = b"\x01\x00" if len(calls) == 1 else b"\x02\x00"
            with wave.open(kwargs["output_path"], "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                wav.writeframes(value * 240)

    monkeypatch.setattr(engine, "_load_index_tts", lambda: Model)
    monkeypatch.setattr(engine, "prepare_reference", lambda *args: tmp_path / "reference.wav")
    speech = engine.SpeechEngine(device=device)
    emotion = EmotionSettings() if automatic else EmotionSettings(
        mode="manual", vector=(0, 0, 0, 0, 0, 0, 0, 1), strength=0.4,
    )
    output = speech.synthesize(f" \r\nplanner's R.{separator}Fits require three minutes.\n ",
                               tmp_path / "joined.wav", emotion, advanced=True)
    assert [call["text"] for call in calls] == ["planner's R.", "Fits require three minutes."]
    assert len(instances) == 1
    assert all(call["emo_alpha"] == emotion.strength for call in calls)
    assert all(call.get("use_emo_text", False) == automatic for call in calls)
    assert all(call["spk_audio_prompt"] == str(tmp_path / "reference.wav") for call in calls)
    with wave.open(str(output), "rb") as wav:
        assert wav.getnframes() == 480 + pause_frames
        assert wav.readframes(480 + pause_frames) == (
            b"\x01\x00" * 240 + b"\x00\x00" * pause_frames + b"\x02\x00" * 240
        )
    assert not list(tmp_path.glob(".paragraphs-*"))


@pytest.mark.parametrize("failure", ["error", "missing", "rate", "invalid", "empty"])
def test_failed_paragraph_never_returns_partial_audio(tmp_path, monkeypatch, failure):
    import wave

    calls = []

    class Model:
        def __init__(self, **kwargs):
            pass

        def infer(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                if failure == "error":
                    raise RuntimeError("second paragraph failed")
                if failure == "missing":
                    return
                if failure == "invalid":
                    Path(kwargs["output_path"]).write_bytes(b"bad")
                    return
            rate = 22050 if failure == "rate" and len(calls) == 2 else 24000
            with wave.open(kwargs["output_path"], "wb") as wav:
                wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                wav.writeframes(b"" if failure == "empty" and len(calls) == 2 else b"\x01\x00" * 240)

    monkeypatch.setattr(engine, "_load_index_tts", lambda: Model)
    monkeypatch.setattr(engine, "prepare_reference", lambda *args: tmp_path / "reference.wav")
    output = tmp_path / "joined.wav"
    with pytest.raises(RuntimeError):
        engine.SpeechEngine().synthesize("First.\nSecond.", output, advanced=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".paragraphs-*"))


def test_paragraph_join_preserves_stereo_unsigned_pcm(tmp_path):
    import wave

    paths = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for path in paths:
        with wave.open(str(path), "wb") as wav:
            wav.setparams((2, 1, 1000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x90\x70" * 10)
    output = tmp_path / "joined.wav"
    engine._join_paragraph_audio(paths, output, [0.3])
    with wave.open(str(output), "rb") as wav:
        assert wav.getparams()[:4] == (2, 1, 1000, 320)
        assert wav.readframes(320) == b"\x90\x70" * 10 + b"\x80\x80" * 300 + b"\x90\x70" * 10


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
