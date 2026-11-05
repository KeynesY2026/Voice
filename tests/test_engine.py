from pathlib import Path

import pytest

from voice_reader import engine
from voice_reader.engine import EmotionSettings


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
