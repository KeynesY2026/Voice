import pytest


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_emotion_model_loading_uses_explicit_device(device, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from voice_reader.model import load_emotion_model

    calls = []
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda path: object())

    def load(path, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load)
    emotion = load_emotion_model("unused-checkpoints", device)
    assert calls[0]["device_map"] == {"": device}
    assert emotion.convert({"emotion": "happy"})["happy"] == 1
    assert len(emotion.desired_vector_order) == 8


def test_explicit_device_adapter_preserves_pinned_emotion_configuration(monkeypatch):
    from indextts import infer_v2_5
    from indextts.infer_v2_5 import QwenEmotion
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from voice_reader.model import load_emotion_model

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda path: None)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer_v2_5.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: None)
    original = QwenEmotion("unused")
    configured = load_emotion_model("unused", "cpu")
    assert vars(configured) == vars(original)
