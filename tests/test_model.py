import pytest


@pytest.mark.parametrize("runtime", ["standard", "low-vram"])
def test_qwen_loads_only_on_first_automatic_emotion_request(monkeypatch, runtime):
    from types import SimpleNamespace

    from indextts.infer_v2_5 import IndexTTS2
    from voice_reader import model as module

    loads = []

    def initialize(self, **kwargs):
        self.cfg = SimpleNamespace(qwen_emo_path="emotion")
        self.qwen_emo = None
        self.device = "cpu"

    def configure(self, device):
        self.segment_budget = 32

    monkeypatch.setattr(module, "select_runtime", lambda *args: runtime)
    monkeypatch.setattr(module.torch.cuda, "set_per_process_memory_fraction", lambda *args: None)
    monkeypatch.setattr(IndexTTS2, "__init__", initialize)
    monkeypatch.setattr(IndexTTS2, "infer", lambda self, *args, **kwargs: "audio")
    monkeypatch.setattr(module.ConfiguredIndexTTS, "_configure_low_vram", configure)

    def load(path, device):
        loads.append((path, device))
        return SimpleNamespace(model=module.nn.Linear(1, 1))

    monkeypatch.setattr(module, "load_emotion_model", load)
    model = module.ConfiguredIndexTTS(
        cfg_path="config", model_dir="checkpoints", device="cpu",
        use_bf16=False, use_qwen_emo=True, use_cuda_kernel=False,
    )
    assert loads == []
    model.infer(text="Fast mode")
    model.infer(text="Manual mode", emo_vector=[0] * 8)
    assert loads == []
    model.infer(text="Auto mode", use_emo_text=True)
    model.infer(text="Auto mode again", use_emo_text=True)
    assert len(loads) == 1
    assert loads[0][1] == "cpu"
    assert isinstance(model.qwen_emo.model, module.BoundedEmotionModel) == (runtime == "low-vram")


def test_small_gpu_selects_resident_core_but_cpu_keeps_standard_runtime(monkeypatch):
    from types import SimpleNamespace

    import torch

    from voice_reader.model import select_runtime

    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(total_memory=4 * 1024**3))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert select_runtime("cuda:0", "auto") == "low-vram"
    assert select_runtime("cuda:0", "standard") == "standard"
    assert select_runtime("cpu", "auto") == "standard"


def test_low_vram_requires_cuda_bf16_and_rejects_unknown_runtime(monkeypatch):
    import torch

    from voice_reader.model import select_runtime

    with pytest.raises(ValueError, match="runtime"):
        select_runtime("cpu", "invalid")
    with pytest.raises(ValueError, match="CUDA"):
        select_runtime("cpu", "low-vram")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimeError, match="BF16"):
        select_runtime("cuda:0", "low-vram")


def test_cpu_condition_adapter_preserves_values_and_propagates_errors():
    import torch
    from torch import nn

    from voice_reader.model import CPUConditionModel

    layer = nn.Linear(2, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, 3.0]]))
        layer.bias.fill_(1.0)
    adapter = CPUConditionModel(layer)
    assert adapter(torch.tensor([[4.0, 5.0]])).item() == 24.0
    assert next(adapter.parameters()).device.type == "cpu"
    with pytest.raises(RuntimeError):
        adapter(torch.zeros(1, 3))


def test_low_vram_moves_only_core_and_converts_gpt_before_transfer():
    from types import SimpleNamespace

    import torch
    from torch import nn

    from voice_reader.model import ConfiguredIndexTTS

    model = ConfiguredIndexTTS.__new__(ConfiguredIndexTTS)
    model.gpt = nn.Linear(2, 2)
    model.semantic_model = nn.Linear(2, 2)
    model.semantic_codec = nn.Linear(2, 2)
    model.s2mel = nn.Linear(2, 2)
    transformer = nn.Module()
    transformer.freqs_cis = torch.ones(2, 2)
    transformer.causal_mask = torch.ones(2, 2, dtype=torch.bool)
    estimator = nn.Module()
    estimator.transformer = transformer
    cfm = nn.Module()
    cfm.estimator = estimator
    model.s2mel.models = nn.ModuleDict({"cfm": cfm})
    model.bigvgan = nn.Linear(2, 2)
    model.campplus_model = nn.Linear(2, 2)
    model.spk_matrix = (torch.ones(1, 2),)
    model.emo_matrix = (torch.ones(1, 2),)
    model.cfg = SimpleNamespace()
    model._configure_low_vram("cpu")
    assert next(model.gpt.parameters()).dtype == torch.bfloat16
    assert model.semantic_model.weight.dtype == torch.float32
    assert model.s2mel.weight.dtype == torch.float32
    assert model.bigvgan.weight.dtype == torch.float32
    assert model.dtype == torch.bfloat16
    assert model.low_vram is False
    model._configure_low_vram("meta")
    assert transformer.freqs_cis.device.type == "meta"
    assert transformer.causal_mask.device.type == "meta"


def test_reference_encoder_runs_on_its_own_device_and_returns_condition(monkeypatch):
    from types import SimpleNamespace

    import torch
    from torch import nn
    from indextts.infer_v2_5 import IndexTTS2

    from voice_reader.model import ConfiguredIndexTTS

    model = ConfiguredIndexTTS.__new__(ConfiguredIndexTTS)
    model.runtime = "standard"
    model.semantic_model = nn.Linear(2, 2)
    model.device = "cpu"

    def encode(self, features, mask):
        assert features.device == next(self.semantic_model.parameters()).device
        return features + mask.unsqueeze(-1)

    monkeypatch.setattr(IndexTTS2, "get_emb", encode)
    features = torch.tensor([[[1.0, 2.0]]])
    result = model.get_emb(features, torch.tensor([[3]]))
    assert torch.equal(result, torch.tensor([[[4.0, 5.0]]]))


def test_inference_does_not_retain_reference_autograd_graphs(monkeypatch):
    import torch
    from torch import nn
    from indextts.infer_v2_5 import IndexTTS2

    from voice_reader.model import ConfiguredIndexTTS

    layer = nn.Linear(2, 2)
    monkeypatch.setattr(IndexTTS2, "infer", lambda self, **kwargs: layer(torch.ones(1, 2)))
    model = ConfiguredIndexTTS.__new__(ConfiguredIndexTTS)
    model.runtime = "standard"
    result = model.infer(text="Hello.")
    assert result.requires_grad is False
    assert torch.is_grad_enabled()


def test_reference_emotion_cache_reuses_conditions_but_not_strength_or_mutations():
    import torch
    from torch import nn

    from voice_reader.model import CachedGPT

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def get_emovec(self, condition, lengths):
            self.calls += 1
            return condition.sum(dim=-1)

    source = Model()
    cached = CachedGPT(source)
    speaker = torch.tensor([[1.0, 2.0]])
    emotion = torch.tensor([[3.0, 4.0]])
    lengths = torch.tensor([2])
    assert cached.merge_emovec(speaker, emotion, lengths, lengths, 0.5).item() == 5.0
    assert cached.merge_emovec(speaker, emotion, lengths, lengths, 1.0).item() == 7.0
    assert source.calls == 2
    emotion.add_(1.0)
    assert cached.merge_emovec(speaker, emotion, lengths, lengths, 1.0).item() == 9.0
    assert source.calls == 3
    cached.merge_emovec(speaker, emotion, lengths + 1, lengths + 1, 1.0)
    assert source.calls == 5


def test_low_runtime_bounds_tokens_and_restores_precision_after_error(monkeypatch):
    import torch
    from indextts.infer_v2_5 import IndexTTS2

    from voice_reader.model import ConfiguredIndexTTS

    observed = []

    def fail(self, **kwargs):
        observed.append((kwargs, torch.get_float32_matmul_precision()))
        raise RuntimeError("generation failed")

    monkeypatch.setattr(IndexTTS2, "infer", fail)
    model = ConfiguredIndexTTS.__new__(ConfiguredIndexTTS)
    model.runtime = "low-vram"
    model.segment_budget = 32
    before = torch.get_float32_matmul_precision()
    with pytest.raises(RuntimeError, match="generation failed"):
        model.infer(text="Hello.", max_text_tokens_per_segment=120, num_beams=3)
    assert observed[0][0]["max_text_tokens_per_segment"] == 32
    assert observed[0][0]["max_mel_tokens"] == 700
    assert observed[0][0]["num_beams"] == 3
    assert observed[0][1] == "high"
    assert torch.get_float32_matmul_precision() == before


def test_low_runtime_retries_memory_pressure_with_smaller_segments(monkeypatch, caplog):
    import torch
    from indextts.infer_v2_5 import IndexTTS2

    from voice_reader.model import ConfiguredIndexTTS

    attempts = []

    def generate(self, **kwargs):
        attempts.append(kwargs["max_text_tokens_per_segment"])
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("test memory limit")
        return "complete.wav"

    monkeypatch.setattr(IndexTTS2, "infer", generate)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = ConfiguredIndexTTS.__new__(ConfiguredIndexTTS)
    model.runtime = "low-vram"
    model.segment_budget = 32
    with caplog.at_level("WARNING"):
        assert model.infer(text="Long text.") == "complete.wav"
    assert attempts == [32, 24]
    assert model.segment_budget == 24
    assert "retry" in caplog.text.lower()


def test_gpt_rejects_truncated_codes_instead_of_returning_success():
    import torch
    from torch import nn

    from voice_reader.model import CachedGPT, IncompleteSpeechError

    class Model(nn.Module):
        stop_mel_token = 7

        def inference_speech(self, *, complete):
            return torch.tensor([[3, 7 if complete else 4]]), torch.ones(1, 2)

    model = CachedGPT(Model())
    assert model.inference_speech(complete=True)[0].tolist() == [[3, 7]]
    with pytest.raises(IncompleteSpeechError):
        model.inference_speech(complete=False)


def test_emotion_generation_has_a_bounded_budget_and_reports_nontermination():
    import torch
    from torch import nn

    from voice_reader.model import BoundedEmotionModel

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.terminate = True

        def generate(self, *, input_ids, max_new_tokens):
            count = 1 if self.terminate else max_new_tokens
            return torch.cat((input_ids, torch.ones(1, count, dtype=torch.long)), dim=1)

    source = Model()
    bounded = BoundedEmotionModel(source)
    assert bounded.device.type == "cpu"
    assert bounded.generate(input_ids=torch.ones(1, 2), max_new_tokens=32768).shape == (1, 3)
    source.terminate = False
    with pytest.raises(RuntimeError, match="token budget"):
        bounded.generate(input_ids=torch.ones(1, 2), max_new_tokens=32768)


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
