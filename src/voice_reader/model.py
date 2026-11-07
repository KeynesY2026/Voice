import logging
from pathlib import Path

import torch
from torch import nn
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.infer_v2_5 import IndexTTS2, QwenEmotion
from transformers import AutoModelForCausalLM, AutoTokenizer

from voice_reader.device import RuntimeMode

logger = logging.getLogger(__name__)


class IncompleteSpeechError(RuntimeError):
    pass


def select_runtime(device: str, runtime: RuntimeMode) -> str:
    if runtime not in {"auto", "standard", "low-vram"}:
        raise ValueError("Unsupported runtime; choose auto, standard, or low-vram.")
    if runtime == "standard" or (runtime == "auto" and device == "cpu"):
        return "standard"
    if not device.startswith("cuda"):
        raise ValueError("The low-vram runtime requires CUDA.")
    if runtime == "auto" and torch.cuda.get_device_properties(device).total_memory >= 8 * 1024**3:
        return "standard"
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "The low-vram runtime requires BF16 support. Select the standard runtime "
            "or explicitly use CPU on this GPU."
        )
    return "low-vram"


class CPUConditionModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.cpu().eval()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features.to("cpu")).to(features.device)


class BoundedEmotionModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(self, **kwargs) -> torch.Tensor:
        budget = min(kwargs.get("max_new_tokens", 256), 256)
        kwargs["max_new_tokens"] = budget
        result = self.model.generate(**kwargs)
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("Emotion analysis returned an unsupported generation result.")
        if result.shape[-1] - kwargs["input_ids"].shape[-1] >= budget:
            raise RuntimeError("Emotion analysis exceeded its token budget without terminating.")
        return result


class CachedGPT(nn.Module):
    def __init__(self, model: UnifiedVoice) -> None:
        super().__init__()
        self.model = model
        self._conditions: list[tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]] = []

    @property
    def text_pos_embedding(self) -> nn.Module:
        return self.model.text_pos_embedding

    def inference_speech(self, *args, **kwargs):
        output, latent = self.model.inference_speech(*args, **kwargs)
        codes = output if isinstance(output, torch.Tensor) else output.sequences
        if codes.shape[-1] == 0 or (codes[:, -1] != self.model.stop_mel_token).any():
            raise IncompleteSpeechError("Speech generation reached its token budget without terminating.")
        return output, latent

    def _emovec(self, condition: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if condition.is_inference():
            return self.model.get_emovec(condition, lengths)
        for cached, version, cached_lengths, result in self._conditions:
            if cached is condition and version == condition._version and torch.equal(lengths, cached_lengths):
                return result
        result = self.model.get_emovec(condition, lengths)
        self._conditions = [entry for entry in self._conditions if entry[0] is not condition]
        self._conditions.append((condition, condition._version, lengths.clone(), result))
        self._conditions = self._conditions[-2:]
        return result

    def merge_emovec(
        self, speech_conditioning_latent: torch.Tensor, emo_speech_conditioning_latent: torch.Tensor,
        cond_lengths: torch.Tensor, emo_cond_lengths: torch.Tensor, alpha: float = 1.0,
    ) -> torch.Tensor:
        emotion = self._emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base = self._emovec(speech_conditioning_latent, cond_lengths)
        return base + alpha * (emotion - base)


class DeviceQwenEmotion(QwenEmotion):
    """Keep upstream inference, but prevent its device_map='auto' from ignoring CPU."""

    def __init__(self, model_dir: str, device: str) -> None:
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype="float16", device_map={"": device},
        )
        self.prompt = "\u6587\u672c\u60c5\u611f\u5206\u7c7b"
        self.desired_vector_order = [
            "\u9ad8\u5174", "\u6124\u6012", "\u60b2\u4f24", "\u6050\u60e7",
            "\u53cd\u611f", "\u4f4e\u843d", "\u60ca\u8bb6", "\u81ea\u7136",
        ]
        self.cn_key_to_en = dict(zip(self.desired_vector_order, (
            "happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm",
        )))
        self.melancholic_words = {
            "\u4f4e\u843d", "melancholy", "melancholic", "depression", "depressed", "gloomy",
        }
        self.min_score = 0.0
        self.max_score = 1.2


def load_emotion_model(model_dir: str, device: str) -> QwenEmotion:
    return DeviceQwenEmotion(model_dir, device)


class ConfiguredIndexTTS(IndexTTS2):
    def __init__(
        self,
        *,
        cfg_path: str,
        model_dir: str,
        device: str,
        use_bf16: bool,
        use_qwen_emo: bool,
        use_cuda_kernel: bool,
        runtime: RuntimeMode = "auto",
    ) -> None:
        self.runtime = select_runtime(device, runtime)
        low_vram = self.runtime == "low-vram"
        if low_vram:
            torch.set_num_threads(min(4, torch.get_num_threads()))
            torch.cuda.set_per_process_memory_fraction(0.90, device)
        super().__init__(
            cfg_path=cfg_path, model_dir=model_dir, device="cpu" if low_vram else device,
            use_bf16=use_bf16, use_qwen_emo=False,
            use_cuda_kernel=False if low_vram else use_cuda_kernel,
            use_deepspeed=False, use_accel=False, use_torch_compile=False,
        )
        if low_vram:
            self._configure_low_vram(device)
            print(">> Runtime: low-vram; BF16 GPT and FP32 audio models on CUDA; "
                  "TF32 matmul, 32-token segments, CPU reference/emotion (no weight offload).")
        if use_qwen_emo:
            self.qwen_emo = load_emotion_model(
                str(Path(model_dir) / self.cfg.qwen_emo_path), "cpu" if low_vram else device,
            )
            if low_vram:
                self.qwen_emo.model = BoundedEmotionModel(self.qwen_emo.model)

    def _configure_low_vram(self, device: str) -> None:
        self.device = device
        self.use_bf16 = True
        self.dtype = torch.bfloat16
        self.low_vram = False
        self.segment_budget = 32
        # Convert on CPU first so an FP32 copy never occupies scarce VRAM.
        self.gpt = self.gpt.bfloat16().to(device)
        self.gpt = CachedGPT(self.gpt)
        self.semantic_codec = self.semantic_codec.to(device)
        self.s2mel = self.s2mel.to(device)
        # Upstream caches are ordinary tensors, not registered buffers.
        transformer = self.s2mel.models["cfm"].estimator.transformer
        transformer.freqs_cis = transformer.freqs_cis.to(device)
        transformer.causal_mask = transformer.causal_mask.to(device)
        self.bigvgan = self.bigvgan.to(device)
        self.campplus_model = CPUConditionModel(self.campplus_model)
        self.emo_matrix = tuple(matrix.to(device) for matrix in self.emo_matrix)
        self.spk_matrix = tuple(matrix.to(device) for matrix in self.spk_matrix)

    def get_emb(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_device = next(self.semantic_model.parameters()).device
        return super().get_emb(
            input_features.to(encoder_device), attention_mask.to(encoder_device),
        ).to(self.device)

    @torch.no_grad()
    def infer(self, *args, **kwargs):
        if self.runtime != "low-vram":
            return super().infer(*args, **kwargs)
        requested = args[12] if len(args) > 12 else kwargs.get("max_text_tokens_per_segment", 32)
        initial = min(requested, self.segment_budget)
        budgets = list(dict.fromkeys((initial, min(initial, 24), min(initial, 16))))
        kwargs["max_mel_tokens"] = min(kwargs.get("max_mel_tokens", 700), 700)
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            for index, budget in enumerate(budgets):
                self.segment_budget = budget
                if len(args) > 12:
                    attempt_args = (*args[:12], budget, *args[13:])
                else:
                    attempt_args = args
                    kwargs["max_text_tokens_per_segment"] = budget
                try:
                    return super().infer(*attempt_args, **kwargs)
                except (torch.cuda.OutOfMemoryError, IncompleteSpeechError) as error:
                    if index == len(budgets) - 1:
                        raise
                    logger.warning(
                        "Generation hit a memory or token limit (%s); retrying the complete "
                        "request with %d-token segments, without CPU offload.",
                        error, budgets[index + 1],
                    )
                torch.cuda.empty_cache()
        finally:
            torch.set_float32_matmul_precision(precision)
