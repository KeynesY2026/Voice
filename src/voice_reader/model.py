from pathlib import Path

from indextts.infer_v2_5 import IndexTTS2, QwenEmotion
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    ) -> None:
        super().__init__(
            cfg_path=cfg_path, model_dir=model_dir, device=device,
            use_bf16=use_bf16, use_qwen_emo=False, use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=False, use_accel=False, use_torch_compile=False,
        )
        if use_qwen_emo:
            self.qwen_emo = load_emotion_model(
                str(Path(model_dir) / self.cfg.qwen_emo_path), device,
            )
