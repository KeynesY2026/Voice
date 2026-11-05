from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from indextts.infer_v2_5 import IndexTTS2

from voice_reader.config import CACHE_DIR, CHECKPOINT_DIR, DEFAULT_REFERENCE
from voice_reader.reference import prepare_reference

EMOTION_CATEGORIES = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)


def _load_index_tts() -> type[IndexTTS2]:
    from indextts.infer_v2_5 import IndexTTS2

    return IndexTTS2


@dataclass(frozen=True)
class EmotionSettings:
    mode: str = "auto"
    vector: tuple[float, ...] | None = None
    strength: float = 0.6

    def validate(self) -> None:
        if self.mode not in {"auto", "manual"}:
            raise ValueError("Emotion mode must be 'auto' or 'manual'.")
        if not math.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("Emotion strength must be between 0.0 and 1.0.")
        if self.mode == "auto" and self.vector is not None:
            raise ValueError("Automatic emotion mode does not accept a manual vector.")
        if self.mode == "manual":
            if self.vector is None or len(self.vector) != 8:
                raise ValueError("Manual emotion mode requires an eight-value vector.")
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.vector):
                raise ValueError("Emotion vector values must be between 0.0 and 1.0.")


def make_emotion_settings(mode: str, category: str, strength: float) -> EmotionSettings:
    if mode == "auto":
        settings = EmotionSettings(strength=strength)
    elif mode == "manual" and category in EMOTION_CATEGORIES:
        selected = EMOTION_CATEGORIES.index(category)
        vector = tuple(float(index == selected) for index in range(len(EMOTION_CATEGORIES)))
        settings = EmotionSettings(mode="manual", vector=vector, strength=strength)
    else:
        raise ValueError(f"Unsupported emotion selection: {mode}/{category}")
    settings.validate()
    return settings


class SpeechEngine:
    def __init__(
        self,
        reference_path: Path = DEFAULT_REFERENCE,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self.reference_path = Path(reference_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cache_dir = Path(cache_dir)
        self._model: IndexTTS2 | None = None
        self._reference_audio: Path | None = None

    def _get_model(self) -> IndexTTS2:
        if self._model is None:
            self._model = _load_index_tts()(
                cfg_path=str(self.checkpoint_dir / "config.yaml"),
                model_dir=str(self.checkpoint_dir),
                use_bf16=False,
                use_qwen_emo=True,
            )
        return self._model

    def _get_reference_audio(self) -> Path:
        if self._reference_audio is None:
            self._reference_audio = prepare_reference(self.reference_path, self.cache_dir)
        return self._reference_audio

    def synthesize(
        self,
        text: str,
        output_path: Path,
        emotion: EmotionSettings | None = None,
    ) -> Path:
        if not text.strip():
            raise ValueError("Text must not be empty.")
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".wav":
            raise ValueError("Output path must end in .wav.")
        if emotion is None:
            emotion = EmotionSettings()
        emotion.validate()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)

        options = {"emo_alpha": emotion.strength}
        if emotion.mode == "auto":
            options["use_emo_text"] = True
        else:
            options["emo_vector"] = list(emotion.vector)

        self._get_model().infer(
            spk_audio_prompt=str(self._get_reference_audio()),
            text=text,
            lang="EN",
            output_path=str(output_path),
            **options,
        )
        if not output_path.is_file() or not output_path.stat().st_size:
            raise RuntimeError("IndexTTS completed without creating a non-empty WAV.")
        return output_path
