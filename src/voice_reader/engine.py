from __future__ import annotations

import math
import re
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice_reader.model import ConfiguredIndexTTS

from voice_reader.config import CACHE_DIR, CHECKPOINT_DIR, DEFAULT_REFERENCE
from voice_reader.device import DeviceInfo, DeviceMode, RuntimeMode, resolve_device
from voice_reader.profiles import ProfileStore
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


def _join_paragraph_audio(paths: list[Path], output: Path, pauses: list[float]) -> None:
    with wave.open(str(paths[0]), "rb") as first:
        channels, width, rate, _, compression, _ = first.getparams()
    if compression != "NONE" or width not in {1, 2, 3, 4}:
        raise RuntimeError("Paragraph audio must use uncompressed PCM.")
    silence_byte = b"\x80" if width == 1 else b"\x00"
    with wave.open(str(output), "wb") as joined:
        joined.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        for index, path in enumerate(paths):
            with wave.open(str(path), "rb") as source:
                if (source.getnchannels(), source.getsampwidth(), source.getframerate(),
                    source.getcomptype()) != (channels, width, rate, compression):
                    raise RuntimeError("Paragraph audio formats do not match.")
                if source.getnframes() == 0:
                    raise RuntimeError("Paragraph generation returned empty audio.")
                if index:
                    pause = silence_byte * (channels * width * round(rate * pauses[index - 1]))
                    joined.writeframesraw(pause)
                while frames := source.readframes(65536):
                    joined.writeframesraw(frames)


def _load_index_tts() -> type[ConfiguredIndexTTS]:
    from voice_reader.model import ConfiguredIndexTTS

    return ConfiguredIndexTTS


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
        device: DeviceMode = "auto",
        profile_store: ProfileStore | None = None,
        runtime: RuntimeMode = "auto",
    ) -> None:
        if runtime not in {"auto", "standard", "low-vram"}:
            raise ValueError("Unsupported runtime; choose auto, standard, or low-vram.")
        self.reference_path = Path(reference_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.runtime = runtime
        self.profile_store = profile_store if profile_store is not None else ProfileStore()
        self._device_info: DeviceInfo | None = None
        self._lock = threading.RLock()
        self._model: ConfiguredIndexTTS | None = None

    @property
    def device_info(self) -> DeviceInfo:
        with self._lock:
            if self._device_info is None:
                self._device_info = resolve_device(self.device)
            return self._device_info

    @property
    def runtime_description(self) -> str:
        with self._lock:
            selected = self._model.runtime if self._model is not None else self.runtime
            if selected == "low-vram":
                budget = self._model.segment_budget if self._model is not None else 32
                return (
                    f"Runtime: low-vram (BF16 GPT / TF32 matmul; {budget}-token segments; "
                    "CPU reference/emotion; resident CUDA audio models)."
                )
            if self._model is None:
                return f"Runtime: {selected} (selected when the model loads)."
            return "Runtime: standard."

    def _get_model(self) -> ConfiguredIndexTTS:
        if self._model is None:
            self._model = _load_index_tts()(
                cfg_path=str(self.checkpoint_dir / "config.yaml"),
                model_dir=str(self.checkpoint_dir),
                use_bf16=False,
                use_qwen_emo=True,
                device=self.device_info.resolved,
                use_cuda_kernel=False,
                runtime=self.runtime,
            )
        return self._model

    def _get_reference_audio(self, profile_id: str | None) -> Path:
        if profile_id is not None:
            return self.profile_store.get(profile_id).reference_path
        try:
            return prepare_reference(self.reference_path, self.cache_dir)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Default reference audio is missing. Create and select a voice profile "
                f"or provide {self.reference_path}."
            ) from error

    def synthesize(
        self,
        text: str,
        output_path: Path,
        emotion: EmotionSettings | None = None,
        *,
        profile_id: str | None = None,
    ) -> Path:
        if not text.strip():
            raise ValueError("Text must not be empty.")
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".wav":
            raise ValueError("Output path must end in .wav.")
        if emotion is None:
            emotion = EmotionSettings()
        emotion.validate()
        with self._lock:
            resolved_output = output_path.resolve()
            if resolved_output.is_relative_to(self.profile_store.root):
                raise ValueError("Output cannot be inside the voice profile directory.")
            if resolved_output == self.reference_path.resolve():
                raise ValueError("Output cannot overwrite the default reference recording.")
            reference = self._get_reference_audio(profile_id)
            if resolved_output == reference.resolve():
                raise ValueError("Output cannot overwrite the reference recording.")
            model = self._get_model()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.unlink(missing_ok=True)
            def generate(part: str, destination: Path) -> None:
                if emotion.mode == "auto":
                    model.infer(
                        spk_audio_prompt=str(reference), text=part, lang="EN",
                        output_path=str(destination), emo_alpha=emotion.strength, use_emo_text=True,
                    )
                else:
                    assert emotion.vector is not None
                    model.infer(
                        spk_audio_prompt=str(reference), text=part, lang="EN",
                        output_path=str(destination), emo_alpha=emotion.strength,
                        emo_vector=list(emotion.vector),
                    )
                if not destination.is_file() or not destination.stat().st_size:
                    raise RuntimeError("IndexTTS completed without creating a non-empty WAV.")

            paragraphs: list[str] = []
            pauses: list[float] = []
            pending_pause = 0.0
            for index, part in enumerate(re.split(r"(\r\n|\r|\n| {2,})", text)):
                if index % 2:
                    pause = 0.5 if "\r" in part or "\n" in part else len(part) / 10
                    pending_pause = max(pending_pause, pause)
                elif part.strip():
                    if paragraphs:
                        pauses.append(pending_pause)
                    paragraphs.append(part.strip())
                    pending_pause = 0.0
            if len(paragraphs) <= 1:
                generate(paragraphs[0], output_path)
            else:
                with TemporaryDirectory(prefix=".paragraphs-", dir=output_path.parent) as temporary:
                    paths = []
                    for index, paragraph in enumerate(paragraphs):
                        print(f">> Explicit text block {index + 1}/{len(paragraphs)}: {paragraph}")
                        path = Path(temporary) / f"{index}.wav"
                        generate(paragraph, path)
                        paths.append(path)
                    joined = Path(temporary) / "joined.wav"
                    try:
                        _join_paragraph_audio(paths, joined, pauses)
                    except (wave.Error, EOFError) as error:
                        raise RuntimeError(f"Invalid paragraph WAV: {error}") from error
                    joined.replace(output_path)
            if not output_path.is_file() or not output_path.stat().st_size:
                raise RuntimeError("IndexTTS completed without creating a non-empty WAV.")
        return output_path
