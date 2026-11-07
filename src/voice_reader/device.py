from dataclasses import dataclass
from typing import Literal

DeviceMode = Literal["auto", "cpu", "cuda"]
RuntimeMode = Literal["auto", "standard", "low-vram"]


@dataclass(frozen=True)
class DeviceInfo:
    requested: DeviceMode
    resolved: str
    name: str
    reason: str

    def describe(self) -> str:
        return f"Device: {self.resolved} ({self.name}). {self.reason}"


def resolve_device(mode: DeviceMode) -> DeviceInfo:
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported device; choose auto, cpu, or cuda.")
    if mode == "cpu":
        return DeviceInfo(mode, "cpu", "CPU", "CPU explicitly requested.")

    import torch

    if torch.cuda.is_available():
        return DeviceInfo(mode, "cuda:0", torch.cuda.get_device_name(0), "CUDA is available.")
    if mode == "cuda":
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install the gpu extra and a compatible "
            "NVIDIA driver, or explicitly select --device cpu."
        )
    return DeviceInfo(mode, "cpu", "CPU", "CUDA is unavailable; auto selected CPU.")
