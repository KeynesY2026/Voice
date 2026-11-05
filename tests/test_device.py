import pytest
import torch


@pytest.mark.parametrize(
    ("mode", "available", "expected"),
    [
        ("auto", False, "cpu"),
        ("auto", True, "cuda:0"),
        ("cpu", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cuda", True, "cuda:0"),
    ],
)
def test_device_selection(mode, available, expected, monkeypatch):
    from voice_reader.device import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Test NVIDIA GPU")
    info = resolve_device(mode)
    assert info.resolved == expected
    assert info.requested == mode
    assert info.reason
    assert info.name == ("CPU" if expected == "cpu" else "Test NVIDIA GPU")


def test_explicit_cuda_does_not_fall_back(monkeypatch):
    from voice_reader.device import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_invalid_device_is_rejected():
    from voice_reader.device import resolve_device

    with pytest.raises(ValueError, match="device"):
        resolve_device("mps")


def test_device_detection_failure_is_not_hidden(monkeypatch):
    from voice_reader.device import resolve_device

    def fail():
        raise RuntimeError("driver probe failed")

    monkeypatch.setattr(torch.cuda, "is_available", fail)
    with pytest.raises(RuntimeError, match="driver probe failed"):
        resolve_device("auto")
