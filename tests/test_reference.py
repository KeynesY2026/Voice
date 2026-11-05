from pathlib import Path
from types import SimpleNamespace

import pytest

from voice_reader import reference


def test_missing_reference_fails_without_invoking_ffmpeg(tmp_path, monkeypatch):
    invoked = False

    def run(*args, **kwargs):
        nonlocal invoked
        invoked = True

    monkeypatch.setattr(reference.subprocess, "run", run)

    with pytest.raises(FileNotFoundError, match="Reference audio not found"):
        reference.prepare_reference(tmp_path / "missing.m4a", tmp_path / "cache")

    assert not invoked


def test_reference_is_converted_once_and_reused(tmp_path, monkeypatch):
    source = tmp_path / "voice.m4a"
    source.write_bytes(b"sample")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"wav")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(reference.imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg.exe")
    monkeypatch.setattr(reference.subprocess, "run", run)

    first = reference.prepare_reference(source, tmp_path / "cache")
    second = reference.prepare_reference(source, tmp_path / "cache")

    assert first == second
    assert first.read_bytes() == b"wav"
    assert len(calls) == 1
    assert calls[0][calls[0].index("-ac") + 1] == "1"
    assert calls[0][calls[0].index("-ar") + 1] == "24000"


def test_changed_reference_gets_a_new_cached_wav(tmp_path, monkeypatch):
    source = tmp_path / "voice.m4a"
    source.write_bytes(b"first")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"wav")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(reference.imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg.exe")
    monkeypatch.setattr(reference.subprocess, "run", run)

    first = reference.prepare_reference(source, tmp_path / "cache")
    source.write_bytes(b"second")
    second = reference.prepare_reference(source, tmp_path / "cache")

    assert first != second
    assert len(calls) == 2


def test_failed_conversion_removes_partial_output_and_reports_stderr(tmp_path, monkeypatch):
    source = tmp_path / "voice.m4a"
    source.write_bytes(b"sample")

    def run(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stderr="invalid input")

    monkeypatch.setattr(reference.imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg.exe")
    monkeypatch.setattr(reference.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="invalid input"):
        reference.prepare_reference(source, tmp_path / "cache")

    assert not list((tmp_path / "cache").glob("*.wav"))
