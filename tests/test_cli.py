from pathlib import Path
from typing import ClassVar

import pytest

from voice_reader import cli
from voice_reader.engine import EmotionSettings


class FakeSpeechEngine:
    instances: ClassVar[list] = []
    failure: ClassVar[Exception | None] = None

    def __init__(self, *, device="auto", runtime="auto"):
        from voice_reader.device import DeviceInfo
        self.device = device
        self.runtime = runtime
        self.runtime_description = f"Runtime: {runtime}"
        self.device_info = DeviceInfo(device, "cpu", "CPU", "Test CPU")
        self.calls = []
        self.instances.append(self)

    def synthesize(self, text, output_path, emotion, *, profile_id=None):
        self.profile_id = profile_id
        self.calls.append((text, Path(output_path), emotion))
        if self.failure:
            raise self.failure
        return Path(output_path)


@pytest.fixture(autouse=True)
def reset_engine_instances(monkeypatch):
    FakeSpeechEngine.instances.clear()
    FakeSpeechEngine.failure = None
    monkeypatch.setattr(cli, "SpeechEngine", FakeSpeechEngine)


def test_cli_accepts_direct_text_and_outputs_path(tmp_path, capsys):
    output = tmp_path / "speech.wav"

    result = cli.main(["speak", "--text", "Hello.", "--output", str(output)])

    assert result == 0
    assert FakeSpeechEngine.instances[0].calls[0] == ("Hello.", output, EmotionSettings())
    assert str(output) in capsys.readouterr().out


def test_cli_reads_utf8_text_file(tmp_path):
    text_file = tmp_path / "input.txt"
    text_file.write_text("Good morning.", encoding="utf-8")

    assert cli.main(["speak", "--text-file", str(text_file)]) == 0
    assert FakeSpeechEngine.instances[0].calls[0][0] == "Good morning."


def test_cli_reads_stdin(monkeypatch):
    monkeypatch.setattr(cli.sys, "stdin", type("Input", (), {"read": lambda self: "A test."})())

    assert cli.main(["speak", "--stdin"]) == 0
    assert FakeSpeechEngine.instances[0].calls[0][0] == "A test."


@pytest.mark.parametrize(
    "arguments",
    [
        ["speak"],
        ["speak", "--text", "One", "--stdin"],
    ],
)
def test_cli_requires_exactly_one_text_source(arguments):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(arguments)

    assert exc_info.value.code == 2
    assert not FakeSpeechEngine.instances


def test_cli_maps_manual_emotion_to_model_vector():
    assert cli.main(["speak", "--text", "Surprise!", "--emotion", "surprised", "--strength", "0.4"]) == 0

    settings = FakeSpeechEngine.instances[0].calls[0][2]
    assert settings == EmotionSettings(
        mode="manual",
        vector=(0, 0, 0, 0, 0, 0, 1, 0),
        strength=0.4,
    )


def test_cli_reports_generation_failure_with_nonzero_exit(capsys):
    FakeSpeechEngine.failure = RuntimeError("model failed")

    result = cli.main(["speak", "--text", "Hello."])

    assert result == 1
    assert "model failed" in capsys.readouterr().err


def test_cli_passes_profile_and_device(tmp_path):
    assert cli.main(["speak", "--text", "Hello.", "--profile", "chosen-id", "--device", "cpu"]) == 0
    speech = FakeSpeechEngine.instances[0]
    assert speech.profile_id == "chosen-id"
    assert speech.device == "cpu"


def test_cli_can_select_standard_runtime():
    assert cli.main(["speak", "--text", "Hello.", "--runtime", "standard"]) == 0
    assert FakeSpeechEngine.instances[0].runtime == "standard"


def test_cli_profile_lifecycle_without_model(tmp_path, monkeypatch, capsys):
    import wave

    from voice_reader.profiles import ProfileStore

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 72000)
    store = ProfileStore(tmp_path / "profiles")
    monkeypatch.setattr(cli, "ProfileStore", lambda: store, raising=False)
    assert cli.main(["profiles", "create", "--name", "Voice", "--audio", str(source)]) == 0
    profile = store.list_profiles()[0]
    assert profile.id in capsys.readouterr().out
    assert cli.main(["profiles", "list"]) == 0
    assert "Voice" in capsys.readouterr().out
    assert cli.main(["profiles", "rename", profile.id, "--name", "Renamed"]) == 0
    assert store.get(profile.id).name == "Renamed"
    assert cli.main(["profiles", "delete", profile.id]) == 1
    assert store.get(profile.id).name == "Renamed"
    assert cli.main(["profiles", "delete", profile.id, "--yes"]) == 0
    assert store.list_profiles() == []
    assert not FakeSpeechEngine.instances


def test_cli_model_stdout_does_not_pollute_output_path(tmp_path, capsys, monkeypatch):
    def noisy_synthesis(self, text, output_path, emotion, *, profile_id=None):
        print("upstream model diagnostic")
        return Path(output_path)

    monkeypatch.setattr(FakeSpeechEngine, "synthesize", noisy_synthesis)
    output = tmp_path / "output.wav"
    assert cli.main(["speak", "--text", "Hi.", "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == str(output)
    assert "upstream model diagnostic" in captured.err
