from pathlib import Path
from typing import ClassVar

import pytest

from voice_reader import cli
from voice_reader.engine import EmotionSettings


class FakeSpeechEngine:
    instances: ClassVar[list] = []
    failure: ClassVar[Exception | None] = None

    def __init__(self):
        self.calls = []
        self.instances.append(self)

    def synthesize(self, text, output_path, emotion):
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
