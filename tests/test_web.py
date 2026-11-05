from pathlib import Path

import gradio as gr

from voice_reader import web


def test_ui_logs_generation_errors(tmp_path, monkeypatch, caplog):
    class FailingSpeechEngine:
        def synthesize(self, text, output_path, emotion):
            raise OSError(22, "Invalid argument")

    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path)
    app = web.build_ui(FailingSpeechEngine())
    callback = next(
        function.fn for function in app.fns.values()
        if getattr(function.fn, "__name__", "") == "generate"
    )

    with caplog.at_level("ERROR", logger=web.__name__):
        audio_path, status = callback("Test.", "manual", "happy", 0.6)

    assert audio_path is None
    assert "Invalid argument" in status
    assert "Speech generation failed" in caplog.text


def test_ui_callback_uses_shared_engine_and_manual_emotion(tmp_path, monkeypatch):
    calls = []

    class FakeSpeechEngine:
        def synthesize(self, text, output_path, emotion):
            calls.append((text, Path(output_path), emotion))
            Path(output_path).write_bytes(b"wav")
            return Path(output_path)

    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path)
    app = web.build_ui(FakeSpeechEngine())
    callback = next(
        function.fn for function in app.fns.values()
        if getattr(function.fn, "__name__", "") == "generate"
    )

    audio_path, status = callback("That is great.", "manual", "happy", 0.7)

    assert Path(audio_path).read_bytes() == b"wav"
    assert "ready" in status.lower()
    assert calls[0][0] == "That is great."
    assert calls[0][2].mode == "manual"
    assert calls[0][2].vector == (1, 0, 0, 0, 0, 0, 0, 0)
    assert calls[0][2].strength == 0.7
    assert isinstance(app, gr.Blocks)


def test_web_launch_binds_only_to_loopback(monkeypatch):
    launches = []

    class FakeApp:
        def queue(self):
            return self

        def launch(self, **kwargs):
            launches.append(kwargs)

    monkeypatch.setattr(web, "SpeechEngine", lambda: object())
    monkeypatch.setattr(web, "build_ui", lambda engine: FakeApp())

    web.launch()

    assert launches == [{"server_name": "127.0.0.1", "server_port": 7860, "share": False}]
