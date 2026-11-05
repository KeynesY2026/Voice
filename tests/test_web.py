from pathlib import Path

import gradio as gr
from gradio.data_classes import FileData

from voice_reader import web
from voice_reader.device import DeviceInfo
from voice_reader.profiles import ProfileStore


class DeviceStatus:
    device_info = DeviceInfo("cpu", "cpu", "CPU", "Explicit CPU")
    profile_store = ProfileStore()


def test_ui_logs_generation_errors(tmp_path, monkeypatch, caplog):
    class FailingSpeechEngine(DeviceStatus):
        def synthesize(self, text, output_path, emotion, *, profile_id=None):
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

    class FakeSpeechEngine(DeviceStatus):
        def synthesize(self, text, output_path, emotion, *, profile_id=None):
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

    monkeypatch.setattr(web, "SpeechEngine", lambda **kwargs: DeviceStatus())
    monkeypatch.setattr(web, "build_ui", lambda engine, **kwargs: FakeApp())

    web.launch([])

    assert launches[0]["server_name"] == "127.0.0.1"
    assert launches[0]["server_port"] == 7860
    assert launches[0]["share"] is False
    assert launches[0]["max_file_size"] == 100 * 1024 * 1024


def callback(app, name):
    return next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == name)


def test_web_profile_lifecycle_and_delete_confirmation(tmp_path):
    import wave

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 72000)
    store = ProfileStore(tmp_path / "profiles")
    app = web.build_ui(DeviceStatus(), profile_store=store)
    selection, status = callback(app, "create_profile")("Voice", str(source))
    profile_id = selection["value"]
    assert store.get(profile_id).name == "Voice"
    assert "created" in status.lower()
    selection, status = callback(app, "rename_profile")(profile_id, "New name")
    assert selection["value"] == profile_id
    assert store.get(profile_id).name == "New name"
    selection, status, confirmed = callback(app, "delete_profile")(profile_id, False)
    assert "confirm" in status.lower()
    assert store.get(profile_id).id == profile_id
    selection, status, confirmed = callback(app, "delete_profile")(profile_id, True)
    assert selection["value"] is None
    assert confirmed is False
    assert store.list_profiles() == []


def test_web_uses_requested_profile_and_unique_output(tmp_path, monkeypatch):
    calls = []

    class Engine(DeviceStatus):
        def synthesize(self, text, output_path, emotion, *, profile_id=None):
            calls.append((profile_id, output_path))
            return output_path

    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path)
    app = web.build_ui(Engine(), profile_store=ProfileStore(tmp_path / "profiles"))
    generate = callback(app, "generate")
    generate("Text.", "auto", "calm", 0.6, "first-id")
    generate("Text.", "auto", "calm", 0.6, "second-id")
    assert [item[0] for item in calls] == ["first-id", "second-id"]
    assert calls[0][1] != calls[1][1]
    output, status = generate("Text.", "auto", "calm", 0.6, None)
    assert output is None
    assert "select" in status.lower()
    assert len(calls) == 2


def test_web_launch_accepts_device(monkeypatch):
    devices = []

    class App:
        def queue(self):
            return self

        def launch(self, **kwargs):
            pass

    def engine(**kwargs):
        devices.append(kwargs["device"])
        return DeviceStatus()

    monkeypatch.setattr(web, "SpeechEngine", engine)
    monkeypatch.setattr(web, "build_ui", lambda *args, **kwargs: App())
    web.launch(["--device", "cuda"])
    assert devices == ["cuda"]


def test_web_defaults_to_engines_profile_store(tmp_path):
    import wave

    from voice_reader.engine import SpeechEngine

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 72000)
    store = ProfileStore(tmp_path / "profiles")
    profile = store.create("Shared voice", source)
    app = web.build_ui(SpeechEngine(device="cpu", profile_store=store))
    selection, _status = callback(app, "refresh_profiles")(profile.id)
    assert selection["value"] == profile.id
    assert (f"Shared voice ({profile.id})", profile.id) in selection["choices"]


def test_profile_name_matching_default_remains_distinguishable(tmp_path):
    import wave

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 72000)
    store = ProfileStore(tmp_path / "profiles")
    profile = store.create("Default reference recording", source)
    app = web.build_ui(DeviceStatus(), profile_store=store)
    selection, _status = callback(app, "refresh_profiles")(web.DEFAULT_PROFILE)
    assert selection["value"] == web.DEFAULT_PROFILE
    assert ("Default reference recording", web.DEFAULT_PROFILE) in selection["choices"]
    assert (f"Default reference recording ({profile.id})", profile.id) in selection["choices"]


def test_profile_input_allows_long_audio_to_be_edited_before_creation(tmp_path):
    import wave

    source = tmp_path / "long.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 24000 * 45)
    store = ProfileStore(tmp_path / "profiles")
    app = web.build_ui(DeviceStatus(), profile_store=store)
    create = next(fn for fn in app.fns.values() if fn.fn == callback(app, "create_profile"))
    audio_input = create.inputs[1]

    assert isinstance(audio_input, gr.Audio)
    assert audio_input.interactive is True
    assert audio_input.editable is True
    assert audio_input.sources == ["upload"]
    assert audio_input.preprocess(FileData(path=str(source))) == str(source)
    selection, status = create.fn("Not cropped", str(source))
    assert "value" not in selection
    assert "crop" in status.lower()
    assert store.list_profiles() == []
    with wave.open(str(source), "rb") as wav:
        assert wav.getnframes() == 24000 * 45


def test_web_saves_cropped_audio_without_depending_on_upload(tmp_path):
    import wave

    clip = tmp_path / "cropped.wav"
    frames = b"\x23\x01" * 24000 * 5
    with wave.open(str(clip), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(frames)
    store = ProfileStore(tmp_path / "profiles")
    app = web.build_ui(DeviceStatus(), profile_store=store)
    create = next(fn for fn in app.fns.values() if fn.fn == callback(app, "create_profile"))
    audio_input = create.inputs[1]
    assert isinstance(audio_input, gr.Audio)
    selection, status = create.fn("Selected clip", audio_input.preprocess(FileData(path=str(clip))))
    assert "created" in status.lower()
    clip.unlink()

    profile = ProfileStore(store.root).get(selection["value"])
    assert profile.name == "Selected clip"
    assert profile.duration_seconds == 5
    with wave.open(str(profile.reference_path), "rb") as wav:
        assert wav.readframes(wav.getnframes()) == frames
