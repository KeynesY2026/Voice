import asyncio
import os
import time
from pathlib import Path

import gradio as gr
from fastapi.testclient import TestClient
from gradio.data_classes import FileData

from voice_reader import web
from voice_reader.device import DeviceInfo
from voice_reader.profiles import ProfileStore


class DeviceStatus:
    device = "cpu"
    device_info = DeviceInfo("cpu", "cpu", "CPU", "Explicit CPU")
    profile_store = ProfileStore()
    runtime_description = "Runtime: standard"

    def cancel(self, owner="local"):
        return True


def test_ui_logs_generation_errors(tmp_path, monkeypatch, caplog):
    class FailingSpeechEngine(DeviceStatus):
        def synthesize(self, text, output_path, emotion, *, profile_id=None, language="auto", device=None, owner="local", advanced=False):
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


def test_ui_callback_uses_shared_engine_and_manual_emotion(tmp_path, monkeypatch, capsys):
    calls = []

    class FakeSpeechEngine(DeviceStatus):
        def synthesize(self, text, output_path, emotion, *, profile_id=None, language="auto", device=None, owner="local", advanced=False):
            calls.append((text, Path(output_path), emotion))
            Path(output_path).write_bytes(b"wav")
            return Path(output_path)

    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path)
    app = web.build_ui(FakeSpeechEngine())
    callback = next(
        function.fn for function in app.fns.values()
        if getattr(function.fn, "__name__", "") == "generate"
    )

    clock = iter((10.0, 12.375))
    monkeypatch.setattr(web.time, "perf_counter", lambda: next(clock))
    audio_path, status = callback(
        "That is great.", "manual", "happy", 0.7, advanced_enabled=True,
    )

    assert Path(audio_path).read_bytes() == b"wav"
    assert "本轮推理总用时：2.38 秒" in capsys.readouterr().out
    assert "耗时 2.4 秒" in status
    assert "生成完成" in status
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


def test_web_launch_translates_framework_audio_for_browser_locales(monkeypatch):
    launches = []

    class FakeApp:
        def queue(self):
            return self

        def launch(self, **kwargs):
            launches.append(kwargs)

    monkeypatch.setattr(web, "SpeechEngine", lambda **kwargs: DeviceStatus())
    monkeypatch.setattr(web, "build_ui", lambda engine, **kwargs: FakeApp())
    web.launch([])
    translations = launches[0]["i18n"].translations_dict
    for locale in ("en", "zh-CN", "fr"):
        assert translations[locale]["upload_text.drop_audio"] == "将音频拖放到此处"
        assert translations[locale]["upload_text.click_to_upload"] == "点击上传"
        assert translations[locale]["audio.drop_to_upload"] == "将音频文件拖放到此处以上传"

def test_web_launch_accepts_runtime_selection(monkeypatch):
    settings = []

    class App:
        def queue(self):
            return self

        def launch(self, **kwargs):
            pass

    def create_engine(**kwargs):
        settings.append(kwargs)
        return DeviceStatus()

    monkeypatch.setattr(web, "SpeechEngine", create_engine)
    monkeypatch.setattr(web, "build_ui", lambda *args, **kwargs: App())
    web.launch(["--runtime", "standard"])
    assert settings[0]["runtime"] == "standard"


def callback(app, name):
    return next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == name)


def test_web_cache_uses_gradio_native_cleanup():
    app = web.build_ui(DeviceStatus())

    assert app.delete_cache == (60, 23 * 60 * 60 + 58 * 60)


def test_cleanup_expires_web_outputs_after_24_hours_and_preserves_other_files(tmp_path):
    now = 2_000_000_000
    expired = tmp_path / "web-expired.wav"
    boundary = tmp_path / "web-boundary.wav"
    cli_output = tmp_path / "speech.wav"
    for path in (expired, boundary, cli_output):
        path.write_bytes(b"audio")
    os.utime(expired, (now - 24 * 60 * 60 - 1,) * 2)
    os.utime(boundary, (now - 24 * 60 * 60,) * 2)
    os.utime(cli_output, (now - 30 * 24 * 60 * 60,) * 2)

    web._cleanup_web_outputs(tmp_path, now=now)

    assert not expired.exists()
    assert boundary.exists()
    assert cli_output.exists()


def test_cleanup_limits_web_outputs_to_100_oldest_first(tmp_path):
    now = 2_000_000_000
    outputs = []
    for index in range(101):
        path = tmp_path / f"web-{index:03}.wav"
        path.write_bytes(b"audio")
        modified = now - (100 - index)
        os.utime(path, (modified, modified))
        outputs.append(path)

    web._cleanup_web_outputs(tmp_path, now=now)

    assert len(list(tmp_path.glob("web-*.wav"))) == 100
    assert not outputs[0].exists()
    assert all(path.exists() for path in outputs[1:])


def test_generated_audio_cleanup_runs_while_web_is_active(tmp_path, monkeypatch):
    class FakeSpeechEngine(DeviceStatus):
        def synthesize(self, text, output_path, emotion, **kwargs):
            Path(output_path).write_bytes(b"generated wav")
            return Path(output_path)

    now = time.time()
    expired = tmp_path / "web-expired.wav"
    expired.write_bytes(b"old audio")
    old_time = now - 24 * 60 * 60 - 1
    os.utime(expired, (old_time, old_time))
    existing_outputs = []
    for index in range(100):
        path = tmp_path / f"web-existing-{index:03}.wav"
        path.write_bytes(b"existing audio")
        modified = now - (100 - index)
        os.utime(path, (modified, modified))
        existing_outputs.append(path)
    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path)
    app = web.build_ui(FakeSpeechEngine())

    output, status = callback(app, "generate")("Test.", "manual", "happy", 0.6)

    assert "生成完成" in status
    assert not expired.exists()
    assert not existing_outputs[0].exists()
    assert all(path.exists() for path in existing_outputs[1:])
    assert len(list(tmp_path.glob("web-*.wav"))) == 100
    assert Path(output).exists()


def test_generated_audio_keeps_the_audio_player_download_button(tmp_path, monkeypatch):
    class FakeSpeechEngine(DeviceStatus):
        def synthesize(self, text, output_path, emotion, **kwargs):
            Path(output_path).write_bytes(b"generated wav")
            return Path(output_path)

    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path)
    app = web.build_ui(FakeSpeechEngine())
    generate = next(fn for fn in app.fns.values() if fn.fn == callback(app, "generate"))
    audio = next(
        block for block in app.blocks.values()
        if isinstance(block, gr.Audio) and block.label == "生成的音频"
    )

    assert audio in generate.outputs
    assert audio.show_download_button is True
    audio_path, status = callback(app, "generate")(
        "Test.", "manual", "happy", 0.6,
    )

    assert "生成完成" in status
    assert Path(audio_path).read_bytes() == b"generated wav"

    result = asyncio.run(app.process_api(
        generate,
        inputs=["Test.", "manual", "happy", 0.6, "builtin-male", "auto", "cpu", False],
    ))
    audio_file = result["data"][0]
    assert audio_file["orig_name"].endswith(".wav")
    with TestClient(app.app) as client:
        response = client.get(audio_file["url"])
    assert response.status_code == 200
    assert response.content == b"generated wav"


def test_generation_and_cancel_buttons_toggle_exclusive_visibility():
    app = web.build_ui(DeviceStatus())
    generate_button = next(
        block for block in app.blocks.values()
        if isinstance(block, gr.Button) and block.value == "生成语音"
    )
    cancel_button = next(
        block for block in app.blocks.values()
        if isinstance(block, gr.Button) and block.value == "取消生成"
    )

    assert generate_button.visible is True
    assert cancel_button.visible is False
    start = callback(app, "start_generation")
    finish = callback(app, "finish_generation")
    starting = start()
    finished = finish()
    assert (starting[0]["visible"], starting[1]["visible"]) == (False, True)
    assert starting[2]["value"] is None
    assert (finished[0]["visible"], finished[1]["visible"]) == (True, False)

    dependencies = app.config["dependencies"]
    by_name = {
        app.fns[dependency["id"]].fn.__name__: dependency
        for dependency in dependencies if dependency["backend_fn"]
    }
    generation = by_name["generate"]
    assert generation["trigger_after"] == by_name["start_generation"]["id"]
    finish_triggers = {
        dependency["trigger_after"] for dependency in dependencies
        if dependency["backend_fn"]
        and app.fns[dependency["id"]].fn.__name__ == "finish_generation"
    }
    assert finish_triggers == {generation["id"], by_name["cancel_generation"]["id"]}
    cancel_dependency = next(
        dependency for dependency in dependencies
        if dependency["types"]["cancel"]
        and (cancel_button._id, "click") in dependency["targets"]
    )
    assert generation["id"] in cancel_dependency["cancels"]


def test_profile_creation_and_rename_use_separate_names_and_chinese_labels(tmp_path):
    app = web.build_ui(DeviceStatus(), profile_store=ProfileStore(tmp_path / "profiles"))
    create = next(fn for fn in app.fns.values() if fn.fn == callback(app, "create_profile"))
    rename = next(fn for fn in app.fns.values() if fn.fn == callback(app, "rename_profile"))
    generate = next(fn for fn in app.fns.values() if fn.fn == callback(app, "generate"))
    assert create.inputs[0] is not rename.inputs[1]
    assert create.inputs[0].label == "档案名称"
    assert rename.inputs[1].label == "新名称"
    assert ("自动", "auto") in generate.inputs[5].choices
    assert ("开心", "happy") in generate.inputs[2].choices
    assert [tab.label for tab in app.blocks.values() if isinstance(tab, gr.Tab)] == [
        "生成语音", "设置", "声音档案",
    ]


def test_deletion_reports_actual_generation_selection(tmp_path):
    import wave

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 72000)
    store = ProfileStore(tmp_path / "profiles")
    saved = store.create("Voice", source)
    app = web.build_ui(DeviceStatus(), profile_store=store)
    selected, status, _ = callback(app, "delete_profile")(saved.id, True, saved.id)
    assert selected["value"] == "builtin-male"
    assert "男声" in status


def test_rename_does_not_change_empty_generation_selection(tmp_path):
    import wave

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 72000)
    store = ProfileStore(tmp_path / "profiles")
    saved = store.create("Voice", source)
    app = web.build_ui(DeviceStatus(), profile_store=store)
    generation, _ = callback(app, "rename_profile")(saved.id, "Renamed", None)
    assert generation["value"] is None


def test_failed_creation_keeps_management_selection_and_builtin_is_protected(tmp_path):
    import wave

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 72000)
    store = ProfileStore(tmp_path / "profiles")
    saved = store.create("Voice", source)
    app = web.build_ui(DeviceStatus(), profile_store=store)
    update, status = callback(app, "create_profile")("Other", None)
    assert "value" not in update and "上传" in status
    management, name, *_ = callback(app, "refresh_after_create")("builtin-male", saved.id, status)
    assert management["value"] == saved.id and name == "Voice"
    generation, status, confirmed = callback(app, "delete_profile")("builtin-female", True, saved.id)
    assert "value" not in generation and "删除失败" in status and confirmed is False
    assert store.get(saved.id).name == "Voice"

def test_ui_has_defaults_language_device_cancel_and_browser_memory():
    app = web.build_ui(DeviceStatus())
    generate = next(fn for fn in app.fns.values() if fn.fn == callback(app, "generate"))
    assert generate.inputs[3].value == 0.4
    assert generate.inputs[5].value == "auto"
    assert isinstance(generate.inputs[6], gr.Dropdown)
    assert generate.inputs[7].value is True
    assert next(block for block in app.blocks.values()
                if isinstance(block, gr.Accordion) and block.label == "文本写法案例").open is False
    assert isinstance(next(block for block in app.blocks.values() if isinstance(block, gr.BrowserState)), gr.BrowserState)
    assert callback(app, "cancel_generation")() == "生成已取消，下次生成将重新加载模型。"
    restore = callback(app, "restore_inputs")
    values = restore(dict(text="Hello  there.\nAgain.", strength=0.3, language="en", device="cpu"))
    assert values[0] == "Hello  there.\nAgain."
    assert values[3] == 0.3
    assert values[5] == "en"
    assert values[6] == "cpu"
    assert values[7] is True
    panels = [block for block in app.blocks.values() if isinstance(block, gr.Tab)]
    assert any(panel.label == "设置" for panel in panels)
    assert any(block.value and "<行|HANG2>" in str(block.value)
               for block in app.blocks.values() if isinstance(block, gr.Markdown))
    assert callback(app, "set_advanced")(True)["visible"] is True
    assert callback(app, "set_advanced")(False)["visible"] is False


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
    assert "已创建" in status
    selection, status = callback(app, "rename_profile")(profile_id, "New name")
    assert selection["value"] is None
    assert store.get(profile_id).name == "New name"
    selection, status, confirmed = callback(app, "delete_profile")(profile_id, False)
    assert "确认" in status
    assert store.get(profile_id).id == profile_id
    selection, status, confirmed = callback(app, "delete_profile")(profile_id, True)
    assert selection["value"] is None
    assert confirmed is False
    assert store.list_profiles() == []


def test_profile_management_has_its_own_loaded_selector_and_action_inputs(tmp_path):
    import wave

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 72000)
    store = ProfileStore(tmp_path / "profiles")
    saved = store.create("My voice", source)
    app = web.build_ui(DeviceStatus(), profile_store=store)
    rename = next(fn for fn in app.fns.values() if fn.fn == callback(app, "rename_profile"))
    delete = next(fn for fn in app.fns.values() if fn.fn == callback(app, "delete_profile"))
    generate = next(fn for fn in app.fns.values() if fn.fn == callback(app, "generate"))
    selector = rename.inputs[0]
    assert selector is not generate.inputs[4]
    assert selector is delete.inputs[0]
    assert (f"My voice ({saved.id})", saved.id) in selector.choices
    assert all(value not in {"builtin-male", "builtin-female"} for _, value in selector.choices)
    selection, name, confirmed, empty, rename_state, delete_state = callback(app, "refresh_management")(saved.id)
    assert selection["value"] == saved.id
    assert name == "My voice" and confirmed["value"] is False
    assert confirmed["interactive"] and rename_state["interactive"] and delete_state["interactive"]
    assert empty == ""
    generation_selection, _ = callback(app, "rename_profile")(saved.id, "Renamed", "builtin-female")
    assert generation_selection["value"] == "builtin-female"
    selection, name, *_ = callback(app, "refresh_management")(saved.id)
    assert name == "Renamed"
    generation_selection, _, _ = callback(app, "delete_profile")(saved.id, True, "builtin-female")
    assert generation_selection["value"] == "builtin-female"
    selection, name, confirmed, empty, rename_state, delete_state = callback(app, "refresh_management")(saved.id)
    assert selection["choices"] == [] and selection["value"] is None
    assert name == "" and confirmed["value"] is False
    assert empty == "暂无已保存档案"
    assert not confirmed["interactive"] and not rename_state["interactive"] and not delete_state["interactive"]


def test_web_uses_requested_profile_and_unique_output(tmp_path, monkeypatch):
    calls = []

    class Engine(DeviceStatus):
        def synthesize(self, text, output_path, emotion, *, profile_id=None, language="auto", device=None, owner="local", advanced=False):
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
    assert "请选择" in status
    assert len(calls) == 2


def test_web_advanced_switch_is_forwarded_and_fast_ignores_hidden_emotion(tmp_path, monkeypatch):
    calls = []

    class Engine(DeviceStatus):
        def synthesize(self, text, output_path, emotion, **kwargs):
            calls.append((emotion, kwargs))
            return output_path

    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path)
    app = web.build_ui(Engine())
    generate = callback(app, "generate")
    generate("Hello.\nNext.", "manual", "happy", 0.8, advanced_enabled=False)
    generate("Hello.\nNext.", "manual", "happy", 0.8, advanced_enabled=True)
    assert calls[0][0] is None
    assert calls[0][1]["advanced"] is False
    assert calls[1][0].strength == 0.8
    assert calls[1][1]["advanced"] is True


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
    selection, _status = callback(app, "refresh_profiles")("default")
    assert selection["value"] == "builtin-male"
    assert all(value != "default" for _, value in selection["choices"])
    assert (f"Default reference recording ({profile.id})", profile.id) in selection["choices"]
    restored = callback(app, "restore_inputs")({"profile": "default", "text": "Keep this."})
    assert restored[4]["value"] == "builtin-male"
    assert restored[0] == "Keep this."


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
    assert "裁剪" in status
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
    assert "已创建" in status
    clip.unlink()

    profile = ProfileStore(store.root).get(selection["value"])
    assert profile.name == "Selected clip"
    assert profile.duration_seconds == 5
    with wave.open(str(profile.reference_path), "rb") as wav:
        assert wav.readframes(wav.getnframes()) == frames
