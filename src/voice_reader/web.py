import argparse
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from voice_reader.config import OUTPUT_DIR, PRIVATE_DATA_DIR, PROJECT_ROOT
from voice_reader.console import configure_utf8_output
from voice_reader.engine import EMOTION_CATEGORIES, SpeechEngine, make_emotion_settings
from voice_reader.device import resolve_device
from voice_reader.builtin_profiles import BUILTIN_PROFILES
from voice_reader.inference_service import GenerationCancelled, InferenceService
from voice_reader.language import LANGUAGE_CHOICES
from voice_reader.preferences import DEFAULT_PREFERENCES, load_browser_secret, restore_preferences
from voice_reader.profiles import MAX_AUDIO_BYTES, ProfileStore

os.environ.setdefault("GRADIO_TEMP_DIR", str(PRIVATE_DATA_DIR / "web-cache"))

import gradio as gr

logger = logging.getLogger(__name__)
DEFAULT_PROFILE = "builtin-male"

LANGUAGE_LABELS = {
    "auto": "自动", "en": "英语", "zh": "中文", "ja": "日语", "es": "西班牙语", "ar": "阿拉伯语",
}
EMOTION_LABELS = {
    "happy": "开心", "angry": "愤怒", "sad": "悲伤", "afraid": "害怕",
    "disgusted": "厌恶", "melancholic": "忧郁", "surprised": "惊讶", "calm": "平静",
}
DEVICE_LABELS = {"auto": "自动", "cpu": "CPU", "cuda": "GPU"}
BUILTIN_LABELS = {"builtin-male": "内置男声", "builtin-female": "内置女声"}

_FRAMEWORK_ZH = {
    "common.built_with": "构建于", "common.built_with_gradio": "使用 Gradio 构建",
    "common.clear": "清除", "common.download": "下载", "common.edit": "编辑",
    "common.empty": "空", "common.error": "错误", "common.hosted_on": "托管于",
    "common.loading": "加载中", "common.logo": "标志", "common.or": "或",
    "common.remove": "移除", "common.share": "分享", "common.submit": "提交",
    "common.undo": "撤销", "common.settings": "设置", "common.no_devices": "未找到设备",
    "common.language": "语言", "common.display_theme": "显示主题", "common.pwa": "渐进式 Web 应用",
    "common.screen_studio": "屏幕录制", "common.stop_recording": "停止录制",
    "audio.audio": "音频", "audio.drop_to_upload": "将音频文件拖放到此处以上传",
    "audio.play": "播放", "audio.pause": "暂停", "audio.stop": "停止", "audio.resume": "继续",
    "audio.record": "录制", "audio.waiting": "等待中",
    "audio.allow_recording_access": "请允许访问麦克风以进行录音。",
    "audio.record_from_microphone": "从麦克风录制", "audio.stop_recording": "停止录制",
    "audio.no_microphone": "找不到麦克风", "audio.no_device_support": "无法访问音频设备，请检查浏览器权限。",
    "upload_text.drop_audio": "将音频拖放到此处", "upload_text.click_to_upload": "点击上传",
    "blocks.waiting_for_inputs": "正在等待文件上传完成，请稍后重试。",
    "blocks.lost_connection": "连接已断开，正在重新加入队列。",
    "blocks.connection_can_break": "移动设备切换页面或休眠时，连接可能中断。",
    "blocks.long_requests_queue": "当前有较多请求正在排队。",
    "radio.radio": "单选框", "slider.slider": "滑块",
    "errors.use_via_api": "通过 API 使用", "errors.use_via_api_or_mcp": "通过 API 或 MCP 使用",
}
_FRAMEWORK_LOCALES = (
    "ar", "ca", "ckb", "de", "en", "es", "eu", "fa", "fi", "fr", "he", "hi", "id",
    "ja", "ko", "lt", "nb", "nl", "pl", "pt-BR", "pt", "ro", "ru", "sv", "ta",
    "th", "tr", "uk", "ur", "uz", "zh-CN", "zh-TW",
)
FRAMEWORK_I18N = gr.I18n(**{locale: _FRAMEWORK_ZH for locale in _FRAMEWORK_LOCALES})


def _error_detail(error: Exception) -> str:
    message = str(error)
    known = {
        "Profile name must not be empty.": "档案名称不能为空。",
        "Upload a reference recording first.": "请先上传参考录音。",
        "Text must not be empty.": "请输入要朗读的文本。",
        "Profile audio must be WAV, MP3, M4A, or FLAC.": "录音须为 WAV、MP3、M4A 或 FLAC 格式。",
        "Profile audio must be non-empty and at most 100 MiB.": "录音不能为空，且不得超过 100 MiB。",
    }
    if message in known:
        return known[message]
    if message.startswith("Profile audio must be between 3 and 30 seconds."):
        return "录音须为 3–30 秒，请裁剪后再试；不会自动裁剪。"
    if message.startswith("A profile named ") and message.endswith(" already exists."):
        return f"档案名称已存在：{message}"
    return f"详情：{message}"


def _device_description(info) -> str:
    reasons = {
        "CPU explicitly requested.": "已指定 CPU。",
        "Explicit CPU": "已指定 CPU。",
        "CUDA is available.": "CUDA 可用。",
        "CUDA is unavailable; auto selected CPU.": "CUDA 不可用，已自动选择 CPU。",
    }
    return f"设备：{info.resolved}（{info.name}）。{reasons.get(info.reason, info.reason)}"


def _runtime_description(description: str) -> str:
    return (description.replace("Runtime:", "运行策略：")
            .replace("selected when the model loads", "模型加载时确定")
            .replace("standard", "标准").replace("low-vram", "低显存")
            .replace("auto", "自动").replace("token segments", "token 分段")
            .replace("CPU reference/emotion", "参考音频和情绪使用 CPU")
            .replace("resident CUDA audio models", "音频模型常驻 CUDA")
            .replace("matmul", "矩阵乘法"))


def _speech_tips() -> str:
    source = PROJECT_ROOT / "docs" / "speech-tips.md"
    packaged = Path(__file__).parent / "assets" / "speech-tips.md"
    return (source if source.is_file() else packaged).read_text(encoding="utf-8")


def build_ui(engine: SpeechEngine | InferenceService, profile_store: ProfileStore | None = None) -> gr.Blocks:
    if isinstance(engine, SpeechEngine):
        engine = InferenceService(engine)
    store = profile_store if profile_store is not None else engine.profile_store
    devices = [("自动", "auto"), ("CPU", "cpu")]
    if resolve_device("auto").resolved.startswith("cuda"):
        devices.append(("GPU", "cuda"))

    def profile_choices():
        return [(BUILTIN_LABELS[key], key) for key in BUILTIN_PROFILES] + [
            (f"{profile.name} ({profile.id})", profile.id) for profile in store.list_profiles()
        ]

    with gr.Blocks(
        title="本地语音生成器", analytics_enabled=False, delete_cache=(86400, 86400),
        css=".speech-examples { max-height: 460px; overflow-y: auto; padding: 0 12px; }",
    ) as app:
        gr.Markdown("# 本地语音生成器")
        saved_inputs = gr.BrowserState(
            dict(DEFAULT_PREFERENCES, device=engine.device), storage_key="voice-reader-inputs-v1",
            secret=load_browser_secret(),
        )
        initialized = gr.State(False)
        with gr.Tab("生成语音"):
            with gr.Row():
                profile = gr.Dropdown(
                    choices=profile_choices(), value="builtin-male", label="声音", interactive=True,
                )
                language = gr.Dropdown(
                    [(LANGUAGE_LABELS[key], key) for _, key in LANGUAGE_CHOICES],
                    value="auto", label="阅读语言",
                )
            text = gr.Textbox(label="朗读文本", lines=8, placeholder="输入要朗读的文本")
            settings_summary = gr.Markdown("**高级模式** · 语言：自动 · 设备：自动")
            advanced = gr.Checkbox(
                value=True, label="开启高级功能（可能更慢）",
            )
            with gr.Column(visible=True) as advanced_panel:
                gr.Markdown(
                    "换行 / 多空格会分段生成，可能更慢；自动情绪会额外分析文本。"
                )
                with gr.Row():
                    mode = gr.Radio([("自动", "auto"), ("手动", "manual")], value="auto", label="情绪模式")
                    category = gr.Dropdown([(EMOTION_LABELS[key], key) for key in EMOTION_CATEGORIES], value="calm",
                                           label="手动情绪", interactive=False)
                    strength = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="情绪强度")
                with gr.Accordion("文本写法案例", open=False):
                    gr.Markdown(_speech_tips(), elem_classes="speech-examples")
            with gr.Row():
                generate_button = gr.Button("生成语音", variant="primary")
                cancel_button = gr.Button("取消生成", variant="stop")
            audio = gr.Audio(
                label="生成的音频", type="filepath", interactive=False, show_download_button=True,
            )
            status = gr.Markdown()
        with gr.Tab("设置"):
            device = gr.Dropdown(devices, value=engine.device, label="推理设备（下次生成生效）")
            gr.Markdown("## 运行详情")
            gr.Textbox(value=_device_description(engine.device_info), label="初始运行设备",
                       interactive=False)
            gr.Markdown(_runtime_description(engine.runtime_description))
            gr.Markdown(
                "文本、声音和配置保存在当前浏览器的同一站点，刷新后恢复；清除站点数据会清除记录。"
                "高级功能首次默认开启，之后恢复你上次选择的开关。取消会停止当前推理进程，"
                "下次生成重新加载模型，不从取消位置续读。"
            )
        with gr.Tab("声音档案") as profiles_tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 已保存档案")
                    management_profile = gr.Dropdown(
                        choices=[(f"{p.name} ({p.id})", p.id) for p in store.list_profiles()],
                        value=None, label="已保存档案", interactive=True,
                    )
                    empty_profiles = gr.Markdown("暂无已保存档案" if not store.list_profiles() else "")
                    gr.Markdown("内置声音不可修改或删除。")
                    refresh_button = gr.Button("刷新档案")
                    rename_name = gr.Textbox(label="新名称")
                    rename_button = gr.Button("改名", interactive=False)
                    confirm = gr.Checkbox(label="确认永久删除", value=False, interactive=False)
                    delete_button = gr.Button("删除档案", variant="stop", interactive=False)
                with gr.Column():
                    gr.Markdown("## 创建档案")
                    name = gr.Textbox(label="档案名称")
                    upload = gr.Audio(
                        label="参考录音（上传、裁剪并试听）",
                        sources=["upload"], type="filepath", format=None, interactive=True, editable=True,
                    )
                    gr.Markdown("仅使用本人或已获授权的录音。WAV / MP3 / M4A / FLAC，最大 100 MiB；"
                                "保存片段 3–30 秒，建议 3–15 秒，选择清晰的单人声音。")
                    create_button = gr.Button("创建档案", variant="primary")
            profile_status = gr.Markdown()

        def set_category_interactive(selected_mode: str):
            return gr.update(interactive=selected_mode == "manual")

        def set_advanced(enabled: bool):
            return gr.update(visible=enabled)

        def summarize_settings(selected_language, selected_device, enabled):
            language_label = LANGUAGE_LABELS[selected_language]
            return (
                f"**{'高级模式' if enabled else '快速模式'}**"
                f" · 语言：{language_label} · 设备：{DEVICE_LABELS[selected_device]}"
            )

        def generate(
            text_value: str, selected_mode: str, selected_category: str, selected_strength: float,
            selected_profile: str | None = DEFAULT_PROFILE,
            selected_language: str = "auto", selected_device: str | None = None,
            advanced_enabled: bool = False,
            request: gr.Request = None,
        ):
            started = time.perf_counter()
            try:
                if selected_profile is None or selected_profile == "default":
                    raise ValueError("请选择内置声音或已保存的声音档案。")
                emotion = (make_emotion_settings(selected_mode, selected_category, selected_strength)
                           if advanced_enabled else None)
                output = engine.synthesize(
                    text_value, OUTPUT_DIR / f"web-{uuid4()}.wav", emotion,
                    profile_id=selected_profile,
                    language=selected_language, device=selected_device or engine.device,
                    owner=request.session_hash if request is not None else "local",
                    advanced=advanced_enabled,
                )
            except GenerationCancelled:
                logger.info("Speech generation cancelled.")
                return None, "生成已取消，下次生成将重新加载模型。"
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Speech generation failed.")
                return None, f"**生成失败：**{_error_detail(error)}"
            elapsed = time.perf_counter() - started
            print(f"本轮推理总用时：{elapsed:.2f} 秒", flush=True)
            return str(output), (
                f"生成完成，耗时 {elapsed:.1f} 秒。"
                f"{'高级' if advanced_enabled else '快速'}模式。"
                f"{_device_description(engine.device_info)} {_runtime_description(engine.runtime_description)}"
            )

        def refresh_profiles(selected_profile):
            try:
                choices = profile_choices()
                if selected_profile == "default":
                    selected_profile = DEFAULT_PROFILE
                selected = selected_profile if any(value == selected_profile for _, value in choices) else None
                return gr.update(choices=choices, value=selected), "档案已刷新。"
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Profile refresh failed.")
                return gr.update(), f"**刷新失败：**{_error_detail(error)}"

        def refresh_management(selected_profile):
            try:
                saved = store.list_profiles()
                selected = next((p for p in saved if p.id == selected_profile), None)
                return (
                    gr.update(choices=[(f"{p.name} ({p.id})", p.id) for p in saved],
                              value=selected.id if selected else None),
                    selected.name if selected else "",
                    gr.update(value=False, interactive=bool(selected)),
                    "" if saved else "暂无已保存档案",
                    gr.update(interactive=bool(selected)), gr.update(interactive=bool(selected)),
                )
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Profile management refresh failed.")
                raise gr.Error(f"档案刷新失败：{_error_detail(error)}") from error

        def create_profile(profile_name, source):
            try:
                if not source:
                    raise ValueError("Upload a reference recording first.")
                created = store.create(profile_name, Path(source))
                return gr.update(choices=profile_choices(), value=created.id), f"档案已创建：{created.name}"
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Profile creation failed.")
                return gr.update(), f"**创建失败：**{_error_detail(error)}"

        def rename_profile(profile_id, profile_name, selected_profile=None):
            try:
                if not profile_id or profile_id == DEFAULT_PROFILE or profile_id in BUILTIN_PROFILES:
                    raise ValueError("请选择要改名的已保存档案。")
                renamed = store.rename(profile_id, profile_name)
                choices = profile_choices()
                selected = selected_profile if any(value == selected_profile for _, value in choices) else None
                return gr.update(choices=choices, value=selected), f"档案已改名：{renamed.name}"
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Profile rename failed.")
                return gr.update(), f"**改名失败：**{_error_detail(error)}"

        def delete_profile(profile_id, confirmed, selected_profile=None):
            try:
                if not confirmed:
                    raise ValueError("请先确认永久删除。")
                if not profile_id or profile_id == DEFAULT_PROFILE or profile_id in BUILTIN_PROFILES:
                    raise ValueError("请选择要删除的已保存档案。")
                store.delete(profile_id)
                choices = profile_choices()
                selected = selected_profile if any(value == selected_profile for _, value in choices) else (
                    DEFAULT_PROFILE if selected_profile is not None else None
                )
                message = "档案已删除。"
                if selected_profile == profile_id and selected == DEFAULT_PROFILE:
                    message += "生成声音已切换为内置男声。"
                return gr.update(choices=choices, value=selected), message, False
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Profile deletion failed.")
                return gr.update(), f"**删除失败：**{_error_detail(error)}", False

        def refresh_after_create(selected_profile, selected_management, result):
            target = selected_profile if result.startswith("档案已创建：") else selected_management
            return refresh_management(target)

        def cancel_generation(request: gr.Request = None):
            cancelled = engine.cancel(request.session_hash if request is not None else "local")
            return ("生成已取消，下次生成将重新加载模型。" if cancelled
                    else "当前浏览器没有正在生成的任务，排队任务已取消。")

        def restore_inputs(saved):
            choices = profile_choices()
            values, warning = restore_preferences(
                saved, {key for _, key in choices}, {key for _, key in devices},
            )
            if warning:
                logger.warning(warning)
                warning = "部分浏览器保存的设置无效或不可用，已恢复默认值。"
            return (
                values["text"], values["mode"],
                gr.update(value=values["category"], interactive=values["mode"] == "manual"),
                values["strength"], gr.update(choices=choices, value=values["profile"]),
                values["language"], values["device"], values["advanced"],
                set_advanced(values["advanced"]),
                summarize_settings(values["language"], values["device"], values["advanced"]),
                warning, True,
            )

        def remember_inputs(text_value, selected_mode, selected_category, selected_strength,
                            selected_profile, selected_language, selected_device, enabled, ready):
            if not ready:
                return gr.skip()
            return dict(text=text_value, mode=selected_mode, category=selected_category,
                        strength=selected_strength, profile=selected_profile,
                        language=selected_language, device=selected_device, advanced=enabled)

        controls = (text, mode, category, strength, profile, language, device, advanced)
        mode.change(set_category_interactive, inputs=mode, outputs=category, queue=False)
        advanced.change(set_advanced, inputs=advanced, outputs=advanced_panel, queue=False)
        gr.on(
            triggers=[language.change, device.change, advanced.change], fn=summarize_settings,
            inputs=(language, device, advanced), outputs=settings_summary, queue=False,
        )
        generation = generate_button.click(
            generate, inputs=controls, outputs=(audio, status),
            concurrency_id="voice", concurrency_limit=1,
        )
        cancel_button.click(
            cancel_generation, outputs=status, queue=False, cancels=[generation],
        )
        gr.on(
            triggers=[control.change for control in controls], fn=remember_inputs,
            inputs=(*controls, initialized), outputs=saved_inputs, queue=False,
            trigger_mode="always_last", show_progress="hidden",
        )
        management_outputs = (management_profile, rename_name, confirm, empty_profiles, rename_button, delete_button)
        management_profile.input(
            refresh_management, inputs=management_profile, outputs=management_outputs, queue=False,
        )
        profiles_tab.select(
            refresh_management, inputs=management_profile, outputs=management_outputs,
            concurrency_id="voice", concurrency_limit=1,
        )
        app.load(
            refresh_management, inputs=management_profile, outputs=management_outputs,
            concurrency_id="voice", concurrency_limit=1,
        )
        create_button.click(
            create_profile, inputs=(name, upload), outputs=(profile, profile_status),
            concurrency_id="voice", concurrency_limit=1,
        ).then(
            refresh_after_create, inputs=(profile, management_profile, profile_status), outputs=management_outputs,
            concurrency_id="voice", concurrency_limit=1,
        )
        rename_button.click(
            rename_profile, inputs=(management_profile, rename_name, profile), outputs=(profile, profile_status),
            concurrency_id="voice", concurrency_limit=1,
        ).then(
            refresh_management, inputs=management_profile, outputs=management_outputs,
            concurrency_id="voice", concurrency_limit=1,
        )
        delete_button.click(
            delete_profile, inputs=(management_profile, confirm, profile), outputs=(profile, profile_status, confirm),
            concurrency_id="voice", concurrency_limit=1,
        ).then(
            refresh_management, inputs=management_profile, outputs=management_outputs,
            concurrency_id="voice", concurrency_limit=1,
        )
        refresh_button.click(
            refresh_profiles, inputs=profile, outputs=(profile, profile_status),
            concurrency_id="voice", concurrency_limit=1,
        ).then(
            refresh_management, inputs=management_profile, outputs=management_outputs,
            concurrency_id="voice", concurrency_limit=1,
        )
        app.load(
            restore_inputs, inputs=saved_inputs,
            outputs=(*controls, advanced_panel, settings_summary, status, initialized), queue=False,
        )
    return app


def launch(argv: Sequence[str] | None = None) -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(prog="voice-reader-web")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--runtime", choices=("auto", "standard", "low-vram"), default="auto")
    args = parser.parse_args(argv)
    store = ProfileStore()
    engine = SpeechEngine(device=args.device, profile_store=store, runtime=args.runtime)
    try:
        app = build_ui(engine, profile_store=store)
    except (OSError, RuntimeError, ValueError) as error:
        logger.exception("Web startup failed.")
        parser.exit(1, f"Error: {error}\n")
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        max_file_size=MAX_AUDIO_BYTES,
        blocked_paths=[str(store.root)],
        i18n=FRAMEWORK_I18N,
    )
