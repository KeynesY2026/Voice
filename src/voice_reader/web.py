import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from voice_reader.config import OUTPUT_DIR, PRIVATE_DATA_DIR
from voice_reader.engine import EMOTION_CATEGORIES, SpeechEngine, make_emotion_settings
from voice_reader.profiles import MAX_AUDIO_BYTES, ProfileStore

os.environ.setdefault("GRADIO_TEMP_DIR", str(PRIVATE_DATA_DIR / "web-cache"))

import gradio as gr

logger = logging.getLogger(__name__)
DEFAULT_PROFILE = "default"


def build_ui(engine: SpeechEngine, profile_store: ProfileStore | None = None) -> gr.Blocks:
    store = profile_store if profile_store is not None else engine.profile_store

    def profile_choices():
        return [("Default reference recording", DEFAULT_PROFILE)] + [
            (f"{profile.name} ({profile.id})", profile.id) for profile in store.list_profiles()
        ]

    with gr.Blocks(title="Local Voice Reader", analytics_enabled=False, delete_cache=(86400, 86400)) as app:
        gr.Markdown("# Local Voice Reader")
        gr.Textbox(value=engine.device_info.describe(), label="Runtime device (restart to change)", interactive=False)
        profile = gr.Dropdown(
            choices=[("Default reference recording", DEFAULT_PROFILE)],
            value=DEFAULT_PROFILE, label="Voice profile", interactive=True,
        )
        with gr.Tab("Generate speech"):
            text = gr.Textbox(label="English text", lines=8, placeholder="Enter text to speak...")
            mode = gr.Radio(("auto", "manual"), value="auto", label="Emotion mode")
            category = gr.Dropdown(EMOTION_CATEGORIES, value="calm", label="Manual emotion", interactive=False)
            strength = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Emotion strength")
            generate_button = gr.Button("Generate speech", variant="primary")
            audio = gr.Audio(label="Generated WAV", type="filepath")
            status = gr.Markdown()
        with gr.Tab("Voice profiles"):
            gr.Markdown(
                "Create a reusable voice from your own or an authorized recording. "
                "This is zero-shot cloning, not model training. "
                "Upload WAV / MP3 / M4A / FLAC, up to **100 MiB**. "
                "Long recordings are welcome: click the scissors icon (Trim audio to selection), "
                "drag the selection boundaries, then click Trim and play the clip "
                "before creating a profile. "
                "The saved clip must be **3-30 seconds**; **3-15 seconds is recommended** "
                "because the model uses only the first 15 seconds. Choose a single clear speaker. "
                "Files stay local and the original file is not modified."
            )
            name = gr.Textbox(label="Profile name")
            upload = gr.Audio(
                label="Reference recording - crop and preview",
                sources=["upload"], type="filepath", format=None, interactive=True, editable=True,
            )
            create_button = gr.Button("Create profile")
            refresh_button = gr.Button("Refresh profiles")
            rename_button = gr.Button("Rename selected profile")
            confirm = gr.Checkbox(label="I confirm permanent deletion of the selected profile", value=False)
            delete_button = gr.Button("Delete selected profile", variant="stop")
            profile_status = gr.Markdown()

        def set_category_interactive(selected_mode: str):
            return gr.update(interactive=selected_mode == "manual")

        def generate(
            text_value: str, selected_mode: str, selected_category: str, selected_strength: float,
            selected_profile: str | None = DEFAULT_PROFILE,
        ):
            try:
                if selected_profile is None:
                    raise ValueError("Select a voice profile or the default reference recording.")
                emotion = make_emotion_settings(selected_mode, selected_category, selected_strength)
                output = engine.synthesize(
                    text_value, OUTPUT_DIR / f"web-{uuid4()}.wav", emotion,
                    profile_id=None if selected_profile == DEFAULT_PROFILE else selected_profile,
                )
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Speech generation failed.")
                return None, f"**Generation failed:** {error}"
            return str(output), f"Audio ready. {engine.device_info.describe()}"

        def refresh_profiles(selected_profile):
            try:
                choices = profile_choices()
                selected = selected_profile if any(value == selected_profile for _, value in choices) else None
                return gr.update(choices=choices, value=selected), "Profiles refreshed."
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Profile refresh failed.")
                return gr.update(), f"**Profile refresh failed:** {error}"

        def create_profile(profile_name, source):
            try:
                if not source:
                    raise ValueError("Upload a reference recording first.")
                created = store.create(profile_name, Path(source))
                return gr.update(choices=profile_choices(), value=created.id), f"Profile created: {created.name}"
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Profile creation failed.")
                return gr.update(), f"**Profile creation failed:** {error}"

        def rename_profile(profile_id, profile_name):
            try:
                if not profile_id or profile_id == DEFAULT_PROFILE:
                    raise ValueError("Select a saved profile to rename.")
                renamed = store.rename(profile_id, profile_name)
                return gr.update(choices=profile_choices(), value=renamed.id), f"Profile renamed: {renamed.name}"
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Profile rename failed.")
                return gr.update(), f"**Profile rename failed:** {error}"

        def delete_profile(profile_id, confirmed):
            try:
                if not confirmed:
                    raise ValueError("Confirm permanent deletion before deleting.")
                if not profile_id or profile_id == DEFAULT_PROFILE:
                    raise ValueError("Select a saved profile to delete.")
                store.delete(profile_id)
                return gr.update(choices=profile_choices(), value=None), "Profile deleted. Select a voice before generating.", False
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Profile deletion failed.")
                return gr.update(), f"**Profile deletion failed:** {error}", False

        mode.change(set_category_interactive, inputs=mode, outputs=category)
        generate_button.click(
            generate, inputs=(text, mode, category, strength, profile), outputs=(audio, status),
            concurrency_id="voice", concurrency_limit=1,
        )
        create_button.click(
            create_profile, inputs=(name, upload), outputs=(profile, profile_status),
            concurrency_id="voice", concurrency_limit=1,
        )
        rename_button.click(
            rename_profile, inputs=(profile, name), outputs=(profile, profile_status),
            concurrency_id="voice", concurrency_limit=1,
        )
        delete_button.click(
            delete_profile, inputs=(profile, confirm), outputs=(profile, profile_status, confirm),
            concurrency_id="voice", concurrency_limit=1,
        )
        refresh_button.click(
            refresh_profiles, inputs=profile, outputs=(profile, profile_status),
            concurrency_id="voice", concurrency_limit=1,
        )
        app.load(
            refresh_profiles, inputs=profile, outputs=(profile, profile_status),
            concurrency_id="voice", concurrency_limit=1,
        )
    return app


def launch(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="voice-reader-web")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    store = ProfileStore()
    engine = SpeechEngine(device=args.device, profile_store=store)
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
    )
