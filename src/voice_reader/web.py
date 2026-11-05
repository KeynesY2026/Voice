import logging

import gradio as gr

from voice_reader.config import OUTPUT_DIR
from voice_reader.engine import EMOTION_CATEGORIES, SpeechEngine, make_emotion_settings

logger = logging.getLogger(__name__)


def build_ui(engine: SpeechEngine) -> gr.Blocks:
    with gr.Blocks(title="Local Voice Reader") as app:
        gr.Markdown("# Local Voice Reader")
        text = gr.Textbox(label="English text", lines=8, placeholder="Enter text to speak...")
        mode = gr.Radio(("auto", "manual"), value="auto", label="Emotion mode")
        category = gr.Dropdown(EMOTION_CATEGORIES, value="calm", label="Manual emotion", interactive=False)
        strength = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Emotion strength")
        generate_button = gr.Button("Generate speech", variant="primary")
        audio = gr.Audio(label="Generated WAV", type="filepath")
        status = gr.Markdown()

        def set_category_interactive(selected_mode: str):
            return gr.update(interactive=selected_mode == "manual")

        def generate(text_value: str, selected_mode: str, selected_category: str, selected_strength: float):
            try:
                emotion = make_emotion_settings(selected_mode, selected_category, selected_strength)
                output = engine.synthesize(text_value, OUTPUT_DIR / "web-speech.wav", emotion)
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Speech generation failed.")
                return None, f"**Generation failed:** {error}"
            return str(output), "Audio ready."

        mode.change(set_category_interactive, inputs=mode, outputs=category)
        generate_button.click(generate, inputs=(text, mode, category, strength), outputs=(audio, status))
    return app


def launch() -> None:
    build_ui(SpeechEngine()).queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
