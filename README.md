# Local Voice Reader

Generate English speech locally with the voice in `private-data/reference.m4a`. The tool uses IndexTTS-2.5 zero-shot voice cloning with automatic emotion detection by default, plus optional manual emotion control. It does not fine-tune or upload text or audio; model and package downloads require an internet connection.

## Setup

Requirements: Windows, Python 3.11, and [`uv`](https://docs.astral.sh/uv/).

```powershell
uv sync --group dev
uv run hf download IndexTeam/IndexTTS-2.5 --local-dir checkpoints
```

Keep the model files in `checkpoints/`. Place your reference recording at `private-data/reference.m4a`. The first run converts it into a cached mono WAV under `private-data/cache/`; the source recording remains unchanged.

The entire `private-data/` directory is excluded from Git. Keep private scripts and other sensitive inputs there too. Generated audio defaults to `private-data/outputs/`; `.m4a` and `.wav` files are also ignored elsewhere as a safeguard. Model files and runtime caches are excluded from source control. Git ignore rules are not encryption or access controls.

The verified machine has no NVIDIA GPU. CPU generation works but is slow: the supplied passage took about 3.7–4.2 minutes to produce 19–21 seconds of audio. The first generation also loads the model into memory.

## Command line

Speak a string:

```powershell
uv run voice-reader speak --text "Thank you for watching."
```

Speak a UTF-8 text file:

```powershell
uv run voice-reader speak --text-file .\private-data\script.txt --output .\private-data\outputs\script.wav
```

Read from standard input:

```powershell
Get-Content -Raw .\private-data\script.txt | uv run voice-reader speak --stdin
```

Automatic emotion is the default. Choose a manual category and strength from `0.0` to `1.0`:

```powershell
uv run voice-reader speak --text "That is wonderful news." --emotion happy --strength 0.7
```

Supported manual categories: `happy`, `angry`, `sad`, `afraid`, `disgusted`, `melancholic`, `surprised`, and `calm`. By default, the command writes `private-data/outputs/speech.wav`.

## Local web UI

```powershell
uv run voice-reader-web
```

Open `http://127.0.0.1:7860`. The server binds to loopback only and does not create a public share link. The page accepts English text, automatic or manual emotion, and emotion strength; generated audio is available in the browser.

## Notes

- Place the reference recording at `private-data/reference.m4a` or change `DEFAULT_REFERENCE` in `src/voice_reader/config.py`.
- Place the IndexTTS-2.5 checkpoint files in `checkpoints/`.
- The project pins the IndexTTS source revision in `pyproject.toml` and `uv.lock`.
- Follow the upstream [license and usage terms](https://github.com/index-tts/index-tts).
