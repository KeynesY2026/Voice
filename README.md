# Local Voice Reader

Generate English speech locally using saved voice profiles or the original default recording. CPU and NVIDIA CUDA installations share the same application, models, and profiles.

The tool uses IndexTTS-2.5 **zero-shot voice cloning**, with automatic or manual emotion control. Creating a profile stores a reference recording; it does **not** train or fine-tune model weights. Text and recordings are not uploaded. Package and model downloads require internet access.

## Setup

Requirements: Windows, Python 3.11, and [`uv`](https://docs.astral.sh/uv/). The declared Python range remains 3.10–3.11.

Choose **one** installation:

### CPU

```powershell
uv sync --extra cpu --group dev
uv run --extra cpu hf download IndexTeam/IndexTTS-2.5 --local-dir checkpoints
uv run --extra cpu voice-reader-web --device cpu
```

This installs CPU-only PyTorch and does not require an NVIDIA GPU.

### NVIDIA GPU (CUDA)

```powershell
uv sync --extra gpu --group dev
uv run --extra gpu hf download IndexTeam/IndexTTS-2.5 --local-dir checkpoints
uv run --extra gpu voice-reader-web --device auto
```

Requires a CUDA 12.8-compatible NVIDIA driver and sufficient GPU memory. The CUDA runtime comes with PyTorch; a local CUDA compiler toolkit, DeepSpeed, FlashAttention and custom CUDA kernels are not required. Reduced precision and compilation-based accelerators are disabled in this version.

Always use the matching `--extra cpu` or `--extra gpu` on subsequent `uv run` commands. Do not select both extras or use `--all-extras`. To keep both environments installed without replacing PyTorch in the CPU environment:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-gpu"
uv sync --extra gpu --group dev
uv run --extra gpu voice-reader-web --device auto
# After stopping the server, return to the default .venv:
Remove-Item Env:\UV_PROJECT_ENVIRONMENT
```

Keep downloaded models in `checkpoints/`. On first generation, upstream may also download supporting model assets. The first request loads the model; CPU generation can be slow.

### Device selection

Both `voice-reader speak` and `voice-reader-web` accept `--device auto|cpu|cuda`:

| Selection          | Behavior                                                               |
| ------------------ | ---------------------------------------------------------------------- |
| `auto` (default) | Use `cuda:0` when CUDA is available; otherwise explicitly report CPU |
| `cpu`            | Force CPU, including the text-emotion model                            |
| `cuda`           | Require CUDA; report an error if unavailable                           |

The CUDA installation can also run on CPU. The CPU installation cannot use CUDA. Errors during model loading or inference, including insufficient GPU memory, are reported; they do not silently retry on CPU. The UI displays the actual selected device. Restart the web server to change devices; switching voice profiles does not reload the model.

## Voice profiles

Use your own recording or a recording you have permission to use. Provide one clear speaker without music, preferably in English:

- WAV, MP3, M4A or FLAC.
- The saved reference clip must be between **3 and 30 seconds**, inclusive. Choose **3-15 seconds**
  of clear speech when possible: the current model uses only the first 15 seconds.
- At most **100 MiB** per file.
- The web UI accepts longer recordings for manual waveform cropping before profile creation.
  CLI imports still require an already prepared 3-30 second clip.
- Invalid, empty or overlong reference clips are rejected, not automatically trimmed.

Create a profile and list its stable ID:

```powershell
uv run --extra cpu voice-reader profiles create --name "Narrator" --audio .\private-data\sample.wav
uv run --extra cpu voice-reader profiles list
```

Use the returned ID, not the display name:

```powershell
uv run --extra cpu voice-reader speak --text "Hello there." --profile PROFILE_ID --device cpu
uv run --extra cpu voice-reader profiles rename PROFILE_ID --name "My narrator"
uv run --extra cpu voice-reader profiles delete PROFILE_ID --yes
```

Replace `PROFILE_ID` with the ID from the list. Names support Unicode and cannot be empty or duplicate another name (case-insensitive). Rename preserves the ID. Delete requires explicit confirmation and removes only that profile's stored data; it does not delete the original recording or previously generated audio.

Profiles are stored in `private-data/profiles/<id>/`, including metadata, an imported recording copy, and a 24 kHz mono PCM WAV. They persist across restarts and remain usable after the original upload is removed. Back up the entire profile directory to retain the voice. Profile management does not load the speech model or need a GPU.

Run only one profile-management instance at a time; simultaneous CLI/web writes from multiple processes are not supported.

## Command line

Speak a string:

```powershell
uv run --extra cpu voice-reader speak --text "Thank you for watching."
```

Speak a UTF-8 text file:

```powershell
uv run --extra cpu voice-reader speak --text-file .\private-data\script.txt --output .\private-data\outputs\script.wav
```

Read from standard input:

```powershell
Get-Content -Raw .\private-data\script.txt | uv run --extra cpu voice-reader speak --stdin
```

Automatic emotion is the default. Choose a manual category and strength from `0.0` to `1.0`:

```powershell
uv run --extra cpu voice-reader speak --text "That is wonderful news." --emotion happy --strength 0.7
```

Supported manual categories: `happy`, `angry`, `sad`, `afraid`, `disgusted`, `melancholic`, `surprised`, and `calm`. By default, the command writes `private-data/outputs/speech.wav`. The output path is printed to stdout; diagnostics go to stderr.

Omitting `--profile` preserves the old default reference at `private-data/reference.m4a`. The first run converts it to a cached WAV without changing the source. If that recording is absent, create/select a profile or supply the default recording. An invalid explicit profile never falls back to another voice.

## Local web UI

On Windows, after setup, double-click `start-auto.bat`. It uses `.venv-gpu` when present,
otherwise `.venv` with its installed PyTorch variant, and automatically selects CUDA when
available or CPU otherwise. Keep the command window open while using the app.

To explicitly start the CPU version:

```powershell
uv run --extra cpu voice-reader-web --device cpu
```

Open `http://127.0.0.1:7860`. The server binds to loopback only and does not create a public share link.

1. In **Voice profiles**, upload a recording under **Reference recording - crop and preview**.
2. For a long recording, click the scissors icon (**Trim audio to selection**), then drag the
   left/right selection boundaries to choose a 3-30 second clip (3-15 seconds recommended)
   and click **Trim**. Play the resulting clip to check it before saving.
   To choose another section, upload the original again.
3. Enter a name and click **Create profile**. Only the current, cropped audio is imported;
   the original file on your computer is unchanged. The saved profile remains usable after
   the temporary upload is removed. An unedited recording already within the limits also works.
4. The newly created profile is selected automatically. Choose another profile in the dropdown at any time; saved profiles show their names and IDs.
5. In **Generate speech**, enter English text, select emotion settings and generate a WAV.
6. The profile tab also provides refresh, rename and confirmed deletion. Deleting the selected profile clears the selection; explicitly choose a voice before generating again.

Requests are serialized to protect the shared model. Each web generation gets a separate output file so later requests do not overwrite earlier audio.

## Privacy and troubleshooting

- All profile metadata, uploaded copies, web caches and generated outputs are under ignored `private-data/`. This spelling differs from `private_data`; the application does not migrate files between them.
- Do not add recordings to Git. The entire `private-data/` directory, models and environments are excluded; `.wav` and `.m4a` also have global ignore rules. MP3/FLAC source files outside the private directory are **not** globally ignored.
- The stored reference directory is blocked from direct Gradio downloads. Uploaded temporary copies are separate from saved profiles and may remain in the private web cache until cleanup. Generated outputs remain until you remove them.
- Git ignore rules are not encryption or access controls. Use operating-system permissions appropriate for private recordings.
- **CUDA unavailable:** use the GPU installation, verify `nvidia-smi` and the driver, or explicitly select CPU.
- **GPU out of memory:** close other GPU workloads, use shorter text, or restart with `--device cpu`. Automatic emotion detection loads an additional model.
- **Invalid recording:** provide a decodable, single-speaker clip within the limits above.
  For a long recording, apply a crop in the web waveform editor before creating the profile;
  no automatic clipping is performed. Browser decoding and waveform editing can be slow for
  large recordings; use a shorter local file if your browser cannot load it.
- **Corrupt/missing profile:** the error includes the profile ID. Restore its directory from a backup; the app does not silently hide corruption or substitute another voice.

## Development checks

```powershell
uv lock --check
uv run --extra cpu pytest -q
```

Unit tests include real FFmpeg import checks and model doubles for deterministic inference contracts. Passing them does not by itself establish GPU synthesis performance or voice similarity; those require real checkpoints, hardware and listening tests.

## Notes

- The IndexTTS source revision remains pinned in `pyproject.toml` and `uv.lock`. It is installed from the revision's source archive so upstream Git development-source overrides cannot force CUDA wheels into the CPU installation.
- The pinned upstream emotion loader independently selects devices. A small local adapter preserves its inference and emotion configuration while loading its weights on the explicitly selected device.
- Follow the upstream [license and usage terms](https://github.com/index-tts/index-tts).
