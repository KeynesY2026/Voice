from __future__ import annotations

import atexit
import logging
import multiprocessing
import threading
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory

from voice_reader.device import DeviceInfo, DeviceMode
from voice_reader.console import configure_utf8_output
from voice_reader.engine import EmotionSettings, SpeechEngine
from voice_reader.profiles import ProfileStore

logger = logging.getLogger(__name__)


class GenerationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class _Job:
    text: str
    output: Path
    emotion: EmotionSettings | None
    profile_id: str | None
    language: str
    advanced: bool


@dataclass(frozen=True)
class _Result:
    device: DeviceInfo | None = None
    runtime: str = ""
    error: str | None = None


def _inference_worker(connection: Connection, config: dict, factory: Callable[..., SpeechEngine]) -> None:
    configure_utf8_output()
    try:
        store = ProfileStore(config.pop("profile_root"))
        engine = factory(profile_store=store, **config)
        while True:
            job = connection.recv()
            if job is None:
                break
            try:
                engine.synthesize(
                    job.text, job.output, job.emotion, profile_id=job.profile_id,
                    language=job.language,
                    advanced=job.advanced,
                )
                connection.send(_Result(engine.device_info, engine.runtime_description))
            except Exception as error:
                # This process boundary must report unexpected model errors to the UI too.
                logger.exception("Inference worker failed.")
                connection.send(_Result(error=f"{type(error).__name__}: {error}"))
    except (EOFError, BrokenPipeError):
        logger.info("Inference worker connection closed.")
    finally:
        connection.close()


class InferenceService:
    """Keep one warm model process; cancellation discards that process and its caches."""

    def __init__(
        self, engine: SpeechEngine, *, engine_factory: Callable[..., SpeechEngine] = SpeechEngine,
    ) -> None:
        self.profile_store = engine.profile_store
        self.device = engine.device
        self._info = engine.device_info
        self._runtime = engine.runtime_description
        self._config = dict(
            reference_path=engine.reference_path, checkpoint_dir=engine.checkpoint_dir,
            cache_dir=engine.cache_dir, runtime=engine.runtime, profile_root=self.profile_store.root,
        )
        self._factory = engine_factory
        self._context = multiprocessing.get_context("spawn")
        self._process = None
        self._connection: Connection | None = None
        self._worker_device: DeviceMode | None = None
        self._lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._owner: str | None = None
        self._cancelled = threading.Event()
        atexit.register(self.close)

    @property
    def device_info(self) -> DeviceInfo:
        return self._info

    @property
    def runtime_description(self) -> str:
        return self._runtime

    @property
    def worker_pid(self) -> int | None:
        with self._lock:
            return self._process.pid if self._process is not None else None

    def _stop_worker(self) -> None:
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
            self._process.close()
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._worker_device = None

    def _start_worker(self, device: DeviceMode) -> Connection:
        if self._process is None or not self._process.is_alive() or self._worker_device != device:
            self._stop_worker()
            parent, child = self._context.Pipe()
            config = {**self._config, "device": device}
            if device == "cpu" and config["runtime"] == "low-vram":
                config["runtime"] = "standard"
            process = self._context.Process(
                target=_inference_worker, args=(child, config, self._factory), daemon=True,
            )
            process.start()
            child.close()
            self._process, self._connection, self._worker_device = process, parent, device
        assert self._connection is not None
        return self._connection

    def synthesize(
        self, text: str, output_path: Path, emotion: EmotionSettings | None, *,
        profile_id: str | None = None, language: str = "auto",
        device: DeviceMode | None = None, owner: str = "local",
        advanced: bool = False,
    ) -> Path:
        selected_device = device if device is not None else self.device
        if selected_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Unsupported device selection.")
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".wav":
            raise ValueError("Output path must end in .wav.")
        resolved = output_path.resolve()
        if resolved.is_relative_to(self.profile_store.root):
            raise ValueError("Output cannot be inside the voice profile directory.")
        if resolved == Path(self._config["reference_path"]).resolve():
            raise ValueError("Output cannot overwrite the reference recording.")
        if resolved.is_relative_to(Path(self._config["cache_dir"]).resolve()):
            raise ValueError("Output cannot be inside the reference cache directory.")
        with self._generation_lock:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".inference-", dir=output_path.parent) as temporary:
                scratch = Path(temporary) / "audio.wav"
                try:
                    with self._lock:
                        self._cancelled.clear()
                        self._owner = owner
                        connection = self._start_worker(selected_device)
                        connection.send(_Job(text, scratch, emotion, profile_id, language, advanced))
                    while not connection.poll(0.05):
                        if self._cancelled.is_set():
                            raise GenerationCancelled("Generation cancelled.")
                        if self._process is None or not self._process.is_alive():
                            raise RuntimeError("Inference worker exited unexpectedly.")
                    result = connection.recv()
                    with self._lock:
                        if self._cancelled.is_set():
                            raise GenerationCancelled("Generation cancelled.")
                        if result.error:
                            raise RuntimeError(result.error)
                        if not scratch.is_file() or scratch.stat().st_size == 0:
                            raise RuntimeError("Inference worker returned no audio.")
                        assert result.device is not None
                        self._info, self._runtime = result.device, result.runtime
                        scratch.replace(output_path)
                        self._owner = None
                    return output_path
                except (EOFError, OSError) as error:
                    if self._cancelled.is_set():
                        raise GenerationCancelled("Generation cancelled.") from error
                    raise RuntimeError(f"Inference worker connection failed: {error}") from error
                finally:
                    with self._lock:
                        self._owner = None
                        if self._cancelled.is_set() or self._process is None or not self._process.is_alive():
                            self._stop_worker()

    def cancel(self, owner: str = "local") -> bool:
        with self._lock:
            if self._owner != owner:
                return False
            self._cancelled.set()
            if self._process is not None and self._process.is_alive():
                self._process.terminate()
            return True

    def close(self) -> None:
        with self._lock:
            self._cancelled.set()
            self._stop_worker()

    def __enter__(self) -> InferenceService:
        return self

    def __exit__(self, *args) -> None:
        self.close()
