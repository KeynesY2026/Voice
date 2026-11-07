import threading
from pathlib import Path

import pytest

from voice_reader.device import DeviceInfo
from voice_reader.engine import EmotionSettings, SpeechEngine
from voice_reader.inference_service import GenerationCancelled, InferenceService


class TestWorkerEngine:
    __test__ = False

    def __init__(self, **kwargs):
        self.device_info = DeviceInfo(kwargs["device"], kwargs["device"], "Test", "Test")
        self.runtime_description = "Test worker"

    def synthesize(self, text, output_path, emotion, *, profile_id=None, language="auto", advanced=False):
        import time
        Path(output_path).write_text(f"{text}:{language}:{self.device_info.resolved}")
        if text == "wait":
            while True:
                time.sleep(0.05)
        if text == "fail":
            raise RuntimeError("Deliberate worker failure")
        return Path(output_path)


def service(tmp_path):
    return InferenceService(
        SpeechEngine(device="cpu", cache_dir=tmp_path / "cache"),
        engine_factory=TestWorkerEngine,
    )


def test_worker_reuses_model_and_switches_device(tmp_path):
    with service(tmp_path) as worker:
        first = worker.synthesize("First", tmp_path / "first.wav", EmotionSettings(), language="zh")
        pid = worker.worker_pid
        second = worker.synthesize("Second", tmp_path / "second.wav", EmotionSettings())
        assert first.read_text() == "First:zh:cpu"
        assert second.read_text() == "Second:auto:cpu"
        assert worker.worker_pid == pid
        # Fake engine supports this device without initializing real CUDA.
        worker.synthesize("Third", tmp_path / "third.wav", EmotionSettings(), device="auto")
        assert worker.worker_pid != pid
    assert worker.worker_pid is None


def test_cancel_stops_worker_cleans_partial_output_and_can_generate_again(tmp_path):
    with service(tmp_path) as worker:
        errors = []

        def run():
            try:
                worker.synthesize("wait", tmp_path / "partial.wav", EmotionSettings(), owner="one")
            except GenerationCancelled as error:
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        import time
        deadline = time.monotonic() + 10
        while not list(tmp_path.glob(".inference-*/audio.wav")) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert list(tmp_path.glob(".inference-*/audio.wav"))
        assert not worker.cancel("other")
        assert worker.cancel("one")
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert not (tmp_path / "partial.wav").exists()
        assert not list(tmp_path.glob(".inference-*"))
        output = worker.synthesize("Again", tmp_path / "again.wav", EmotionSettings())
        assert output.read_text().startswith("Again")


def test_worker_reports_failure_without_returning_partial_audio(tmp_path):
    with service(tmp_path) as worker:
        with pytest.raises(RuntimeError, match="Deliberate"):
            worker.synthesize("fail", tmp_path / "failed.wav", EmotionSettings())
        assert not (tmp_path / "failed.wav").exists()
        assert not list(tmp_path.glob(".inference-*"))


def test_service_cannot_replace_reference_or_profile_audio(tmp_path):
    engine = SpeechEngine(reference_path=tmp_path / "reference.wav", cache_dir=tmp_path / "cache")
    with InferenceService(engine, engine_factory=TestWorkerEngine) as worker:
        for output in (engine.reference_path, engine.profile_store.root / "anything.wav",
                       engine.cache_dir / "reference.wav"):
            with pytest.raises(ValueError, match="reference|profile"):
                worker.synthesize("Hello", output, EmotionSettings())
