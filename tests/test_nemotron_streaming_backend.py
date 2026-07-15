import contextlib
import io
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends.nemotron_streaming_backend import NemotronStreamingBackend
from backends.nemotron_languages import language_id_for


class FakeConfig:
    def __init__(self, values=None):
        self.values = {
            "language": "pl-PL",
            "recording_mode": "toggle",
            "nemotron_streaming_buffer_max_seconds": 3.0,
            "nemotron_streaming_finalize_timeout": 2.0,
            "nemotron_streaming_use_vad": False,
        }
        self.values.update(values or {})

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


class FakeManager:
    def __init__(self, config=None):
        self.config = config or FakeConfig()
        self.temp_dir = None
        self.ready = False
        self.current_model = None
        self._last_use_time = 0.0
        self._streaming_partial_callback = None
        self._realtime_partial_callback = None


class FakeProcessor:
    def __init__(self, _model):
        self.options = {}

    def set_option(self, key, value):
        self.options[key] = value

    def process(self, chunk):
        return np.asarray(chunk, dtype=np.float32)

    def flush(self):
        return None


class SlowProcessor(FakeProcessor):
    def process(self, chunk):
        time.sleep(0.08)
        return super().process(chunk)


class FakeTokenizerStream:
    def decode(self, token):
        return str(token)


class FakeTokenizer:
    def __init__(self, _model):
        pass

    def create_stream(self):
        return FakeTokenizerStream()


class FakeGenerator:
    def __init__(self, _model, _params):
        self._tokens = []
        self._next = []

    def set_runtime_option(self, _key, _value):
        pass

    def set_inputs(self, inputs):
        self._tokens = [f"w{len(inputs)} "]

    def is_done(self):
        return not self._tokens

    def generate_next_token(self):
        self._next = [self._tokens.pop(0)]

    def get_next_tokens(self):
        result = self._next
        self._next = []
        return result


class FakeGeneratorParams:
    def __init__(self, _model):
        pass


class FakeOG:
    StreamingProcessor = FakeProcessor
    Tokenizer = FakeTokenizer
    Generator = FakeGenerator
    GeneratorParams = FakeGeneratorParams


def ready_backend(config=None, og=FakeOG):
    manager = FakeManager(config)
    backend = NemotronStreamingBackend(manager)
    backend._og = og
    backend._model = object()
    backend._model_sample_rate = 16000
    backend._chunk_samples = 4
    manager.ready = True
    return backend


class NemotronLanguageTests(unittest.TestCase):
    def test_polish_locale_uses_polish_prompt_id(self):
        self.assertEqual(language_id_for("pl-PL"), 17)
        self.assertEqual(language_id_for("pl"), 17)

    def test_missing_and_unknown_language_use_auto_detection(self):
        self.assertEqual(language_id_for(None), 101)
        self.assertEqual(language_id_for("xx-ZZ"), 101)


class NemotronStreamingTests(unittest.TestCase):
    def test_backend_is_first_class_local_streaming_backend(self):
        backend = ready_backend()
        self.assertEqual(backend.name, "nemotron-streaming")
        self.assertTrue(backend.is_local)
        self.assertTrue(backend.supports_streaming_capture)
        self.assertTrue(backend.supports_partial_transcripts)
        self.assertTrue(backend.requires_streaming_capture)

    def test_audio_callback_is_non_blocking_while_worker_is_slow(self):
        slow_og = types.SimpleNamespace(
            StreamingProcessor=SlowProcessor,
            Tokenizer=FakeTokenizer,
            Generator=FakeGenerator,
            GeneratorParams=FakeGeneratorParams,
        )
        backend = ready_backend(og=slow_og)
        callback = backend.get_streaming_callback()
        started = time.perf_counter()
        callback(np.ones(4, dtype=np.float32))
        callback_elapsed = time.perf_counter() - started
        self.assertLess(callback_elapsed, 0.05)
        self.assertEqual(
            backend.transcribe(np.ones(4, dtype=np.float32)), "w4"
        )

    def test_streaming_session_emits_partial_and_final_text(self):
        backend = ready_backend()
        partials = []
        backend.apply_partial_callback(partials.append)
        callback = backend.get_streaming_callback()
        callback(np.ones(8, dtype=np.float32))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and "w4 w4" not in partials:
            time.sleep(0.005)
        self.assertEqual(
            backend.transcribe(np.ones(8, dtype=np.float32)), "w4 w4"
        )
        self.assertIn("w4 w4", partials)

    def test_batch_fallback_sizes_queue_for_long_form_audio(self):
        backend = ready_backend(
            FakeConfig({"nemotron_streaming_buffer_max_seconds": 0.001})
        )
        audio = np.ones(160, dtype=np.float32)
        backend._chunk_samples = 40
        result = backend.transcribe(audio, sample_rate=100)
        self.assertTrue(result)
        self.assertGreater(len(result.split()), 4)

    def test_cancel_discards_session_and_clears_preview(self):
        backend = ready_backend()
        partials = []
        backend.apply_partial_callback(partials.append)
        backend.get_streaming_callback()(np.ones(4, dtype=np.float32))
        self.assertTrue(backend.cancel_stream(timeout=1.0))
        self.assertIsNone(backend._active_session)
        self.assertEqual(partials[-1], "")

    def test_continuous_mode_swaps_session_before_finalizing(self):
        config = FakeConfig({"recording_mode": "continuous"})
        backend = ready_backend(config)
        callback = backend.get_streaming_callback()
        callback(np.ones(4, dtype=np.float32))
        self.assertEqual(
            backend.transcribe(np.ones(4, dtype=np.float32)), "w4"
        )
        self.assertIsNotNone(backend._active_session)
        config.values["recording_mode"] = "toggle"
        backend._active_session.enqueue(np.ones(4, dtype=np.float32))
        self.assertEqual(
            backend.transcribe(np.ones(4, dtype=np.float32)), "w4"
        )

    def test_queue_overflow_is_reported_only_once(self):
        backend = ready_backend(
            FakeConfig({"nemotron_streaming_buffer_max_seconds": 0.001})
        )
        callback = backend.get_streaming_callback()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            callback(np.ones(32, dtype=np.float32))
            callback(np.ones(32, dtype=np.float32))
            callback(np.ones(32, dtype=np.float32))
            self.assertEqual(
                backend.transcribe(np.ones(32, dtype=np.float32)), ""
            )
        self.assertEqual(output.getvalue().count("worker fell behind"), 1)

    def test_capture_sample_rate_is_applied_before_first_audio(self):
        backend = ready_backend()
        callback = backend.get_streaming_callback()
        callback.set_input_sample_rate(48000)
        self.assertEqual(backend._active_session.input_sample_rate, 48000)

    def test_metrics_report_compute_rtf_not_session_wall_time(self):
        backend = ready_backend()
        callback = backend.get_streaming_callback()
        callback(np.ones(8, dtype=np.float32))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                backend.transcribe(np.ones(8, dtype=np.float32)), "w4 w4"
            )
        metrics = output.getvalue()
        self.assertIn("compute_rtf=", metrics)
        self.assertIn("finalize=", metrics)
        self.assertIn("max_queue=", metrics)

    def test_close_cancels_without_unloading_model(self):
        backend = ready_backend()
        backend.get_streaming_callback()
        model = backend._model
        backend.close()
        self.assertIsNone(backend._active_session)
        self.assertIs(backend._model, model)


if __name__ == "__main__":
    unittest.main()
