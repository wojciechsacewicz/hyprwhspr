import sys
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "lib" / "src"))

from test_nemotron_streaming_backend import (
    FakeGenerator,
    FakeGeneratorParams,
    FakeProcessor,
    FakeTokenizer,
    ready_backend,
)


class BlockingProcessor(FakeProcessor):
    entered = threading.Event()
    release = threading.Event()

    def process(self, chunk):
        self.entered.set()
        self.release.wait(timeout=3.0)
        return super().process(chunk)


class NemotronWorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        BlockingProcessor.entered.clear()
        BlockingProcessor.release.clear()

    def tearDown(self):
        BlockingProcessor.release.set()

    def test_unload_waits_for_every_registered_worker(self):
        runtime = types.SimpleNamespace(
            StreamingProcessor=BlockingProcessor,
            Tokenizer=FakeTokenizer,
            Generator=FakeGenerator,
            GeneratorParams=FakeGeneratorParams,
        )
        backend = ready_backend(og=runtime)
        model = backend._model
        callback = backend.get_streaming_callback()
        callback(np.ones(4, dtype=np.float32))
        self.assertTrue(BlockingProcessor.entered.wait(timeout=1.0))

        result = []
        unload_thread = threading.Thread(
            target=lambda: result.append(backend.unload()), daemon=True
        )
        unload_thread.start()
        time.sleep(0.05)

        self.assertTrue(unload_thread.is_alive())
        self.assertIs(backend._model, model)
        self.assertTrue(backend._sessions)

        BlockingProcessor.release.set()
        unload_thread.join(timeout=2.0)

        self.assertFalse(unload_thread.is_alive())
        self.assertEqual(result, [True])
        self.assertIsNone(backend._model)
        self.assertFalse(backend._sessions)


if __name__ == "__main__":
    unittest.main()
