"""Non-blocking audio queue and isolated ONNX streaming session."""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

try:
    from ..dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package("numpy")

from .nemotron_languages import _LANGUAGE_TAG_RE, language_id_for

_FINISH = object()


class _NemotronSession:
    """One isolated cache/generator lifecycle for one dictated utterance."""

    def __init__(
        self,
        backend: "NemotronStreamingBackend",
        generation_id: int,
        input_sample_rate: int,
        language: Optional[str],
        publish_partials: bool,
    ):
        self.backend = backend
        self.generation_id = generation_id
        self.input_sample_rate = int(input_sample_rate)
        self.language = language
        self.publish_partials = publish_partials

        self._queue: "queue.Queue[object]" = queue.Queue()
        self._queue_lock = threading.Lock()
        self._queued_input_samples = 0
        self._received_input_samples = 0
        self._finish_requested = threading.Event()
        self._cancel_requested = threading.Event()
        self._done = threading.Event()
        self._error: Optional[str] = None
        self._text = ""
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name=f"hyprwhspr-nemotron-{generation_id}",
            daemon=True,
        )

        max_seconds = float(
            backend.config.get_setting("nemotron_streaming_buffer_max_seconds", 3.0)
        )
        self._max_queued_input_samples = max(
            1, int(max_seconds * self.input_sample_rate)
        )
        self._thread.start()

    @property
    def has_audio(self) -> bool:
        with self._queue_lock:
            return self._received_input_samples > 0

    def set_input_sample_rate(self, sample_rate: int) -> None:
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            return
        with self._queue_lock:
            if self._received_input_samples:
                if sample_rate != self.input_sample_rate:
                    print(
                        "[NEMOTRON] Ignoring sample-rate change after audio started "
                        f"({self.input_sample_rate} -> {sample_rate})",
                        flush=True,
                    )
                return
            self.input_sample_rate = sample_rate
            max_seconds = float(
                self.backend.config.get_setting(
                    "nemotron_streaming_buffer_max_seconds", 3.0
                )
            )
            self._max_queued_input_samples = max(1, int(max_seconds * sample_rate))

    def enqueue(self, audio_chunk: np.ndarray) -> bool:
        if self._finish_requested.is_set() or self._cancel_requested.is_set():
            return False

        audio = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if not audio.size:
            return True
        if not audio.flags.c_contiguous:
            audio = np.ascontiguousarray(audio, dtype=np.float32)

        # AudioCapture already passes a copy. Keep this path allocation-light and
        # never wait on inference or a bounded Queue while PortAudio holds its lock.
        with self._queue_lock:
            projected = self._queued_input_samples + len(audio)
            if projected > self._max_queued_input_samples:
                self._error = (
                    "streaming worker fell behind "
                    f"(buffer>{self._max_queued_input_samples / self.input_sample_rate:.1f}s)"
                )
                self._cancel_requested.set()
                try:
                    self._queue.put_nowait(_FINISH)
                except queue.Full:
                    pass
                return False
            self._queued_input_samples = projected
            self._received_input_samples += len(audio)

        self._queue.put_nowait(audio)
        return True

    def finish(self, timeout: float) -> str:
        if not self._finish_requested.is_set():
            self._finish_requested.set()
            self._queue.put_nowait(_FINISH)

        if not self._done.wait(timeout=max(0.1, float(timeout))):
            self._error = f"finalization timed out after {timeout:.1f}s"
            self.cancel()
            print(f"[NEMOTRON] {self._error}", flush=True)
            return ""

        if self._error:
            print(f"[NEMOTRON] Session failed: {self._error}", flush=True)
            return ""
        return self._text.strip()

    def cancel(self) -> None:
        self._cancel_requested.set()
        self._finish_requested.set()
        try:
            self._queue.put_nowait(_FINISH)
        except queue.Full:
            pass
        self._done.wait(timeout=1.0)

    def _run(self) -> None:
        try:
            self._run_stream()
        except Exception as exc:
            self._error = str(exc)
        finally:
            self._done.set()

    def _run_stream(self) -> None:
        og = self.backend._og
        model = self.backend._model
        if og is None or model is None:
            raise RuntimeError("Nemotron model is not loaded")

        # AudioCapture announces the concrete device rate immediately before the
        # first callback. Waiting for that first queue item avoids constructing a
        # 16 kHz resampler from the temporary default before a 44.1/48 kHz device
        # has supplied its actual rate.
        first_item = self._queue.get()
        if first_item is _FINISH or self._cancel_requested.is_set():
            return

        processor = og.StreamingProcessor(model)
        use_vad = bool(
            self.backend.config.get_setting("nemotron_streaming_use_vad", False)
        )
        processor.set_option("use_vad", "true" if use_vad else "false")

        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
        generator = og.Generator(model, og.GeneratorParams(model))
        generator.set_runtime_option("lang_id", str(language_id_for(self.language)))

        resampler = None
        if self.input_sample_rate != self.backend._model_sample_rate:
            try:
                import soxr
            except ImportError as exc:
                raise RuntimeError(
                    "python-soxr is required when the microphone does not run at 16 kHz"
                ) from exc
            resampler = soxr.ResampleStream(
                self.input_sample_rate,
                self.backend._model_sample_rate,
                1,
                dtype="float32",
                quality="HQ",
            )

        pending = np.empty(0, dtype=np.float32)
        item = first_item
        while not self._cancel_requested.is_set():
            if item is _FINISH:
                break
            audio = item
            with self._queue_lock:
                self._queued_input_samples = max(
                    0, self._queued_input_samples - len(audio)
                )

            if resampler is not None:
                audio = resampler.resample_chunk(audio, last=False)
            if audio.size:
                pending = np.concatenate((pending, audio.astype(np.float32, copy=False)))
                pending = self._process_complete_chunks(
                    pending, processor, generator, tokenizer_stream
                )
            item = self._queue.get()

        if self._cancel_requested.is_set():
            return

        if resampler is not None:
            tail = resampler.resample_chunk(
                np.empty(0, dtype=np.float32), last=True
            )
            if tail.size:
                pending = np.concatenate((pending, tail.astype(np.float32, copy=False)))

        pending = self._process_complete_chunks(
            pending, processor, generator, tokenizer_stream
        )
        if pending.size:
            self._process_model_input(
                pending, processor, generator, tokenizer_stream
            )

        with self.backend._inference_lock:
            inputs = processor.flush()
            if inputs is not None:
                generator.set_inputs(inputs)
                self._drain_generator(generator, tokenizer_stream)

        elapsed = time.monotonic() - self._started_at
        audio_seconds = self._received_input_samples / max(1, self.input_sample_rate)
        rtf = elapsed / audio_seconds if audio_seconds else 0.0
        print(
            f"[NEMOTRON] Finalized {audio_seconds:.2f}s audio in {elapsed:.2f}s "
            f"(RTF={rtf:.2f})",
            flush=True,
        )

    def _process_complete_chunks(self, pending, processor, generator, tokenizer_stream):
        step = self.backend._chunk_samples
        while len(pending) >= step and not self._cancel_requested.is_set():
            chunk = pending[:step]
            pending = pending[step:]
            self._process_model_input(
                chunk, processor, generator, tokenizer_stream
            )
        return pending

    def _process_model_input(self, chunk, processor, generator, tokenizer_stream):
        with self.backend._inference_lock:
            inputs = processor.process(
                np.ascontiguousarray(chunk, dtype=np.float32)
            )
            if inputs is not None:
                generator.set_inputs(inputs)
                self._drain_generator(generator, tokenizer_stream)

    def _drain_generator(self, generator, tokenizer_stream) -> None:
        changed = False
        while not generator.is_done():
            generator.generate_next_token()
            tokens = generator.get_next_tokens()
            if len(tokens) <= 0:
                continue
            piece = tokenizer_stream.decode(tokens[0])
            piece = _LANGUAGE_TAG_RE.sub("", piece or "")
            if piece:
                self._text += piece
                changed = True
        if changed and self.publish_partials:
            self.backend._publish_partial(self, self._text.strip())
