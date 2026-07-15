# Local Nemotron 3.5 streaming

This optional backend adds fully local, cache-aware streaming transcription
with NVIDIA Nemotron 3.5 ASR Streaming 0.6B and ONNX Runtime GenAI.

It is opt-in and does not change hyprwhspr's default backend.

## Model and runtime

hyprwhspr currently supports one tested export:

- model: `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`
- revision: `8364d9e2dd9da23789b480bdbba9e423717e42ee`
- audio: 16 kHz mono
- streaming chunk: 8,960 samples / 560 ms
- runtime: ONNX Runtime GenAI 0.14.1

The ONNX repository is roughly 793 MB to download. Runtime memory usage is
higher than the model files and depends on the execution provider and session;
benchmark it on the target hardware instead of assuming a fixed RAM figure.

The base NVIDIA model is licensed under OpenMDW 1.1. The backend code remains
under hyprwhspr's repository license; model weights keep their own terms.

## Architecture

`nemotron-streaming` is a first-class local backend. It uses the same generic
streaming-capture capability as cloud realtime providers, without pretending to
be a WebSocket transport.

The PortAudio callback only copies audio into a sample-bounded queue. A worker
thread performs stateful SoXR resampling, forms the model's required 560 ms
chunks, advances ONNX cache state, and publishes partial transcripts.

Queue overflow fails the utterance explicitly. Audio is not silently dropped.
Runtime startup uses the local Hugging Face cache only; model downloads happen
through setup, never during systemd service startup.

## Setup

```bash
hyprwhspr nemotron setup --device auto --language pl-PL
```

`--device auto` selects CUDA when a working NVIDIA device is detected and CPU
otherwise. Omitting `--language` preserves the user's existing language setting.
An explicit locale such as `pl-PL` generally performs better than automatic
language detection for Polish.

Useful commands:

```bash
hyprwhspr nemotron status
hyprwhspr nemotron validate
hyprwhspr nemotron download
hyprwhspr nemotron benchmark ./sample.wav --language pl-PL
hyprwhspr model unload
hyprwhspr model reload
```

## Configuration

```jsonc
{
  "transcription_backend": "nemotron-streaming",
  "language": "pl-PL",
  "nemotron_streaming_device": "cpu",
  "nemotron_streaming_buffer_max_seconds": 3.0,
  "nemotron_streaming_finalize_timeout": 5.0,
  "nemotron_streaming_use_vad": false
}
```

Advanced model and revision overrides exist for testing compatible exports, but
only the pinned model above is supported. Other Nemotron exports may use a
different chunk size, graph structure, language prompt mapping, or runtime API.

Whisper prompts, beam search, and speech translation do not affect Nemotron.

## Recording modes

- `toggle`, `push_to_talk`, and `auto`: partials are previewed while recording;
  one final text is injected after stop.
- `continuous`: capture remains open while each utterance is finalized. The
  queue is bounded, so hardware that cannot finalize quickly enough fails
  explicitly instead of silently losing audio.
- `long_form` and file-based transcription: supplied audio is replayed through
  the same streaming runtime with queue capacity sized for the recording.
- cancel and recording-start recovery clear only the active utterance while
  keeping the model resident.

Partials are preview-only. The backend does not edit unstable partial text in
the focused application.

## Validation and benchmarks

Fast tests use a fake ONNX runtime and do not download model weights:

```bash
python -m unittest tests.test_nemotron_streaming_backend
python -m unittest tests.test_nemotron_commands
python -m unittest tests.test_streaming_backend_capabilities
python -m unittest tests.test_nemotron_worker_lifecycle
```

The real-runtime test needs an actual speech fixture:

```bash
HYPRWHSPR_RUN_MODEL_TESTS=1 \
HYPRWHSPR_NEMOTRON_TEST_AUDIO=/path/to/polish.wav \
HYPRWHSPR_NEMOTRON_EXPECTED_TEXT="oczekiwane stabilne słowa" \
python -m unittest tests.test_nemotron_model_integration
```

For realtime suitability, measure:

- average and maximum inference time per 560 ms chunk,
- compute RTF (`inference_seconds / audio_seconds`),
- first-partial latency,
- finalization latency after recording stops,
- maximum queued audio,
- peak RSS and CPU usage,
- at least ten minutes without queue growth or rejected chunks.

A practical target is compute RTF at or below 0.8, leaving headroom for load
spikes. Results from English Nemotron checkpoints or server CPUs must not be
assumed to apply to the multilingual INT4 export on a laptop.

## Known limitations

- The INT4 ONNX export is community-maintained and separate from NVIDIA's full
  NeMo/Transformers checkpoint.
- Polish is a broad-coverage language, not the model's strongest quality tier.
- Accuracy varies significantly by domain, especially for names, punctuation,
  Polish-English code switching, and noisy microphones.
- VAD is disabled by default until it is benchmarked against short dictation and
  push-to-talk behavior.
- Only the pinned 560 ms export is supported.
