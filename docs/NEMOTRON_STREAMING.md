# Local Nemotron 3.5 streaming

This optional provider adds fully local, cache-aware streaming transcription to
hyprwhspr with NVIDIA Nemotron 3.5 ASR Streaming 0.6B and ONNX Runtime GenAI.
It is opt-in and does not change the default backend.

## Architecture

The implementation deliberately reuses hyprwhspr's existing `realtime-ws`
lifecycle. `RealtimeBackendDispatcher` selects the normal cloud WebSocket
backend for every existing provider and selects the local backend only when:

```json
{
  "transcription_backend": "realtime-ws",
  "websocket_provider": "nemotron-local"
}
```

This keeps recording modes, OSD partial previews, capture mode, cancellation,
microphone recovery, and final text injection on the same tested path as the
existing realtime integrations.

The PortAudio callback never runs inference. It copies audio into a bounded
queue and returns. A worker thread performs stateful SoXR resampling, forms the
model's exact 8,960-sample chunks (560 ms at 16 kHz), advances the ONNX cache,
and publishes partial transcripts.

Queue overflow is an explicit failed utterance. Audio is never silently
discarded.

## Requirements

1. Install hyprwhspr normally first so its private venv exists.
2. CPU: recent x86-64 Linux and roughly 1 GB free RAM for the quantized model.
3. CUDA is optional and uses the separate `onnxruntime-genai-cuda` wheel.

The model is pinned to:

- repository: `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`
- revision: `8364d9e2dd9da23789b480bdbba9e423717e42ee`
- runtime: `onnxruntime-genai==0.14.0`

## Setup

From the repository checkout:

```bash
bash ./bin/hyprwhspr-nemotron setup --device cpu --language pl-PL
```

For NVIDIA CUDA:

```bash
bash ./bin/hyprwhspr-nemotron setup --device cuda --language pl-PL
```

The command installs optional packages into hyprwhspr's existing venv,
downloads the pinned snapshot, backs up `config.json`, writes the provider
configuration, and restarts the user service when available.

Useful commands:

```bash
bash ./bin/hyprwhspr-nemotron status
bash ./bin/hyprwhspr-nemotron validate
bash ./bin/hyprwhspr-nemotron download
bash ./bin/hyprwhspr-nemotron unload
bash ./bin/hyprwhspr-nemotron reload
bash ./bin/hyprwhspr-nemotron restart
bash ./bin/hyprwhspr-nemotron benchmark ./sample.wav --language pl-PL
```

`unload` stops `hyprwhspr.service`, which releases the local model. `reload`
starts it again. The regular `hyprwhspr model unload` command treats every
`realtime-ws` provider as remote, so use the dedicated commands for this
provider.

## Configuration

```json
{
  "transcription_backend": "realtime-ws",
  "websocket_provider": "nemotron-local",
  "websocket_model": "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4",
  "realtime_mode": "transcribe",
  "language": "pl-PL",
  "nemotron_streaming_model": "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4",
  "nemotron_streaming_revision": "8364d9e2dd9da23789b480bdbba9e423717e42ee",
  "nemotron_streaming_device": "cpu",
  "nemotron_streaming_buffer_max_seconds": 3.0,
  "nemotron_streaming_finalize_timeout": 5.0,
  "nemotron_streaming_use_vad": false
}
```

Use an explicit locale such as `pl-PL` where possible. `null` uses model-side
automatic language detection, which is less accurate for Polish.

Unsupported settings are not emulated: Whisper prompts, beam search, and
speech translation do not affect Nemotron.

## Recording-mode behavior

- `toggle`, `push_to_talk`, and `auto`: partials appear in OSD; one final text is
  injected after stop.
- `continuous`: the backend swaps to a fresh cache before finalizing each
  utterance so capture stays open without mixing adjacent segments.
- `long_form` and file-based tests: `transcribe()` replays the supplied audio
  through the same streaming runtime.
- cancel, mute cancellation, and recording-start errors clear the active
  session but keep the model loaded for the next utterance.

Partials are preview-only. The provider does not edit unstable partial text in
the focused application.

## Tests

Fast tests do not download model weights:

```bash
python -m unittest tests.test_nemotron_streaming_backend
python -m unittest tests.test_realtime_backend_dispatcher
python -m unittest tests.test_nemotron_commands
```

The real-runtime smoke test is opt-in:

```bash
HYPRWHSPR_RUN_MODEL_TESTS=1 \
  python -m unittest tests.test_nemotron_model_integration
```

Before an upstream PR, benchmark the intended hardware and language. A useful
minimum gate for realtime use is RTF <= 0.8, no queue growth during a ten-minute
session, zero dropped chunks, and finalization below one second for normal
utterances.

## Known limitations

- The ONNX export is community-maintained and newer than the official NVIDIA
  checkpoint.
- Polish accuracy may be lower than cloud Scribe-style services, especially for
  names, punctuation, and Polish-English code switching.
- The checked-in smoke test validates runtime compatibility but is skipped in
  normal CI because it downloads a large model.
- Only the pinned 560 ms export is supported. Other chunk sizes require matching
  model exports and are intentionally not exposed as fake configurability.
