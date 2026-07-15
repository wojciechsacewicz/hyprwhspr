"""Backend utilities and constants for hyprwhspr."""

import re

_HARDWARE_DEVICE_TYPES = (
    "INTEGRATED_GPU",
    "DISCRETE_GPU",
    "VIRTUAL_GPU",
)
_DEVICE_TYPE_LINE_RE = re.compile(
    r"deviceType\s*=\s*PHYSICAL_DEVICE_TYPE_(\w+)"
)


def vulkaninfo_has_hardware_gpu(summary: str) -> bool:
    """Return True when vulkaninfo lists at least one non-software GPU."""
    for match in _DEVICE_TYPE_LINE_RE.finditer(summary):
        if match.group(1) in _HARDWARE_DEVICE_TYPES:
            return True
    return False


def normalize_backend(backend: str) -> str:
    """Normalize legacy backend names."""
    if backend == "local":
        return "pywhispercpp"
    if backend == "remote":
        return "rest-api"
    if backend == "amd":
        return "vulkan"
    return backend


BACKEND_DISPLAY_NAMES = {
    "pywhispercpp": "Local (pywhispercpp)",
    "onnx-asr": "Parakeet TDT V3 (onnx-asr, CPU/GPU)",
    "nemotron-streaming": "Nemotron 3.5 Streaming (local ONNX)",
    "cohere-transcribe": "Cohere Transcribe 2B (transformers, CPU/GPU)",
    "rest-api": "REST API",
    "realtime-ws": "Realtime WebSocket",
    "cpu": "Whisper CPU (pywhispercpp)",
    "nvidia": "Whisper NVIDIA (CUDA)",
    "amd": "Whisper AMD/Intel (Vulkan)",
    "vulkan": "Whisper AMD/Intel (Vulkan)",
    "faster-whisper": "faster-whisper (CTranslate2, CPU/CUDA)",
}
