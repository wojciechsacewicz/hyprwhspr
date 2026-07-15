"""Transcription backend registry."""

from .base import TranscriptionBackend
from .cohere_backend import CohereBackend
from .faster_whisper_backend import FasterWhisperBackend
from .nemotron_streaming_backend import NemotronStreamingBackend
from .onnx_asr_backend import OnnxAsrBackend
from .pywhispercpp_backend import PywhispercppBackend
from .realtime_ws_backend import RealtimeWsBackend
from .rest_api_backend import RestApiBackend

BACKENDS = {
    CohereBackend.name: CohereBackend,
    FasterWhisperBackend.name: FasterWhisperBackend,
    NemotronStreamingBackend.name: NemotronStreamingBackend,
    OnnxAsrBackend.name: OnnxAsrBackend,
    PywhispercppBackend.name: PywhispercppBackend,
    RealtimeWsBackend.name: RealtimeWsBackend,
    RestApiBackend.name: RestApiBackend,
}

__all__ = [
    "BACKENDS",
    "TranscriptionBackend",
    "CohereBackend",
    "FasterWhisperBackend",
    "NemotronStreamingBackend",
    "OnnxAsrBackend",
    "PywhispercppBackend",
    "RealtimeWsBackend",
    "RestApiBackend",
]
