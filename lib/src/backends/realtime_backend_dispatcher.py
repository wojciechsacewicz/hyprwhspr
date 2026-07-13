"""Choose the cloud or local implementation of the realtime backend."""

from .nemotron_streaming_backend import NemotronStreamingBackend
from .realtime_ws_backend import RealtimeWsBackend


LOCAL_NEMOTRON_PROVIDER = "nemotron-local"


def realtime_backend_class(config):
    """Return the implementation selected by the realtime provider setting."""
    provider = config.get_setting("websocket_provider", None)
    if provider == LOCAL_NEMOTRON_PROVIDER:
        return NemotronStreamingBackend
    return RealtimeWsBackend


class RealtimeBackendDispatcher:
    """Factory registered under ``realtime-ws`` in the backend registry."""

    name = "realtime-ws"

    def __new__(cls, manager):
        implementation = realtime_backend_class(manager.config)
        return implementation(manager)
