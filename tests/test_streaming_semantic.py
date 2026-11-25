import numpy as np
import pytest
from gamst import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.streaming_semantic import StreamingSemanticContext


class Backend:
    model_id = "StreamingVLM"
    model_version = "pinned-test"

    def __init__(self):
        self.calls = []

    def predict(self, frames, timestamps_ms, previous_state):
        self.calls.append((len(frames), list(timestamps_ms), previous_state))
        return {"state": {"phase": "moving"}, "events": [{"type": "motion"}],
                "summary": "object moving", "evidence_ids": [f"frame-{x}" for x in timestamps_ms]}

    def close(self):
        pass


def frame(index, timestamp):
    sensor = SensorMetadata("camera", "video", index, coordinate_frame="pixels")
    return InputFrame(np.zeros((32, 48, 3), dtype=np.uint8), "video", timestamp, sensor=sensor)


def test_streaming_provider_keeps_previous_state_and_bounded_windows():
    backend = Backend()
    provider = StreamingSemanticContext(backend, window_frames=2)
    assert provider.process(frame(0, 1000)) is None
    first = provider.process(frame(1, 1500))
    assert first["state"]["phase"] == "moving"
    assert backend.calls[0][2] is None
    assert provider.process(frame(2, 2000)) is None
    second = provider.process(frame(3, 2500))
    assert backend.calls[1][2] == {"phase": "moving"}
    assert second["window"]["frames"] == 2


def test_streaming_provider_rejects_nonchronological_frames():
    provider = StreamingSemanticContext(Backend(), window_frames=2)
    provider.process(frame(0, 1000))
    with pytest.raises(ValueError, match="strictly chronological"):
        provider.process(frame(1, 999))
