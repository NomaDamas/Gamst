"""Stateful video-only streaming semantic provider contract.

The backend is explicit and model-owned: GAMST supplies ordered bounded frames
and the previous semantic state; it never downloads or selects a model.
"""
from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import json
import os
from typing import Any, Protocol
from urllib.parse import urlparse
import urllib.request

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json
from ..plugins import InputFrame


class StreamingSemanticBackend(Protocol):
    model_id: str
    model_version: str

    def predict(self, frames: list[Any], timestamps_ms: list[int], previous_state: dict[str, Any] | None) -> dict[str, Any]: ...
    def close(self) -> None: ...


class OpenAIStreamingBackend:
    """Explicit local OpenAI-compatible bridge for a pinned StreamingVLM server.

    The server must host the pinned StreamingVLM checkpoint; GAMST does not
    claim that a generic chat endpoint implements KV-cache streaming semantics.
    """

    def __init__(self, endpoint: str, model: str, revision: str, api_key_env: str | None = None,
                 request_options: dict | None = None, max_tokens: int = 256):
        parsed = urlparse(endpoint)
        if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("streaming VLM endpoint must be local")
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += "/chat/completions"
        self.model_id, self.model_version = model, revision
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        # Server-specific request fields, e.g. {"chat_template_kwargs": {"enable_thinking": false},
        # "response_format": {"type": "json_object"}} for a Qwen3.5 vLLM server.
        options = dict(request_options or {})
        if set(options) & {"model", "messages"}:
            raise ValueError("request_options cannot override model or messages")
        if type(max_tokens) is not int or not 16 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be 16..4096")
        self.request_options, self.max_tokens = options, max_tokens

    def predict(self, frames, timestamps_ms, previous_state):
        import cv2
        content = [{"type": "text", "text": json.dumps({
            "timestamps_ms": timestamps_ms, "previous_state": previous_state,
            "instruction": "Update the video-only semantic state. Return JSON with state, events, summary, evidence_ids. Do not infer audio or hidden causes.",
        }, ensure_ascii=False)}]
        for frame in frames:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError("streaming VLM JPEG encoding failed")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")}})
        body = json.dumps({"model": self.model_id, "messages":[{"role":"user","content":content}],
                           "temperature":0, "max_tokens":self.max_tokens, **self.request_options}).encode()
        headers={"Content-Type":"application/json"}
        if self.api_key: headers["Authorization"]="Bearer "+self.api_key
        request=urllib.request.Request(self.endpoint, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            payload=json.loads(response.read(2*1024*1024+1))
        text=payload["choices"][0]["message"]["content"]
        value=json.loads(text)
        value.setdefault("evidence_ids", [f"video:{ts}" for ts in timestamps_ms])
        return value

    def close(self):
        return None


class StreamingSemanticBackend(Protocol):
    model_id: str
    model_version: str

    def predict(self, frames: list[Any], timestamps_ms: list[int], previous_state: dict[str, Any] | None) -> dict[str, Any]: ...
    def close(self) -> None: ...


class StreamingSemanticContext:
    """Video-only stateful window adapter for a pinned streaming VLM backend."""

    def __init__(self, backend: StreamingSemanticBackend, *, window_frames: int = 16,
                 max_gap_ms: int = 2000):
        if type(window_frames) is not int or not 1 <= window_frames <= 64:
            raise ValueError("window_frames must be 1..64")
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 10000:
            raise ValueError("max_gap_ms must be 1..10000")
        if not callable(getattr(backend, "predict", None)) or not callable(getattr(backend, "close", None)):
            raise ValueError("streaming backend requires predict and close")
        self.backend = backend
        self.window_frames = window_frames
        self.max_gap_ms = max_gap_ms
        self.closed = False
        self.frames: list[Any] = []
        self.timestamps: list[int] = []
        self.previous_state: dict[str, Any] | None = None
        self.segment_id = None

    @staticmethod
    def capability(**config):
        return CapabilitySpec(
            "vision.streaming_semantic", "vision.semantic_state", "video",
            "streamingvlm.local", "1", (
                OutputField("state", "object"),
                OutputField("events", "array"),
                OutputField("summary", "string"),
                OutputField("evidence_ids", "array"),
                OutputField("window", "object"),
                OutputField("model", "object"),
            ), "window", window_ms=10000, freshness_ms=2000,
            stateful=True, interpretation="semantic_description",
            devices=(config.get("device", "cuda"),),
            sensor_requirements=("uint8-bgr-camera", "capture-time-source-metadata"),
            license="adapter:Apache-2.0;model:verify-separately")

    def reset(self):
        self.frames.clear()
        self.timestamps.clear()
        self.previous_state = None
        self.segment_id = None

    def close(self):
        if not self.closed:
            self.closed = True
            self.reset()
            self.backend.close()

    def process(self, frame: InputFrame):
        if self.closed:
            raise RuntimeError("streaming semantic provider closed")
        if frame.modality != "video" or frame.sample_rate is not None or not isinstance(frame.sensor, SensorMetadata):
            raise ValueError("video frame with source metadata required")
        if type(frame.timestamp_ms) is not int:
            raise ValueError("capture timestamp required")
        if self.timestamps and frame.timestamp_ms <= self.timestamps[-1]:
            raise ValueError("streaming frames must be strictly chronological")
        if self.timestamps and frame.timestamp_ms - self.timestamps[-1] > self.max_gap_ms:
            self.reset()
        self.frames.append(frame.data.copy() if hasattr(frame.data, "copy") else frame.data)
        self.timestamps.append(frame.timestamp_ms)
        if len(self.frames) < self.window_frames:
            return None
        result = bounded_json(self.backend.predict(self.frames, self.timestamps, self.previous_state))
        if not isinstance(result, dict) or set(result) != {"state", "events", "summary", "evidence_ids"}:
            raise ValueError("streaming backend must return state/events/summary/evidence_ids")
        if not isinstance(result["state"], dict) or not isinstance(result["events"], list):
            raise ValueError("invalid streaming semantic result")
        self.previous_state = deepcopy(result["state"])
        evidence = list(result["evidence_ids"])[-self.window_frames:]
        value = {
            "state": result["state"], "events": result["events"],
            "summary": str(result["summary"])[:8192], "evidence_ids": evidence,
            "window": {"start_ms": self.timestamps[0], "end_ms": self.timestamps[-1],
                       "frames": len(self.frames), "input_sha256": hashlib.sha256(
                           json.dumps(self.timestamps).encode()).hexdigest()},
            "model": {"id": getattr(self.backend, "model_id", "streaming-vlm"),
                      "version": getattr(self.backend, "model_version", "unverified")},
        }
        self.frames.clear()
        self.timestamps.clear()
        return value
