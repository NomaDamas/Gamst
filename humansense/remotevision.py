"""Optional local/explicitly remote image-model scene descriptions.

Provider descriptions are unverified interpretations, never measured human
identity, inner emotion, diagnosis, or intent. Camera frames exist in memory and
are transmitted only to the deliberately configured endpoint; none are stored.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request

from .plugins import InputFrame, PerceptionPlugin
from .reasoning import ChatCompletionsBackend, MAX_RESPONSE_BYTES, _hypothesis


OBSERVATIONS = {"person_count", "head_orientation_description", "posture_description", "visible_action"}
EXPRESSIONS = {"confusion", "interest", "frustration", "smile"}
SYSTEM_PROMPT = """Describe only visible scene evidence. Never identify people,
infer hidden intentions, diagnose illness, judge deception, or assert inner
emotional truth. Treat expression scores as uncalibrated recognized patterns.
Return ONLY a JSON object with exactly these two list fields:
observations: [{name,value,evidence_description}], where name is person_count,
head_orientation_description, posture_description, or visible_action. person_count
value is a nonnegative integer, other values are concise visual descriptions.
expressions: [{name,model_score}], where name is confusion, interest, frustration,
or smile, and model_score is null or 0..1. Empty lists are appropriate when the
frame does not support a claim. Do not output probability, confidence, calibrated,
identity, medical, demographic, or intent fields. Scene descriptions are uncertain
model interpretations, not verified measurements."""


def parse_description(payload: object) -> dict:
    if not isinstance(payload, dict) or set(payload) != {"observations", "expressions"}:
        raise ValueError("Vision model must return exactly observations and expressions")
    observations, expressions = payload["observations"], payload["expressions"]
    if not isinstance(observations, list) or len(observations) > 16 or not isinstance(expressions, list) or len(expressions) > 4:
        raise ValueError("Vision model lists exceed allowed bounds")
    seen = set()
    for item in observations:
        if not isinstance(item, dict) or set(item) != {"name", "value", "evidence_description"}:
            raise ValueError("Unsupported visual observation fields")
        name, value = item["name"], item["value"]
        if not isinstance(name, str) or name not in OBSERVATIONS:
            raise ValueError("Unsupported visual observation name")
        if name in seen:
            raise ValueError("Duplicate visual observation")
        seen.add(name)
        if name == "person_count":
            if type(value) is not int or not 0 <= value <= 1000:
                raise ValueError("person_count must be an integer in [0, 1000]")
        elif not isinstance(value, str) or len(value) > 500:
            raise ValueError("Visual description must be a string of at most 500 characters")
        else:
            _hypothesis(value)
        evidence = item["evidence_description"]
        if not isinstance(evidence, str) or len(evidence) > 1000:
            raise ValueError("Visual evidence description must be at most 1000 characters")
        _hypothesis(evidence)
    seen.clear()
    for item in expressions:
        if not isinstance(item, dict) or set(item) != {"name", "model_score"}:
            raise ValueError("Unsupported expression fields")
        name, score = item["name"], item["model_score"]
        if not isinstance(name, str) or name not in EXPRESSIONS or name in seen:
            raise ValueError("Unsupported or duplicated expression name")
        seen.add(name)
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Expression model_score must be null or finite in [0, 1]")
    return {"observations": observations, "expressions": expressions,
            "interpretation": "unverified_model_hypotheses", "probability": None,
            "calibrated": False, "identity_assessed": False}


class VisionDescriptionPlugin(PerceptionPlugin):
    """Rate-limited JSON image-model adapter; all outputs have insufficient quality.

    Constructor does not open devices or send data. Configure loopback models by
    default; remote HTTPS requires allow_remote=True separately from capture
    consent. ``model_version`` is provider_unspecified unless known by the caller.
    """
    name = "generic_vision_description"
    modalities = ("video",)

    def __init__(self, endpoint: str, model: str, *, api_key_env: str | None = None,
                 allow_remote: bool = False, timeout: float = 30,
                 max_side: int = 640, interval_ms: int = 1000,
                 model_version: str = "provider_unspecified", opener=None):
        if type(max_side) is not int or not 64 <= max_side <= 2048:
            raise ValueError("max_side must be an integer in [64, 2048]")
        if type(interval_ms) is not int or interval_ms < 100:
            raise ValueError("interval_ms must be an integer of at least 100")
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version must be a nonempty string")
        self.transport = ChatCompletionsBackend(endpoint, model, api_key_env=api_key_env,
            allow_remote=allow_remote, timeout=timeout, opener=opener)
        self.model_id, self.model_version = model, model_version
        self.max_side, self.interval_ms = max_side, interval_ms
        self._last: dict[str, int] = {}
        self._generation = 0
        self._lock = asyncio.Lock()

    def reset(self):
        self._generation += 1
        self._last.clear()

    async def process(self, frame: InputFrame) -> list[dict]:
        if frame.modality != "video":
            raise ValueError("VisionDescriptionPlugin accepts video frames only")
        generation = self._generation
        async with self._lock:
            if generation != self._generation:
                return []
            last = self._last.get(frame.subject_id)
            if last is not None and frame.timestamp_ms - last < self.interval_ms:
                return []
            description = await asyncio.to_thread(self._describe_sync, frame.data)
            if generation != self._generation:
                return []
            # Update only after success so errors do not silently suppress retry.
            self._last[frame.subject_id] = frame.timestamp_ms
            return [{"name": "scene.model_description", "value": description,
                     "modality": "video", "timestamp_ms": frame.timestamp_ms,
                     "subject_id": frame.subject_id, "model_id": self.model_id,
                     "model_version": self.model_version,
                     "quality": {"status": "insufficient", "flags": ["model_interpretation", "not_verified"]}}]

    def _encode_jpeg(self, image) -> str:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('Install humansense-sdk[vision] to encode vision model input') from exc
        shape = getattr(image, "shape", ())
        if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) <= 0:
            raise ValueError("Expected an OpenCV BGR image with three channels")
        height, width = shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        if scale < 1:
            image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            raise ValueError("JPEG encoding failed")
        raw = encoded.tobytes()
        if len(raw) > 2_000_000:
            raise ValueError("Encoded image exceeds 2 MB")
        return base64.b64encode(raw).decode("ascii")

    def _describe_sync(self, image) -> dict:
        encoded = self._encode_jpeg(image)
        transport = self.transport
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if transport.api_key_env:
            key = os.environ.get(transport.api_key_env)
            if not key:
                raise ValueError(f"API key environment variable {transport.api_key_env} is not set")
            headers["Authorization"] = "Bearer " + key
        payload = {"model": self.model_id, "temperature": 0, "max_tokens": 1000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": [{"type": "text", "text": "Describe supported visible evidence in this frame."},
                             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}]}]}
        request = Request(transport.endpoint, data=json.dumps(payload, allow_nan=False).encode(), headers=headers, method="POST")
        response = None
        try:
            response = transport._opener(request, timeout=transport.timeout)
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise RuntimeError(f"Vision provider returned HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Vision provider request failed or timed out") from None
        finally:
            if response is not None:
                response.close()
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Vision provider response exceeds 1 MiB")
        try:
            content = json.loads(raw)["choices"][0]["message"]["content"]
            payload = json.loads(content)
        except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError):
            raise ValueError("Vision provider did not return a JSON completion") from None
        return parse_description(payload)


def vision_from_env() -> VisionDescriptionPlugin | None:
    """Optional shared MCP/dashboard image model; unset URL means no image calls."""
    endpoint = os.environ.get("HUMANSENSE_VISION_URL")
    if not endpoint:
        return None
    model = os.environ.get("HUMANSENSE_VISION_MODEL")
    if not model:
        raise ValueError("HUMANSENSE_VISION_MODEL is required with HUMANSENSE_VISION_URL")
    try:
        interval = int(os.environ.get("HUMANSENSE_VISION_INTERVAL_MS", "1000"))
    except ValueError:
        raise ValueError("HUMANSENSE_VISION_INTERVAL_MS must be an integer") from None
    return VisionDescriptionPlugin(endpoint, model,
        api_key_env=os.environ.get("HUMANSENSE_VISION_KEY_ENV"),
        allow_remote=os.environ.get("HUMANSENSE_VISION_ALLOW_REMOTE", "").lower() == "true",
        interval_ms=interval)
