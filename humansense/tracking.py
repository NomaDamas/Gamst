"""Deterministic bounding-box continuity within a camera stream.

Tracks are temporary geometric associations. They do not identify a person,
link audio speakers, or survive expiry as a biometric identity would.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math


def normalize_bbox(box) -> dict[str, float]:
    """Return normalized ``x,y,width,height`` without guessing pixel units.

    Accept mappings with x/y/width/height or x1/y1/x2/y2, and four-item
    sequences in x/y/width/height order. Coordinates must lie within [0, 1].
    Rejecting malformed detections avoids silently associating invalid boxes.
    """
    if isinstance(box, Mapping):
        if all(key in box for key in ("x", "y", "width", "height")):
            raw = [box[key] for key in ("x", "y", "width", "height")]
        elif all(key in box for key in ("x1", "y1", "x2", "y2")):
            try:
                if any(isinstance(box[key], bool) for key in ("x1", "y1", "x2", "y2")):
                    raise ValueError("boolean coordinates are invalid")
                raw = [box["x1"], box["y1"], float(box["x2"]) - float(box["x1"]),
                       float(box["y2"]) - float(box["y1"])]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("bbox coordinates must be finite numbers") from exc
        else:
            raise ValueError("bbox mapping needs x/y/width/height or x1/y1/x2/y2")
    elif isinstance(box, Sequence) and not isinstance(box, (str, bytes, bytearray)) and len(box) == 4:
        raw = list(box)
    else:
        raise ValueError("bbox must be a mapping or a four-item xywh sequence")
    try:
        if any(isinstance(value, bool) for value in raw):
            raise ValueError("boolean coordinates are invalid")
        x, y, width, height = [float(value) for value in raw]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bbox coordinates must be finite numbers") from exc
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("bbox coordinates must be finite numbers")
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
        raise ValueError("bbox must have positive area inside normalized [0, 1] coordinates")
    return {"x": x, "y": y, "width": width, "height": height}


def intersection_over_union(left, right) -> float:
    """IoU of normalized xywh boxes."""
    a, b = normalize_bbox(left), normalize_bbox(right)
    intersection = max(0., min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])) * max(
        0., min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union else 0.


class PersonTracker:
    """Greedy IoU association with expiry and explicit ambiguity flags.

    ``update`` returns visible and recently missing tracks, each containing a
    normalized bbox, stream-local ``track_id``, first/last observation times,
    ``visible`` and ``association_quality``. A near tie for either detection or
    previous track marks all involved tracks ``uncertain``. Deterministic tie
    breaking does not make that association a verified human identity.

    Missing detections retain their last observed box, never an extrapolated
    location. Older timestamps raise ValueError before changing any state.
    """

    def __init__(self, iou_threshold: float = .2, max_age_ms: int = 1500,
                 ambiguity_margin: float = .1):
        if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be in (0, 1]")
        if isinstance(max_age_ms, bool) or not isinstance(max_age_ms, int) or max_age_ms < 0:
            raise ValueError("max_age_ms must be a nonnegative integer")
        if not math.isfinite(ambiguity_margin) or not 0 <= ambiguity_margin <= 1:
            raise ValueError("ambiguity_margin must be in [0, 1]")
        self.iou_threshold = iou_threshold
        self.max_age_ms = max_age_ms
        self.ambiguity_margin = ambiguity_margin
        self._streams: dict[str, dict] = {}
        self._counters: dict[str, int] = {}

    def update(self, boxes, timestamp_ms: int, stream_id: str = "camera-0") -> list[dict]:
        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ValueError("stream_id must be non-empty")
        if isinstance(boxes, (str, bytes, Mapping)):
            raise ValueError("boxes must be an iterable of detections")
        normalized = [normalize_bbox(box) for box in boxes]
        previous = self._streams.get(stream_id)
        if previous is not None and timestamp_ms < previous["timestamp_ms"]:
            raise ValueError("out-of-order timestamp; tracker state is unchanged")
        # Work on a copy: validation and callers' edits cannot partially mutate
        # stream state or alter previously returned observations.
        state = deepcopy(previous) if previous is not None else {
            "timestamp_ms": timestamp_ms, "counter": self._counters.get(stream_id, 0), "tracks": {}}
        tracks = {key: value for key, value in state["tracks"].items()
                  if timestamp_ms - value["last_seen_ms"] <= self.max_age_ms}
        candidates = []
        by_track: dict[str, list] = {}
        by_detection: dict[int, list] = {}
        for track_id, track in tracks.items():
            for index, box in enumerate(normalized):
                iou = intersection_over_union(track["bbox"], box)
                if iou >= self.iou_threshold:
                    candidate = (iou, track_id, index)
                    candidates.append(candidate)
                    by_track.setdefault(track_id, []).append(candidate)
                    by_detection.setdefault(index, []).append(candidate)
        uncertain_tracks, uncertain_detections = set(), set()
        for alternatives in [*by_track.values(), *by_detection.values()]:
            alternatives = sorted(alternatives, key=lambda item: (-item[0], item[1], item[2]))
            if len(alternatives) >= 2 and alternatives[0][0] - alternatives[1][0] <= self.ambiguity_margin:
                for iou, track_id, index in alternatives:
                    if alternatives[0][0] - iou <= self.ambiguity_margin:
                        uncertain_tracks.add(track_id)
                        uncertain_detections.add(index)
        matched_tracks, matched_detections = set(), set()
        for track in tracks.values():
            track.update(visible=False, timestamp_ms=timestamp_ms, association_quality="missing",
                         association_reason="no_matching_detection", association_iou=None)
        for iou, track_id, index in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
            if track_id in matched_tracks or index in matched_detections:
                continue
            ambiguous = track_id in uncertain_tracks or index in uncertain_detections
            tracks[track_id].update(bbox=normalized[index], visible=True, last_seen_ms=timestamp_ms,
                association_quality="uncertain" if ambiguous else "matched",
                association_reason="ambiguous_geometry" if ambiguous else "unique_geometric_match",
                association_iou=iou,
                association_history_uncertain=tracks[track_id].get("association_history_uncertain", False) or ambiguous)
            matched_tracks.add(track_id)
            matched_detections.add(index)
        for index, box in enumerate(normalized):
            if index in matched_detections:
                continue
            state["counter"] += 1
            track_id = f"{stream_id}:person-{state['counter']}"
            ambiguous = index in uncertain_detections or index in by_detection
            tracks[track_id] = {"track_id": track_id, "stream_id": stream_id, "bbox": box,
                "timestamp_ms": timestamp_ms, "first_seen_ms": timestamp_ms, "last_seen_ms": timestamp_ms,
                "visible": True, "identity_verified": False, "speaker_association": "unavailable",
                "association_quality": "uncertain" if ambiguous else "new",
                "association_reason": "ambiguous_geometry" if ambiguous else "new_detection",
                "association_iou": None, "association_history_uncertain": ambiguous}
        for track_id in uncertain_tracks - matched_tracks:
            tracks[track_id].update(association_quality="uncertain", association_reason="ambiguous_geometry",
                                    association_history_uncertain=True)
        state.update(timestamp_ms=timestamp_ms, tracks=tracks)
        self._streams[stream_id] = state
        self._counters[stream_id] = state["counter"]
        return deepcopy(sorted(tracks.values(), key=lambda track: track["track_id"]))

    def reset(self, stream_id: str | None = None):
        """Forget geometric history, retaining only non-personal ID counters.

        IDs are not recycled within this tracker instance, including after a
        pause/reset, so stored events cannot mistake a new person for an old
        geometric track. Create a new tracker for an entirely new session.
        """
        if stream_id is None:
            self._streams.clear()
        else:
            self._streams.pop(stream_id, None)
