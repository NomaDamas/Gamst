"""Local MediaPipe body keypoints and geometric, stream-local continuity.

No identity, trained gesture classifier, emotion or face/speaker association is
inferred. Wrist-above-shoulder flags are visibility-gated coordinate comparisons.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
from pathlib import Path
import threading

from ..plugins import PerceptionPlugin, InputFrame, _signal, _version
from ..tracking import create_tracker
from ..visualization import drawing_of


def _image_size(data):
    """Image size of the normalized coordinates, so a consumer can undo a non-square image."""
    shape = getattr(data, "shape", None)
    return {} if shape is None else dict(image_width=int(shape[1]), image_height=int(shape[0]))


def associate(poses, tracker, frame):
    """Give each pose its stream-local ``track_id`` (geometric continuity only); returns the tracks."""
    boxes = [p["bbox"] for p in poses if p["bbox"] is not None]
    tracks = tracker.update(boxes, frame.timestamp_ms, stream_id=frame.subject_id + ":body")
    for pose in poses:
        if pose["bbox"] is None:
            continue
        # Exact matching uses only this frame's boxes. Duplicate boxes
        # abstain rather than falsely assigning the same track twice.
        candidates = [t for t in tracks if t["visible"] and t["bbox"] == pose["bbox"]]
        unique = sum(p["bbox"] == pose["bbox"] for p in poses) == 1
        if unique and len(candidates) == 1:
            track = candidates[0]
            pose["association_quality"] = track["association_quality"]
            pose["association_history_uncertain"] = track["association_history_uncertain"]
            if track["association_quality"] != "uncertain":
                pose["track_id"] = track["track_id"]
        else:
            pose["association_quality"] = "uncertain"
    return tracks


class MediaPipePosePlugin(PerceptionPlugin):
    """MediaPipe body on single images in this process. Records carry boxes, tracks and geometry; joint positions
    stay here for the local screen and leave only as the derived input ``joints``, when a Pipeline asks for it."""
    name = "mediapipe_pose_landmarker"
    modalities = ("video",)
    joints = None

    def __init__(self, model_path: str, max_poses: int = 4, tracker: str = 'iou', joints=None, **tracker_options):
        path = Path(model_path)
        if not path.is_file():
            raise ValueError("Supply an existing local pose_landmarker.task file")
        if type(max_poses) is not int or not 1 <= max_poses <= 8:
            raise ValueError("max_poses must be an integer in [1, 8]")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError("Install gamst-sdk[mediapipe]") from exc
        self.mp, self.joints = mp, joints
        self.max_poses = max_poses
        self.model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE, num_poses=max_poses,
            output_segmentation_masks=False))
        self._native_lock = threading.Lock()
        self._process_lock = asyncio.Lock()
        self._generation = 0
        self._closed = False
        self.tracker = create_tracker(tracker, **tracker_options)
        self.tracker_name = tracker

    @staticmethod
    def _landmarks(raw):
        if len(raw) != 33:
            return None
        points = []
        for p in raw:
            values = {k: float(getattr(p, k, 0) or 0) for k in ("x", "y", "z", "visibility", "presence")}
            if not all(math.isfinite(v) for v in values.values()):
                return None
            for key in ("visibility", "presence"):
                values[key] = max(0., min(1., values[key]))
            points.append(values)
        return points

    @staticmethod
    def _bbox(points):
        visible = [p for p in points if min(p["visibility"], p["presence"]) >= .5]
        if len(visible) < 3:
            return None
        x0 = max(0., min(1., min(p["x"] for p in visible)))
        y0 = max(0., min(1., min(p["y"] for p in visible)))
        x1 = max(0., min(1., max(p["x"] for p in visible)))
        y1 = max(0., min(1., max(p["y"] for p in visible)))
        if x1 <= x0 or y1 <= y0:
            return None
        return dict(x=x0, y=y0, width=x1-x0, height=y1-y0)

    @staticmethod
    def _gestures(points):
        result = {}
        for side, shoulder, wrist in (("left", 11, 15), ("right", 12, 16)):
            pair = (points[shoulder], points[wrist])
            reliable = all(min(p["visibility"], p["presence"]) >= .5 for p in pair)
            result[side + "_wrist_above_shoulder"] = pair[1]["y"] < pair[0]["y"] if reliable else None
        return result

    async def process(self, frame: InputFrame) -> list[dict]:
        generation = self._generation
        async with self._process_lock:
            if generation != self._generation:
                return []
            if self._closed:
                raise RuntimeError("MediaPipePosePlugin is closed")
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                  data=frame.data[:, :, ::-1].copy())

            def detect():
                with self._native_lock:
                    if self._closed or generation != self._generation:
                        return None
                    return self.detector.detect(image)

            try:
                result = await asyncio.to_thread(detect)
            except asyncio.CancelledError:
                self.reset()
                raise
            if result is None or generation != self._generation or self._closed:
                return []
            poses = []
            size = _image_size(frame.data)
            for index, raw in enumerate(result.pose_landmarks[:self.max_poses]):
                points = self._landmarks(raw)
                if points is None:
                    continue
                poses.append(dict(frame_pose_index=index, landmarks=points,
                    bbox=self._bbox(points), geometry=self._gestures(points),
                    track_id=None, association_quality="unavailable",
                    identity_verified=False, speaker_association="unavailable", **size))
            tracks = associate(poses, self.tracker, frame)
            drawing_of(self).put("scene.poses", frame, [pose["landmarks"] for pose in poses])
            return committed(frame, poses, tracks, self.name, f"{_version('mediapipe')}:sha256:{self.model_hash}",
                             getattr(self, 'tracker_name', 'iou'), self.joints)

    def reset(self):
        self._generation += 1
        self.tracker.reset()
        drawing_of(self).clear()

    async def close(self):
        self.reset()
        self._closed = True
        def close_native():
            with self._native_lock:
                self.detector.close()
        await asyncio.to_thread(close_native)


def committed(frame, poses, tracks, name, version, tracker_name, joints=None):
    """What a pose module with private joints returns: records without joint positions, and the joints on
    the derived input ``joints`` (when the recipe asked for them)."""
    from ..pipeline import Derived
    from ..private_payload import PrivatePoses, public_pose
    output = [_signal(frame, "scene.poses", [dict(public_pose(pose), landmark_count=len(pose["landmarks"])) for pose in poses], name, version),
              _signal(frame, "scene.pose_count", len(poses), name, version),
              _signal(frame, "scene.body_tracks", tracks, f"{name}:{tracker_name}-tracker", "0.1")]
    if joints is not None:
        output.append(Derived(joints, PrivatePoses(poses), "video"))
    return output


class PoseWorkerPlugin(PerceptionPlugin):
    """MediaPipe body in VIDEO mode in its own process, owned by this module.

    Records carry boxes, tracks and geometry. Joint positions are the worker's private payload; they
    leave this module only as the derived input ``joints``, which the Pipeline never commits.
    """
    name = "mediapipe_pose_landmarker"
    modalities = ("video",)
    owns_process = True

    def __init__(self, worker, *, joints=None, tracker='iou', opener=None, **tracker_options):
        self.worker, self.joints, self.version = worker, joints, f"mediapipe:{worker.spec.version}"
        self.tracker = create_tracker(tracker, **tracker_options)
        self.tracker_name, self._opener, self._tracker_options = tracker, opener, tracker_options
        if opener is None:
            self.fork = None

    async def fork(self):
        """Another worker of the same model with a fresh tracker (an L2 chain of the Pipeline)."""
        return type(self)(await self._opener(), joints=self.joints, tracker=self.tracker_name, opener=self._opener,
                          **self._tracker_options)

    async def process(self, frame: InputFrame) -> list[dict]:
        from dataclasses import replace
        from ..crew.modules import bounded_image
        from ..private_payload import CONTRACTS, landmarks
        value, private = await self.worker.query(replace(frame, data=bounded_image(frame.data)))
        size = dict(image_width=value["summary"]["image_width"], image_height=value["summary"]["image_height"])
        columns = CONTRACTS["mediapipe.body.video"].columns
        poses = [dict(frame_pose_index=pose["frame_pose_index"], landmarks=landmarks(rows, columns), bbox=pose["bbox"],
                      geometry=pose["geometry"], track_id=None, association_quality="unavailable",
                      identity_verified=False, speaker_association="unavailable", **size)
                 for pose, rows in zip(value["poses"], private.take("joints"))]
        tracks = associate(poses, self.tracker, frame)
        return committed(frame, poses, tracks, self.name, self.version, self.tracker_name, self.joints)

    async def reset(self):
        self.tracker.reset()
        await self.worker.reset()

    async def close(self):
        await self.worker.close()
