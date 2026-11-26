"""Decode one media file once and publish it to the shared rings at wall-clock pace.

``speed=1.0`` replays at capture rate (a live camera/microphone equivalent):
frames that a model cannot keep up with are dropped by that model, never
queued without bound. Video is published as uint8 BGR, audio as 16 kHz mono
int16, both stamped on the same media clock.
"""
from __future__ import annotations

import subprocess
import threading
import time

import numpy as np

from .media_bus import AudioRing, FrameRing


def probe(path: str, max_side: int = 1280) -> dict:
    import cv2
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    scale = min(1.0, max_side / max(width, height))
    out_w, out_h = int(round(width * scale / 2) * 2), int(round(height * scale / 2) * 2)
    has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                     "stream=index", "-of", "csv=p=0", path],
                                    capture_output=True, text=True).stdout.strip())
    return {"fps": fps, "width": out_w, "height": out_h, "frames": frames, "duration_s": frames / fps,
            "has_audio": has_audio}


class MediaSource:
    def __init__(self, path: str, frame_ring: FrameRing | None, audio_ring: AudioRing | None, info: dict,
                 speed: float = 1.0):
        self.path, self.frame_ring, self.audio_ring, self.info, self.speed = path, frame_ring, audio_ring, info, speed
        self.start_wall = None
        self.published_frames = 0
        self.late_frames = 0
        self.done = threading.Event()

    def clock(self) -> float:
        """Media seconds elapsed on the shared clock."""
        if self.start_wall is None:
            return 0.0
        return (time.time() - self.start_wall) * self.speed

    def _pace(self, media_s: float) -> None:
        delay = self.start_wall + media_s / self.speed - time.time()
        if delay > 0:
            time.sleep(delay)

    def _video(self) -> None:
        import cv2
        capture = cv2.VideoCapture(self.path)
        index = 0
        size = (self.info["width"], self.info["height"])
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            pts_s = index / self.info["fps"]
            if time.time() - (self.start_wall + pts_s / self.speed) > 0.05:
                self.late_frames += 1
            self._pace(pts_s)
            self.frame_ring.write(np.ascontiguousarray(frame), int(round(pts_s * 1000)))
            self.published_frames += 1
            index += 1
        capture.release()

    def _audio(self) -> None:
        rate = self.audio_ring.sample_rate
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", self.path, "-f", "s16le", "-ac", "1", "-ar",
                                 str(rate), "-"], stdout=subprocess.PIPE)
        block = rate // 50  # 20 ms
        written = 0
        while True:
            raw = proc.stdout.read(block * 2)
            if not raw:
                break
            pcm = np.frombuffer(raw, np.int16)
            self._pace((written + len(pcm)) / rate)
            self.audio_ring.write(pcm)
            written += len(pcm)
        proc.wait()

    def run(self) -> None:
        self.start_wall = time.time()
        threads = []
        if self.frame_ring is not None:
            threads.append(threading.Thread(target=self._video, daemon=True))
        if self.audio_ring is not None:
            threads.append(threading.Thread(target=self._audio, daemon=True))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.done.set()
