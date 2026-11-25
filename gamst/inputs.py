"""Input modules: the front layer that turns devices and recordings into capture-stamped frames.

Each module has ``frames()``, an async iterator of :class:`gamst.plugins.InputFrame` whose
``timestamp_ms`` is the capture time, and ``close()``. A :class:`gamst.pipeline.Pipeline` pumps them;
hosts that own capture themselves keep using ``Gamst.push_video`` / ``push_audio`` or
``Pipeline.push``. Third-party inputs plug in the same way (any object with ``frames()``).

Recorded inputs are re-timed onto the wall clock so the newest frame is not in the future (the
runtime rejects capture times more than 5 s ahead), keeping the recording's own spacing.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time

from .plugins import InputFrame


def _now_ms():
    return time.time_ns() // 1_000_000


class CameraInput:
    """A local camera (OpenCV). Capture time is the host's monotonic clock at read."""
    modality = 'video'

    def __init__(self, index=0, *, max_frames=None):
        self.index, self.max_frames = index, max_frames
        self._cap = None

    async def frames(self):
        import cv2
        backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f'cannot open camera {self.index}')
        base, start, last, count = _now_ms(), time.monotonic(), None, 0
        while self.max_frames is None or count < self.max_frames:
            ok, frame = await asyncio.to_thread(self._cap.read)
            if not ok:
                break
            stamp = base + round(1000 * (time.monotonic() - start))
            stamp = stamp if last is None or stamp > last else last + 1
            last, count = stamp, count + 1
            yield InputFrame(frame, 'video', stamp)

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFileInput:
    """A video file; ``capture_fps`` overrides file timing (high-speed clips saved at 30 fps)."""
    modality = 'video'

    def __init__(self, path, *, capture_fps=None, max_frames=None):
        self.path, self.capture_fps, self.max_frames = str(path), capture_fps, max_frames
        self._cap = None

    def fps(self):
        import cv2
        cap = cv2.VideoCapture(self.path)
        try:
            return self.capture_fps or cap.get(cv2.CAP_PROP_FPS) or None
        finally:
            cap.release()

    async def frames(self):
        import cv2
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f'cannot open video {self.path}')
        fps = self.capture_fps or self._cap.get(cv2.CAP_PROP_FPS) or 30.
        total = self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        base = _now_ms() - round(1000 * total / fps) - 1000
        sequence, last = 0, None
        while self.max_frames is None or sequence < self.max_frames:
            ok, frame = self._cap.read()
            if not ok:
                break
            offset = sequence * 1000 / self.capture_fps if self.capture_fps else self._cap.get(cv2.CAP_PROP_POS_MSEC)
            stamp = base + round(offset)
            stamp = stamp if last is None or stamp > last else last + 1
            last, sequence = stamp, sequence + 1
            yield InputFrame(frame, 'video', stamp)

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class TakeInput:
    """A GAMST 240 take (folder or ``raw.gmst``); capture times are the phone's sensor times."""
    modality = 'video'

    def __init__(self, path, *, max_frames=None):
        path = Path(path)
        self.path = path / 'raw.gmst' if path.is_dir() else path
        if not self.path.is_file():
            raise FileNotFoundError(f'no GAMST 240 recording at {self.path}')
        self.max_frames = max_frames

    def fps(self):
        summary = self.path.parent / 'summary.json'
        if summary.is_file():
            import json
            fps = json.loads(summary.read_text()).get('capture_fps')
            if fps:
                return float(fps)
        from .demos.rotation_live import read_recording
        frames = read_recording(self.path)
        span = frames[-1][0].seconds - frames[0][0].seconds if len(frames) > 1 else 0
        return (len(frames) - 1) / span if span else None

    async def frames(self):
        from .demos.rotation_live import decode, read_recording
        recording = read_recording(self.path)
        if not recording:
            return
        first, end = recording[0][0].seconds, recording[-1][0].seconds
        base = _now_ms() - round(1000 * (end - first)) - 1000
        last, count = None, 0
        for info, _, payload in recording:
            if self.max_frames is not None and count >= self.max_frames:
                break
            frame = decode(info, payload)
            if frame is None:
                continue
            stamp = base + round(1000 * (info.seconds - first))
            stamp = stamp if last is None or stamp > last else last + 1
            last, count = stamp, count + 1
            yield InputFrame(frame, 'video', stamp)
            await asyncio.sleep(0)


class AudioFileInput:
    """A PCM WAV file as mono float32 chunks at its own sample rate (16 kHz expected by speech modules)."""
    modality = 'audio'

    def __init__(self, path, *, chunk_ms=100):
        self.path, self.chunk_ms = str(path), int(chunk_ms)

    async def frames(self):
        import wave
        import numpy as np
        with wave.open(self.path, 'rb') as source:
            rate, width, channels = source.getframerate(), source.getsampwidth(), source.getnchannels()
            total = source.getnframes()
            if width not in (2, 4):
                raise ValueError('16- or 32-bit PCM WAV required')
            chunk = max(1, rate * self.chunk_ms // 1000)
            base = _now_ms() - round(1000 * total / rate) - 1000
            done = 0
            while True:
                raw = source.readframes(chunk)
                if not raw:
                    break
                samples = np.frombuffer(raw, np.int16 if width == 2 else np.int32).astype(np.float32)
                samples /= 32768. if width == 2 else 2147483648.
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                done += len(samples)
                # The frame's time is the capture time of its last sample, like push_audio.
                yield InputFrame(samples, 'audio', base + round(1000 * done / rate), sample_rate=rate)
                await asyncio.sleep(0)


class MicrophoneInput:
    """The default microphone as mono float32 chunks (needs ``sounddevice``)."""
    modality = 'audio'

    def __init__(self, *, sample_rate=16000, chunk_ms=100, device=None, max_chunks=None, stream_factory=None):
        self.sample_rate, self.chunk_ms, self.device = sample_rate, chunk_ms, device
        self.max_chunks, self._factory, self._stream = max_chunks, stream_factory, None

    async def frames(self):
        if self._factory is None:
            try:
                import sounddevice
            except ImportError as exc:
                raise RuntimeError('MicrophoneInput needs: pip install sounddevice') from exc
            factory = lambda **options: sounddevice.InputStream(**options)
        else:
            factory = self._factory
        chunk = self.sample_rate * self.chunk_ms // 1000
        self._stream = factory(samplerate=self.sample_rate, channels=1, dtype='float32',
                               blocksize=chunk, device=self.device)
        self._stream.start()
        count = 0
        try:
            while self.max_chunks is None or count < self.max_chunks:
                samples, _ = await asyncio.to_thread(self._stream.read, chunk)
                count += 1
                yield InputFrame(samples.reshape(-1).copy(), 'audio', _now_ms(), sample_rate=self.sample_rate)
        finally:
            self.close()

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def open_source(kind, value=None, **options):
    """The input module for a recipe source name."""
    if kind == 'camera':
        return CameraInput(int(value or 0), **options)
    if kind == 'video':
        return VideoFileInput(value, **options)
    if kind == 'take':
        return TakeInput(value, **options)
    if kind == 'audio':
        return AudioFileInput(value, **options)
    if kind == 'microphone':
        return MicrophoneInput(**options)
    raise ValueError(f'unknown input source {kind!r}')
