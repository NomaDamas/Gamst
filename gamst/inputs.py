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
import collections
from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import socket
import struct
import sys
import threading
import time
import typing

from .capabilities import SensorMetadata
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


# ---------------------------------------------------------------- GAMST 240 (iPhone high-fps app) wire format

# "GMS3" | u32 sequence | f64 capture seconds | u16 width | u16 height | u8 flags | u8 zoom x10 |
# u16 clockwise degrees to upright | u16 x4 crop rect (sensor-normalized x65535) | u32 jpeg bytes.
# Frames arrive in sensor orientation; the Mac rotates them upright. "GMS2" frames are already upright
# and uncropped; "GMST" (first app build) has no flags and counts as one continuous take.
# The Mac drives the phone with newline-delimited JSON commands written back on the same socket.
HEADER_SIZE = {b'GMS3': 36, b'GMS2': 28, b'GMST': 24}
RECORDING, FRONT_CAMERA = 1, 2
FULL = (0., 0., 1., 1.)


class FrameHeader(typing.NamedTuple):
    sequence: int
    seconds: float
    width: int
    height: int
    flags: int
    zoom: float
    size: int
    angle: int = 0
    crop: tuple = FULL


def parse_header(header):
    magic = header[:4]
    if magic == b'GMS3':
        sequence, seconds, width, height, flags, zoom, angle, cx, cy, cw, ch, size = struct.unpack(
            '<IdHHBBH4HI', header[4:36])
        return FrameHeader(sequence, seconds, width, height, flags, (zoom or 10) / 10, size, angle % 360,
                           (cx / 65535, cy / 65535, cw / 65535, ch / 65535))
    if magic == b'GMS2':
        sequence, seconds, width, height, flags, zoom, size = struct.unpack('<IdHHBB2xI', header[4:28])
        return FrameHeader(sequence, seconds, width, height, flags, (zoom or 10) / 10, size)
    if magic == b'GMST':
        sequence, seconds, width, height, size = struct.unpack('<IdHHI', header[4:24])
        return FrameHeader(sequence, seconds, width, height, RECORDING, 1., size)
    raise ValueError('not a GAMST 240 frame stream')


def read_recording(path):
    """[(FrameHeader, header bytes, jpeg bytes)] of a raw.gmst stream."""
    data = Path(path).read_bytes()
    frames, offset = [], 0
    while offset + 4 <= len(data):
        size = HEADER_SIZE.get(data[offset:offset + 4])
        if size is None:
            raise SystemExit(f'{path}: not a GAMST 240 recording')
        info = parse_header(data[offset:offset + size])
        offset += size
        frames.append((info, data[offset - size:offset], data[offset:offset + info.size]))
        offset += info.size
    return frames


def decode(info, payload):
    """The frame upright, as BGR."""
    import cv2
    import numpy as np
    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    turn = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(info.angle)
    return frame if frame is None or turn is None else cv2.rotate(frame, turn)


def sensor_to_upright(x, y, angle):
    """Full-sensor normalized point -> full upright normalized point (clockwise ``angle``)."""
    return {0: (x, y), 90: (1 - y, x), 180: (1 - x, 1 - y), 270: (y, 1 - x)}[angle]


def upright_to_sensor(x, y, angle):
    return {0: (x, y), 90: (y, 1 - x), 180: (1 - x, 1 - y), 270: (1 - y, x)}[angle]


def frame_to_upright(info, fx, fy):
    """Point in the received (cropped, upright) frame, normalized -> full upright normalized."""
    # Undo the rotation inside the crop, then place the crop in the full sensor frame.
    lx, ly = upright_to_sensor(fx, fy, info.angle)
    cx, cy, cw, ch = info.crop
    return sensor_to_upright(cx + lx * cw, cy + ly * ch, info.angle)


def upright_rect_to_sensor(rect, angle):
    x, y, w, h = rect
    points = [upright_to_sensor(px, py, angle) for px, py in ((x, y), (x + w, y + h))]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


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
        frames = read_recording(self.path)
        span = frames[-1][0].seconds - frames[0][0].seconds if len(frames) > 1 else 0
        return (len(frames) - 1) / span if span else None

    async def frames(self):
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


@dataclass
class PhoneFrame:
    """One GAMST 240 frame as received: its header, the JPEG as sent, and the upright image on demand.

    Decoding a 240 fps JPEG stream costs more than a frame interval on full frames, so a consumer
    that only records the frame (``header + payload``) or shows one frame in eight never pays for
    the others; ``image()`` decodes once.
    """
    info: FrameHeader
    header: bytes
    payload: bytes
    _image: typing.Any = field(default=None, repr=False, compare=False)

    @property
    def recording(self):
        """Recorded between Start and Stop on the phone (or REC on the Mac): part of a take."""
        return bool(self.info.flags & RECORDING)

    def image(self):
        if self._image is None:
            self._image = decode(self.info, self.payload)
        return self._image


@dataclass(frozen=True)
class PhoneEvent:
    """Not a frame: the phone disconnected (``kind``), so a take in progress ends here."""
    kind: str
    reason: str = ''
    recording = False


class IPhone240Input:
    """The GAMST 240 iPhone app (``apps/ios-highspeed-camera``): live over TCP, or a ``raw.gmst`` replayed.

    Each frame's capture time is the phone's sensor time, re-timed onto the runtime clock by
    :meth:`clock` (fixed at the first frame, keeping the phone's spacing). Its data is a
    :class:`PhoneFrame`; ``decode=True`` yields upright BGR arrays instead, for generic video modules.

    Takes are marked by the phone: frames between Start and Stop carry the recording flag
    (``PhoneFrame.recording``). Live, every recording frame is kept in order, while a standby frame
    is skipped when a newer one is already waiting (standby only frames the shot); the first standby
    frame after a take is always kept, so the end of the take is never lost. A disconnect is
    yielded as a :class:`PhoneEvent`.

    :meth:`command` drives the phone like its own buttons (``record``, ``camera``, ``zoom``,
    ``crop``); the phone echoes its state in each frame header (:meth:`state`).
    """

    def __init__(self, port=None, *, replay=None, host='0.0.0.0', decode=False, max_frames=None):
        if (port is None) == (replay is None):
            raise ValueError('IPhone240Input listens on a port or replays a recording, not both')
        if replay is not None:
            replay = Path(replay)
            replay = replay / 'raw.gmst' if replay.is_dir() else replay
            if not replay.is_file():
                raise FileNotFoundError(f'no GAMST 240 recording at {replay}')
        self.port, self.replay, self.host = port, replay, host
        self.decode, self.max_frames = decode, max_frames
        self.modality = 'video' if decode else 'gamst240'
        self.live = replay is None
        self.received = 0
        self._queue = queue.Queue()
        self._capture = collections.deque(maxlen=241)
        self._phone = dict(sock=None, recording=False, front=False, zoom=1., angle=0, crop=FULL)
        self._listener = None
        self._stopped = False
        self._first = self._base = None
        self._span_ms = 0

    # ---------------------------------------------------------------- clock and state

    def clock(self, capture_ms):
        """Phone capture time (ms) -> runtime ms, so records land at the moment things happened."""
        if self._first is None:
            raise RuntimeError('no frame received yet')
        return self._base + round((capture_ms / 1000 - self._first) * 1000)

    def fps(self):
        return TakeInput(self.replay).fps() if self.replay is not None else None

    def state(self):
        window, phone = list(self._capture), self._phone
        fps = (len(window) - 1) / (window[-1] - window[0]) if len(window) > 1 and window[-1] > window[0] else 0.
        return dict(live=self.live, connected=phone['sock'] is not None, recording=phone['recording'],
                    front=phone['front'], zoom=phone['zoom'], angle=phone['angle'], crop=tuple(phone['crop']),
                    capture_fps=fps, received=self.received)

    # ---------------------------------------------------------------- phone commands

    def command(self, name, *, around=None):
        """Send a phone command; returns the message, or None when no phone is connected.

        ``crop`` with ``around=(info, x, y)``, a normalized point in that received frame, moves the
        phone's native-resolution strip there: full frame height (the object flies up), narrow
        enough to keep 240 fps. ``crop`` alone returns to the full frame.
        """
        phone = self._phone
        if name == 'record':
            message = dict(cmd='record', on=not phone['recording'])
        elif name == 'camera':
            message = dict(cmd='camera', position='back' if phone['front'] else 'front')
        elif name == 'zoom':
            message = dict(cmd='zoom', factor=1 if phone['zoom'] >= 3 else round(phone['zoom']) + 1)
        elif name == 'crop':
            if around is None:
                rect = list(FULL)
            else:
                info, fx, fy = around
                ux, _ = frame_to_upright(info, fx, fy)
                left = min(max(0., ux - .2), .6)
                rect = [round(v, 4) for v in upright_rect_to_sensor((left, 0., .4, 1.), info.angle)]
            message = dict(cmd='crop', rect=rect)
        else:
            raise ValueError(f'unknown phone command {name!r}; commands: record, camera, zoom, crop')
        sock = phone['sock']
        if sock is None:
            return None
        try:
            sock.sendall((json.dumps(message) + '\n').encode())
        except OSError:
            return None
        print(json.dumps({'command': message}), flush=True)
        return message

    # ---------------------------------------------------------------- frames

    def _note(self, info):
        self.received += 1
        self._capture.append(info.seconds)
        self._phone.update(recording=bool(info.flags & RECORDING), front=bool(info.flags & FRONT_CAMERA),
                           zoom=info.zoom, angle=info.angle, crop=info.crop)

    def _serve(self, listener):
        """Socket thread: reads the phone stream, so a busy consumer never back-pressures the phone."""
        while True:
            try:
                sock, peer = listener.accept()
            except OSError:
                return
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(json.dumps({'iphone': 'connected', 'peer': str(peer)}), flush=True)
            self._phone['sock'] = sock
            stream = sock.makefile('rb', buffering=1 << 20)
            reason = 'stream ended'
            try:
                while True:
                    magic = stream.read(4)
                    if magic not in HEADER_SIZE:
                        raise ConnectionError('stream ended' if len(magic) < 4 else 'not a GAMST 240 frame stream')
                    header = magic + stream.read(HEADER_SIZE[magic] - 4)
                    info = parse_header(header)
                    payload = stream.read(info.size)
                    if len(payload) < info.size:
                        raise ConnectionError('stream ended')
                    self._note(info)
                    self._queue.put(PhoneFrame(info, header, payload))
            except (OSError, ValueError, struct.error) as error:
                reason = str(error)
                print(json.dumps({'iphone': 'disconnected', 'reason': reason}), flush=True)
            finally:
                if self._phone['sock'] is sock:
                    self._phone['sock'] = None
                self._phone['recording'] = False
                self._queue.put(PhoneEvent('disconnected', reason))
                stream.close()
                sock.close()

    async def _received(self):
        self._listener = socket.create_server((self.host, self.port), reuse_port=False)
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._serve, args=(self._listener,), daemon=True).start()
        print(json.dumps({'listening': self.port}), flush=True)
        standby = False  # the last item handed on was a standby frame
        while not self._stopped:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(.003)
                continue
            if isinstance(item, PhoneFrame) and not item.recording:
                if standby and not self._queue.empty():
                    continue  # a newer frame is waiting; standby frames only frame the shot
                standby = True
            else:
                standby = False
            yield item

    async def _replayed(self):
        recording = read_recording(self.replay)
        self._span_ms = round(1000 * (recording[-1][0].seconds - recording[0][0].seconds)) if recording else 0
        for info, header, payload in recording:
            if self._stopped:
                return
            self._note(info)
            yield PhoneFrame(info, header, payload)

    async def frames(self):
        last, count = None, 0
        async for item in (self._received() if self.live else self._replayed()):
            if isinstance(item, PhoneEvent):
                if not self.decode:
                    last = _now_ms() if last is None else last + 1
                    yield InputFrame(item, self.modality, last)
                continue
            info = item.info
            if self._first is None:
                # The runtime rejects capture times in the future; a replay ends 1 s before now.
                self._first, self._base = info.seconds, _now_ms() - 1000 - self._span_ms
            data = item.image() if self.decode else item
            if data is None:
                continue
            stamp = self.clock(info.seconds * 1000)
            stamp = stamp if last is None or stamp > last else last + 1
            last, count = stamp, count + 1
            sensor = SensorMetadata(source_id='gamst240', sensor_id='camera', sequence=info.sequence,
                                    capture_sequence=info.sequence)
            yield InputFrame(data, self.modality, stamp, sensor=sensor)
            await asyncio.sleep(0)
            if self.max_frames is not None and count >= self.max_frames:
                break

    def stop(self):
        """End ``frames()`` (the owner quit); the phone's socket thread ends with the process."""
        self._stopped = True
        if self._listener is not None:
            self._listener.close()
            self._listener = None

    def close(self):
        self.stop()


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
    if kind == 'iphone':
        return IPhone240Input(int(value or 9123), **options)
    if kind == 'gamst240-take':
        return IPhone240Input(replay=value, **options)
    if kind == 'audio':
        return AudioFileInput(value, **options)
    if kind == 'microphone':
        return MicrophoneInput(**options)
    raise ValueError(f'unknown input source {kind!r}')
