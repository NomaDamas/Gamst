"""Input modules: the front layer that turns devices and recordings into capture-stamped frames.

Each module has ``frames()``, an async iterator of :class:`gamst.plugins.InputFrame` whose
``timestamp_ms`` is the capture time, and ``close()``. A :class:`gamst.pipeline.Pipeline` pumps them;
hosts that own capture themselves keep using ``Gamst.push_video`` / ``push_audio`` or
``Pipeline.push``. Third-party inputs plug in the same way (any object with ``frames()``).

A source may say more: ``live`` (default True; a recorded source waits for the modules, a live one
cannot), ``missed`` (how many frames or chunks it lost so far; the Pipeline turns an increase into a gap
before the next frame) and ``status()``.

``IPhone240Input`` is the GAMST 240 iPhone app (live over TCP, or a ``raw.gmst`` replayed): its
frames carry the phone's Start/Stop marks and it takes phone commands, so a view can drive it.

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


class _DeviceReader:
    """A thread that takes everything a device delivers, so a busy event loop never makes the device drop.

    ``read()`` runs in the thread and returns (item, bytes), or None at the end. Items wait in order up to
    ``max_bytes``; beyond that the oldest is given up, and the count travels with the item after it.
    """

    def __init__(self, read, max_bytes, name):
        self.read, self.max_bytes = read, max_bytes
        self.waiting, self.bytes, self.lock = collections.deque(), 0, threading.Lock()
        self.stop, self.ended, self.error = threading.Event(), False, None
        self.loop, self.ready = asyncio.get_running_loop(), asyncio.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=name)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                found = self.read()
                if found is None:
                    break
                with self.lock:
                    self.waiting.append([*found, 0])
                    self.bytes += found[1]
                    while self.bytes > self.max_bytes and len(self.waiting) > 1:
                        _, size, lost = self.waiting.popleft()
                        self.bytes -= size
                        self.waiting[0][2] += lost + 1
                self.loop.call_soon_threadsafe(self.ready.set)
        except BaseException as error:   # raised by items(), never lost in the thread
            self.error = error
        self.ended = True
        try:
            self.loop.call_soon_threadsafe(self.ready.set)
        except RuntimeError:             # the loop is already closed
            pass

    async def items(self):
        """(item, items lost just before it), until the device ends."""
        while True:
            self.ready.clear()
            with self.lock:
                found = self.waiting.popleft() if self.waiting else None
                if found is not None:
                    self.bytes -= found[1]
            if found is not None:
                yield found[0], found[2]
            elif self.ended:
                if self.error is not None and not self.stop.is_set():
                    raise self.error
                return
            else:
                await self.ready.wait()

    def close(self):
        """True when the thread is out of the device, so the device may be released."""
        self.stop.set()
        if self.thread is not threading.current_thread():
            self.thread.join(2)
        return not self.thread.is_alive()


class CameraInput:
    """A local camera (OpenCV). Capture time is the host's monotonic clock at read.

    Every frame the driver delivers is read and waits for the consumer (up to ``MAX_WAITING_BYTES``;
    beyond that the oldest is given up and counted in ``missed``). Frames the camera or its driver
    dropped before the read are not visible here and are not counted.
    """
    modality = 'video'
    live = True
    MAX_WAITING_BYTES = 256 * 1024 * 1024

    def __init__(self, index=0, *, max_frames=None):
        self.index, self.max_frames, self.missed = index, max_frames, 0
        self._cap = self._reader = None

    async def frames(self):
        import cv2
        backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
        cap = self._cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            raise RuntimeError(f'cannot open camera {self.index}')
        base, start, last, count = _now_ms(), time.monotonic(), None, 0

        def read():
            ok, frame = cap.read()
            return ((base + round(1000 * (time.monotonic() - start)), frame), frame.nbytes) if ok else None

        self._reader = _DeviceReader(read, self.MAX_WAITING_BYTES, f'gamst-camera-{self.index}')
        try:
            async for (stamp, frame), lost in self._reader.items():
                if self.max_frames is not None and count >= self.max_frames:
                    break
                self.missed += lost
                stamp = stamp if last is None or stamp > last else last + 1
                last, count = stamp, count + 1
                yield InputFrame(frame, 'video', stamp)
                if count == self.max_frames:
                    break
        finally:
            self.close()

    def status(self):
        reader = self._reader
        oldest = reader.waiting[0][0][0] if reader is not None and reader.waiting else None
        return dict(waiting=len(reader.waiting) if reader is not None else 0,
                    backlog_ms=0 if oldest is None else max(0, _now_ms() - oldest))

    def close(self):
        reader, self._reader = self._reader, None
        cap, self._cap = self._cap, None
        if cap is not None and (reader is None or reader.close()):
            cap.release()   # never while a read is still inside the driver


class VideoFileInput:
    """A video file; ``capture_fps`` overrides file timing (high-speed clips saved at 30 fps).

    ``start_s`` / ``end_s`` read only that part of the file. ``origin_ms`` is the capture time of the file's
    first frame, so ``timestamp_ms - origin_ms`` is a position in the file whichever part is read.
    """
    modality = 'video'
    live = False

    def __init__(self, path, *, capture_fps=None, max_frames=None, start_s=0., end_s=None):
        self.path, self.capture_fps, self.max_frames = str(path), capture_fps, max_frames
        self.start_s, self.end_s, self.origin_ms = float(start_s), end_s, None
        self._cap = None

    def fps(self):
        import cv2
        cap = cv2.VideoCapture(self.path)
        try:
            return self.capture_fps or cap.get(cv2.CAP_PROP_FPS) or None
        finally:
            cap.release()

    def duration_s(self):
        """From the file's frame count and rate; None when the container does not say."""
        import cv2
        cap = cv2.VideoCapture(self.path)
        try:
            if not cap.isOpened():
                raise RuntimeError(f'cannot open video {self.path}')
            fps, total = self.capture_fps or cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            return total / fps if fps and total and total > 0 else None
        finally:
            cap.release()

    async def frames(self):
        import cv2
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f'cannot open video {self.path}')
        fps = self.capture_fps or self._cap.get(cv2.CAP_PROP_FPS) or 30.
        total = self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        base = self.origin_ms = _now_ms() - round(1000 * total / fps) - 1000
        position = first = round(self.start_s * fps)
        if first and not (self._cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                          and round(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) == first):
            self._cap.release()      # this container cannot seek: decode up to the start
            self._cap = cv2.VideoCapture(self.path)
            for _ in range(first):
                if not self._cap.grab():
                    break
        last = None
        while self.max_frames is None or position - first < self.max_frames:
            ok, frame = self._cap.read()
            if not ok:
                break
            offset = position * 1000 / self.capture_fps if self.capture_fps else self._cap.get(cv2.CAP_PROP_POS_MSEC)
            if self.end_s is not None and offset > 1000 * self.end_s:
                break
            stamp = base + round(offset)
            stamp = stamp if last is None or stamp > last else last + 1
            last, position = stamp, position + 1
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
    live = False

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
    live = False

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
    """The default microphone as mono float32 chunks (needs ``sounddevice``).

    A chunk's time is the capture time of its last sample on the device's sample clock: it starts at the
    first read (arrival minus what still waits in the device buffer and the latency the device reports) and
    advances by samples, so scheduling delays do not move it. An overflow reported by the device, or a
    sample clock more than ``resync_ms`` away from arrival (loss the device did not report, or clock
    drift), is a gap: what was buffered around an overflow is dropped, the clock starts again, and
    ``missed`` grows by the lost time in chunks (at least one).
    """
    modality = 'audio'
    live = True
    resync_ms = 250
    MAX_WAITING_BYTES = 64 * 1024 * 1024

    def __init__(self, *, sample_rate=16000, chunk_ms=100, device=None, max_chunks=None, stream_factory=None):
        self.sample_rate, self.chunk_ms, self.device = sample_rate, chunk_ms, device
        self.max_chunks, self._factory, self._stream = max_chunks, stream_factory, None
        self.missed = self.overflows = self.resyncs = self.lost_ms = 0
        self._reader = None

    async def frames(self):
        if self._factory is None:
            try:
                import sounddevice
            except ImportError as exc:
                raise RuntimeError('MicrophoneInput needs: pip install sounddevice') from exc
            factory = lambda **options: sounddevice.InputStream(**options)
        else:
            factory = self._factory
        rate = self.sample_rate
        chunk = rate * self.chunk_ms // 1000
        stream = self._stream = factory(samplerate=rate, channels=1, dtype='float32', blocksize=chunk,
                                        device=self.device)
        stream.start()
        clock = dict(anchor=None, done=0, last=None, lost=False)

        def read():
            while True:
                samples, overflowed = stream.read(chunk)
                waiting = int(getattr(stream, 'read_available', 0) or 0)
                arrived = _now_ms() - 1000 * waiting / rate - 1000 * float(getattr(stream, 'latency', 0) or 0)
                if not overflowed:
                    break
                self.overflows += 1
                clock.update(anchor=None, lost=True)
                if waiting:
                    stream.read(waiting)   # the loss is at an unknown place in it
            samples = samples.reshape(-1).copy()
            clock['done'] += len(samples)
            if clock['anchor'] is not None and abs(
                    arrived - clock['anchor'] - 1000 * clock['done'] / rate) > self.resync_ms:
                self.resyncs += 1
                clock.update(anchor=None, lost=True)
            if clock['anchor'] is None:
                clock.update(done=len(samples), anchor=arrived - 1000 * len(samples) / rate)
            stamp, last = round(clock['anchor'] + 1000 * clock['done'] / rate), clock['last']
            if last is not None and stamp <= last:
                stamp = last + 1
            missed = 0
            if clock['lost']:
                gone = 0 if last is None else max(0, round(stamp - 1000 * len(samples) / rate - last))
                self.lost_ms += gone
                missed = max(1, round(gone / self.chunk_ms))
            clock.update(last=stamp, lost=False)
            return (samples, stamp, missed), samples.nbytes

        self._reader, count = _DeviceReader(read, self.MAX_WAITING_BYTES, 'gamst-microphone'), 0
        try:
            async for (samples, stamp, missed), lost in self._reader.items():
                self.missed += missed + lost
                count += 1
                yield InputFrame(samples, 'audio', stamp, sample_rate=rate)
                if count == self.max_chunks:
                    break
        finally:
            self.close()

    def status(self):
        return dict(overflows=self.overflows, resyncs=self.resyncs, lost_ms=self.lost_ms,
                    waiting=len(self._reader.waiting) if self._reader is not None else 0)

    def close(self):
        reader, self._reader = self._reader, None
        stream, self._stream = self._stream, None
        if reader is not None:
            reader.close()
        if stream is not None:
            stream.stop()
            stream.close()


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
