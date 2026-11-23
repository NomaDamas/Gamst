"""GAMST 240 live: an iPhone high-fps camera feeding GAMST ``vision.rotation`` with a HUD.

The phone streams continuously so the HUD can frame the shot; Start/Stop on the phone
marks a *take*. Only take frames go through GAMST (every frame, never dropped), and each
take is saved under ``--record-dir``:

* ``raw.gmst``      the phone stream as received (replay with ``--iphone-replay``)
* ``hud.mp4``       the HUD at real time (30 fps, 1920x1080), ready to upload
* ``slowmo.mp4``    every captured frame with overlays at 30 fps (8x slow motion at 240 fps),
                    rendered by a background process a few seconds after the take
* ``measurements.jsonl``  GAMST's per-frame rotation measurement, aligned with ``raw.gmst``
* ``summary.json``  tosses, rotation events and capture statistics

GAMST owns measurement, records, events and Watch alarms; this module only receives,
renders and writes files.
"""
from __future__ import annotations

import asyncio
import collections
import json
import queue
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import typing
from pathlib import Path

from ..providers.rotation import RotationPlugin
from ..providers.target_tracks import measure_exemplar

# "GMS3" | u32 sequence | f64 capture seconds | u16 width | u16 height | u8 flags | u8 zoom x10 |
# u16 clockwise degrees to upright | u16 x4 crop rect (sensor-normalized x65535) | u32 jpeg bytes.
# Frames arrive in sensor orientation; the Mac rotates them upright. "GMS2" frames are already upright
# and uncropped; "GMST" (first app build) has no flags and counts as one continuous take.
# The Mac drives the phone with newline-delimited JSON commands written back on the same socket.
HEADER_SIZE = {b'GMS3': 36, b'GMS2': 28, b'GMST': 24}
RECORDING, FRONT_CAMERA = 1, 2
FULL = (0., 0., 1., 1.)
MEASURE_PIXELS = 300_000  # GAMST measures frames scaled down to at most this many pixels


def measured_size(frame):
    height, width = frame.shape[:2]
    scale = min(1., (MEASURE_PIXELS / (width * height)) ** .5)
    return max(2, round(width * scale)), max(2, round(height * scale))


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


def pixel_density(info, measured_width, frame_width):
    """Measured pixels per unit of sqrt(full-sensor area) for this frame's mode and crop."""
    _, _, cw, ch = info.crop
    native = ((info.width / cw) * (info.height / ch)) ** .5
    return native * measured_width / frame_width


# ---------------------------------------------------------------- rendering

BG, PANEL, LINE = (18, 14, 11), (34, 28, 23), (60, 52, 44)
WHITE, MUTED, DIM = (240, 240, 240), (160, 152, 146), (100, 94, 90)
ACCENT, GOOD, WARN, REC = (255, 196, 64), (120, 214, 96), (60, 180, 255), (70, 70, 235)
CANVAS = (1920, 1080)
VIDEO = (32, 104, 1376, 774)  # x, y, width, height (16:9)
FONT_FILES = {'sans': '/System/Library/Fonts/SFNS.ttf', 'mono': '/System/Library/Fonts/SFNSMono.ttf'}


class Text:
    """Anti-aliased system-font text, cached per string so 240 fps rendering stays cheap."""

    def __init__(self):
        from PIL import ImageFont
        self._font_cls = ImageFont
        self._fonts, self._cache = {}, collections.OrderedDict()

    def _font(self, family, size, weight):
        key = (family, size, weight)
        if key not in self._fonts:
            try:
                font = self._font_cls.truetype(FONT_FILES[family], size)
                if weight != 'regular':
                    try:
                        font.set_variation_by_name(weight.capitalize())
                    except (OSError, ValueError):
                        pass
            except OSError:
                font = self._font_cls.load_default(size)
            self._fonts[key] = font
        return self._fonts[key]

    def patch(self, text, size, color, weight='regular', family='sans'):
        import numpy as np
        from PIL import Image, ImageDraw
        key = (text, size, color, weight, family)
        cached = self._cache.get(key)
        if cached is None:
            font = self._font(family, size, weight)
            left, top, right, bottom = font.getbbox(text)
            width, height = max(1, right - left + 2), max(1, bottom - top + 2)
            image = Image.new('L', (width, height), 0)
            ImageDraw.Draw(image).text((-left + 1, -top + 1), text, font=font, fill=255)
            alpha = np.asarray(image, dtype=np.float32)[..., None] / 255
            cached = (alpha, np.array(color, np.float32), top)
            self._cache[key] = cached
            if len(self._cache) > 2048:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return cached

    def put(self, image, text, x, y, size, color=WHITE, weight='regular', family='sans', anchor='left'):
        """Draw ``text`` with its cap-height top at ``y``; returns the drawn width."""
        alpha, rgb, _ = self.patch(text, size, color, weight, family)
        h, w = alpha.shape[:2]
        if anchor == 'right':
            x -= w
        elif anchor == 'center':
            x -= w // 2
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(image.shape[1], int(x) + w), min(image.shape[0], int(y) + h)
        if x1 <= x0 or y1 <= y0:
            return w
        a = alpha[y0 - int(y):y1 - int(y), x0 - int(x):x1 - int(x)]
        region = image[y0:y1, x0:x1]
        region[:] = (region * (1 - a) + rgb * a).astype(image.dtype)
        return w


def _pill(image, text_engine, x, y, label, color, filled=True, size=22):
    import cv2
    alpha, _, _ = text_engine.patch(label, size, WHITE, 'semibold')
    width, height = alpha.shape[1] + 36, size + 20
    overlay = image.copy()
    cv2.rectangle(overlay, (x + height // 2, y), (x + width - height // 2, y + height), color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + height // 2, y + height // 2), height // 2, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + width - height // 2, y + height // 2), height // 2, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, .9 if filled else .35, image, .1 if filled else .65, 0, image)
    text_engine.put(image, label, x + 18, y + 9, size, WHITE, 'semibold')
    return width


class Hud:
    """Composes the 1920x1080 HUD from a frame and GAMST's latest rotation measurement."""

    def __init__(self, *, show_all=False):
        import numpy as np
        self.np = np
        self.text = Text()
        self.show_all = show_all
        self.trails = {}
        self.base = self._static()

    def _static(self):
        import cv2
        np = self.np
        canvas = np.full((CANVAS[1], CANVAS[0], 3), BG, np.uint8)
        x, y, w, h = VIDEO
        cv2.rectangle(canvas, (x - 2, y - 2), (x + w + 1, y + h + 1), LINE, 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (1432, y - 2), (1888, y + h + 1), PANEL, -1)
        cv2.rectangle(canvas, (x - 2, 900), (1888, 1052), PANEL, -1)
        t = self.text
        width = self._logo(canvas, 28, 20)
        t.put(canvas, 'Halo  ·  vision.rotation  ·  moment capture', 28 + width + 20, 44, 22, MUTED)
        t.put(canvas, 'LAST TOSS', 1456, 124, 18, MUTED, 'semibold')
        t.put(canvas, 'THIS TAKE', 1456, 560, 18, MUTED, 'semibold')
        t.put(canvas, 'GAMST CONTEXT   Watch: rotation.ended  ->  timestamped records an LLM can read later',
              48, 914, 18, MUTED, 'semibold')
        return canvas

    def _logo(self, canvas, x, y):
        """Alpha-blend the packaged GAMST logo; returns its width."""
        from PIL import Image
        np = self.np
        logo = Image.open(Path(__file__).parent / 'assets' / 'gamst-logo.png').convert('RGBA')
        rgba = np.asarray(logo, dtype=np.float32)
        alpha = rgba[..., 3:] / 255
        h, w = rgba.shape[:2]
        region = canvas[y:y + h, x:x + w]
        region[:] = (region * (1 - alpha) + rgba[..., 2::-1] * alpha).astype(np.uint8)
        return w

    def update(self, value):
        """Feed every measured frame so motion trails stay continuous (measurement coordinates)."""
        seen = set()
        for item in (value or {}).get('objects', []):
            if not item['observed']:
                continue
            x, y, size, _ = item['box']
            trail = self.trails.setdefault(item['track_id'], collections.deque(maxlen=48))
            trail.append((x + size / 2, y + size / 2))
            seen.add(item['track_id'])
        for track_id in [key for key in self.trails if key not in seen]:
            trail = self.trails[track_id]
            trail.popleft()
            if not trail:
                del self.trails[track_id]

    def video_layer(self, frame, value):
        """The frame letterboxed into the video rect with GAMST overlays (also used for slow motion)."""
        import cv2
        np = self.np
        _, _, w, h = VIDEO
        layer = np.zeros((h, w, 3), np.uint8)
        fh, fw = frame.shape[:2]
        scale = min(w / fw, h / fh)
        dw, dh = int(fw * scale) // 2 * 2, int(fh * scale) // 2 * 2
        ox, oy = (w - dw) // 2, (h - dh) // 2
        layer[oy:oy + dh, ox:ox + dw] = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        measured = measured_size(frame)
        sx, sy = dw / measured[0], dh / measured[1]
        for trail in self.trails.values():
            points = [(int(ox + px * sx), int(oy + py * sy)) for px, py in trail]
            for index in range(1, len(points)):
                a, b = points[index - 1], points[index]
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 120:  # a re-seeded track, not motion
                    continue
                cv2.line(layer, a, b, ACCENT, 1 + 3 * index // len(points), cv2.LINE_AA)
        for item in (value or {}).get('objects', []):
            if not item['observed']:
                continue
            x, y, size, _ = item['box']
            x0, y0 = int(ox + x * sx), int(oy + y * sy)
            x1, y1 = int(ox + (x + size) * sx), int(oy + (y + size) * sy)
            status = item['status']
            color = ACCENT if status == 'counting' else WARN if status == 'undersampled' else DIM
            corner = max(6, (x1 - x0) // 4)
            for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                cv2.line(layer, (cx, cy), (cx + dx * corner, cy), color, 2, cv2.LINE_AA)
                cv2.line(layer, (cx, cy), (cx, cy + dy * corner), color, 2, cv2.LINE_AA)
            if status != 'idle' and item.get('flip_rev_s'):
                label = f"{item['flip_rev_s']:.1f} rev/s"
                self.text.put(layer, label, x0, max(2, y0 - 26), 20, color, 'semibold')
        return layer

    def render(self, frame, value, state):
        canvas = self.base.copy()
        x, y, w, h = VIDEO
        if frame is not None:
            canvas[y:y + h, x:x + w] = self.video_layer(frame, value)
        else:
            self.text.put(canvas, 'waiting for the iPhone', x + w // 2, y + h // 2 - 20, 34, MUTED, anchor='center')
        self._top(canvas, state)
        self._panel(canvas, state)
        self._feed(canvas, state)
        return canvas

    def _top(self, canvas, state):
        t = self.text
        mode = state['mode']
        if mode == 'recording':
            label, color = f"●  REC  {state['take_seconds']:05.1f} s", REC
        elif mode == 'processing':
            label, color = f"GAMST catching up  ·  {state['lag_s']:.1f} s behind", WARN
        elif mode == 'saved':
            label, color = f"saved  {state['saved']}", GOOD
        elif mode == 'standby':
            label, color = 'STANDBY  ·  REC on the iPhone or here (space) starts a take', DIM
        else:
            label, color = 'waiting for the iPhone', DIM
        alpha, _, _ = t.patch(label, 22, WHITE, 'semibold')
        _pill(canvas, t, 960 - (alpha.shape[1] + 36) // 2, 32, label, color)
        right = f"camera {state['capture_fps']:.0f} fps"
        if state.get('frame_size'):
            right += f"  ·  {state['frame_size'][0]}x{state['frame_size'][1]}"
        if state.get('target_px'):
            right += f"  ·  target {state['target_px']:.0f} px"
        t.put(canvas, right, 1888, 34, 22, WHITE, 'semibold', anchor='right')
        t.put(canvas, f"countable up to {state['capture_fps'] / 4:.0f} rev/s flipping", 1888, 64, 18, MUTED, anchor='right')

    def _panel(self, canvas, state):
        t = self.text
        toss = state['last_toss']
        if toss is None:
            t.put(canvas, '0.0', 1450, 150, 150, LINE, 'bold')
            t.put(canvas, 'toss a coin through the frame', 1456, 340, 22, MUTED)
        else:
            number = f"{toss['turns']:.1f}"
            width = t.put(canvas, number, 1450, 150, 150, WHITE, 'bold')
            t.put(canvas, 'turns', 1456 + width + 8, 262, 30, MUTED, 'semibold')
            t.put(canvas, f"{toss['rev_s']:.1f} rev/s", 1456, 338, 34, ACCENT, 'semibold')
            t.put(canvas, f"{toss['duration_ms'] / 1000:.2f} s in view  ·  {toss['capture_fps']:.0f} fps",
                  1456, 388, 22, MUTED)
            kind = (toss.get('trajectory') or {}).get('kind')
            chip = {'free_fall': ('FREE FALL', GOOD), 'stationary': ('SPINNING IN PLACE', ACCENT)}.get(
                kind, ('NOT AIRBORNE', DIM))
            _pill(canvas, t, 1456, 430, chip[0], chip[1], size=18)
        t.put(canvas, f"{state['tosses']} measured", 1456, 596, 30, WHITE, 'semibold')
        t.put(canvas, f"{state['frames']} frames  ·  {state['ignored']} motions ignored", 1456, 640, 20, MUTED)
        for index, item in enumerate(state['history'][-6:][::-1]):
            line = f"{item['turns']:4.1f} turns   {item['rev_s']:5.1f} rev/s"
            t.put(canvas, line, 1456, 690 + index * 30, 22, WHITE if index == 0 else MUTED, family='mono')

    def _feed(self, canvas, state):
        for index, line in enumerate(state['feed'][-4:]):
            self.text.put(canvas, line, 48, 950 + index * 24, 17, WHITE if index == len(state['feed'][-4:]) - 1 else MUTED,
                          family='mono')


# ---------------------------------------------------------------- files

class VideoSink:
    """H.264 via ffmpeg when available (upload-ready), else OpenCV mp4v."""

    def __init__(self, path, fps, size):
        self.path, self.size = str(path), size
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            self.process = subprocess.Popen(
                [ffmpeg, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                 '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-', '-c:v', 'libx264',
                 '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', self.path],
                stdin=subprocess.PIPE)
            self.writer = None
        else:
            import cv2
            self.process = None
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    def write(self, image):
        if self.process is not None:
            self.process.stdin.write(image.tobytes())
        else:
            self.writer.write(image)

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
        else:
            self.writer.release()


class Take:
    """One Start→Stop take: raw stream, per-frame measurements, real-time HUD video and a summary.

    ``slowmo.mp4`` is rendered by a separate process after the take closes, so it never
    competes with GAMST for the live loop.
    """

    def __init__(self, root):
        stamp = time.strftime('%Y%m%d-%H%M%S')
        self.dir = Path(root) / f'take-{stamp}'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.raw = open(self.dir / 'raw.gmst', 'wb')
        self.measurements = open(self.dir / 'measurements.jsonl', 'w')
        self.hud = VideoSink(self.dir / 'hud.mp4', 30, CANVAS)
        self.events, self.frames, self.first, self.last = [], 0, None, None

    def add(self, header, payload, seconds, value, toss):
        self.raw.write(header + payload)
        self.measurements.write(json.dumps(dict(objects=value['objects'], toss=toss), separators=(',', ':')) + '\n')
        self.frames += 1
        self.first = seconds if self.first is None else self.first
        self.last = seconds

    def close(self, summary):
        self.raw.close()
        self.measurements.close()
        self.hud.close()
        span = (self.last - self.first) if self.frames > 1 else 0
        summary.update(frames=self.frames, seconds=round(span, 3),
                       capture_fps=round((self.frames - 1) / span, 1) if span else None, events=self.events)
        (self.dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        subprocess.Popen([sys.executable, '-m', 'gamst.demos.rotation_live', 'slowmo', str(self.dir)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return self.dir


def render_slowmo(directory):
    """Every captured frame of a take with its GAMST overlay at 30 fps (8x slow at 240 fps)."""
    directory = Path(directory)
    hud, sink = Hud(), None
    with open(directory / 'measurements.jsonl') as measurements:
        for (info, _, payload), line in zip(read_recording(directory / 'raw.gmst'), measurements):
            frame = decode(info, payload)
            if frame is None:
                continue
            value = json.loads(line)
            hud.update(value)
            layer = hud.video_layer(frame, value)
            hud.text.put(layer, 'GAMST  ·  every captured frame  ·  slow motion', 24, 20, 24, WHITE, 'semibold')
            toss = value.get('toss')
            if toss is not None:
                hud.text.put(layer, f"{toss['turns']:.1f} turns  ·  {toss['rev_s']:.1f} rev/s", 24, 56, 34, ACCENT, 'bold')
            if sink is None:
                sink = VideoSink(directory / 'slowmo.part.mp4', 30, (layer.shape[1], layer.shape[0]))
            sink.write(layer)
    if sink is not None:
        sink.close()
        (directory / 'slowmo.part.mp4').rename(directory / 'slowmo.mp4')


# ---------------------------------------------------------------- run

def _toss_line(event):
    kind = (event.get('trajectory') or {}).get('kind', 'unknown')
    turns = '-' if event.get('turns') is None else f"{event['turns']:.1f} turns"
    rev = '' if event.get('rev_s') is None else f"  {event['rev_s']:.1f} rev/s"
    clock = time.strftime('%H:%M:%S', time.localtime(event['timestamp_ms'] / 1000))
    return (f"{clock}.{event['timestamp_ms'] % 1000:03d}  rotation.ended  {event['axis']}  {turns}{rev}"
            f"  {event['status']}  {kind}  {event.get('capture_fps') or 0:.0f} fps")


def is_toss(event, show_all=False, targeted=False):
    """HUD policy: thrown flips, or a sustained spin in place (>=1 turn over >=250 ms) when no target
    is set. A rate above ``expected_max_rev_s`` is not a coin flip but shape noise, so it never counts."""
    if event.get('axis') != 'flip' or event.get('status') != 'counting' or not event.get('turns'):
        return False
    if show_all:
        return True
    if event.get('rev_s') and event.get('expected_max_rev_s') and event['rev_s'] > event['expected_max_rev_s']:
        return False
    kind = (event.get('trajectory') or {}).get('kind')
    if kind == 'stationary' and not targeted:
        return event['turns'] >= 1 and event.get('duration_ms', 0) >= 250
    return kind == 'free_fall'


async def run(args):
    import cv2
    import numpy as np
    from .. import Gamst

    record_dir = Path(args.record_dir or 'gamst-takes')
    # Every take frame, in order; GAMST must see the full capture cadence. Filled by a socket thread
    # so a busy GAMST loop never stops reading and back-pressures the phone into dropping frames.
    take_queue = queue.Queue()
    latest = {'frame': None}       # standby view: newest frame only
    stats = dict(received=0, capture=collections.deque(maxlen=241))

    def receive(info, header, payload):
        stats['received'] += 1
        stats['capture'].append(info.seconds)
        if info.flags & RECORDING:
            take_queue.put(('frame', info, header, payload))
        else:
            latest['frame'] = (info, payload)

    recording_flag = {'on': False}
    phone = dict(sock=None, recording=False, front=False, zoom=1., angle=0, crop=FULL)

    def mark(flags):
        on = bool(flags & RECORDING)
        if on != recording_flag['on']:
            recording_flag['on'] = on
            take_queue.put(('start',) if on else ('stop',))

    def send(message):
        sock = phone['sock']
        if sock is None:
            return
        try:
            sock.sendall((json.dumps(message) + '\n').encode())
        except OSError:
            return
        print(json.dumps({'command': message}), flush=True)

    def command(name):
        """Drive the phone exactly like its own buttons; the phone echoes state in each frame header."""
        if name == 'record':
            send(dict(cmd='record', on=not phone['recording']))
        elif name == 'camera':
            send(dict(cmd='camera', position='back' if phone['front'] else 'front'))
        elif name == 'zoom':
            send(dict(cmd='zoom', factor=1 if phone['zoom'] >= 3 else round(phone['zoom']) + 1))
        elif name == 'target':
            clear_target()

    def serve(listener):
        while True:
            try:
                sock, peer = listener.accept()
            except OSError:
                return
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(json.dumps({'iphone': 'connected', 'peer': str(peer)}), flush=True)
            phone['sock'] = sock
            stream = sock.makefile('rb', buffering=1 << 20)
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
                    phone.update(recording=bool(info.flags & RECORDING), front=bool(info.flags & FRONT_CAMERA),
                                 zoom=info.zoom, angle=info.angle, crop=info.crop)
                    mark(info.flags)
                    receive(info, header, payload)
            except (OSError, ValueError, struct.error) as error:
                print(json.dumps({'iphone': 'disconnected', 'reason': str(error)}), flush=True)
            finally:
                if phone['sock'] is sock:
                    phone['sock'] = None
                mark(0)
                stream.close()
                sock.close()

    async def replay(path):
        frames = read_recording(path)
        stats['span_ms'] = round(1000 * (frames[-1][0].seconds - frames[0][0].seconds)) if frames else 0
        for info, header, payload in frames:
            mark(info.flags)
            receive(info, header, payload)
            await asyncio.sleep(0)
        mark(0)
        take_queue.put(('end',))

    listener = feeder = None
    if args.iphone_replay:
        feeder = asyncio.create_task(replay(args.iphone_replay))
    else:
        listener = socket.create_server(('0.0.0.0', args.iphone), reuse_port=False)
        threading.Thread(target=serve, args=(listener,), daemon=True).start()
        print(json.dumps({'listening': args.iphone, 'takes': str(record_dir.resolve())}), flush=True)

    hud = Hud(show_all=args.show_all)
    title = 'GAMST 240 - vision.rotation'
    buttons = []  # (name, x, y, width, height) in HUD pixels; drawn on the window only, never recorded
    drag = dict(start=None, now=None, button=None)
    view = dict(info=None, shape=None)  # header and size of the frame on screen, to map clicks
    target = dict(norm=None, applied=None)  # exemplar size per unit of sensor density

    def to_frame(x, y):
        """HUD pixel -> normalized point in the frame on screen (None outside the picture)."""
        if view['shape'] is None:
            return None
        fh, fw = view['shape'][:2]
        scale = min(VIDEO[2] / fw, VIDEO[3] / fh)
        dw, dh = int(fw * scale) // 2 * 2, int(fh * scale) // 2 * 2
        fx = (x - VIDEO[0] - (VIDEO[2] - dw) // 2) / dw
        fy = (y - VIDEO[1] - (VIDEO[3] - dh) // 2) / dh
        return (fx, fy) if 0 <= fx <= 1 and 0 <= fy <= 1 else None

    def set_target_from_box(a, b):
        """The user's box around the object fixes its size and moves the phone's native-resolution
        strip over it: full frame height (the object flies up), narrow enough to keep 240 fps."""
        info, shape = view['info'], view['shape']
        if info is None or a is None or b is None:
            return
        fh, fw = shape[:2]
        size_px = max(abs(a[0] - b[0]) * fw, abs(a[1] - b[1]) * fh)
        if view.get('frame') is not None:
            # Boxes are drawn loosely; the object's own outline inside the box gives its real size.
            box = (min(a[0], b[0]) * fw, min(a[1], b[1]) * fh, max(a[0], b[0]) * fw, max(a[1], b[1]) * fh)
            measured = measure_exemplar(view['frame'], box)
            size_px = measured if measured is not None else .5 * size_px
        measured_w = measured_size(np.empty((fh, fw, 1), np.uint8))[0]
        if size_px * measured_w / fw < 3:
            return
        target.update(norm=size_px * measured_w / fw / pixel_density(info, measured_w, fw), applied=None)
        ux, _ = frame_to_upright(info, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        left = min(max(0., ux - .2), .6)
        send(dict(cmd='crop', rect=[round(v, 4) for v in upright_rect_to_sensor((left, 0., .4, 1.), info.angle)]))
        print(json.dumps({'target': {'size_px_on_screen_frame': round(size_px, 1), 'norm': round(target['norm'], 6)}}),
              flush=True)

    def clear_target():
        target.update(norm=None, applied=None)
        if plugin_ref:
            plugin_ref[0].set_target(None)
        send(dict(cmd='crop', rect=list(FULL)))

    plugin_ref = []
    if not args.headless:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1440, 810)
        cv2.moveWindow(title, 40, 60)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                drag.update(start=None, now=None, button=None)
                for name, bx, by, bw, bh in buttons:
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        drag['button'] = name
                        return
                if to_frame(x, y) is not None:
                    drag.update(start=(x, y), now=(x, y))
            elif event == cv2.EVENT_MOUSEMOVE and drag['start'] is not None:
                drag['now'] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                if drag['button'] is not None:
                    command(drag['button'])
                elif drag['start'] is not None and abs(x - drag['start'][0]) + abs(y - drag['start'][1]) > 6:
                    set_target_from_box(to_frame(*drag['start']), to_frame(x, y))
                drag.update(start=None, now=None, button=None)

        cv2.setMouseCallback(title, on_mouse)
    pending_alarms = []
    state = dict(mode='waiting', take_seconds=0., lag_s=0., saved='', capture_fps=0., frame_size=None,
                 last_toss=None, tosses=0, ignored=0, frames=0, history=[], feed=[], target_px=None)
    take = None
    first = base = last_stamp = None
    hud_due = take_started = None
    last_view = 0.
    shown_frame = None
    saved = []

    def capture_fps():
        window = stats['capture']
        return (len(window) - 1) / (window[-1] - window[0]) if len(window) > 1 and window[-1] > window[0] else 0.

    def controls(image):
        """Phone controls for the Mac window (space/r: record, c: camera, z: zoom, t: clear target, q: quit)."""
        import cv2
        buttons.clear()
        connected = phone['sock'] is not None
        items = [('record', '■  STOP' if phone['recording'] else '●  REC', REC if connected else DIM),
                 ('camera', f"camera  {'front' if phone['front'] else 'back'}", PANEL if connected else DIM),
                 ('zoom', f"zoom  {phone['zoom']:g}x", PANEL if connected else DIM),
                 ('target', 'target  ✕' if target['norm'] else 'target: drag a box', GOOD if target['norm'] else PANEL)]
        widths = [hud.text.patch(label, 24, WHITE, 'semibold')[0].shape[1] + 36 for _, label, _ in items]
        x = VIDEO[0] + (VIDEO[2] - sum(widths) - 16 * (len(items) - 1)) // 2
        y = VIDEO[1] + VIDEO[3] - 72
        for (name, label, color), width in zip(items, widths):
            _pill(image, hud.text, x, y, label, color, size=24)
            buttons.append((name, x, y, width, 44))
            x += width + 16
        hint = ('drag a box around the coin   ·   space  record   ·   c  camera   ·   z  zoom   ·   t  clear   ·   q  quit'
                if connected else 'phone not connected')
        hud.text.put(image, hint, VIDEO[0] + VIDEO[2] // 2, y + 54, 16, MUTED, anchor='center')
        if drag['start'] is not None and drag['now'] is not None:
            cv2.rectangle(image, drag['start'], drag['now'], GOOD, 2, cv2.LINE_AA)
        return image

    def show(image):
        nonlocal last_view
        if args.headless:
            return False
        cv2.imshow(title, controls(image.copy()) if not args.iphone_replay else image)
        last_view = time.monotonic()
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(' '), ord('r')):
            command('record')
        elif key == ord('c'):
            command('camera')
        elif key == ord('z'):
            command('zoom')
        elif key == ord('t'):
            clear_target()
        return key in (27, ord('q'))

    try:
        async with Gamst(consent=True, store=args.store) as human:
            plugin = RotationPlugin(human, expected_max_rev_s=args.expected_max_rev_s, region_source='motion',
                                    region_refresh_ms=20, record_every_ms=33)
            human.register(plugin)
            plugin_ref.append(plugin)
            watch = human.watches.create(owner='user', description='Tell me how many turns each rotation made.',
                rule=dict(conditions=[dict(capability_id='vision.rotation', path='events', op='contains',
                                           value='rotation.ended', within_ms=1000)],
                          cooldown_ms=0, max_per_minute=600, context_window_ms=3000))
            human.watches.add_sink(pending_alarms.append)
            while True:
                try:
                    item = take_queue.get_nowait()
                except queue.Empty:
                    item = None
                if item is None:
                    # Standby: show the newest live frame at ~30 fps, no measurement.
                    state['capture_fps'] = capture_fps()
                    if take is None and (state['mode'] == 'waiting' or (
                            state['mode'] == 'saved' and time.monotonic() - state['saved_at'] > 4)):
                        state['mode'] = 'standby' if latest['frame'] is not None else 'waiting'
                    if take is None and latest['frame'] is not None and time.monotonic() - last_view > 1 / 30:
                        info, payload = latest['frame']
                        frame = decode(info, payload)
                        if frame is not None:
                            state['frame_size'] = (frame.shape[1], frame.shape[0])
                            shown_frame = frame
                            view.update(info=info, shape=frame.shape, frame=frame)
                            if show(hud.render(frame, None, state)):
                                break
                    elif not args.headless and time.monotonic() - last_view > .25:
                        if show(hud.render(shown_frame, None, state)):
                            break
                    await asyncio.sleep(.003)
                    continue
                kind = item[0]
                if kind == 'end':
                    break
                if kind == 'start':
                    take = Take(record_dir)
                    plugin.reset()
                    hud.trails.clear()
                    state.update(mode='recording', last_toss=None, tosses=0, ignored=0, frames=0, history=[])
                    take_started = hud_due = None
                    print(json.dumps({'take': 'started', 'dir': str(take.dir)}), flush=True)
                    continue
                if kind == 'stop':
                    if take is not None:
                        summary = dict(tosses=state['history'], ignored_motions=state['ignored'],
                                       expected_max_rev_s=args.expected_max_rev_s)
                        directory = take.close(summary)
                        saved.append(str(directory))
                        state.update(mode='saved', saved=directory.name, saved_at=time.monotonic())
                        print(json.dumps({'take': 'saved', 'dir': str(directory), 'tosses': len(state['history'])}),
                              flush=True)
                        take = None
                    continue
                _, info, header, payload = item
                if take is None:
                    continue
                frame = decode(info, payload)
                if frame is None:
                    continue
                size = measured_size(frame)
                measured = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                seconds = info.seconds
                density = pixel_density(info, size[0], frame.shape[1])
                if args.target_px and target['norm'] is None:
                    target['norm'] = args.target_px / density
                if target['norm']:
                    # Keep the exemplar's size in this frame's pixels (crop and format change the scale).
                    wanted = min(4096., max(4., target['norm'] * density))
                    if target['applied'] is None or abs(wanted / target['applied'] - 1) > .05:
                        plugin.set_target(wanted)
                        target['applied'] = wanted
                    state['target_px'] = wanted
                else:
                    state['target_px'] = None
                if first is None:
                    first, base = seconds, time.time_ns() // 1_000_000 - 1000 - stats.get('span_ms', 0)
                stamp = base + round((seconds - first) * 1000)
                if last_stamp is not None and stamp <= last_stamp:
                    stamp = last_stamp + 1
                last_stamp = stamp
                if take_started is None:
                    take_started = hud_due = seconds
                await human.push_video(measured, timestamp_ms=stamp)
                value = plugin.latest
                hud.update(value)
                for event in value['events']:
                    if event['type'] != 'rotation.ended':
                        continue
                    take.events.append(event)
                    if is_toss(event, args.show_all, bool(target['norm'])):
                        toss = dict(turns=event['turns'], rev_s=event['rev_s'], duration_ms=event['duration_ms'],
                                    capture_fps=event['capture_fps'] or 0, trajectory=event.get('trajectory'),
                                    timestamp_ms=event['timestamp_ms'])
                        state['last_toss'] = toss
                        state['history'].append(toss)
                        state['tosses'] += 1
                    elif event['axis'] == 'flip':
                        state['ignored'] += 1
                while pending_alarms:
                    alarm = pending_alarms.pop(0)
                    for record in alarm['context']['records']:
                        if record['id'] != alarm['trigger_record_id']:
                            continue
                        for event in record['observation']['value']['events']:
                            if event['type'] == 'rotation.ended' and is_toss(event, args.show_all, bool(target['norm'])):
                                state['feed'].append(_toss_line(event))
                                print(json.dumps({'watch': watch['id'], 'rotation.ended': event}), flush=True)
                state['feed'] = state['feed'][-8:]
                state['frames'] += 1
                state['frame_size'] = (frame.shape[1], frame.shape[0])
                state['capture_fps'] = capture_fps()
                state['take_seconds'] = seconds - take_started
                state['lag_s'] = max(0., (stats['capture'][-1] if stats['capture'] else seconds) - seconds)
                if not recording_flag['on']:
                    state['mode'] = 'processing'
                take.add(header, payload, seconds, value, state['last_toss'])
                shown_frame = frame
                view.update(info=info, shape=frame.shape, frame=frame)
                if seconds >= hud_due:
                    # hud.mp4 advances on capture time, so it plays in real time however far GAMST lags.
                    image = hud.render(frame, value, state)
                    while seconds >= hud_due:
                        take.hud.write(image)
                        hud_due += 1 / 30
                    if time.monotonic() - last_view > 1 / 30 and show(image):
                        break
                await asyncio.sleep(0)
            if take is not None:
                directory = take.close(dict(tosses=state['history'], ignored_motions=state['ignored'],
                                            expected_max_rev_s=args.expected_max_rev_s))
                saved.append(str(directory))
            plugin.close()
    finally:
        if listener is not None:
            listener.close()
        if feeder is not None:
            feeder.cancel()
        if not args.headless:
            cv2.destroyAllWindows()
    print(json.dumps({'takes': saved}), flush=True)
    return saved


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'slowmo':
        render_slowmo(sys.argv[2])
    else:
        raise SystemExit('usage: python -m gamst.demos.rotation_live slowmo TAKE_DIR')
