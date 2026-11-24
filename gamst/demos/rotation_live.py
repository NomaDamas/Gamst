"""GAMST 240 live: an iPhone high-fps camera feeding GAMST, with a HUD and throw counting.

The phone streams continuously so the HUD can frame the shot; Start/Stop on the phone
marks a *take*. The HUD is a camera monitor: it shows the stream, never a live count (a
per-frame tracker cheap enough for 240 fps miscounted glinting coins). Each take is saved
under ``--record-dir``:

* ``raw.gmst``      the phone stream as received (replay with ``--iphone-replay``)
* ``hud.mp4``       the HUD at real time (30 fps, 1920x1080)
* ``summary.json``  capture statistics

With ``--refine LABEL`` a separate process then counts the take's throws of that named object
(:mod:`gamst.providers.rotation_refine`) and writes:

* ``refined.json``  every throw: turns, flips, rate, flight window, coverage and the per-frame trace
* ``refined.mp4``   each throw at 8x slow motion with the followed outline and the running count

The live process stores each throw as a timestamped ``vision.rotation.refined`` observation and
remembers the named object, so agents read the counts from GAMST context.
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
    """The 1920x1080 live view: camera monitor, take status and the counted throws."""

    def __init__(self):
        import numpy as np
        self.np = np
        self.text = Text()
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
        t.put(canvas, 'Halo  ·  GAMST 240  ·  moment capture', 28 + width + 20, 44, 22, MUTED)
        t.put(canvas, 'LAST THROW', 1456, 124, 18, MUTED, 'semibold')
        t.put(canvas, 'THIS SESSION', 1456, 560, 18, MUTED, 'semibold')
        t.put(canvas, 'GAMST CONTEXT   vision.rotation.refined  ->  timestamped throw records an agent can read',
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

    def video_layer(self, frame):
        """The frame letterboxed into the video rect."""
        import cv2
        np = self.np
        _, _, w, h = VIDEO
        layer = np.zeros((h, w, 3), np.uint8)
        fh, fw = frame.shape[:2]
        scale = min(w / fw, h / fh)
        dw, dh = int(fw * scale) // 2 * 2, int(fh * scale) // 2 * 2
        ox, oy = (w - dw) // 2, (h - dh) // 2
        layer[oy:oy + dh, ox:ox + dw] = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        return layer

    def render(self, frame, state):
        canvas = self.base.copy()
        x, y, w, h = VIDEO
        if frame is not None:
            canvas[y:y + h, x:x + w] = self.video_layer(frame)
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
        t.put(canvas, right, 1888, 34, 22, WHITE, 'semibold', anchor='right')
        t.put(canvas, f"countable up to {state['capture_fps'] / 4:.0f} rev/s flipping", 1888, 64, 18, MUTED, anchor='right')

    def _panel(self, canvas, state):
        t = self.text
        throw = state['last_throw']
        if throw is None:
            t.put(canvas, '0', 1450, 150, 150, LINE, 'bold')
            hint = ('counting the last take ...' if state['pending'] else
                    'REC, toss, STOP: counted after the take')
            t.put(canvas, hint, 1456, 340, 22, MUTED)
        else:
            width = t.put(canvas, f"{throw['flips']:.1f}", 1450, 150, 150, WHITE, 'bold')
            t.put(canvas, 'flips', 1456 + width + 8, 262, 30, MUTED, 'semibold')
            t.put(canvas, f"{throw['turns']:.1f} turns  ·  {throw['rev_s']:.1f} rev/s", 1456, 338, 34, ACCENT, 'semibold')
            t.put(canvas, f"{throw['duration_ms'] / 1000:.2f} s in the air  ·  seen {throw['coverage']:.0%}",
                  1456, 388, 22, MUTED)
            chip = (('THROW', GOOD) if throw['status'] == 'counting' and throw['trajectory'] == 'free_fall' else
                    ('PARTIAL  ·  LOWER BOUND', WARN) if throw['status'] == 'partial' else ('NOT A THROW', DIM))
            _pill(canvas, t, 1456, 430, chip[0], chip[1], size=18)
        t.put(canvas, f"{len(state['history'])} throws counted", 1456, 596, 30, WHITE, 'semibold')
        t.put(canvas, f"{state['frames']} frames this take", 1456, 640, 20, MUTED)
        for index, item in enumerate(state['history'][-6:][::-1]):
            line = f"{item['flips']:5.1f} flips  {item['rev_s']:5.1f} rev/s"
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
    """One Start→Stop take: the raw stream, a real-time HUD video and a summary.

    With ``refine`` arguments a separate process counts the take's throws afterwards
    (``refined.json``, ``refined.mp4``), so the analysis never competes with capture.
    """

    def __init__(self, root, refine=None):
        self.refine = refine
        stamp = time.strftime('%Y%m%d-%H%M%S')
        self.dir = Path(root) / f'take-{stamp}'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.raw = open(self.dir / 'raw.gmst', 'wb')
        self.hud = VideoSink(self.dir / 'hud.mp4', 30, CANVAS)
        self.frames, self.first, self.last = 0, None, None
        self.refine_process = None

    def add(self, header, payload, seconds):
        self.raw.write(header + payload)
        self.frames += 1
        self.first = seconds if self.first is None else self.first
        self.last = seconds

    def close(self):
        self.raw.close()
        self.hud.close()
        span = (self.last - self.first) if self.frames > 1 else 0
        summary = dict(frames=self.frames, seconds=round(span, 3),
                       capture_fps=round((self.frames - 1) / span, 1) if span else None)
        (self.dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        if self.refine:
            with open(self.dir / 'refine.log', 'w') as log:
                self.refine_process = subprocess.Popen(
                    [sys.executable, '-m', 'gamst.demos.rotation_live', 'refine', str(self.dir), *self.refine],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return self.dir


def render_refined(directory):
    """``refined.mp4``: each counted throw at 8x slow motion (every 240 fps frame at 30 fps).

    Shows the followed object's outline, its axis-ratio signal (one dip per flip) and GAMST's
    running count up to each frame. Replays the per-frame ellipses kept in ``refined.json``;
    no model runs.
    """
    import cv2
    import numpy as np
    from ..providers.rotation_refine import count_spins
    directory = Path(directory)
    output = json.loads((directory / 'refined.json').read_text())
    if not output['results']:
        return None
    recording = read_recording(directory / 'raw.gmst')
    stamps = [info.seconds * 1000 for info, _, _ in recording]
    text, sink = Text(), None
    width, height, view_h = 1600, 1000, 960
    for number, event in enumerate(output['results']):
        shapes = {row[0]: tuple(row[1:]) for row in event['shapes']}
        first, last = event['flight_frames']
        lo, hi = max(event['frames'][0], first - 36), min(event['frames'][1], last + 30)
        seen = [i for i in sorted(shapes) if first <= i <= last]
        edge = {i for a, b in zip(seen, seen[1:]) if 1 < b - a <= 7 for i in range(a + 1, b)}
        ratio = {i: shapes[i][3] / shapes[i][2] if i in shapes else 0. for i in seen + sorted(edge)}
        running = {}
        for i in range(first, last + 1):
            counted = count_spins(stamps, shapes, span=(first, i))
            running[i] = counted[0]['turns'] if counted and counted[0]['turns'] is not None else None
        shown = 0.
        for i in range(lo, hi + 1):
            info, _, payload = recording[i]
            frame = decode(info, payload)
            canvas = np.full((height, width, 3), BG, np.uint8)
            fh, fw = frame.shape[:2]
            scale = view_h / fh
            view = cv2.resize(frame, (int(fw * scale), view_h))
            flying = first <= i <= last
            if i in shapes:
                cx, cy, major, minor, angle = shapes[i]
                cv2.ellipse(view, (round(cx * scale), round(cy * scale)),
                            (max(1, round(major * scale / 2)), max(1, round(minor * scale / 2))), angle, 0, 360,
                            GOOD if flying else WARN, 2, cv2.LINE_AA)
            canvas[20:20 + view_h, 20:20 + view.shape[1]] = view
            x = 60 + view.shape[1]
            near = min(shapes, key=lambda k: abs(k - i))
            zx = int(max(0, min(fw - 110, shapes[near][0] - 55)))
            zy = int(max(0, min(fh - 110, shapes[near][1] - 55)))
            canvas[70:430, x:x + 360] = cv2.resize(frame[zy:zy + 110, zx:zx + 110], (360, 360),
                                                   interpolation=cv2.INTER_CUBIC)
            if flying and running.get(i) is not None:
                shown = running[i]
            final = i > last
            turns = event['turns'] if final else shown
            phase = 'caught  ·  final' if final else 'in the air  ·  counting' if flying else 'before release'
            text.put(canvas, f'GAMST  ·  "{event["label"]}"  ·  one named object', x, 24, 26, WHITE, 'semibold')
            text.put(canvas, f'throw {number + 1}/{len(output["results"])}  ·  frame {i}  ·  '
                     f't {stamps[i] - stamps[first]:+.0f} ms', x + 380, 80, 20, MUTED)
            text.put(canvas, '8x slow  ·  every captured frame', x + 380, 110, 20, MUTED)
            text.put(canvas, phase, x + 380, 170, 26, GOOD if flying else ACCENT, 'semibold')
            text.put(canvas, 'flips', x, 470, 30, MUTED, 'semibold')
            number_text = ('–' if turns is None else f'{2 * turns:.1f}') if final or not flying else (
                f'{2 * turns:.1f}' if turns else '…')
            text.put(canvas, number_text, x, 505, 120, WHITE, 'bold')
            if turns:
                text.put(canvas, f'= {turns:.1f} turns', x + 360, 560, 30, MUTED)
            if final:
                text.put(canvas, f"{event['rev_s'] or 0:.1f} rev/s  ·  {event['duration_ms']} ms in the air  ·  "
                         f"seen {event['coverage']:.0%}  ·  {event['status']}", x + 380, 230, 20, WHITE)
            g0, g1, top, bottom = x, width - 40, 700, 880
            cv2.rectangle(canvas, (g0, top), (g1, bottom), LINE, 1)
            px = lambda j: int(g0 + (j - lo) / max(1, hi - lo) * (g1 - g0))  # noqa: E731
            points = [(px(j), int(bottom - ratio[j] * (bottom - top))) for j in sorted(ratio) if j <= i]
            if len(points) > 1:
                cv2.polylines(canvas, [np.array(points, np.int32)], False, WARN, 2, cv2.LINE_AA)
            cv2.line(canvas, (px(i), top), (px(i), bottom), DIM, 1)
            cv2.line(canvas, (px(first), 920), (px(last), 920), GOOD, 6)
            text.put(canvas, 'outline minor/major (1 = face, 0 = edge)  ·  one dip = one flip', x, 668, 18, MUTED)
            if sink is None:
                sink = VideoSink(directory / 'refined.part.mp4', 30, (width, height))
            for _ in range(30 if i == hi else 1):
                sink.write(canvas)
    sink.close()
    (directory / 'refined.part.mp4').rename(directory / 'refined.mp4')
    return directory / 'refined.mp4'


async def refine_take(directory, *, label, models, device, efficienttam_source, worker_python,
                      detect_every=4, threshold=.2):
    """Re-count a saved take's throws of one named object; writes ``refined.json``.

    YOLOE (GAMST's pinned worker, text prompt ``label``) looks at every ``detect_every``-th frame
    in native-resolution tiles; its hits seed EfficientTAM, which follows that one object through
    each throw. See :mod:`gamst.providers.rotation_refine`.
    """
    import numpy as np
    from ..capabilities import SensorMetadata
    from ..plugins import InputFrame
    from ..providers.rotation_refine import EfficientTAMVideo, refine, tiles
    from .. import ProcessProvider
    from .rotation import REFINE_ASSETS, detector_config
    directory, models = Path(directory), Path(models)
    recording = read_recording(directory / 'raw.gmst')
    frames = [(info.seconds * 1000, (lambda info=info, payload=payload: decode(info, payload)))
              for info, _, payload in recording]
    worker = await ProcessProvider.open(worker_python, 'yoloe.local',
                                        detector_config(models, [label], device, threshold))
    detections = []
    try:
        for index in range(0, len(frames), detect_every):
            stamp, load = frames[index]
            image = load()
            if image is None:
                continue
            for number, (x, y) in enumerate(tiles(*image.shape[:2])):
                tile = np.ascontiguousarray(image[y:y + 640, x:x + 640])
                sensor = SensorMetadata(source_id='gamst.take', sensor_id=f'tile-{number}', sequence=index,
                                        capture_sequence=index, coordinate_frame='image', stream_epoch='refine')
                value = await worker.process(InputFrame(tile, 'video', int(stamp), subject_id='local-user',
                                                        sensor=sensor))
                h, w = tile.shape[:2]
                for item in value['objects']:
                    box = item['bbox']
                    detections.append((stamp, index, item['score'], (
                        x + box['x'] * w, y + box['y'] * h,
                        x + (box['x'] + box['width']) * w, y + (box['y'] + box['height']) * h)))
    finally:
        await worker.close()
    name = 'efficienttam_ti_512x512.pt'
    segmenter = EfficientTAMVideo(efficienttam_source, models / name, REFINE_ASSETS[name][1], device=device)
    results = refine(frames, detections, segmenter, label=label)
    output = dict(label=label, detections=len(detections), results=results)
    (directory / 'refined.part.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    (directory / 'refined.part.json').rename(directory / 'refined.json')
    render_refined(directory)
    return output


# ---------------------------------------------------------------- run


async def run(args):
    import cv2
    from .. import Gamst

    record_dir = Path(args.record_dir or 'gamst-takes')
    # Every take frame, in order, goes to raw.gmst. A socket thread fills the queue so a busy loop
    # never stops reading and back-pressures the phone into dropping frames.
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
        elif name == 'crop':
            clear_crop()

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

    hud = Hud()
    title = 'GAMST 240'
    buttons = []  # (name, x, y, width, height) in HUD pixels; drawn on the window only, never recorded
    drag = dict(start=None, now=None, button=None)
    view = dict(info=None, shape=None)  # header and size of the frame on screen, to map clicks

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

    def crop_to_box(a, b):
        """Move the phone's native-resolution strip over the user's box: full frame height (the
        object flies up), narrow enough to keep 240 fps."""
        info = view['info']
        if info is None or a is None or b is None:
            return
        ux, _ = frame_to_upright(info, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        left = min(max(0., ux - .2), .6)
        send(dict(cmd='crop', rect=[round(v, 4) for v in upright_rect_to_sensor((left, 0., .4, 1.), info.angle)]))

    def clear_crop():
        send(dict(cmd='crop', rect=list(FULL)))

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
                    crop_to_box(to_frame(*drag['start']), to_frame(x, y))
                drag.update(start=None, now=None, button=None)

        cv2.setMouseCallback(title, on_mouse)
    state = dict(mode='waiting', take_seconds=0., saved='', capture_fps=0., frame_size=None,
                 last_throw=None, frames=0, history=[], feed=[], pending=False)
    take = None
    first = base = None
    hud_due = take_started = None
    last_view = 0.
    shown_frame = None
    saved = []

    def capture_fps():
        window = stats['capture']
        return (len(window) - 1) / (window[-1] - window[0]) if len(window) > 1 and window[-1] > window[0] else 0.

    def controls(image):
        """Phone controls for the Mac window (space/r: record, c: camera, z: zoom, t: full frame, q: quit)."""
        import cv2
        buttons.clear()
        connected = phone['sock'] is not None
        items = [('record', '■  STOP' if phone['recording'] else '●  REC', REC if connected else DIM),
                 ('camera', f"camera  {'front' if phone['front'] else 'back'}", PANEL if connected else DIM),
                 ('zoom', f"zoom  {phone['zoom']:g}x", PANEL if connected else DIM),
                 ('crop', 'strip  ✕' if tuple(phone['crop']) != FULL else 'strip: drag a box',
                  GOOD if tuple(phone['crop']) != FULL else PANEL)]
        widths = [hud.text.patch(label, 24, WHITE, 'semibold')[0].shape[1] + 36 for _, label, _ in items]
        x = VIDEO[0] + (VIDEO[2] - sum(widths) - 16 * (len(items) - 1)) // 2
        y = VIDEO[1] + VIDEO[3] - 72
        for (name, label, color), width in zip(items, widths):
            _pill(image, hud.text, x, y, label, color, size=24)
            buttons.append((name, x, y, width, 44))
            x += width + 16
        hint = ('drag a box: the phone streams that strip at native resolution   ·   space  record   ·   '
                'c  camera   ·   z  zoom   ·   t  full frame   ·   q  quit'
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
            clear_crop()
        return key in (27, ord('q'))

    refine_args = None if not getattr(args, 'refine', None) else [
        '--label', args.refine, '--efficienttam-source', args.efficienttam_source, '--models', args.models,
        '--device', args.device, '--worker-python', args.worker_python]
    pending_refine = []  # saved takes whose refine.json has not been published yet

    async def collect_refined(human):
        """Publish finished take refinements: registered observations + named objects in ObjectMemory."""
        from ..providers.rotation_refine import publish
        for item in list(pending_refine):
            path = item['dir'] / 'refined.json'
            if not path.exists():
                if item['process'].poll() is not None:
                    pending_refine.remove(item)
                    print(json.dumps({'refine_failed': str(item['dir']), 'log': str(item['dir'] / 'refine.log')}),
                          flush=True)
                continue
            pending_refine.remove(item)
            output = json.loads(path.read_text())
            records = await publish(human, output['results'], take_id=item['dir'].name,
                                    clock=lambda ms, item=item: item['base'] + round((ms / 1000 - item['first']) * 1000))
            for event, record in zip(output['results'], records):
                attributes = record['observation']['value']['objects'][0]['attributes']
                throw = dict(attributes, trajectory=attributes['trajectory'])
                if throw['turns'] is not None:
                    state['history'].append(throw)
                    state['last_throw'] = throw
                clock = time.strftime('%H:%M:%S', time.localtime(attributes['ended_ms'] / 1000))
                state['feed'].append(f"{clock}  {event['label']}  {throw['flips'] or 0:.1f} flips  "
                                     f"{throw['rev_s'] or 0:.1f} rev/s  {throw['status']}  seen {throw['coverage']:.0%}")
            state['feed'] = state['feed'][-8:]
            state['pending'] = bool(pending_refine)
            objects = await human.find_objects(text=output['label'])
            print(json.dumps({'refined': str(item['dir']), 'records': [record['id'] for record in records],
                              'objects': [dict(entity_id=entry['entity_id'], label=entry['label'],
                                               track_id=entry['track_id']) for entry in objects]}), flush=True)

    try:
        async with Gamst(consent=True, store=args.store) as human:
            while True:
                try:
                    item = take_queue.get_nowait()
                except queue.Empty:
                    item = None
                if item is None:
                    # Standby: show the newest live frame at ~30 fps.
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
                            view.update(info=info, shape=frame.shape)
                            if show(hud.render(frame, state)):
                                break
                    elif not args.headless and time.monotonic() - last_view > .25:
                        if show(hud.render(shown_frame, state)):
                            break
                    if pending_refine:
                        await collect_refined(human)
                    await asyncio.sleep(.003)
                    continue
                kind = item[0]
                if kind == 'end':
                    break
                if kind == 'start':
                    take = Take(record_dir, refine_args)
                    state.update(mode='recording', frames=0)
                    take_started = hud_due = None
                    print(json.dumps({'take': 'started', 'dir': str(take.dir)}), flush=True)
                    continue
                if kind == 'stop':
                    if take is not None:
                        directory = take.close()
                        saved.append(str(directory))
                        if refine_args:
                            pending_refine.append(dict(dir=directory, first=first, base=base,
                                                       process=take.refine_process))
                        state.update(mode='saved', saved=directory.name, saved_at=time.monotonic(),
                                     pending=bool(pending_refine))
                        print(json.dumps({'take': 'saved', 'dir': str(directory), 'frames': take.frames}), flush=True)
                        take = None
                    continue
                _, info, header, payload = item
                if take is None:
                    continue
                seconds = info.seconds
                if first is None:
                    # Capture time -> runtime clock, so refined throws land at the moment they happened.
                    first, base = seconds, time.time_ns() // 1_000_000 - 1000 - stats.get('span_ms', 0)
                if take_started is None:
                    take_started = hud_due = seconds
                take.add(header, payload, seconds)
                state['frames'] += 1
                state['capture_fps'] = capture_fps()
                state['take_seconds'] = seconds - take_started
                # Only frames that reach the screen or hud.mp4 are decoded; raw.gmst keeps every frame.
                live = not args.headless and time.monotonic() - last_view > 1 / 30
                if seconds < hud_due and not live:
                    continue
                frame = decode(info, payload)
                if frame is None:
                    continue
                state['frame_size'] = (frame.shape[1], frame.shape[0])
                shown_frame = frame
                view.update(info=info, shape=frame.shape)
                image = hud.render(frame, state)
                while seconds >= hud_due:
                    # hud.mp4 advances on capture time, so it plays in real time.
                    take.hud.write(image)
                    hud_due += 1 / 30
                if live and show(image):
                    break
                await asyncio.sleep(0)
            if take is not None:
                directory = take.close()
                saved.append(str(directory))
                if refine_args:
                    pending_refine.append(dict(dir=directory, first=first, base=base, process=take.refine_process))
            if args.iphone_replay:
                # A replay ends with its take: wait for the refined count so it reaches GAMST too.
                while pending_refine:
                    await collect_refined(human)
                    await asyncio.sleep(.5)
    finally:
        if listener is not None:
            listener.close()
        if feeder is not None:
            feeder.cancel()
        if not args.headless:
            cv2.destroyAllWindows()
    print(json.dumps({'takes': saved}), flush=True)
    return saved


def _refine_main(argv):
    import argparse
    from .rotation import DEFAULT_MODELS
    parser = argparse.ArgumentParser(prog='python -m gamst.demos.rotation_live refine')
    parser.add_argument('take_dir')
    parser.add_argument('--label', required=True, help='open-vocabulary text for the object to follow')
    parser.add_argument('--efficienttam-source', required=True)
    parser.add_argument('--models', default=str(DEFAULT_MODELS))
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--worker-python', default=sys.executable)
    args = parser.parse_args(argv)
    output = asyncio.run(refine_take(args.take_dir, label=args.label, models=args.models, device=args.device,
                                     efficienttam_source=args.efficienttam_source,
                                     worker_python=args.worker_python))
    print(json.dumps(dict(detections=output['detections'], throws=[
        {key: event.get(key) for key in ('track_id', 'turns', 'rev_s', 'status', 'coverage', 'flight_frames')}
        | dict(trajectory=event['trajectory'].get('kind')) for event in output['results']])))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'render':
        render_refined(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == 'refine':
        _refine_main(sys.argv[2:])
    else:
        raise SystemExit('usage: python -m gamst.demos.rotation_live refine TAKE_DIR --label ... | render TAKE_DIR')
