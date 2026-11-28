"""The Halo coin HUD: the GAMST 240 app's screen for the coin-flip recipe, as a view module.

A view is application UI, outside GAMST's context boundary (:mod:`gamst.views`):

- **input**: keys and mouse become commands. REC, camera, zoom and the strip crop go to the phone
  (:meth:`gamst.inputs.IPhone240Input.command`), exactly like its own buttons; ANALYZE goes to the
  counting module (:meth:`gamst.takes.TakeFlips.analyze`); q quits.
- **output**: the 1920x1080 camera monitor, the take status and the counted throws. Throws are
  read back from the counting module's committed ``vision.rotation.refined`` records, the same
  records an agent reads over MCP; the HUD never shows a live count and measures nothing.

It also records its screen into each take's ``hud.mp4`` at real time (capture clock), with or
without a window. :func:`render_refined` draws ``refined.mp4``, each counted throw at 8x slow
motion, from ``refined.json``; no model runs.
"""
from __future__ import annotations

import asyncio
import collections
import json
from pathlib import Path
import time

from .inputs import FULL


# ---------------------------------------------------------------- rendering

# ---------------------------------------------------------------- rendering

BG, PANEL, LINE = (18, 14, 11), (34, 28, 23), (60, 52, 44)
WHITE, MUTED, DIM = (240, 240, 240), (160, 152, 146), (100, 94, 90)
ACCENT, GOOD, WARN, REC = (255, 196, 64), (120, 214, 96), (60, 180, 255), (70, 70, 235)
CANVAS = (1920, 1080)
VIDEO = (32, 104, 1376, 774)  # x, y, width, height (16:9)
FONT_FILES = {'sans': '/System/Library/Fonts/SFNS.ttf', 'mono': '/System/Library/Fonts/SFNSMono.ttf'}
# Linux (e.g. a cloud VM rendering a HUD): DejaVu, used when the macOS files are not there
LINUX_FONT_FILES = {'sans': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    'mono': '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'}


def font_file(family):
    import os
    path = FONT_FILES[family]
    return path if os.path.exists(path) else LINUX_FONT_FILES[family]


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
                font = self._font_cls.truetype(font_file(family), size)
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
        logo = Image.open(Path(__file__).parent / 'demos' / 'assets' / 'gamst-logo.png').convert('RGBA')
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
        if state['pending']:
            # The audience watches the wait: what GAMST is doing, on how many frames, how far along.
            progress = state.get('progress') or {}
            stage = dict(finding='finding the coin', tracking='following it through the flight',
                         rendering='drawing the slow motion').get(progress.get('stage'), 'starting')
            t.put(canvas, 'GAMST is counting', 1456, 150, 40, WHITE, 'bold')
            t.put(canvas, f"{state['take_frames']:,} frames at {state['take_fps']:.0f} fps", 1456, 210, 26, ACCENT, 'semibold')
            t.put(canvas, stage, 1456, 262, 22, MUTED)
            done, total = progress.get('done', 0), max(1, progress.get('total', 1))
            import cv2
            cv2.rectangle(canvas, (1456, 302), (1856, 316), LINE, -1)
            cv2.rectangle(canvas, (1456, 302), (1456 + int(400 * min(1., done / total)), 316), ACCENT, -1)
            t.put(canvas, f"{state['pending']:.0f} s", 1456, 336, 22, MUTED)
        elif throw is None:
            t.put(canvas, '0', 1450, 150, 150, LINE, 'bold')
            t.put(canvas, 'REC, toss, STOP, ANALYZE', 1456, 340, 22, MUTED)
        else:
            width = t.put(canvas, f"{throw['flips']:.1f}", 1450, 150, 150, WHITE, 'bold')
            t.put(canvas, 'flips', 1456 + width + 8, 262, 30, MUTED, 'semibold')
            t.put(canvas, f"{throw['turns']:.1f} turns  ·  {throw['rev_s']:.1f} rev/s", 1456, 338, 34, ACCENT, 'semibold')
            t.put(canvas, f"{throw['duration_ms'] / 1000:.2f} s in the air  ·  seen {throw['coverage']:.0%}",
                  1456, 388, 22, MUTED)
            chip = (('THROW', GOOD) if throw['status'] == 'counting' and throw['trajectory'] == 'free_fall' else
                    ('PARTIAL  ·  LOWER BOUND', WARN) if throw['status'] == 'partial' else ('NOT A THROW', DIM))
            _pill(canvas, t, 1456, 430, chip[0], chip[1], size=18)
        count = len(state['history'])
        t.put(canvas, f"{count} throw{'' if count == 1 else 's'} counted", 1456, 596, 30, WHITE, 'semibold')
        t.put(canvas, f"{state['frames']} frames this take", 1456, 640, 20, MUTED)
        for index, item in enumerate(state['history'][-6:][::-1]):
            line = f"{item['flips']:5.1f} flips  {item['rev_s']:5.1f} rev/s"
            t.put(canvas, line, 1456, 690 + index * 30, 22, WHITE if index == 0 else MUTED, family='mono')

    def _feed(self, canvas, state):
        for index, line in enumerate(state['feed'][-4:]):
            self.text.put(canvas, line, 48, 950 + index * 24, 17, WHITE if index == len(state['feed'][-4:]) - 1 else MUTED,
                          family='mono')


def render_refined(directory):
    """``refined.mp4``: each counted throw at 8x slow motion (every 240 fps frame at 30 fps).

    Shows the followed object's outline, its axis-ratio signal (one dip per flip) and GAMST's
    running count up to each frame. Replays the per-frame ellipses kept in ``refined.json``;
    no model runs.
    """
    import cv2
    import numpy as np
    from .inputs import decode, read_recording
    from .providers.rotation_refine import count_spins
    from .takes import VideoSink
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



# ---------------------------------------------------------------- the view module

class HaloCoinView:
    """The view of a Pipeline with an :class:`IPhone240Input`, a :class:`TakeRecorder` and, to count,
    a :class:`TakeFlips`. ``update`` is the Pipeline's ``on_frame``; ``idle`` keeps the window alive
    while no frame arrives (phone not connected, counting after the take)."""
    title = 'GAMST 240'

    def __init__(self, pipeline, *, headless=False, capabilities=()):
        from .inputs import IPhone240Input
        from .takes import TakeFlips, TakeRecorder
        self.pipeline, self.headless = pipeline, headless
        sources = [entry['source'] for entry in pipeline.inputs.values()]
        self.source = next((item for item in sources if isinstance(item, IPhone240Input)), None)
        if self.source is None:
            raise ValueError('the halo-coin view needs a gamst240 input')
        modules = [entry['module'] for entry in pipeline.modules.values()]
        self.recorder = next((item for item in modules if isinstance(item, TakeRecorder)), None)
        self.counter = next((item for item in modules if isinstance(item, TakeFlips)), None)
        if self.recorder is not None:
            self.recorder.hud_size = CANVAS
        self.hud = Hud()
        self.buttons = []  # (name, x, y, width, height) in HUD pixels; drawn on the window only, never recorded
        self.drag = dict(start=None, now=None, button=None)
        self.view = dict(info=None, shape=None)  # header and size of the frame on screen, to map clicks
        self.shown, self.frame_size, self.seconds, self.frames = None, None, None, 0
        self.last_view, self.quit = 0., False
        self.saved_seen, self.saved_at, self.saved_name = 0, None, ''
        self.seen, self.history, self.last_throw, self.feed = 0, [], None, []
        if not headless:
            import cv2
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, 1440, 810)
            cv2.moveWindow(self.title, 40, 60)
            cv2.setMouseCallback(self.title, self._mouse)

    # ------------------------------------------------------------ output: state read back from modules and records

    def _throws(self):
        """Newly committed vision.rotation.refined records -> the panel's throws and the context feed."""
        records = self.counter.records if self.counter is not None else []
        for record in records[self.seen:]:
            item = record['observation']['value']['objects'][0]
            throw = dict(item['attributes'])
            if throw['turns'] is not None and throw['trajectory'] == 'free_fall':
                self.history.append(throw)
                self.last_throw = throw
            clock = time.strftime('%H:%M:%S', time.localtime(throw['ended_ms'] / 1000))
            self.feed.append(f"{clock}  {item['label']}  {throw['flips'] or 0:.1f} flips  "
                             f"{throw['rev_s'] or 0:.1f} rev/s  {throw['status']}  seen {throw['coverage']:.0%}")
        self.seen = len(records)
        self.feed = self.feed[-8:]

    def state(self):
        self._throws()
        phone = self.source.state()
        take = self.recorder.take if self.recorder is not None else None
        if self.recorder is not None and len(self.recorder.saved) > self.saved_seen:
            self.saved_seen, self.saved_at = len(self.recorder.saved), time.monotonic()
            self.saved_name = self.recorder.saved[-1].name
        if take is not None:
            mode = 'recording'
            self.frames = take.frames
        elif self.saved_at is not None and time.monotonic() - self.saved_at <= 4:
            mode = 'saved'
        else:
            mode = 'standby' if self.source.received else 'waiting'
        counting = self.counter.status() if self.counter is not None else {}
        return dict(mode=mode, take_seconds=(self.seconds - take.first) if take is not None and take.first is not None
                    and self.seconds is not None else 0., saved=self.saved_name, capture_fps=phone['capture_fps'],
                    frame_size=self.frame_size, last_throw=self.last_throw, frames=self.frames, history=self.history,
                    feed=self.feed, pending=counting.get('pending_s', 0.), progress=counting.get('progress'),
                    take_frames=counting.get('take_frames', 0), take_fps=counting.get('take_fps', 0.))

    async def update(self, input_id, frame, records):
        """on_frame: draw the frames that reach the screen (~30 fps) or hud.mp4; raw.gmst keeps every frame."""
        from .inputs import PhoneFrame
        item = frame.data
        if not isinstance(item, PhoneFrame):
            return not self.quit
        take = self.recorder.take if self.recorder is not None else None
        recorded = take is not None and item.recording
        seconds = item.info.seconds
        if recorded:
            self.seconds = seconds
        live = not self.headless and time.monotonic() - self.last_view > 1 / 30
        if not live and not (recorded and take.hud_wanted(seconds)):
            return not self.quit
        image = item.image()
        if image is None:
            return not self.quit
        self.frame_size = (image.shape[1], image.shape[0])
        self.shown = image
        self.view.update(info=item.info, shape=image.shape)
        canvas = self.hud.render(image, self.state())
        if recorded:
            take.write_hud(canvas, seconds)
        if live:
            self._show(canvas)
        return not self.quit

    async def idle(self):
        """Keep the window drawn and the keys read while no frame arrives."""
        while not self.quit and not self.headless:
            if time.monotonic() - self.last_view > .25:
                self._show(self.hud.render(self.shown, self.state()))
            await asyncio.sleep(.03)

    # ------------------------------------------------------------ input: keys, buttons, drag -> commands

    def command(self, name):
        if name == 'analyze':
            if self.counter is not None:
                self.counter.analyze()
        elif name == 'crop':
            self.source.command('crop')
        else:
            self.source.command(name)

    def _to_frame(self, x, y):
        """HUD pixel -> normalized point in the frame on screen (None outside the picture)."""
        if self.view['shape'] is None:
            return None
        fh, fw = self.view['shape'][:2]
        scale = min(VIDEO[2] / fw, VIDEO[3] / fh)
        dw, dh = int(fw * scale) // 2 * 2, int(fh * scale) // 2 * 2
        fx = (x - VIDEO[0] - (VIDEO[2] - dw) // 2) / dw
        fy = (y - VIDEO[1] - (VIDEO[3] - dh) // 2) / dh
        return (fx, fy) if 0 <= fx <= 1 and 0 <= fy <= 1 else None

    def _crop_to_box(self, a, b):
        """The phone streams a native-resolution strip over the dragged box (IPhone240Input.command)."""
        info = self.view['info']
        if info is None or a is None or b is None:
            return
        self.source.command('crop', around=(info, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2))

    def _mouse(self, event, x, y, flags, param):
        import cv2
        drag = self.drag
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.update(start=None, now=None, button=None)
            for name, bx, by, bw, bh in self.buttons:
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    drag['button'] = name
                    return
            if self._to_frame(x, y) is not None:
                drag.update(start=(x, y), now=(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drag['start'] is not None:
            drag['now'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if drag['button'] is not None:
                self.command(drag['button'])
            elif drag['start'] is not None and abs(x - drag['start'][0]) + abs(y - drag['start'][1]) > 6:
                self._crop_to_box(self._to_frame(*drag['start']), self._to_frame(x, y))
            drag.update(start=None, now=None, button=None)

    def _controls(self, image, state):
        """Phone controls for the Mac window (space/r: record, c: camera, z: zoom, t: full frame, q: quit)."""
        import cv2
        hud, phone = self.hud, self.source.state()
        self.buttons.clear()
        connected = phone['connected']
        items = [('record', '■  STOP' if phone['recording'] else '●  REC', REC if connected else DIM),
                 ('camera', f"camera  {'front' if phone['front'] else 'back'}", PANEL if connected else DIM),
                 ('zoom', f"zoom  {phone['zoom']:g}x", PANEL if connected else DIM),
                 ('crop', 'strip  ✕' if phone['crop'] != FULL else 'strip: drag a box',
                  GOOD if phone['crop'] != FULL else PANEL)]
        if self.counter is not None:
            # Counting is not live: it runs on the last saved take when asked (or on Stop).
            ready = self.counter.status()['ready']
            label = (f"COUNTING  {state['pending']:.0f} s" if state['pending'] else
                     'ANALYZE  ▶' if ready else 'ANALYZE')
            items.append(('analyze', label, ACCENT if ready or state['pending'] else DIM))
        widths = [hud.text.patch(label, 24, WHITE, 'semibold')[0].shape[1] + 36 for _, label, _ in items]
        x = VIDEO[0] + (VIDEO[2] - sum(widths) - 16 * (len(items) - 1)) // 2
        y = VIDEO[1] + VIDEO[3] - 72
        for (name, label, color), width in zip(items, widths):
            _pill(image, hud.text, x, y, label, color, size=24)
            self.buttons.append((name, x, y, width, 44))
            x += width + 16
        hint = ('drag a box: the phone streams that strip at native resolution   ·   space  record   ·   '
                'c  camera   ·   z  zoom   ·   t  full frame   ·   a  analyze   ·   q  quit'
                if connected else 'phone not connected')
        hud.text.put(image, hint, VIDEO[0] + VIDEO[2] // 2, y + 54, 16, MUTED, anchor='center')
        if self.drag['start'] is not None and self.drag['now'] is not None:
            cv2.rectangle(image, self.drag['start'], self.drag['now'], GOOD, 2, cv2.LINE_AA)
        return image

    def _show(self, image):
        if self.headless:
            return
        import cv2
        shown = self._controls(image.copy(), self.state()) if self.source.live else image
        cv2.imshow(self.title, shown)
        self.last_view = time.monotonic()
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(' '), ord('r')):
            self.command('record')
        elif key == ord('c'):
            self.command('camera')
        elif key == ord('z'):
            self.command('zoom')
        elif key == ord('t'):
            self.command('crop')
        elif key == ord('a'):
            self.command('analyze')
        elif key in (27, ord('q')):
            self.quit = True
            self.source.stop()

    def close(self):
        if not self.headless:
            import cv2
            cv2.destroyAllWindows()
