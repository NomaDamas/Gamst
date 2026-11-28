"""View modules: the application's screen for a running recipe (the HUD).

A view is application UI, outside GAMST's context boundary. It does two things:

- **input**: keys (and, in richer apps, buttons) become commands to the Pipeline: quit, reset the
  modules, switch a module on or off;
- **output**: it draws the camera frame with values read back from committed records, the same
  records an agent reads over MCP. A view never computes a measurement itself.

``gamst recipe run`` picks the view a recipe names. ``--headless`` replaces it with JSON lines.
The GAMST 240 HUD of the coin-flip recipe (``halo-coin``) is :class:`gamst.halo.HaloCoinView`.
"""
from __future__ import annotations

import json
import textwrap


class _Window:
    title = 'GAMST'

    def __init__(self, pipeline, *, headless=False, capabilities=()):
        self.pipeline, self.headless = pipeline, headless
        self.capabilities = tuple(capabilities)
        self.latest = {}
        self.printed, self.windows = set(), {}
        self.out = None                 # where headless JSON lines go (None: stdout)

    async def _refresh(self, records):
        for record in records:
            capability = ((record.get('observation') or {}).get('capability') or {}).get('id')
            if capability in self.capabilities:
                self.latest[capability] = record
                events = record['observation']['value'].get('events') if isinstance(
                    record['observation']['value'], dict) else None
                if events and record['id'] not in self.printed:
                    self.printed.add(record['id'])
                    print(json.dumps(dict(capability=capability, timestamp_ms=record['timestamp_ms'],
                                          events=events), ensure_ascii=False), file=self.out, flush=True)

    def _title(self, input_id):
        """One window per video input; the first has the plain title."""
        return self.windows.setdefault(input_id, f'{self.title}: {input_id}' if self.windows else self.title)

    def _keys(self):
        import cv2
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            return False
        if key == ord('r'):
            for name in self.pipeline.modules:
                self.pipeline.disable(name)
                self.pipeline.enable(name)
        return True

    def close(self):
        if not self.headless:
            import cv2
            cv2.destroyAllWindows()


# The live HUD's states, in the colours of the video HUD (gamst.count_video, BGR).
STATE_COLOR = {'heard': (200, 222, 236), 'provisional': (40, 170, 255), 'final': (110, 220, 120),
               'revoked': (80, 80, 240), 'uncertain': (230, 130, 190)}
KEPT_RECORDS = 4096


def event_line(event):
    """One readable line of a committed event or revision, e.g. ``-> revoked  뽀또  (other_speaker)``."""
    revision = event.get('event') == 'event.revision'
    state = event.get('state') or ('final' if not revision else '?')
    what = event.get('term') or event.get('label') or event.get('event', '')
    parts = [('-> ' if revision else '+ ') + state, str(what)]
    if event.get('event') == 'term.other_voice':
        parts[0] = '  other voice'
    if type(event.get('speaker_score')) in (int, float):
        parts.append(f"voice {event['speaker_score']:.2f}")
    if event.get('candidate'):
        parts.append('candidate')
    if event.get('reason'):
        parts.append(f"({event['reason']})")
    return '  '.join(parts)


class CountView(_Window):
    """Any recipe that counts: the frame (or, with sound only, a panel of its own), then per declared capability its
    count, how many events are heard / provisional / final / revoked / uncertain (the revision states of
    :mod:`gamst.counting`, in the video HUD's colours), and the newest events and corrections as they are decided."""
    title = 'GAMST recipe'
    every_ms = 100      # a panel without video is redrawn at most this often

    def __init__(self, pipeline, *, headless=False, capabilities=()):
        super().__init__(pipeline, headless=headless, capabilities=capabilities)
        self.rows, self.recent, self._drawn_ms, self._text = {}, {}, None, None

    def _put(self, panel, line, x, y, color):
        """One line with its baseline at ``y``; returns its width. The video HUD's font (Korean words too) when Pillow is
        there, else OpenCV's (ASCII only)."""
        if self._text is None:
            try:
                from .count_video import KoreanText
                self._text = KoreanText()
            except ImportError:
                self._text = False
        if self._text:
            return self._text.put(panel, line, x, y - 14, 17, color)
        import cv2
        line = line.encode('ascii', 'replace').decode()
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1)
        return cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)[0][0]

    async def _refresh(self, records):
        await super()._refresh(records)
        for record in records:
            capability = ((record.get('observation') or {}).get('capability') or {}).get('id')
            value = (record.get('observation') or {}).get('value')
            events = value.get('events') if isinstance(value, dict) else None
            if capability not in self.capabilities or not events:
                continue
            kept = self.rows.setdefault(capability, [])     # only records with events: what the states are made of
            kept.append(record)
            del kept[:-KEPT_RECORDS]
            recent = self.recent.setdefault(capability, [])
            recent += [event for event in events or () if isinstance(event, dict)]
            del recent[:-6]

    def _video(self):
        return any(item['modality'] == 'video' for item in self.pipeline.inputs.values())

    async def update(self, input_id, frame, records):
        await self._refresh(records)
        if self.headless:
            return True
        if frame.modality != 'video':
            if self._video():
                return True       # drawn with the video frames
            if self._drawn_ms is not None and frame.timestamp_ms - self._drawn_ms < self.every_ms:
                return True
            self._drawn_ms = frame.timestamp_ms
        import cv2
        import numpy as np
        from .counting import STATES, count_events
        if frame.modality == 'video':
            image = frame.data
        else:
            image = np.full((540, 480, 3), 16, np.uint8)
            self._put(image, 'sound only: no camera', 20, 40, (150, 150, 150))
        panel = np.full((max(image.shape[0], 540), 560, 3), 24, np.uint8)
        if image.shape[0] < panel.shape[0]:
            image = np.vstack([image, np.full((panel.shape[0] - image.shape[0], image.shape[1], 3), 16, np.uint8)])
        mode = getattr(self.pipeline, 'mode', 'count')
        lines = [(f'GAMST  {self.pipeline.name}   mode: {mode}', (80, 220, 255)), ('', None)]
        if mode == 'search':
            lines.insert(1, ('SEARCH: unclear words count as candidates', STATE_COLOR['provisional']))
        spill = (getattr(self.pipeline, 'notes', None) or {}).get('video_memory')
        if spill:                         # the spill file was bounded by free disk (gamst/providers/video_memory.py)
            lines.insert(1, ('SPILL LIMITED: ' + spill[:70], STATE_COLOR['revoked']))
        choice = (getattr(self.pipeline, 'notes', None) or {}).get('pose_choice')
        if isinstance(choice, dict):      # which pose model counts, as measured at start (recipes.choose_pose)
            kept = choice['chosen'] == choice['model']
            lines.insert(1, (f"pose {choice['chosen']}: {choice['live_fps']:g}+{choice['l2_fps']:g} fps measured, "
                             f"input {choice['input_fps']:g}" + ('' if kept else f" ({choice['model']} too slow)"),
                             (150, 150, 150) if kept else STATE_COLOR['revoked']))
        refused = (getattr(self.pipeline, 'notes', None) or {}).get('who_refused')
        if refused:     # an instruction naming a person was refused (gamst/person_reference.py, same words as CLI and MCP)
            lines += [(refused[:110], STATE_COLOR['revoked']), ('', None)]
        quality = getattr(self.pipeline, 'voice_quality', None)
        if isinstance(quality, dict) and quality.get('low'):      # gamst/voice_quality.py (ASCII: cv2 text)
            lines += [(f"VOICE REFERENCE QUALITY LOW {quality['quality']['mean']:.2f} -> {quality['used']}",
                       STATE_COLOR['revoked']), ('', None)]
        tallies = []
        for capability in self.capabilities:
            record = self.latest.get(capability)
            value = record['observation']['value'] if record else {}
            lines.append((capability, None))
            lines.append((f"  count {value.get('count', 0)}   {value.get('status', '' if record else 'waiting')}",
                          (255, 255, 255)))
            states = count_events(self.rows.get(capability, ()), max_evidence=0).get('states') or {}
            if any(states.values()):
                tallies.append((len(lines), [(name, states.get(name, 0)) for name in STATES]))
                lines.append(('', None))
            if 'distance' in value:
                lines.append((f"  distance {value['distance']:.3f}", None))
            for event in reversed(self.recent.get(capability, ())[-4:]):      # newest first
                state = event.get('state') or 'final'
                late = 'LATE ' if event.get('correction') else ''     # an L2 correction, counted after the live lane
                lines.append(('  ' + late + event_line(event)[:60], STATE_COLOR.get(state)))
            lines.append(('', None))
        lost = [self.pipeline.lost(capability) for capability in self.capabilities]
        if any(item['gaps'] for item in lost):
            lines += [(f"INCOMPLETE: {max(item['frames'] for item in lost)} frames lost", STATE_COLOR['revoked']),
                      ('', None)]
        status = self.pipeline.status()['modules']
        lines.append(('modules (r: reset, q: quit)', None))
        lines += [(f"  {name}: {'on' if item['enabled'] else 'off'}  {item['last_latency_ms'] or 0:.0f} ms", None)
                  for name, item in status.items()]
        rows = max(1, panel.shape[0] // 22 - 1)
        for index, (line, color) in enumerate(lines[:rows]):
            if line:
                self._put(panel, line, 10, 26 + 22 * index, color or (255, 255, 255))
        for index, tally in tallies:
            if index >= rows:
                continue
            x = 22
            for name, number in tally:
                x += self._put(panel, f'{name} {number}', x, 26 + 22 * index, STATE_COLOR[name]) + 14
        cv2.imshow(self._title(input_id if frame.modality == 'video' else 'sound'), np.hstack([image, panel]))
        return self._keys()


class RotationView(_Window):
    """The rotation panel: tracked regions, turns and rev/s from ``vision.rotation`` records."""
    title = 'GAMST rotation'

    async def update(self, input_id, frame, records):
        # The rotation module commits its own records, so read the newest one back.
        rows = await self.pipeline.human.get_observations(capability_id='vision.rotation', window='1h', limit=1)
        record = rows['items'][0] if rows['items'] else None
        await self._refresh([record] if record else [])
        if self.headless or frame.modality != 'video':
            return True
        import cv2
        from .demos.rotation import Panel
        cv2.imshow(self._title(input_id), Panel().draw(frame.data.copy(), record, [], 0))
        return self._keys()


VIEWS = {'count': CountView, 'rotation': RotationView}
