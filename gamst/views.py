"""View modules: the application's screen for a running recipe (the HUD).

A view is application UI, outside GAMST's context boundary. It does two things:

- **input**: keys (and, in richer apps, buttons) become commands to the Pipeline: quit, reset the
  modules, switch a module on or off;
- **output**: it draws the camera frame with values read back from committed records, the same
  records an agent reads over MCP. A view never computes a measurement itself.

Before a run, ``gamst recipe hud`` shows the setup step (gamst/hud_setup.py) in the same window as :class:`CountView`.

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
STATE_WORD = {'heard': '들림', 'provisional': '잠정', 'final': '확정', 'revoked': '취소', 'uncertain': '불확실'}
LATE_COLOR, SELECTED, OTHER, FAINT = (60, 230, 255), (110, 220, 120), (200, 160, 120), (110, 110, 110)
KEPT_RECORDS = 4096
NOT_OCCURRENCES = ('exemplar.ended', 'event.revision')      # an end marker and a decision are not occurrences
KEYS_LINE = ('q/Esc 끝내기 · r 모듈 초기화 · a 모든 사람 상자', '클릭: 타임라인 → 근거 · 사람 → 선택')


class Writer:
    """The HUD's text: the video HUD's font (Korean words too) when Pillow and the font are there (``korean``), else
    OpenCV's, ASCII only (the views then use their English words)."""

    def __init__(self):
        self.text = None
        try:
            from pathlib import Path
            from .count_video import KOREAN, KoreanText
            if Path(KOREAN).exists():
                self.text = KoreanText()
        except ImportError:
            pass
        self.korean = self.text is not None

    def say(self, korean, english):
        return korean if self.korean else english

    def put(self, image, line, x, y, color, size=17):
        """One line with its baseline at ``y``; returns its width."""
        if self.text:
            return self.text.put(image, line, x, y - int(size * .82), size, color)
        import cv2
        line = line.encode('ascii', 'replace').decode()
        cv2.putText(image, line, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size / 35, color, 1)
        return cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, size / 35, 1)[0][0]

    def width(self, line, size=17):
        if self.text:
            return self.text.patch(line, size, (255, 255, 255))[0].shape[1]
        import cv2
        return cv2.getTextSize(line.encode('ascii', 'replace').decode(), cv2.FONT_HERSHEY_SIMPLEX, size / 35, 1)[0][0]


def event_line(event, korean=False):
    """One readable line of a committed event or revision, e.g. ``-> revoked  뽀또  (other_speaker)`` (``korean``:
    ``-> 취소  뽀또  (other_speaker)``)."""
    revision = event.get('event') == 'event.revision'
    state = event.get('state') or ('final' if not revision else '?')
    what = event.get('term') or short(event.get('label')) or event.get('event', '')
    parts = [('-> ' if revision else '+ ') + (STATE_WORD.get(state, state) if korean else state), str(what)]
    if event.get('event') == 'term.other_voice':
        parts[0] = '  다른 목소리' if korean else '  other voice'
    if korean and event.get('who') not in (None, '', 'unknown'):
        parts.append(str(event['who']))
    if type(event.get('speaker_score')) in (int, float):
        parts.append(f"{'목소리' if korean else 'voice'} {event['speaker_score']:.2f}")
    if event.get('candidate'):
        parts.append('후보' if korean else 'candidate')
    if event.get('reason'):
        parts.append(f"({event['reason']})")
    return '  '.join(parts)


def occurrences(rows, binding_rows=()):
    """Every event with a span (``start_ms`` and ``end_ms``) of the kept rows (``{capability: [record]}``), oldest
    first, each with its revision state, ``who`` and ``person_id`` as the person binding says now (retroactive within a
    track; one action of a person seen on two tracks once: :meth:`gamst.engine.Gamst._with_who`), and ``capability``."""
    from .counting import revisions, with_state
    from .engine import Gamst
    from .providers.person_binding import binding_events, intervals_of
    events = binding_events(binding_rows)
    who, persons = intervals_of(events), intervals_of(events, key='person_id')
    found = []
    for capability, kept in rows.items():
        if capability == 'vision.person_binding':
            continue
        revised = revisions(kept)
        for record in Gamst._with_who(kept, who, persons) if events else kept:
            value = (record.get('observation') or {}).get('value')
            for item in (value.get('events') if isinstance(value, dict) else None) or ():
                if not isinstance(item, dict) or item.get('event') in NOT_OCCURRENCES or \
                        type(item.get('start_ms')) not in (int, float) or type(item.get('end_ms')) not in (int, float):
                    continue
                found.append(dict(with_state(item, revised), capability=capability, record_ms=record['timestamp_ms']))
    return sorted(found, key=lambda item: (item['start_ms'], item['end_ms']))


def short(label, most=16):
    """A movement's label as the panel shows it: the name before a colon (``관제탑: 두 팔을 ...`` -> ``관제탑``)."""
    label = str(label or '').split(':')[0].strip()
    return label if len(label) <= most else label[:most - 1] + '…'


def person_of(event):
    """The person an occurrence is of: its name when one is bound, else its person id, else None."""
    who = event.get('who')
    if isinstance(who, str) and who and who != 'unknown':
        return who
    return event.get('person_id') or None


class CountView(_Window):
    """Any recipe that counts: the frame (or, with sound only, a panel of its own), then per declared capability its
    count, how many events are heard / provisional / final / revoked / uncertain (the revision states of
    :mod:`gamst.counting`, in the video HUD's colours), the count per person (the named person and the others, by
    the person binding), and the newest events and corrections as they are decided (an L2 correction says LATE).

    Below, a timeline strip: each occurrence with its span; a click shows its evidence (the rule's steps with their
    times, the nearest example with its distance). The box of the selected person is drawn (the named person at
    first; a click on a box selects that person), the others faint; ``a`` shows them all. The panel says which pose
    model counts (measured at the start in realtime) and what was left out. Korean words when the font is there.

    ``hold_at_end``: after the last frame the window stays until q / Esc, so the evidence can still be read (the
    setup step, gamst/hud_setup.py, sets it)."""
    title = 'GAMST recipe'
    every_ms = 100      # a panel without video is redrawn at most this often
    max_height = 720    # a taller picture is shown smaller (clicks are mapped back)
    strip_h = 92
    hold_at_end = False

    def __init__(self, pipeline, *, headless=False, capabilities=()):
        super().__init__(pipeline, headless=headless, capabilities=capabilities)
        self.rows, self.recent, self._drawn_ms, self.writer = {}, {}, None, None
        self.binding_rows, self.poses, self.tracks, self.bound = [], None, None, {}
        self.selected, self.show_all, self.card = None, False, None
        self.first_ms = self.now_ms = None
        self.hits, self.clicks, self._mouse, self._found, self._last = [], [], set(), (None, []), None

    def _put(self, panel, line, x, y, color):
        """One line with its baseline at ``y``; returns its width (:class:`Writer`)."""
        if self.writer is None:
            self.writer = Writer()
        return self.writer.put(panel, line, x, y, color)

    @property
    def korean(self):
        if self.writer is None:
            self.writer = Writer()
        return self.writer.korean

    def _say(self, korean, english):
        return korean if self.korean else english

    def _fit(self, line, width):
        """``line``, cut short to ``width`` pixels."""
        if self.writer.width(line) <= width:
            return line
        while line and self.writer.width(line + '…') > width:
            line = line[:-1]
        return line + '…'

    async def _refresh(self, records):
        await super()._refresh(records)
        for record in records:
            capability = ((record.get('observation') or {}).get('capability') or {}).get('id')
            value = (record.get('observation') or {}).get('value')
            if ((record.get('observation') or {}).get('name') == 'scene.poses' or capability == 'scene.poses') and \
                    isinstance(value, list):
                self.poses = (record['timestamp_ms'], value)       # boxes (0..1 of the frame) with their track
            if capability == 'vision.person_binding' and isinstance(value, dict):
                self.tracks = (record['timestamp_ms'], {item['track_id']: item for item in value.get('tracks') or ()
                                                        if isinstance(item, dict) and item.get('track_id')})
                self.bound = dict(value.get('bound') or {})
                if value.get('events'):
                    self.binding_rows.append(record)
                    del self.binding_rows[:-KEPT_RECORDS]
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

    def _occurrences(self):
        """:func:`occurrences`, computed again only when a record was kept."""
        key = (sum(len(kept) for kept in self.rows.values()), len(self.binding_rows))
        if self._found[0] != key:
            self._found = (key, occurrences(self.rows, self.binding_rows))
        return self._found[1]

    def _named(self):
        """The named persons of the run (the binding's ``bound``), the selected one first."""
        names = [name for name in self.bound if name and name != 'unknown']
        if self.selected is None and names:
            self.selected = names[0]
        return names

    # -- the panel
    def _notes(self):
        """Lines about how the run was set up: the pose model (measured in realtime), what was left out, refusals."""
        notes = getattr(self.pipeline, 'notes', None) or {}
        context = getattr(self.pipeline, 'recipe_context', None) or {}
        recipe = context.get('recipe') or {}
        lines = []
        choice = notes.get('pose_choice')
        pose = next((item for item in recipe.get('modules', ()) if item.get('use') in ('rtmw-pose', 'mediapipe-pose',
                                                                                       'sapiens2-pose')), None)
        if isinstance(choice, dict):      # which pose model counts, as measured at start (recipes.choose_pose)
            kept = choice['chosen'] == choice['model']
            lines.append((self._say(f"자세 모델 {choice['chosen']}: 측정 {choice['live_fps']:g}+{choice['l2_fps']:g} fps, "
                                    f"입력 {choice['input_fps']:g}" + ('' if kept else f" ({choice['model']} 너무 느림)"),
                                    f"pose {choice['chosen']}: {choice['live_fps']:g}+{choice['l2_fps']:g} fps measured, "
                                    f"input {choice['input_fps']:g}" + ('' if kept else f" ({choice['model']} too slow)")),
                          (150, 150, 150) if kept else STATE_COLOR['revoked']))
        elif pose is not None:
            timing = notes.get('timing')
            lines.append((self._say(f"자세 모델 {pose.get('candidate') or pose['use']}" + (
                ' (나중에: 모든 프레임, 가장 정확한 모델)' if timing == 'post' else ''),
                f"pose {pose.get('candidate') or pose['use']}" + (' (post: every frame)' if timing == 'post' else '')),
                (150, 150, 150)))
        running = set(getattr(self.pipeline, 'modules', {}) or {})
        left = list(dict.fromkeys([*((recipe.get('composed') or {}).get('left_out') or ()),     # compose's left_out
                                   *(item['name'] for item in recipe.get('modules', ()) if running and
                                     item['name'] not in running)]))
        built = context.get('notes') or {}
        if left:              # modules of the recipe this run does not run (optional ones off, or not given)
            lines.append((self._say('뺀 모듈: ', 'left out: ') + ', '.join(left), (150, 150, 150)))
        for key, text in built.items():     # what a module left out while it was built (the report's setup_notes)
            lines.append((self._say('빠진 모델: ', 'not set up: ') + f'{key}: {text}', (150, 150, 150)))
        spill = notes.get('video_memory')
        if spill:                         # the spill file was bounded by free disk (gamst/providers/video_memory.py)
            lines.append((self._say('영상 저장 제한: ', 'SPILL LIMITED: ') + spill[:70], STATE_COLOR['revoked']))
        refused = notes.get('who_refused')
        if refused:     # an instruction naming a person was refused (gamst/person_reference.py, same words as CLI and MCP)
            lines.append((refused[:110], STATE_COLOR['revoked']))
        quality = getattr(self.pipeline, 'voice_quality', None)
        if isinstance(quality, dict) and quality.get('low'):      # gamst/voice_quality.py
            lines.append((self._say(f"목소리 기준 품질 낮음 {quality['quality']['mean']:.2f} -> {quality['used']}",
                                    f"VOICE REFERENCE QUALITY LOW {quality['quality']['mean']:.2f} -> {quality['used']}"),
                          STATE_COLOR['revoked']))
        return lines

    def _per_person(self, capability, found):
        """``[(text, colour)]``: the count per person of one capability's occurrences (named person and the others)."""
        from .counting import COUNTED
        mine = [event for event in found if event['capability'] == capability and event.get('state', 'final') in COUNTED
                and event.get('event') != 'term.other_voice']
        if not mine or not any('track_id' in event for event in mine):
            others = sum(1 for event in found if event['capability'] == capability
                         and event.get('event') == 'term.other_voice')
            return [[(self._say(f'다른 목소리 {others} (세지 않음)', f'other voice {others} (not counted)'), FAINT)]] \
                if others else []
        counts = {}
        for event in mine:
            key = person_of(event) or '?'
            counts[key] = counts.get(key, 0) + 1
        named = [key for key in counts if key in self.bound or key == self.selected]
        others = {key: number for key, number in counts.items() if key not in named}
        parts = [(f'{key} {counts[key]}', SELECTED if key == self.selected else OTHER) for key in named]
        if others:
            detail = ' · '.join(f'{key} {number}' for key, number in sorted(others.items(), key=lambda item: -item[1])[:4])
            parts.append((self._say(f'다른 사람 {sum(others.values())} ({detail})',
                                    f'others {sum(others.values())} ({detail})'), OTHER))
        return [parts]      # drawn part by part, each in its colour

    def _panel(self, height, found):
        import numpy as np
        from .counting import STATES, count_events
        panel = np.full((height, 600, 3), 24, np.uint8)
        mode = getattr(self.pipeline, 'mode', 'count')
        timing = (getattr(self.pipeline, 'notes', None) or {}).get('timing')
        lines = [(self._say(f"GAMST  {self.pipeline.name}   모드: {'찾기' if mode == 'search' else '세기'}" +
                            (f"   때: {'실시간' if timing == 'realtime' else '나중에'}" if timing else ''),
                            f'GAMST  {self.pipeline.name}   mode: {mode}'), (80, 220, 255))]
        if mode == 'search':
            lines.append((self._say('찾기: 애매한 것도 후보로 셉니다', 'SEARCH: unclear words count as candidates'),
                          STATE_COLOR['provisional']))
        lines += self._notes()
        lines.append(('', None))
        tallies, resolved = [], {_key(event): person_of(event) for event in found if person_of(event)}
        for capability in self.capabilities:
            record = self.latest.get(capability)
            value = record['observation']['value'] if record else {}
            if capability == 'vision.person_binding':
                persons = (value.get('persons') or {}) if isinstance(value, dict) else {}
                names = self._named()
                lines.append((self._say('사람 ', 'persons ') + (' · '.join(
                    f'{name} = {next((key for key, track in persons.items() if track == value["bound"].get(name)), "?")}'
                    for name in names) or self._say('이름 붙은 사람 없음', 'nobody named')) + self._say(
                    f'   보이는 사람 {len(self.tracks[1]) if self.tracks else 0}명',
                    f'   tracks {len(self.tracks[1]) if self.tracks else 0}'), None))
                lines.append(('', None))
                continue
            label = next((event.get('label') for event in reversed(self.recent.get(capability, ())) if event.get('label')),
                         None)
            lines.append(((f'{short(label)}  ' if label else '') + capability, None))
            lines.append((self._say(f"  모듈 셈 {value.get('count', 0)}   {value.get('status', '' if record else '기다림')}",
                                    f"  count {value.get('count', 0)}   {value.get('status', '' if record else 'waiting')}"),
                          (255, 255, 255)))
            for parts in self._per_person(capability, found):
                tallies.append((len(lines), parts))
                lines.append(('', None))
            states = count_events(self.rows.get(capability, ()), max_evidence=0).get('states') or {}
            if any(states.values()):
                tallies.append((len(lines), [(f'{self._say(STATE_WORD[name], name)} {states.get(name, 0)}',
                                              STATE_COLOR[name]) for name in STATES]))
                lines.append(('', None))
            if 'distance' in value:
                lines.append((self._say(f"  거리 {value['distance']:.3f}", f"  distance {value['distance']:.3f}"), None))
            for event in reversed(self.recent.get(capability, ())[-4:]):      # newest first
                state = event.get('state') or 'final'
                late = 'LATE ' if event.get('correction') else ''     # an L2 correction, counted after the live lane
                now = resolved.get(_key(dict(event, capability=capability)))
                if now is not None:     # who the binding says it is now (retroactive within the track)
                    event = dict(event, who=now)
                lines.append(('  ' + late + event_line(event, self.korean),
                              LATE_COLOR if late else STATE_COLOR.get(state)))
            lines.append(('', None))
        lost = [self.pipeline.lost(capability) for capability in self.capabilities]
        if any(item['gaps'] for item in lost):
            lines += [(self._say(f"불완전: {max(item['frames'] for item in lost)} 프레임 잃음",
                                 f"INCOMPLETE: {max(item['frames'] for item in lost)} frames lost"), STATE_COLOR['revoked']),
                      ('', None)]
        status = self.pipeline.status()['modules']
        lines.append((self._say('모듈 (r: 초기화, q: 끝내기)', 'modules (r: reset, q: quit)'), None))
        lines += [(f"  {name}: {self._say('켬' if item['enabled'] else '끔', 'on' if item['enabled'] else 'off')}  "
                   f"{item['last_latency_ms'] or 0:.0f} ms", None) for name, item in status.items()]
        rows = max(1, (height - 50) // 22 - 1)
        for index, (line, color) in enumerate(lines[:rows]):
            if line:
                self._put(panel, self._fit(line, 580), 10, 26 + 22 * index, color or (255, 255, 255))
        for index, tally in tallies:
            if index >= rows:
                continue
            x = 22
            for text, color in tally:
                x += self._put(panel, text, x, 26 + 22 * index, color) + 14
        for index, line in enumerate(KEYS_LINE if self.korean else ('q/Esc quit · r reset modules · a all boxes',
                                                                    'click: timeline -> evidence · person -> select')):
            self._put(panel, line, 10, height - 32 + 20 * index, (150, 150, 150))
        return panel

    # -- the picture: boxes of the persons
    def _boxes(self, image, frame_ms, scale):
        import cv2
        if not self.poses or self.poses[0] < frame_ms - 1500:
            return
        tracks = self.tracks[1] if self.tracks else {}
        height, width = image.shape[:2]
        self._named()
        for pose in self.poses[1]:
            box = pose.get('bbox') if isinstance(pose, dict) else None
            if not isinstance(box, dict):
                continue
            entry = tracks.get(pose.get('track_id')) or {}
            if entry.get('duplicate_of'):
                continue
            who = person_of(entry)
            x0, y0 = int(box['x'] * width), int(box['y'] * height)
            x1, y1 = int((box['x'] + box['width']) * width), int((box['y'] + box['height']) * height)
            chosen = who is not None and who == self.selected
            if chosen or self.show_all:
                color = SELECTED if chosen else OTHER
                cv2.rectangle(image, (x0, y0), (x1, y1), color, 3 if chosen else 2)
                self._put(image, who or self._say('모름', 'unknown'), x0 + 4, max(18, y0 - 6), color)
            else:
                cv2.rectangle(image, (x0, y0), (x1, y1), FAINT, 1)
            self.hits.append(((x0, y0, x1, y1), ('person', who)))

    # -- the timeline strip and the evidence card
    def _strip(self, width, found):
        import cv2
        import numpy as np
        strip = np.full((self.strip_h, width, 3), 20, np.uint8)
        first = self.first_ms or 0
        last = max(self.now_ms or first, max((event['end_ms'] for event in found), default=first), first + 10000)
        left, right, top = 14, width - 14, 30
        px = lambda ms: int(left + (right - left) * (ms - first) / max(1, last - first))
        cv2.rectangle(strip, (left, top), (right, top + 30), (36, 36, 36), -1)
        self._put(strip, self._say(f'타임라인: 동작 {len(found)}개 · 초록 {self.selected or "선택한 사람"} · 파랑 다른 사람 · '
                                   '노랑 LATE (L2가 늦게 셈) · 누르면 근거',
                                   f'timeline: {len(found)} occurrences · green selected · blue others · yellow LATE · '
                                   'click for evidence'), left, 20, (150, 150, 150))
        for index, event in enumerate(found):
            x0, x1 = px(event['start_ms']), max(px(event['end_ms']), px(event['start_ms']) + 3)
            state = event.get('state', 'final')
            color = STATE_COLOR[state] if state in ('revoked', 'uncertain') else \
                SELECTED if person_of(event) == self.selected and self.selected else \
                FAINT if event.get('event') == 'term.other_voice' else OTHER
            cv2.rectangle(strip, (x0, top + 3), (x1, top + 27), color, -1)
            if state == 'provisional':
                cv2.rectangle(strip, (x0, top + 3), (x1, top + 27), STATE_COLOR['provisional'], 1)
            if event.get('correction'):
                cv2.rectangle(strip, (x0, top - 4), (x1, top + 1), LATE_COLOR, -1)
            if self.card is not None and self.card == _key(event):
                cv2.rectangle(strip, (x0 - 2, top - 6), (x1 + 2, top + 33), (255, 255, 255), 2)
            self.hits.append(((x0 - 3, top - 8, x1 + 3, top + 36), ('span', index)))
        if self.now_ms is not None:
            cv2.line(strip, (px(self.now_ms), top - 6), (px(self.now_ms), top + 36), (40, 170, 255), 2)
        self._put(strip, self._say('0초', '0 s'), left, top + 52, (150, 150, 150))
        end = f'{(last - first) / 1000:.1f}'
        self._put(strip, self._say(f'{end}초', f'{end} s'), right - 60, top + 52, (150, 150, 150))
        return strip

    def _evidence(self, event):
        """The lines of an occurrence's evidence card: who, when, state, the rule's steps, the nearest example."""
        first = self.first_ms or 0
        at = lambda ms: f'{(ms - first) / 1000:.2f}' + self._say('초', ' s')
        who = person_of(event) or self._say('모름', 'unknown')
        lines = [(f"#{event.get('index', '?')} {short(event.get('label'), 40) or event.get('term') or event.get('event')} · {who}" +
                  (f" ({event['person_id']})" if event.get('person_id') and event.get('person_id') != who else ''),
                  (80, 220, 255)),
                 (f"{at(event['start_ms'])} - {at(event['end_ms'])} ({(event['end_ms'] - event['start_ms']) / 1000:.2f}"
                  f"{self._say('초', ' s')}) · {self._say(STATE_WORD.get(event.get('state', 'final'), '?'), event.get('state', 'final'))}" +
                  (self._say(' · LATE: L2 정정 (실시간 길이 못 따라온 프레임을 늦게 셈)', ' · LATE: an L2 correction')
                   if event.get('correction') else ''), LATE_COLOR if event.get('correction') else (255, 255, 255))]
        evidence = event.get('evidence') or {}
        rule, example = evidence.get('rule'), evidence.get('example')
        if evidence.get('combine'):
            lines.append((self._say('셈: 규칙 그리고 예시' if evidence['combine'] == 'all' else '셈: 규칙 또는 예시',
                                    f"combine: {evidence['combine']}"), (150, 150, 150)))
        if isinstance(rule, dict):
            steps = rule.get('steps_ms') or []
            lines.append((self._say('규칙: ' + ('맞음' if rule.get('satisfied') else '안 맞음'),
                                    'rule: ' + ('satisfied' if rule.get('satisfied') else 'not satisfied')),
                          STATE_COLOR['final'] if rule.get('satisfied') else STATE_COLOR['revoked']))
            if steps:
                lines.append(('  ' + ' · '.join(self._say(f'{number}단계 {at(ms)}', f'step {number} {at(ms)}')
                                                for number, ms in enumerate(steps, 1) if type(ms) in (int, float)),
                              (255, 255, 255)))
        if isinstance(example, dict):
            distance = example.get('distance')
            lines.append((self._say(f"가장 가까운 예시: {example.get('index', '?')}번, 거리 "
                                    f"{distance:.4f}" if type(distance) in (int, float) else '가장 가까운 예시: 없음',
                                    f"nearest example #{example.get('index', '?')}, distance {distance}") +
                          (self._say(f" (기준 {example['enter']:.3f})", f" (enter {example['enter']:.3f})")
                           if type(example.get('enter')) in (int, float) else ''), (255, 255, 255)))
            negative = example.get('negative_distance')
            lines.append(('  ' + self._say('반례 거리: ' + (f'{negative:.4f}' if type(negative) in (int, float) else '없음'),
                                           f'negative distance: {negative}') +
                          (f"  · {self._say('예시 구간', 'example span')} {at(example['start_ms'])} - {at(example['end_ms'])}"
                           if type(example.get('start_ms')) in (int, float) and type(example.get('end_ms')) in (int, float)
                           else ''), (150, 150, 150)))
        if not isinstance(rule, dict) and not isinstance(example, dict) and type(event.get('distance')) in (int, float):
            lines.append((self._say(f"거리 {event['distance']:.4f}", f"distance {event['distance']:.4f}"), (255, 255, 255)))
        if event.get('track_id'):
            lines.append((self._say('트랙 ', 'track ') + str(event['track_id'])[-40:], (150, 150, 150)))
        if event.get('reason'):
            lines.append((self._say('이유: ', 'reason: ') + str(event['reason']), (150, 150, 150)))
        lines.append((self._say('카드를 누르면 닫힙니다', 'click the card to close'), (110, 110, 110)))
        return lines

    def _card(self, image, found):
        import cv2
        event = next((item for item in found if _key(item) == self.card), None)
        if event is None:
            return
        lines = self._evidence(event)
        width, height = min(image.shape[1] - 20, 720), 18 + 24 * len(lines)
        x0, y0 = 10, max(0, image.shape[0] - height - 10)
        overlay = image[y0:y0 + height, x0:x0 + width]
        overlay[:] = (overlay * .25 + 12).astype(overlay.dtype)
        cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height), (80, 220, 255), 1)
        for index, (line, color) in enumerate(lines):
            self._put(image, line, x0 + 12, y0 + 26 + 24 * index, color)
        self.hits.append(((x0, y0, x0 + width, y0 + height), ('card',)))

    # -- input: clicks and keys
    def _click(self, x, y):
        for (x0, y0, x1, y1), action in reversed(self.hits):
            if x0 <= x < x1 and y0 <= y < y1:
                found = self._occurrences()
                if action[0] == 'card':
                    self.card = None
                elif action[0] == 'span' and action[1] < len(found):
                    self.card = _key(found[action[1]])
                elif action[0] == 'person' and action[1]:
                    self.selected = action[1]
                return action
        return None

    def _keys(self, wait_ms=1):
        import cv2
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('a'):
            self.show_all = not self.show_all
        while self.clicks:
            self._click(*self.clicks.pop(0))
        if key in (27, ord('q')):
            return False
        if key == ord('r'):
            for name in self.pipeline.modules:
                self.pipeline.disable(name)
                self.pipeline.enable(name)
        return True

    def _listen(self, title):
        import cv2
        if title in self._mouse:
            return
        self._mouse.add(title)
        try:
            cv2.setMouseCallback(title, lambda event, x, y, flags, param: self.clicks.append((x, y))
                                 if event == cv2.EVENT_LBUTTONDOWN else None)
        except cv2.error:
            pass

    def draw(self, frame):
        """The whole window for one frame (a video frame, or the sound-only panel); also used to hold the last one."""
        import cv2
        import numpy as np
        self.hits = []
        found = self._occurrences()
        scale = 1.
        if frame.modality == 'video':
            image = frame.data
            if image.shape[0] > self.max_height:
                scale = self.max_height / image.shape[0]
                image = cv2.resize(image, (int(image.shape[1] * scale), self.max_height), interpolation=cv2.INTER_AREA)
            else:
                image = image.copy()
            self._boxes(image, frame.timestamp_ms, scale)
        else:
            image = np.full((540, 480, 3), 16, np.uint8)
            self._put(image, self._say('소리만: 카메라 없음', 'sound only: no camera'), 20, 40, (150, 150, 150))
        height = max(image.shape[0], 540)
        if image.shape[0] < height:
            image = np.vstack([image, np.full((height - image.shape[0], image.shape[1], 3), 16, np.uint8)])
        self._card(image, found)
        panel = self._panel(height, found)
        top = np.hstack([image, panel])
        strip = self._strip(top.shape[1], found)
        for index, (rect, action) in enumerate(self.hits):
            if action[0] == 'span':
                self.hits[index] = ((rect[0], rect[1] + height, rect[2], rect[3] + height), action)
        return np.vstack([top, strip])

    async def update(self, input_id, frame, records):
        await self._refresh(records)
        if self.headless:
            return True
        if self.first_ms is None:
            self.first_ms = frame.timestamp_ms
        self.now_ms = max(self.now_ms or frame.timestamp_ms, frame.timestamp_ms)
        if frame.modality != 'video':
            if self._video():
                return True       # drawn with the video frames
            if self._drawn_ms is not None and frame.timestamp_ms - self._drawn_ms < self.every_ms:
                return True
            self._drawn_ms = frame.timestamp_ms
        import cv2
        title = self._title(input_id if frame.modality == 'video' else 'sound')
        self._last = (title, frame)
        cv2.imshow(title, self.draw(frame))
        self._listen(title)
        return self._keys()

    def close(self):
        if not self.headless and self.hold_at_end and self._last is not None:
            import cv2
            title, frame = self._last
            print(self._say('끝: 타임라인을 누르면 근거를 봅니다 · q/Esc 닫기', 'done: click the timeline for evidence · q/Esc close'),
                  flush=True)
            while True:
                cv2.imshow(title, self.draw(frame))
                if not self._keys(30):
                    break
        super().close()


def _key(event):
    """One occurrence, the same across redraws."""
    return (event.get('capability'), event.get('event_id') or (event.get('start_ms'), event.get('end_ms'),
                                                               event.get('track_id'), event.get('label')))


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
