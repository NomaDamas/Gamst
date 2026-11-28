"""The setup step of the live HUD: the user-defined input module of the perception input, shown before the run starts.

    gamst recipe hud [--input FILE|DEVICE ...]
                     [--who NAME] [--who-text TEXT] [--photo JPG ...] [--voice V.json]
                     [--action TEXT] [--rule RULE.json] [--example FILE@A-B[@X,Y] ...] [--negative ...] [--words W,W]
                     [--mode count|search] [--timing realtime|post] [--out-dir DIR] [--profile P.json]
                     [--allow-remote-asr] [-- more options of gamst recipe run]

One window, the live view's (:class:`gamst.views.CountView`): first the setup step, then, on Start, the same window
shows the run. Everything the command line gives is filled in already; the rest is set in the window.

- **입력 (input)**: files or devices; the user never picks the kind, it is detected from the path or device
  (:func:`detect_input`): a video file, a WAV file, a sensor declaration (the JSON object of ``compose_recipe``'s
  ``inputs.sensors``: columns and time are declared, never guessed), a camera number, or the microphone.
- **누구 (who, optional)**: a name or description AND photos (owner decision 59: both, or nobody). Photos come from files
  or are grabbed from the camera or video; they are enrolled at once in the background (:func:`compose.photo_reference`,
  embeddings only) and the person check (:func:`gamst.person_reference.check`: faces agree, one person per photo) is
  shown with the photo count.
- **무엇 (what)**: an action text with a rule file, and/or example clips and negatives trimmed here on the input video
  (mark start and end, save as an example or a negative; a click on the picture names the person to learn the clip
  from: ``at: [x, y]``), and/or words to count.
- **방식 (mode)**: count / search; **때 (timing)**: realtime / post.

Validation is :func:`gamst.recipes.compose.validate_request`, the rules of ``compose_recipe`` themselves. While something
is missing, Start is off and one Korean line (:func:`korean`) says what to add. Start composes through
:func:`gamst.recipes.compose.compose` and runs the composed recipe in this window.

Nothing here asks for consent (owner, 2026-11-28): GAMST is open source and installing it is the consent for the camera,
the microphone and recognition. The person reference format still keeps a consent record; it is filled in as
:data:`INSTALL_CONSENT`. What is said before a run is a cost: sound goes to a hosted, paid recogniser only with
``--allow-remote-asr`` (a cost notice, not consent).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ACCENT, GOOD, BAD, LATE, MUTED, WHITE = (80, 220, 255), (110, 220, 120), (80, 80, 240), (40, 170, 255), (150, 150, 150), \
    (240, 240, 240)
BG, PANEL, EDGE = (16, 16, 16), (24, 24, 24), (70, 70, 70)
PREVIEW_W, PREVIEW_H, BAR_H, PANEL_W, HEIGHT = 960, 540, 64, 640, 760
INSTALL_CONSENT = 'GAMST 설치·사용자 동의'     # the consent record a person reference needs (installing is the consent)


# ---------------------------------------------------------------- what is missing, in one Korean line

KOREAN = [   # (a phrase of compose's sentence, the Korean line); the first that matches
    ('inputs: give at least one input', '입력을 넣으세요: 영상이나 소리(.wav) 파일, 센서 설명(.json), 카메라 번호 또는 마이크.'),
    ('one picture source', '화면 입력은 하나만 됩니다: 카메라와 영상 파일 중 하나를 빼세요.'),
    ('one sound source', '소리 입력은 하나만 됩니다: 마이크와 소리 파일 중 하나를 빼세요.'),
    ('a movement and a person are seen on pictures', '동작과 사람은 화면으로 봅니다: 영상 파일이나 카메라 번호를 넣으세요.'),
    ('words are heard', '단어는 소리로 듣습니다: 마이크, 소리 파일, 또는 소리가 있는 영상 파일을 넣으세요.'),
    ('nothing in this request reads', '센서를 읽는 것이 없습니다: 리듬(채널, 대역)을 고르거나 센서를 빼세요.'),
    ('timing: post counts a finished file', '카메라와 마이크는 실시간만 됩니다: 때를 실시간으로 바꾸세요.'),
    ('say who it is in words as well', '누구인지 글로도 적어 주세요(이름이나 설명): 사진만으로는 누구를 세는지 모릅니다.'),
    ('alone never identifies a person', '사람 사진을 추가하세요: 이름만으로는 사람을 알아볼 수 없습니다(한 장에 한 사람).'),
    ('no consent record', '동의를 적어 주세요: 누가, 무엇에 동의했는지.'),
    ('1..16 photos', '사진은 1-16장, 한 장에 한 사람만 나와야 합니다.'),
    ('exactly one person', '사진마다 한 사람만 나와야 합니다: 여러 사람이나 아무도 없는 사진을 빼세요.'),
    ('face crops disagree', '사진의 얼굴이 서로 다릅니다: 같은 사람의 사진만 넣으세요.'),
    ('usable face crop', '쓸 수 있는 얼굴이 모자랍니다(3개 이상): 얼굴이 정면으로 보이는 사진을 더 넣으세요.'),
    ('only whole-body appearance', '얼굴이 보이는 사진이 필요합니다: 옷차림은 같은 영상 안에서만 믿습니다.'),
    ('no face, body or voice samples', '사진에서 사람을 찾지 못했습니다: 사람이 잘 보이는 사진을 넣으세요.'),
    ('who.voice', '그 사람이 한 말을 세려면 목소리 등록이 필요합니다: --voice 파일을 주세요.'),
    ('enrolls', '사람 파일의 이름이 적은 이름과 다릅니다.'),
    ('examples_from', '예시를 배울 사람(누구)을 정하세요.'),
    ('no hidden LLM', '동작을 글로만 줄 수는 없습니다: 규칙 파일을 고르거나 예시 구간을 저장하세요.'),
    ('a rule needs the text', '규칙이 나타내는 동작을 글로 적어 주세요.'),
    ('what.movement.rule', '규칙 파일이 맞지 않습니다: gamst/providers/movement_rule.py 의 규칙 JSON 을 고르세요.'),
    ('look-alikes go with examples', '반례만 있습니다: 동작 예시 구간도 저장하세요.'),
    ('start_s < end_s', '구간은 시작이 끝보다 앞이고, 한 동작만(30초 이하) 담아야 합니다.'),
    ('a point goes with', '사람을 고른 점은 구간과 함께 저장합니다.'),
    ('what: say what to look for', '무엇을 셀지 정하세요: 동작(글과 규칙, 또는 예시 구간), 단어, 또는 센서 리듬.'),
    ('what.words', '단어를 쉼표 없이 하나씩 적어 주세요.'),
    ('is hosted and paid', '비용 안내: 단어 인식기는 유료 원격 서비스라 돈이 듭니다. 비용을 확인했으면 --allow-remote-asr 로 시작하세요.'),
    ('hosted diarization', '비용 안내: 원격 화자 분리는 유료입니다. 비용을 확인했으면 --allow-remote-diarization 을 주세요.'),
    ('there is no file', '파일이 없습니다: 다시 고르세요.'),
    ('is not one of', '이 형식의 파일은 쓸 수 없습니다: 다시 고르세요.'),
]


def korean(sentence):
    """One Korean line for one of compose's sentences (the rule itself stays compose's)."""
    for phrase, line in KOREAN:
        if phrase in sentence:
            return line
    return '추가할 것: ' + sentence[:90]


# ---------------------------------------------------------------- the setup model

def parse_clip(value):
    """{file, start_s, end_s[, at]} of ``FILE@A-B[@X,Y]`` (compose's form), or None."""
    from .recipes.compose import CLIP
    found = CLIP.fullmatch(value) if isinstance(value, str) else None
    if not found:
        return None
    clip = dict(file=found[1], start_s=float(found[2]), end_s=float(found[3]))
    if found[4]:
        clip['at'] = [float(found[4]), float(found[5])]
    return clip


def detect_input(value):
    """``(kind, value)`` of an input given as a path or a device, or ``(None, a Korean line)``. The user never picks the
    kind (owner, 2026-11-28): a camera number (``0``, ``camera 1``, ``카메라``), the microphone (``mic``, ``마이크``),
    a video file, a WAV file (``audio``) or a sensor declaration (``.json``; ``sensor``) by its suffix."""
    from .recipes.compose import SENSOR_FILES, SOUND_FILES, VIDEO_FILES
    if isinstance(value, int) and not isinstance(value, bool):
        return ('camera', value) if value >= 0 else (None, '카메라 번호는 0 이상입니다.')
    text = str(value or '').strip()
    low = text.lower()
    if low in ('camera', '카메라', 'cam'):
        return 'camera', 0
    number = re.fullmatch(r'(?:camera|cam|카메라)?\s*:?\s*(\d{1,2})', low)
    if number:
        return 'camera', int(number[1])
    if low in ('mic', 'microphone', '마이크'):
        return 'microphone', True
    path = Path(text).expanduser()
    if not text or not path.is_file():
        return None, f'파일이나 장치를 찾지 못했습니다: {text or "(빈칸)"} (영상·소리 파일, 센서 설명 .json, 카메라 번호, 마이크)'
    suffix = path.suffix.lower()
    if suffix in VIDEO_FILES:
        return 'video', path.absolute()
    if suffix in SOUND_FILES:
        return 'audio', path.absolute()
    if suffix == '.json':
        return 'sensor', path.absolute()
    if suffix in SENSOR_FILES:
        return None, '센서 기록은 설명 파일(.json: id, file, modality, columns, time)로 넣으세요: 열과 시간은 적어 줘야 합니다.'
    return None, f'이 형식의 파일은 입력으로 쓸 수 없습니다: {path.name} (영상 {" ".join(VIDEO_FILES)}, 소리 .wav, 센서 설명 .json)'


class Setup:
    """What the setup step holds; :meth:`request` is a ``compose_recipe`` request and :meth:`state` says whether it
    can start. Nothing here draws (:class:`SetupWindow` does)."""

    def __init__(self, out_dir, *, profile=None, enroll=None):
        self.out_dir, self.profile = Path(out_dir), profile
        self.pictures = None            # ('camera', index) | ('video', path)
        self.microphone = False
        self.audio = None               # a sound file (WAV)
        self.sensors, self.rhythm = [], None        # sensor declarations; {input, channel, band}
        self.name = self.text = ''
        self.consent = INSTALL_CONSENT  # never asked: installing GAMST is the consent (owner, 2026-11-28)
        self.photos, self.voice = [], None
        self.action, self.rule = '', None
        self.examples, self.negatives, self.words = [], [], []
        self.mode, self.timing = 'count', None
        self.allow_remote_asr = False
        self.mark = dict(start=None, end=None, at=None)    # the clip being trimmed
        self.clip_file = None           # trim source when the input is a camera
        self.person = None              # {key, file, check} | {key, error} of the last enrollment
        self.enrolling = None           # key being enrolled
        self.notice = None              # a short line of the window's own (e.g. a grabbed photo)
        self._enroll = enroll or self._enroll_photos
        self._lock = threading.Lock()

    # -- inputs
    @property
    def video(self):
        return self.pictures[1] if self.pictures and self.pictures[0] == 'video' else None

    @property
    def trim_file(self):
        """The video clips are trimmed on: the input video, else a clip file chosen for it."""
        return self.video or self.clip_file

    def add_input(self, value):
        """An input given as a path or a device, its kind detected (:func:`detect_input`); an error line or None. A
        picture source replaces the other picture source, a sound source the other sound source."""
        kind, found = detect_input(value)
        if kind is None:
            return found
        if kind in ('camera', 'video'):
            self.pictures = (kind, found)
        elif kind == 'microphone':
            self.microphone, self.audio = True, None
        elif kind == 'audio':
            self.microphone, self.audio = False, found
        else:
            problem = self.add_sensor(found)
            if problem:
                return problem
            if not self.rhythm:
                self.cycle_rhythm('none')
        return None

    def clear_inputs(self):
        self.pictures, self.microphone, self.audio, self.sensors, self.rhythm = None, False, None, [], None

    def add_sensor(self, path):
        """A sensor declaration file ({id, file, modality, columns, time}); returns an error line or None."""
        try:
            item = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return '센서 설명 파일(JSON)을 읽지 못했습니다.'
        if not isinstance(item, dict) or not {'id', 'file', 'columns', 'time'} <= set(item):
            return '센서는 설명 파일로 고르세요: {id, file, modality, columns, time} (열과 시간은 적어 줘야 합니다).'
        if not Path(item['file']).is_absolute():
            item = dict(item, file=str(Path(path).parent / item['file']))
        self.sensors = [other for other in self.sensors if other.get('id') != item['id']] + [item]
        return None

    def cycle_rhythm(self, part):
        """Next channel or band of the rhythm of the first sensor."""
        from .providers.rhythm import BANDS
        if not self.sensors:
            return
        sensor = self.sensors[0]
        columns = [column.get('name') for column in sensor['columns'] if isinstance(column, dict)]
        rhythm = dict(self.rhythm or dict(input=sensor['id'], channel=columns[0] if columns else '', band='heart'))
        if part == 'channel' and columns:
            rhythm['channel'] = columns[(columns.index(rhythm['channel']) + 1) % len(columns)] \
                if rhythm['channel'] in columns else columns[0]
        elif part == 'band':
            bands = list(BANDS)
            rhythm['band'] = bands[(bands.index(rhythm['band']) + 1) % len(bands)]
        self.rhythm = rhythm

    # -- who
    def _key(self):
        return (self.name.strip() or self.text.strip(), tuple(str(path) for path in self.photos),
                self.consent.strip() or INSTALL_CONSENT, self.text.strip(), self.video)

    def enrolled(self):
        """The person file enrolled from exactly these photos, name and consent, or None."""
        person = self.person
        return person['file'] if person and person.get('file') and person['key'] == self._key() else None

    def _enroll_photos(self, key):
        from .recipes.compose import photo_reference
        name, photos, consent, text, video = key
        path = self.out_dir / f"{_slug(name)}.person.json"
        verdict, problem = photo_reference(dict(name=name, text=text or None, photos=list(photos), consent=consent),
                                           path, profile=self.profile, session_file=video)
        return dict(key=key, file=path, check=verdict) if verdict else dict(key=key, error=problem)

    def enroll(self, *, wait=False):
        """Enroll the photos in the background when a name (or description) and photos are given and changed."""
        key = self._key()
        if not (key[0] and key[1] and key[2]) or self.enrolling or (self.person and self.person['key'] == key):
            return None
        self.enrolling = key

        def work():
            try:
                found = self._enroll(key)
            except Exception as error:      # a model that failed to load is said, not raised into the window
                found = dict(key=key, error=f'{type(error).__name__}: {error}')
            with self._lock:
                self.person, self.enrolling = found, None
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        if wait:
            worker.join()
        return worker

    def add_photos(self, paths):
        self.photos = list(dict.fromkeys([*self.photos, *[Path(path).absolute() for path in paths]]))[:16]

    # -- what
    def set_mark(self, part, value):
        self.mark[part] = value
        start, end = self.mark['start'], self.mark['end']
        if part == 'start' and end is not None and end <= value or part == 'end' and start is not None and start >= value:
            self.mark['end' if part == 'start' else 'start'] = None

    def save_clip(self, negative=False):
        """The marked span as an example (or a negative) of the trim video; an error line or None."""
        start, end = self.mark['start'], self.mark['end']
        if not self.trim_file:
            return '구간은 영상에서 자릅니다: 영상 파일을 고르세요.'
        if start is None or end is None:
            return '구간의 시작([)과 끝(])을 먼저 표시하세요.'
        clip = dict(file=str(self.trim_file), start_s=round(start, 3), end_s=round(end, 3))
        if self.mark['at'] is not None:
            clip['at'] = [round(value, 3) for value in self.mark['at']]
        (self.negatives if negative else self.examples).append(clip)
        self.mark = dict(start=None, end=None, at=None)
        return None

    # -- the request and whether it can start
    def request(self):
        inputs = {}
        if self.pictures:
            inputs[self.pictures[0]] = self.pictures[1] if self.pictures[0] == 'camera' else str(self.pictures[1])
        if self.microphone:
            inputs['microphone'] = True
        elif self.audio:
            inputs['audio'] = str(self.audio)
        if self.sensors:
            inputs['sensors'] = [dict(item) for item in self.sensors]
        request = dict(inputs=inputs, mode=self.mode)
        if self.timing:
            request['timing'] = self.timing
        if self.name.strip() or self.text.strip() or self.photos:
            who = {key: value.strip() for key, value in (('name', self.name), ('text', self.text)) if value.strip()}
            if self.enrolled():
                who['person'] = str(self.enrolled())
            elif self.photos:
                who.update(photos=[str(path) for path in self.photos], consent=self.consent.strip() or INSTALL_CONSENT)
            if self.voice:
                who['voice'] = str(self.voice)
            request['who'] = who
        what = {}
        if self.action.strip() or self.rule or self.examples or self.negatives:
            movement = {}
            if self.action.strip():
                movement['text'] = self.action.strip()
            if self.rule:
                movement['rule'] = str(self.rule)
            if self.examples:
                movement['examples'] = [dict(item) for item in self.examples]
            if self.negatives:
                movement['negatives'] = [dict(item) for item in self.negatives]
            what['movement'] = movement
        if self.words:
            what['words'] = list(self.words)
        if self.rhythm:
            what['rhythm'] = dict(self.rhythm)
        if what:
            request['what'] = what
        if self.allow_remote_asr:           # the cost of a hosted, paid recogniser was accepted (a cost, not consent)
            request['consent'] = dict(remote_asr=True)
        return request

    def state(self):
        """``(ok, line)``: whether Start is on, and the one line that says what to add (or that it is ready)."""
        from .recipes.compose import validate_request
        checked = validate_request(self.request())
        if not checked['ok']:
            return False, korean(checked['missing'][0])
        if self.photos and not self.enrolled():
            if self.enrolling:
                return False, f'사진 {len(self.photos)}장을 검사하는 중입니다 (얼굴이 같은 사람인지, 한 장에 한 사람인지).'
            if self.person and self.person.get('key') == self._key() and self.person.get('error'):
                return False, korean(self.person['error'])
            return False, '사진 검사를 기다립니다.'
        if checked['before_run']:
            return False, korean(checked['before_run'][0])
        return True, '준비됐습니다: 시작을 누르세요 (Enter).'

    def photo_line(self):
        """``(text, colour)``: the photo count and the person check (faces agree, one person per photo)."""
        if not self.photos:
            return '사진 0장 (한 장에 한 사람, 3장 이상)', MUTED
        if self.enrolling:
            return f'사진 {len(self.photos)}장 · 검사 중...', LATE
        person = self.person if self.person and self.person.get('key') == self._key() else None
        if person is None:
            return f'사진 {len(self.photos)}장 · 이름이나 설명을 적으면 검사합니다', MUTED
        if person.get('error'):
            return f'사진 {len(self.photos)}장 · 검사 실패: ' + korean(person['error']), BAD
        face = person['check']['face']
        return (f"사진 {len(self.photos)}장 · 얼굴 {face['n']}개 일치 {face['mean']:.2f} (기준 0.5) · 사진마다 한 사람"
                if face.get('mean') is not None else f"사진 {len(self.photos)}장 · 얼굴 {face['n']}개"), GOOD


def _slug(text):
    found = re.sub(r'[^0-9A-Za-z가-힣_-]+', '-', text).strip('-')
    return found[:40] or 'person'


# ---------------------------------------------------------------- dialogs (text with Korean input, files)

def ask_text(prompt, default=''):
    """A line typed by the user (the system's own dialog, so Korean input works), or None when cancelled."""
    if sys.platform == 'darwin':
        script = ['-e', 'on run argv', '-e', 'text returned of (display dialog (item 1 of argv) default answer '
                  '(item 2 of argv) with title "GAMST")', '-e', 'end run']
        done = subprocess.run(['osascript', *script, prompt, default or ''], capture_output=True, text=True)
        return done.stdout.rstrip('\n') if done.returncode == 0 else None
    try:
        import tkinter
        from tkinter import simpledialog
        root = tkinter.Tk()
        root.withdraw()
        try:
            return simpledialog.askstring('GAMST', prompt, initialvalue=default or '', parent=root)
        finally:
            root.destroy()
    except Exception:
        try:
            return input(f'{prompt} [{default}]: ') or default
        except EOFError:
            return None


def ask_files(prompt, multiple=True):
    """Files chosen by the user (the system's own dialog), or []."""
    if sys.platform == 'darwin':
        script = ['-e', 'on run argv', '-e', 'set picked to choose file with prompt (item 1 of argv)' +
                  (' with multiple selections allowed' if multiple else ''), '-e', 'set out to ""',
                  '-e', 'if class of picked is not list then set picked to {picked}',
                  '-e', 'repeat with f in picked', '-e', 'set out to out & POSIX path of f & linefeed', '-e',
                  'end repeat', '-e', 'return out', '-e', 'end run']
        done = subprocess.run(['osascript', *script, prompt], capture_output=True, text=True)
        return [line for line in done.stdout.splitlines() if line] if done.returncode == 0 else []
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        try:
            found = filedialog.askopenfilenames(title=prompt) if multiple else filedialog.askopenfilename(title=prompt)
            return list(found) if multiple else [found] if found else []
        finally:
            root.destroy()
    except Exception:
        try:
            return [item for item in input(f'{prompt} (paths, separated by ;): ').split(';') if item.strip()]
        except EOFError:
            return []


# ---------------------------------------------------------------- the preview (camera or video) and its trim bar

class Preview:
    """The picture of the setup step: a video file (read only, scrubbed and played) or the camera."""

    def __init__(self):
        self.capture, self.source, self.frame = None, None, None
        self.position_s, self.duration_s, self.fps, self.playing = 0., None, 30., False
        self._shown_at = None

    def open(self, source):
        import cv2
        if source == self.source:
            return
        self.close()
        self.source = source
        if source is None:
            return
        kind, value = source
        self.capture = cv2.VideoCapture(int(value) if kind == 'camera' else str(value))
        if not self.capture.isOpened():
            self.capture = None
            return
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.
        frames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration_s = frames / self.fps if kind == 'video' and frames > 0 else None
        self.position_s, self.playing = 0., kind == 'camera'
        self._read()

    def _read(self):
        import cv2
        ok, frame = self.capture.read()
        if ok:
            self.frame = frame
            if self.duration_s is not None:
                self.position_s = max(0., self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.)
        elif self.duration_s is not None:
            self.playing = False

    def seek(self, seconds):
        import cv2
        if self.capture is None or self.duration_s is None:
            return
        seconds = min(max(0., seconds), max(0., self.duration_s - 1. / self.fps))
        self.capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.)
        self._read()

    def step(self):
        """Advance while playing, at the file's own pace."""
        if self.capture is None or not self.playing:
            return
        now = time.monotonic()
        if self._shown_at is None or now - self._shown_at >= 1. / self.fps or self.duration_s is None:
            self._shown_at = now
            self._read()

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.capture, self.source, self.frame, self.duration_s = None, None, None, None


def fit(frame, width, height):
    """``(image, scale, x0, y0)``: ``frame`` letterboxed into width x height."""
    import cv2
    import numpy as np
    canvas = np.full((height, width, 3), 8, np.uint8)
    if frame is None:
        return canvas, 1., 0, 0
    scale = min(width / frame.shape[1], height / frame.shape[0])
    w, h = max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))
    x0, y0 = (width - w) // 2, (height - h) // 2
    canvas[y0:y0 + h, x0:x0 + w] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return canvas, scale, x0, y0


# ---------------------------------------------------------------- the window

KEYS_LINE = ('스페이스 재생/멈춤 · , . 0.1초 · < > 1초 · [ 시작 · ] 끝 · e 예시 저장 · n 반례 저장 · 화면 클릭: 그 사람 고르기',
             'Enter 시작 · q 또는 Esc 끝내기')


class SetupWindow:
    """The setup step drawn in the live view's window; :meth:`loop` returns compose's answer once Start succeeded."""

    def __init__(self, setup, *, title=None, ask_text=ask_text, ask_files=ask_files, start_at_once=False):
        from .views import CountView, Writer
        self.setup, self.title = setup, title or CountView.title
        self.writer, self.preview = Writer(), Preview()
        self.ask_text, self.ask_files = ask_text, ask_files
        self.hits, self.composing, self.answer, self.start_at_once = [], None, None, start_at_once
        self.view = dict(scale=1., x0=0, y0=0)

    # -- drawing
    def put(self, image, line, x, y, color=WHITE, size=17):
        return self.writer.put(image, line, x, y, color, size)

    def button(self, image, x, y, label, action, *, on=False, enabled=True, size=16):
        import cv2
        width = self.writer.width(label, size) + 18
        fill = (60, 110, 60) if on else (44, 44, 44)
        cv2.rectangle(image, (x, y - 20), (x + width, y + 6), fill if enabled else (34, 34, 34), -1)
        cv2.rectangle(image, (x, y - 20), (x + width, y + 6), GOOD if on else EDGE, 1)
        self.put(image, label, x + 9, y, WHITE if enabled else MUTED, size)
        if enabled:
            self.hits.append(((x, y - 20, x + width, y + 6), action))
        return x + width + 8

    def draw(self):
        import numpy as np
        setup = self.setup
        self.hits = []
        image = np.full((HEIGHT, PREVIEW_W + PANEL_W, 3), PANEL[0], np.uint8)
        image[:, :PREVIEW_W] = BG[0]
        # left: the picture and the trim bar
        picture, scale, x0, y0 = fit(self.preview.frame, PREVIEW_W, PREVIEW_H)
        self.view = dict(scale=scale, x0=x0, y0=y0, w=picture.shape[1], h=picture.shape[0],
                         fw=self.preview.frame.shape[1] if self.preview.frame is not None else 1,
                         fh=self.preview.frame.shape[0] if self.preview.frame is not None else 1)
        if self.preview.frame is None:
            line = ('소리만: 화면 입력 없음' if setup.microphone or setup.audio or setup.sensors else
                    '입력(파일이나 장치)을 넣으면 여기에 화면이 보입니다')
            self.put(picture, line, 30, 60, MUTED, 20)
        image[:PREVIEW_H, :PREVIEW_W] = picture
        self._points(image)
        self.hits.append(((0, 0, PREVIEW_W, PREVIEW_H), ('pick',)))
        self._trim_bar(image, PREVIEW_H)
        y = PREVIEW_H + BAR_H + 30
        for line in KEYS_LINE:
            self.put(image, line, 16, y, MUTED, 15)
            y += 24
        if setup.notice:
            self.put(image, setup.notice, 16, y + 4, LATE, 16)
        # right: the panel
        x, y = PREVIEW_W + 20, 36
        self.put(image, 'GAMST 준비: 무엇을, 누구를 셀지 정합니다', x, y, ACCENT, 21)
        y += 44
        self.put(image, '1 입력', x, y, ACCENT, 18)
        at = x + 70
        at = self.button(image, at, y, '파일', ('input_file',))
        at = self.button(image, at, y, '장치', ('input_device',))
        self.button(image, at, y, '비우기', ('clear', 'inputs'),
                    enabled=bool(setup.pictures or setup.microphone or setup.audio or setup.sensors))
        y += 28
        shown = []
        if setup.pictures:
            shown.append(f'카메라 {setup.pictures[1]}' if setup.pictures[0] == 'camera' else
                         f'영상 {Path(setup.video).name}' + (f' ({_clock(self.preview.duration_s)})'
                                                            if self.preview.duration_s else ''))
        if setup.microphone:
            shown.append('마이크')
        if setup.audio:
            shown.append(f'소리 {Path(setup.audio).name}')
        for item in setup.sensors:
            shown.append(f"센서 {item['id']}")
        self.put(image, ' · '.join(shown) or '아직 없음 (종류는 파일이나 장치를 보고 정합니다)', x + 16, y,
                 WHITE if shown else MUTED, 15)
        if setup.sensors:
            y += 26
            rhythm = setup.rhythm or {}
            at = self.put(image, f"리듬: {rhythm.get('channel', '-')} / {rhythm.get('band', '-')}", x + 16, y, WHITE, 15)
            at = self.button(image, x + 30 + at, y, '채널', ('rhythm', 'channel'), size=14)
            self.button(image, at, y, '대역', ('rhythm', 'band'), size=14)
        y += 40
        self.put(image, '2 누구 (선택)', x, y, ACCENT, 18)
        at = self.button(image, x + 130, y, '이름', ('ask', 'name', '누구를 셀까요? 이름'))
        self.button(image, at, y, '설명', ('ask', 'text', '그 사람의 설명 (예: 탈색 짧은 머리, 검은 뿔테)'))
        y += 28
        named = ' · '.join(part for part in (setup.name.strip(), setup.text.strip()) if part)
        self.put(image, named or '이름이나 설명 없음 (사람을 정하지 않으면 모두를 셉니다)', x + 16, y, WHITE if named else MUTED, 15)
        y += 56
        at = self.button(image, x + 16, y, '사진 파일 추가', ('photos',), size=15)
        at = self.button(image, at, y, '지금 화면을 사진으로', ('grab',), enabled=self.preview.frame is not None, size=15)
        self.button(image, at, y, '사진 비우기', ('clear', 'photos'), enabled=bool(setup.photos), size=15)
        y += 28
        line, color = setup.photo_line()
        self.put(image, line[:80], x + 16, y, color, 15)
        y += 40
        self.put(image, '3 무엇', x, y, ACCENT, 18)
        at = self.button(image, x + 80, y, '동작 글', ('ask', 'action', '셀 동작을 글로 (예: 두 팔을 앞으로 오므렸다가 양옆으로 폈다가 다시 접는다)'))
        at = self.button(image, at, y, '규칙 파일', ('rule',))
        self.button(image, at, y, '단어', ('words',))
        y += 28
        self.put(image, '동작: ' + (setup.action.strip()[:52] or '없음'), x + 16, y, WHITE if setup.action.strip() else MUTED,
                 15)
        y += 24
        self.put(image, '규칙: ' + (Path(setup.rule).name if setup.rule else '없음'), x + 16, y,
                 WHITE if setup.rule else MUTED, 15)
        y += 24
        clips = f'예시 {len(setup.examples)}개 · 반례 {len(setup.negatives)}개'
        if setup.trim_file and not setup.video:
            clips += f' (영상 {Path(setup.trim_file).name})'
        at = self.put(image, clips, x + 16, y, WHITE if setup.examples else MUTED, 15)
        at = self.button(image, x + 32 + at, y, '구간 비우기', ('clear', 'clips'), enabled=bool(setup.examples or
                                                                                         setup.negatives), size=14)
        if setup.pictures and setup.pictures[0] == 'camera':
            self.button(image, at, y, '카메라 보기' if setup.clip_file else '예시 영상 고르기',
                        ('camera_view',) if setup.clip_file else ('clipfile',), size=14)
        y += 24
        mark = setup.mark
        marked = (f"구간: 시작 {_seconds(mark['start'])} · 끝 {_seconds(mark['end'])}" + (
            f" · 사람 ({mark['at'][0]:.2f}, {mark['at'][1]:.2f})" if mark['at'] else ' · 화면을 눌러 사람을 고르세요'))
        self.put(image, marked, x + 16, y, LATE if mark['start'] is not None or mark['end'] is not None else MUTED, 15)
        y += 24
        self.put(image, '단어: ' + (', '.join(setup.words) or '없음'), x + 16, y, WHITE if setup.words else MUTED, 15)
        y += 40
        self.put(image, '4 방식', x, y, ACCENT, 18)
        at = self.button(image, x + 80, y, '세기', ('mode', 'count'), on=setup.mode == 'count')
        at = self.button(image, at, y, '찾기', ('mode', 'search'), on=setup.mode == 'search')
        at = self.put(image, '때', at + 20, y, ACCENT, 18) + at + 32
        timing = setup.timing or ('realtime' if setup.pictures and setup.pictures[0] == 'camera' or setup.microphone
                                  else 'post')
        at = self.button(image, at, y, '실시간', ('timing', 'realtime'), on=timing == 'realtime')
        self.button(image, at, y, '나중에(파일 전체)', ('timing', 'post'), on=timing == 'post')
        y += 26
        self.put(image, ('찾기: 놓치지 않기, 애매한 것은 후보로 셉니다' if setup.mode == 'search' else
                         '세기: 규칙과 예시가 모두 맞을 때 셉니다') + ' · ' +
                 ('실시간: 늦은 프레임은 L2가 나중에 고칩니다(LATE)' if timing == 'realtime' else '나중에: 모든 프레임, 늦는 것 없음'),
                 x + 16, y, MUTED, 14)
        # the one line, and Start
        ok, line = setup.state()
        if self.composing:
            ok, line = False, self.composing
        y = HEIGHT - 86
        self.put(image, line[:70], x, y, GOOD if ok else BAD, 17)
        if len(line) > 70:
            self.put(image, line[70:140], x, y + 22, GOOD if ok else BAD, 17)
        self.button(image, x, HEIGHT - 26, '  시작  ', ('start',), on=ok, enabled=ok, size=20)
        return image, ok

    def _trim_bar(self, image, top):
        import cv2
        setup, duration = self.setup, self.preview.duration_s
        left, right = 16, PREVIEW_W - 16
        cv2.rectangle(image, (left, top + 14), (right, top + 38), (40, 40, 40), -1)
        if not duration:
            self.put(image, '구간 자르기: 영상 파일을 고르면 여기에서 시작과 끝을 표시합니다', left, top + 58, MUTED, 14)
            return
        px = lambda seconds: int(left + (right - left) * min(max(seconds / duration, 0.), 1.))
        for clip, color in [(item, GOOD) for item in setup.examples] + [(item, BAD) for item in setup.negatives]:
            if str(clip['file']) == str(setup.trim_file):
                cv2.rectangle(image, (px(clip['start_s']), top + 14), (max(px(clip['end_s']), px(clip['start_s']) + 2),
                                                                       top + 38), color, -1)
        start, end = setup.mark['start'], setup.mark['end']
        if start is not None:
            cv2.line(image, (px(start), top + 8), (px(start), top + 44), LATE, 2)
        if end is not None:
            cv2.line(image, (px(end), top + 8), (px(end), top + 44), LATE, 2)
        if start is not None and end is not None:
            cv2.rectangle(image, (px(start), top + 30), (px(end), top + 38), LATE, -1)
        cv2.line(image, (px(self.preview.position_s), top + 4), (px(self.preview.position_s), top + 48), WHITE, 2)
        self.put(image, f'{_seconds(self.preview.position_s)} / {_seconds(duration)}' +
                 (' · 재생 중' if self.preview.playing else ''), left, top + 62, WHITE, 14)
        self.put(image, '초록 예시 · 빨강 반례 · 주황 지금 표시한 구간', right - 300, top + 62, MUTED, 14)
        self.hits.append(((left, top, right, top + 50), ('seek', left, right)))

    def _points(self, image):
        """The person chosen for the marked clip, and those of saved clips showing now."""
        import cv2
        view, setup = self.view, self.setup
        points = [(setup.mark['at'], LATE)] if setup.mark['at'] else []
        now = self.preview.position_s
        points += [(clip['at'], GOOD if clip in setup.examples else BAD) for clip in setup.examples + setup.negatives
                   if clip.get('at') and clip['start_s'] <= now <= clip['end_s'] and str(clip['file']) == str(setup.trim_file)]
        for (x, y), color in points:
            cx, cy = int(view['x0'] + x * view['fw'] * view['scale']), int(view['y0'] + y * view['fh'] * view['scale'])
            cv2.circle(image, (cx, cy), 14, color, 2)
            cv2.line(image, (cx - 22, cy), (cx + 22, cy), color, 1)
            cv2.line(image, (cx, cy - 22), (cx, cy + 22), color, 1)

    # -- actions
    def act(self, action, x=None, y=None):
        setup, kind = self.setup, action[0]
        setup.notice = None
        if kind == 'input_file':        # the kind is read from the file, never asked
            for path in self.ask_files('입력 파일: 영상, 소리(.wav), 센서 설명(.json)'):
                setup.notice = setup.add_input(path) or setup.notice
        elif kind == 'input_device':
            value = self.ask_text('장치: 카메라 번호(0이 첫 카메라) 또는 마이크', '0')
            if value is not None:
                setup.notice = setup.add_input(value)
        elif kind == 'rhythm':
            setup.cycle_rhythm(action[1])
        elif kind == 'ask':
            value = self.ask_text(action[2], getattr(setup, action[1]))
            if value is not None:
                setattr(setup, action[1], value)
        elif kind == 'photos':
            setup.add_photos(self.ask_files('사람 사진 (한 장에 한 사람)'))
        elif kind == 'grab':
            self.grab()
        elif kind == 'clear':
            if action[1] == 'photos':
                setup.photos, setup.person = [], None
            elif action[1] == 'inputs':
                setup.clear_inputs()
            else:
                setup.examples, setup.negatives = [], []
        elif kind == 'rule':
            found = self.ask_files('동작 규칙 파일 (JSON, 예: docs/movements/gwanjetap.rule.json)', multiple=False)
            setup.rule = Path(found[0]).absolute() if found else setup.rule
        elif kind == 'clipfile':
            found = self.ask_files('예시를 자를 영상 파일', multiple=False)
            if found:
                setup.clip_file = Path(found[0]).absolute()
        elif kind == 'camera_view':
            setup.clip_file = None
        elif kind == 'words':
            value = self.ask_text('셀 단어 (쉼표로 나눕니다; re:패턴, word:낱말 도 됩니다)', ', '.join(setup.words))
            if value is not None:
                setup.words = [word.strip() for word in value.split(',') if word.strip()]
        elif kind == 'mode':
            setup.mode = action[1]
        elif kind == 'timing':
            setup.timing = action[1]
        elif kind == 'seek':
            left, right = action[1], action[2]
            if self.preview.duration_s:
                self.preview.seek((x - left) / (right - left) * self.preview.duration_s)
        elif kind == 'pick':
            view = self.view
            if self.preview.frame is not None and setup.trim_file and view['x0'] <= x < view['x0'] + view['fw'] * \
                    view['scale'] and view['y0'] <= y < view['y0'] + view['fh'] * view['scale']:
                setup.mark['at'] = [round((x - view['x0']) / (view['fw'] * view['scale']), 3),
                                    round((y - view['y0']) / (view['fh'] * view['scale']), 3)]
        elif kind == 'start':
            self.start()

    def source(self):
        """What the picture shows: the clip file being trimmed (a camera input), else the input."""
        setup = self.setup
        return ('video', setup.clip_file) if setup.clip_file and not setup.video else setup.pictures

    def grab(self):
        """The picture shown now, saved as a photo of the person (into the setup folder) and added."""
        import cv2
        if self.preview.frame is None:
            return
        folder = self.setup.out_dir / 'photos'
        folder.mkdir(parents=True, exist_ok=True)
        stem = Path(self.setup.video).stem[:24] if self.setup.video else 'camera'
        path = folder / f"{_slug(stem)}_{self.preview.position_s:08.3f}.jpg"
        cv2.imwrite(str(path), self.preview.frame)
        self.setup.add_photos([path])
        self.setup.notice = f'사진 추가: {path.name} (지금 화면)'

    def start(self):
        """Compose in the background (example clips are enrolled, which runs the pose model); the answer ends the loop."""
        from .recipes.compose import compose
        ok, _ = self.setup.state()
        if not ok or self.composing:
            return
        request = self.setup.request()
        self.composing = '준비 중: 레시피를 짜고 예시 구간을 배우는 중입니다...'

        def work():
            try:
                answer = compose(request, out_dir=self.setup.out_dir, profile=self.setup.profile)
            except Exception as error:
                answer = dict(ok=False, missing=[f'{type(error).__name__}: {error}'])
            if answer.get('ok') and not answer.get('before_run'):
                self.answer = answer
            else:
                self.setup.notice = korean((answer.get('missing') or answer.get('before_run') or ['?'])[0])
            self.composing = None
        threading.Thread(target=work, daemon=True).start()

    def key(self, key):
        setup, preview = self.setup, self.preview
        if key in (27, ord('q')):
            return False
        if key == ord(' '):
            preview.playing = not preview.playing
        elif key in (ord(','), ord('.'), ord('<'), ord('>'), 63234, 63235, 65361, 65363, 2424832, 2555904):
            step = {ord(','): -.1, ord('.'): .1, ord('<'): -1., ord('>'): 1., 63234: -1., 65361: -1., 2424832: -1.}.get(
                key, 1.)
            preview.playing = False
            preview.seek(preview.position_s + step)
        elif key == ord('['):
            setup.set_mark('start', round(preview.position_s, 3))
        elif key == ord(']'):
            setup.set_mark('end', round(preview.position_s, 3))
        elif key in (ord('e'), ord('n')):
            setup.notice = setup.save_clip(negative=key == ord('n'))
        elif key in (13, 10):
            self.start()
        return True

    def loop(self):
        """Draw and listen until Start succeeds (compose's answer) or the user quits (None)."""
        import cv2
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)

        def mouse(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            for (x0, y0, x1, y1), action in reversed(self.hits):
                if x0 <= x < x1 and y0 <= y < y1:
                    self.pending.append((action, x, y))
                    return
        self.pending = []
        cv2.setMouseCallback(self.title, mouse)
        try:
            while self.answer is None:
                self.preview.open(self.source())
                self.preview.step()
                self.setup.enroll()
                image, ok = self.draw()
                cv2.imshow(self.title, image)
                if ok and self.start_at_once:
                    self.start_at_once = False
                    self.start()
                key = cv2.waitKeyEx(15)
                while self.pending:
                    action, x, y = self.pending.pop(0)
                    self.act(action, x, y)
                if key != -1 and not self.key(key if key > 255 and key in (63234, 63235, 65361, 65363, 2424832,
                                                                            2555904) else key & 0xFF):
                    return None
        finally:
            self.preview.close()
        return self.answer


def _seconds(value):
    return '?' if value is None else f'{value:.2f}초'


def _clock(seconds):
    seconds = int(round(seconds or 0))
    return f'{seconds // 60}분 {seconds % 60}초' if seconds >= 60 else f'{seconds}초'


# ---------------------------------------------------------------- command line

def setup_from(args):
    """A :class:`Setup` filled in with what the command line gives."""
    setup = Setup(args.out_dir, profile=args.profile)
    given = list(args.input or ())          # the older per-kind options are read the same way (the kind is detected)
    given += [str(args.camera)] if args.camera is not None else []
    given += [args.video] if args.video else []
    given += ['microphone'] if args.microphone else []
    given += list(args.sensor or ())
    for value in given:
        problem = setup.add_input(value)
        if problem:
            raise SystemExit(f'--input {value}: {problem}')
    setup.name, setup.text = args.who or '', args.who_text or ''
    setup.add_photos(args.photo or ())
    setup.voice = args.voice
    setup.action, setup.rule = args.action or '', Path(args.rule).absolute() if args.rule else None
    for value, kept in ((args.example, setup.examples), (args.negative, setup.negatives)):
        for item in value or ():
            clip = parse_clip(item)
            if clip is None:
                raise SystemExit(f'{item}: a clip is FILE@START-END[@X,Y]')
            kept.append(dict(clip, file=str(Path(clip['file']).absolute())))
    setup.words = [word.strip() for word in (args.words or '').split(',') if word.strip()]
    setup.mode, setup.timing, setup.allow_remote_asr = args.mode, args.timing, args.allow_remote_asr
    return setup


def run_command(answer, setup, extra=()):
    """The ``gamst recipe run`` arguments of compose's answer, with a window (not headless), the setup profile, and a
    video file replayed at its own pace when the timing is realtime (a camera stand-in)."""
    argv = [item for item in answer['run'][3:] if item != '--headless']
    if setup.profile:
        argv += ['--profile', str(setup.profile)]
    if setup.video and 'realtime' in argv[argv.index('--timing') + 1:argv.index('--timing') + 2]:
        argv.append('--realtime')
    return argv + list(extra)


def options(parser):
    parser.add_argument('--input', action='append', metavar='FILE|DEVICE',
                        help='an input; its kind is detected: a video file (its sound for words), a WAV file, a sensor '
                             'declaration .json ({id, file, modality, columns, time}), a camera number (0) or mic')
    parser.add_argument('--camera', type=int, help=argparse.SUPPRESS)         # the same as --input N
    parser.add_argument('--video', help=argparse.SUPPRESS)                    # the same as --input FILE
    parser.add_argument('--microphone', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--sensor', action='append', help=argparse.SUPPRESS)
    parser.add_argument('--who', metavar='NAME', help='who to count (a name); photos are needed too')
    parser.add_argument('--who-text', metavar='TEXT', help='a description of the person')
    parser.add_argument('--photo', action='append', metavar='JPG', help='a photo of the person (one person in it)')
    parser.add_argument('--voice', metavar='V.json', help="the person's enrolled voice (words said by the person)")
    parser.add_argument('--action', metavar='TEXT', help='the movement in words (with --rule, and/or example clips)')
    parser.add_argument('--rule', metavar='RULE.json', help='the movement as a rule (gamst/providers/movement_rule.py)')
    parser.add_argument('--example', action='append', metavar='FILE@A-B[@X,Y]', help='an example clip of the movement')
    parser.add_argument('--negative', action='append', metavar='FILE@A-B[@X,Y]', help='a look-alike that is not it')
    parser.add_argument('--words', metavar='W,W', help='words to count')
    parser.add_argument('--mode', choices=('count', 'search'), default='count')
    parser.add_argument('--timing', choices=('realtime', 'post'))
    parser.add_argument('--allow-remote-asr', action='store_true',
                        help='accept the cost of a hosted, paid recogniser for words (a cost notice, not consent)')
    parser.add_argument('--out-dir', default='gamst-hud', help='where the composed recipe, enrolled files and grabbed '
                        'photos are saved (default ./gamst-hud)')
    parser.add_argument('--profile', help='setup profile (the models photos and clips are enrolled and counted with)')
    parser.add_argument('--start', action='store_true', help='start at once when nothing is missing')


def main(argv=None):
    given = list(sys.argv[1:] if argv is None else argv)
    extra = given[given.index('--') + 1:] if '--' in given else []
    given = given[:given.index('--')] if '--' in given else given
    parser = argparse.ArgumentParser(prog='gamst recipe hud', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    options(parser)
    args = parser.parse_args(given)
    setup = setup_from(args)
    answer = SetupWindow(setup, start_at_once=args.start).loop()
    if answer is None:
        import cv2
        cv2.destroyAllWindows()
        return None
    argv = run_command(answer, setup, extra)
    print(json.dumps(dict(composed=answer['saved'], run=['gamst', 'recipe', 'run', *argv]), ensure_ascii=False),
          file=sys.stderr, flush=True)
    from .recipes import main as recipe
    from .views import CountView
    CountView.hold_at_end = True        # the evidence can still be read after the last frame
    return recipe(['run', *argv])


if __name__ == '__main__':
    main()
