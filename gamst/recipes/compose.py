"""Compose a recipe from a structured request: inputs, who, what, mode, timing (owner decision 46).

GAMST has no hidden LLM. The calling LLM (or the owner) fills in a request; GAMST picks the lanes that answer each part
of it, links them by ``consumes``, validates the graph and returns a recipe that can be saved and re-run, with the reason
for every choice. A part that is missing comes back as one sentence that says what to provide.

  gamst recipe compose REQUEST.json [--out FILE] [--start]       (MCP: compose_recipe, the same request)

Request::

  {"name": "shotgun-by-gamst",                        optional; default composed-<digest of the request>
   "recipe": "person-action-count",                   optional: use a built-in recipe as it is, or customised
   "inputs": {"camera": 0 | "video": FILE,            pictures: a camera index or a video file (its sound too)
              "microphone": true | "audio": FILE,     sound
              "sensors": [{"id", "file", "modality", "columns", "time"}]},
   "who": {"name": "P", "text": "a description", "photos": [JPG, ...] | "person": P.json, "consent": "...",
           "voice": V.json} | null,    a text (name or description) AND a photo few-shot are required (decision 59)
   "what": {"movement": {"text", "rule": {...} | FILE, "examples": [CLIP, ...], "negatives": [CLIP, ...],
                         "combine": "all" | "any"} | {"exemplar": FILE},
            (text + rule: zero-shot; examples: few-shot; text + rule + examples: the rule is the skeleton, the clips set
             its numbers and are compared too, decision 58; CLIP = {"file", "start_s", "end_s"} or "FILE@START-END")
            "words": ["W", ...] | FILE,
            "rhythm": {"input", "channel", "band"}},
   "mode": "count" | "search", "timing": "realtime" | "post",
   "pose": CANDIDATE, "asr": CANDIDATE, "with": [...], "without": [...], "set": ["MODULE.KEY=VALUE"],
   "consent": {"capture": true, "remote_asr": true}, "start": false, "wait": false}

How the lanes are chosen (:data:`CAPABILITIES`): each capability names the lane that answers it, taken from the built-in
recipe it was measured in, so a composed plan is the hand-assembled one (issue #369's acceptance); every ``use`` is a
part of :data:`gamst.recipes.parts.PARTS`. The pose model is the measured rule of decision 38 (RTMW-l, benchmarked at
the start of a realtime run with MediaPipe full as its fallback; post mode counts every frame with it). Who adds the
person binding and its few-shot reference, checked here by :mod:`gamst.person_reference`. Words add the word lanes of
``person-term-count`` unchanged (decision 48). The semantic VLM lane is in every composed recipe with a picture input,
OFF (decision 47).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

KEYS = ('name', 'recipe', 'inputs', 'who', 'what', 'mode', 'timing', 'pose', 'asr', 'with', 'without', 'set', 'consent',
        'start', 'wait')
INPUT_KEYS = ('camera', 'video', 'microphone', 'audio', 'sensors')
WHAT_KEYS = ('movement', 'words', 'rhythm')
VIDEO_FILES = ('.mp4', '.mov', '.m4v', '.mkv', '.webm', '.avi')
SOUND_FILES = ('.wav',)
SENSOR_FILES = ('.csv', '.jsonl', '.ndjson')
MAX_FILE = 4 << 20
MAX_MEDIA = 8 << 30

# capability -> (the built-in recipe it was measured in, its module items, its references, what it answers)
CAPABILITIES = {
    'presence': ('person-action-count', ('pose', 'person_binding'), ('person',),
                 'when an enrolled person is on screen: pose tracks every person, the binding names the tracks'),
    'movement': ('person-action-count', ('pose', 'person_binding', 'target_person', 'pose_channels', 'exemplar',
                                         'repetition'), ('exemplar', 'person'),
                 'a movement counted for every tracked person, each event with start, end and who'),
    'words': ('person-term-count', None, ('voice', 'person', 'terms'),
              'words counted as they are said, with the word lanes of person-term-count unchanged (decision 48)'),
    'semantic': ('person-action-count', ('semantic',), (),
                 'a VLM interpretation of segments as a causal state chain; OFF (decision 47)'),
}
POSE_RULE = ('decision 38: RTMW-l 384x288 is the default; {timing}')
REALTIME_POSE = ('realtime: at the start a short benchmark measures it (live and one L2 worker at once) and keeps it '
                 'when together they reach the input frame rate, else MediaPipe full counts and the report, HUD and '
                 'get_pipeline say so (gamst.recipes.choose_pose); never two models in one run')
POST_POSE = 'post: a finished file is counted on every frame with nothing late, so the most accurate model counts'


class Missing(Exception):
    """What the request lacks, one sentence each."""

    def __init__(self, sentences):
        super().__init__('\n'.join(sentences))
        self.sentences = list(sentences)


def _items(raw, names):
    found = {item['name']: item for item in raw['modules']}
    return [deepcopy(found[name]) for name in (names or found)]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


class _Files:
    """Request files: anywhere for the owner on the command line (``folders`` None), only inside the owner's ``--clips``
    folders over MCP (:func:`gamst.recipes.instructions.inside`)."""

    def __init__(self, folders, missing):
        self.folders, self.missing = folders, missing

    def get(self, value, suffixes, where, limit=MAX_FILE):
        if not isinstance(value, str) or not value:
            self.missing.append(f'{where}: name a file ({", ".join(suffixes)}).')
            return None
        if self.folders is not None:
            from .instructions import inside
            try:
                return inside(self.folders, value, suffixes, limit)
            except ValueError as error:
                self.missing.append(f'{where}: {error}.')
                return None
        path = Path(value).expanduser()
        if not path.is_file():
            self.missing.append(f'{where}: there is no file {value}.')
            return None
        if path.suffix.lower() not in suffixes:
            self.missing.append(f'{where}: {value} is not one of {", ".join(suffixes)}.')
            return None
        return path.absolute()


def _json_file(path, kind, where, missing):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        missing.append(f'{where}: {path} is not readable JSON ({error}).')
        return None
    if not isinstance(value, dict) or value.get('kind') != kind:
        missing.append(f'{where}: {path} is not an enrolled {kind} file (gamst recipe enroll --kind {kind}).')
        return None
    return value


def _inputs(request, files, missing):
    """{'pictures': (kind, value) | None, 'sound': (kind, value) | None, 'sensors': [item]}"""
    given = request.get('inputs') or {}
    if not isinstance(given, dict) or set(given) - set(INPUT_KEYS):
        missing.append(f'inputs: an object of {", ".join(INPUT_KEYS)}, each optional.')
        return dict(pictures=None, sound=None, sensors=[])
    pictures = sound = None
    if 'camera' in given and 'video' in given:
        missing.append('inputs: give one picture source, camera or video, not both.')
    elif 'camera' in given:
        index = 0 if given['camera'] is True else given['camera']
        if type(index) is not int or index < 0:
            missing.append('inputs.camera: a camera index (0 for the first camera).')
        else:
            pictures = ('camera', index)
    elif 'video' in given:
        path = files.get(given['video'], VIDEO_FILES, 'inputs.video', MAX_MEDIA)
        pictures = ('video', path) if path else None
    if 'microphone' in given and 'audio' in given:
        missing.append('inputs: give one sound source, microphone or audio, not both.')
    elif given.get('microphone') is True:
        sound = ('microphone', True)
    elif 'microphone' in given:
        missing.append('inputs.microphone: true for the default microphone.')
    elif 'audio' in given:
        path = files.get(given['audio'], SOUND_FILES, 'inputs.audio', MAX_MEDIA)
        sound = ('audio', path) if path else None
    sensors = []
    for index, item in enumerate(given.get('sensors') or ()):
        where = f'inputs.sensors[{index}]'
        if not isinstance(item, dict) or not {'id', 'file', 'columns', 'time'} <= set(item):
            missing.append(f'{where}: {{id, file, modality, columns: [{{name, unit}}], time: {{column, unit}}}}.')
            continue
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,31}', str(item['id'])) or item['id'] in ('camera', 'microphone'):
            missing.append(f'{where}.id: a short lowercase name other than camera and microphone.')
            continue
        path = files.get(item['file'], SENSOR_FILES, f'{where}.file', MAX_MEDIA)
        if path:
            sensors.append(dict(item, file=path))
    if not (pictures or sound or sensors) and not missing:
        missing.append('inputs: give at least one input: camera (an index), video (a file), microphone (true), audio '
                       '(a WAV file) or sensors.')
    return dict(pictures=pictures, sound=sound, sensors=sensors)


def _timing(request, inputs, missing):
    live = (inputs['pictures'] or ('',))[0] == 'camera' or (inputs['sound'] or ('',))[0] == 'microphone'
    timing = request.get('timing') or ('realtime' if live else 'post')
    if timing not in ('realtime', 'post'):
        missing.append('timing: realtime (count as frames arrive, late frames corrected by L2) or post (a finished '
                       'file, every frame, nothing late).')
    elif timing == 'post' and live:
        missing.append('timing: post counts a finished file; a camera or microphone is realtime: say realtime, or give '
                       'a video or audio file.')
    mode = request.get('mode') or 'count'
    if mode not in ('count', 'search'):
        missing.append('mode: count (default) or search (never miss the person: unclear ones become candidates).')
    return timing, mode


def _who(request, inputs, files, missing, *, words):
    """{name, person (path), voice (path) | None, check} or None."""
    who = request.get('who')
    if who in (None, {}):
        return None
    from ..person_reference import check, name_of, refusal, session_of
    if isinstance(who, str):
        who = dict(name=who)
    if not isinstance(who, dict) or set(who) - {'name', 'text', 'photos', 'person', 'consent', 'voice'}:
        missing.append('who: {"name" or "text": who it is (a name or a description), "photos": photos of that person '
                       '(few-shot, with "consent") or "person": an enrolled person file, "voice": an enrolled voice file '
                       '(for words)}.')
        return None
    texts = [who.get(key) for key in ('name', 'text') if isinstance(who.get(key), str) and who[key].strip()]
    if not texts:
        missing.append('who: say who it is in words as well ("name" or "text": a name or a description); photos alone '
                       'do not say whom the count is for (decision 59).')
        return None
    name = (who.get('name') or who['text']).strip()[:64]
    description = who.get('text') if isinstance(who.get('text'), str) else None
    if who.get('photos') and who.get('person'):
        missing.append('who: give photos (enrolled now) or an enrolled person file, not both.')
        return None
    if who.get('photos'):
        photos = who['photos']
        if type(photos) is not list or not 1 <= len(photos) <= 16:
            missing.append('who.photos: 1..16 photos of the person, exactly one person in each.')
            return None
        if not isinstance(who.get('consent'), str) or not who['consent'].strip():
            missing.append(refusal(name, ['no consent record (who consented, to what, when): add who.consent']))
            return None
        from .instructions import PHOTOS
        paths = [files.get(item, PHOTOS, f'who.photos[{index}]', MAX_FILE * 8) for index, item in enumerate(photos)]
        if None in paths:
            return None
        if words:
            missing.append('who.voice: words said by a person are told apart by the voice: enroll the person with a voice '
                           'first (gamst recipe enroll --kind person ... --voice-file V.json) and give who.person.')
            return None
        return dict(name=name, text=description, person=None, photos=paths, consent=who['consent'], voice=None,
                    check=None)
    if not who.get('person'):
        missing.append(refusal(name, [f'a text ({name!r}) alone never identifies a person: add who.photos (with '
                                      'who.consent) or who.person (an enrolled person file)']))
        return None
    path = files.get(who['person'], ('.json',), 'who.person')
    value = _json_file(path, 'person', 'who.person', missing) if path else None
    if value is None:
        return None
    if who.get('name') and name_of(value, name) != name:
        missing.append(f'who.person: {who["person"]} enrolls {name_of(value)!r}, not {name!r}: give that person\'s file '
                       f'or ask for who {name_of(value)!r}.')
        return None
    pictures = inputs['pictures']
    session = session_of(pictures[1]) if pictures and pictures[0] == 'video' else None
    verdict = check(value, session=session)
    if not verdict['ok']:
        missing.append(refusal(name, verdict['problems']))
        return None
    voice = None
    if words:
        if not who.get('voice'):
            missing.append(f"who.voice: words said by {name} are told apart by {name}'s voice: give a voice file "
                           "(gamst recipe enroll person-term-count --kind voice ... --out V.json), or leave who out to "
                           'count words said by anyone.')
        else:
            voice = files.get(who['voice'], ('.json',), 'who.voice')
            if voice and _json_file(voice, 'voice', 'who.voice', missing) is None:
                voice = None
    return dict(name=name_of(value, name) if not who.get('name') else name, text=description, person=path, photos=None,
                voice=voice, check=verdict)


def _clip_text(item, files, where):
    """``FILE@START-END`` (the enroll form) of one clip given as that string or as {file, start_s, end_s}; None if bad."""
    from .instructions import VIDEO
    if isinstance(item, str):
        found = re.fullmatch(r'(.+)@(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)', item)
        item = dict(file=found[1], start_s=float(found[2]), end_s=float(found[3])) if found else dict(file=item)
    if not isinstance(item, dict) or set(item) - {'file', 'start_s', 'end_s'}:
        files.missing.append(f'{where}: a clip is {{"file", "start_s", "end_s"}} or "FILE@START-END".')
        return None
    path = files.get(item.get('file'), VIDEO, where, MAX_MEDIA)
    if path is None:
        return None
    start, end = item.get('start_s'), item.get('end_s')
    if start is None and end is None:
        return str(path)
    from .movement import MAX_CLIP_S
    if type(start) not in (int, float) or type(end) not in (int, float) or not 0 < end - start <= MAX_CLIP_S or start < 0:
        files.missing.append(f'{where}: start_s < end_s, at most {MAX_CLIP_S:g} s apart (one movement per clip).')
        return None
    return f'{path}@{start:g}-{end:g}'


MOVEMENT_KEYS = ('text', 'rule', 'examples', 'negatives', 'combine', 'exemplar', 'label', 'clips')   # clips: examples


def _movement(what, files, missing):
    """{exemplar path | None, rule, text, label, examples, negatives, combine} of what.movement, or None.

    Text only is refused (GAMST has no hidden LLM to read it): text + rule (zero-shot), examples (few-shot) or text +
    rule + examples (decision 58), or an enrolled movement file."""
    movement = what.get('movement')
    if movement is None:
        return None
    if not isinstance(movement, dict) or set(movement) - set(MOVEMENT_KEYS):
        missing.append('what.movement: {"text" with "rule" (JSON of gamst/providers/movement_rule.py, written by the '
                       'caller), and/or "examples" (clips of the movement, with optional "negatives"), or "exemplar" (an '
                       'enrolled movement file)}.')
        return None
    if movement.get('clips'):
        movement = dict(movement, examples=movement['clips'])
    examples, negatives = movement.get('examples') or [], movement.get('negatives') or []
    from ..providers.movement_match import MAX_EXAMPLES
    if type(examples) is not list or type(negatives) is not list or len(examples) > MAX_EXAMPLES or \
            len(negatives) > MAX_EXAMPLES:
        missing.append(f'what.movement.examples / negatives: lists of at most {MAX_EXAMPLES} clips.')
        return None
    if negatives and not examples:
        missing.append('what.movement.negatives: look-alikes go with examples of the movement (what.movement.examples).')
        return None
    if movement.get('combine') not in (None, 'all', 'any'):
        missing.append("what.movement.combine: 'all' (the rule and an example: count mode's default) or 'any' (either: "
                       "search mode's default).")
        return None
    if movement.get('exemplar') and (movement.get('rule') or examples):
        missing.append('what.movement: an enrolled movement file stands alone; give it, or a rule and/or examples.')
        return None
    if movement.get('exemplar'):
        path = files.get(movement['exemplar'], ('.json',), 'what.movement.exemplar')
        if path is None:
            return None
        try:
            value = json.loads(Path(path).read_text())
        except (OSError, ValueError) as error:
            missing.append(f'what.movement.exemplar: {movement["exemplar"]} is not readable JSON ({error}).')
            return None
        if not isinstance(value, dict) or value.get('kind') not in (None, 'exemplar'):
            missing.append(f'what.movement.exemplar: {movement["exemplar"]} is not an enrolled movement file '
                           '(gamst recipe enroll).')
            return None
        return dict(exemplar=path, rule=None, text=movement.get('text') or value.get('text'),
                    label=value.get('label') or movement.get('text') or Path(path).stem, examples=[], negatives=[])
    rule = movement.get('rule')
    if rule is None and not examples:
        missing.append('what.movement: GAMST has no hidden LLM to read a movement in words: write it as a rule '
                       '(what.movement.rule, the JSON of gamst/providers/movement_rule.py, with its text), give example '
                       'clips (what.movement.examples), both, or an enrolled movement file (what.movement.exemplar).')
        return None
    clips = [_clip_text(item, files, f'what.movement.examples[{index}]') for index, item in enumerate(examples)]
    looks = [_clip_text(item, files, f'what.movement.negatives[{index}]') for index, item in enumerate(negatives)]
    if None in clips or None in looks:
        return None
    text = movement.get('text')
    if rule is None:
        return dict(exemplar=None, rule=None, text=text, label=movement.get('label') or text or 'movement',
                    examples=clips, negatives=looks, combine=None)
    if isinstance(rule, str):
        path = files.get(rule, ('.json',), 'what.movement.rule')
        if path is None:
            return None
        try:
            rule = json.loads(Path(path).read_text())
        except (OSError, ValueError) as error:
            missing.append(f'what.movement.rule: {rule} is not readable JSON ({error}).')
            return None
        if isinstance(rule, dict) and rule.get('kind') == 'exemplar' and isinstance(rule.get('rule'), dict):
            rule = rule['rule']     # an enrolled rule file: its rule
    if not text:
        missing.append('what.movement.text: a rule needs the text it stands for (e.g. "raises both arms").')
        return None
    from ..providers.movement_rule import validate_rule
    try:
        rule = validate_rule(rule)
    except ValueError as error:
        missing.append(f'what.movement.rule: {error}.')
        return None
    return dict(exemplar=None, rule=rule, text=text, label=movement.get('label') or text, examples=clips,
                negatives=looks, combine=movement.get('combine'))


def _words(what, files, missing):
    words = what.get('words')
    if words is None:
        return None
    if isinstance(words, list) and words and all(isinstance(word, str) and word.strip() and ',' not in word
                                                 for word in words):
        return dict(terms=','.join(word.strip() for word in words), file=None)
    if isinstance(words, str):
        path = files.get(words, ('.json',), 'what.words')
        if path and _json_file(path, 'terms', 'what.words', missing) is not None:
            return dict(terms=str(path), file=path)
        return None
    missing.append('what.words: a list of words (re:PATTERN, word:W and word:W* allowed; no commas) or a terms file of '
                   'gamst recipe enroll --kind terms.')
    return None


def _rhythm(what, inputs, missing):
    rhythm = what.get('rhythm')
    if rhythm is None:
        return None
    from ..providers.rhythm import BANDS
    ids = [item['id'] for item in inputs['sensors']]
    if not isinstance(rhythm, dict) or rhythm.get('input') not in ids or not isinstance(rhythm.get('channel'), str) \
            or rhythm.get('band') not in BANDS:
        missing.append(f'what.rhythm: {{"input": a sensor id ({", ".join(ids) or "none given"}), "channel": one of its '
                       f'columns, "band": {" | ".join(BANDS)}}}.')
        return None
    sensor = next(item for item in inputs['sensors'] if item['id'] == rhythm['input'])
    if rhythm['channel'] not in [column.get('name') for column in sensor['columns'] if isinstance(column, dict)]:
        missing.append(f"what.rhythm.channel: {rhythm['channel']!r} is not a column of sensor {rhythm['input']}.")
        return None
    return rhythm


def _merge_reference(references, reference):
    same = next((item for item in references if item['name'] == reference['name']), None)
    if same is None:
        references.append(deepcopy(reference))
        return
    wanted = list(dict.fromkeys([*_listed(same.get('for')), *_listed(reference.get('for'))]))
    if wanted:
        same['for'] = wanted
    same['optional'] = bool(same.get('optional')) or bool(reference.get('optional'))


def _listed(value):
    return [] if value is None else [value] if isinstance(value, str) else list(value)


def _composed(request, inputs, timing, who, movement, words, rhythm):
    """(raw recipe, reasons, exclude)"""
    from . import read_recipe
    lanes = [name for name, wanted in (('movement', movement), ('presence', who and not movement and not words),
                                        ('words', words)) if wanted]
    if inputs['pictures'] and lanes != ['words']:
        lanes.append('semantic')        # in every composed recipe with pictures, OFF
    reasons, modules, references, records, limits, names = [], [], [], [], [], set()
    for lane in lanes:
        source, wanted, kinds, answers = CAPABILITIES[lane]
        raw = read_recipe(source)[1]
        for item in _items(raw, wanted):
            if item['name'] not in names:
                names.add(item['name'])
                modules.append(item)
        for reference in raw.get('references', ()):
            if reference['name'] in kinds:
                _merge_reference(references, reference)
        for record in raw.get('records', ()) if lane != 'semantic' else ():    # an interpretation is never counted
            if lane == 'presence' and (record if isinstance(record, str) else record['id']) != 'vision.person_binding':
                continue
            if record not in records:
                records.append(deepcopy(record))
        if raw.get('limits'):
            limits.append(f"{source}: {raw['limits']}")
        if lane != 'semantic':      # said once, with its state, below
            reasons.append(dict(capability=lane, lanes=[item['name'] for item in _items(raw, wanted)], recipe=source,
                                why=answers))
    if 'presence' in lanes:      # only who: the binding is the answer, it is not optional here
        for item in modules:
            if item['name'] == 'person_binding':
                item.pop('optional', None)
        for reference in references:
            if reference['name'] == 'person':
                reference.pop('optional', None)
    if rhythm:
        name = f"rhythm_{rhythm['input']}"
        modules.append(dict(name=name, use='rhythm', channel=rhythm['channel'], config=dict(band=rhythm['band']),
                            consumes=[rhythm['input']]))
        records.append(f"temporal.rhythm.{rhythm['band']}")
        reasons.append(dict(capability='rhythm', lanes=[name], why=f"the {rhythm['band']} rate of channel "
                            f"{rhythm['channel']} of sensor {rhythm['input']}, computed (gamst/providers/rhythm.py)"))
    recipe_inputs = []
    if inputs['pictures'] and any('camera' in item.get('consumes', ()) for item in modules):
        recipe_inputs.append(dict(id='camera', use='camera', sources=['camera', 'video', 'take']))
    if inputs['sound'] or (words and inputs['pictures'] and inputs['pictures'][0] == 'video'):
        if any('microphone' in item.get('consumes', ()) for item in modules):
            recipe_inputs.append(dict(id='microphone', use='microphone', sources=['microphone', 'audio', 'video']))
    for item in inputs['sensors']:
        recipe_inputs.append({'id': item['id'], 'use': 'sensor', 'sources': ['sensor'],
                              **{key: item[key] for key in ('modality', 'columns', 'time') if key in item}})
    exclude = []
    if words and not who:
        exclude.append('voice')     # words said by anyone: the voice parts of person-term-count stay out
    raw = dict(name=_name(request), summary=_summary(who, movement, words, rhythm),
               hardware=dict(min_fps=None, camera='as the lanes it takes: ' + ', '.join(
                   sorted({reason.get('recipe') for reason in reasons if reason.get('recipe')})) or 'none',
                   compute='as the lanes it takes'),
               inputs=recipe_inputs, references=references, modules=modules, view='count', records=records,
               limits=' | '.join(limits) or 'see the lanes', measured='not as composed; each lane is measured in the '
               'built-in recipe it comes from')
    return raw, reasons, exclude


def _name(request):
    return request.get('name') or f"composed-{_digest({key: value for key, value in request.items() if key not in ('start', 'wait')})}"


def _summary(who, movement, words, rhythm):
    parts = []
    if movement:
        parts.append(f"count the movement {movement['label']!r} for every tracked person" +
                     (f", who = {who['name']}" if who else ', who = unknown until a person is enrolled'))
    elif who and not words:
        parts.append(f"when {who['name']} is on screen")
    if words:
        parts.append(f"count the words {words['terms'] if not words['file'] else 'of ' + Path(words['file']).name}" +
                     (f" said by {who['name']}" if who else ' said by anyone'))
    if rhythm:
        parts.append(f"the {rhythm['band']} rate of {rhythm['input']}.{rhythm['channel']}")
    return 'Composed: ' + '; '.join(parts) + '.'


def _pose_reason(recipe, timing, pose):
    item = next((item for item in recipe['modules'] if item.get('use') in ('rtmw-pose', 'mediapipe-pose',
                                                                         'sapiens2-pose')), None)
    if item is None:
        return None
    if pose:
        return dict(capability='pose model', chose=pose, why='named in the request (one model counts every frame; '
                    'MediaPipe full, RTMW-l and Sapiens2 opt-in are the vision.pose candidates)')
    return dict(capability='pose model', chose=item.get('candidate'),
                why=POSE_RULE.format(timing=REALTIME_POSE if timing == 'realtime' else POST_POSE))


def _hosted_asr(recipe, include, asr):
    from ..crew.registry import ROLES
    from .parts import PARTS
    found = []
    for item in recipe['modules']:
        part = PARTS.get(item.get('use'))
        if part is None or not part.transcribes:
            continue
        candidate = asr if asr and not item.get('votes_for') else item.get('candidate', part.candidate)
        chosen = ROLES['audio.asr'].candidates.get(candidate)
        if chosen is not None and 'api' in (chosen.config or {}):
            found.append(candidate)
    return list(dict.fromkeys(found))


ENROLL_RECIPE = 'person-action-count'     # the movement and who lanes of a composed recipe are this recipe's


def _enroll(argv):
    """``gamst recipe enroll`` with ``argv`` (its JSON line goes to stderr: MCP's stdout is the protocol); an error
    sentence or None."""
    import contextlib
    from . import main as recipe
    try:
        with contextlib.redirect_stdout(sys.stderr):
            recipe(['enroll', ENROLL_RECIPE, *argv])
    except SystemExit as error:
        if error.code not in (None, 0):
            return str(error.code)
    return None


def enroll_examples(movement, path, *, profile=None, actor='user'):
    """Enroll the request's example clips (and its rule, when given) into ``path``; an error sentence or None."""
    argv = ['--mode', 'sequence', '--name', movement['label'], '--out', str(path)]
    for clip in movement['examples']:
        argv += ['--video', clip]
    for clip in movement['negatives']:
        argv += ['--negative', clip]
    if movement.get('text'):
        argv += ['--text', movement['text']]
    if movement['rule'] is not None:
        rule_path = Path(path).with_suffix('.rule.json')
        rule_path.write_text(json.dumps(movement['rule'], ensure_ascii=False))
        argv += ['--rule', str(rule_path)]
    if movement.get('combine'):
        argv += ['--combine', movement['combine']]
    if actor != 'user':
        argv.append('--no-confirm')
    if profile:
        argv += ['--profile', str(profile)]
    return _enroll(argv)


def enroll_photos(who, path, *, profile=None):
    """Enroll the request's photos of the person into ``path``; an error sentence or None."""
    argv = ['--kind', 'person', '--name', who['name'], '--consent', who['consent'], '--out', str(path)]
    for photo in who['photos']:
        argv += ['--image', str(photo)]
    if profile:
        argv += ['--profile', str(profile)]
    return _enroll(argv)


def compose(request, *, folders=None, out=None, out_dir=None, actor='user', profile=None):
    """The recipe for ``request`` (a dict, the module docstring), validated, with ``reasons`` and the ``run`` command.

    ``folders``: the owner's ``--clips`` folders every file must be inside (MCP), or None (the command line: any path).
    ``out``: the recipe file to write (a rule becomes ``<out>.movement.json`` beside it); None writes nothing;
    ``out_dir``: write it there as ``<name>.json`` instead.
    ``actor``: who wrote a rule (``user``: confirmed; ``jev``: recorded as unconfirmed, as with set_instruction).
    ``profile``: the setup profile example clips and photos are enrolled with (they run the pose and identity models).
    Returns ``{ok: false, missing: [sentence, ...]}`` when something must be provided first.
    """
    from . import check as sampling, not_given, read_recipe, roles as setup_roles, validate, with_pose, with_settings
    from ..person_reference import refusal
    missing = []
    if not isinstance(request, dict):
        return dict(ok=False, missing=['the request is a JSON object: inputs, who, what, mode, timing.'])
    unknown = sorted(set(request) - set(KEYS))
    if unknown:
        missing.append(f'request: unknown {", ".join(unknown)}; the keys are {", ".join(KEYS)}.')
    files = _Files(folders, missing)
    inputs = _inputs(request, files, missing)
    timing, mode = _timing(request, inputs, missing)
    what = request.get('what') or {}
    if not isinstance(what, dict) or set(what) - set(WHAT_KEYS):
        missing.append(f'what: an object of {", ".join(WHAT_KEYS)}.')
        what = {}
    base = request.get('recipe')
    movement, words, rhythm = _movement(what, files, missing), _words(what, files, missing), _rhythm(what, inputs, missing)
    who = _who(request, inputs, files, missing, words=bool(words))
    if not base and not (movement or words or rhythm or who) and not missing:
        missing.append('what: say what to look for: what.movement (a rule with its text, or an enrolled movement file), '
                       'what.words (words to count), what.rhythm (a sensor channel), or who alone (when that person is '
                       'on screen).')
    if (movement or (who and not words and not base)) and not inputs['pictures'] and not missing:
        missing.append('inputs: a movement and a person are seen on pictures: add inputs.camera (an index) or '
                       'inputs.video (a file).')
    if words and not (inputs['sound'] or (inputs['pictures'] and inputs['pictures'][0] == 'video')) and not missing:
        missing.append('inputs: words are heard: add inputs.microphone (true), inputs.audio (a WAV file) or '
                       'inputs.video (a file with its sound).')
    if inputs['sensors'] and not rhythm and not base and not missing:
        missing.append(f"inputs.sensors: nothing in this request reads {inputs['sensors'][0]['id']}: add what.rhythm "
                       '{input, channel, band}, or leave the sensor out.')
    for key in ('with', 'without', 'set'):
        if request.get(key) is not None and (not isinstance(request[key], list) or not all(
                isinstance(item, str) and item for item in request[key])):
            missing.append(f'{key}: a list of names' + (' MODULE.KEY=VALUE' if key == 'set' else '') + '.')
    if request.get('name') is not None and not (isinstance(request['name'], str) and re.fullmatch(
            r'[a-z0-9][a-z0-9-]{0,63}', request['name'])):
        missing.append('name: lowercase letters, digits and dashes, at most 64.')
    if missing:
        return dict(ok=False, missing=missing)
    include, exclude = list(request.get('with') or ()), list(request.get('without') or ())
    reasons = []
    if base:
        if folders is not None and (not isinstance(base, str) or not re.fullmatch(r'[a-z0-9-]{1,64}', base)):
            return dict(ok=False, missing=['recipe: the name of a built-in recipe (gamst recipe list).'])
        try:
            raw = deepcopy(read_recipe(base)[1])
        except SystemExit as error:
            return dict(ok=False, missing=[f'recipe: {error}.'])
        reasons.append(dict(capability='recipe', chose=raw['name'], why='named in the request: used as it is' + (
            ', customised by with / without / set / pose / asr' if include or exclude or request.get('set')
            or request.get('pose') or request.get('asr') else '')))
        if words and not who and any(item.get('parts') and 'voice' in _listed(item.get('parts'))
                                     for item in raw['modules'] if item.get('optional') == 'on'):
            exclude.append('voice')
    else:
        raw, composed_reasons, composed_exclude = _composed(request, inputs, timing, who, movement, words, rhythm)
        reasons += composed_reasons
        exclude += [name for name in composed_exclude if name not in include]
    # what is given for each kind of reference
    given = dict(exemplar=movement, person=who, voice=who and who.get('voice'), terms=words)
    if who and not any(item.get('use') in ('person-binding', 'target-person', 'speaker-verify') for item in raw['modules']):
        return dict(ok=False, missing=[refusal(who['name'], [f"{raw['name']} has nothing that tells people apart "
                                                             '(person binding or a voice)'])])
    if who and movement and not any(item.get('use') == 'person-binding' for item in raw['modules']):
        return dict(ok=False, missing=[refusal(who['name'], [f"{raw['name']} has no person-binding module"])])
    try:
        # what the run leaves out by itself: optional parts whose optional reference is not given
        unread = [name for name in not_given(raw, argparse.Namespace(**{kind: bool(value) for kind, value in
                                                                       given.items()}), include) if name not in exclude]
        label = f'composed recipe {raw["name"]}' if not base else raw['name']
        chosen = validate(deepcopy(raw), label, include, [*exclude, *unread])
        if request.get('pose'):
            chosen = with_pose(chosen, request['pose'])
            raw = with_pose(raw, request['pose'])
        if request.get('set'):
            chosen = with_settings(chosen, request['set'])
            raw = with_settings(raw, request['set'])
    except (ValueError, SystemExit) as error:
        return dict(ok=False, missing=[f'{error}'.rstrip('.') + '.'])
    lacking = [reference for reference in chosen.get('references', ()) if not reference.get('optional')
               and reference['kind'] in given and not given[reference['kind']]]
    hints = dict(exemplar='what.movement (a rule with its text, or an enrolled movement file)',
                 terms='what.words (the words to count)', voice='who.voice (an enrolled voice file)',
                 person='who.person (an enrolled person file)')
    if lacking:
        return dict(ok=False, missing=[f"{chosen['name']} needs {reference['label']}: give {hints[reference['kind']]}."
                                       for reference in lacking])
    pose = _pose_reason(chosen, timing, request.get('pose'))
    if pose:
        reasons.insert(0, pose)
    if who and not who.get('photos'):
        reasons.append(dict(capability='who', chose=who['name'], why=f"few-shot person reference checked "
                            f"(gamst/person_reference.py): identified by {', '.join(who['check']['basis'])}; events of "
                            'a track take the person bound to it, also retroactively within the track'))
    elif movement:
        reasons.append(dict(capability='who', chose=None, why='nobody named: every track is counted with who = unknown; '
                            'the binding runs with nobody enrolled, so enroll_person can add a person while it runs'))
    if words:
        hosted = _hosted_asr(chosen, include, request.get('asr'))
        reasons.append(dict(capability='words', chose=hosted or request.get('asr'), why=(
            'the state-of-the-art live recogniser by published results (decision 31) is hosted and paid: it needs '
            'consent.remote_asr' if hosted else 'a local recogniser') + ('; said by ' + who['name'] + ', told apart by '
                                                                        'the enrolled voice' if who else
                                                                        '; said by anyone (the voice parts stay out)')))
    if any(item['name'] == 'semantic' for item in raw['modules']):
        reasons.append(dict(capability='semantic', chose='off', why='the VLM interpretation stays OFF (decision 47); '
                            'with "with": ["semantic"] it runs as a causal state chain (explain_state)'))
    reasons.append(dict(capability='timing', chose=timing, why=(
        'realtime: counts as frames arrive; what the live lane cannot reach in time is counted late by L2, as '
        'corrections' if timing == 'realtime' else 'post: a finished file, every frame in the largest batches, nothing '
        'late')))
    if mode == 'search':
        reasons.append(dict(capability='mode', chose='search', why='never miss the person: what would be uncertain or '
                            'revoked for its voice or identity is counted as a candidate with its reason'))
    # files written for the run, and the command that runs it
    saved, before = {}, []
    target = Path(out) if out else Path(out_dir) / f"{raw['name']}.json" if out_dir else None
    recipe_path = target if target else Path(f"{raw['name']}.json")
    if (movement and movement['examples'] or who and who.get('photos')) and not target:
        return dict(ok=False, missing=['example clips and photos are enrolled into files beside the recipe: give --out '
                                       'FILE (over MCP the recipe is saved in the first --clips folder).'])
    if who and who.get('photos'):
        from ..person_reference import check, refusal as refused, session_of
        person_path = recipe_path.with_suffix('.person.json')
        person_path.parent.mkdir(parents=True, exist_ok=True)
        problem = enroll_photos(who, person_path, profile=profile)
        if problem:
            return dict(ok=False, missing=[refused(who['name'], [problem])])
        value = json.loads(person_path.read_text())
        if who.get('text'):
            value['description'] = who['text']
            from .enrollment import _save
            _save(str(person_path), value)
        pictures = inputs['pictures']
        verdict = check(value, session=session_of(pictures[1]) if pictures and pictures[0] == 'video' else None)
        if not verdict['ok']:
            return dict(ok=False, missing=[refused(who['name'], verdict['problems'])])
        who = dict(who, person=person_path, check=verdict)
        saved['person'] = str(person_path)
        reasons.append(dict(capability='who', chose=who['name'], why=f"few-shot person reference enrolled from "
                            f"{len(who['photos'])} photo(s) and checked (gamst/person_reference.py): identified by "
                            f"{', '.join(verdict['basis'])}; events of a track take the person bound to it, also "
                            'retroactively within the track'))
    if movement and movement['examples']:
        movement_path = recipe_path.with_suffix('.movement.json')
        movement_path.parent.mkdir(parents=True, exist_ok=True)
        problem = enroll_examples(movement, movement_path, profile=profile, actor=actor)
        if problem:
            return dict(ok=False, missing=[f'what.movement.examples: {problem}'])
        enrolled = json.loads(movement_path.read_text())
        saved['movement'] = str(movement_path)
        movement = dict(movement, exemplar=movement_path)
        reasons.append(dict(capability='movement', chose=f"{enrolled.get('mode')} {enrolled.get('id')}", why=(
            f"{len(movement['examples'])} example clip(s)" + (f", {len(movement['negatives'])} negative(s)" if
                                                              movement['negatives'] else '') +
            (f" with the rule for {movement['text']!r} as the skeleton ({enrolled.get('rule_source')}): counted when "
             f"{'the rule and an example both' if (enrolled.get('combine') or ('any' if mode == 'search' else 'all')) == 'all' else 'the rule or an example'} "
             'say so (decision 58)' if movement['rule'] is not None else ', compared by dynamic time warping') +
            f"; thresholds {enrolled.get('threshold_source')} (enter {enrolled.get('enter')}, exit {enrolled.get('exit')})")))
    elif movement and movement['rule'] is not None:
        from .movement import _rule
        movement_path = recipe_path.with_suffix('.movement.json')
        args = argparse.Namespace(name=movement['label'], text=movement['text'], out=None)
        saved_movement = _rule(raw, args, movement['rule'], confirmed=actor == 'user')
        if target:
            from .enrollment import _save
            movement_path.parent.mkdir(parents=True, exist_ok=True)
            _save(str(movement_path), saved_movement)
            saved['movement'] = str(movement_path)
        movement = dict(movement, exemplar=movement_path)
        reasons.append(dict(capability='movement', chose=f"rule {saved_movement['id']}", why=(
            f"the rule the caller wrote for {movement['text']!r}, validated (gamst/providers/movement_rule.py)" +
            ('' if actor == 'user' else '; written by jev, so its events say confirmed: false'))))
    elif movement:
        reasons.append(dict(capability='movement', chose=Path(movement['exemplar']).name, why='an enrolled movement file'))
    run = ['gamst', 'recipe', 'run', str(recipe_path)]
    pictures, sound = inputs['pictures'], inputs['sound']
    if pictures:
        run += ['--camera', str(pictures[1])] if pictures[0] == 'camera' else ['--video', str(pictures[1])]
    if sound:
        run += ['--microphone'] if sound[0] == 'microphone' else ['--audio', str(sound[1])]
    for item in inputs['sensors']:
        run += (['--sensor', str(item['file'])] if len(inputs['sensors']) == 1 else
                ['--input', f"{item['id']}=sensor:{item['file']}"])
    if movement:
        run += ['--exemplar', str(movement['exemplar'])]
    kinds = {reference['kind'] for reference in chosen.get('references', ())}
    if who:
        if 'person' in kinds:
            run += ['--person', str(who['person'])]
        if who.get('voice'):
            run += ['--voice', str(who['voice'])]
        if movement:
            run += ['--who', who['name']]
    if words:
        run += ['--terms', words['terms']]
    if request.get('asr'):
        run += ['--asr', request['asr']]
    run += ['--timing', timing] + (['--mode', 'search'] if mode == 'search' else [])
    for name in include:
        run += ['--with', name]
    for name in exclude:
        run += ['--without', name]
    if request.get('pose'):
        run += ['--pose', request['pose']]
    consent = request.get('consent') or {}
    if words and _hosted_asr(chosen, include, request.get('asr')):
        if consent.get('remote_asr') is True:
            run.append('--allow-remote-asr')
        else:
            before.append(f"words: {', '.join(_hosted_asr(chosen, include, request.get('asr')))} is hosted and paid: "
                          'set consent.remote_asr true (the owner\'s consent to sending the sound), or name a local '
                          'recogniser in asr.')
    if any(name in include for name in ('verification', 'hosted', 'vote', 'tracks', 'tse')):
        before.append('with: the verification and hosted diarization parts send sound to paid services; run with '
                      '--allow-remote-diarization once the owner consents.')
    if (pictures and pictures[0] == 'camera' or sound and sound[0] == 'microphone') and consent.get('capture') is not True:
        before.append('consent.capture: a camera or microphone is opened only with the owner\'s explicit consent: set '
                      'consent.capture true.')
    run.append('--headless')
    recipe = dict(raw, composed=dict(request={key: value for key, value in request.items() if key not in (
        'start', 'wait', 'consent')}, reasons=reasons, run=run, by='gamst.recipes.compose', version=1))
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + '\n')
        saved['recipe'] = str(target)
    try:
        roles = setup_roles(chosen)
    except Exception as error:      # what setup would install is a hint; composing never needs it
        roles = {'unknown': str(error)}
    verdict = sampling(chosen)
    return dict(ok=True, name=raw['name'], recipe=recipe, reasons=reasons, run=run, saved=saved or None,
                before_run=before, inputs=[item['id'] for item in chosen['inputs']],
                modules=[item['name'] for item in chosen['modules']],
                left_out=[item['name'] for item in raw['modules'] if item['name'] not in
                          {other['name'] for other in chosen['modules']}],
                records=list(chosen.get('records', ())), setup_roles=roles, required_fps=verdict.get('required_fps'),
                setup=f'gamst recipe setup {recipe_path} --dry-run')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='gamst recipe compose', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('request', help='the request as a JSON file, or - for stdin')
    parser.add_argument('--out', help='recipe file to write (a rule is written beside it as NAME.movement.json)')
    parser.add_argument('--start', action='store_true', help='run it at once, as gamst recipe run (needs --out)')
    parser.add_argument('--profile', help='setup profile that example clips and photos are enrolled with')
    args = parser.parse_args(argv)
    try:
        request = json.loads(sys.stdin.read() if args.request == '-' else Path(args.request).read_text())
    except (OSError, ValueError) as error:
        raise SystemExit(f'{args.request}: {error}') from None
    start = args.start or isinstance(request, dict) and request.get('start') is True
    if start and not args.out:
        raise SystemExit('--start runs a saved recipe: give --out FILE')
    result = compose(request, out=args.out, profile=args.profile)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)
    if not result['ok']:
        raise SystemExit(1)
    if start:
        if result['before_run']:
            raise SystemExit('\n'.join(result['before_run']))
        from . import main as recipe
        return recipe(result['run'][2:])
    return result
