"""Executable recipes: one JSON file names the inputs, perception modules and view a task needs.

A recipe composes parts GAMST already has (``gamst/recipes/parts.py``) into a
:class:`gamst.pipeline.Pipeline`; it adds no runtime of its own.

  gamst recipe list
  gamst recipe show person-action-count
  gamst recipe setup person-action-count [--dry-run]     install only what the recipe uses (gamst setup's apply)
  gamst recipe check rotation --fps 240 --symmetry 3     test a camera against the sampling limit
  gamst recipe enroll person-action-count --video a.mp4 --video b.mp4@2-3.5 --negative other.mp4 --out move.json
  gamst recipe run person-action-count --camera 0 --exemplar move.json
  gamst recipe find person-action-count --video long.mp4 --exemplar move.json --out found.csv --format csv
  gamst recipe propose person-action-count --video long.mp4 --text "raises both arms"     a VLM finds candidates
  gamst recipe proposals list | confirm ID... | reject ID...                              the owner decides
  gamst recipe enroll person-action-count --text "raises both arms" --proposals FILE --out move.json
  gamst recipe enroll person-action-count --text "raises both arms" --rule rule.json --out move.json
  gamst recipe export --store run.db --format csv --out events.csv
  gamst recipe run rotation --take DIR --roi .3,.2,.4,.6 --symmetry 3
  gamst recipe enroll person-term-count --kind voice --microphone --seconds 20 --consent '...' --out me.json
  gamst recipe run person-term-count --microphone --voice me.json --terms 대박 [--truth truth.json]
  gamst recipe enroll person-term-count --kind terms --video a.mp4@3-5 --audio b.wav        propose the word said in clips
  gamst recipe run coin-flip --iphone 9123 | --take DIR
  gamst recipe run scene --camera 0 --microphone --without voice ...    several inputs on one capture clock
  gamst recipe run scene --video FILE ...                               a file's pictures and its own sound
  gamst recipe run person-action-count ... --with-role vision.depth@500 any registry role, here every 500 ms
  gamst recipe roles                                                    which roles that can be, and why not

Recipe file::

  {"name", "summary", "skill", "hardware": {"min_fps", "camera", "compute"},
   "inputs": [{"id": "camera", "use": "camera", "sources": ["camera", "video", "take"]},
              {"id": "microphone", "use": "microphone", "sources": ["microphone", "audio", "video"], "offset_ms": 0},
              {"id": "imu", "use": "sensor", "sources": ["sensor"], "modality": "imu",
               "columns": [{"name": "ax", "unit": "m/s2"}], "time": {"column": "t", "unit": "s"}}],
   "references": [{"name": "exemplar", "kind": "exemplar", "for": "exemplar"}],   user-defined inputs
   "modules": [{"name": "pose", "use": "mediapipe-pose", "consumes": ["camera"], ...options},
               {"name": "depth", "use": "role", "role": "vision.depth", "consumes": ["camera"]},
               {"name": "find", "use": "yoloe", "in": "count"}],     a stage that runs inside "count"
   "view": "count" | "rotation-view" | "halo-coin",
   "records": ["temporal.exemplar_events", {"id": "speech.term_events", "for": "terms"}], "measured": "..."}

A module may be "optional": "on" | "off" and belong to named "parts"; ``--with`` / ``--without`` choose a module
or a part. What is left out takes its references, records and (in a recipe with several inputs) the inputs
nothing reads any more with it. An optional module pauses while others are behind, unless "pausable": false.

``run`` builds the Pipeline, feeds it the chosen source and shows the view; agents read the same
records over MCP (``get_observations``, ``count_events``) and switch modules with
``configure_pipeline_module``. ``coin-flip`` is the GAMST 240 app: the ``gamst240`` input (the
iPhone, or a take replayed), a take recorder, the after-take flip counter (YOLOE and EfficientTAM
are its stages, in its own process) and the ``halo-coin`` HUD.

Sampling limit. A part's ``sampling`` rule says which frame rate counts its target without
aliasing, e.g. an object whose look repeats ``symmetry`` times per turn is countable only below
``fps / (2 * symmetry)`` rev/s, and a movement repeated r times per second needs more than 2r
frames per second. ``check`` and ``run`` refuse a known frame rate below the limit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import platform
import re
import sys
import time

from ..demos import rotation as demo
from .movement import PROPOSALS
from .parts import PARTS

RECIPES = Path(__file__).resolve().parent
SOURCES_BY_INPUT = {'camera': ('camera', 'video', 'take'), 'gamst240': ('iphone', 'take'),
                    'microphone': ('microphone', 'audio', 'video'), 'sensor': ('sensor',), 'plugin': ('plugin',)}
MODALITY_BY_INPUT = {'camera': 'video', 'video': 'video', 'take': 'video', 'microphone': 'audio', 'audio': 'audio'}
SOURCES = ('iphone', 'take', 'camera', 'video', 'audio', 'microphone', 'sensor')     # command-line options
LIVE = ('iphone', 'camera', 'microphone')
MAX_LIVE_OFFSET_MS = 4000      # the runtime refuses capture times further ahead
REFERENCE_KINDS = ('voice', 'person', 'face', 'body', 'exemplar', 'terms', 'object')
SEMANTIC = 'vision.streaming_semantic'     # an interpretation: never counted


def names():
    return sorted(path.stem for path in RECIPES.glob('*.json'))


def _listed(value):
    return [] if value is None else [value] if isinstance(value, str) else list(value)


def choose(recipe, include=(), exclude=()):
    """Resolve ``"optional": "on" | "off"`` modules (``--with`` / ``--without``, by module or part name).

    A module may name what it reads when an optional one is absent (``consumes_without``), what it reads
    only while an optional one is present (``consumes_optional``), what it reads instead while an optional one is
    present (``consumes_with``; a key ``"a+b"`` while all of them are present, applied after single keys in the order
    given), and a reference or record what it is for (``for``); both follow the choice. An
    optional module that does the work of others (``replaces``) leaves them out while it runs, unless they are
    asked for by name too.
    """
    optional = {item['name']: item for item in recipe.get('modules', ()) if item.get('optional')}

    def expand(wanted):
        found = set()
        for name in wanted:
            members = [key for key, item in optional.items() if key == name or name in _listed(item.get('parts'))]
            if not members:
                parts = sorted({part for item in optional.values() for part in _listed(item.get('parts'))})
                raise SystemExit(f"{recipe['name']} has no optional module {name}; optional: "
                                 f"{', '.join([*sorted(optional), *parts]) or 'none'}")
            found.update(members)
        return found

    included, excluded = expand(include), expand(exclude)
    absent = {name for name, item in optional.items()
              if name in excluded or (item['optional'] == 'off' and name not in included)}
    for name, item in optional.items():
        if name not in absent:
            absent |= {other for other in _listed(item.get('replaces')) if other in optional and other not in included}
    modules = []
    for item in recipe.get('modules', ()):
        if item['name'] in absent:
            continue
        for name in [name for name in item.get('consumes_without', {}) if name in absent]:
            item = dict(item, consumes=item['consumes_without'][name])
        for name in [name for name in item.get('consumes_with', {})     # "a+b": while all of them are present
                     if all(part in optional and part not in absent for part in name.split('+'))]:
            item = dict(item, consumes=item['consumes_with'][name])
        if item.get('consumes_optional'):    # read only while the optional module that gives it is in the run
            present = [source for source in item['consumes_optional'] if not any(
                other['name'] in absent and (source.split(':', 1)[0] == other['name'] or source in other.get('emits', ()))
                for other in recipe['modules'])]
            item = dict(item, consumes=[*item.get('consumes', ()), *present])
        modules.append(item)
    for item in modules:
        for source in item.get('consumes', ()):
            owner = next((other['name'] for other in recipe['modules'] if other['name'] in absent and (
                source.split(':', 1)[0] == other['name'] or source in other.get('emits', ()))), None)
            if owner is not None:
                wanted = ', '.join(_listed(item.get('parts'))) or 'that has no part'
                raise SystemExit(f"{item['name']} reads {owner}, which is left out: leave {item['name']} out too "
                                 f"(--without {item['name']}, or its part: {wanted})" if item.get('optional') else
                                 f"{item['name']} needs {owner}; it cannot be left out")

    def kept(item):
        wanted = _listed(item.get('for')) if isinstance(item, dict) else []
        return not wanted or any(name not in absent for name in wanted)

    return dict(recipe, modules=modules,
                references=[item for item in recipe.get('references', ()) if kept(item)],
                records=[item['id'] if isinstance(item, dict) else item for item in recipe.get('records', ()) if kept(item)])


POSE_PARTS = {'mediapipe': 'mediapipe-pose', 'rtmw': 'rtmw-pose', 'sapiens2': 'sapiens2-pose', 'yolo26': 'yolo26-pose'}
# yolo26-pose reads like RTMW: its 17 COCO points are the first 17 of COCO-WholeBody (the same indices)
POSE_SOURCES = {'mediapipe-pose': None, 'rtmw-pose': 'rtmw', 'sapiens2-pose': 'sapiens2', 'yolo26-pose': 'rtmw'}


def with_pose(recipe, candidate, *, module=None):
    """The recipe with ``candidate`` (a vision.pose candidate) as its pose module (``module``: by name, default the
    first pose item); the modules that read its joints read them that way. One pose model counts: the parts it
    selects are :data:`POSE_PARTS` (a new pose model adds its part there and in ``POSE_SOURCES``)."""
    from ..crew.registry import ROLES
    if candidate not in ROLES['vision.pose'].candidates:
        raise SystemExit(f"--pose: no vision.pose candidate {candidate!r}; candidates: "
                         f"{', '.join(sorted(ROLES['vision.pose'].candidates))}")
    part = POSE_PARTS[candidate.split('.', 1)[0]]
    items = recipe['modules']
    target = next((item for item in items if item['use'] in POSE_SOURCES and (
        item['name'] == module if module else True)), None)
    if target is None:
        raise SystemExit(f"--pose: {recipe['name']} has no pose module")
    joints = set(target.get('emits', ())) | {target['name']}
    source = POSE_SOURCES[part]

    def changed(item):
        if item is target:
            kept = {key: value for key, value in item.items() if key not in ('delegate', 'candidate', 'use')}
            return dict(kept, use=part, candidate=candidate)
        reads = {name.split(':', 1)[0] for name in item.get('consumes', ())} | {
            name.split(':', 1)[0] for names in (item.get('consumes_without') or {}).values() for name in names}
        if item['use'] in ('pose-channels', 'exemplar') and reads & joints:
            return {key: value for key, value in dict(item, source=source).items() if value is not None}
        return item
    # what reads the joints of what reads them (target_person passes the pose joints on as its own)
    for item in items:
        if item['use'] == 'target-person' and set(item.get('consumes', ())) & joints:
            joints |= set(item.get('emits', ())) | {item['name']}
    return dict(recipe, modules=[changed(item) for item in items])


WARM_UP_MS = 200      # a pose model's own state after a cut (tracking boxes, VIDEO mode): about 12 frames at 60 fps


def pre_roll_ms(movement):
    """How much video before a cut an L2 worker (or the live lane after its jump) needs to see a movement that ends
    after the cut: an ordered rule's ``within_ms`` and ``hold_ms`` summed plus three rate windows, or the longest
    example; plus the pose model's warm-up."""
    from ..providers.movement_rule import RATE_MS
    rule = movement.get('rule') if isinstance(movement, dict) else None
    if isinstance(rule, dict) and isinstance(rule.get('steps'), list):
        steps = rule['steps']
        return int(sum(step.get('within_ms', 0) + step.get('hold_ms', 0) for step in steps) + 3 * RATE_MS + WARM_UP_MS)
    spans = [example.get('span_ms') for example in (movement.get('positives') or ()) if isinstance(example, dict)] \
        if isinstance(movement, dict) else []
    spans = [span for span in spans if type(span) in (int, float)]
    return int(max(spans) + 3 * RATE_MS + WARM_UP_MS) if spans else 1000


def with_pre_roll(recipe, exemplar):
    """``catch_up`` of a module item with ``"max_lookback_ms": "pre_roll"`` / ``"tail_ms": "pre_roll"``: the
    movement's :func:`pre_roll_ms`."""
    wanted = [item for item in recipe['modules'] if 'pre_roll' in (item.get('catch_up') or {}).values()]
    if not wanted:
        return recipe
    try:
        movement = json.loads(Path(exemplar).read_text())
    except (OSError, ValueError):
        movement = {}
    pre = pre_roll_ms(movement)
    return dict(recipe, modules=[dict(item, catch_up={key: pre if value == 'pre_roll' else value
                                                      for key, value in item['catch_up'].items()})
                                 if item in wanted else item for item in recipe['modules']])


def with_settings(recipe, settings):
    """``MODULE.KEY=VALUE`` settings of module items (VALUE as JSON, else as text)."""
    modules = [dict(item) for item in recipe['modules']]
    for setting in settings:
        name, _, value = setting.partition('=')
        module, _, key = name.partition('.')
        item = next((item for item in modules if item['name'] == module), None)
        if not key or not _ or item is None:
            raise SystemExit(f"--set {setting!r}: MODULE.KEY=VALUE of a module of {recipe['name']} "
                             f"({', '.join(item['name'] for item in modules)})")
        try:
            item[key] = json.loads(value)
        except ValueError:
            item[key] = value
    return dict(recipe, modules=modules)


def input_modality(item):
    """The modality of an input's frames, or None when only its own recipe's modules read them."""
    if item['use'] == 'plugin':
        return item.get('modality')
    return item.get('modality', 'imu') if item['use'] == 'sensor' else MODALITY_BY_INPUT.get(item['use'])


def selected_plugin(item, plugins):
    """The entry point a plugin item names, if the owner selected it; a recipe file alone never runs installed code."""
    name = item.get('plugin')
    if not isinstance(name, str) or not name:
        raise SystemExit(f"{item.get('name') or item.get('id')!r}: a plugin item names its entry point: \"plugin\": NAME")
    if name not in (plugins or ()):
        raise SystemExit(f'plugin {name} is code installed on this computer, run as it is with your rights (trusted code); '
                         f'if you trust it: --plugin {name}')
    return name


def add_role(recipe, wanted):
    """``ROLE[=CANDIDATE][@MS]`` as one more optional module of a recipe whose inputs fit its modality."""
    from ..crew.registry import ROLES
    found = re.fullmatch(r'([a-z_.0-9]+)(?:=([^@]+))?(?:@(\d+))?', wanted)
    if found is None or found[1] not in ROLES:
        raise SystemExit(f'--with-role {wanted}: ROLE[=CANDIDATE][@MS], a role of gamst recipe roles')
    role = found[1]
    _refuse_role(ROLES[role], found[2])
    modality = ROLES[role].modality
    source = next((item['id'] for item in recipe['inputs'] if input_modality(item) == modality), None)
    if source is None:
        raise SystemExit(f"{role} reads {modality}; {recipe['name']} has no {modality} input")
    same = next((item['name'] for item in recipe['modules'] if item.get('use') in PARTS and _role(item)[0] == role), None)
    if same is not None:
        raise SystemExit(f"{role} already runs in {recipe['name']} as {same}")
    item = dict(name=role, use='role', role=role, consumes=[source], optional='on')
    if found[2]:
        item['candidate'] = found[2]
    if found[3]:
        item['every_ms'] = int(found[3])
    return dict(recipe, modules=[*recipe['modules'], item])


def _refuse_role(role, candidate=None):
    """Why no candidate (or not this one) can be a generic module, as an error."""
    from ..crew.modules import refusal
    reasons = {name: refusal(role.id, name) if name in role.candidates else f"unknown; choose from {', '.join(role.candidates)}"
               for name in ([candidate] if candidate else role.candidates)}
    if all(reasons.values()):
        raise SystemExit(f'{role.id} cannot be a module: ' + '; '.join(
            f"{', '.join(name for name in reasons if reasons[name] == reason)}: {reason}"
            for reason in dict.fromkeys(reasons.values())))


def _check_role(path, recipe, item):
    from ..crew.registry import ROLES
    role = ROLES.get(item.get('role'))
    if role is None:
        raise ValueError(f"{path}: module {item['name']!r} names no role of the registry: {item.get('role')!r}")
    _refuse_role(role, item.get('candidate'))
    for source in item.get('consumes', ()):
        entry = next((entry for entry in recipe['inputs'] if entry['id'] == source), None)
        if entry is not None and input_modality(entry) != role.modality:
            raise SystemExit(f"{role.id} reads {role.modality}; input {source!r} is {input_modality(entry) or entry['use']}")


def read_recipe(name):
    """(path, content) of a built-in recipe by name, or of a recipe file path; nothing chosen yet."""
    path = Path(name)
    if not (path.suffix == '.json' and path.is_file()):
        path = RECIPES / f'{name}.json'
        if not path.is_file():
            raise SystemExit(f'unknown recipe {name!r}; built-in: {", ".join(names())}')
    return path, json.loads(path.read_text())


PART_SETTINGS = ('segmentation', 'hop_seconds', 'margin_seconds', 'max_seconds', 'beam_size', 'window_seconds',
                 'pause_seconds', 'config', 'python')


def asr_part(candidate):
    """The transcribing part that builds an ``audio.asr`` candidate (``Part.serves``)."""
    from ..crew.registry import ROLES
    chosen = ROLES['audio.asr'].candidates.get(candidate)
    if chosen is None:
        raise SystemExit(f"--asr: no audio.asr candidate {candidate!r}; candidates: "
                         f"{', '.join(sorted(ROLES['audio.asr'].candidates))}")
    key = chosen.in_process or chosen.provider_profile
    found = [name for name, part in PARTS.items() if part.transcribes and key in part.serves]
    if not found:
        raise SystemExit(f'--asr: no recipe part builds {candidate} yet')
    return found[0]


def with_asr(recipe, candidate):
    """The recipe with its transcribing module built from ``candidate``; what it reads and emits stays."""
    part = asr_part(candidate)
    modules, changed = [], False
    for item in recipe['modules']:
        if PARTS[item['use']].transcribes and not item.get('votes_for'):     # a voter keeps its own recogniser
            if item['use'] != part:     # the old part's own settings (phrase buffer, windows) do not carry over
                item = {key: value for key, value in item.items() if key not in PART_SETTINGS}
            item, changed = dict(item, use=part, candidate=candidate), True
        modules.append(item)
    if not changed:
        raise SystemExit(f"--asr: {recipe['name']} has no module that transcribes")
    return dict(recipe, modules=modules)


def _override(modules, changes, where):
    """Module items with ``changes`` ({module: {key: value | null}}) merged in; ``{"remove": true}`` drops one."""
    names = {item['name'] for item in modules}
    unknown = set(changes) - names
    if unknown:
        raise ValueError(f"{where}: no module {', '.join(sorted(unknown))} to change")
    result = []
    for item in modules:
        change = changes.get(item['name'])
        if change is None:
            result.append(item)
            continue
        if change.get('remove'):
            continue
        merged = dict(item)
        for key, value in change.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        result.append(merged)
    return result


def stack_names(raw):
    return [name for name in (raw.get('stacks') or {}) if name != 'default']


def variant(raw, *, stack=None, sport=None, timing=None):
    """The recipe with its sport preset, then its stack, then the stack's timing variant applied (docs/sports.md).

    ``"sports"``: {sport: {"modules": {...}}} plus ``"default"``; the sport's physics preset
    (:mod:`gamst.sports.presets`) also goes to every ``ball-track`` item as ``sport``. ``"stacks"``: {stack:
    {"summary", "modules", "realtime": {"modules"}, "post": {"modules"}}} plus ``"default"``. A recipe without them is
    returned unchanged (asking for a stack or sport it does not have is an error)."""
    sports, stacks = raw.get('sports') or {}, raw.get('stacks') or {}
    if sport is not None and sport not in sports:
        raise SystemExit(f"{raw['name']} has no sport {sport!r}; sports: {', '.join(k for k in sports if k != 'default') or 'none'}")
    if stack is not None and stack not in stacks:
        raise SystemExit(f"{raw['name']} has no stack {stack!r}; stacks: {', '.join(stack_names(raw)) or 'none'}")
    sport = sport or sports.get('default')
    stack = stack or stacks.get('default')
    if not sport and not stack:
        return raw
    modules = [dict(item) for item in raw['modules']]
    chosen = {}
    if sport:
        from ..sports.presets import SPORTS
        for item in modules:
            if item['use'] == 'ball-track' and sport in SPORTS:
                item.setdefault('sport', sport)
        modules = _override(modules, (sports[sport] or {}).get('modules') or {}, f"{raw['name']} sport {sport}")
        chosen['sport'] = sport
    if stack:
        entry = stacks[stack]
        modules = _override(modules, entry.get('modules') or {}, f"{raw['name']} stack {stack}")
        if timing in TIMINGS and entry.get(timing):
            modules = _override(modules, entry[timing].get('modules') or {}, f"{raw['name']} stack {stack} {timing}")
        chosen.update(stack=stack, stack_summary=entry.get('summary'))
    if timing:
        chosen['timing'] = timing
    return dict(raw, modules=modules, chosen=chosen)


def load(name, include=(), exclude=(), with_roles=(), keep=None, *, stack=None, sport=None, timing=None):
    """A built-in recipe by name, or a recipe file path; validated, optional modules resolved.

    ``keep(input item)``: an input that stays although nothing reads it (a camera that is only shown). ``stack``,
    ``sport``, ``timing``: the recipe's variant (:func:`variant`).
    """
    path, raw = read_recipe(name)
    raw = variant(raw, stack=stack, sport=sport, timing=timing)
    recipe = choose(raw, include, exclude)
    for wanted in with_roles:
        recipe = add_role(recipe, wanted)
    inputs = recipe.get('inputs') or []
    if len(inputs) > 1:     # nothing is opened that nobody reads
        read = inputs_of(recipe, recipe['modules'])
        inputs = [item for item in inputs if item in read or keep is not None and keep(item)]
        recipe = dict(recipe, inputs=inputs)
        if not inputs:
            raise SystemExit(f"{recipe['name']}: nothing that is left reads an input")
    if not inputs or len({item['id'] for item in inputs}) != len(inputs):
        raise ValueError(f'{path}: a recipe has one or more inputs with distinct ids')
    for item in inputs:
        part = PARTS.get(item.get('use'))
        if part is None or part.kind not in ('input', 'plugin'):
            raise ValueError(f'{path}: unknown input part {item.get("use")!r}')
        if item['use'] == 'plugin':
            from ..capabilities import MODALITIES
            if item.get('modality') not in MODALITIES:
                raise ValueError(f"{path}: plugin input {item['id']!r} declares its modality")
            item.setdefault('sources', ['plugin'])
        supported = SOURCES_BY_INPUT.get(item['use'], (item['use'],))
        if not item.get('sources') or set(item['sources']) - set(supported):
            raise ValueError(f"{path}: input {item['use']!r} reads from {supported}")
        if item['use'] == 'sensor' and not {'columns', 'time'} <= set(item):
            raise ValueError(f"{path}: a sensor input declares its columns (name, unit) and its time column")
    known = {item['id'] for item in inputs}
    for reference in recipe.get('references', ()):
        if reference.get('kind') not in REFERENCE_KINDS:
            raise ValueError(f'{path}: reference kind must be one of {REFERENCE_KINDS}')
    for item in recipe.get('modules', ()):
        part = PARTS.get(item.get('use'))
        if part is None or part.kind not in ('module', 'plugin'):
            raise ValueError(f'{path}: unknown part {item.get("use")!r}')
        if item.get('in') and item['in'] not in {other['name'] for other in recipe['modules']}:
            raise ValueError(f"{path}: module {item['name']!r} runs in {item['in']!r}, which is not a module")
        for source in item.get('consumes', ()):
            if source not in known and source.split(':', 1)[0] not in known:
                raise ValueError(f"{path}: module {item['name']!r} consumes {source!r}, which no input or "
                                 'earlier module provides')
        if item['use'] == 'role':
            _check_role(path, recipe, item)
        known.add(item['name'])
        known.update(item.get('emits', ()))
    if SEMANTIC in recipe['records']:
        raise ValueError(f'{path}: {SEMANTIC} is an interpretation and is never counted; it cannot be in "records"')
    view = PARTS.get(recipe.get('view'))
    if view is None or view.kind != 'view':
        raise ValueError(f'{path}: unknown view {recipe.get("view")!r}')
    return recipe


def upstream(modules, name):
    """The module ``name`` and everything it reads, directly or through other modules or derived inputs, in order."""
    provider = {key: item['name'] for item in modules for key in (item['name'], *item.get('emits', ()))}
    wanted, needed = [name], set()
    while wanted:
        current = wanted.pop()
        if current in needed:
            continue
        needed.add(current)
        item = next(item for item in modules if item['name'] == current)
        wanted += [owner for source in item.get('consumes', ())
                   if (owner := provider.get(source) or provider.get(source.split(':', 1)[0])) is not None]
    return [item for item in modules if item['name'] in needed]


def inputs_of(recipe, modules):
    """The recipe's inputs that these modules read."""
    read = {source for item in modules for source in item.get('consumes', ())}
    return [item for item in recipe['inputs'] if item['id'] in read]


def _sampled(recipe):
    return [item for item in recipe.get('modules', ()) if PARTS[item['use']].sampling is not None]


def _camera_fps(recipe, item, fps):
    """``fps`` is a camera's: it says nothing about a module that only reads sound or another sensor."""
    read = inputs_of(recipe, upstream(recipe['modules'], item['name'])) if not item.get('in') else []
    return fps if not read or any(entry['use'] not in ('microphone', 'sensor') for entry in read) else None


def check(recipe, *, fps=None, max_rev_s=None, symmetry=None, max_reps_s=None):
    """Hardware and sampling verdict for a camera frame rate (``fps=None``: what to buy)."""
    min_fps = (recipe.get('hardware') or {}).get('min_fps') or 0
    result = dict(recipe=recipe['name'])
    required, problems, ok = min_fps, [], True
    for item in _sampled(recipe):
        verdict = PARTS[item['use']].sampling(item, _camera_fps(recipe, item, fps), symmetry=symmetry, max_rev_s=max_rev_s,
                                              max_reps_s=max_reps_s)
        result.update({key: value for key, value in verdict.items() if key not in ('ok', 'problem')},
                      count=item['use'])
        required = max(required, verdict['required_fps'])
        if verdict.get('ok') is False:
            ok = False
            problems.append(verdict['problem'])
    result['required_fps'] = round(required, 1)
    if fps is None:
        return result
    # min_fps is the camera mode the recipe was measured with (a nominal 240 fps camera measures
    # ~239 fps); the sampling limit is physics and has no tolerance.
    if fps < .95 * min_fps:
        ok = False
        problems.insert(0, f'the recipe was measured at {min_fps:g} fps; this camera gives {fps:g} fps')
    return dict(result, fps=fps, ok=ok, problems=problems)


def _items(recipe):
    """(item, part) for the recipe's input, modules and view, in order."""
    items = list(recipe['inputs']) + list(recipe.get('modules', ())) + [dict(use=recipe['view'])]
    return [(item, PARTS[item['use']]) for item in items]


def _parts(recipe):
    return [part for _, part in _items(recipe)]


def _role(item):
    """(role, candidate) a module or input item needs; a role item without a candidate takes what is set up."""
    part = PARTS[item['use']]
    return (item['role'], item.get('candidate')) if item['use'] == 'role' else (part.role, item.get('candidate', part.candidate))


def roles(recipe):
    """``{role: candidate}`` of the role registry that the recipe's parts need."""
    from ..crew.registry import ROLES
    selected, open_choice = {}, []
    for item, part in _items(recipe):
        role, model = _role(item)
        chosen = ROLES[role].candidates.get(model) if role in ROLES and model else None
        if item.get('votes_for') and chosen is not None and 'api' in (chosen.config or {}):
            continue        # a hosted voter: nothing to set up, and its role keeps the recipe's own recogniser
        for role, model in [_role(item)] + [(role, item.get(key, default)) for key, role, default in part.extra_roles]:
            if role is None:
                continue
            if model is None:
                open_choice.append(role)
            elif selected.setdefault(role, model) != model:
                raise ValueError(f"{recipe['name']}: {role} is both {selected[role]} and {model}")
    for role in open_choice:    # what gamst setup chooses for this machine
        if role not in selected:
            from ..crew.setup import detect_hardware
            selected[role] = ROLES[role].defaults[detect_hardware()['tier']]
    return selected


def plan(recipe, models=None, hardware=None, isolation='venv'):
    """The setup plan for exactly the recipe's parts (``gamst.crew.setup.make_plan``; installs nothing).

    Same isolation as ``gamst setup``: one environment per model environment; only what the
    Pipeline process itself imports goes into this interpreter. ``models`` replaces the weight root.
    """
    from ..crew.setup import make_plan, models_root
    steps = make_plan(roles=roles(recipe), dependencies=False, isolation=isolation, hardware=hardware)
    steps['recipe'] = recipe['name']
    steps['models_dir'] = str(Path(models).expanduser().absolute() if models else models_root())
    return steps


def weight_files(steps):
    return [weight['name'] for selection in steps['roles'].values() for weight in selection['weights']]


def setup(recipe, *, models=None, dry_run=False, prefix=None, profile=None, isolation='venv'):
    """Install only the recipe's parts with the same apply as ``gamst setup``; writes the profile."""
    from ..crew.setup import SetupError, apply_plan
    steps = plan(recipe, models, isolation=isolation)
    print(json.dumps(steps, indent=2))
    if dry_run:
        return steps
    try:
        result = apply_plan(steps, prefix=prefix, profile_path=profile, models_dir=steps['models_dir'])
    except SetupError as error:
        raise SystemExit(f'setup: {error}') from error
    print(json.dumps(result, indent=2))
    return result


def installed(recipe, item, profile=None):
    """The profile entry setup wrote for this module's role and candidate."""
    from ..profile import read_profile
    role, model = _role(item)
    try:
        read = read_profile(profile)
    except FileNotFoundError:
        read = {}
    entry = (read.get('roles') or {}).get(role)
    if model is not None and (entry is None or entry.get('model') != model):
        # another candidate of the role, set up beside the profile's default one
        entry = (read.get('candidates') or {}).get(model)
        entry = entry if isinstance(entry, dict) and entry.get('model') == model else None
    if entry is None or model is not None and entry.get('model') != model:
        raise SystemExit(f"{role}{' ' + model if model else ''} is not set up; run: gamst recipe setup {recipe['name']}" + (
            f' --with-role {role}' if item['use'] == 'role' else ''))
    return entry


# ---------------------------------------------------------------- run

def _take_recording(path):
    path = Path(path)
    raw = path / 'raw.gmst' if path.is_dir() else path
    if not raw.is_file():
        raise SystemExit(f'no GAMST 240 recording at {raw}')
    return raw


def _take_fps(raw):
    from ..inputs import TakeInput
    return TakeInput(raw).fps()


def _source_fps(args):
    """The input's frame rate when it is known before the run (a take or a file), else None."""
    if getattr(args, 'take', None):
        return _take_fps(_take_recording(args.take))
    if getattr(args, 'video', None):
        return source_fps('video', args.video, args.capture_fps)
    return None


def source_fps(kind, path, capture_fps=None):
    if kind != 'video' or capture_fps:
        return capture_fps
    import cv2
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS) if video.isOpened() else 0
    video.release()
    return fps or None


def store_options(args):
    """Store arguments of a run: ``--max-records`` raises what is retained (a pose run commits several per frame)."""
    options = dict(store=getattr(args, 'store', None) or ':memory:')
    if getattr(args, 'max_records', None):
        options.update(max_records=args.max_records, max_payload_bytes=max(32 << 20, 8192 * args.max_records))
    return options


def _device():
    return 'mps' if sys.platform == 'darwin' and platform.machine() == 'arm64' else 'cpu'


def _given(args):
    return [name for name in SOURCES if getattr(args, name, None) is not None and getattr(args, name) is not False]


TIMINGS = ('realtime', 'post')


def timing(args):
    """``realtime`` (a camera, ``--as-live``, ``--realtime``: counts as frames arrive; frames the live lane cannot
    reach in time go to L2 and are counted late, as corrections) or ``post`` (a finished file: every frame, the
    largest batches, nothing is late). ``--timing`` names it; without it a live input or pacing is realtime."""
    live = any(name in LIVE for name in _given(args)) or getattr(args, 'as_live', False) or getattr(args, 'realtime', None)
    wanted = getattr(args, 'timing', None)
    if wanted == 'post' and live:
        raise SystemExit('--timing post counts a finished file; a camera, --as-live or --realtime is realtime')
    return wanted or ('realtime' if live else 'post')


async def choose_pose(recipe, args, pipeline):
    """Realtime, when the owner did not name ``--pose``: measure the recipe's pose model now (its live worker and, as
    far as the memory budget allows, an L2 worker, at the same time) and keep it if together they give at least the
    input's frame rate; else the item's ``choose.fallback`` (never both in one run). Returns what was measured and
    decided, or None when nothing was chosen."""
    import asyncio
    from ..memory_budget import budget, free_pct
    item = next((item for item in recipe['modules'] if item['use'] in POSE_SOURCES and item.get('choose')), None)
    if item is None or getattr(args, 'pose', None) or timing(args) != 'realtime':
        return None
    module = pipeline.modules[item['name']]['module']
    if not callable(getattr(module, 'benchmark', None)):
        return None
    choose, catch = item['choose'], item.get('catch_up') or {}
    fps = _source_fps(args) or float(choose.get('camera_fps', 30))
    persons, seconds = int(choose.get('persons', 1)), float(choose.get('seconds', 2))
    replicas = budget(replica_mb=float(catch.get('replica_mb') or 0), min_free_pct=float(catch.get('min_free_pct') or 0),
                      most=1) if catch else 0
    spare = await module.prestart() if replicas else None
    measured = await asyncio.gather(module.benchmark(seconds=seconds),
                                    *([spare.benchmark(seconds=seconds)] if spare is not None else []))

    def rate(found):     # frames per second for ``persons`` people, the detector at its interval
        detect = found['detect_ms'] or 0.
        per_frame = found['ms_per_frame'] * persons + (detect * 1000. / fps / found['det_every_ms']
                                                      if found['det_every_ms'] else detect)
        return 1000. / per_frame
    live = rate(measured[0])
    late = rate(measured[1]) if len(measured) > 1 else 0.
    keeps = live + late >= fps
    chosen = item['candidate'] if keeps else choose['fallback']
    reason = (f"{item['candidate']} gives {live:.1f} fps live" + (f" + {late:.1f} fps on an L2 worker" if late else "") +
              f" measured now for {persons} person(s); the input is {fps:g} fps: " +
              ('it keeps up' if keeps else f"too heavy for this machine, {choose['fallback']} counts instead"))
    return dict(model=item['candidate'], chosen=chosen, input_fps=fps, live_fps=round(live, 1), l2_fps=round(late, 1),
                l2_workers=len(measured) - 1, persons=persons, free_memory_pct=free_pct(), measured=list(measured),
                reason=reason)


MEASURED_PARTS = ('detect', 'ball-model')     # model parts whose realtime candidate is chosen by a start benchmark


async def choose_models(recipe, args, pipeline):
    """Realtime: measure each ``detect`` / ``ball-model`` item that has ``"choose"`` on synthetic frames now (``size``,
    ``seconds``), keep its candidate if it gives at least the input's frame rate, else its ``choose.fallback``; the
    same rule as :func:`choose_pose` (one model per part in a run, the reason said). Post: nothing is chosen, the
    item's candidate counts every frame."""
    import time
    import numpy as np
    from ..capabilities import SensorMetadata
    from ..plugins import InputFrame
    if timing(args) != 'realtime':
        return []
    choices = []
    for item in recipe['modules']:
        if item['use'] not in MEASURED_PARTS or not item.get('choose') or item['name'] not in pipeline.modules:
            continue
        module, choose = pipeline.modules[item['name']]['module'], item['choose']
        fps = _source_fps(args) or float(choose.get('camera_fps', 30))
        height, width = choose.get('size', [720, 1280])
        image = np.full((int(height), int(width), 3), 127, np.uint8)
        seconds, calls, started = float(choose.get('seconds', 2)), 0, None
        for index in range(10_000):
            frame = InputFrame(image, 'video', index + 1, 'local-user', None, SensorMetadata(
                source_id='gamst.choose', sensor_id='benchmark', sequence=index, coordinate_frame='camera.normalized'))
            await module.process(frame)
            if index == 0:
                started = time.perf_counter()        # the first call also warms the model up
                continue
            calls += 1
            if time.perf_counter() - started >= seconds and calls >= 3:
                break
        ms = (time.perf_counter() - started) * 1000 / max(1, calls)
        reset = getattr(module, 'reset', None)
        if callable(reset):
            value = reset()
            if hasattr(value, '__await__'):
                await value
        keeps = 1000 / ms >= fps
        chosen = item['candidate'] if keeps or not choose.get('fallback') else choose['fallback']
        choices.append(dict(module=item['name'], model=item['candidate'], chosen=chosen, ms_per_frame=round(ms, 1),
                            fps_possible=round(1000 / ms, 1), input_fps=fps, size=[int(height), int(width)],
                            reason=f"{item['candidate']} takes {ms:.1f} ms per {width}x{height} frame now "
                                   f"({1000 / ms:.1f} fps); the input is {fps:g} fps: " +
                                   ('it keeps up' if keeps else f"too slow for this machine, {chosen} counts instead"
                                    if chosen != item['candidate'] else 'too slow, and no fallback: frames will wait')))
    return choices


def with_candidates(recipe, chosen):
    """The recipe with ``chosen`` ({module: candidate}) as those modules' candidates."""
    return dict(recipe, modules=[dict(item, candidate=chosen[item['name']]) if item['name'] in chosen else item
                                 for item in recipe['modules']])


def _pairs(values, option):
    """``ID=VALUE`` options as a dict."""
    result = {}
    for value in values or ():
        key, equals, rest = value.partition('=')
        if not equals or not key or not rest or key in result:
            raise SystemExit(f'{option} {value}: ID=VALUE, once per input')
        result[key] = rest
    return result


def _source(recipe, args, inputs=None):
    """(input item, source name) of the one input that is read."""
    inputs = recipe['inputs'] if inputs is None else inputs
    ((item, chosen, _),) = _sources(recipe, args, inputs[:1])
    return item, chosen


def _leave_out(recipe, item):
    """What to leave out so that this input is not needed, for a message."""
    readers = [entry for entry in recipe['modules'] if item in inputs_of(recipe, [entry])]
    if not readers or not all(entry.get('optional') for entry in readers):
        return ''
    parts = sorted({(_listed(entry.get('parts')) or [entry['name']])[0] for entry in readers})
    return ', or leave out what reads it: ' + ' '.join(f'--without {name}' for name in parts)


def _unread(recipe, option):
    return f"{recipe['name']} reads nothing from --{option} here: drop it" + (
        ' (or run without --headless to see it, or with --serve so that an agent can add what reads it)' if option == 'camera'
        else ' (or add --serve, so that an agent can add what reads it)' if option in LIVE else '')


def _sources(recipe, args, inputs):
    """[(input item, source name, value)]: where each input is read from."""
    given, named = _given(args), _pairs(getattr(args, 'input', None), '--input')
    unknown = set(named) - {item['id'] for item in inputs}
    if unknown:
        raise SystemExit(f"{recipe['name']} has no input {', '.join(sorted(unknown))} to read here; inputs: "
                         f"{', '.join(item['id'] for item in inputs)}")
    result, used = [], set()
    for item in inputs:
        if item['use'] == 'plugin':
            chosen, value = 'plugin', item.get('plugin')
        elif item['id'] in named:
            chosen, _, value = named[item['id']].partition(':')
            value = True if chosen == 'microphone' else int(value or 0) if chosen == 'camera' else value
            if value == '':
                raise SystemExit(f"--input {item['id']}={chosen}:FILE")
        elif len(inputs) == 1:
            chosen = given[0] if given else item['sources'][0]
            value = getattr(args, chosen, None)
        else:
            found = [name for name in given if name in item['sources']]
            if item['use'] == 'microphone' and len(found) > 1 and 'video' in found:
                found.remove('video')       # its own sound only where no other sound is given
            if not found:
                raise SystemExit(f"{recipe['name']}: give a source for {item['id']}: "
                                 + ' | '.join(f'--{name}' for name in item['sources']) + _leave_out(recipe, item))
            if len(found) > 1:
                raise SystemExit(f"{recipe['name']}: {' and '.join('--' + name for name in found)} could both be {item['id']}; "
                                 f"say which: --input {item['id']}=SOURCE[:VALUE]")
            chosen = found[0]
            value = getattr(args, chosen)
        if chosen not in item['sources']:
            raise SystemExit(f"{recipe['name']} reads from {', '.join(item['sources'])}, not {chosen}")
        used.add(chosen)
        result.append((item, chosen, value))
    if len(inputs) > 1:
        unused = [name for name in given if name not in used]
        if unused:
            raise SystemExit(_unread(recipe, unused[0]))
        places = [(input_modality(item), chosen, str(value)) for item, chosen, value in result]
        twice = next((place for place in places if places.count(place) > 1), None)
        if twice is not None:
            raise SystemExit(f"{recipe['name']}: two inputs would read the same {twice[1]}; give one with "
                             '--input ID=SOURCE[:VALUE]')
    return result


def _offsets(recipe, args, inputs):
    """({input id: offset_ms}, the ids whose offset was stated)."""
    stated = _pairs(getattr(args, 'offset_ms', None), '--offset-ms')
    unknown = set(stated) - {item['id'] for item in inputs}
    if unknown:
        raise SystemExit(f"--offset-ms: {recipe['name']} reads no input {', '.join(sorted(unknown))} here")
    try:
        offsets = {item['id']: int(stated.get(item['id'], item.get('offset_ms', 0))) for item in inputs}
    except ValueError as error:
        raise SystemExit(f'--offset-ms ID=MS: {error}') from error
    if any(item['use'] == 'gamst240' and offsets[item['id']] for item in inputs):
        raise SystemExit('a GAMST 240 input keeps its own clock; it takes no offset_ms')
    return offsets, set(stated) | {item['id'] for item in inputs if 'offset_ms' in item}


def open_inputs(recipe, args, inputs):
    """{input id: source}, {input id: offset_ms}: every input opened on one capture clock."""
    from ..inputs import RecordedClock, media_streams, open_source
    chosen = _sources(recipe, args, inputs)
    offsets, stated = _offsets(recipe, args, inputs)
    given = {}
    for item, kind, value in chosen:
        if kind == 'plugin':
            from ..extensions import load_input
            from ..inputs import PluginInput
            selected_plugin(item, getattr(args, 'plugins', None))
            try:
                given[item['id']] = PluginInput(value, load_input(value, **(item.get('options') or {})), item['modality'])
            except (ValueError, TypeError, ImportError, RuntimeError, OSError) as error:
                raise SystemExit(f"input {item['id']}: {error}") from error
            if not given[item['id']].live and len(chosen) > 1:
                raise SystemExit(f"input {item['id']}: a recorded plugin input has no clock shared with others; run it alone")
    live = [given[item['id']].live if item['id'] in given else kind in LIVE for item, kind, _ in chosen]
    if any(live) and not all(live):
        raise SystemExit('a live input and a recording have no common capture clock; record both, or run both live')
    if any(live) and any(abs(value) > MAX_LIVE_OFFSET_MS for value in offsets.values()):
        raise SystemExit(f'a live input is off by at most {MAX_LIVE_OFFSET_MS} ms')
    clock, media, sources, streams = None, {}, {}, {}
    for item, kind, value in chosen:
        if kind == 'video':
            streams.setdefault(value, set()).add('audio' if item['use'] == 'microphone' else 'video')
    if len(chosen) > 1 and not any(live):
        clock = RecordedClock()
        if any(kind == 'take' for _, kind, _ in chosen):
            raise SystemExit('a take cannot be combined with another input')
        files = {}
        for item, _, value in chosen:
            files.setdefault(str(Path(value).expanduser().resolve()), []).append(item['id'])
        for path in list(files)[1:]:
            unplaced = [key for key in files[path] if key not in stated]
            if unplaced:
                raise SystemExit(f'{Path(path).name} and {Path(next(iter(files))).name} are different recordings; say how '
                                 f'they relate: --offset-ms {unplaced[0]}=MS (0: they start together)')
    try:
        for item, kind, value in chosen:
            if item['id'] in given:
                sources[item['id']] = given[item['id']]
                continue
            options = {}
            if kind == 'video' and item['use'] == 'microphone':
                kind = 'sound'
                if getattr(args, 'capture_fps', None):
                    raise SystemExit("--capture-fps changes the file's timing; its own sound does not follow it")
            if kind in ('video', 'sound') and (kind == 'sound' or clock is not None):
                info = media.get(value) or media.setdefault(value, media_streams(value))
                if kind == 'sound' and info['audio'] is None:
                    raise SystemExit(f'{value} has no sound track{_leave_out(recipe, item)}')
                own = info['audio' if kind == 'sound' else 'video']
                if own is None:
                    raise SystemExit(f'{value} has no pictures')
                options['at_s'] = own - min(info[name] for name in streams[value] if info[name] is not None)
                length = info['duration_s']
            if kind == 'video' and getattr(args, 'capture_fps', None):
                options['capture_fps'] = args.capture_fps
            if getattr(args, 'frames', None) and len(chosen) == 1 and kind in ('camera', 'video', 'take'):
                options['max_frames'] = args.frames
            if kind == 'sensor':
                options.update({key: item[key] for key in ('columns', 'time', 'modality') if key in item})
            # A GAMST 240 take replays through the phone's input, so its Start/Stop marks stay.
            kind = 'gamst240-take' if item['use'] == 'gamst240' and kind == 'take' else kind
            if clock is not None:
                options['clock'] = clock
            if kind == 'camera' and getattr(args, 'camera_max_side', None):
                options['max_side'] = args.camera_max_side
            source = sources[item['id']] = open_source(kind, None if value is True else value, **options)
            if getattr(args, 'realtime', None) and kind not in LIVE:
                from ..inputs import Paced
                sources[item['id']] = Paced(source, args.realtime)
            if clock is not None:
                if kind not in ('video', 'sound'):
                    length = source.duration_s()
                clock.add(length + options.get('at_s', 0.) + max(0, offsets[item['id']]) / 1000)
    except (RuntimeError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if getattr(args, 'as_live', False):
        from ..inputs import PacedInput
        if any(live):
            raise SystemExit('--as-live paces a recording; a live input has its own pace')
        pace = {}
        sources = {key: PacedInput(source, pace) for key, source in sources.items()}
    return sources, offsets


def from_rotation_demo(args):
    """``gamst demo rotation --iphone/--iphone-replay`` options -> (the coin-flip recipe, its run options).

    The demo keeps its own defaults: count each take on Stop (``--refine-on button``: on ANALYZE),
    one process per take that counts every throw unless ``--refine-fast``, and no counting at all
    without ``--refine`` (a camera monitor that records takes).
    """
    recipe = load('coin-flip')
    if not args.refine:
        recipe = dict(recipe, modules=[item for item in recipe['modules'] if item['use'] == 'take-recorder'],
                      records=[])
    options = argparse.Namespace(
        iphone=args.iphone, take=args.iphone_replay, label=args.refine, max_rev_s=args.expected_max_rev_s,
        device=args.device, models=args.models, worker_python=args.worker_python, store=args.store,
        headless=args.headless, report=args.report, record_dir=args.record_dir, refine_show=args.refine_show,
        refine_cached=args.refine_cached, efficienttam_source=args.efficienttam_source,
        count_on='analyze' if args.refine_on == 'button' else 'stop', fast=args.refine_fast)
    return recipe, options


def _check_exemplar(saved, path):
    """Refuse a pose exemplar whose channels were computed another way (it would silently not match)."""
    from ..providers.pose_exemplar import CHANNELS_VERSION
    channels = saved.get('channels', 'vision.pose_channels@1.')  # files without the field are version 1
    current = f'vision.pose_channels@{CHANNELS_VERSION}.'
    if channels.startswith('vision.pose_channels@') and not channels.startswith(current):
        raise SystemExit(f'{path} was enrolled before pose channels were corrected for the image aspect ratio; '
                         'enroll it again: gamst recipe enroll')


def _without(recipe, reference):
    """How to leave out the part that needs this reference, for a message."""
    readers = [item for item in recipe['modules'] if item['name'] in _listed(reference.get('for'))]
    if not readers or not all(item.get('optional') for item in readers):
        return ''
    return f"; or leave that part out: --without {(_listed(readers[0].get('parts')) or [readers[0]['name']])[-1]}"


def reference_values(recipe, args):
    """[(reference, value, consent)]; says everything that is missing at once, and how to run without it."""
    found, missing = [], []
    for reference in recipe.get('references', ()):
        try:
            found.append((reference, *_reference_value(reference, args)))
        except SystemExit as error:
            if reference['kind'] not in ('voice', 'person', 'exemplar', 'terms') or getattr(args, reference['kind'], None):
                raise       # given, but unusable
            missing.append(f'{error}{_without(recipe, reference)}')
    if missing:
        raise SystemExit('\n'.join(missing))
    return found


def _reference_value(reference, args):
    """(value, consent record or None) of one user-defined input from its command-line option."""
    kind, name = reference['kind'], reference['name']
    if kind in ('voice', 'person'):
        from .enrollment import saved_reference
        path = getattr(args, kind, None)
        if not path:
            raise SystemExit(f"this recipe needs an enrolled {kind}: run 'gamst recipe enroll --kind {kind}' and "
                             f'pass --{kind} FILE ({name})')
        value, consent = saved_reference(path, kind)
        if kind == 'voice' and hasattr(args, 'voice_min_quality'):    # a run: is the reference one person's voice?
            from ..voice_quality import from_args, hud_line
            value, consent, args.voice_quality = from_args(value, consent, args)
            if args.voice_quality['low']:
                print(f"note: {hud_line(args.voice_quality)}: the voice reference's pieces agree only "
                      f"{args.voice_quality['quality']['mean']:.2f} (mean pairwise, < {args.voice_quality['min_quality']:g}); "
                      f"used: {args.voice_quality['used']}", file=sys.stderr)
        if kind == 'voice' and getattr(args, 'voice_wav', None):
            value['wav'] = args.voice_wav   # extraction models condition on the clip itself
        return value, consent
    if kind == 'exemplar':
        if not args.exemplar:
            raise SystemExit(f"this recipe needs a demonstrated exemplar: run 'gamst recipe enroll' and pass "
                             f'--exemplar FILE ({name})')
        path = args.exemplar
        saved = json.loads(Path(path).read_text())
        if isinstance(saved, dict) and 'value' in saved:
            _check_exemplar(saved, path)
        if isinstance(saved, dict) and saved.get('format') == 2:
            _check_exemplar(saved, path)
            return saved, None
        return (saved['value'] if isinstance(saved, dict) and 'value' in saved else saved), None
    if kind == 'terms':
        if not args.terms:
            raise SystemExit(f'this recipe needs --terms word1,word2, or a file of gamst recipe enroll --kind terms ({name})')
        if args.terms.endswith('.json') and Path(args.terms).is_file():
            from .enrollment import saved_terms
            return saved_terms(args.terms), None
        return [item.strip() for item in args.terms.split(',') if item.strip()], None
    raise SystemExit(f'reference kind {kind!r} is not enrollable from the command line yet')


async def build_pipeline(recipe, args, human, *, sources=None, modules=None, lanes=True):
    """The recipe's Pipeline on ``human``: inputs, user-defined references, modules (no view).

    Modules that own a process run in their own lane (``gamst.pipeline``; an item may set ``"queue"``), and
    the recipe's ``"optional"`` ones pause while others are behind. ``lanes=False`` runs everything on the
    frame itself: enrollment pairs each frame with its own result and keeps no real time. ``modules``: only
    these, with the inputs and references they need. ``sources`` ({input id: source}; None: the host pushes
    the frames, ``Pipeline.push``) replaces what the command line names.
    """
    from ..pipeline import Pipeline
    from .perception_parts import remote_asr_preflight
    inputs = recipe['inputs']
    if modules is None:
        modules = recipe['modules']
    else:
        inputs = inputs_of(recipe, modules) or inputs
        present = {item['name'] for item in modules}
        recipe = dict(recipe, references=[item for item in recipe.get('references', ()) if not item.get('for')
                                          or present & set(_listed(item['for']))])
    remote_asr_preflight(recipe, modules, getattr(args, 'allow_remote_asr', False))
    pipeline = Pipeline(human, re.sub(r'[^A-Za-z0-9_.:-]', '-', recipe['name']))
    references = reference_values(recipe, args)
    pipeline.voice_quality = getattr(args, 'voice_quality', None)      # shown by the live view when low
    mode, search = _mode(args)
    pipeline.mode, pipeline.search = mode, search       # gamst.counting: count (default) or search
    if sources is None:
        sources, offsets = open_inputs(recipe, args, inputs)
        replay = any(kind == 'take' for _, kind, _ in _sources(recipe, args, inputs))
    else:
        (offsets, _), replay = _offsets(recipe, args, inputs), False
        if set(sources) != {item['id'] for item in inputs}:
            raise ValueError(f"sources must name {', '.join(item['id'] for item in inputs)}")
    for item in inputs:
        pipeline.add_input(item['id'], sources[item['id']], modality=input_modality(item), offset_ms=offsets[item['id']])
    for reference, value, consent in references:
        label = value.get('label') if reference['kind'] == 'exemplar' and isinstance(value, dict) else None
        pipeline.add_reference(reference['name'], reference['kind'], value, label=label or reference.get('label'),
                               consent=consent)
    context = dict(human=human, labels=args.label.split(',') if getattr(args, 'label', None) else None,
                   roi=getattr(args, 'roi', None), symmetry=getattr(args, 'symmetry', None),
                   max_rev_s=getattr(args, 'max_rev_s', None), max_reps_s=getattr(args, 'max_reps_s', None),
                   device=getattr(args, 'device', None) or _device(),
                   worker_python=getattr(args, 'worker_python', None),
                   installed=lambda item: installed(recipe, item, getattr(args, 'profile', None)),
                   models=getattr(args, 'models', None), enter=getattr(args, 'enter', None),
                   exit=getattr(args, 'exit', None), language=getattr(args, 'language', None),
                   speaker_threshold=getattr(args, 'speaker_threshold', None), inputs=sources, recipe=recipe,
                   mode=mode, search=search,
                   speaker_decision=_speaker_decision(getattr(args, 'speaker_decision', None)), l1=_l1(args), partials=_partials(args),
                   modules=modules, vlm_context=getattr(args, 'vlm_context', None),
                   remote_context=getattr(args, 'vlm_remote_context', False), plugins=getattr(args, 'plugins', None),
                   references={reference['name']: value for reference, value, _ in references},
                   allow_remote_asr=getattr(args, 'allow_remote_asr', False),
                   allow_remote_diarization=getattr(args, 'allow_remote_diarization', False),
                   record_dir=getattr(args, 'record_dir', None), replay=replay,
                   efficienttam_source=getattr(args, 'efficienttam_source', None),
                   count_on=getattr(args, 'count_on', None), fast=getattr(args, 'fast', None),
                   show=getattr(args, 'refine_show', False), cached=getattr(args, 'refine_cached', None))
    context['timing'] = timing(args)
    if getattr(args, 'court', None):     # the sports court reference: four marked corners (gamst/sports/court.py)
        from ..sports.court import court_map
        court = json.loads(Path(args.court).read_text())
        court_map(court)        # refused here, before any model starts, when the points do not fix a plane
        context['court'] = court
    context['stages'] = {item['name']: item for item in modules if item.get('in')}
    slow = slowdowns(getattr(args, 'debug_slowdown', None))
    unknown = set(slow) - {item['name'] for item in modules}
    if unknown:
        raise SystemExit(f"--debug-slowdown: no module {', '.join(sorted(unknown))} in this run")
    modules = [dict(item, slowdown=slow[item['name']]) if item['name'] in slow else item for item in modules]
    for item in modules:
        if item.get('in'):
            continue  # runs inside the module it names (its build reads context['stages'])
        try:
            built = await build_module(item, context)
            pipeline.add_module(item['name'], built['module'], **module_options(item, built, lanes=lanes))
        except BaseException:
            await pipeline.close()   # workers of the modules already built
            raise
    pipeline.recipe_context = context    # what a module added while it runs is built with
    return pipeline


async def build_module(item, context):
    """What the part's builder makes of one module item; a worker that does not start is named with its reason."""
    build = PARTS[item['use']].build
    if build is None:
        raise SystemExit(f"part {item['use']!r} runs inside the {context['recipe']['view']} view")
    try:
        return await build(item, context)
    except (RuntimeError, OSError, asyncio.TimeoutError) as error:
        from ..redact import public_text
        raise SystemExit(f"{item['name']} did not start: {public_text(error)}") from error


def _catch_up(config):
    """A module item's ``catch_up``; a ``pre_roll`` that no movement resolved (:func:`with_pre_roll`) is 1 s."""
    if not isinstance(config, dict):
        return config
    return {key: 1000 if value == 'pre_roll' else value for key, value in config.items()}


def module_options(item, built, *, lanes=True):
    """How the Pipeline runs a built module item (``Pipeline.add_module``)."""
    return dict(consumes=item['consumes'], spec=built.get('spec'), every_ms=built.get('every_ms'),
                queue=item.get('queue', built.get('queue', 'auto')) if lanes else None,
                optional=bool(item.get('optional')) and item.get('pausable', True), emits=tuple(item.get('emits', ())),
                enabled=item.get('enabled', True), catch_up=_catch_up(item.get('catch_up')) if lanes else False,
                slowdown=item.get('slowdown'), sample=item.get('sample') if lanes else None,
                backlog=item.get('backlog') if lanes else None, batch=item.get('batch') if lanes else None)


def slowdowns(values):
    """``--debug-slowdown MODULE=MS@FROM-TO[,FROM-TO]`` -> {module: {"ms", "spans_s"}} (seconds from the first frame)."""
    found = {}
    for value in values or ():
        try:
            name, rest = value.split('=', 1)
            ms, spans = rest.split('@', 1)
            found[name] = dict(ms=float(ms), spans_s=[[float(part) for part in span.split('-', 1)]
                                                      for span in spans.split(',')])
        except ValueError:
            raise SystemExit(f'--debug-slowdown {value!r}: write MODULE=MS@FROM-TO[,FROM-TO] (seconds)') from None
    return found


def stacks_table(name):
    """Per stack and timing: which part and candidate each changed module uses, and the stack's summary."""
    _, raw = read_recipe(name)
    table = dict(recipe=raw['name'], default=(raw.get('stacks') or {}).get('default'), stacks={})
    for stack in stack_names(raw):
        rows = {}
        for each in TIMINGS:
            recipe = variant(raw, stack=stack, timing=each)
            rows[each] = {item['name']: {key: item[key] for key in ('use', 'candidate', 'tracker', 'settings', 'embed')
                                         if key in item}
                          for item in recipe['modules'] if item['use'] in MODEL_PARTS or 'tracker' in item}
        table['stacks'][stack] = dict(summary=raw['stacks'][stack].get('summary'), **rows)
    return table


MODEL_PARTS = ('detect', 'ball-model', 'yolo26-pose', 'rtmw-pose', 'mediapipe-pose', 'sapiens2-pose', 'object-track',
               'team-split', 'team-split-siglip')


def _view_class(name):
    if name == 'halo-coin':
        from ..halo import HaloCoinView
        return HaloCoinView
    from ..views import CountView, RotationView
    return RotationView if name == 'rotation-view' else CountView


def speaker_bound(modules, name=None):
    """Whether what is heard (by module ``name``) is tied to an enrolled voice."""
    from .perception_parts import VOICE_PARTS
    return any(item['use'] in VOICE_PARTS for item in (modules if name is None else upstream(modules, name)))


class _LiveMCP:
    """The run's own session on the loopback server (token required), which ``gamst-mcp-live`` bridges to MCP.

    An agent reads the same records, timeline and counts, and switches modules, while the recipe runs. It cannot
    open another camera or microphone on this session: the recipe owns its inputs.
    """

    def __init__(self, human, port, proposals=None):
        from ..mcp import MCPServer
        self.port, self.server, self.task = port, MCPServer(human, movement_proposals=proposals), None
        for tool in ('start_capture', 'stop_capture', 'pause_capture', 'resume_capture'):
            self.server.tools.pop(tool, None)

    async def start(self):
        from ..server import serve_websocket
        self.task = asyncio.create_task(serve_websocket(self.server, self.port))
        await asyncio.sleep(.2)
        if self.task.done():
            try:
                self.task.result()
            except (OSError, RuntimeError) as error:
                raise SystemExit(f'--serve {self.port}: {error}') from error
        print(json.dumps(dict(serving=f'ws://127.0.0.1:{self.port}',
                              mcp=f'gamst-mcp-live --url ws://127.0.0.1:{self.port}')), flush=True)

    async def close(self):
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.server.close()


async def run_pipeline(recipe, args):
    from .. import Gamst
    from ..counting import MAX_SCAN, REVISION, count_events, current_instructions, is_current, newest_instructions, \
        revisions, with_state
    from ..engine import timestamp
    from .instructions import Instructions
    recipe = dict(recipe, records=list(recipe.get('records', ())))     # an instruction may add one
    report = dict(recipe=recipe['name'], frames=0, mode=_mode(args)[0],
                  **({'paced': True} if getattr(args, 'as_live', False) else {}))
    if getattr(args, 'voice_sample', None):     # where the person's voice sample came from (recipes/voice_sample.py)
        report['voice_sample'] = {key: value for key, value in args.voice_sample.items() if key != 'segments_s'}
        report['voice_sample']['excluded_s'] = args.voice_sample.get('segments_s') or (
            [[args.voice_sample['from_s'], args.voice_sample['to_s']]] if 'from_s' in args.voice_sample else [])
    inputs, first = recipe['inputs'], recipe['inputs'][0]['id']    # several: "frames" are those of the first input
    if len(inputs) > 1:
        report['input_frames'] = {item['id']: 0 for item in inputs}
    counted = next((item['id'] for item in inputs if item['use'] not in ('microphone', 'sensor')), first)   # of --frames
    truth = json.loads(Path(args.truth).read_text()) if getattr(args, 'truth', None) else None
    args_frames = getattr(args, 'frames', None)
    wanted, committed, earlier = set(recipe.get('records', ())), {}, set()
    if truth:
        wanted.add(truth['capability'])

    def keep(records):
        """Event records as they are committed: the store keeps a bounded number and evicts the oldest."""
        for record in records:
            observation = record.get('observation') or {}
            if record['id'] not in earlier and (observation.get('capability') or {}).get('id') in wanted and isinstance(
                    observation.get('value'), dict) and observation['value'].get('events'):
                committed[record['id']] = record

    # Every record is committed; the engine's human.inference record is made once a second instead of with every
    # one. Not for speed (0.3 ms): each is stored, so it would halve how long a run's own records are retained.
    async with Gamst(consent=True, sensor_inference_interval_ms=1000, **store_options(args)) as human:
        stored = [record for capability in wanted for record in human.store.query(
            'local-user', kind='observation.signal', limit=MAX_SCAN, registered_only=True, capability_id=capability)]
        earlier = {record['id'] for record in stored}   # a store file may hold other runs
        pipeline = await build_pipeline(recipe, args, human)
        report['timing'] = timing(args)
        choice = await choose_pose(recipe, args, pipeline)
        if choice is not None:
            print(json.dumps(dict(pose_choice={key: choice[key] for key in choice if key != 'measured'})),
                  file=sys.stderr, flush=True)
            if choice['chosen'] != choice['model']:
                await pipeline.close()
                recipe = with_pose(recipe, choice['chosen'])
                pipeline = await build_pipeline(recipe, args, human)
            report['pose_choice'] = choice
            pipeline.notes['pose_choice'] = choice
        models = await choose_models(recipe, args, pipeline)
        if models:
            print(json.dumps(dict(model_choice=models)), file=sys.stderr, flush=True)
            changed = {entry['module']: entry['chosen'] for entry in models if entry['chosen'] != entry['model']}
            if changed:
                await pipeline.close()
                recipe = with_candidates(recipe, changed)
                pipeline = await build_pipeline(recipe, args, human)
            report['model_choice'] = models
            pipeline.notes['model_choice'] = models
        if recipe.get('chosen'):
            report['variant'] = recipe['chosen']
            pipeline.notes['variant'] = recipe['chosen']
        pipeline.notes['timing'] = report['timing']
        view, idle, live, warned = None, None, None, dict(at=0., seen=set())
        pipeline.notify = lambda text: print(text, file=sys.stderr, flush=True)

        def added(capability):      # an instruction added a counting module
            wanted.add(capability)
            if capability not in recipe['records']:
                recipe['records'].append(capability)
            if view is not None:
                view.capabilities = tuple(dict.fromkeys((*view.capabilities, capability)))

        try:
            human.instructions = Instructions(recipe, args, pipeline, folders=getattr(args, 'clips', None) or (),
                                              proposals=getattr(args, 'proposals', None), on_capability=added)
            if getattr(args, 'serve', None):
                live = _LiveMCP(human, args.serve, getattr(args, 'proposals', None))
                await live.start()
            view = _view_class(recipe['view'])(pipeline, headless=args.headless,
                                               capabilities=recipe.get('records', ()))
        except BaseException:
            if live is not None:
                await live.close()
            await pipeline.close()
            raise

        async def on_frame(input_id, frame, records):
            started = frame.timestamp_ms - (
                round(1000 * len(frame.data) / frame.sample_rate) if frame.sample_rate else 0)
            if len(inputs) > 1:     # the run starts with the input that starts first
                if not report['input_frames'][input_id]:
                    report['start_ms'] = min(started, report.get('start_ms', started))
                report['input_frames'][input_id] += 1
            if 'start_ms' not in report and getattr(args, 'realtime', None):
                # replayed at its pace on the recording's own clock: knowledge time - this = capture clock
                report.update(realtime=args.realtime, clock_shift_ms=timestamp() - frame.timestamp_ms)
            report.setdefault('start_ms', started)
            report['frames'] += input_id == first
            keep(records)
            if time.monotonic() - warned['at'] > 5:
                warned['at'] = time.monotonic()
                for warning in pipeline.warnings():
                    if (warning['lane'], warning['input']) not in warned['seen']:
                        warned['seen'].add((warning['lane'], warning['input']))
                        print(f"warning: {warning['text']}", file=sys.stderr, flush=True)
            shown = await view.update(input_id, frame, records)
            if len(inputs) > 1 and args_frames and report['input_frames'][counted] >= args_frames:
                return False
            return shown

        if callable(getattr(view, 'idle', None)):
            idle = asyncio.create_task(view.idle())  # an interactive view stays drawn between frames
        try:
            began = time.monotonic()
            keep(await pipeline.run(on_frame=on_frame, stop_on_failure=bool(getattr(args, 'stop_on_failure', False))) or [])
            report['run_s'] = round(time.monotonic() - began, 2)   # the run itself: models were loaded before
            now = timestamp()
            for capability in wanted:   # plus the records of modules that commit their own
                keep(human.store.query('local-user', report.get('start_ms', 0), now, 'observation.signal',
                                       limit=MAX_SCAN, registered_only=True, capability_id=capability))
            report['counts'], report['events'], report['lost'] = {}, {}, {}
            rows = sorted(committed.values(), key=lambda record: record['timestamp_ms'])
            for capability in recipe.get('records', ()):
                own = [row for row in rows if row['observation']['capability']['id'] == capability]
                changes = [change for _, change in human.instruction_changes() if change['start_ms'] >= report.get('start_ms', 0)]
                counted = count_events(own, group_by='@event', max_evidence=100, current=current_instructions(
                    [change for change in changes if capability in change['capabilities']]))
                report['counts'][capability] = counted['groups']
                if counted.get('other_instructions'):   # what was looked for before: never added to the count
                    report.setdefault('other_instructions', {})[capability] = counted['other_instructions']
                report['events'][capability] = [item['event'] for item in counted['evidence']]
                if counted.get('states'):   # provisional + final are counted; revoked ones are not
                    report.setdefault('states', {})[capability] = counted.get('group_states') or counted['states']
                    report.setdefault('event_states', {})[capability] = event_states(
                        own, human.store.known_times([row['id'] for row in own]), paced=bool(report.get('paced')))
                    report['decision_clock'] = 'capture (paced as live: knowledge time)' if report.get('paced') else \
                        'capture (record time: the run was not paced, knowledge times are on another clock)'
                elif (report.get('paced') or 'clock_shift_ms' in report) and counted['groups']:   # when each was known
                    report.setdefault('event_states', {})[capability] = event_states(
                        own, human.store.known_times([row['id'] for row in own]), paced=True,
                        events=set(counted['groups']) - {'exemplar.ended'})
                    report['decision_clock'] = 'capture (paced as live: knowledge time)'
                report['lost'][capability] = pipeline.lost(capability)
            # Complete: no frame the count depends on was lost, failed or captured while a module was off.
            report['complete'] = not any(lost['gaps'] for lost in report['lost'].values())
            report['failures'], optional = pipeline.failures(), {key for key, item in pipeline.modules.items() if item['optional']}
            if any(item.get('module') not in optional for item in report['failures']):
                report['complete'] = False
            report['warnings'] = pipeline.warnings()
            report['mode'] = pipeline.mode            # gamst.counting; an instruction may have switched it
            if pipeline.mode_changes:
                report['mode_changes'] = [dict(item) for item in pipeline.mode_changes]
            if human.instruction_changes():
                report['instruction_changes'] = [change for _, change in human.instruction_changes()]
            if truth:
                from ..scoring import score
                own = [row for row in rows if row['observation']['capability']['id'] == truth['capability']]
                newest = newest_instructions(own, current_instructions(
                    [change for _, change in human.instruction_changes() if truth['capability'] in change['capabilities']]))
                found = [event for row in own for event in row['observation']['value']['events']
                         if is_current(event, newest) and event.get('event') != REVISION]
                revised = revisions(own)
                current = [event for event in (with_state(item, revised) for item in found)
                           if event.get('state') not in ('revoked', 'uncertain')]
                report['score'] = dict(score(truth, current, start_ms=report.get('start_ms', 0)),
                                       complete=not pipeline.lost(truth['capability'])['gaps'])
                if revised or any('state' in event for event in found):     # what was counted at once, before revisions
                    report['score_provisional'] = score(truth, [event for event in found if 'replaces' not in event],
                                                        start_ms=report.get('start_ms', 0))
            if any(PARTS[item['use']].transcribes or item['use'] == 'prosody' for item in recipe['modules']):
                report['speaker'] = 'enrolled' if speaker_bound(recipe['modules']) else 'not_bound'
                if getattr(args, 'voice_quality', None):    # how self-consistent the voice reference is
                    report['voice_quality'] = args.voice_quality      # (gamst/voice_quality.py)
            report['pipeline'] = pipeline.status()
            report['errors'] = {name: item['last_error'] for name, item in report['pipeline']['modules'].items()
                                if item['errors']}
        finally:
            if idle is not None:
                idle.cancel()
            view.close()
            if live is not None:
                await live.close()
            await pipeline.close()
    print(json.dumps({key: report[key] for key in ('recipe', 'mode', 'frames', 'input_frames', 'speaker', 'counts', 'other_instructions', 'complete', 'lost',
                                                   'states', 'failures', 'warnings', 'errors', 'score', 'score_provisional')
                      if key in report and (key not in ('lost', 'errors', 'failures', 'warnings', 'mode') or key == 'lost' and not report['complete']
                                            or key == 'mode' and report[key] != 'count' or key not in ('lost', 'mode') and report[key])},
                     ensure_ascii=False), flush=True)
    if getattr(args, 'report', None):
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    failed = [item for item in report['failures'] if 'input' in item]
    if failed and not any(item['frames'] for item in report['pipeline']['inputs'].values()):
        raise SystemExit('; '.join(f"{item['input']}: {item['error']}" for item in failed))
    return report


def event_states(rows, known, *, paced=False, events=None):
    """Each event with a revision state as it stands now, and when it was committed and settled.

    ``committed_ms`` / ``settled_ms``: when the Store learned the event and its revision (``known``: record ID ->
    knowledge time); ``capture_ms`` / ``settled_capture_ms``: the capture times of those records; ``committed_state``:
    the state it was committed in (``final`` when it said none).

    ``history``: every step of the event, oldest first: the commit, then each revision (an event may be revised more
    than once: uncertain, later final). Each step has ``state``, ``capture_ms`` (the capture time of the record that
    holds it) and ``decided_ms``: the moment it was decided on the capture clock, i.e. when it appears live. In a run
    paced as live (``paced``, ``--as-live``) that is when the Store learned it (the capture clock is the wall clock);
    in a run faster than real time the knowledge time is on another clock, so it is the record's capture time (the
    frame whose arrival decided it). A revision step also says ``reason``, ``basis`` and ``by`` (the lane that settled
    it, e.g. ``verification``, ``tse``) where it has them.
    """
    from ..counting import REVISION, revision_history, revisions, with_state
    revised, history, found = revisions(rows), revision_history(rows), []

    def decided(record):
        at = known.get(record['id']) if paced else None
        return record['timestamp_ms'] if at is None else at

    for row in rows:
        for item in row['observation']['value'].get('events') or ():
            if not isinstance(item, dict) or item.get('event') == REVISION:
                continue
            if 'event_id' not in item:   # final as committed: only when it was asked for (``events``: these names)
                if events is not None and item.get('event') in events:
                    found.append(dict(item, state=item.get('state', 'final'), committed_ms=known.get(row['id']),
                                      capture_ms=row['timestamp_ms'], committed_state=item.get('state', 'final'),
                                      history=[dict(state=item.get('state', 'final'), capture_ms=row['timestamp_ms'],
                                                    decided_ms=decided(row))]))
                continue
            entry = dict(with_state(item, revised), committed_ms=known.get(row['id']), capture_ms=row['timestamp_ms'],
                         committed_state=item.get('state', 'final'))
            revision = revised.get(item['event_id'])
            if revision is not None:
                entry.update(settled_ms=known.get(revision[1]['id']), settled_capture_ms=revision[1]['timestamp_ms'],
                             **({'settled_slot_score': revision[0]['slot_score']} if 'slot_score' in revision[0] else {}))
            steps = [dict(state=item.get('state', 'final'), capture_ms=row['timestamp_ms'], decided_ms=decided(row),
                          **({'by': item['decided_by']} if item.get('decided_by') else {}))]
            for change, record in history.get(item['event_id'], ()):
                steps.append(dict(state=change['state'], capture_ms=record['timestamp_ms'], decided_ms=decided(record),
                                  **{key: change[key] for key in ('reason', 'basis', 'by', 'candidate') if change.get(key)}))
            entry['history'] = steps
            found.append(entry)
    return found


async def enroll(recipe, args):
    """Save the owner's movement (clips, confirmed proposals or a rule), or an enrolled voice or person."""
    kind = getattr(args, 'kind', None) or 'exemplar'
    if kind == 'exemplar':
        from .movement import enroll as enroll_movement
        return await enroll_movement(recipe, args)
    if kind == 'terms':
        from .terms import enroll_terms
        return await enroll_terms(recipe, args)
    for name in ('video', 'audio'):
        values = getattr(args, name) or []
        if len(values) > 1:
            raise SystemExit(f'--kind {kind} reads one --{name}')
        setattr(args, name, values[0] if values else None)
    args.start = args.start or 0.
    if args.seconds is None:     # a voice needs speech: a whole recording (bounded), or 20 s of a microphone
        args.seconds = 1.5 if kind != 'voice' else 15. if getattr(args, 'main_speaker', False) else \
            20. if getattr(args, 'microphone', False) else 300.
    return await _enroll_identity(recipe, args, kind)


async def _enroll_identity(recipe, args, kind):
    from . import enrollment
    context = dict(installed=lambda item: installed(recipe, item, getattr(args, 'profile', None)))
    if kind == 'person':
        return await enrollment.enroll_person(recipe, args, build_pipeline=build_pipeline, context=context)
    item = next((item for item in recipe['modules'] if item['use'] in ('speaker-verify', 'speaker-tracks')), None)
    if item is None:
        raise SystemExit(f"{recipe['name']} has no speaker-verify module to enroll for")
    (inputs, sources) = (inputs_of(recipe, upstream(recipe['modules'], item['name'])), None)
    sources, _ = open_inputs(recipe, args, inputs)
    return await enrollment.enroll_voice(recipe, args, context=context, source=next(iter(sources.values())))


# ---------------------------------------------------------------- command line

def _kept(args):
    """Which inputs stay open although nothing reads them: a camera that is shown, and live ones an agent may need later."""
    named = _pairs(args.input, '--input')

    def keep(item):
        given = {name for name in LIVE if name in item.get('sources', ()) and name in _given(args)}
        if item['id'] in named:
            given |= {named[item['id']].partition(':')[0]} & set(LIVE)
        return bool(given) if args.serve else 'camera' in given and not args.headless
    return keep


def _optional_options(parser):
    parser.add_argument('--with', dest='include', action='append', default=[], metavar='MODULE',
                        help='add an optional module of the recipe')
    parser.add_argument('--without', dest='exclude', action='append', default=[], metavar='MODULE',
                        help='leave an optional module out')
    parser.add_argument('--with-role', dest='with_roles', action='append', default=[], metavar='ROLE[=CANDIDATE][@MS]',
                        help='add a registry role as a module (gamst recipe roles); @MS: look at one frame per MS')


def _variant_options(parser):
    parser.add_argument('--stack', metavar='NAME',
                        help="which models fill the recipe's parts (sports recipes: ultralytics, roboflow, sota; "
                             "gamst recipe stacks RECIPE lists them)")
    parser.add_argument('--sport', metavar='NAME',
                        help="the sport preset of a sports recipe (court, ball, fastest speed, detector labels)")


def _module_options(parser):
    parser.add_argument('--pose', metavar='CANDIDATE',
                        help='the vision.pose candidate that counts (person-action-count: rtmw.l_384x288 by default; '
                             'e.g. mediapipe.pose_full, mediapipe.pose_heavy); the part that builds it replaces the '
                             'recipe\'s, joints are read its way')
    parser.add_argument('--set', dest='settings', action='append', default=[], metavar='MODULE.KEY=VALUE',
                        help='one setting of a module item (VALUE is JSON, else text), e.g. pose.delegate=gpu, '
                             'pose.batch=4 (frames per pose call), pose.device=cpu, pose.sample=newest (a lane that skips to the '
                             'newest frame while busy; not the default: every frame is counted)')


def _asr_options(parser, consent=True):
    """``--asr`` (and ``--allow-remote-asr``) for the actions that transcribe or set up the transcribing module."""
    if consent:
        parser.add_argument('--allow-remote-asr', action='store_true',
                            help="consent to sending the sound to a hosted ASR module's paid API (stream-words, "
                                 "hosted-words; the default recogniser elevenlabs.scribe_v3_live.hosted is one)")
    parser.add_argument('--asr', metavar='CANDIDATE',
                        help='the audio.asr candidate that transcribes (e.g. elevenlabs.scribe_v3_live.hosted, '
                             'openai.gpt_live_transcribe.hosted, sherpa.zipformer_ko_streaming, '
                             'voxtral.mini_4b_realtime_2602.mlx); the part that builds it replaces the recipe\'s')


def _speed_options(parser):
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument('--max-rev-s', type=float, help="fastest turn rate to count (default: the recipe's)")
    speed.add_argument('--max-rpm', type=float, help='the same in revolutions per minute')
    parser.add_argument('--symmetry', type=int, choices=range(1, 65), metavar='N',
                        help='times the object looks the same per in-plane turn, e.g. blades of a fan '
                             "(default: the recipe's)")
    parser.add_argument('--max-reps-s', type=float, help="fastest repeated movement to count, per second")


def _source_options(parser, *, clips=False):
    source = parser
    many = dict(action='append', default=None) if clips else {}
    source.add_argument('--iphone', type=int, metavar='PORT', help='live GAMST 240 app (default port 9123)')
    source.add_argument('--take', metavar='DIR', help='a recorded GAMST 240 take (folder or raw.gmst)')
    source.add_argument('--camera', type=int, metavar='INDEX', help='a local camera')
    source.add_argument('--video', metavar='FILE[@start-end]' if clips else 'FILE', **many,
                        help='a video file (add --capture-fps for high-speed files)' + (
                            '; repeat it for more' if clips else ''))
    source.add_argument('--audio', metavar='FILE[@start-end]' if clips else 'FILE', **many,
                        help='a PCM WAV file' + ('; repeat it for more' if clips else ''))
    source.add_argument('--microphone', action='store_true', help='the default microphone')
    source.add_argument('--sensor', metavar='FILE[@start-end]' if clips else 'FILE', **many,
                        help='CSV or JSONL samples of a sensor input (the recipe declares columns, units and time)')
    parser.add_argument('--plugin', dest='plugins', action='append', metavar='NAME',
                        help='allow a user-made module or input the recipe names (an installed entry point): trusted code, run as it is')
    parser.add_argument('--input', action='append', metavar='ID=SOURCE[:VALUE]',
                        help='the source of one input by name, e.g. left=camera:1 or sound=audio:room.wav')
    parser.add_argument('--offset-ms', action='append', metavar='ID=MS',
                        help="a measured correction added to an input's capture times (default 0)")
    parser.add_argument('--camera-max-side', type=int, metavar='PX',
                        help='shrink camera frames to at most this many pixels on the longer side (less to compute)')
    parser.add_argument('--capture-fps', type=float, help='override file timing for high-fps captures')
    parser.add_argument('--frames', type=int, help='stop after this many frames (several inputs: of the video)')
    parser.add_argument('--as-live', action='store_true',
                        help='give a recording at its own pace, as a live capture: nothing waits for slow modules')
    parser.add_argument('--timing', choices=TIMINGS,
                        help='realtime: count as frames arrive, frames the live lane cannot reach in time are counted '
                             'late by L2 (corrections); post: a finished file, every frame in the largest batches, '
                             'nothing late (default: realtime for a camera, --as-live or --realtime, else post)')
    parser.add_argument('--models', help='weight root or an existing flat weights directory (default: the setup profile)')
    parser.add_argument('--worker-python', help='python for model worker processes (default: the setup profile)')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--label', help="what to find (default: the recipe's label)")
    parser.add_argument('--roi', type=lambda value: [float(item) for item in value.split(',')],
                        help='normalized x,y,width,height')
    parser.add_argument('--profile', help='setup profile to read (default ~/.config/gamst/profile.json)')


def _store_options(parser):
    parser.add_argument('--store', default=':memory:', help='SQLite store for the records (default: memory)')
    parser.add_argument('--max-records', type=int, metavar='N',
                        help='records the store retains before it evicts the oldest (default 5000)')


def _movement_options(parser):
    """What a run and a search need to know the movement and the person."""
    parser.add_argument('--exemplar', help='movement JSON from gamst recipe enroll')
    parser.add_argument('--person', help='person JSON from gamst recipe enroll --kind person')
    parser.add_argument('--voice', help='voice JSON from gamst recipe enroll --kind voice')
    parser.add_argument('--voice-wav', help="the enrolled person's clip, for separation models only")
    parser.add_argument('--speaker-threshold', type=float, help='cosine score that counts as the enrolled voice')
    parser.add_argument('--speaker-decision', metavar='JSON|FILE',
                        help='speaker decision v2: compare with the references AND the other voices present '
                             '(gamst/providers/voice_bank.py; {"l1": {...}, "l4": {...}, "bank": {...}})')
    parser.add_argument('--l1', choices=['instant', 'threshold'],
                        help="live lane (L1) with a verification lane: 'instant' counts every heard word at once and "
                             "excludes only a clearly other voice (person-term-count's default); 'threshold' counts a "
                             'word at once only when its voice score reaches --speaker-threshold (the old behaviour)')
    parser.add_argument('--l1-clear-other', type=float, metavar='SCORE',
                        help="with --l1 instant: a word whose voice score is below this is another person's at once "
                             '(default 0.1)')
    parser.add_argument('--l1-clear-margin', type=float, metavar='COSINE',
                        help='with --l1 instant and speaker decision v2: excluded at once when a known other voice is '
                             'this much closer than the person (default: off)')
    parser.add_argument('--partials', choices=['on', 'off'],
                        help="count a word the moment the live recogniser's partial hypothesis holds it steadily "
                             "('heard', corrected by the final; gamst/providers/partial_terms.py): 'off' waits for the "
                             "recogniser's final, as before (default: the recipe's)")
    parser.add_argument('--partial-revisions', type=int, metavar='N',
                        help='a word is heard once N partials in a row held it (default 2)')
    parser.add_argument('--partial-ms', type=float, metavar='MS',
                        help='... or once partials held it for MS of capture time (default: off)')
    parser.add_argument('--voice-fallback', action='append', default=[], metavar='FILE',
                        help='another enrolled voice of the same person (e.g. an owner sample from another recording): '
                             'used when the voice reference is not self-consistent (gamst/voice_quality.py)')
    parser.add_argument('--voice-min-quality', type=float, metavar='COSINE',
                        help='least self-consistency (mean pairwise similarity of its pieces) of a voice reference '
                             'used alone (default 0.5)')
    parser.add_argument('--voice-fallback-mode', choices=['replace', 'combined'],
                        help="a low-quality reference with a --voice-fallback: 'replace' (default: the fallback alone) "
                             "or 'combined' (both, equal weight)")
    parser.add_argument('--mode', choices=['count', 'search'],
                        help="'count' (default): today's decisions; 'search' (owner decision 33): never miss the person, "
                             'over-count rather than miss: words or moments that would be uncertain or revoked for their '
                             'voice or identity are counted as candidates with their reason (gamst/counting.py)')
    parser.add_argument('--search-revoke-margin', type=float, metavar='COSINE',
                        help="with --mode search: verification still revokes a word whose voice scores are all this much "
                             'under the speaker threshold (clearly another person; default 0.3)')
    parser.add_argument('--enter', type=float, help="match threshold (default: the movement file's)")
    parser.add_argument('--exit', type=float, help='release threshold')


def _l1(args):
    """The live lane's policy from ``--l1``, ``--l1-clear-other`` and ``--l1-clear-margin`` (None: the recipe's)."""
    found = {key: value for key, value in (('policy', getattr(args, 'l1', None)),
                                           ('clear_other', getattr(args, 'l1_clear_other', None)),
                                           ('clear_margin', getattr(args, 'l1_clear_margin', None))) if value is not None}
    if found and 'policy' not in found:
        found['policy'] = 'instant'
    return found or None


def _mode(args):
    """(mode, search settings or None) from ``--mode``, ``--search-revoke-margin`` and ``--l1-clear-other``."""
    from ..counting import search_settings
    mode = getattr(args, 'mode', None) or 'count'
    if mode != 'search':
        return mode, None
    given = {key: value for key, value in (('revoke_margin', getattr(args, 'search_revoke_margin', None)),
                                           ('l1_clear_other', getattr(args, 'l1_clear_other', None))) if value is not None}
    try:
        return mode, search_settings(given)
    except ValueError as error:
        raise SystemExit(f'--mode search: {error}') from None


def _partials(args):
    """Heard words from ``--partials``, ``--partial-revisions`` and ``--partial-ms`` (None: the recipe's; False: off)."""
    if getattr(args, 'partials', None) == 'off':
        return False
    found = {key: value for key, value in (('min_revisions', getattr(args, 'partial_revisions', None)),
                                           ('min_ms', getattr(args, 'partial_ms', None))) if value is not None}
    if getattr(args, 'partials', None) == 'on' or found:
        return found
    return None


def _speaker_decision(value):
    """``--speaker-decision``: a JSON object or a file of one (speaker decision v2, gamst/providers/voice_bank.py)::

        {"l1": {"floor": 0.4, "margin": 0.05},                 # the live lane: each word vs references and negatives
         "l4": {"decision": "relative", "floor": 0.4, "margin": 0.05, "word_margin": 0.05,
                "anchor_ms": [10000, 5000]},                    # verification: speakers by comparison, word-anchored
         "bank": {"combine": "max", "grow": true}}              # references (+ final words), negatives
    """
    if not value:
        return None
    try:
        found = json.loads(Path(value).read_text() if Path(value).is_file() else value)
    except (OSError, ValueError) as error:
        raise SystemExit(f'--speaker-decision is a JSON object or a file of one: {error}') from None
    if not isinstance(found, dict) or set(found) - {'l1', 'l4', 'bank'}:
        raise SystemExit('--speaker-decision has "l1", "l4" and "bank"')
    return found


def _max_rev_s(args):
    return args.max_rpm / 60 if getattr(args, 'max_rpm', None) else getattr(args, 'max_rev_s', None)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='gamst recipe', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('list', help='built-in recipes')
    table = actions.add_parser('roles', help='every registry role: usable as a module, refused (why), or prepared by hand')
    table.add_argument('--json', action='store_true', help='also what has been verified, as JSON lines')
    show = actions.add_parser('show', help='the recipe, what setup installs, and the frame rate it needs')
    show.add_argument('recipe')
    install = actions.add_parser('setup', help="install only the recipe's parts")
    install.add_argument('recipe')
    install.add_argument('--models', help='weight root (default GAMST_MODELS or ~/.cache/gamst/models); files go into '
                                          '<root>/<model>/, an existing flat directory is read in place')
    install.add_argument('--dry-run', action='store_true', help='print the plan, install nothing')
    install.add_argument('--prefix', help='environments and source checkouts (default ~/.local/share/gamst)')
    install.add_argument('--isolation', choices=['venv', 'current'], default='venv',
                         help='venv: one environment per model (default); current: everything into this python')
    install.add_argument('--profile', help='profile to write (default ~/.config/gamst/profile.json)')
    _asr_options(install, consent=False)
    install.add_argument('--timing', choices=TIMINGS,
                         help="set up the stack's models for this timing only (default: both realtime and post)")
    listing = actions.add_parser('stacks', help="a recipe's stacks: which candidate fills each part, realtime and post")
    listing.add_argument('recipe')
    verify = actions.add_parser('check', help='can this camera run the recipe without aliasing?')
    verify.add_argument('recipe')
    _variant_options(verify)
    verify.add_argument('--fps', type=float, help='camera frames per second (omit to get the required rate)')
    _speed_options(verify)
    learn = actions.add_parser('enroll', help='save a user-defined movement (example clips, a text or a rule), voice or person')
    learn.add_argument('recipe')
    _source_options(learn, clips=True)
    learn.add_argument('--negative', action='append', default=[], metavar='FILE[@start-end]',
                       help='a clip of something similar that must not count; repeat it')
    learn.add_argument('--start', type=float, help='seconds into the input where the demonstration starts')
    learn.add_argument('--seconds', type=float, help='length of the demonstration (a file: all of it; live: 1.5; a voice: 20 live, up to 300 of a file)')
    learn.add_argument('--mode', choices=['hold', 'sequence'], help="hold: one pose; sequence: a movement")
    learn.add_argument('--name', help='label of the movement in its records (default: the text)')
    learn.add_argument('--text', help='the movement in words: with --proposals or --rule')
    learn.add_argument('--rule', metavar='FILE', help='a rule as JSON (gamst/providers/movement_rule.py)')
    learn.add_argument('--proposals', metavar='FILE', help='use the confirmed (and rejected) proposals of --text')
    learn.add_argument('--proposal', action='append', default=[], metavar='ID', help='only this proposal; repeat it')
    learn.add_argument('--no-confirm', action='store_true',
                       help='use open proposals too; their records are marked confirmed: false')
    learn.add_argument('--kind', choices=['exemplar', 'voice', 'person', 'terms'], default='exemplar',
                       help='voice / person: save embeddings only (needs --consent); terms: the words said in example '
                            'clips, proposed first and saved when confirmed')
    learn.add_argument('--consent', help='who consented to being enrolled, and to what')
    learn.add_argument('--main-speaker', action='store_true',
                       help='--kind voice without a marked range: enroll the speaker who speaks alone longest in the '
                            'window (default the first 15 s), by pyannoteAI Precision-3 (needs --allow-remote-diarization)')
    learn.add_argument('--allow-remote-diarization', action='store_true',
                       help="consent to sending the window to pyannoteAI's paid API (--main-speaker)")
    learn.add_argument('--language', help="spoken language of term clips, e.g. ko (default: the recipe's)")
    _asr_options(learn)
    learn.add_argument('--out', help='enrollment JSON to write')
    start = actions.add_parser('run', help='run the recipe')
    start.add_argument('recipe')
    _source_options(start)
    _speed_options(start)
    _movement_options(start)
    from .voice_sample import options as voice_sample_options
    voice_sample_options(start)
    start.add_argument('--terms', help='comma-separated words to count (re:PATTERN for a pattern, word:W for W only as a whole word, word:W* '
                            'for whole words made of W, each repetition counted), or a terms file')
    start.add_argument('--language', help="spoken language, e.g. ko (default: the recipe's)")
    start.add_argument('--vlm-context', metavar='NAMES', help='what a VLM module gets as text besides the frames: '
                       "transcript,counts (default: both), one of them, or none")
    start.add_argument('--vlm-remote-context', action='store_true',
                       help='consent to sending the transcript and counts to a hosted VLM beside the frames (default: frames only)')
    _asr_options(start)
    start.add_argument('--allow-remote-diarization', action='store_true',
                       help="consent to sending the sound to a hosted diarization service's paid API (the verification "
                            "lane's pyannoteAI Precision-3 windows); separate from --allow-remote-asr")
    start.add_argument('--debug-slowdown', action='append', metavar='MODULE=MS@FROM-TO[,FROM-TO]',
                       help='test and debug: sleep MS before each audio frame of MODULE captured in those seconds of the '
                            'run, so its live lane falls behind on purpose (it catches up without losing audio)')
    start.add_argument('--realtime', nargs='?', type=float, const=1., metavar='SPEED',
                       help='replay a recording at its own speed (or SPEED times it), as a live capture arrives')
    start.add_argument('--truth', help='ground-truth JSON to score the run against (gamst/scoring.py)')
    start.add_argument('--court', metavar='FILE',
                       help='sports recipes: the court reference, JSON {"court": "tennis.singles", "points": [[x, y] x 4]} '
                            '(the four corners the camera sees, far left, far right, near right, near left; normalised '
                            'image coordinates, or pixels with "image_size")')
    _store_options(start)
    start.add_argument('--headless', action='store_true', help='no window; events as JSON lines')
    start.add_argument('--serve', nargs='?', type=int, const=8765, metavar='PORT',
                       help='let an agent read and switch this run over MCP: gamst-mcp-live --url ws://127.0.0.1:PORT '
                            '(this computer only, token required; default port 8765)')
    start.add_argument('--clips', action='append', metavar='DIR',
                       help='a folder whose example clips and enrolled files an instruction may use while the run goes on; repeat it')
    start.add_argument('--proposals', metavar='FILE', help='proposals file an agent may list and add to, and whose confirmed '
                       'proposals may become instructions')
    start.add_argument('--stop-on-failure', action='store_true',
                       help='end the run when an input or a needed module fails (default: the others go on, the run is incomplete)')
    start.add_argument('--report', help='write frames, counts and pipeline status as JSON')
    search = actions.add_parser('find', help='list where a movement occurs in recordings (the same counter, offline)')
    search.add_argument('recipe')
    _source_options(search, clips=True)
    _movement_options(search)
    _store_options(search)
    search.add_argument('--max-reps-s', type=float, help='fastest repeated movement to count, per second')
    seek = actions.add_parser('propose', help='a VLM looks for a described movement in a recording; counts nothing')
    seek.add_argument('recipe')
    seek.add_argument('--video', action='append', metavar='FILE', help='a recording to search; repeat it')
    seek.add_argument('--text', required=True, help='the movement in words')
    seek.add_argument('--proposals', metavar='FILE', default=PROPOSALS)
    seek.add_argument('--fps', type=float, default=2., help='frames per second the VLM sees')
    seek.add_argument('--window-s', type=float, default=8., help='seconds per VLM call')
    seek.add_argument('--overlap-s', type=float, default=2., help='seconds shared by neighbouring calls')
    seek.add_argument('--max-windows', type=int, help='most VLM calls per recording')
    seek.add_argument('--capture-fps', type=float, help='override file timing for high-fps captures')
    seek.add_argument('--profile', help='setup profile to read (default ~/.config/gamst/profile.json)')
    decide = actions.add_parser('proposals', help="list, confirm or reject proposed movements (the owner's decision)")
    decide.add_argument('decision', choices=['list', 'confirm', 'reject'])
    decide.add_argument('ids', nargs='*', metavar='ID[@start-end | =WORD,...]',
                        help='confirm may correct an interval, or the spellings of a term (the first names it)')
    decide.add_argument('--proposals', metavar='FILE', default=PROPOSALS)
    decide.add_argument('--status', choices=['open', 'confirmed', 'rejected'])
    decide.add_argument('--text')
    data = actions.add_parser('export', help='the events of a store as JSONL or CSV')
    data.add_argument('--store', required=True, help='SQLite store of a run (--store)')
    data.add_argument('--capability', help='only this capability, e.g. temporal.exemplar_events')
    data.add_argument('--event', help='only this event, e.g. exemplar.matched')
    data.add_argument('--subject', default='local-user')
    for command in (search, data):
        command.add_argument('--format', choices=['jsonl', 'csv'], default='jsonl')
        command.add_argument('--out', help='file to write')
    for command in (show, install, learn, start, search, seek):
        _optional_options(command)
        _module_options(command)
        _variant_options(command)
    takes = start.add_argument_group('GAMST 240 takes (coin-flip)')
    takes.add_argument('--record-dir', help='one folder per take: raw.gmst, hud.mp4, refined.json/mp4 '
                                            '(default gamst-takes)')
    takes.add_argument('--refine-show', action='store_true',
                       help='open refined.mp4 (8x slow motion with the running count) when it is ready')
    takes.add_argument('--refine-cached', metavar='TAKE_DIR',
                       help='stage backup: ANALYZE shows this already counted take at once')
    takes.add_argument('--all-throws', action='store_true',
                       help='count every throw of a take, one process per take (default: the best-named throw, '
                            'models kept loaded)')
    given = sys.argv[1:] if argv is None else list(argv)
    if given[:1] == ['bench']:       # its own options; the rest goes to each run
        from .bench import main as bench
        return bench(given[1:])
    actions.add_parser('bench', help='one recipe on several owner clips, paced as live, one table (gamst/recipes/bench.py)')
    args = parser.parse_args(argv)
    if getattr(args, 'all_throws', False):
        args.fast = False
    if args.action == 'list':
        for name in names():
            print(f"{name:20} {load(name)['summary']}")
        return
    if args.action == 'roles':
        from ..crew.modules import role_report
        for row in role_report():
            print(json.dumps(row, ensure_ascii=False) if args.json else
                  f"{row['role']:26} {row['candidate']:34} {row['state']:8} {row['reason'] or ''}")
        return
    if args.action in ('proposals', 'export'):
        from . import movement
        return getattr(movement, args.action)(args)
    if args.action == 'stacks':
        return print(json.dumps(stacks_table(args.recipe), indent=2, ensure_ascii=False))
    variants = dict(stack=getattr(args, 'stack', None), sport=getattr(args, 'sport', None))
    if args.action in ('run', 'enroll', 'find'):
        variants['timing'] = timing(args)
    elif args.action == 'setup' and stack_names(read_recipe(args.recipe)[1]):
        # Both timing variants of the stack (or the one named), and the realtime fallbacks a start benchmark may pick;
        # setup skips what an earlier pass installed.
        for each in [args.timing] if getattr(args, 'timing', None) else TIMINGS:
            each_recipe = load(args.recipe, getattr(args, 'include', ()), getattr(args, 'exclude', ()),
                               getattr(args, 'with_roles', ()), None, **variants, timing=each)
            fallbacks = {item['name']: item['choose']['fallback'] for item in each_recipe['modules']
                         if each == 'realtime' and item['use'] in MEASURED_PARTS
                         and (item.get('choose') or {}).get('fallback')}
            for one in [each_recipe] + ([with_candidates(each_recipe, fallbacks)] if fallbacks else []):
                setup(one, models=args.models, dry_run=args.dry_run, prefix=args.prefix, profile=args.profile,
                      isolation=args.isolation)
        return
    else:
        variants['timing'] = getattr(args, 'timing', None)
    recipe = load(args.recipe, getattr(args, 'include', ()), getattr(args, 'exclude', ()), getattr(args, 'with_roles', ()),
                  _kept(args) if args.action == 'run' else None, **variants)
    if getattr(args, 'asr', None):
        recipe = with_asr(recipe, args.asr)
    if getattr(args, 'pose', None):
        recipe = with_pose(recipe, args.pose)
    if getattr(args, 'settings', None):
        recipe = with_settings(recipe, args.settings)
    if getattr(args, 'timing', None) == 'realtime' and not getattr(args, 'realtime', None) and \
            not any(name in LIVE for name in _given(args)):
        args.as_live = True        # a file counted in realtime: given at its own pace, as a camera would
    if getattr(args, 'exemplar', None):
        recipe = with_pre_roll(recipe, args.exemplar)
    if len(recipe['inputs']) == 1 and args.action in ('run', 'enroll', 'find') and len(_given(args)) > 1:
        unread = [name for name in _given(args) if name not in recipe['inputs'][0]['sources']]
        if unread and len(read_recipe(args.recipe)[1]['inputs']) > 1:
            raise SystemExit(_unread(recipe, unread[0]))
        parser.error(' and '.join('--' + name for name in _given(args)) + f": {recipe['name']} reads one source")
    if args.action == 'enroll' and not args.out and args.kind != 'terms':
        parser.error('the following arguments are required: --out')
    if args.action in ('find', 'propose'):
        from . import movement
        return asyncio.run(getattr(movement, args.action)(recipe, args))
    if args.action == 'show':
        print(json.dumps(dict(recipe, setup=plan(recipe), sampling=check(recipe)), indent=2, ensure_ascii=False))
    elif args.action == 'setup':
        setup(recipe, models=args.models, dry_run=args.dry_run, prefix=args.prefix, profile=args.profile,
              isolation=args.isolation)
    elif args.action == 'check':
        verdict = check(recipe, fps=args.fps, max_rev_s=_max_rev_s(args), symmetry=args.symmetry,
                        max_reps_s=args.max_reps_s)
        print(json.dumps(verdict, indent=2))
        if verdict.get('ok') is False:
            raise SystemExit(1)
    elif args.action == 'enroll':
        return asyncio.run(enroll(recipe, args))
    else:
        args.max_rev_s = _max_rev_s(args)
        verdict = check(recipe, fps=_source_fps(args), max_rev_s=args.max_rev_s, symmetry=args.symmetry,
                        max_reps_s=args.max_reps_s)
        if verdict.get('ok') is False:
            raise SystemExit('; '.join(verdict['problems']))
        print(json.dumps(dict(recipe=recipe['name'], sampling=verdict)), flush=True)
        from . import voice_sample
        args.voice_sample = asyncio.run(voice_sample.prepare(recipe, args))
        try:
            return asyncio.run(run_pipeline(recipe, args))
        finally:
            voice_sample.cleanup(args)
