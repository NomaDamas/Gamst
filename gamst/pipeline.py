"""The composition layer: which inputs feed which perception modules, switched on and off at runtime.

GAMST's layers, front to back::

    inputs (camera, microphone, files, takes, user-defined inputs, derived inputs)
      -> Pipeline (this module: what is connected and what is on)
      -> perception modules (light models, math and physics)
      -> context (Store records, Watches, MCP)

A recipe builds a Pipeline; an agent or the owner switches its modules on and off while capture
runs (``enable`` / ``disable``, also over MCP). The Pipeline executes nothing of its own:

- a *plugin* module (``process(frame) -> [signal, ...]``) commits through
  ``Gamst.ingest_plugin_output``, like a registered plugin;
- a *provider* module (a ``CapabilitySpec`` plus ``process(frame) -> value``) commits through
  ``Gamst.push_observation``, like a SensorStreams lane.

Two kinds of links connect modules:

- **record links**: a module consumes another module's output by naming it in ``consumes``
  (``"pose"``, or ``"pose:scene.poses"`` for one output name); it receives each committed value as
  a frame at the same capture time, with the record's modality;
- **derived inputs**: a module returns ``Derived(...)`` items (for example one person's voice
  separated from the room). They become frames of a new input, routed to the modules that consume
  it, with the original capture time and ``derived_from`` provenance. Derived media never enters the
  Store or any context; only records do.

A module added with ``queue=N`` runs in its own bounded lane, so a slow model never blocks the input
or the modules after it; this is how an isolated worker (``gamst.crew.modules.WorkerModule``) runs
next to per-frame math. ``queue='auto'`` (recipes, ``ProfileRuntime``) gives a lane to every module
that owns a process and keeps math inline.

Delivery is lossless: every frame reaches every module, once and in capture order. Live capture does
not wait, so a module that is slower than its input builds a backlog (``status()``: ``backlog``,
``backlog_ms``), bounded in frames and, across all lanes, in bytes (modules reading the same frame share
its array). When a bound comes near, modules declared ``optional`` (which no other module needs) pause
until the backlog is clear; when it is exceeded anyway, the oldest frames of that lane are given up as
one declared *gap*: counted, committed as a ``pipeline.gap`` event, and marked on the events and counts
downstream. What happens to a frame that does not fit is one decision (:meth:`Pipeline._overflow`). Each
gap says why: ``input`` and ``capture_paused`` are data that never reached GAMST; ``error`` is a frame a
module failed on; ``backlog``, ``paused`` and ``disabled`` are captured frames a module did not process.
``complete_until_ms`` of a module only advances over capture time that was processed or declared so.
Audio is never joined across a gap.

The one exception is declared sampling: a module with ``every_ms`` looks at one frame per interval; the
others are reported as ``sampled``, and are no gap.

A module that computes without a process and reads a module in a lane joins that lane: one frame runs
the whole chain (pose -> person -> channels -> count) in capture order, also where such a module reads
the input itself. A module with ``replicas`` (stateless workers of one spec) works on several frames of
its lane at once; results are still committed in capture order. Recorded inputs (``live = False``) wait
for the lanes, so nothing builds up and nothing is given up.
``PipelineIntake`` feeds a host-fed input from the engine's own intake (``push_video`` /
``push_audio``, MCP ``start_capture``, the dashboard), which is how ``ProfileRuntime`` runs the
roles chosen by setup as one Pipeline.

User-defined reference inputs (a voice sample, photos, a demonstrated pose, a list of words) are
given with ``add_reference``. Modules read them in ``bind(pipeline)``; the pipeline keeps only what
the module returns from ``bind`` (an embedding, a feature template), never the raw sample, unless
the module keeps it itself.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field, replace
import inspect
import time
from typing import Any

from .capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from .plugins import InputFrame

GAPS = CapabilitySpec('pipeline.gaps', 'pipeline.gaps', 'events', 'gamst.pipeline', '1',
                      (OutputField('events', 'array'),), 'frame', interpretation='measurement', license='Apache-2.0')
BACKLOG = 900                            # frames a lane holds (30 s of 30 fps video) ...
BACKLOG_BYTES = 512 * 1024 * 1024        # ... and what all lanes of a Pipeline hold together
LOG = 512                                # gap intervals and timings kept per module
GAP_RECORD_S = 1.                        # at most one gap record per place and second; merged in between


@dataclass(frozen=True)
class Derived:
    """A module output that becomes a frame of another input (media stays out of context)."""
    input: str
    data: Any
    modality: str
    sample_rate: int | None = None
    timestamp_ms: int | None = None   # capture time of this item; default: the frame it came from


@dataclass(eq=False)
class _Item:
    """One frame in a lane, for the modules of that lane that read it."""
    names: list
    frame: InputFrame
    media: bool          # a frame of an input (not a record): audio media is never joined across a gap
    gap: bool = False    # frames were lost just before this one


@dataclass
class _Context:
    """Where a module runs: the lane whose task this is (None: on the input), and what failed under it."""
    lane: dict | None = None
    failures: list = field(default_factory=list)


def _size(frame):
    data = frame.data
    return len(data) if isinstance(data, (bytes, bytearray, memoryview)) else int(getattr(data, 'nbytes', 0) or 0)


def _percentiles(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(p50=ordered[len(ordered) // 2], p95=ordered[min(len(ordered) - 1, int(.95 * len(ordered)))],
                samples=len(ordered))


def _now_ms():
    return time.time_ns() // 1_000_000


async def _call(function, *args):
    value = function(*args)
    return await value if inspect.isawaitable(value) else value


class Pipeline:
    def __init__(self, human, name='pipeline', *, max_backlog_bytes=BACKLOG_BYTES):
        identifier(name, 'pipeline name')
        self.max_backlog_bytes = int(max_backlog_bytes)
        self.human, self.name = human, name
        self.inputs: dict[str, dict] = {}
        self.modules: dict[str, dict] = {}
        self.references: dict[str, dict] = {}
        self._sequence: dict[str, int] = {}
        self._closed = False
        self._lanes: list[dict] = []
        self._held: dict[int, list] = {}     # id(array) -> [lane items holding it, bytes]
        self._held_bytes = 0
        self._paced = False                  # recorded inputs, or the input ended: lanes are waited for
        self._recorded = False               # capture times are not now: ages and lags would mean nothing
        self._collected = None               # records committed by lanes, until the next on_frame
        self._running, self._failure = False, None
        self._gaps: dict[str, dict] = {}     # not yet committed, by place
        self._gap_at: dict[str, float] = {}
        self._gap_sequence = 0
        self._pausable: dict[str, bool] = {}
        human.pipeline = self

    # ------------------------------------------------------------------ composition

    def add_input(self, input_id, source=None, *, modality=None, derived_from=None):
        """``source`` has ``frames()`` (an async iterator of InputFrame); None means the host feeds it."""
        identifier(input_id, 'input id')
        if input_id in self.inputs or input_id in self.modules:
            raise ValueError(f'{input_id!r} is already an input or module')
        if source is not None and not callable(getattr(source, 'frames', None)):
            raise TypeError('an input source needs frames()')
        self.inputs[input_id] = dict(source=source, modality=modality or getattr(source, 'modality', None),
                                     derived_from=derived_from, frames=0, last_ms=None, done_ms=None, missed=0, gaps=0, off=None)
        return self

    def add_reference(self, name, kind, value, *, label=None, consent=None):
        """A user-defined reference input; modules turn it into what they keep in ``bind``."""
        identifier(name, 'reference name')
        if kind not in ('voice', 'person', 'face', 'body', 'exemplar', 'terms', 'object'):
            raise ValueError('reference kind must be voice, person, face, body, exemplar, terms or object')
        if kind in ('voice', 'person', 'face', 'body') and not consent:
            raise ValueError('a reference to a person needs a consent record')
        self.references[name] = dict(kind=kind, value=value, label=label or name, consent=consent)
        return self

    def add_module(self, name, module, *, consumes, spec=None, enabled=True, every_ms=None, emits=(), queue=None,
                   optional=False):
        """``queue``: None (inline, or in the lane of the module it reads), a lane of N frames, or 'auto'.

        ``every_ms`` declares sampling (else every frame is processed). ``optional`` allows pausing the
        module while others are behind. A module with ``replicas`` > 1 gets a lane.
        """
        identifier(name, 'module name')
        if queue not in (None, 'auto') and (type(queue) is not int or not 1 <= queue <= 10000):
            raise ValueError("queue must be an integer from 1 to 10000, 'auto' or None")
        if name in self.modules or name in self.inputs:
            raise ValueError(f'{name!r} is already an input or module')
        if not callable(getattr(module, 'process', None)):
            raise TypeError('a module needs process(frame)')
        if getattr(module, 'private', False):
            raise PermissionError('a private worker answers its owning module only')
        consumes = [consumes] if isinstance(consumes, str) else list(consumes)
        if not consumes:
            raise ValueError('a module consumes at least one input or module')
        width = int(getattr(module, 'replicas', 1))
        if width > 1 and (spec is None or spec.stateful):
            raise ValueError(f'{name!r} cannot run in replicas: only a module whose spec is stateless may')
        links = [tuple(item.split(':', 1)) if ':' in item and item not in self.inputs else (item, None)
                 for item in consumes]
        if spec is not None:
            self.human.register_capability(spec)
        for derived in emits:
            self.add_input(derived, derived_from=name)
        bind = getattr(module, 'bind', None)
        if callable(bind):
            bind(self)
        if queue == 'auto' or queue is None and width > 1:
            queue = None if width == 1 and not getattr(module, 'owns_process', False) else 1 if every_ms else BACKLOG
        if queue is not None:
            lane = dict(name=name, sampled=bool(every_ms), frames=deque(), busy=[], capacity=queue, task=None,
                        dropped=0, lost=False, width=width, max_backlog_ms=0)
            self._lanes.append(lane)
        else:
            lane = self._upstream_lane(links)
        self._pausable.clear()
        self.modules[name] = dict(module=module, consumes=consumes, links=links, spec=spec, enabled=bool(enabled),
                                  every_ms=every_ms, last_run_ms=None, processed=0, errors=0,
                                  last_error=None, last_latency_ms=None, needs_reset=False, emits=tuple(emits),
                                  last_admit_ms=None, lane=lane, optional=bool(optional), paused=False, pauses=0,
                                  off=None, skipped=0, gaps=0, gap_frames=0, gap_out=False, gap_log=deque(maxlen=LOG),
                                  newest_ms=None, until=None, latency=deque(maxlen=LOG), lag=deque(maxlen=LOG))
        return self

    def _upstream_lane(self, links):
        """The lane of a module this one reads: both then see each capture in the same order."""
        for source, _ in links:
            entry = self.modules.get(source) or self.modules.get(self.inputs.get(source, {}).get('derived_from'))
            if entry is not None and entry['lane'] is not None and not entry['lane']['sampled']:
                return entry['lane']
        return None

    def enable(self, name):
        self._module(name)['enabled'] = True
        return self.status()['modules'][name]

    def disable(self, name):
        entry = self._module(name)
        entry['enabled'] = False
        entry['needs_reset'] = True  # what it saw before the gap must not join what it sees after
        lane = entry['lane']
        if lane is not None:
            for item in list(lane['frames']):
                if name in item.names:
                    item.names.remove(name)
                    self._miss(entry, item.frame)
                if not item.names:
                    lane['frames'].remove(item)
                    self._release(item)
        return self.status()['modules'][name]

    async def configure(self, name, options):
        """Pass module options (for example a MediaPipe graph's branches or side packets) to ``configure``."""
        entry = self._module(name)
        configure = getattr(entry['module'], 'configure', None)
        if not callable(configure):
            raise ValueError(f'module {name!r} has no options to configure')
        await _call(configure, dict(options))
        return self.status()['modules'][name]

    async def reset(self):
        """Discard queued frames and reset every module before its next frame (a capture gap)."""
        tasks = []
        for entry in self.modules.values():
            entry['needs_reset'] = True
        for lane in self._lanes:
            if lane['task'] is not None and not lane['task'].done():
                lane['task'].cancel()
                tasks.append(lane['task'])
            for item in [*lane['busy'], *lane['frames']]:
                ms = item.frame.timestamp_ms
                self._gap(lane['name'], item.names, ms, ms, 1, 'reset', record=False)   # the host discarded it
            while lane['frames']:
                self._release(lane['frames'].popleft())
            lane['task'] = None
        await asyncio.gather(*tasks, return_exceptions=True)

    def _module(self, name):
        if name not in self.modules:
            raise KeyError(f'unknown module {name!r}; modules: {sorted(self.modules)}')
        return self.modules[name]

    # ------------------------------------------------------------------ execution

    def _frame(self, input_id, data, modality, timestamp_ms, *, subject_id='local-user', sample_rate=None,
               sensor=None):
        sequence = self._sequence.get(input_id, 0)
        self._sequence[input_id] = sequence + 1
        if sensor is None:
            # Detectors and trackers require the frame to say its boxes are in image coordinates.
            sensor = SensorMetadata(source_id=self.name, sensor_id=input_id, sequence=sequence,
                                    coordinate_frame='image' if modality == 'video' else None)
        else:
            capture = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
            sensor = replace(sensor, sensor_id=input_id, sequence=sequence, capture_sequence=capture)
        return InputFrame(data=data, modality=modality, timestamp_ms=timestamp_ms, subject_id=subject_id,
                          sample_rate=sample_rate, sensor=sensor)

    async def push(self, input_id, data, *, timestamp_ms, modality=None, subject_id='local-user',
                   sample_rate=None):
        """Host-fed input: one capture-stamped frame of ``input_id``."""
        entry = self.inputs.get(input_id)
        if entry is None:
            raise KeyError(f'unknown input {input_id!r}')
        frame = self._frame(input_id, data, modality or entry['modality'], int(timestamp_ms),
                            subject_id=subject_id, sample_rate=sample_rate)
        return await self.feed(input_id, frame)

    async def feed(self, source, frame, *, output=None, gap=False, context=None):
        """Route one frame of ``source`` (an input, or a module's ``output``) to the modules consuming it.

        ``gap``: frames of this input were lost just before it. ``context``: set while a module's own
        output is routed on; without it, the first failure of a module run here is raised at the end,
        after every other module got the frame.
        """
        if self._closed or self.human.paused or self.human.closed or not self.human.consent:
            if source in self.inputs and not self._closed:
                self._miss(self.inputs[source], frame, 'capture_paused')
            return []
        own = context if context is not None else _Context()
        media = source in self.inputs
        if media:
            entry = self.inputs[source]
            entry['frames'] += 1
            entry['last_ms'] = frame.timestamp_ms
            if entry['off'] is not None:
                off, entry['off'] = entry['off'], None
                gap = True
                self._gap(source, [source], off['start_ms'], off['end_ms'], off['frames'], off['reason'],
                          subject_id=off['subject_id'])
        records, queued = [], []
        for name, entry in self.modules.items():
            if not any(src == source and wanted in (None, output) for src, wanted in entry['links']):
                continue
            if not entry['enabled'] or entry['paused']:
                self._miss(entry, frame)
                continue
            other = entry['lane'] is not None and entry['lane'] is not own.lane
            last = entry['last_admit_ms'] if other else entry['last_run_ms']
            if entry['every_ms'] and last is not None and frame.timestamp_ms - last < entry['every_ms']:
                entry['skipped'] += 1
                continue
            if entry['off'] is not None:
                self._declare_off(name, entry)
            if not other:
                records += await self._run_module(name, entry, frame, own, media=media, gap=gap)
                continue
            entry['last_admit_ms'] = frame.timestamp_ms
            item = next((item for lane, item in queued if lane is entry['lane']), None)
            if item is None:
                item = _Item([], frame, media, gap)
                queued.append((entry['lane'], item))
            item.names.append(name)
        for lane, item in queued:
            self._enqueue(lane, item)
        if media:
            self.inputs[source]['done_ms'] = frame.timestamp_ms
        if self._gaps:
            await self.commit_gaps()
        if context is None and own.failures:
            raise own.failures[0][1]
        return records

    # ------------------------------------------------------------------ lanes

    def _enqueue(self, lane, item):
        if item.media and lane['lost']:
            item.gap, lane['lost'] = True, False
        data = id(item.frame.data)
        if data not in self._held:
            self._held[data] = [0, _size(item.frame)]
            self._held_bytes += self._held[data][1]
        self._held[data][0] += 1
        lane['frames'].append(item)
        if not self._paced:
            if not lane['sampled']:
                self._relieve()
            self._fit(lane)
        self._start(lane)

    def _fit(self, lane):
        """Find every lane that no longer fits its bounds and ask :meth:`_overflow`, until all fit."""
        if len(lane['frames']) > lane['capacity']:
            self._overflow(lane, 'frames')
        while self._held_bytes > self.max_backlog_bytes:
            held, _, largest = max((sum(_size(item.frame) for item in other['frames']), index, other)
                                   for index, other in enumerate(self._lanes))
            if not held:
                break                  # only frames being processed are left
            self._overflow(largest, 'bytes')

    def _overflow(self, lane, bound):
        """The one decision about frames that do not fit; ``bound`` is 'frames' (the lane's) or 'bytes' (all lanes').

        It must leave fewer waiting frames in ``lane``. Default: a sampled lane skips its older sample (that is
        its declaration); any other lane declares its oldest tenth not processed (reason ``backlog``).
        """
        self._give_up(lane, 1 if lane['sampled'] else max(1, len(lane['frames']) // 10))

    def _start(self, lane):
        if lane['frames'] and (lane['task'] is None or lane['task'].done()) and not self._closed:
            lane['task'] = asyncio.create_task(self._drain_lane(lane))

    def _release(self, item):
        held = self._held[id(item.frame.data)]
        held[0] -= 1
        if not held[0]:
            self._held_bytes -= held[1]
            del self._held[id(item.frame.data)]

    def _give_up(self, lane, count):
        """The ``count`` oldest waiting frames of a lane: a declared gap, or skipped samples."""
        gone = [lane['frames'].popleft() for _ in range(min(count, len(lane['frames'])))]
        for item in gone:
            self._release(item)
        after = next((item for item in lane['frames'] if item.media), None)
        if after is not None:
            after.gap = True
        else:
            lane['lost'] = True
        names = sorted({name for item in gone for name in item.names})
        if lane['sampled']:
            for item in gone:
                for name in item.names:
                    self.modules[name]['skipped'] += 1
            return
        lane['dropped'] += len(gone)
        for name in names:   # each module with exactly the frames it did not get
            mine = [item.frame for item in gone if name in item.names]
            self._gap(lane['name'], [name], mine[0].timestamp_ms, mine[-1].timestamp_ms, len(mine), 'backlog',
                      subject_id=mine[0].subject_id)

    def _is_pausable(self, name):
        """Declared optional, and nothing that is not optional needs it."""
        if name not in self._pausable:
            self._pausable[name] = all(self.modules[other]['optional'] for other in self._affected([name], sampled=True))
        return self._pausable[name]

    def _needed(self):
        return [lane for lane in self._lanes if not lane['sampled'] and not self._is_pausable(lane['name'])]

    def _relieve(self):
        """Half of a bound is used by a module that is needed: pause the optional ones."""
        needed = self._needed()
        if any(len(lane['frames']) > max(1, lane['capacity'] // 2) for lane in needed) or (
                self._held_bytes * 2 > self.max_backlog_bytes and any(lane['frames'] for lane in needed)):
            for name, entry in self.modules.items():
                if entry['optional'] and entry['enabled'] and not entry['paused'] and self._is_pausable(name):
                    entry['paused'] = entry['needs_reset'] = True
                    entry['pauses'] += 1

    def _resume(self, *, force=False):
        paused = [entry for entry in self.modules.values() if entry['paused']]
        if paused and (force or not any(lane['frames'] for lane in self._needed())):
            for entry in paused:
                entry['paused'] = False
                if entry['lane'] is not None:
                    self._start(entry['lane'])

    async def _drain_lane(self, lane):
        leader, running = self.modules[lane['name']], deque()
        try:
            while not self._closed:
                while lane['frames'] and len(running) < lane['width'] and not all(
                        self.modules[name]['paused'] for name in lane['frames'][0].names):
                    item = lane['frames'].popleft()
                    lane['busy'].append(item)
                    parallel = lane['width'] > 1 and lane['name'] in item.names and leader['enabled']
                    running.append((item, asyncio.create_task(self._process(leader, item.frame)) if parallel else None))
                if not running:
                    return
                item, task = running[0]
                await self._run_item(lane, item, task)
                running.popleft()
                lane['busy'].remove(item)
                self._release(item)
                if not self._recorded:
                    lane['max_backlog_ms'] = max(lane['max_backlog_ms'], _now_ms() - item.frame.timestamp_ms)
                self._resume()
        finally:
            for item, task in running:
                if task is not None:
                    task.cancel()
                lane['busy'].remove(item)
                self._release(item)

    async def _run_item(self, lane, item, task):
        context = _Context(lane)
        for name in item.names:
            entry = self.modules[name]
            if not entry['enabled'] or self.human.paused or self.human.closed:
                self._miss(entry, item.frame)
                continue
            records = await self._run_module(name, entry, item.frame, context, media=item.media, gap=item.gap,
                                             task=task if name == lane['name'] else None)
            if self._collected is not None:
                self._collected += records
        for name, exc in context.failures:   # counted on the module; the lane keeps running
            errors = getattr(self.human, 'plugin_errors', None)
            if isinstance(errors, list):
                errors.append(dict(plugin=f'{self.name}:{name}', error=str(exc), timestamp_ms=item.frame.timestamp_ms))
                del errors[:-100]
            if self._running and not lane['sampled'] and not self._is_pausable(name):
                self._failure = self._failure or exc   # as inline: a run does not go on counting nothing

    # ------------------------------------------------------------------ gaps and completeness

    def _affected(self, names, *, sampled=False):
        """``names`` (inputs or modules) and the modules that depend on every frame of them."""
        found, wanted = [], list(names)
        while wanted:
            name = wanted.pop()
            for other, entry in self.modules.items():
                if other not in found and (other == name or any(
                        src == name or self.inputs.get(src, {}).get('derived_from') == name
                        for src, _ in entry['links'])) and (sampled or not entry['every_ms']):
                    found.append(other)
                    wanted.append(other)
        return found

    @staticmethod
    def _capability(entry):
        return getattr(entry['spec'] or getattr(entry['module'], 'spec', None), 'id', None)

    @staticmethod
    def _miss(entry, frame, reason=None):
        """A frame captured while a module (or capture) was off; declared as one gap when it is on again."""
        off = entry['off']
        if off is None:
            entry['off'] = dict(start_ms=frame.timestamp_ms, end_ms=frame.timestamp_ms, frames=1,
                                subject_id=frame.subject_id,
                                reason=reason or ('paused' if entry['paused'] else 'disabled'))
        else:
            off.update(start_ms=min(off['start_ms'], frame.timestamp_ms),
                       end_ms=max(off['end_ms'], frame.timestamp_ms), frames=off['frames'] + 1)

    def _declare_off(self, name, entry):
        off, entry['off'] = entry['off'], None
        if entry['last_run_ms'] is not None or entry['last_admit_ms'] is not None:   # else it starts here
            self._gap(name, [name], off['start_ms'], off['end_ms'], off['frames'], off['reason'],
                      subject_id=off['subject_id'])

    def _gap(self, place, origins, start_ms, end_ms, frames, reason, *, subject_id='local-user', record=True):
        """``frames`` captured in [start_ms, end_ms] never reached ``origins`` (inputs or modules)."""
        affected = self._affected(origins)
        for name in affected:
            entry = self.modules[name]
            entry['gaps'] += 1
            entry['gap_frames'] += frames
            entry['gap_log'].append(dict(start_ms=start_ms, end_ms=end_ms, frames=frames, reason=reason, source=place))
        if not affected or not record:
            return
        pending = self._gaps.setdefault(place, dict(start_ms=start_ms, end_ms=end_ms, frames=0, gaps=0, reasons=set(),
                                                    modules=set(), capabilities=set(), subject_id=subject_id))
        pending.update(start_ms=min(pending['start_ms'], start_ms), end_ms=max(pending['end_ms'], end_ms),
                       frames=pending['frames'] + frames, gaps=pending['gaps'] + 1)
        pending['reasons'].add(reason)
        pending['modules'].update(affected)
        pending['capabilities'].update(filter(None, (self._capability(self.modules[name]) for name in affected)))

    async def commit_gaps(self, *, force=False):
        """Commit what was lost as ``pipeline.gaps`` records (one per place and second, merged in between)."""
        records = []
        for place in list(self._gaps):
            last = self._gap_at.get(place)
            if place not in self._gaps or not force and last is not None and time.monotonic() - last < GAP_RECORD_S:
                continue
            if self._closed or self.human.paused or self.human.closed or not self.human.consent:
                break
            gap = self._gaps.pop(place)
            self._gap_at[place] = time.monotonic()
            event = dict(event='pipeline.gap', source=place, start_ms=gap['start_ms'], end_ms=gap['end_ms'],
                         lost_frames=gap['frames'], gaps=gap['gaps'], reasons=sorted(gap['reasons']),
                         modules=sorted(gap['modules']), capabilities=sorted(gap['capabilities']))
            self.human.register_capability(GAPS)
            self._gap_sequence += 1
            records.append(await self.human.push_observation(
                GAPS.id, dict(events=[event]), subject_id=gap['subject_id'], timestamp_ms=gap['end_ms'],
                sensor=SensorMetadata(source_id=self.name, sensor_id=place, sequence=self._gap_sequence)))
        return records

    def lost(self, capability=None):
        """``{gaps, frames}`` lost under ``capability`` (default: under any module); gaps 0: nothing was."""
        found = [(entry['gaps'], entry['gap_frames']) for entry in self.modules.values()
                 if capability in (None, self._capability(entry))]
        gaps, frames = max(found, default=(0, 0))
        return dict(gaps=gaps, frames=frames)

    def _mark(self, entry, value, timestamp_ms):
        """Events that span a known gap say so (``gap_frames``): their times and count may miss something."""
        events = value.get('events') if isinstance(value, dict) else None
        if not isinstance(events, list) or not entry['gap_log']:
            return value
        marked = []
        for event in events:
            if isinstance(event, dict):
                start, end = event.get('start_ms', timestamp_ms), event.get('end_ms', timestamp_ms)
                if type(start) in (int, float) and type(end) in (int, float):
                    frames = sum(gap['frames'] for gap in entry['gap_log']
                                 if gap['start_ms'] <= end and gap['end_ms'] >= start)
                    if frames:
                        event = dict(event, gap_frames=frames)
            marked.append(event)
        return dict(value, events=marked)

    def _complete_until(self, name):
        """Capture time up to which everything for ``name`` was processed or declared a gap, or None."""
        if name in self.inputs:
            entry = self.inputs[name]
            return entry['done_ms'] if entry['derived_from'] is None else self._complete_until(entry['derived_from'])
        entry = self.modules[name]
        sources = [self._complete_until(src) for src, _ in entry['links'] if src in self.inputs or src in self.modules]
        if not sources or None in sources:
            return entry['until']
        bounds = list(sources)
        if entry['lane'] is not None:
            bounds += [item.frame.timestamp_ms - 1 for item in (*entry['lane']['busy'], *entry['lane']['frames'])
                       if name in item.names]
        if entry['off'] is not None:
            bounds.append(entry['off']['start_ms'] - 1)
        held = getattr(entry['module'], 'pending_since_ms', None)   # results a module still holds inside
        if held is not None:
            bounds.append(int(held) - 1)
        value = min(bounds)
        entry['until'] = value if entry['until'] is None else max(entry['until'], value)
        return entry['until']

    async def drain(self):
        """Wait until every queued module has processed what it was given."""
        while True:
            tasks = [lane['task'] for lane in self._lanes if lane['task'] is not None and not lane['task'].done()]
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_module(self, name, entry, frame, context, *, media=False, gap=False, task=None):
        """Process and commit one frame; a failure is counted, declared a gap and left in ``context``."""
        if gap and media and frame.modality == 'audio':
            entry['needs_reset'] = entry['gap_out'] = True   # never join audio across a lost interval
        try:
            output = await (self._process(entry, frame) if task is None else task)
            return await self._commit(name, entry, frame, output, context)
        except Exception as exc:
            entry['errors'] += 1
            entry['last_error'] = f'{type(exc).__name__}: {exc}'[:300]
            entry['needs_reset'] = True
            self._gap(name, [name], frame.timestamp_ms, frame.timestamp_ms, 1, 'error', subject_id=frame.subject_id)
            context.failures.append((name, exc))
            return []

    async def _process(self, entry, frame):
        module = entry['module']
        if entry['needs_reset']:
            entry['needs_reset'] = False
            if callable(getattr(module, 'reset', None)):
                await _call(module.reset)
        started = time.monotonic()
        output = await _call(module.process, frame)
        entry['last_run_ms'] = max(frame.timestamp_ms, entry['last_run_ms'] or 0)
        entry['processed'] += 1
        entry['last_latency_ms'] = round((time.monotonic() - started) * 1000, 3)
        entry['latency'].append(entry['last_latency_ms'])
        return output

    async def _commit(self, name, entry, frame, output, context):
        if output is None:
            return []
        records, derived = [], []
        if entry['spec'] is not None:
            outputs = [output] if not isinstance(output, list) else output
            for item in outputs:
                if isinstance(item, Derived):
                    derived.append(item)
                    continue
                sensor = frame.sensor or SensorMetadata(source_id=self.name, sensor_id=name, sequence=0)
                records.append(await self.human.push_observation(
                    entry['spec'].id, self._mark(entry, item, frame.timestamp_ms), sensor=sensor, subject_id=frame.subject_id,
                    timestamp_ms=frame.timestamp_ms))
        else:
            signals = []
            for item in output:
                if isinstance(item, Derived):
                    derived.append(item)
                else:
                    signals.append(dict(item, value=self._mark(entry, item['value'], frame.timestamp_ms))
                                   if 'value' in item else item)
            if signals:
                records += [record for record in
                            await self._ingest(signals, frame, entry['module'], name) if record]
        newest = max((record.get('timestamp_ms', frame.timestamp_ms) for record in records),
                     default=frame.timestamp_ms if derived else None)
        if newest is not None:
            entry['newest_ms'] = max(newest, entry['newest_ms'] or newest)
            if not self._recorded:
                entry['lag'].append(max(0, _now_ms() - newest))
        for record in list(records):
            observation = record.get('observation') or {}
            # A record keeps its own capture time: a module running in parallel (its own process)
            # commits earlier frames' records while later frames are already flowing.
            link = self._frame(name, observation.get('value'), observation.get('modality') or frame.modality,
                               record.get('timestamp_ms', frame.timestamp_ms), subject_id=frame.subject_id,
                               sensor=frame.sensor if record.get('timestamp_ms') == frame.timestamp_ms else None)
            records += await self.feed(name, link, output=observation.get('name'), context=context)
        for item in derived:
            if item.input not in entry['emits']:
                raise ValueError(f'{name} emitted {item.input!r} but declares emits={entry["emits"]}')
            derived_frame = self._frame(item.input, item.data, item.modality,
                                        frame.timestamp_ms if item.timestamp_ms is None else item.timestamp_ms,
                                        subject_id=frame.subject_id, sample_rate=item.sample_rate,
                                        sensor=frame.sensor)
            gap = entry['gap_out'] and item.modality == 'audio'
            if gap:
                entry['gap_out'] = False
            records += await self.feed(item.input, derived_frame, gap=gap, context=context)
        return records

    async def _ingest(self, signals, frame, module, name):
        """Commit plugin signals; returns the committed records (same path as registered plugins)."""
        records = []
        for signal in signals:
            signal = dict(signal)
            signal.setdefault('subject_id', frame.subject_id)
            signal.setdefault('timestamp_ms', frame.timestamp_ms)
            signal.setdefault('modality', frame.modality if frame.modality in ('video', 'audio') else 'text')
            signal.setdefault('model_id', getattr(module, 'name', name))
            signal.setdefault('model_version', getattr(module, 'version', 'unspecified'))
            is_text = signal.pop('type', None) == 'text' or signal.get('name') in ('text.transcript',
                                                                                   'speech.transcript')
            records.append(await self.human.push_signal(**signal))
            if is_text:
                await self.human.push_text(str(signal['value']), subject_id=signal['subject_id'],
                                           timestamp_ms=signal['timestamp_ms'])
        return records

    async def run(self, *, stop=None, on_frame=None):
        """Pump every source-backed input until all end (or ``stop`` is set); then finish modules.

        ``on_frame(input_id, frame, records)`` is the application's hook (a view, a report); returning
        False stops the run. ``records`` are those committed since the last call: modules in lanes answer
        while later frames are already captured, so a record's time is its own, not the frame's.
        A recorded input (``live = False``) waits for the lanes, frame by frame; a live one never waits.
        """
        stop = stop or asyncio.Event()
        sources = [entry['source'] for entry in self.inputs.values() if entry['source'] is not None]
        self._paced = self._recorded = bool(sources) and not any(getattr(source, 'live', True) for source in sources)
        self._collected = None if on_frame is None else []
        self._running, self._failure = True, None

        def check():   # a module the run needs failed in its lane
            if self._failure is not None:
                raise self._failure

        async def pump(input_id, source):
            entry, missed = self.inputs[input_id], getattr(source, 'missed', 0)
            async for frame in source.frames():
                if stop.is_set():
                    break
                lost = getattr(source, 'missed', 0) - missed
                missed += lost
                entry['missed'] += lost
                if lost:
                    entry['gaps'] += 1
                    last = entry['last_ms']
                    first = frame.timestamp_ms - 1 if last is None else min(last + 1, frame.timestamp_ms - 1)
                    self._gap(input_id, [input_id], first, frame.timestamp_ms - 1, lost, 'input',
                              subject_id=frame.subject_id)
                frame = self._frame(input_id, frame.data, frame.modality, frame.timestamp_ms,
                                    subject_id=frame.subject_id, sample_rate=frame.sample_rate,
                                    sensor=frame.sensor)
                records = await self.feed(input_id, frame, gap=bool(lost))
                if self._paced:
                    await self.drain()
                check()
                if self._collected:
                    records, self._collected = self._collected + records, []
                if on_frame is not None and await _call(on_frame, input_id, frame, records) is False:
                    stop.set()
                await asyncio.sleep(0)

        tasks = [asyncio.create_task(pump(input_id, entry['source']))
                 for input_id, entry in self.inputs.items() if entry['source'] is not None]
        try:
            if tasks:
                await asyncio.gather(*tasks)
            self._paced = True          # nothing arrives any more: what waits is worked off, not given up
            self._resume(force=True)
            await self.drain()
            check()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._running = self._paced = False
        records = await self.finish()
        collected, self._collected = self._collected or [], None
        return collected + records

    async def finish(self):
        """End of input: commit what stateful modules still hold (e.g. a count at the last frame)."""
        paced, self._paced = self._paced, True
        records = []
        try:
            self._resume(force=True)
            for name, entry in self.modules.items():
                await self.drain()     # what the modules before it still gave it
                finish = getattr(entry['module'], 'finish', None)
                if not entry['enabled'] or not callable(finish):
                    continue
                output = await _call(finish)
                last = entry['last_run_ms']
                if output is None or last is None or (entry['spec'] is None and not isinstance(output, list)):
                    continue  # a plugin that commits its own records returns its final value only
                context = _Context()
                records += await self._commit(name, entry, self._frame(name, None, 'record', last), output, context)
                if context.failures:
                    raise context.failures[0][1]
            await self.drain()
            records += await self.commit_gaps(force=True)
        finally:
            self._paced = paced
        return records

    async def close(self):
        self._closed = True
        lanes = [lane['task'] for lane in self._lanes if lane['task'] is not None]
        for task in lanes:
            task.cancel()
        await asyncio.gather(*lanes, return_exceptions=True)
        for entry in self.inputs.values():
            close = getattr(entry['source'], 'close', None)
            if callable(close):
                await _call(close)
        for entry in self.modules.values():
            close = getattr(entry['module'], 'close', None)
            if callable(close):
                await _call(close)
        if getattr(self.human, 'pipeline', None) is self:
            self.human.pipeline = None

    def _backlog(self, lane):
        if lane is None:
            return {}
        waiting = [*lane['busy'], *lane['frames']]
        age = None if self._recorded else max((_now_ms() - item.frame.timestamp_ms for item in waiting), default=0)
        return dict(lane=lane['name'], queue=lane['capacity'], queued=len(lane['frames']), backlog=len(waiting),
                    backlog_ms=age, max_backlog_ms=None if self._recorded else lane['max_backlog_ms'],
                    dropped=lane['dropped'])

    def status(self):
        """What is connected, on, behind, lost and complete; times are capture times in ms.

        Ages (``backlog_ms``, ``newest_result_age_ms``, ``capture_to_result_ms``) exist for live input only.
        ``complete_until_ms``: everything captured up to then was processed by the module or declared in
        ``gap_intervals``; a module that answers about a window may still add results for that window.
        """
        now = _now_ms()
        return dict(
            name=self.name, complete=not any(value['gaps'] for value in self.modules.values()),
            backlog_bytes=self._held_bytes, max_backlog_bytes=self.max_backlog_bytes,
            inputs={key: dict(modality=value['modality'], frames=value['frames'], last_ms=value['last_ms'],
                              derived_from=value['derived_from'], host_fed=value['source'] is None,
                              missed=value['missed'], gaps=value['gaps'],
                              **({'state': value['source'].status()}
                                 if callable(getattr(value['source'], 'status', None)) else {}))
                    for key, value in self.inputs.items()},
            references={key: dict(kind=value['kind'], label=value['label']) for key, value in self.references.items()},
            modules={key: dict(enabled=value['enabled'], consumes=value['consumes'], emits=list(value['emits']),
                               capability=value['spec'].id if value['spec'] is not None else None,
                               every_ms=value['every_ms'], processed=value['processed'], errors=value['errors'],
                               last_error=value['last_error'], last_latency_ms=value['last_latency_ms'],
                               latency_ms=_percentiles(value['latency']), capture_to_result_ms=_percentiles(value['lag']),
                               newest_result_age_ms=None if value['newest_ms'] is None or self._recorded
                               else max(0, now - value['newest_ms']),
                               gaps=value['gaps'], gap_frames=value['gap_frames'], complete=not value['gaps'],
                               gap_intervals=list(value['gap_log'])[-64:],
                               complete_until_ms=self._complete_until(key),
                               replicas=int(getattr(value['module'], 'replicas', 1)),
                               optional=value['optional'], paused=value['paused'], pauses=value['pauses'],
                               **({'sampled': dict(every_ms=value['every_ms'], skipped=value['skipped'])}
                                  if value['every_ms'] else {}),
                               **self._backlog(value['lane']),
                               **({'state': value['module'].status()}
                                  if callable(getattr(value['module'], 'status', None)) else {}))
                     for key, value in self.modules.items()})


class PipelineIntake:
    """Engine plugin that feeds pushed media into one host-fed Pipeline input.

    Registered with ``human.register``, it lets every intake path (``push_video`` /
    ``push_audio``, MCP ``start_capture``, the dashboard) reach the Pipeline. The Pipeline
    commits the records, so the plugin itself returns no signals; an engine reset (pause,
    stop, delete) resets the Pipeline.
    """

    version = '1'

    def __init__(self, pipeline, input_id, modality):
        if input_id not in pipeline.inputs or pipeline.inputs[input_id]['source'] is not None:
            raise ValueError(f'{input_id!r} must be a host-fed input of the pipeline')
        self.pipeline, self.input_id = pipeline, input_id
        self.name = f'pipeline:{pipeline.name}:{input_id}'
        self.modalities = (modality,)

    async def process(self, frame):
        await self.pipeline.push(self.input_id, frame.data, timestamp_ms=frame.timestamp_ms, modality=frame.modality,
                                 subject_id=frame.subject_id, sample_rate=frame.sample_rate)
        return []

    async def reset(self):
        await self.pipeline.reset()
