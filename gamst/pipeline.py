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
until the backlog is clear; when it is exceeded anyway, video is thinned: single frames *between* waiting
ones are given up, every second one before any two in a row, so no whole interval goes unseen. Audio
cannot be thinned; its oldest chunks are given up together. Either is a declared *gap*: counted, committed as a ``pipeline.gap`` event, and marked on the events and counts
downstream. What happens to a frame that does not fit is one decision (:meth:`Pipeline._overflow`). Each
gap says why: ``input``, ``input_failed`` and ``capture_paused`` are data that never reached GAMST; ``error`` is a
frame a module failed on; ``thinned``, ``backlog``, ``paused`` and ``disabled`` are captured frames a
module did not process (``thinned``: frames between these two times were processed, ``frames`` of them were not).
``complete_until_ms`` of a module only advances over capture time that was processed or declared so.
Audio is never joined across a gap.

Catch-up (live audio). A module in a lane of its own that can ``fork()`` does not give audio up: when its lane is more
than ``lag_ms`` behind, it skips to now and the skipped capture range becomes a TODO; a pool of catch-up workers (forks
of the module, given its ``snapshot()`` from the TODO's start, hearing from its last pause) works the TODOs off from the
input's kept audio (:meth:`keep_audio`, memory with a spill file) and commits their results at their capture times. A
pending TODO holds ``complete_until_ms`` back, so what reads the module (a verification lane) waits for it
(:meth:`_skip`, docs/catch-up-verification.md).

The one exception is declared sampling: a module with ``every_ms`` looks at one frame per interval; the
others are reported as ``sampled``, and are no gap.

A module that computes without a process and reads a module in a lane joins that lane: one frame runs
the whole chain (pose -> person -> channels -> count) in capture order, also where such a module reads
the input itself. A module with ``replicas`` (stateless workers of one spec) works on several frames of
its lane at once; results are still committed in capture order. Recorded inputs (``live = False``) wait
for the lanes, so nothing builds up and nothing is given up; several recorded inputs are read in capture order
(one frame of each is held), and live and recorded inputs cannot run together, having no common clock. An
input's ``offset_ms`` is the owner's measured correction, added to its capture times.
``PipelineIntake`` feeds a host-fed input from the engine's own intake (``push_video`` /
``push_audio``, MCP ``start_capture``, the dashboard), which is how ``ProfileRuntime`` runs the
roles chosen by setup as one Pipeline.

Failures. In a run, one input or module that fails does not end the others: what it did not deliver is a declared
gap, ``status()['failures']`` names it, and the run is incomplete. A live input is opened again with growing waits
(``REOPEN_S``, bounded) and says when it resumed; a module that fails ``FAILURES`` times in a row is given up until it is
switched on again; the run ends when nothing is left. ``run(stop_on_failure=True)`` ends it at the first failure.

User-defined reference inputs (a voice sample, photos, a demonstrated pose, a list of words) are
given with ``add_reference``. Words (``terms``) and movements (``exemplar``) may be replaced or removed while a run goes
on (:meth:`Pipeline.set_reference`, :meth:`Pipeline.remove_reference`): from a stated capture time on, with a revision
that every event and count carries. References to a person are set by the owner before the run only.
Modules read references in ``bind(pipeline)``; the pipeline keeps only what
the module returns from ``bind`` (an embedding, a feature template), never the raw sample, unless
the module keeps it itself.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field, replace
import inspect
import time
import hashlib
import json
from typing import Any

from .capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from .plugins import InputFrame
from .redact import public_error, public_text

CATCH_UP = dict(lag_ms=2000, max_lookback_ms=10000, tail_ms=1500, workers=2, speed=None, min_free_pct=None,
                replica_mb=None, post_chunk_ms=None)
# Video (L2 for video, see Pipeline._skip_video): ``lag_ms`` is the deadline of a waiting frame, ``max_lookback_ms`` the
# pre-roll (frames before the cut that only build state), ``tail_ms`` how far past the newest waiting frame the worker
# owns (the live lane's own warm-up after it resumes), ``workers`` the L2 chains; ``min_free_pct`` / ``replica_mb``: a
# chain is started only while free system memory stays above that share after taking ``replica_mb`` more.
# ``post_chunk_ms``: a recording that is not paced (timing post) is also cut, into chunks of this length, whenever an L2
# chain is free: chunks run in parallel with the lane (their events are not corrections: nothing is late in post).

GAPS = CapabilitySpec('pipeline.gaps', 'pipeline.gaps', 'events', 'gamst.pipeline', '1',
                      (OutputField('events', 'array'),), 'frame', interpretation='measurement', license='Apache-2.0')
INSTRUCTIONS = CapabilitySpec('pipeline.instructions', 'pipeline.instructions', 'events', 'gamst.pipeline', '1',
                              (OutputField('events', 'array'),), 'frame', interpretation='measurement', license='Apache-2.0')
INSTRUCTABLE = ('terms', 'exemplar')     # reference kinds that may change while a run goes on; the others identify a person
BACKLOG = 900                            # frames a lane holds (30 s of 30 fps video) ...
BACKLOG_BYTES = 512 * 1024 * 1024        # ... and what all lanes of a Pipeline hold together
SPILL_FRAMES = 1_000_000                 # frames a ``backlog="spill"`` lane may hold (bounded by its video memory)
SAMPLES = ('newest',)                    # ``sample``: how a lane that is busy takes frames (declared, not lost)
BACKLOGS = ('spill',)                    # ``backlog``: how a lane keeps the frames it has not processed
SAME_EVENT_MS = 100                      # two lanes' events this close (same label, person) are one (L2 safety net)
MAX_OFFSET_MS = 600_000                  # an input's declared clock correction
LOG = 512                                # gap intervals and timings kept per module
GAP_RECORD_S = 1.                        # at most one gap record per place and second; merged in between
FAILED_RECORD_S = 30.                    # ... and per half minute for a module that was given up
FAILURES = 25                            # failures in a row after which a run gives a module up
REOPEN_S = (.5, 1., 2., 4., 8., 16.)     # waits before a failed live input is opened again; then it is given up
REOPENED_S = 10.                         # an input that delivered this long after that starts over with the waits
SLOW_FRAMES = 30                         # frames measured before a lane is called slower than its input


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
    thinned: int = 0     # how often the frame next to it was given up (2 ** thinned: it stands for that many)
    source: str | None = None    # the input (or module) the frame is of
    kept: Any = None     # in the video memory (a ``backlog="spill"`` lane): the frame's data is there until it runs


@dataclass
class _Context:
    """Where a module runs: the lane whose task this is (None: on the input), and what failed under it."""
    lane: dict | None = None
    failures: list = field(default_factory=list)
    forks: dict | None = None     # an L2 chain: its modules (forks) by name; frames go only to them
    keep: Any = None              # ... and what of their results is theirs: keep(start_ms, end_ms)
    correction: bool = False      # ... whose events are corrections (counted late)


def _size(frame):
    data = frame.data
    return len(data) if isinstance(data, (bytes, bytearray, memoryview)) else int(getattr(data, 'nbytes', 0) or 0)


def _percentiles(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(p50=ordered[len(ordered) // 2], p95=ordered[min(len(ordered) - 1, int(.95 * len(ordered)))],
                samples=len(ordered))


def reference_id(value):
    """Which words or which movement this is: the file's own ID, else a hash of the content."""
    if isinstance(value, dict) and isinstance(value.get('id'), str):
        return value['id'][:64]
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _now_ms():
    return time.time_ns() // 1_000_000


async def _call(function, *args):
    value = function(*args)
    return await value if inspect.isawaitable(value) else value


def _with_events(item, events):
    """A record value (or a signal holding one) with only these events."""
    if 'value' in item and isinstance(item['value'], dict) and 'events' in item['value']:
        return dict(item, value=dict(item['value'], events=events))
    return dict(item, events=events)


def _corrected(item):
    """An L2 result: its events are corrections, decided late (``correction``, ``decided_by``)."""
    if isinstance(item, Derived) or not isinstance(item, dict):
        return item
    value = item['value'] if isinstance(item.get('value'), dict) else item
    events = value.get('events')
    if not isinstance(events, list) or not events:
        return item
    marked = [dict(event, correction=True, decided_by='catch_up') if isinstance(event, dict) else event
              for event in events]
    return _with_events(item, marked)


def memory_allows(more_mb, min_free_pct):
    """Free system memory stays at or above ``min_free_pct`` of all after ``more_mb`` more (unknown: no)."""
    from .memory_budget import free_and_total
    free, total = free_and_total()
    return free is not None and total and (free - more_mb * 1024 * 1024) / total * 100 >= min_free_pct


async def _nth(called, index):
    """The output of one frame of a batched call."""
    return (await called)[index]


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
        self._stop, self._stop_on_failure = None, False
        self.reopen_s = REOPEN_S
        self.notify = None                   # the application's hook: what failed or came back, as a line of text
        self.notes = {}                      # what a run decided before it started (e.g. which pose model, measured)
        self.mode, self.search, self.mode_changes = 'count', None, []    # counting mode (gamst.counting, set_mode)
        self._gaps: dict[str, dict] = {}     # not yet committed, by place
        self._gap_at: dict[str, float] = {}
        self._gap_sequence = 0
        self._pausable: dict[str, bool] = {}
        self.memories: dict[str, Any] = {}   # input id -> AudioMemory: audio kept for catch-up and verification
        self._video_memory = None            # VideoMemory of the ``backlog="spill"`` lanes (created by the first one)
        self._keeping_up = False             # a recording replayed at its pace: sampling and spilling lanes are not waited for
        human.pipeline = self

    # ------------------------------------------------------------------ composition

    def add_input(self, input_id, source=None, *, modality=None, derived_from=None, offset_ms=0):
        """``source`` has ``frames()`` (an async iterator of InputFrame); None means the host feeds it.

        ``offset_ms`` is added to its capture times: how much later than stamped this input really happened
        (negative for a camera whose frames arrive late). Set by the owner from a measurement, never estimated here.
        """
        identifier(input_id, 'input id')
        if input_id in self.inputs or input_id in self.modules:
            raise ValueError(f'{input_id!r} is already an input or module')
        if source is not None and not callable(getattr(source, 'frames', None)):
            raise TypeError('an input source needs frames()')
        if type(offset_ms) is not int or abs(offset_ms) > MAX_OFFSET_MS:
            raise ValueError(f'offset_ms must be an integer of at most {MAX_OFFSET_MS} ms')
        self.inputs[input_id] = dict(source=source, modality=modality or getattr(source, 'modality', None),
                                     derived_from=derived_from, frames=0, first_ms=None, last_ms=None, done_ms=None, missed=0, gaps=0, off=None,
                                     offset_ms=offset_ms, ended=False, error=None, down=None, retries=0, opened=None,
                                     outages=deque(maxlen=16), intervals=deque(maxlen=64), size=None)
        return self

    def add_reference(self, name, kind, value, *, label=None, consent=None):
        """A user-defined reference input; modules turn it into what they keep in ``bind``."""
        identifier(name, 'reference name')
        if kind not in ('voice', 'person', 'face', 'body', 'exemplar', 'terms', 'object'):
            raise ValueError('reference kind must be voice, person, face, body, exemplar, terms or object')
        if kind in ('voice', 'person', 'face', 'body') and not consent:
            raise ValueError('a reference to a person needs a consent record')
        self.references[name] = dict(kind=kind, value=value, label=label or name, consent=consent)
        if kind in INSTRUCTABLE:
            self.references[name].update(id=reference_id(value), revision=1, from_ms=0, set_by='owner', source='start',
                                         set_at_ms=_now_ms())
        return self

    # ------------------------------------------------------------------ instructions while a run goes on

    def change_ms(self, inputs):
        """The first capture time of these inputs that has not been taken in yet: where a change takes effect."""
        return 1 + max((self.inputs[key]['last_ms'] for key in inputs if self.inputs[key]['last_ms'] is not None), default=-1)

    def _reading(self, name):
        return {key: entry for key, entry in self.modules.items()
                if getattr(entry['module'], 'reference_name', None) == name and callable(getattr(entry['module'], 'schedule', None))}

    async def set_reference(self, name, kind, value, *, label=None, set_by='user', source='value', add=()):
        """Give the modules that read ``name`` other words or another movement, without restarting anything.

        Frames captured before ``effective_from_ms`` (the answer's) are still measured by the old one, later ones by the
        new one only: nothing before it was looked for. ``add``: ``(name, module, add_module options)`` of modules to
        connect with it. Raises before anything is changed.
        """
        identifier(name, 'reference name')
        if kind not in INSTRUCTABLE:
            raise PermissionError(f'a {kind} reference identifies a person: the owner sets it, with consent, before the run')
        before = self.references.get(name)
        if before is not None and before['kind'] != kind:
            raise ValueError(f"{name!r} is a {before['kind']} reference")
        readers, added = self._reading(name), [item[0] for item in add]
        if bool(readers) == bool(add):
            raise ValueError(f'{name!r} is read already; nothing to add' if add else f'no module of this run reads a reference {name!r}')
        self._instructable()
        reference = dict(kind=kind, value=value, label=label or name, consent=None, id=reference_id(value),
                         revision=before['revision'] + 1 if before is not None else 1, from_ms=0, set_by=set_by, source=source,
                         set_at_ms=_now_ms())
        ready = {key: entry['module'].prepare(reference) for key, entry in readers.items()}    # refuses what it cannot use
        for module in (item[1] for item in add):
            if getattr(module, 'reference_name', None) == name and callable(getattr(module, 'prepare', None)):
                module.prepare(reference)
        roots = {root for key in readers for root in self._roots(key)}
        for source in (source for _, _, options in add for source in options['consumes']):
            owner = self.inputs[source]['derived_from'] if source in self.inputs else source.split(':', 1)[0]
            roots |= {source} if owner is None else self._roots(owner)
        # Several changes before a new frame: the last one takes the place of the others.
        reference['from_ms'] = self.change_ms(roots)
        for key, entry in readers.items():
            entry['module'].schedule(ready[key], reference['from_ms'])
            self._asked(entry, reference)
        self.references[name] = reference
        try:
            for key, module, options in add:
                self.add_module(key, module, **options)
                self.modules[key]['from_ms'] = self.change_ms(self._roots(key))     # of its own inputs
        except BaseException:
            for key in added:
                self._discard(key)
            if before is None:
                del self.references[name]
            else:
                self.references[name] = before
            raise
        return await self._instructed('instruction.set', name, reference, before)

    async def remove_reference(self, name, *, set_by='user'):
        """Stop looking for what ``name`` describes; what was counted stays, under its own revision."""
        before = self.references.get(name)
        if before is None:
            raise ValueError(f'no reference {name!r} is set')
        if before['kind'] not in INSTRUCTABLE:
            raise PermissionError(f"a {before['kind']} reference is the owner's; it is not removed while the run goes on")
        if before.get('removed'):
            raise ValueError(f'{name!r} is removed already')
        readers = self._reading(name)
        self._instructable()
        reference = dict(before, removed=True, value=None, revision=before['revision'] + 1, set_by=set_by, source='removed',
                         set_at_ms=_now_ms(), from_ms=self.change_ms({root for key in readers for root in self._roots(key)}))
        for entry in readers.values():
            entry['module'].schedule(None, reference['from_ms'])
            self._asked(entry, reference)
        self.references[name] = reference
        return await self._instructed('instruction.removed', name, reference, before)

    def _instructable(self):
        if self._closed or self.human.closed:
            raise RuntimeError('the run is over')
        if self.human.paused or not self.human.consent:
            raise PermissionError('capture is paused; resume it first')

    @staticmethod
    def _asked(entry, reference):
        """The capture intervals in which this module was asked for something (``[from, until)``; older ones are merged)."""
        asked, at = entry['asked'], reference['from_ms']
        if asked and asked[-1][1] is None:
            asked[-1][1] = at
        if not reference.get('removed'):
            asked.append([at, None])
        asked[:] = [item for item in asked if item[1] is None or item[1] > item[0]]
        if len(asked) > 16:
            asked[:2] = [[asked[0][0], asked[1][1]]]

    def _discard(self, name):
        entry = self.modules.pop(name, None)
        if entry is not None:
            self._lanes = [lane for lane in self._lanes if lane['name'] != name]
            self._pausable.clear()
            for derived in entry['emits']:
                self.inputs.pop(derived, None)

    async def _instructed(self, event, name, reference, before):
        capabilities = sorted(filter(None, (self._capability(entry) for entry in self._reading(name).values())))
        change = dict(event=event, reference=name, kind=reference['kind'], label=str(reference['label'])[:100],
                      reference_id=reference['id'], reference_revision=reference['revision'], start_ms=reference['from_ms'],
                      end_ms=reference['from_ms'], set_by=reference['set_by'], source=reference['source'], capabilities=capabilities)
        if before is not None:
            change.update(previous_id=before['id'], previous_revision=before['revision'])
        self.human.register_capability(INSTRUCTIONS)
        self._gap_sequence += 1
        record = await self.human.push_observation(
            INSTRUCTIONS.id, dict(events=[change]), timestamp_ms=reference['from_ms'] or _now_ms(),
            sensor=SensorMetadata(source_id=self.name, sensor_id='instructions', sequence=self._gap_sequence))
        return dict(change, record_id=record['id'], effective_from_ms=reference['from_ms'])

    def add_module(self, name, module, *, consumes, spec=None, enabled=True, every_ms=None, emits=(), queue=None,
                   optional=False, catch_up=None, slowdown=None, sample=None, backlog=None, batch=None):
        """``queue``: None (inline, or in the lane of the module it reads), a lane of N frames, or 'auto'.

        ``every_ms`` declares sampling (else every frame is processed). ``optional`` allows pausing the
        module while others are behind. A module with ``replicas`` > 1 gets a lane.

        ``catch_up``: how a lane that falls behind its live audio catches up (:meth:`_skip`); None: the defaults
        (:data:`CATCH_UP`) for a module in its own lane that can ``fork``, False: never (audio is given up as a
        ``backlog`` gap instead), or a dict of :data:`CATCH_UP` keys. ``slowdown`` (a test and debug option):
        ``{"ms": N, "spans_s": [[from, to], ...]}`` sleeps N ms before each frame captured in those seconds of the run,
        so a live lane falls behind on purpose.

        ``sample="newest"``: the module gets a lane of its own that, while it is busy, keeps only the newest frame;
        the frames it does not take are a declared choice (``status()['modules'][name]['sampled']['skipped']``), not
        lost, and the frames it takes are spread evenly over time. Its modules see real capture times, so rates stay
        right at the lower frame rate. ``backlog="spill"``: a lane of its own that never gives a frame up: waiting
        video frames are kept in the Pipeline's :class:`gamst.providers.video_memory.VideoMemory` (memory, then a
        spill file), so a slower module behind a live input (a verification lane) still processes every frame. A
        recording replayed at its own pace (``--realtime``) does not wait for either kind of lane.

        ``batch=N`` (a module with ``process_batch(frames)``, which answers one output per frame, in order): its lane
        hands it up to N frames at once, the frames that are waiting when it is free (it never waits for more, so a
        lane that keeps up still gets one frame at a time). Every frame is processed; only how many go per call
        changes. Its ``latency_ms`` is then per frame (a call's time divided by its frames).
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
        if sample not in (None, *SAMPLES) or backlog not in (None, *BACKLOGS):
            raise ValueError(f'sample is None or one of {SAMPLES}; backlog is None or one of {BACKLOGS}')
        if sample and backlog:
            raise ValueError('a lane either samples the newest frame or keeps every frame, not both')
        if batch is not None and (type(batch) is not int or not 1 <= batch <= 64):
            raise ValueError('batch is None or 1..64 frames')
        batch = batch if batch and batch > 1 and callable(getattr(module, 'process_batch', None)) else None
        width = int(getattr(module, 'replicas', 1))
        if width > 1 and (spec is None or spec.stateful):
            raise ValueError(f'{name!r} cannot run in replicas: only a module whose spec is stateless may')
        links = [tuple(item.split(':', 1)) if ':' in item and item not in self.inputs else (item, None)
                 for item in consumes]
        if spec is not None:
            self.human.register_capability(spec)
        if name in emits:
            raise ValueError(f'{name!r} cannot emit an input of its own name: name the derived input otherwise')
        for derived in emits:
            self.add_input(derived, derived_from=name)
        bind = getattr(module, 'bind', None)
        if callable(bind):
            bind(self)
        reference = self.references.get(getattr(module, 'reference_name', None), {})
        if queue == 'auto' or queue is None and width > 1:
            queue = None if width == 1 and not getattr(module, 'owns_process', False) else 1 if every_ms else BACKLOG
        if sample or backlog:
            queue = 1 if sample else SPILL_FRAMES
        if batch and queue is None:
            queue = BACKLOG
        if queue is not None:
            lane = dict(name=name, sampled=bool(every_ms) or bool(sample), frames=deque(), busy=[], capacity=queue,
                        task=None, dropped=0, lost=False, width=width, max_backlog_ms=0, sample=sample, backlog=backlog,
                        batch=batch, batches=0, batched=0, largest=0)
            self._lanes.append(lane)
        else:
            lane = self._upstream_lane(links)
        catch_up = self._catch_up_config(name, module, catch_up, lane, width, every_ms)
        if slowdown is not None and (not isinstance(slowdown, dict) or type(slowdown.get('ms')) not in (int, float)
                                     or not all(len(span) == 2 for span in slowdown.get('spans_s') or ())):
            raise ValueError('slowdown is {"ms": N, "spans_s": [[from_s, to_s], ...]}')
        self._pausable.clear()
        self.modules[name] = dict(module=module, consumes=consumes, links=links, spec=spec, enabled=bool(enabled),
                                  every_ms=every_ms, last_run_ms=None, processed=0, errors=0,
                                  last_error=None, last_latency_ms=None, needs_reset=False, emits=tuple(emits),
                                  asked=[[reference['from_ms'], None]] if 'from_ms' in reference else [],
                                  last_admit_ms=None, lane=lane, optional=bool(optional), paused=False, pauses=0,
                                  off=None, failing=0, failed=None, skipped=0, gaps=0, gap_frames=0, gap_out=False, gap_log=deque(maxlen=LOG),
                                  from_ms=1 + max((value['last_ms'] for value in self.inputs.values()
                                                   if value['last_ms'] is not None), default=-1),
                                  newest_ms=None, until=None, latency=deque(maxlen=LOG), lag=deque(maxlen=LOG),
                                  busy_ms=0., catch_up=catch_up, todos=deque(), claims=deque(maxlen=64), workers=[],
                                  catch=dict(skips=0, skipped_ms=0, done=0, failed=0, max_pending=0, max_pending_ms=0,
                                             heard_ms=0, busy_ms=0., wait_ms=deque(maxlen=LOG), todos=deque(maxlen=64)),
                                  slowdown=slowdown)
        if catch_up:
            for source, _ in links:
                if source in self.inputs and self.inputs[source]['modality'] == 'audio':
                    self.keep_audio(source)
        return self

    # ------------------------------------------------------------------ catch-up: a live lane that fell behind

    @staticmethod
    def _catch_up_config(name, module, catch_up, lane, width, every_ms):
        if catch_up is False:
            return None
        forks = callable(getattr(module, 'fork', None)) and getattr(module, 'can_fork', True)
        if catch_up is None and (not forks or tuple(getattr(module, 'modalities', ())) == ('video',)):
            return None     # video: only where a recipe asks for it (it keeps frames for its pre-roll)
        if not forks:
            raise ValueError(f'{name!r} cannot catch up: its module has no fork()')
        if lane is None or lane['name'] != name or width != 1 or every_ms:
            if catch_up is None:
                return None
            raise ValueError(f'{name!r} catches up only in a lane of its own, one frame at a time, unsampled')
        config = dict(CATCH_UP, **(catch_up if isinstance(catch_up, dict) else {}))
        unknown = set(config) - set(CATCH_UP)
        if unknown:
            raise ValueError(f'unknown catch_up settings {sorted(unknown)}; known: {sorted(CATCH_UP)}')
        if not 0 < config['lag_ms'] <= 600000 or not 0 <= config['max_lookback_ms'] <= 120000 \
                or not 0 <= config['tail_ms'] <= 30000 or not 1 <= int(config['workers']) <= 16:
            raise ValueError('catch_up needs 0 < lag_ms <= 600000, 0 <= max_lookback_ms <= 120000, '
                             '0 <= tail_ms <= 30000 and 1 <= workers <= 16')
        if config['speed'] is not None and not float(config['speed']) > 0:
            raise ValueError('catch_up speed must be positive (None: as fast as the module goes)')
        if config['post_chunk_ms'] is not None and not 100 <= float(config['post_chunk_ms']) <= 600000:
            raise ValueError('catch_up post_chunk_ms is 100..600000 (None: a recording is not cut)')
        if config['min_free_pct'] is not None and not 0 <= float(config['min_free_pct']) < 100 or \
                config['replica_mb'] is not None and not 0 <= float(config['replica_mb']) <= 1 << 20:
            raise ValueError('catch_up min_free_pct is 0..100 and replica_mb 0..1048576 (None: not checked)')
        return config

    def keep_audio(self, input_id, **options):
        """Keep every audio frame of ``input_id`` until nobody holds it (:class:`gamst.providers.audio_memory.AudioMemory`)."""
        if input_id not in self.inputs:
            raise KeyError(f'unknown input {input_id!r}')
        if input_id not in self.memories:
            from .providers.audio_memory import AudioMemory
            self.memories[input_id] = AudioMemory(**options)
        return self.memories[input_id]

    def audio_memory(self, input_id):
        """The kept audio of ``input_id``, or None (nothing asked for it to be kept)."""
        return self.memories.get(input_id)

    def _lagging(self, lane):
        """The leader of a live lane fell behind its audio by more than its ``lag_ms``: its waiting audio is skipped."""
        entry = self.modules[lane['name']]
        config = entry['catch_up']
        if not config or not lane['frames'] or not entry['enabled'] or entry['failed']:
            return False
        head = next((item for item in lane['frames'] if lane['name'] in item.names), None)
        if head is None or not head.media:
            return False
        if head.frame.modality == 'video':   # L2 for video: by the input's newest frame (a paced replay has its own clock)
            if lane.get('backlog') != 'spill' or entry['paused']:
                return False
            newest = self.inputs.get(head.source, {}).get('last_ms')
            if self._paced and not self._keeping_up:      # timing post: chunks, only while an L2 chain is free
                if not config.get('post_chunk_ms') or not self._chain_free(lane['name']):
                    return False
                due = config['post_chunk_ms']
            else:
                due = config['lag_ms']
            return newest is not None and newest - head.frame.timestamp_ms > due and \
                self._video_chain(lane) is not None
        return not self._paced and head.frame.modality == 'audio' and \
            _now_ms() - head.frame.timestamp_ms > config['lag_ms']

    async def _skip(self, lane):
        """Skip the leader to now. The audio it did not hear becomes a TODO for a catch-up worker; nothing is lost.

        The worker gets a copy of the module's state as it was at the TODO's start (``snapshot()`` / ``restore()``;
        a module that cannot copy its state says where it last paused: ``snapshot()['pause_ms']``) and hears from
        that pause (at most ``max_lookback_ms`` before the start) to ``tail_ms`` after the end. It owns what was said
        from its first sample to the TODO's end: there its results are kept and the live module's are not
        (``trim(value, keep)`` of the module, by capture time); elsewhere the live module's.
        """
        name, entry = lane['name'], self.modules[lane['name']]
        config, module = entry['catch_up'], entry['module']
        if any(item.media and item.frame.modality == 'video' for item in lane['frames']):
            return await self._skip_video(lane)
        items = [item for item in lane['frames'] if name in item.names and item.media and item.frame.modality == 'audio']
        if not items or items[0].source not in self.memories:
            return
        first, last = items[0].frame, items[-1].frame
        start = int(first.timestamp_ms - (1000 * len(first.data) / first.sample_rate if first.sample_rate else 0))
        snapshot = await _call(module.snapshot) if callable(getattr(module, 'snapshot', None)) else None
        pause = snapshot.get('pause_ms') if isinstance(snapshot, dict) else None
        begin = start if pause is None else max(start - config['max_lookback_ms'], min(int(pause), start))
        memory = self.memories[items[0].source]
        if memory.start_ms is not None:
            begin = max(begin, int(memory.start_ms))
        if entry['until'] is not None:      # what was said complete already stays the live module's
            begin = max(begin, min(start, entry['until'] + 1))
        if entry['claims']:                 # nor does a worker hear again what another worker owns
            begin = max(begin, min(start, entry['claims'][-1][1] + 1))
        for item in items:
            item.names.remove(name)
            if not item.names:
                lane['frames'].remove(item)
                self._release(item)
        catch = entry['catch']
        catch['skips'] += 1
        todo = dict(id=catch['skips'], source=items[0].source, start_ms=start, end_ms=int(last.timestamp_ms),
                    from_ms=begin, frames=len(items), snapshot=snapshot, created=time.monotonic(), state='waiting')
        catch['skipped_ms'] += todo['end_ms'] - start
        entry['todos'].append(todo)
        entry['claims'].append((begin, todo['end_ms']))
        memory.hold(f"catch_up:{name}:{todo['id']}", begin)
        pending = [item for item in entry['todos']]
        catch['max_pending'] = max(catch['max_pending'], len(pending))
        catch['max_pending_ms'] = max(catch['max_pending_ms'], sum(item['end_ms'] - item['from_ms'] for item in pending))
        skipped = getattr(module, 'skipped', None)
        if callable(skipped):
            await _call(skipped, start, todo['end_ms'])
        self._say(f"{name} fell {round(_now_ms() - first.timestamp_ms)} ms behind: skipped to now; "
                  f"{todo['end_ms'] - start} ms of audio is caught up by a worker")
        self._start_catch_up(name)

    def _start_catch_up(self, name):
        entry = self.modules[name]
        entry['workers'] = [task for task in entry['workers'] if not task.done()]
        width = min(int(entry['catch_up']['workers']), int(getattr(entry['module'], 'max_forks', 1) or 1))
        for todo in entry['todos']:
            if len(entry['workers']) >= width or self._closed:
                return
            if todo['state'] == 'waiting':
                todo['state'] = 'running'
                work = self._catch_up_video if todo.get('video') else self._catch_up
                entry['workers'].append(asyncio.create_task(work(name, entry, todo)))

    # ------------------------------------------------------------------ L2 for video

    def _once(self, entry, output, late):
        """Drop an event the other lane (live or L2) already committed with the same label and person and an end within
        ``SAME_EVENT_MS``: ownership by end time makes this rare; it is the stored record's safety net."""
        seen = entry.setdefault('ends', deque(maxlen=256))

        def fresh(event):
            if not isinstance(event, dict) or type(event.get('end_ms')) not in (int, float):
                return True
            key = (event.get('event'), event.get('label'), str(event.get('person')))
            if any(other_key == key and other_late != late and abs(end - event['end_ms']) <= SAME_EVENT_MS
                   for end, other_key, other_late in seen):
                return False
            seen.append((event['end_ms'], key, late))
            return True
        kept = []
        for item in output if isinstance(output, list) else [output]:
            value = item['value'] if isinstance(item, dict) and isinstance(item.get('value'), dict) else item
            events = value.get('events') if isinstance(value, dict) else None
            if isinstance(item, Derived) or not isinstance(events, list) or not events:
                kept.append(item)
                continue
            mine = [event for event in events if fresh(event)]
            if mine:
                kept.append(item if len(mine) == len(events) else _with_events(item, mine))
        return kept if isinstance(output, list) else (kept[0] if kept else None)

    def _chain_free(self, name):
        entry = self.modules[name]
        width = min(int(entry['catch_up']['workers']), int(getattr(entry['module'], 'max_forks', 1) or 1))
        return sum(not task.done() for task in entry['workers']) < width

    def _post_chunks(self, lane):
        """A recording read ahead for chunks (timing post): how far ahead, or None."""
        config = self.modules[lane['name']]['catch_up']
        if not config or not config.get('post_chunk_ms') or not self._paced or self._keeping_up or \
                not self._chain_free(lane['name']):
            return None
        return config['post_chunk_ms']

    def _video_chain(self, lane):
        """The modules of a video lane that an L2 chain runs (each needs ``fork()``), in order; None if one cannot
        fork, or if memory does not allow another chain (``min_free_pct``, ``replica_mb``)."""
        names = [name for name, entry in self.modules.items() if entry['lane'] is lane]
        if not all(callable(getattr(self.modules[name]['module'], 'fork', None)) for name in names):
            return None
        config = self.modules[lane['name']]['catch_up']
        if config.get('min_free_pct') is not None and not memory_allows(config.get('replica_mb') or 0,
                                                                        float(config['min_free_pct'])):
            return None
        return names

    def _remember(self, lane, item):
        """A processed frame: given back, or kept for the pre-roll of a later cut (an L2 lane keeps
        ``max_lookback_ms`` of the frames it processed)."""
        config = self.modules[lane['name']]['catch_up']
        if not config or not item.kept or item.frame.modality != 'video':
            self._release(item)
            return
        recent = lane.setdefault('recent', deque())
        item.frame = replace(item.frame, data=None)      # its data stays in the video memory (``kept``) only
        recent.append(item)
        while recent and item.frame.timestamp_ms - recent[0].frame.timestamp_ms > config['max_lookback_ms']:
            self._release(recent.popleft())

    async def _skip_video(self, lane):
        """L2 for video: the frames the live lane has not reached by their deadline become one interval for an L2
        chain (forks of the lane's modules, the same models); the live lane goes on with the frames that come next.

        The chain owns ``[first waiting frame, newest waiting frame + tail_ms]``: the events whose end lies there are its
        (as corrections, counted late), and the live lane's are dropped there; elsewhere the live lane's. It processes
        ``max_lookback_ms`` of frames before that (the pre-roll: state only, its events there are not its) and the frames up
        to the end of what it owns, which the live lane also processes (its own warm-up after the jump). Nothing is
        skipped: every frame is processed by the lane that owns it.
        """
        name, entry = lane['name'], self.modules[lane['name']]
        config = entry['catch_up']
        chain = self._video_chain(lane)
        items = [item for item in lane['frames'] if name in item.names and item.media and item.frame.modality == 'video']
        if chain is None or not items:
            return
        start, newest = items[0].frame.timestamp_ms, items[-1].frame.timestamp_ms
        until = newest + int(config['tail_ms'])
        recent = lane.get('recent') or deque()
        pre = [item for item in recent if item.frame.timestamp_ms >= start - config['max_lookback_ms']]
        for item in list(recent):
            if item not in pre:
                self._release(item)
        recent.clear()
        # frames the live lane has processed but not yet committed are part of the pre-roll too (copies: the live
        # lane gives its own back)
        for item in sorted(lane['busy'], key=lambda item: item.frame.timestamp_ms):
            at = item.frame.timestamp_ms
            if item.media and item.frame.data is not None and start - config['max_lookback_ms'] <= at < start:
                kept = self.video_memory().keep(item.frame.data)
                if kept is not None:
                    pre.append(_Item(list(item.names), replace(item.frame, data=None), True, False, source=item.source,
                                     kept=kept))
        pre.sort(key=lambda item: item.frame.timestamp_ms)
        for item in items:
            lane['frames'].remove(item)
        catch = entry['catch']
        catch['skips'] += 1
        snapshots = {}
        for member in chain:        # the state a fork takes over (a person lock), where a module can give it
            snapshot = getattr(self.modules[member]['module'], 'snapshot', None)
            if callable(snapshot):
                snapshots[member] = await _call(snapshot)
        todo = dict(id=catch['skips'], video=True, source=items[0].source, names=list(items[0].names), chain=chain,
                    snapshots=snapshots,
                    start_ms=start, end_ms=until, from_ms=pre[0].frame.timestamp_ms if pre else start, until_ms=until,
                    frames=len(items), items=pre + items, created=time.monotonic(), state='waiting', snapshot=None)
        catch['skipped_ms'] += until - start
        entry['todos'].append(todo)
        lane.setdefault('collecting', []).append(todo)
        for owner in chain:      # the live lane's events there are the chain's: an event belongs by its end time
            self.modules[owner]['claims'].append((start, until))
            self.modules[owner]['claim_by'] = 'end'
        pending = list(entry['todos'])
        catch['max_pending'] = max(catch['max_pending'], len(pending))
        catch['max_pending_ms'] = max(catch['max_pending_ms'], sum(item['end_ms'] - item['start_ms'] for item in pending))
        self._say(f"{name} fell {newest - start} ms behind its input: frames {start}..{newest} go to an L2 worker "
                  f"(the same models), which owns them to {until} and counts there as corrections")
        self._start_catch_up(name)

    def _collect(self, lane, item):
        """A frame the live lane gets that an open L2 interval also needs (up to the end of what it owns)."""
        for todo in list(lane.get('collecting') or ()):
            if item.frame.timestamp_ms > todo['until_ms'] or todo['state'] in ('done', 'failed', 'cancelled'):
                lane['collecting'].remove(todo)
                continue
            kept = self.video_memory().keep(item.frame.data)
            if kept is None:        # beyond the disk bound: the worker ends there (its end is declared)
                lane['collecting'].remove(todo)
                todo['until_ms'] = min(todo['until_ms'], item.frame.timestamp_ms - 1)
                continue
            todo['items'].append(_Item(list(item.names), replace(item.frame, data=None), True, False,
                                       source=item.source, kept=kept))

    async def _catch_up_video(self, name, entry, todo):
        """One L2 interval: a chain of forks processes its frames (pre-roll, owned, to its end), commits its events
        as corrections at their capture times, and gives the frames back."""
        lane, catch = entry['lane'], entry['catch']
        started, forks, done = time.monotonic(), {}, 0
        low, high = todo['start_ms'], todo['until_ms']

        def keep(start_ms, end_ms):
            return low <= end_ms <= high

        try:
            for member in todo['chain']:
                forks[member] = await _call(self.modules[member]['module'].fork)
                if member in todo['snapshots'] and callable(getattr(forks[member], 'restore', None)):
                    await _call(forks[member].restore, todo['snapshots'][member])
            context = _Context(lane, forks=forks, keep=keep, correction=not (self._paced and not self._keeping_up))
            leader, width = forks[name], int(lane.get('batch') or 1)
            source = self.inputs.get(todo['source'], {})
            while not self._closed:
                waiting = todo['items'][done:]
                if not waiting:
                    if todo in (lane.get('collecting') or ()) and not source.get('ended') and \
                            (source.get('last_ms') or 0) <= todo['until_ms']:
                        await asyncio.sleep(.01)
                        continue
                    break
                batch = [item for item in waiting[:width] if item.frame.timestamp_ms <= todo['until_ms']]
                if not batch:
                    break
                frames = [replace(item.frame, data=self._video_memory.take(item.kept)) for item in batch]
                if len(frames) > 1 and callable(getattr(leader, 'process_batch', None)):
                    outputs = await _call(leader.process_batch, frames)
                else:
                    outputs = [await _call(leader.process, frame) for frame in frames]
                for item, frame, output in zip(batch, frames, outputs):
                    await self._commit(name, entry, frame, output, context, keep=keep)
                    for member in item.names:
                        if member != name and member in forks:
                            await self._commit(member, self.modules[member], frame,
                                               await _call(forks[member].process, frame), context, keep=keep)
                    self._release(item)
                done += len(batch)
                catch['heard_ms'] += round(frames[-1].timestamp_ms - frames[0].timestamp_ms) + 1
            for member, fork in forks.items():
                if callable(getattr(fork, 'finish', None)) and done:
                    output = await _call(fork.finish)
                    await self._commit(member, self.modules[member], frames[-1], output, context, keep=keep)
            todo['state'] = 'done'
            catch['done'] += 1
        except asyncio.CancelledError:
            todo['state'] = 'cancelled'
            raise
        except Exception as exc:
            todo['state'], todo['error'] = 'failed', public_text(f'{type(exc).__name__}: {exc}')[:300]
            catch['failed'] += 1
            entry['errors'] += 1
            entry['last_error'] = todo['error']
            self._gap(name, [name], todo['start_ms'], todo['until_ms'], todo['frames'], 'error')
        finally:
            for item in todo['items'][done:]:
                self._release(item)
            todo['items'] = []
            if todo in (lane.get('collecting') or ()):
                lane['collecting'].remove(todo)
            for member, fork in forks.items():
                back = getattr(self.modules[member]['module'], 'give_back', None)
                try:
                    if callable(back):          # kept for the next interval (its model stays loaded)
                        await _call(back, fork)
                    elif callable(getattr(fork, 'close', None)):
                        await _call(fork.close)
                except Exception:
                    pass
            took = time.monotonic() - started
            catch['busy_ms'] += 1000 * took
            catch['wait_ms'].append(round(1000 * (time.monotonic() - todo['created'])))
            catch['todos'].append(dict(id=todo['id'], start_ms=todo['start_ms'], end_ms=todo['until_ms'],
                                       from_ms=todo['from_ms'], state=todo['state'], frames=done,
                                       took_ms=round(1000 * took), **({'error': todo['error']} if 'error' in todo else {})))
            if todo in entry['todos']:
                entry['todos'].remove(todo)
            if not self._closed:
                self._start_catch_up(name)

    async def _feed_fork(self, source, frame, output, context):
        """Route a result of an L2 chain to the chain's modules that read it (their forks); nothing else sees it."""
        records = []
        for name, fork in context.forks.items():
            entry = self.modules[name]
            if any(src == source and wanted in (None, output) for src, wanted in entry['links']):
                records += await self._commit(name, entry, frame, await _call(fork.process, frame), context,
                                              keep=context.keep)
        return records

    async def _catch_up(self, name, entry, todo):
        """One TODO: a fork of the module hears the kept audio and commits what it says at its capture times."""
        from .capabilities import SensorMetadata as Sensor
        config, memory, catch = entry['catch_up'], self.memories[todo['source']], entry['catch']
        started, heard, sequence, fork, last = time.monotonic(), 0, 0, None, None
        until = todo['end_ms'] + config['tail_ms']
        low, high = todo['from_ms'], todo['end_ms']

        def keep(start_ms, end_ms):
            return low <= (start_ms + end_ms) / 2 <= high

        context = _Context(entry['lane'])
        try:
            fork = await _call(entry['module'].fork)
            if todo['snapshot'] is not None and callable(getattr(fork, 'restore', None)):
                await _call(fork.restore, todo['snapshot'])
            position, paced = todo['from_ms'], time.monotonic()
            while not self._closed:
                chunk = memory.after(position)
                if chunk is None:
                    if self.inputs[todo['source']]['ended'] or self._stop is None and not self._running:
                        break
                    await asyncio.sleep(.02)
                    continue
                end_ms, samples, rate, subject = chunk
                begin = end_ms - 1000. * len(samples) / rate
                if begin >= until:
                    break
                if begin < position:      # the first frame starts before the TODO's first sample
                    samples = samples[int(round((position - begin) * rate / 1000)):]
                if len(samples):
                    sequence += 1
                    last = InputFrame(samples, 'audio', end_ms, subject, rate, Sensor(
                        source_id=self.name, sensor_id=todo['source'], sequence=sequence,
                        stream_epoch=f"catch-up.{name}.{todo['id']}", channels=('mono',)))
                    output = await _call(fork.process, last)
                    await self._commit(name, entry, last, output, context, keep=keep)
                    heard += 1000. * len(samples) / rate
                position = end_ms
                if config['speed']:
                    wait = paced + heard / 1000. / float(config['speed']) - time.monotonic()
                    if wait > 0:
                        await asyncio.sleep(wait)
            if callable(getattr(fork, 'finish', None)) and last is not None:
                output = await _call(fork.finish)
                await self._commit(name, entry, last, output, context, keep=keep)
            todo['state'] = 'done'
            catch['done'] += 1
        except asyncio.CancelledError:
            todo['state'] = 'cancelled'
            raise
        except Exception as exc:
            todo['state'], todo['error'] = 'failed', public_text(f'{type(exc).__name__}: {exc}')[:300]
            catch['failed'] += 1
            entry['errors'] += 1
            entry['last_error'] = todo['error']
            self._gap(name, [name], todo['from_ms'], todo['end_ms'], todo['frames'], 'error')
        finally:
            if fork is not None and callable(getattr(fork, 'close', None)):
                try:
                    await _call(fork.close)
                except Exception:
                    pass
            took = time.monotonic() - started
            catch['busy_ms'] += 1000 * took
            catch['heard_ms'] += round(heard)
            catch['wait_ms'].append(round(1000 * (time.monotonic() - todo['created'])))
            catch['todos'].append(dict(id=todo['id'], start_ms=todo['start_ms'], end_ms=todo['end_ms'],
                                       from_ms=todo['from_ms'], state=todo['state'], heard_ms=round(heard),
                                       took_ms=round(1000 * took), **({'error': todo['error']} if 'error' in todo else {})))
            if todo in entry['todos']:
                entry['todos'].remove(todo)
            memory.hold(f"catch_up:{name}:{todo['id']}", None)
            if not self._closed:
                self._start_catch_up(name)
            if context.failures:
                errors = getattr(self.human, 'plugin_errors', None)
                if isinstance(errors, list):
                    errors.extend(dict(plugin=f'{self.name}:{key}', error=str(exc)) for key, exc in context.failures)
                    del errors[:-100]

    def _trim(self, entry, output, keep):
        """What of ``output`` was said where ``keep(start_ms, end_ms)`` holds (the module's ``trim``, else by record times)."""
        if output is None or keep is None:
            return output
        trim = getattr(entry['module'], 'trim', None)
        kept = []
        for item in output if isinstance(output, list) else [output]:
            if isinstance(item, Derived):
                at = item.timestamp_ms
                if at is None or keep(at, at):
                    kept.append(item)
            elif callable(trim):
                item = trim(item, keep)
                if item is not None:
                    kept.append(item)
            else:
                value = item.get('value', item) if isinstance(item, dict) else item
                times = [value.get(key) for key in ('start_ms', 'end_ms')] if isinstance(value, dict) else []
                events = value.get('events') if isinstance(value, dict) else None
                timed = len(times) == 2 and all(type(at) in (int, float) for at in times)
                if isinstance(events, list) and events and not timed:   # a record of events: each by its own times
                    mine = [event for event in events if not isinstance(event, dict) or not all(
                        type(event.get(key)) in (int, float) for key in ('start_ms', 'end_ms'))
                        or keep(event['start_ms'], event['end_ms'])]
                    if mine:
                        kept.append(item if len(mine) == len(events) else _with_events(item, mine))
                elif len(times) != 2 or not all(type(at) in (int, float) for at in times) or keep(*times):
                    kept.append(item)
        if not isinstance(output, list):
            return kept[0] if kept else None
        return kept

    def _upstream_lane(self, links):
        """The lane of a module this one reads: both then see each capture in the same order."""
        for source, _ in links:
            entry = self.modules.get(source) or self.modules.get(self.inputs.get(source, {}).get('derived_from'))
            if entry is not None and entry['lane'] is not None and not entry['lane']['sampled']:
                return entry['lane']
        return None

    def enable(self, name):
        self._module(name).update(enabled=True, failing=0, failed=None)
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
        entry.update(failing=0, failed=None)
        return self.status()['modules'][name]

    def set_mode(self, mode, *, actor='user'):
        """The run's counting mode from now on (``count`` or ``search``, :mod:`gamst.counting`): every module that has a
        ``set_mode`` switches for what it decides next; what was decided stays. ``actor`` is what the caller says of itself."""
        from .counting import MODES, search_settings
        if mode not in MODES:
            raise ValueError(f"mode is one of {', '.join(MODES)}")
        search = search_settings(self.search) if mode == 'search' else None
        switched = []
        for name, entry in self.modules.items():
            setter = getattr(entry['module'], 'set_mode', None)
            if callable(setter):
                setter(mode, search)
                switched.append(name)
        self.mode = mode
        self.search = search or self.search
        change = dict(mode=mode, actor=str(actor)[:32], at_ms=_now_ms(), modules=switched)
        self.mode_changes.append(change)
        del self.mode_changes[:-32]
        return dict(change, **({'search': search} if search else {}))

    async def reset(self):
        """Discard queued frames and reset every module before its next frame (a capture gap)."""
        tasks = []
        for entry in self.modules.values():
            entry['needs_reset'] = True
            drawing = getattr(entry['module'], 'drawing', None)     # what a screen draws is discarded at once
            if callable(getattr(drawing, 'clear', None)):
                drawing.clear()
        for entry in self.modules.values():
            for task in entry['workers']:
                task.cancel()
                tasks.append(task)
            for todo in entry['todos']:
                self._gap(next(key for key, value in self.modules.items() if value is entry), [], todo['from_ms'],
                          todo['end_ms'], todo['frames'], 'reset', record=False)
            entry['todos'].clear()
            entry['claims'].clear()
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
                                    coordinate_frame='image' if modality == 'video' else None,
                                    channels=('mono',) if modality == 'audio' and getattr(data, 'ndim', None) == 1 else ())
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
        frame = self._frame(input_id, data, modality or entry['modality'], int(timestamp_ms) + entry['offset_ms'],
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
        if context is not None and context.forks is not None:
            return await self._feed_fork(source, frame, output, context)
        own = context if context is not None else _Context()
        media = source in self.inputs
        if media:
            entry = self.inputs[source]
            entry['frames'] += 1
            if source in self.memories and frame.modality == 'audio' and frame.sample_rate:
                self.memories[source].append(frame.data, frame.timestamp_ms, frame.sample_rate, frame.subject_id)
            if entry['last_ms'] is not None:
                entry['intervals'].append(frame.timestamp_ms - entry['last_ms'])
            entry['last_ms'], entry['size'] = frame.timestamp_ms, getattr(frame.data, 'shape', (None, None))[:2]
            if entry['first_ms'] is None:
                entry['first_ms'] = frame.timestamp_ms
            if entry['off'] is not None:
                off, entry['off'] = entry['off'], None
                gap = True
                self._gap(source, [source], off['start_ms'], off['end_ms'], off['frames'], off['reason'],
                          subject_id=off['subject_id'])
        records, queued = [], []
        for name, entry in list(self.modules.items()):
            if not any(src == source and wanted in (None, output) for src, wanted in entry['links']):
                continue
            if entry['failed']:
                self._gap(name, [name], frame.timestamp_ms, frame.timestamp_ms, 1, 'error', subject_id=frame.subject_id,
                          extend=True)
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
                item = _Item([], frame, media, gap, source=source)
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
        if lane.get('backlog') == 'spill' and item.media and item.frame.modality == 'video' and item.frame.data is not None:
            item.kept = self.video_memory().keep(item.frame.data)
            if item.kept is None:      # beyond the memory's disk bound: declared, as a frame the lane could not take
                lane['lost'] = True
                lane['dropped'] += 1
                frame = item.frame
                self._gap(lane['name'], list(item.names), frame.timestamp_ms, frame.timestamp_ms, 1, 'backlog',
                          subject_id=frame.subject_id, extend=True)
                return
            if lane.get('collecting'):
                self._collect(lane, item)
            item.frame = replace(item.frame, data=None)
            lane['frames'].append(item)
            self._start(lane)
            return
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
        elif self._keeping_up and lane.get('sample'):
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
        its declaration); video is thinned (reason ``thinned``), so that no interval is given up as a whole;
        a lane that cannot be thinned declares its oldest tenth not processed (reason ``backlog``).
        """
        count = max(1, len(lane['frames']) // 10)
        if lane['sampled']:
            self._give_up(lane, 1)
        elif not self._thin(lane, count):
            video = all(item.frame.modality == 'video' for item in lane['frames'])
            self._give_up(lane, 1 if video else count)

    def _thin(self, lane, count):
        """Give up every second of the least thinned waiting video frames, oldest first; how many went.

        Of two neighbours with the same history the earlier one goes and the later one stands for both, so the
        newest frame stays and the rate halves evenly along the lane before it halves again anywhere.
        """
        waiting = list(lane['frames'])
        pairs = {}
        index = 0
        while index + 1 < len(waiting):
            first, second = waiting[index], waiting[index + 1]
            if first.thinned == second.thinned and first.frame.modality == second.frame.modality == 'video' \
                    and first.names == second.names and first.frame.subject_id == second.frame.subject_id:
                pairs.setdefault(first.thinned, []).append((first, second))
                index += 2
            else:
                index += 1
        if not pairs:
            return 0
        chosen = pairs[min(pairs)][:count]
        for _, second in chosen:
            second.thinned += 1
        gone = [first for first, _ in chosen]
        ids = {id(item) for item in gone}
        for item in gone:
            self._release(item)
        lane['frames'] = deque(item for item in waiting if id(item) not in ids)
        lane['dropped'] += len(gone)
        for name in sorted({name for item in gone for name in item.names}):
            mine = [item.frame for item in gone if name in item.names]
            self._gap(lane['name'], [name], mine[0].timestamp_ms, mine[-1].timestamp_ms, len(mine), 'thinned',
                      subject_id=mine[0].subject_id)
        return len(gone)

    def _start(self, lane):
        if lane['frames'] and (lane['task'] is None or lane['task'].done()) and not self._closed:
            lane['task'] = asyncio.create_task(self._drain_lane(lane))

    def video_memory(self, **options):
        """The memory that keeps the waiting frames of ``backlog="spill"`` lanes (created on first use)."""
        if self._video_memory is None:
            from .providers.video_memory import VideoMemory
            self._video_memory = VideoMemory(**options)
            memory = self._video_memory
            if memory.disk_limited:     # the disk is part of the budget
                note = (f'video spill bound {memory.disk_bytes / 1024 ** 3:.1f} GB instead of '
                        f'{memory.asked_disk_bytes / 1024 ** 3:.0f} GB: the disk keeps {memory.disk_floor_bytes / 1024 ** 3:.0f} GB free')
                self.notes['video_memory'] = note
                self._say(note)
        return self._video_memory

    def _release(self, item):
        if item.kept is not None:   # a frame of the video memory (False: already given back): never in ``_held``
            if item.kept is not False:
                self._video_memory.release(item.kept)
            item.kept = False
            return
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
        """Lanes that must not fall behind (a spilling lane may: its frames are kept)."""
        return [lane for lane in self._lanes if not lane['sampled'] and not lane.get('backlog')
                and not self._is_pausable(lane['name'])]

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
                if not running and self._lagging(lane):
                    await self._skip(lane)
                # A batching lane starts its next call as soon as the last one answered, while that one's frames are
                # still being committed (the model works while the modules after it run on the last batch).
                called = lane.get('call')
                if running and lane.get('batch') and (called is None or called.done()) and self._lagging(lane):
                    await self._skip(lane)
                if lane.get('batch') and (called is None or called.done()) and leader['enabled'] \
                        and not leader['paused'] and len(running) <= lane['batch']:
                    taken = []
                    while lane['frames'] and len(taken) < lane['batch'] and lane['name'] in lane['frames'][0].names:
                        item = lane['frames'].popleft()
                        if item.kept:
                            item.frame = replace(item.frame, data=self._video_memory.take(item.kept))
                        lane['busy'].append(item)
                        taken.append(item)
                    if taken:
                        lane['call'] = called = asyncio.create_task(
                            self._process_batch(leader, lane, [item.frame for item in taken]))
                        running.extend((item, asyncio.create_task(_nth(called, index))) for index, item in enumerate(taken))
                while lane['frames'] and len(running) < lane['width'] and not all(
                        self.modules[name]['paused'] for name in lane['frames'][0].names):
                    item = lane['frames'].popleft()
                    if item.kept:     # back from the video memory, for as long as the lane runs it
                        item.frame = replace(item.frame, data=self._video_memory.take(item.kept))
                    lane['busy'].append(item)
                    parallel = lane['width'] > 1 and lane['name'] in item.names and leader['enabled']
                    running.append((item, asyncio.create_task(self._process(leader, item.frame)) if parallel else None))
                if not running:
                    return
                item, task = running[0]
                await self._run_item(lane, item, task)
                running.popleft()
                lane['busy'].remove(item)
                self._remember(lane, item)
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
            if entry['failed']:
                ms = item.frame.timestamp_ms
                self._gap(name, [name], ms, ms, 1, 'error', subject_id=item.frame.subject_id, extend=True)
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
            if self._running and self._stop_on_failure and not lane['sampled'] and not self._is_pausable(name):
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
        if entry['last_run_ms'] is None and entry['last_admit_ms'] is None:   # it starts here
            entry['from_ms'] = max(entry['from_ms'], off['end_ms'] + 1)
        else:
            self._gap(name, [name], off['start_ms'], off['end_ms'], off['frames'], off['reason'],
                      subject_id=off['subject_id'])

    def _gap(self, place, origins, start_ms, end_ms, frames, reason, *, subject_id='local-user', record=True, log=True,
             extend=False):
        """``frames`` captured in [start_ms, end_ms] never reached ``origins`` (inputs or modules).

        ``extend``: it goes on from the last gap of the same place and reason, and is not another one.
        """
        affected = self._affected(origins)
        for name in affected if log else ():
            entry = self.modules[name]
            last = entry['gap_log'][-1] if entry['gap_log'] else None
            if extend and last is not None and (last['source'], last['reason']) == (place, reason):
                last.update(end_ms=max(last['end_ms'], end_ms), frames=last['frames'] + frames)
                entry['gap_frames'] += frames
                continue
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
        for input_id in self.inputs:
            self._outage(input_id, record=force)
        for place in list(self._gaps):
            last = self._gap_at.get(place)
            wait = FAILED_RECORD_S if self.modules.get(place, {}).get('failed') else GAP_RECORD_S
            if place not in self._gaps or not force and last is not None and time.monotonic() - last < wait:
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
        still_off = {}      # switched off and not on again yet: not declared as a gap so far, but lost all the same
        for name, entry in self.modules.items():
            if entry['off'] is not None:
                for affected in self._affected([name]):
                    still_off[affected] = max(still_off.get(affected, 0), entry['off']['frames'])
        found = [(entry['gaps'] + (name in still_off), entry['gap_frames'] + still_off.get(name, 0))
                 for name, entry in self.modules.items() if capability in (None, self._capability(entry))]
        gaps, frames = max(found, default=(0, 0))
        return dict(gaps=gaps, frames=frames)

    # ------------------------------------------------------------------ failures

    def _down(self, input_id, error, *, gone):
        """An input failed: nothing it would have delivered from here on is known. ``gone``: it is not opened again."""
        entry = self.inputs[input_id]
        entry['error'] = public_error(error)
        if gone:
            entry['ended'] = True
        if entry['down'] is None:
            start = entry['last_ms'] + 1 if entry['last_ms'] is not None else None if self._recorded else _now_ms()
            entry['down'] = dict(logged=start, recorded=start, since=start)
            entry['outages'].append(dict(from_ms=start, error=entry['error']))
        self._say(f"{input_id} failed: {entry['error']}; " + (
            'not opened again' if gone else f"trying again ({entry['retries'] + 1} of {len(self.reopen_s)})"))
        self._outage(input_id, record=True)
        self._check_left()

    def _say(self, text):
        if callable(self.notify):
            self.notify(text)

    def _outage(self, input_id, *, record=False, end=None):
        """What a failed input has not delivered up to now (a recording: up to what the others reached) is a gap."""
        entry = self.inputs[input_id]
        down = entry['down']
        if down is None:
            return
        others = [value for key, value in self.inputs.items() if key != input_id and value['last_ms'] is not None]
        if end is None:
            end = _now_ms() if not self._recorded else max((value['last_ms'] for value in others), default=None)
        if down['since'] is None and others:     # a recording that never gave a frame: it is missing from the start
            down.update(dict.fromkeys(('logged', 'recorded', 'since'), min(value['first_ms'] for value in others)))
            entry['outages'][-1]['from_ms'] = down['since']
        if end is None or down['since'] is None:
            return
        # How many frames it would have had is unknown: one is the least.
        if end >= down['logged']:
            self._gap(input_id, [input_id], down['logged'], end, int(down['logged'] == down['since']), 'input_failed',
                      record=False, extend=down['logged'] != down['since'])
            down['logged'] = end + 1
        if record and end >= down['recorded']:
            self._gap(input_id, [input_id], down['recorded'], end, int(down['recorded'] == down['since']), 'input_failed',
                      log=False)
            down['recorded'] = end + 1

    def _resumed(self, input_id, stamp):
        """The first frame after a failure."""
        entry = self.inputs[input_id]
        self._outage(input_id, record=True, end=stamp - 1)
        for name in self._affected([input_id]):     # someone may have asked just before this frame arrived
            for gap in self.modules[name]['gap_log']:
                if (gap['source'], gap['reason']) == (input_id, 'input_failed') and gap['start_ms'] < stamp <= gap['end_ms']:
                    gap['end_ms'] = stamp - 1
        entry.update(down=None, error=None, opened=time.monotonic())
        entry['outages'][-1]['resumed_ms'] = stamp
        self._say(f'{input_id} is back')

    def _check_left(self):
        """End a run in which nothing is left that could still report something."""
        if self._stop is None or not self.modules:
            return
        dead = {key for key, entry in self.inputs.items() if entry['error'] and entry['ended']} | {
            key for key, entry in self.modules.items() if entry['failed']}
        while True:
            more = {key for key, entry in self.modules.items() if key not in dead and all(
                src in dead or self.inputs.get(src, {}).get('derived_from') in dead for src, _ in entry['links'])}
            if not more:
                break
            dead |= more
        if all(name in dead for name in self.modules):
            self._stop.set()

    def failures(self):
        """Inputs and modules that failed in this run: what, why, and since when (or that it resumed)."""
        found = [dict(input=key, **outage, given_up=entry['ended'] and 'resumed_ms' not in outage)
                 for key, entry in self.inputs.items() for outage in entry['outages']]
        return found + [dict(module=key, error=entry['failed'], given_up=True)
                        for key, entry in self.modules.items() if entry['failed']]

    def _roots(self, name):
        """The inputs a module gets its frames from, directly or through other modules."""
        found, wanted = set(), [name]
        while wanted:
            entry = self.modules.get(wanted.pop())
            for source, _ in entry['links'] if entry is not None else ():
                owner = self.inputs[source]['derived_from'] if source in self.inputs else source
                if owner is None:
                    found.add(source)
                elif owner not in found:
                    wanted.append(owner)
        return found

    def warnings(self):
        """Lanes that need longer per frame than their live video input takes to give one: they fall behind."""
        if self._recorded:
            return []
        groups, found = {}, []
        for name, entry in self.modules.items():
            if not entry['every_ms'] and entry['enabled'] and not entry['failed']:
                groups.setdefault(entry['lane']['name'] if entry['lane'] is not None else None, []).append((name, entry))
        for lane, members in groups.items():
            frames = max(len(entry['latency']) for _, entry in members)
            rates = [(sorted(value['intervals'])[len(value['intervals']) // 2], key) for key, value in self.inputs.items()
                     if value['modality'] == 'video' and value['source'] is not None and len(value['intervals']) >= SLOW_FRAMES
                     and any(key in self._roots(name) for name, _ in members)]
            if frames < SLOW_FRAMES or not rates:
                continue
            interval, key = min(rates)
            width = members[0][1]['lane']['width'] if lane is not None else 1
            cost = sum(sum(entry['latency']) for _, entry in members) / frames / width
            if cost > interval > 0:
                source, size = self.inputs[key]['source'], self.inputs[key]['size']
                hint = getattr(source, 'slow_hint', None)
                where = f'lane {lane}' if lane is not None else 'the modules on the input itself'
                found.append(dict(kind='slower_than_input', lane=lane, input=key, ms_per_frame=round(cost, 1),
                                  frame_interval_ms=interval, frame_size=[size[1], size[0]] if size and size[0] else None,
                                  text=f"{where} needs {cost:.0f} ms per frame, {key} gives one every {interval:g} ms"
                                       f"{f' ({size[1]}x{size[0]})' if size and size[0] else ''}: it falls behind"
                                       f"{'; ' + hint() if callable(hint) else ''}"))
        return found

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
            if entry['down'] is not None:
                return None if entry['down']['logged'] is None else entry['down']['logged'] - 1
            if entry['ended'] and entry['error'] is None and entry['done_ms'] is not None:
                # Nothing was captured here after its end: that holds nobody back while other inputs go on.
                return max(value['done_ms'] for value in self.inputs.values() if value['done_ms'] is not None)
            return entry['done_ms'] if entry['derived_from'] is None else self._complete_until(entry['derived_from'])
        entry = self.modules[name]
        sources = [self._complete_until(src) for src, _ in entry['links'] if src in self.inputs or src in self.modules]
        if not sources or None in sources:
            return entry['until']
        bounds = list(sources) + self._inner_bounds(name, entry)
        if entry['off'] is not None:
            bounds.append(entry['off']['start_ms'] - 1)
        held = self._holding_from(entry)
        if held is not None:
            bounds.append(held - 1)
        value = min(bounds)
        entry['until'] = value if entry['until'] is None else max(entry['until'], value)
        return entry['until']

    def _inner_bounds(self, name, entry):
        """What a module has not finished yet: frames waiting in its lane (a record from the earliest capture time it
        speaks about) and catch-up TODOs (from the first sample their worker hears)."""
        bounds = []
        if entry['lane'] is not None:
            for item in (*entry['lane']['busy'], *entry['lane']['frames']):
                if name in item.names:
                    data = item.frame.data
                    begin = data.get('start_ms') if isinstance(data, dict) and not item.media else None
                    bounds.append(min(item.frame.timestamp_ms, begin if type(begin) is int else item.frame.timestamp_ms) - 1)
        bounds += [todo['from_ms'] - 1 for todo in entry['todos']]
        return bounds

    def complete_until(self, name):
        """Capture time up to which everything for ``name`` (an input or module) was processed or declared, or None."""
        return self._complete_until(name)

    def inputs_complete_until(self, name):
        """Up to where everything module ``name`` reads is complete and has reached it (not what it holds itself)."""
        entry = self.modules[name]
        sources = [self._complete_until(src) for src, _ in entry['links'] if src in self.inputs or src in self.modules]
        if not sources or None in sources:
            return None
        return min(sources + self._inner_bounds(name, entry))

    def _complete_from(self, name):
        """Capture time from which ``complete_until_ms`` accounts for everything, or None before the first frame."""
        if name in self.inputs:
            entry = self.inputs[name]
            return entry['first_ms'] if entry['derived_from'] is None else self._complete_from(entry['derived_from'])
        entry = self.modules[name]
        sources = [self._complete_from(src) for src, _ in entry['links'] if src in self.inputs or src in self.modules]
        return None if not sources or None in sources else max(entry['from_ms'], *sources)

    @staticmethod
    def _holding_from(entry):
        """Oldest capture time of evidence a module still holds inside (an open utterance, frames in a native graph)."""
        held = getattr(entry['module'], 'pending_since_ms', None)
        return None if held is None else int(held)

    async def drain(self, *, keep_up=False, batching=False):
        """Wait until every queued module has processed what it was given (``keep_up``: but not for a lane that samples
        the newest frame or keeps its backlog in the video memory; ``batching``: a lane that batches frames may keep
        fewer than a batch waiting, so that a recording fills its batches)."""
        while True:
            lanes = [lane for lane in self._lanes if lane['task'] is not None and not lane['task'].done()
                     and not (keep_up and (lane.get('sample') or lane.get('backlog')))]
            filling = [lane for lane in lanes if batching and lane.get('batch')]
            tasks = [lane['task'] for lane in lanes if lane not in filling]
            # an L2 chain of a batching lane runs beside it while frames are read (chunks, intervals); others are waited for
            tasks += [task for entry in self.modules.values()
                      if not (batching and entry['lane'] is not None and entry['lane'].get('batch'))
                      for task in entry['workers'] if not task.done()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            elif any(self._full(lane) for lane in filling):
                await asyncio.sleep(.001)    # a full batch (or chunk) waits: let the lane take it before the next frame
            else:
                return

    def _full(self, lane):
        """A batching lane of a recording has enough waiting: a batch, or, read ahead for chunks, a chunk."""
        ahead = self._post_chunks(lane)
        if ahead is None or not lane['frames']:
            return len(lane['frames']) >= lane['batch']
        return lane['frames'][-1].frame.timestamp_ms - lane['frames'][0].frame.timestamp_ms > ahead

    async def _run_module(self, name, entry, frame, context, *, media=False, gap=False, task=None):
        """Process and commit one frame; a failure is counted, declared a gap and left in ``context``."""
        if gap and media and frame.modality == 'audio':
            entry['needs_reset'] = entry['gap_out'] = True   # never join audio across a lost interval
        try:
            output = await (self._process(entry, frame) if task is None else task)
            records = await self._commit(name, entry, frame, output, context)
            entry['failing'] = 0
            return records
        except Exception as exc:
            entry['errors'] += 1
            entry['last_error'] = f'{type(exc).__name__}: {exc}'[:300]
            entry['needs_reset'] = True
            self._gap(name, [name], frame.timestamp_ms, frame.timestamp_ms, 1, 'error', subject_id=frame.subject_id)
            context.failures.append((name, exc))
            entry['failing'] += 1
            if self._stop is not None and not self._stop_on_failure and entry['failing'] >= FAILURES:
                entry['failed'] = public_text(entry['last_error'])
                self._say(f"{name} failed {FAILURES} times in a row and is given up: {entry['failed']}")
                self._check_left()
            return []

    async def _process(self, entry, frame):
        module = entry['module']
        if entry['needs_reset']:
            entry['needs_reset'] = False
            if callable(getattr(module, 'reset', None)):
                await _call(module.reset)
        started = time.monotonic()
        await self._slow(entry, [frame])
        output = await _call(module.process, frame)
        entry['last_run_ms'] = max(frame.timestamp_ms, entry['last_run_ms'] or 0)
        entry['processed'] += 1
        entry['last_latency_ms'] = round((time.monotonic() - started) * 1000, 3)
        entry['busy_ms'] += entry['last_latency_ms']
        entry['latency'].append(entry['last_latency_ms'])
        return output

    async def _slow(self, entry, frames):
        """A test and debug option (``slowdown``): this lane falls behind on purpose (live lanes only)."""
        slowdown = entry['slowdown']
        if not slowdown:
            return
        origin = min((value['first_ms'] for value in self.inputs.values() if value['first_ms'] is not None),
                     default=frames[0].timestamp_ms)
        slow = sum(any(low <= (frame.timestamp_ms - origin) / 1000 < high for low, high in slowdown.get('spans_s') or ())
                   for frame in frames if frame.modality in ('audio', 'video'))
        if slow:
            await asyncio.sleep(slowdown['ms'] * slow / 1000)

    async def _process_batch(self, entry, lane, frames):
        """One call of ``process_batch`` for consecutive frames of a lane (``batch``); bookkeeping per frame."""
        module = entry['module']
        if entry['needs_reset']:
            entry['needs_reset'] = False
            if callable(getattr(module, 'reset', None)):
                await _call(module.reset)
        started = time.monotonic()
        await self._slow(entry, frames)
        outputs = await _call(module.process_batch, frames)
        if not isinstance(outputs, list) or len(outputs) != len(frames):
            raise ValueError(f'process_batch answered {len(outputs) if isinstance(outputs, list) else outputs!r} '
                             f'outputs for {len(frames)} frames')
        spent = (time.monotonic() - started) * 1000
        entry['last_run_ms'] = max(frames[-1].timestamp_ms, entry['last_run_ms'] or 0)
        entry['processed'] += len(frames)
        entry['last_latency_ms'] = round(spent / len(frames), 3)
        entry['busy_ms'] += spent
        entry['latency'].extend([entry['last_latency_ms']] * len(frames))
        lane['batches'] += 1
        lane['batched'] += len(frames)
        lane['largest'] = max(lane['largest'], len(frames))
        return outputs

    async def _commit(self, name, entry, frame, output, context, *, keep=None):
        if keep is None and entry['claims']:     # catch-up workers own these capture times
            claims, by_end = list(entry['claims']), entry.get('claim_by') == 'end'

            def keep(start_ms, end_ms):
                at = end_ms if by_end else (start_ms + end_ms) / 2
                return not any(low <= at <= high for low, high in claims)
        output = self._trim(entry, output, keep)
        if entry.get('claim_by') == 'end' and output:     # the safety net of L2 ownership: no event twice
            output = self._once(entry, output, bool(context.forks))
        if context.correction and output:
            output = [_corrected(item) for item in output] if isinstance(output, list) else _corrected(output)
        if output is None or output == []:
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
            if frame.sensor is not None and link.sensor.clock_id != frame.sensor.clock_id:   # its times are in that clock
                link = replace(link, sensor=replace(link.sensor, clock_id=frame.sensor.clock_id,
                                                    clock_epoch=frame.sensor.clock_epoch))
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

    async def run(self, *, stop=None, on_frame=None, stop_on_failure=False):
        """Pump every source-backed input until all end (or ``stop`` is set); then finish modules.

        ``on_frame(input_id, frame, records)`` is the application's hook (a view, a report); returning
        False stops the run. ``records`` are those committed since the last call: modules in lanes answer
        while later frames are already captured, so a record's time is its own, not the frame's.
        A recorded input (``live = False``) waits for the lanes, frame by frame; a live one never waits.
        One input or module that fails does not end the others (see the module docstring) unless ``stop_on_failure``.
        """
        stop = stop or asyncio.Event()
        self._stop, self._stop_on_failure = stop, bool(stop_on_failure)
        sources = [entry['source'] for entry in self.inputs.values() if entry['source'] is not None]
        live = [bool(getattr(source, 'live', True)) for source in sources]
        if any(live) and not all(live):
            raise ValueError('live and recorded inputs have no common capture clock; run them separately')
        self._paced = self._recorded = bool(sources) and not any(live)
        from .inputs import Paced
        self._keeping_up = self._paced and any(isinstance(source, Paced) for source in sources)
        self._collected = None if on_frame is None else []
        self._running, self._failure = True, None

        def check():   # a module the run needs failed in its lane
            if self._failure is not None:
                raise self._failure

        async def take(input_id, source, frame, missed):
            """Route one frame of a source; returns how many it had missed by then."""
            entry = self.inputs[input_id]
            lost = getattr(source, 'missed', 0) - missed
            missed += lost
            entry['missed'] += lost
            stamp = frame.timestamp_ms + entry['offset_ms']
            if lost:
                entry['gaps'] += 1
                last = entry['last_ms']
                first = stamp - 1 if last is None else min(last + 1, stamp - 1)
                self._gap(input_id, [input_id], first, stamp - 1, lost, 'input', subject_id=frame.subject_id)
            frame = self._frame(input_id, frame.data, frame.modality, stamp,
                                subject_id=frame.subject_id, sample_rate=frame.sample_rate,
                                sensor=frame.sensor)
            context = _Context()
            if entry['down'] is not None:
                self._resumed(input_id, stamp)
                lost = lost or 1
            records = await self.feed(input_id, frame, gap=bool(lost), context=context)
            if self._stop_on_failure and context.failures:
                raise context.failures[0][1]
            if self._paced:
                await self.drain(keep_up=self._keeping_up, batching=True)
            check()
            if self._collected:
                records, self._collected = self._collected + records, []
            if on_frame is not None and await _call(on_frame, input_id, frame, records) is False:
                stop.set()
            await asyncio.sleep(0)
            return missed

        async def following(input_id, frames):
            """The next frame of a source, or None at its end; a failure is kept on the input."""
            try:
                return await frames.__anext__()
            except StopAsyncIteration:
                self.inputs[input_id]['ended'] = True
                return None
            except Exception as exc:
                self.inputs[input_id].update(ended=True, error=public_error(exc))
                raise

        async def pump(input_id, source):
            """Route what a source gives; a live one that fails is opened again, a bounded number of times."""
            entry, missed = self.inputs[input_id], getattr(source, 'missed', 0)
            while not stop.is_set():
                frames, failure = source.frames().__aiter__(), None
                while not stop.is_set():
                    waiting = asyncio.ensure_future(following(input_id, frames))
                    if len(sources) > 1:    # another input may end the run while this one has nothing to say
                        await asyncio.wait({waiting, stopping}, return_when=asyncio.FIRST_COMPLETED)
                        if not waiting.done():
                            waiting.cancel()
                            await asyncio.gather(waiting, return_exceptions=True)
                            return
                    try:
                        frame = await waiting
                    except Exception as exc:
                        if self._stop_on_failure:
                            raise
                        failure = exc
                        break
                    if frame is None or stop.is_set():
                        return
                    if entry['opened'] is not None and time.monotonic() - entry['opened'] > REOPENED_S:
                        entry.update(retries=0, opened=None)
                    missed = await take(input_id, source, frame, missed)
                if failure is None:
                    return
                again = getattr(source, 'live', True) and entry['retries'] < len(self.reopen_s)
                entry['ended'] = not again
                self._down(input_id, failure, gone=not again)
                if not again:
                    return
                wait = self.reopen_s[entry['retries']]
                entry['retries'] += 1
                reopen = getattr(source, 'reopen', None) or getattr(source, 'close', None)
                try:
                    if callable(reopen):
                        await _call(reopen)
                except Exception as exc:    # the next attempt says whether it matters
                    entry['error'] = public_error(exc)
                await asyncio.wait({stopping}, timeout=wait)

        async def merge():
            """Recorded inputs in capture order: the earliest of the one waiting frame of each."""
            missed = {key: getattr(self.inputs[key]['source'], 'missed', 0) for key in recorded}
            frames = {key: self.inputs[key]['source'].frames().__aiter__() for key in recorded}

            async def after(key):
                try:
                    return await following(key, frames[key])
                except Exception as exc:
                    if self._stop_on_failure:
                        raise
                    self._down(key, exc, gone=True)
                    return None

            try:
                waiting = {key: await after(key) for key in recorded}
                while not stop.is_set():
                    ready = [(frame.timestamp_ms + self.inputs[key]['offset_ms'], order, key)
                             for order, (key, frame) in enumerate(waiting.items()) if frame is not None]
                    if not ready:
                        return
                    key = min(ready)[2]
                    missed[key] = await take(key, self.inputs[key]['source'], waiting[key], missed[key])
                    waiting[key] = None if stop.is_set() else await after(key)
            finally:
                for iterator in frames.values():
                    close = getattr(iterator, 'aclose', None)
                    if callable(close):
                        await close()

        recorded = [key for key, entry in self.inputs.items() if entry['source'] is not None] if self._paced else []
        for entry in self.inputs.values():
            entry.update(ended=False, error=None, down=None, retries=0, opened=None)
            entry['outages'].clear()
        stopping = asyncio.create_task(stop.wait())
        tasks = [asyncio.create_task(merge())] if len(recorded) > 1 else [
            asyncio.create_task(pump(input_id, entry['source']))
            for input_id, entry in self.inputs.items() if entry['source'] is not None]
        try:
            if tasks:
                await asyncio.gather(*tasks)
            self._paced = True          # nothing arrives any more: what waits is worked off, not given up
            self._keeping_up = False
            self._resume(force=True)
            await self.drain()
            check()
        finally:
            for task in (*tasks, stopping):
                task.cancel()
            await asyncio.gather(*tasks, stopping, return_exceptions=True)
            self._running = self._paced = self._keeping_up = False
        try:
            records = await self.finish()
        finally:
            self._stop = None
        collected, self._collected = self._collected or [], None
        return collected + records

    async def finish(self):
        """End of input: commit what stateful modules still hold (e.g. a count at the last frame)."""
        paced, self._paced = self._paced, True
        records = []
        try:
            self._resume(force=True)
            for name, entry in list(self.modules.items()):
                await self.drain()     # what the modules before it still gave it
                finish = getattr(entry['module'], 'finish', None)
                if not entry['enabled'] or entry['failed'] or not callable(finish):
                    continue
                last = entry['last_run_ms']
                try:
                    output = await _call(finish)
                except Exception as exc:
                    if self._stop is None or self._stop_on_failure:
                        raise
                    entry['errors'] += 1
                    entry['last_error'] = f'{type(exc).__name__}: {exc}'[:300]
                    if last is not None:    # what it still held is not reported
                        self._gap(name, [name], self._holding_from(entry) or last, last, 1, 'error')
                    continue
                if output is None or last is None or (entry['spec'] is None and not isinstance(output, list)):
                    continue  # a plugin that commits its own records returns its final value only
                context = _Context()
                records += await self._commit(name, entry, self._frame(name, None, 'record', last), output, context)
                if context.failures and (self._stop is None or self._stop_on_failure):
                    raise context.failures[0][1]
            await self.drain()
            records += await self.commit_gaps(force=True)
        finally:
            self._paced = paced
        return records

    async def close(self):
        self._closed = True
        lanes = [lane['task'] for lane in self._lanes if lane['task'] is not None]
        lanes += [task for entry in self.modules.values() for task in entry['workers']]
        for task in lanes:
            task.cancel()
        await asyncio.gather(*lanes, return_exceptions=True)
        for entry in list(self.inputs.values()):
            close = getattr(entry['source'], 'close', None)
            if callable(close):
                await _call(close)
        for entry in list(self.modules.values()):
            close = getattr(entry['module'], 'close', None)
            if callable(close):
                await _call(close)
        for memory in self.memories.values():
            memory.close()
        if self._video_memory is not None:
            self._video_memory.close()
        if getattr(self.human, 'pipeline', None) is self:
            self.human.pipeline = None

    def _backlog(self, lane):
        if lane is None:
            return {}
        waiting = [*lane['busy'], *lane['frames']]
        age = None if self._recorded else max((_now_ms() - item.frame.timestamp_ms for item in waiting), default=0)
        return dict(lane=lane['name'], queue=lane['capacity'], queued=len(lane['frames']), backlog=len(waiting),
                    **({'sample': lane['sample']} if lane.get('sample') else {}),
                    **({'keeps': lane['backlog']} if lane.get('backlog') else {}),
                    **({'batch': dict(most=lane['batch'], calls=lane['batches'], batched_frames=lane['batched'],
                                      largest=lane['largest'])} if lane.get('batch') else {}),
                    backlog_ms=age, max_backlog_ms=None if self._recorded else lane['max_backlog_ms'],
                    dropped=lane['dropped'])

    @staticmethod
    def _catch_status(entry):
        catch = entry['catch']
        return dict(entry['catch_up'], skips=catch['skips'], skipped_ms=catch['skipped_ms'], done=catch['done'],
                    failed=catch['failed'], pending=len(entry['todos']), max_pending=catch['max_pending'],
                    max_pending_ms=catch['max_pending_ms'], heard_ms=catch['heard_ms'], busy_ms=round(catch['busy_ms']),
                    todo_to_done_ms=_percentiles(catch['wait_ms']), todos=list(catch['todos']))

    def status(self):
        """What is connected, on, behind, lost and complete; times are capture times in ms.

        Ages (``backlog_ms``, ``newest_result_age_ms``, ``capture_to_result_ms``) exist for live input only.
        ``complete_until_ms``: everything captured from ``complete_from_ms`` (its first frame) up to then was
        processed by the module or declared in ``gap_intervals``; ``holding_from_ms``: it still holds evidence
        from then on (a module's ``pending_since_ms``, already counted in ``complete_until_ms``); a module that answers about a window may still add results for that window.
        """
        now = _now_ms()
        for key in self.inputs:
            self._outage(key)
        return dict(
            name=self.name, complete=not any(value['gaps'] for value in self.modules.values()) and not self.failures(),
            failures=self.failures(), warnings=self.warnings(), mode=self.mode,
            **({'notes': dict(self.notes)} if self.notes else {}),
            **({'search': dict(self.search)} if self.mode == 'search' and self.search else {}),
            **({'mode_changes': [dict(item) for item in self.mode_changes]} if self.mode_changes else {}),
            backlog_bytes=self._held_bytes, max_backlog_bytes=self.max_backlog_bytes,
            **({'audio_memory': {key: memory.status() for key, memory in self.memories.items()}} if self.memories else {}),
            **({'video_memory': self._video_memory.status()} if self._video_memory is not None else {}),
            inputs={key: dict(modality=value['modality'], frames=value['frames'], last_ms=value['last_ms'],
                              derived_from=value['derived_from'], host_fed=value['source'] is None,
                              missed=value['missed'], gaps=value['gaps'],
                              offset_ms=value['offset_ms'], ended=value['ended'],
                              **({'error': value['error']} if value['error'] else {}),
                              **({'down_since_ms': value['down']['since']} if value['down'] is not None else {}),
                              **({'outages': [dict(item) for item in value['outages']]} if value['outages'] else {}),
                              **({'state': value['source'].status()}
                                 if callable(getattr(value['source'], 'status', None)) else {}))
                    for key, value in self.inputs.items()},
            references={key: dict(kind=value['kind'], label=value['label'], **(dict(
                reference_id=value['id'], reference_revision=value['revision'], effective_from_ms=value['from_ms'],
                set_by=value['set_by'], removed=bool(value.get('removed'))) if 'revision' in value else {}))
                        for key, value in self.references.items()},
            modules={key: dict(enabled=value['enabled'], consumes=value['consumes'], emits=list(value['emits']),
                               capability=value['spec'].id if value['spec'] is not None else None,
                               every_ms=value['every_ms'], processed=value['processed'], errors=value['errors'],
                               last_error=value['last_error'], last_latency_ms=value['last_latency_ms'],
                               latency_ms=_percentiles(value['latency']), capture_to_result_ms=_percentiles(value['lag']),
                               newest_result_age_ms=None if value['newest_ms'] is None or self._recorded
                               else max(0, now - value['newest_ms']),
                               gaps=value['gaps'], gap_frames=value['gap_frames'], complete=not value['gaps'] and value['off'] is None,
                               gap_intervals=[dict(item) for item in list(value['gap_log'])[-64:]],
                               **({'failed': value['failed']} if value['failed'] else {}),
                               complete_until_ms=self._complete_until(key), complete_from_ms=self._complete_from(key),
                               holding_from_ms=self._holding_from(value),
                               **({'asked': [list(item) for item in value['asked']]}
                                  if value['asked'] not in ([], [[0, None]]) else {}),
                               replicas=int(getattr(value['module'], 'replicas', 1)),
                               optional=value['optional'], paused=value['paused'], pauses=value['pauses'],
                               **({'sampled': dict(every_ms=value['every_ms'], skipped=value['skipped'],
                                                   **({'choice': value['lane']['sample']} if value['lane'] is not None
                                                      and value['lane'].get('sample') else {}))}
                                  if value['every_ms'] or value['lane'] is not None and value['lane'].get('sample') else {}),
                               **self._backlog(value['lane']), busy_ms=round(value['busy_ms']),
                               **({'catch_up': self._catch_status(value)} if value['catch_up'] else {}),
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
