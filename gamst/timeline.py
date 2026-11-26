"""Retained records as rows of capture time: what every module said at T, and whether it was done with T.

``Gamst.get_timeline`` reads the Store once and builds rows; it runs no model, keeps no state and starts
nothing. Three existing parts do the work it needs: the Store's queries, ``gamst.temporal.TemporalAligner``
for sources with their own clock (``Gamst.clocks``), and the Pipeline's own account of what it processed,
read in one place (:func:`progress`). Without that account a row says ``unknown``; it never guesses.

Cell statuses, in the order they are decided for a row ``[t, t + step)``:

``exact``    a record captured in this row, or an interval record that overlaps it;
``gap``      captured, declared lost, never processed;
``off``      the module was switched off (or paused), or nothing was set to look for (``not_instructed``);
``pending``  the module has not finished this capture time (``ref``: its last value, if still fresh);
``sampled``  the module looks at one frame per interval by declaration; this row is between two of them;
``held``     nothing new, and the last value is still within its ``freshness_ms`` (``ref``, ``age_ms``);
``none``     processed, and there was nothing to report;
``unknown``  nothing says whether this was processed.

Only the Store is read on the event loop, in short steps; records are decoded and rows built in a thread. What is
learned during a read is not in it, so every step sees the same state.

Values are never interpolated. Rows, cells and evidence are bounded; what is cut says so.
"""
from __future__ import annotations

import asyncio
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import itertools
import json
import math
import threading
import time
from types import SimpleNamespace

from .capabilities import is_raw_key as _is_raw_key
from .context import compact_observation, is_context_observation
from .counting import GAP_CAPABILITY as GAPS, INSTRUCTION_CAPABILITY as INSTRUCTIONS, REVISION
from .observation_pages import _encode as seal_cursor, _open as open_cursor
from .revisions import LATE_MS, RETRACTION, depends_on, record_span, span_of, time_ms

MAX_ROWS, MAX_COLUMNS, MAX_SCAN = 601, 32, 20_000      # 601: 600 steps that do not start on the grid
MIN_STEP_MS, MAX_STEP_MS = 20, 60_000
LOOKBACK_MS = 60_000        # how far back a still-fresh value is searched when no contract says (a reopened store)
MAX_INTERVALS = 8           # interval records named in one cell
MAX_VALUE_BYTES = 2048      # one value, as ASCII JSON
MAX_LISTED = 8              # gaps listed per column
MAX_AFFECTED = 64           # changed intervals listed in one delta
MAX_DELTA_SCAN = 5000
TURN_S, REST_S = .0003, .00005      # a building thread works this long, then lets the event loop run
HISTORY, SAMPLE_MS = 1500, 200      # Pipeline completeness kept for as_of_ms and cursors (5 minutes)
OFF_REASONS = frozenset(('disabled', 'paused', 'capture_paused'))
NOT_ASKED = 'not_instructed'        # nothing was looked for then: no words or movement was set
MAX_CHANGES = 8                     # instruction changes named in one answer
ITEM_FIELDS = ('events', 'words', 'segments')
DEFAULT_FRESHNESS_MS = 5000


# ---------------------------------------------------------------- completeness: the one adapter to the Pipeline

@dataclass
class Progress:
    """What one Pipeline module says about its own completeness; times are capture times (ms)."""
    module: str
    outputs: frozenset = frozenset()    # capability IDs, and the model ID of unregistered signals
    reported: bool = False              # the Pipeline reports completeness at all
    until: int | None = None            # everything captured up to here was processed or declared lost
    since: int | None = None            # where that account starts
    gaps: tuple = ()                    # (start_ms, end_ms, reason)
    gaps_since: int | None = None       # older gaps are no longer listed
    sampled: bool = False
    off: bool = False
    records: bool = True                # it commits records (not only a derived input)
    asked: tuple = ()                   # [from, until) in which it was asked for something; empty: always


def progress(pipeline):
    """``{module: Progress}`` from ``Pipeline.status()``; the only code that knows its fields.

    Read per module: ``complete_until_ms``, ``gap_intervals`` (dicts with start_ms, end_ms, reason, or
    ``[start, end]``), ``gaps`` (how many were declared in total), ``sampled``, ``enabled``, ``paused``,
    ``complete_from_ms`` (else see :class:`History`) and ``holding_from_ms`` (the oldest capture time of evidence it still holds, e.g.
    an open utterance). Whatever is missing stays unknown.
    """
    if pipeline is None:
        return {}
    result = {}
    for name, state in pipeline.status().get('modules', {}).items():
        module = getattr(pipeline, 'modules', {}).get(name, {}).get('module')
        spec = getattr(module, 'spec', None)
        capability = state.get('capability') or getattr(spec, 'id', None)
        outputs = {item for item in (capability, None if capability else getattr(module, 'name', name)) if item}
        until, holding = time_ms(state.get('complete_until_ms')), time_ms(state.get('holding_from_ms'))
        if until is not None and holding is not None:
            until = min(until, holding - 1)
        listed, gaps = state.get('gap_intervals') or (), []
        for item in listed:
            if isinstance(item, dict):
                start, end, reason = item.get('start_ms'), item.get('end_ms'), str(item.get('reason') or 'lost')
            else:
                start, end, reason = item[0], item[1], 'lost'
            if time_ms(start) is not None and time_ms(end) is not None:
                gaps.append((time_ms(start), time_ms(end), reason))
        total = state.get('gaps')
        cut = type(total) is int and total > len(listed)
        result[name] = Progress(
            name, frozenset(outputs), 'complete_until_ms' in state, until, time_ms(state.get('complete_from_ms')),
            tuple(sorted(gaps)), (min(gap[0] for gap in gaps) if gaps else until) if cut else None,
            bool(state['sampled']) if 'sampled' in state else bool(state.get('every_ms')),
            not state.get('enabled', True) or bool(state.get('paused')),
            bool(capability) or not state.get('emits'),
            tuple((start, end) for start, end in state.get('asked') or ()))
    return result


def _joined(parts):
    """Several modules behind one capability: done only as far as the slowest."""
    if len(parts) == 1:
        return parts[0]
    on = [part for part in parts if not part.off] or parts
    untils = [part.until for part in on]
    sinces = [part.since for part in on]
    cuts = [part.gaps_since for part in on if part.gaps_since is not None]
    return Progress(', '.join(part.module for part in parts), reported=all(part.reported for part in on),
                    until=None if None in untils else min(untils), since=None if None in sinces else max(sinces),
                    gaps=tuple(sorted(gap for part in on for gap in part.gaps)), gaps_since=max(cuts, default=None),
                    sampled=any(part.sampled for part in on), off=all(part.off for part in parts),
                    asked=() if any(not part.asked for part in on) else tuple(sorted(
                        (pair for part in on for pair in part.asked), key=lambda pair: pair[0])))


EMPTY = ({}, {}, None)
is_raw_key = lru_cache(maxsize=4096)(_is_raw_key)      # the same few field names, for every value


class History:
    """What the Pipeline said about its completeness, and how source clocks were calibrated, over knowledge time.

    ``as_of_ms`` and cursors need it: records say when they were learned, completeness does not. Bounded, in
    memory; sampled when records are committed and at every timeline read.
    """

    def __init__(self):
        self._known, self._states = deque(maxlen=HISTORY), deque(maxlen=HISTORY)
        self._pipeline, self._since = None, {}

    def sample(self, pipeline, aligner, known_ms, *, force=False):
        if self._known and not force and known_ms - self._known[-1] < SAMPLE_MS:
            return
        before, error = self._states[-1] if self._states else ({}, {}, None), None
        try:
            modules = progress(pipeline)
        except Exception as exc:    # a module's own status() failed: nothing is known about completeness then
            modules, error = {}, type(exc).__name__
        if pipeline is not self._pipeline:
            self._pipeline, self._since = pipeline, {}
        for name, part in modules.items():
            # A module that does not say where its account starts: not later than where it was first seen to be.
            if part.since is None and part.until is not None:
                part.since = self._since.setdefault(name, part.until)
            if before[0].get(name) == part:
                modules[name] = before[0][name]
        state = modules, {key: aligner.clock_status(key) for key in aligner._clock_samples}, error
        if state != before or not self._states:
            self._known.append(known_ms)
            self._states.append(state)

    def at(self, known_ms):
        """``(modules, clocks, error)`` as of ``known_ms``, or None when that is older than what is kept."""
        index = bisect_right(self._known, known_ms) - 1
        return self._states[index] if index >= 0 else None


_turn = threading.local()


def rest():
    """In a building thread: give the interpreter to the event loop after a short turn.

    Without it the loop waits up to 5 ms (Python's switch interval) each time it needs the interpreter back,
    that is, at every database call of every record it commits.
    """
    now = time.perf_counter()
    if now - getattr(_turn, 'since', 0.) > TURN_S:
        time.sleep(REST_S)
        _turn.since = time.perf_counter()


def _breathe(items):
    for item in items:
        rest()
        yield item


# ---------------------------------------------------------------- clocks

def clock_key(clock_id, clock_epoch):
    """A restarted clock (a new epoch) never reuses an old calibration."""
    return clock_id if clock_epoch is None else f'{clock_id}@{clock_epoch}'


def foreign_clock(observation):
    """The clock a record's times are in, or None for this runtime's own capture clock.

    ``provider:`` names the operations of a local worker process (``gamst.process_provider``); its records
    keep the capture time of the frame it was given. So does a server that echoes the frame's capture time.
    """
    sensor = observation.get('sensor') or {}
    clock_id = sensor.get('clock_id')
    if not isinstance(clock_id, str) or clock_id.startswith('provider:'):
        return None
    return clock_key(clock_id, sensor.get('clock_epoch'))


class _Clock:
    """One calibrated source clock: the linear fit of ``TemporalAligner``."""

    def __init__(self, key, status):
        self.key, self.offset, self.rate = key, status['offset_ms'], 1. + status['drift']
        self.status = {name: status[name] for name in ('sample_count', 'residual_ms', 'drift')}

    def local(self, value):
        return round(self.offset + self.rate * value)

    def source(self, value):
        return round((value - self.offset) / self.rate)


# ---------------------------------------------------------------- records -> evidence

def summary(value, name=None):
    """``(value, flags)``: bounded, without raw or private fields; ``flags`` says what was cut."""
    flags = set()

    def walk(item, depth):
        if isinstance(item, str):
            if len(item) > 400:
                flags.add('truncated')
            return item[:400]
        if item is None or type(item) in (bool, int):
            return item
        if type(item) is float:
            return item if math.isfinite(item) else None
        if isinstance(item, list):
            if len(item) > 8 or depth >= 4 and item:
                flags.add('truncated')
            return [] if depth >= 4 else [walk(child, depth + 1) for child in item[:8]]
        if isinstance(item, dict):
            kept = [(str(key)[:100], child) for key, child in item.items() if not is_raw_key(str(key))]
            if len(kept) < len(item):
                flags.add('raw_fields_removed')
            if len(kept) > 16 or depth >= 4 and kept:
                flags.add('truncated')
            return {} if depth >= 4 else {key: walk(child, depth + 1) for key, child in kept[:16]}
        flags.add('truncated')
        return None

    result = walk(value if name is None else compact_observation(name, value), 0)
    if len(json.dumps(result, ensure_ascii=True)) > MAX_VALUE_BYTES:
        flags.add('truncated')
        if isinstance(result, dict):    # what is left is the short scalar fields, not an arbitrary prefix
            result = {key: child for key, child in result.items()
                      if not isinstance(child, (list, dict)) and not (isinstance(child, str) and len(child) > 100)}
        else:
            result = None
    return result, sorted(flags)


def _split(value, at, span):
    """``(body, items)``: the entries of a value that name their own time, ``(kind, index, start, end)``, taken out of it."""
    if not isinstance(value, dict) or not any(value.get(name) for name in ITEM_FIELDS):
        return value, []
    body, items = dict(value), []
    for name in ITEM_FIELDS:
        entries, rest = value.get(name), []
        if not isinstance(entries, list) or not entries:
            continue
        for index, entry in enumerate(entries):
            pair = span_of(entry)
            if pair is None and isinstance(entry, dict) and span is None and name == 'events':
                pair = at, at       # an event without its own time happened at its record's capture time
            if pair is None:
                rest.append(entry)
            else:
                items.append((name, index, *pair))
        if rest:
            body[name] = rest
        else:
            body.pop(name)
    return body, items


class _Record:
    """What placing a record needs. Its value stays text until it is shown.

    A read holds up to tens of thousands of these. Decoded, they would be millions of objects, and one garbage
    collection or release of them stops every thread, the capture too.
    """
    __slots__ = ('id', 'payload', 'known', 'ts', 'type', 'capability', 'name', 'named', 'source', 'sensor', 'clock',
                 'model', 'models', 'context', 'freshness', 'declared', 'modality', 'supersedes', 'utterance',
                 'revision', 'partial', 'span', 'items', 'gap_events', 'marks')

    def __init__(self, payload, known, record=None, shared=None):
        record = json.loads(payload) if record is None else record
        observation = record['observation']
        sensor, contract = observation.get('sensor') or {}, observation.get('capability') or {}
        value, shared = observation.get('value'), {} if shared is None else shared

        def same(item):     # the same few names and contracts in every record
            return shared.setdefault(item, item)

        self.id, self.payload, self.known, self.ts, self.type = record['id'], payload, known, record['timestamp_ms'], same(record['type'])
        self.capability, self.name = same(contract.get('id')), same(observation.get('name'))
        self.named = self.capability or self.name
        self.source, self.sensor = same(sensor.get('source_id')), same(sensor.get('sensor_id'))
        self.clock, self.modality = same(foreign_clock(observation)), same(observation.get('modality'))
        self.models = same(tuple(f"{model['id']}:{model['version']}"[:200] for model in record.get('models', [])[:2]))
        self.model = same((record.get('models') or [{}])[0].get('id'))
        self.context = record['type'] == RETRACTION or is_context_observation(record)
        self.declared = same((contract.get('freshness_ms', DEFAULT_FRESHNESS_MS), contract.get('interpretation'),
                              contract.get('window_ms', 0)))
        self.freshness = self.declared[0]
        self.supersedes = record.get('supersedes')
        self.span = record_span(record)
        self.utterance = self.revision = None
        self.partial, self.gap_events = False, ()
        if isinstance(value, dict):
            if isinstance(value.get('utterance_id'), str) and self.span is not None:
                self.utterance = value['utterance_id']
                self.revision = value['revision'] if type(value.get('revision')) is int else 0
            self.partial = value.get('status') == 'partial'
            if self.capability == GAPS:
                self.gap_events = value.get('events') or ()
        items, self.marks = [], None
        for kind, index, start, end in _split(value, self.ts, self.span)[1]:
            phase = match = None
            if kind == 'events':
                event = value[kind][index]
                if event.get('event') == REVISION and isinstance(event.get('revises'), str) or isinstance(
                        event.get('event_id'), str):     # revision states (gamst.counting)
                    self.marks = self.marks or {}
                    self.marks[len(items)] = (event.get('event_id'), event.get('revises') if event.get(
                        'event') == REVISION else None, event.get('state'), event.get('reason'))
                family, _, phase = str(event.get('event', event.get('type'))).rpartition('.')
                if phase in ('matched', 'ended'):
                    match = json.dumps([family, event.get('reference'), event.get('reference_id'), event.get('index'),
                                        event.get('start_ms')], default=str)
            items.append((kind, index, start, end, phase if match else None, match))
        self.items = tuple(items)

    def decoded(self):
        return json.loads(self.payload)


class _Item:
    """One thing a column says, at a time (an instant) or over an interval; local capture time."""
    __slots__ = ('start', 'end', 'kind', 'index', 'record', 'clock', 'instant', 'gone', '_built', 'revised')

    def __init__(self, start, end, kind, index, record, clock, instant, gone, revised=None):
        self.start, self.end, self.kind, self.index, self.record = start, end, kind, index, record
        self.clock, self.instant, self.gone, self._built, self.revised = clock, instant, gone, None, revised

    def marked(self, revised):
        """The same event with the state a later revision gave it: ``(state, reason, record ID)``."""
        return _Item(self.start, self.end, self.kind, self.index, self.record, self.clock, self.instant, self.gone,
                     revised)

    def built(self, key):
        """``(ref, entry, digest)``: the digest covers content only, so equal evidence has an equal revision."""
        if self._built is None or self._built[1]['capability'] != key:
            rest()
            record = self.record.decoded()
            observation, value, taken = record['observation'], record['observation'].get('value'), 0
            if self.kind == 'record':
                value, items = _split(value, self.record.ts, self.record.span)
                taken = len(items)
            else:
                value = value[self.kind][self.index]
            shown, flags = summary(value, None if observation.get('capability') else observation.get('name'))
            if self.revised is not None and isinstance(shown, dict):
                state, reason, by = self.revised
                shown = dict(shown, state=state, revised_by=dict(record_id=by, **({'reason': reason} if reason else {})))
            entry = dict(capability=key, kind=self.kind)
            entry.update(dict(at_ms=self.start) if self.instant else dict(start_ms=self.start, end_ms=self.end))
            entry['value'] = shown
            if taken:
                entry['items_as_intervals'] = taken
            if flags:
                entry['flags'] = flags
            quality = record.get('quality') or {}
            if quality.get('level'):
                entry['level'] = quality['level']
            if quality.get('status') not in (None, 'sufficient'):
                entry['quality'] = quality['status']
            if self.clock is not None:
                entry['source_time_ms'] = self.record.ts
            sources = depends_on(record, value)
            changed = [item for item in sources if item in self.gone]
            if changed:     # what it was derived from has been revised or retracted since; GAMST does not recompute it
                entry['depends_on_changed'] = changed[:8]
            digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            ref = hashlib.sha256(f"{self.record.id}|{self.kind}|{digest}".encode()).hexdigest()[:10]
            entry.update(known_ms=self.record.known, evidence_id=self.record.id)
            if sources:
                entry['depends_on'] = sources[:8]
                if len(sources) > 8:
                    entry['depends_on_omitted'] = len(sources) - 8
            if record.get('supersedes'):
                entry.update(supersedes=record['supersedes'], revision=record.get('revision'))
            self._built = ref, entry, digest
        return self._built


def _evidence(record, clock, gone=frozenset()):
    """``(instant or None, intervals)`` of one record, in local capture time."""
    local = (lambda ms: ms) if clock is None else clock.local
    intervals = [_Item(local(start), local(end), kind, index, record, clock, False, gone)
                 for kind, index, start, end, _, _ in record.items]
    if record.span is not None:
        intervals.append(_Item(*map(local, record.span), 'record', None, record, clock, False, gone))
        return None, intervals
    return _Item(local(record.ts), local(record.ts), 'record', None, record, clock, True, gone), intervals


def _marks(records):
    """``(revised, targets)`` of records of one capability: ``{event_id: (state, reason, record ID)}`` of the newest
    revision of each event, and the IDs of the events among them (any source of the capability)."""
    revised, targets = {}, set()
    for record in sorted((record for record in records if record.marks), key=lambda item: (item.known or 0, item.ts)):
        for event_id, revises, state, reason in record.marks.values():
            if revises is not None and state in ('final', 'revoked', 'uncertain'):
                revised[revises] = state, reason, record.id      # the newest revision decides
            elif event_id is not None:
                targets.add(event_id)
    return revised, targets


@dataclass
class _Column:
    name: str                           # capability ID, else signal name, else a module or requested name
    source: str | None = None
    sensor: str | None = None
    clock: _Clock | None = None
    uncalibrated: str | None = None     # clock key: nothing of it can be placed
    records: list = field(default_factory=list)
    instants: list = field(default_factory=list)
    intervals: list = field(default_factory=list)
    progress: Progress | None = None
    processed: int | None = None
    key: str = ''

    def place(self, gone, cache, marks=({}, set())):
        """Replace superseded evidence: a newer revision of an utterance, an event whose end is now known.

        ``marks``: ``(revised, targets)`` of this capability (:func:`_marks`): an event shows the state its newest
        revision gave it, and a revision whose event is shown is not shown on its own.
        """
        latest = {}
        for order, record in enumerate(_breathe(self.records)):
            if record.utterance is not None:
                rank = record.revision, record.ts, order
                if record.utterance not in latest or rank > latest[record.utterance][0]:
                    latest[record.utterance] = rank, record.id
        matched, (revised, targets) = {}, marks
        for record in _breathe(self.records):
            if record.utterance is not None and latest[record.utterance][1] != record.id:
                continue
            which = record.id, self.clock and (self.clock.offset, self.clock.rate), gone
            if which not in cache:
                cache[which] = _evidence(record, self.clock, gone)
            instant, intervals = cache[which]
            if instant is not None:
                self.instants.append(instant)
            for position, (item, (_, _, _, _, phase, match)) in enumerate(zip(intervals, record.items)):
                mark = record.marks.get(position) if record.marks else None
                if mark is not None and mark[1] is not None and mark[1] in targets:
                    continue        # shown as the state of the event it revises
                if mark is not None and mark[0] in revised:
                    item = item.marked(revised[mark[0]])
                if phase == 'ended' and match in matched:
                    self.intervals[matched.pop(match)] = item    # same match, final end
                    continue
                if phase == 'matched':
                    matched[match] = len(self.intervals)
                self.intervals.append(item)
            self.intervals += intervals[len(record.items):]
        self.instants.sort(key=lambda item: item.start)

    def open_from(self):
        """Where an utterance that is still being revised begins: nothing from there on is settled."""
        return min((item.start for item in self.intervals if item.kind == 'record' and item.record.partial), default=None)


# ---------------------------------------------------------------- rows

def _column_cells(column, starts, step, lost_until):
    """``(cell, instant, intervals, settled)`` per row; ``settled``: the module is done with this capture time."""
    size, first, done = len(starts), starts[0], column.progress or Progress('')
    if column.uncalibrated is not None:
        return [(dict(status='unknown', reason='uncalibrated_clock'), None, [], False)] * size
    exact, over = defaultdict(list), defaultdict(list)
    for item in column.instants:
        if first <= item.start < starts[-1] + step:
            exact[(item.start - first) // step].append(item)
    for item in column.intervals:
        low = (item.start - first) // step
        high = (item.end - 1 - first) // step if item.end > item.start else low
        for index in range(max(0, low), min(size - 1, high) + 1):
            over[index].append(item)
    last = [(item.start, item) for item in column.instants]
    if done.sampled:        # between two samples the last one is shown, also when it describes a window
        last += [(item.end, item) for item in column.intervals if item.kind == 'record']
        last.sort(key=lambda pair: pair[0])
    times = [pair[0] for pair in last]
    # Before the module's own account starts, and where records were removed, the absence of a record means nothing.
    floor = max(math.inf if done.since is None else done.since, -math.inf if done.gaps_since is None else done.gaps_since,
                -math.inf if lost_until is None else lost_until + 1)
    cells = []
    for index, start in enumerate(_breathe(starts)):
        stop = start + step
        finished = done.until is not None and done.until >= stop - 1
        gap = next((gap for gap in done.gaps if gap[0] < stop and gap[1] >= start), None)
        off = gap[2] in OFF_REASONS if gap is not None else done.off and not finished
        begun = start
        if done.asked:      # the part of the row in which it was asked for something, or None
            begun = next((max(start, low) for low, high in done.asked if low < stop and (high is None or high > start)), None)
        known = finished and begun is not None and begun >= floor
        position = bisect_left(times, start) - 1
        held = last[position] if position >= 0 and start - times[position] <= last[position][1].record.freshness else None
        instant = None
        if exact.get(index) or over.get(index):
            cell, held = dict(status='exact'), None
            if exact.get(index):
                instant = exact[index][-1]
                if len(exact[index]) > 1:
                    cell['records'] = len(exact[index])
            if gap is not None:
                cell['gap'] = gap[2]
        elif begun is None:
            cell, held = dict(status='off', reason=NOT_ASKED), None
        elif off or gap is not None:
            cell, held = dict(status='off' if off else 'gap'), None
            if gap is not None:
                cell['reason'] = gap[2]
        elif done.reported and not finished:
            cell = dict(status='pending')
        elif done.sampled and known:
            cell = dict(status='sampled')
        elif held is not None and held[1].instant:
            cell = dict(status='held')
        else:
            cell, held = dict(status='none' if known else 'unknown'), None
        if held is not None:
            cell['age_ms'], instant = start - held[0], held[1]
            if any(gap[1] > held[0] and gap[0] < start for gap in done.gaps):
                cell['across_gap'] = True
        intervals = sorted(over.get(index, ()), key=lambda item: (item.start, item.end))
        if len(intervals) > MAX_INTERVALS:
            cell['intervals_omitted'] = len(intervals) - MAX_INTERVALS
            intervals = intervals[-MAX_INTERVALS:]
        cells.append((cell, instant, intervals, known or begun is None or off and gap is None))
    return cells


def _declared_gaps(records):
    """``(start, end, reason, {modules and capabilities})`` of gap records (earlier runs, or no longer listed)."""
    result = []
    for record in records:
        for event in record.gap_events:
            span = span_of(event)
            if span is not None:
                reasons = [str(item) for item in event.get('reasons') or ()]
                # Lost is the stronger statement: "off" only if nothing but switching off caused it.
                lost = [item for item in reasons if item not in OFF_REASONS]
                result.append((*span, (lost or reasons or ['lost'])[0],
                               {str(item) for name in ('modules', 'capabilities') for item in event.get(name) or ()}))
    return result


class _Query:
    """What one read asks for; :meth:`load` gets the records, :meth:`table` builds the rows as of a knowledge time."""

    def __init__(self, *, subject_id, first, stop, step_ms, capabilities, contracts, states):
        self.subject, self.first, self.stop, self.step = subject_id, first, stop, step_ms
        self.capabilities, self.contracts, self.states = capabilities, contracts, states
        self.cache, self.gone = {}, None        # evidence built once, for both knowledge times of a delta
        self.lookback = LOOKBACK_MS if not contracts else max(
            DEFAULT_FRESHNESS_MS, *(spec.freshness_ms for spec in contracts.values()))
        self.wanted = None
        if capabilities is not None:
            self.wanted = set(capabilities).union(*(modules[name].outputs for modules, _, _ in states
                                                    for name in capabilities if name in modules))

    async def load(self, store, known_ms, base_ms=None):
        """Everything the answer needs from the Store, as stored text; the only part that touches it."""
        any_of = None if self.wanted is None else sorted({*self.wanted, GAPS})
        self.scan_truncated, self.cut, self.at = False, [], {}

        async def scan(start, end, **about):
            found = []
            for rows in store.scan(self.subject, start, end, known_until=known_ms, any_of=any_of, **about):
                found += rows
                if len(found) > MAX_SCAN:
                    break
                await asyncio.sleep(0)
            return found

        # What is current, and the values before the window, as of each knowledge time.
        for moment in {known_ms, base_ms} - {None}:
            carried = store.latest_signals(self.subject, max(0, self.first - self.lookback), self.first - 1,
                                           known_until=moment)
            self.at[moment] = dict(
                gone=frozenset(store.superseded(self.subject, moment)), sources=store.source_clocks(self.subject, moment),
                carried=carried, times=store.known_times([record['id'] for record in carried]))
            await asyncio.sleep(0)
        self.window = await scan(self.first, self.stop - 1)
        self.late = await scan(self.stop, self.stop + LATE_MS, spans_only=True)
        self.clocked = {}
        for source in self.at[known_ms]['sources']:
            key = clock_key(source['clock_id'], source['clock_epoch'])
            fits = [_Clock(key, clocks[key]) for _, clocks, _ in self.states if clocks.get(key, {}).get('calibrated')]
            if fits and key not in self.clocked:
                self.clocked[key] = fits[0], await scan(
                    max(0, min(fit.source(self.first - self.lookback) for fit in fits)),
                    max(fit.source(self.stop + LATE_MS) for fit in fits), clock_id=source['clock_id'])
        self.learned, self.roots = [], []
        if base_ms is not None:
            for rows in store.scan_learned(self.subject, base_ms, known_ms, ('observation.signal', RETRACTION)):
                self.learned += rows
                if len(self.learned) > MAX_DELTA_SCAN:
                    break
                await asyncio.sleep(0)
            # What a revision replaced, also when it was stored before: its interval changed too.
            roots = sorted({row[0] for row in store.revisions(self.subject, known_ms, known_after=base_ms)})[:500]
            self.roots = store.get_many(roots)
        self.removed_until, self.removed_known = store.lost_until_ms, store.lost_known_ms

    def decode(self):
        """Stored text -> records (a worker thread)."""
        self.lost_until, self.shared = self.removed_until, {}

        def records(rows, local=lambda ms: ms):
            cut, rows = len(rows) > MAX_SCAN, rows[:MAX_SCAN]
            found = [_Record(payload, known, shared=self.shared) for payload, known in _breathe(reversed(rows))]
            if cut:
                self.scan_truncated = True
                self.lost_until = max(self.lost_until or 0, local(found[0].ts))
            return found

        rows = records(self.window)
        self.scan_truncated = self.scan_truncated or len(self.late) > MAX_SCAN
        rows += records(self.late[:MAX_SCAN])
        self.gap_records = [record for record in _breathe(rows) if record.capability == GAPS]
        self.records = [record for record in rows if record.clock is None]
        self.foreign = {key: [record for record in records(found, fit.local) if record.clock == key]
                        for key, (fit, found) in self.clocked.items()}
        self.window = self.late = self.clocked = None

    def _selected(self, record):
        if not record.context:
            return False
        if self.wanted is None:
            return record.capability not in (GAPS, INSTRUCTIONS)
        return bool({record.capability, record.name} & self.wanted) or not record.capability and record.model in self.wanted

    def table(self, known_ms, state, stop=None):
        modules, calibrations, _ = state
        stop = self.stop if stop is None else stop
        starts = list(range(self.first, stop, self.step))
        now = self.at[known_ms]
        gone = now['gone']
        self.gone = gone = self.gone if gone == self.gone else gone
        clocks, uncalibrated = {}, {}
        for source in now['sources']:
            key = clock_key(source['clock_id'], source['clock_epoch'])
            if calibrations.get(key, {}).get('calibrated'):
                clocks[key] = _Clock(key, calibrations[key])
            else:
                uncalibrated.setdefault(key, []).append(source)
        carried = [_Record(None, now['times'].get(record['id']), record, self.shared) for record in now['carried']]
        for record, source in zip(carried, now['carried']):
            record.payload = json.dumps(source)
        groups = [[record for record in carried if record.clock is None], self.records,
                  *(found for key, found in self.foreign.items() if key in clocks)]
        columns, seen = {}, set()
        # -- columns: one per capability (or signal), source, sensor and clock
        for record in _breathe(itertools.chain(*groups)):
            if record.known is not None and record.known > known_ms or record.id in seen:
                continue
            seen.add(record.id)
            if record.id in gone or not self._selected(record):
                continue
            which = record.named, record.source, record.sensor, record.clock
            if which not in columns:
                columns[which] = _Column(*which[:3], clock=clocks.get(record.clock))
            columns[which].records.append(record)
        for key, sources in uncalibrated.items():
            for source in sources:
                name = source['capability_id'] or source['name']
                if self.wanted is None or {name, source['name']} & self.wanted:
                    columns[name, source['source_id'], source['sensor_id'], key] = _Column(
                        name, source['source_id'], source['sensor_id'], uncalibrated=key)
        answered, covered, marks = set(), set(), {}
        for column in columns.values():
            if any(record.marks for record in column.records):
                marks.setdefault(column.name, []).extend(column.records)
        marks = {name: _marks(records) for name, records in marks.items()}
        for column in columns.values():
            column.place(gone, self.cache, marks.get(column.name, ({}, set())))
            models = {record.model for record in column.records if record.model and not record.capability}
            parts = [part for part in modules.values() if part.outputs & ({column.name} | models)]
            covered.update(part.module for part in parts)
            answered.update((column.name, *models, *(record.name for record in column.records[:1])))
            column.progress = _joined(parts) if parts else None
        # A module that has said nothing yet is a column too: that is what tells "pending" from "nothing there".
        for part in modules.values():
            asked = part.records if self.wanted is None else part.module in self.wanted or part.outputs & self.wanted
            if asked and part.module not in covered:
                name = next(iter(sorted(part.outputs & set(self.contracts))), part.module)
                columns[name, '', '', part.module] = _Column(name, progress=part)
                answered.update((name, *part.outputs))
            answered.add(part.module)
        for name in self.capabilities or ():
            if name not in answered:
                columns[name, '', '', ''] = _Column(name)
        ordered = [columns[which] for which in sorted(columns, key=lambda item: [str(part or '') for part in item])]
        omitted, ordered = [column.name for column in ordered[MAX_COLUMNS:]], ordered[:MAX_COLUMNS]
        counts = defaultdict(int)
        for column in ordered:
            counts[column.name] += 1
        used = set()
        for column in ordered:
            column.key = column.name if counts[column.name] == 1 else f'{column.name}@{column.source}/{column.sensor}'
            while column.key in used:
                column.key += '+'
            used.add(column.key)
        gaps = _declared_gaps([record for record in self.gap_records if record.known is None or record.known <= known_ms])
        for column in ordered:
            self._complete(column, gaps)
        return _Table(starts, self.step, ordered, omitted, uncalibrated,
                      {column.key: _column_cells(column, starts, self.step, self.lost_until) for column in ordered})

    def _complete(self, column, declared):
        """The column's progress in local capture time, with what its contract says may still come."""
        part = column.progress
        labels = {column.name, *(part.module.split(', ') if part else ())}
        listed = min((gap[0] for gap in part.gaps), default=math.inf) if part else math.inf
        more = [gap[:3] for gap in declared if gap[3] & labels and gap[1] < listed]
        if part is None:
            column.progress = Progress('', gaps=tuple(sorted(more))) if more else None
            return
        contract = self.contracts.get(column.name)
        settle = 0 if contract is None else contract.lookahead_ms + (
            contract.window_ms if contract.streaming_mode == 'window' else 0)
        local = (lambda ms: ms) if column.clock is None else column.clock.local
        column.processed = None if part.until is None else local(part.until)
        until, held_from = column.processed, column.open_from()
        if until is not None:
            until = until - settle if held_from is None else min(until - settle, held_from - 1)
        column.progress = Progress(
            part.module, part.outputs, part.reported, until, None if part.since is None else local(part.since),
            tuple(sorted([(local(gap[0]), local(gap[1]), gap[2]) for gap in part.gaps] + more)),
            None if part.gaps_since is None else local(part.gaps_since), part.sampled, part.off, part.records, part.asked)

    def affected(self, before, clocks, limit):
        """Capture intervals before ``before`` that records learned since the cursor changed, newest first."""
        truncated = len(self.learned) > MAX_DELTA_SCAN
        roots = {record['id']: _Record('', None, record, self.shared) for record in self.roots}
        found = defaultdict(list)
        for payload in _breathe(self.learned[:MAX_DELTA_SCAN]):
            record = _Record(payload, None, shared=self.shared)
            change = 'retracted' if record.type == RETRACTION else 'revised' if record.supersedes else 'added'
            for item in ([] if change == 'retracted' else [record]) + [roots.get(record.supersedes)]:
                if item is None or not self._selected(item):
                    continue
                if item.clock is not None and not clocks.get(item.clock, {}).get('calibrated'):
                    continue
                instant, intervals = _evidence(item, None if item.clock is None else _Clock(item.clock, clocks[item.clock]))
                name = item.named
                for part in ([instant] if instant is not None else []) + intervals:
                    if part.start < before:
                        found[name, change].append((part.start, part.end))
        merged = []
        for (name, change), spans in found.items():
            spans.sort()
            start, end = spans[0]
            for low, high in spans[1:]:
                if low > end + self.step:
                    merged.append(dict(capability=name, change=change, start_ms=start, end_ms=end))
                    start, end = low, high
                end = max(end, high)
            merged.append(dict(capability=name, change=change, start_ms=start, end_ms=end))
        merged.sort(key=lambda item: (-item['end_ms'], item['capability'], item['change']))
        return dict(intervals=merged[:limit], omitted=max(0, len(merged) - limit), scan_truncated=truncated)


class _Table:
    def __init__(self, starts, step, columns, omitted, uncalibrated, cells):
        self.starts, self.step, self.columns, self.omitted = starts, step, columns, omitted
        self.uncalibrated, self.cells = uncalibrated, cells
        self.index = {start: index for index, start in enumerate(starts)}

    def complete(self, start):
        index = self.index[start]
        return bool(self.columns) and all(self.cells[column.key][index][3] for column in self.columns)

    def row(self, start):
        """``{key: (cell, instant, intervals, content)}``, complete, revision: the revision covers content only."""
        index, row, complete = self.index[start], {}, self.complete(start)
        for column in self.columns:
            cell, instant, intervals, _ = self.cells[column.key][index]
            content = [cell, instant and instant.built(column.key)[2],
                       [item.built(column.key)[2] for item in intervals]]
            row[column.key] = cell, instant, intervals, content
        digest = hashlib.sha256(json.dumps([start, self.step, complete, [[key, row[key][3]] for key in sorted(row)]],
                                           sort_keys=True).encode()).hexdigest()
        return row, complete, digest[:16]


def _shown(column_key, cell, instant, intervals):
    cell, entries = dict(cell), {}
    if instant is not None:
        cell['ref'], entry, _ = instant.built(column_key)
        entries[cell['ref']] = entry
    if intervals:
        cell['intervals'] = []
        for item in intervals:
            ref, entry, _ = item.built(column_key)
            cell['intervals'].append(ref)
            entries[ref] = entry
    return cell, entries


# ---------------------------------------------------------------- one read

def _cost(value):
    return len(json.dumps(value, ensure_ascii=True))


def _open(store, cursor, scope):
    try:
        value = open_cursor(store, cursor)
        if type(value) is not dict or value.get('kind') != 'timeline':
            raise ValueError()
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError('invalid or expired timeline cursor; restart without cursor') from exc
    if value['scope'] != scope:
        raise ValueError('cursor belongs to a different timeline query')
    return value


NOT_READ = 'records learned after this cursor were removed before they were read'


async def read(store, history, *, subject_id, start_ms, end_ms, known_ms, as_of_ms, now_ms, step_ms, capabilities, settled_only,
         cursor, max_bytes, contracts, changes=()):
    """One answer of ``Gamst.get_timeline``: rows on the absolute step grid, or what changed since ``cursor``."""
    scope = hashlib.sha256(json.dumps([subject_id, step_ms, sorted(capabilities) if capabilities else None,
                                       settled_only]).encode()).hexdigest()[:32]
    first = start_ms // step_ms * step_ms
    stop = max(first + step_ms, -(-end_ms // step_ms) * step_ms)
    if (stop - first) // step_ms > MAX_ROWS:
        raise ValueError(f'{(stop - first) // step_ms} rows asked for; at most {MAX_ROWS}: '
                         'use a larger step_ms or a shorter window')
    at, base, base_stop, not_covered, lost = first, None, None, [], store.lost_until_ms or 0
    known = known_ms if as_of_ms is None else as_of_ms
    state = history.at(known)
    if cursor is not None:
        seen = _open(store, cursor, scope)
        base, base_stop = seen['base'], seen['base_stop']

        def restart(reason, detail):
            return dict(subject_id=subject_id, restart=True, reason=reason, detail=detail, rows=[], evidence={},
                        has_more=False, next_cursor=None)

        if seen['to'] is not None:          # the rest of an answer that did not fit: same two knowledge times
            first, stop, known, at = seen['first'], seen['stop'], seen['to'], seen['at']
            state = history.at(known)
            if lost > seen['lost'] and lost >= at:
                return restart('retention', 'records of the remaining rows were removed; read again without cursor')
        else:
            first = min(first, seen['first'])
            if (stop - first) // step_ms > MAX_ROWS:
                first = stop - MAX_ROWS * step_ms
            if lost > seen['lost'] and lost >= first:
                first = (lost // step_ms + 1) * step_ms
                if first >= min(stop, base_stop):
                    return restart('retention', 'every row this cursor covered lost records; read again without cursor')
            if first > seen['first']:       # still reported in ``affected``, not row by row
                not_covered.append(dict(start_ms=seen['first'], end_ms=min(first, base_stop)))
            at = first
        if base is not None and (store.lost_known_ms or 0) > base:
            return restart('retention', NOT_READ)
        if state is None or base is not None and history.at(base) is None:
            return restart('completeness_history', 'what was complete then is no longer kept; read again without cursor')
    before = None if base is None else history.at(base)
    states = [item for item in (state, before) if item is not None] or [EMPTY]
    query = _Query(subject_id=subject_id, first=first, stop=stop, step_ms=step_ms, capabilities=capabilities,
                   contracts=dict(contracts), states=states)
    signer = SimpleNamespace(_cursor_key=store._cursor_key)     # a new key after a deletion: this answer is then void
    await query.load(store, known, base)
    if base is not None and (query.removed_known or 0) > base:
        return restart('retention', NOT_READ)
    lost = query.removed_until or 0
    if cursor is not None and seen['to'] is not None and lost > seen['lost'] and lost >= at:
        return restart('retention', 'records of the remaining rows were removed; read again without cursor')
    return await asyncio.to_thread(
        _answer, query, signer, scope=scope, at=at, base=base, base_stop=base_stop, known=known, state=state, before=before,
        not_covered=not_covered, lost=lost, now_ms=now_ms, settled_only=settled_only, max_bytes=max_bytes, changes=changes)


def _instructions(changes, columns, first, stop):
    """What was asked for in [first, stop): each reference and revision, from when and until when; and how many are not shown."""
    names, found, last = {column.name for column in columns}, [], {}
    for known, change in changes:
        before = last.get(change['reference'])
        if before is None and 'previous_revision' in change:    # what the run began with
            before = dict(reference=change['reference'], kind=change['kind'], reference_id=change['previous_id'],
                          reference_revision=change['previous_revision'], from_ms=None,
                          capabilities=change.get('capabilities', []))
            found.append(before)
        if before is not None and before.get('until_ms') is None:
            before['until_ms'] = change['start_ms']
        if change['event'] == 'instruction.removed':
            before['removed_by'] = change['set_by']
            continue
        last[change['reference']] = entry = dict(
            {key: change[key] for key in ('reference', 'kind', 'label', 'reference_id', 'reference_revision', 'set_by', 'source')
             if key in change}, from_ms=change['start_ms'], known_ms=known, capabilities=change.get('capabilities', []))
        found.append(entry)
    shown = [entry for entry in found if (entry['from_ms'] or 0) < stop and entry.get('until_ms', stop) > first
             and (not names or names & set(entry['capabilities']))]
    return shown[-MAX_CHANGES:], max(0, len(shown) - MAX_CHANGES)


def _answer(query, signer, *, scope, at, base, base_stop, known, state, before, not_covered, lost, now_ms, settled_only,
            max_bytes, changes=()):
    """Rows and evidence of a loaded query (a worker thread: no Store, no runtime)."""
    query.decode()
    subject_id, first, stop, step_ms, contracts = query.subject, query.first, query.stop, query.step, query.contracts
    table = query.table(known, state or EMPTY)
    old = None if base is None or base_stop <= first else query.table(base, before, base_stop)

    def rows():
        for start in _breathe(table.starts):
            if start < at or settled_only and not table.complete(start):
                continue
            row, complete, revision = table.row(start)
            past = old.row(start) if old is not None and start < base_stop else None
            if past is not None and settled_only and not past[1]:
                past = None
            if past is None or past[2] != revision:
                yield start, row, complete, revision, past

    untils = [column.progress.until if column.progress and column.uncalibrated is None else None
              for column in table.columns if not (column.progress and column.progress.off)]
    result = dict(
        subject_id=subject_id, step_ms=step_ms, start_ms=first, end_ms=stop, known_ms=known, now_ms=now_ms,
        complete_until_ms=None if not untils or None in untils else min(untils),
        columns=[_describe(column, contracts, first, stop) for column in table.columns],
        columns_omitted=table.omitted[:MAX_COLUMNS],
        uncalibrated_clocks=[dict(clock=key, records=sum(source['records'] for source in sources),
                                  sample_count=(state or EMPTY)[1].get(key, {}).get('sample_count', 0),
                                  newest_source_ms=max(source['newest_ms'] for source in sources),
                                  sources=sorted({source['source_id'] for source in sources if source['source_id']})[:8])
                             for key, sources in sorted(table.uncalibrated.items())[:8]],
        scan_truncated=query.scan_truncated, removed_until_ms=query.removed_until, rows=[], evidence={},
        rows_total=len(table.starts), rows_complete=sum(map(table.complete, table.starts)), has_more=True, byte_budget=max_bytes,
        next_cursor='', values='retained records; summarized, never interpolated')
    asked, omitted = _instructions(changes, table.columns, first, stop)
    if asked:
        result['instructions'] = asked      # a count of one is never a count of another; before from_ms nothing was looked for
        if omitted:
            result['instructions_omitted'] = omitted
    if state is None:
        result['completeness_history'] = 'unavailable'      # as_of_ms is older than what is kept: nothing says complete
    elif state[2] is not None:
        result['completeness_error'] = state[2]
    if base is not None:
        result['since_known_ms'] = base
        result['affected'] = query.affected(base_stop, (state or EMPTY)[1], min(MAX_AFFECTED, max_bytes // 512))
        if not_covered:
            result['not_covered'] = not_covered

    def seal(**values):
        return seal_cursor(signer, dict(kind='timeline', scope=scope, lost=lost, first=first, **values))

    more = dict(base=base, base_stop=base_stop, to=known, stop=stop)
    result['next_cursor'] = seal(**more, at=stop)
    size = _cost(result) + 64
    if size > max_bytes:
        raise ValueError(f'{len(table.columns)} capabilities need max_bytes >= {size + 1024}; ask for fewer or raise it')
    rest = None
    for start, row, complete, revision, past in rows():
        shown, entries = dict(t_ms=start, change='added', complete=complete, revision=revision, cells={}), {}
        if past is not None:
            shown.update(change='revised', was={})
        learned = None
        for key in sorted({*row, *(past[0] if past else ())}):
            new, was = row.get(key), past[0].get(key) if past else None
            if past is not None and new is not None and was is not None and new[3] == was[3]:
                continue
            if new is not None:
                shown['cells'][key], items = _shown(key, *new[:3])
                entries.update(items)
                learned = max([learned or 0, *(item['known_ms'] or 0 for item in items.values())])
            if was is not None:
                shown['was'][key], items = _shown(key, *was[:3])
                for ref, entry in items.items():
                    entries.setdefault(ref, dict(entry))
                if was[0]['status'] == 'exact' and (new is None or new[0]['status'] != 'exact'):
                    shown['change'] = 'retracted'
        if learned:
            shown['known_ms'] = learned
        shown.update({} if past is None else dict(was_revision=past[2]))
        fresh = {ref: entry for ref, entry in entries.items() if ref not in result['evidence']}
        cost = _cost(shown) + 2 + sum(_cost(entry) + 16 for entry in fresh.values())
        if size + cost > max_bytes and not result['rows']:
            fresh = {ref: dict(entry, value=None, flags=['value_omitted']) for ref, entry in fresh.items()}
            cost = _cost(shown) + 2 + sum(_cost(entry) + 16 for entry in fresh.values())
            if size + cost > max_bytes:
                raise ValueError(f'one row needs max_bytes >= {size + cost}; ask for fewer capabilities or raise it')
        if size + cost > max_bytes:
            rest = start
            break
        result['rows'].append(shown)
        result['evidence'].update(fresh)
        size += cost
    if rest is None:
        result.update(has_more=False, next_cursor=seal(base=known, base_stop=stop, to=None, stop=stop, at=stop))
    else:
        result['next_cursor'] = seal(**more, at=rest)
    if base is not None:
        gone = query.at[known]['gone']
        for entry in result['evidence'].values():
            if entry['evidence_id'] in gone:
                entry['current'] = False
    return result


def _describe(column, contracts, first, stop):
    done, contract = column.progress, contracts.get(column.name)
    newest = column.records[-1] if column.records else None
    freshness, interpretation, window = newest.declared if newest is not None else (
        (contract.freshness_ms, contract.interpretation, contract.window_ms) if contract is not None
        else (DEFAULT_FRESHNESS_MS, None, 0))
    result = dict(key=column.key, capability=column.name, module=done.module if done and done.module else None,
                  complete_until_ms=done.until if done else None,
                  freshness_ms=freshness)
    if column.processed not in (None, result['complete_until_ms']):
        result['processed_until_ms'] = column.processed   # later, but results for this time may still come
    for key, value in (('source_id', column.source), ('sensor_id', column.sensor),
                       ('interpretation', interpretation), ('modality', newest and newest.modality)):
        if value:
            result[key] = value
    if newest is not None:
        result['models'] = list(newest.models)
    for key, value in (('sampled', done and done.sampled), ('off', done and done.off)):
        if value:
            result[key] = True
    if column.uncalibrated is not None:
        result['uncalibrated_clock'] = column.uncalibrated
    if column.clock is not None:
        result['clock'] = dict(id=column.clock.key, **column.clock.status)
    if any(not item.instant and item.end > item.start for item in column.intervals) and not window:
        result['late_intervals'] = True     # reported when it is over: a complete row can still gain one
    gaps = [gap for gap in done.gaps if gap[0] < stop and gap[1] >= first] if done is not None else []
    if gaps:
        result['gaps'] = [dict(start_ms=gap[0], end_ms=gap[1], reason=gap[2]) for gap in gaps[-MAX_LISTED:]]
        if len(gaps) > MAX_LISTED:
            result['gaps_omitted'] = len(gaps) - MAX_LISTED
    return result
