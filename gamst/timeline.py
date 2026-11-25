"""Retained records as rows of capture time: what every module said at T, and whether it was done with T.

``Gamst.get_timeline`` reads the Store once and builds rows; it runs no model, keeps no state and starts
nothing. Three existing parts do the work it needs: the Store's queries, ``gamst.temporal.TemporalAligner``
for sources with their own clock (``Gamst.clocks``), and the Pipeline's own account of what it processed,
read in one place (:func:`progress`). Without that account a row says ``unknown``; it never guesses.

Cell statuses, in the order they are decided for a row ``[t, t + step)``:

``exact``    a record captured in this row, or an interval record that overlaps it;
``gap``      captured, declared lost, never processed;
``off``      the module was switched off (or paused);
``pending``  the module has not finished this capture time (``ref``: its last value, if still fresh);
``sampled``  the module looks at one frame per interval by declaration; this row is between two of them;
``held``     nothing new, and the last value is still within its ``freshness_ms`` (``ref``, ``age_ms``);
``none``     processed, and there was nothing to report;
``unknown``  nothing says whether this was processed.

Values are never interpolated. Rows, cells and evidence are bounded; what is cut says so.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math

from .capabilities import is_raw_key as _is_raw_key
from .context import compact_observation, is_context_observation
from .counting import GAP_CAPABILITY as GAPS
from .observation_pages import _encode as seal_cursor, _open as open_cursor
from .revisions import LATE_MS, RETRACTION, depends_on, named, record_span, span_of, time_ms

MAX_ROWS, MAX_COLUMNS, MAX_SCAN = 601, 32, 20_000      # 601: 600 steps that do not start on the grid
MIN_STEP_MS, MAX_STEP_MS = 20, 60_000
LOOKBACK_MS = 60_000        # how far back a still-fresh value is searched when no contract says (a reopened store)
MAX_INTERVALS = 8           # interval records named in one cell
MAX_VALUE_BYTES = 2048      # one value, as ASCII JSON
MAX_LISTED = 8              # gaps listed per column
MAX_AFFECTED = 64           # changed intervals listed in one delta
MAX_DELTA_SCAN = 5000
HISTORY, SAMPLE_MS = 1500, 200      # Pipeline completeness kept for as_of_ms and cursors (5 minutes)
OFF_REASONS = frozenset(('disabled', 'paused', 'capture_paused'))
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
            bool(capability) or not state.get('emits'))
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
                    sampled=any(part.sampled for part in on), off=all(part.off for part in parts))


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


@dataclass
class _Item:
    """One thing a column says, at a time (an instant) or over an interval; local capture time."""
    start: int
    end: int
    kind: str                # record | events | words | segments
    record: dict
    value: object
    known: int | None = None # knowledge time of its record
    clock: _Clock | None = None
    instant: bool = False
    items: int = 0           # of its list entries, those that became intervals of their own
    gone: frozenset = frozenset()
    _built: tuple = None

    @property
    def freshness(self):
        contract = self.record['observation'].get('capability') or {}
        return contract.get('freshness_ms', DEFAULT_FRESHNESS_MS)

    def built(self, key):
        """``(ref, entry, digest)``: the digest covers content only, so equal evidence has an equal revision."""
        if self._built is None or self._built[1]['capability'] != key:
            observation, record = self.record['observation'], self.record
            value, flags = summary(self.value, None if observation.get('capability') else observation.get('name'))
            entry = dict(capability=key, kind=self.kind)
            entry.update(dict(at_ms=self.start) if self.instant else dict(start_ms=self.start, end_ms=self.end))
            entry['value'] = value
            if self.items:
                entry['items_as_intervals'] = self.items
            if flags:
                entry['flags'] = flags
            quality = record.get('quality') or {}
            if quality.get('level'):
                entry['level'] = quality['level']
            if quality.get('status') not in (None, 'sufficient'):
                entry['quality'] = quality['status']
            if self.clock is not None:
                entry['source_time_ms'] = record['timestamp_ms']
            sources = depends_on(record, self.value)
            changed = [item for item in sources if item in self.gone]
            if changed:     # what it was derived from has been revised or retracted since; GAMST does not recompute it
                entry['depends_on_changed'] = changed[:8]
            digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            ref = hashlib.sha256(f"{record['id']}|{self.kind}|{digest}".encode()).hexdigest()[:10]
            entry.update(known_ms=self.known, evidence_id=record['id'])
            if sources:
                entry['depends_on'] = sources[:8]
                if len(sources) > 8:
                    entry['depends_on_omitted'] = len(sources) - 8
            if record.get('supersedes'):
                entry.update(supersedes=record['supersedes'], revision=record.get('revision'))
            self._built = ref, entry, digest
        return self._built


def _evidence(record, clock, known=None, gone=frozenset()):
    """``(instant or None, intervals)`` of one record, in local capture time."""
    observation, at = record['observation'], record['timestamp_ms']
    value, span = observation.get('value'), record_span(record)
    local = (lambda ms: ms) if clock is None else clock.local
    intervals, body, taken = [], value, 0
    if isinstance(value, dict) and any(value.get(name) for name in ITEM_FIELDS):
        body = dict(value)
        for name in ITEM_FIELDS:
            entries, rest = value.get(name), []
            if not isinstance(entries, list) or not entries:
                continue
            for entry in entries:
                pair = span_of(entry)
                if pair is None and isinstance(entry, dict) and span is None and name == 'events':
                    pair = at, at       # an event without its own time happened at its record's capture time
                if pair is None:
                    rest.append(entry)
                else:
                    taken += 1
                    intervals.append(_Item(local(pair[0]), local(pair[1]), name, record, entry, known, clock, gone=gone))
            if rest:
                body[name] = rest
            else:
                body.pop(name)
    if span is not None:
        intervals.append(_Item(local(span[0]), local(span[1]), 'record', record, body, known, clock, False, taken, gone))
        return None, intervals
    return _Item(local(at), local(at), 'record', record, body, known, clock, True, taken, gone), intervals


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

    def place(self, known, gone, cache):
        """Replace superseded evidence: a newer revision of an utterance, an event whose end is now known."""
        latest = {}
        for order, record in enumerate(self.records):
            value = record['observation'].get('value')
            utterance = value.get('utterance_id') if isinstance(value, dict) else None
            if isinstance(utterance, str) and record_span(record) is not None:
                revision = value.get('revision')
                rank = revision if type(revision) is int else 0, record['timestamp_ms'], order
                if utterance not in latest or rank > latest[utterance][0]:
                    latest[utterance] = rank, record['id']
        matched = {}
        for record in self.records:
            value = record['observation'].get('value')
            utterance = value.get('utterance_id') if isinstance(value, dict) else None
            if isinstance(utterance, str) and utterance in latest and latest[utterance][1] != record['id']:
                continue
            which = record['id'], self.clock and (self.clock.offset, self.clock.rate), gone
            if which not in cache:
                cache[which] = _evidence(record, self.clock, known.get(record['id']), gone)
            instant, intervals = cache[which]
            if instant is not None:
                self.instants.append(instant)
            for item in intervals:
                if item.kind == 'events':
                    event = item.value
                    family, _, phase = str(event.get('event', event.get('type'))).rpartition('.')
                    which = json.dumps([family, event.get('reference'), event.get('reference_id'),
                                        event.get('index'), event.get('start_ms')], default=str)
                    if phase == 'ended' and which in matched:
                        self.intervals[matched.pop(which)] = item    # same match, final end
                        continue
                    if phase == 'matched':
                        matched[which] = len(self.intervals)
                self.intervals.append(item)
        self.instants.sort(key=lambda item: item.start)

    def open_from(self):
        """Where an utterance that is still being revised begins: nothing from there on is settled."""
        return min((item.start for item in self.intervals if item.kind == 'record' and isinstance(item.value, dict)
                    and item.value.get('status') == 'partial'), default=None)


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
    for index, start in enumerate(starts):
        stop = start + step
        finished = done.until is not None and done.until >= stop - 1
        gap = next((gap for gap in done.gaps if gap[0] < stop and gap[1] >= start), None)
        off = gap[2] in OFF_REASONS if gap is not None else done.off and not finished
        known = finished and start >= floor
        position = bisect_left(times, start) - 1
        held = last[position] if position >= 0 and start - times[position] <= last[position][1].freshness else None
        instant = None
        if exact.get(index) or over.get(index):
            cell, held = dict(status='exact'), None
            if exact.get(index):
                instant = exact[index][-1]
                if len(exact[index]) > 1:
                    cell['records'] = len(exact[index])
            if gap is not None:
                cell['gap'] = gap[2]
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
        cells.append((cell, instant, intervals, known or off and gap is None))
    return cells


def _declared_gaps(records):
    """``(start, end, reason, {modules and capabilities})`` of gap records (earlier runs, or no longer listed)."""
    result = []
    for record in records:
        for event in (record['observation'].get('value') or {}).get('events') or ():
            span = span_of(event)
            if span is not None:
                reasons = [str(item) for item in event.get('reasons') or ()]
                # Lost is the stronger statement: "off" only if nothing but switching off caused it.
                lost = [item for item in reasons if item not in OFF_REASONS]
                result.append((*span, (lost or reasons or ['lost'])[0],
                               {str(item) for name in ('modules', 'capabilities') for item in event.get(name) or ()}))
    return result


class _Query:
    """What one read asks for, and the records it needs; :meth:`table` builds the rows as of a knowledge time."""

    def __init__(self, store, *, subject_id, first, stop, step_ms, capabilities, contracts, known_ms, states):
        self.store, self.subject, self.first, self.stop, self.step = store, subject_id, first, stop, step_ms
        self.capabilities, self.contracts = capabilities, contracts
        self.cache, self.gone = {}, None        # evidence built once, for both knowledge times of a delta
        self.lookback = LOOKBACK_MS if not contracts else max(
            DEFAULT_FRESHNESS_MS, *(spec.freshness_ms for spec in contracts.values()))
        self.wanted = None
        if capabilities is not None:
            self.wanted = set(capabilities).union(*(modules[name].outputs for modules, _, _ in states
                                                    for name in capabilities if name in modules))
        bounds = dict(any_of=None if self.wanted is None else sorted(self.wanted), known_until=known_ms, with_known=True,
                      limit=MAX_SCAN + 1)
        rows = store.query(subject_id, first, stop - 1, 'observation.signal', **bounds)
        late = store.query(subject_id, stop, stop + LATE_MS, 'observation.signal', spans_only=True, **bounds)
        self.scan_truncated = len(rows) > MAX_SCAN or len(late) > MAX_SCAN
        self.lost_until = store.lost_until_ms
        if len(rows) > MAX_SCAN:
            rows = rows[-MAX_SCAN:]
            self.lost_until = max(self.lost_until or 0, rows[0][0]['timestamp_ms'])
        self.records = [pair for pair in rows + late if foreign_clock(pair[0]['observation']) is None]
        self.foreign = {}
        for source in store.source_clocks(subject_id, known_ms):
            key = clock_key(source['clock_id'], source['clock_epoch'])
            fits = [_Clock(key, clocks[key]) for _, clocks, _ in states if clocks.get(key, {}).get('calibrated')]
            if not fits or key in self.foreign:
                continue
            extra = store.query(subject_id, max(0, min(fit.source(first - self.lookback) for fit in fits)),
                                max(fit.source(stop + LATE_MS) for fit in fits), 'observation.signal',
                                clock_id=source['clock_id'], **bounds)
            if len(extra) > MAX_SCAN:
                self.scan_truncated, extra = True, extra[-MAX_SCAN:]
                self.lost_until = max(self.lost_until or 0, fits[0].local(extra[0][0]['timestamp_ms']))
            self.foreign[key] = [pair for pair in extra if foreign_clock(pair[0]['observation']) == key]
        self.gap_records = store.query(subject_id, first, stop + LATE_MS, 'observation.signal', registered_only=True,
                                       capability_id=GAPS, known_until=known_ms, with_known=True, limit=1000)

    def _selected(self, record):
        observation = record['observation']
        if not is_context_observation(record):
            return False
        capability, name = (observation.get('capability') or {}).get('id'), observation.get('name')
        if self.wanted is None:
            return capability != GAPS
        model = (record.get('models') or [{}])[0].get('id')
        return bool({capability, name} & self.wanted) or not capability and model in self.wanted

    def table(self, known_ms, state, stop=None):
        modules, calibrations, _ = state
        stop = self.stop if stop is None else stop
        starts = list(range(self.first, stop, self.step))
        gone = frozenset(self.store.superseded(self.subject, known_ms))
        self.gone = gone = self.gone if gone == self.gone else gone
        known, clocks, uncalibrated, records = {}, {}, {}, []
        carried = self.store.latest_signals(self.subject, max(0, self.first - self.lookback), self.first - 1,
                                            known_until=known_ms)
        for source in self.store.source_clocks(self.subject, known_ms):
            key = clock_key(source['clock_id'], source['clock_epoch'])
            if calibrations.get(key, {}).get('calibrated'):
                clocks[key] = _Clock(key, calibrations[key])
            else:
                uncalibrated.setdefault(key, []).append(source)
        times = self.store.known_times([record['id'] for record in carried])
        groups = [((record, times.get(record['id'])) for record in carried
                   if foreign_clock(record['observation']) is None), self.records,
                  *(pairs for key, pairs in self.foreign.items() if key in clocks)]
        for group in groups:
            for record, learned in group:
                if (learned is None or learned <= known_ms) and record['id'] not in known:
                    known[record['id']] = learned
                    records.append(record)
        # -- columns: one per capability (or signal), source, sensor and clock
        columns = {}
        for record in records:
            if record['id'] in gone or not self._selected(record):
                continue
            observation = record['observation']
            sensor, key = observation.get('sensor') or {}, foreign_clock(observation)
            which = (named(record), sensor.get('source_id'),
                     sensor.get('sensor_id'), key)
            if which not in columns:
                columns[which] = _Column(*which[:3], clock=clocks.get(key))
            columns[which].records.append(record)
        for key, sources in uncalibrated.items():
            for source in sources:
                name = source['capability_id'] or source['name']
                if self.wanted is None or {name, source['name']} & self.wanted:
                    columns[name, source['source_id'], source['sensor_id'], key] = _Column(
                        name, source['source_id'], source['sensor_id'], uncalibrated=key)
        answered, covered = set(), set()
        for column in columns.values():
            column.place(known, gone, self.cache)
            models = {record['models'][0]['id'] for record in column.records
                      if record.get('models') and not record['observation'].get('capability')}
            parts = [part for part in modules.values() if part.outputs & ({column.name} | models)]
            covered.update(part.module for part in parts)
            answered.update((column.name, *models, *(record['observation']['name'] for record in column.records[:1])))
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
        gaps = _declared_gaps([record for record, learned in self.gap_records if learned is None or learned <= known_ms])
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
            None if part.gaps_since is None else local(part.gaps_since), part.sampled, part.off)

    def affected(self, after_ms, known_ms, before, clocks, limit):
        """Capture intervals before ``before`` that records learned in (after_ms, known_ms] changed, newest first."""
        bounds = dict(known_after=after_ms, known_until=known_ms, limit=MAX_DELTA_SCAN + 1)
        any_of = None if self.wanted is None else sorted(self.wanted)
        records = self.store.query(self.subject, None, None, 'observation.signal', any_of=any_of, **bounds)
        retractions = self.store.query(self.subject, None, None, RETRACTION, **bounds)
        truncated = len(records) > MAX_DELTA_SCAN or len(retractions) > MAX_DELTA_SCAN
        roots = {record['id']: record for record in self.store.get_many(
            list({record['supersedes'] for record in records + retractions if record.get('supersedes')})[:500])}
        found = defaultdict(list)
        for record in records + retractions:
            change = 'retracted' if record['type'] == RETRACTION else 'revised' if record.get('supersedes') else 'added'
            for item in ([] if change == 'retracted' else [record]) + [roots.get(record.get('supersedes'))]:
                if item is None or not self._selected(item):
                    continue
                key = foreign_clock(item['observation'])
                if key is not None and not clocks.get(key, {}).get('calibrated'):
                    continue
                instant, intervals = _evidence(item, None if key is None else _Clock(key, clocks[key]))
                name = named(item)
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


def read(store, history, *, subject_id, start_ms, end_ms, known_ms, as_of_ms, now_ms, step_ms, capabilities, settled_only,
         cursor, max_bytes, contracts):
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
        if base is not None and store.lost_known_ms is not None and store.lost_known_ms > base:
            return restart('retention', 'records learned after this cursor were removed before they were read')
        if state is None or base is not None and history.at(base) is None:
            return restart('completeness_history', 'what was complete then is no longer kept; read again without cursor')
    before = None if base is None else history.at(base)
    states = [item for item in (state, before) if item is not None] or [EMPTY]
    query = _Query(store, subject_id=subject_id, first=first, stop=stop, step_ms=step_ms, capabilities=capabilities,
                   contracts=contracts, known_ms=known, states=states)
    table = query.table(known, state or EMPTY)
    old = None if base is None or base_stop <= first else query.table(base, before, base_stop)

    def rows():
        for start in table.starts:
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
        scan_truncated=query.scan_truncated, removed_until_ms=store.lost_until_ms, rows=[], evidence={},
        rows_total=len(table.starts), rows_complete=sum(map(table.complete, table.starts)), has_more=True, byte_budget=max_bytes,
        next_cursor='', values='retained records; summarized, never interpolated')
    if state is None:
        result['completeness_history'] = 'unavailable'      # as_of_ms is older than what is kept: nothing says complete
    elif state[2] is not None:
        result['completeness_error'] = state[2]
    if base is not None:
        result['since_known_ms'] = base
        result['affected'] = query.affected(base, known, base_stop, (state or EMPTY)[1],
                                            min(MAX_AFFECTED, max_bytes // 512))
        if not_covered:
            result['not_covered'] = not_covered

    def seal(**values):
        return seal_cursor(store, dict(kind='timeline', scope=scope, lost=lost, first=first, **values))

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
        gone = store.superseded(subject_id, known)
        for entry in result['evidence'].values():
            if entry['evidence_id'] in gone:
                entry['current'] = False
    return result


def _describe(column, contracts, first, stop):
    done, contract = column.progress, contracts.get(column.name)
    newest = column.records[-1] if column.records else None
    observed = (newest or {}).get('observation', {})
    declared = observed.get('capability') or ({} if contract is None else dict(
        freshness_ms=contract.freshness_ms, interpretation=contract.interpretation, window_ms=contract.window_ms))
    result = dict(key=column.key, capability=column.name, module=done.module if done and done.module else None,
                  complete_until_ms=done.until if done else None,
                  freshness_ms=declared.get('freshness_ms', DEFAULT_FRESHNESS_MS))
    if column.processed not in (None, result['complete_until_ms']):
        result['processed_until_ms'] = column.processed   # later, but results for this time may still come
    for key, value in (('source_id', column.source), ('sensor_id', column.sensor),
                       ('interpretation', declared.get('interpretation')), ('modality', observed.get('modality'))):
        if value:
            result[key] = value
    if newest is not None:
        result['models'] = [f"{model['id']}:{model['version']}"[:200] for model in newest.get('models', [])[:2]]
    for key, value in (('sampled', done and done.sampled), ('off', done and done.off)):
        if value:
            result[key] = True
    if column.uncalibrated is not None:
        result['uncalibrated_clock'] = column.uncalibrated
    if column.clock is not None:
        result['clock'] = dict(id=column.clock.key, **column.clock.status)
    if any(not item.instant and item.end > item.start for item in column.intervals) and not declared.get('window_ms'):
        result['late_intervals'] = True     # reported when it is over: a complete row can still gain one
    gaps = [gap for gap in done.gaps if gap[0] < stop and gap[1] >= first] if done is not None else []
    if gaps:
        result['gaps'] = [dict(start_ms=gap[0], end_ms=gap[1], reason=gap[2]) for gap in gaps[-MAX_LISTED:]]
        if len(gaps) > MAX_LISTED:
            result['gaps_omitted'] = len(gaps) - MAX_LISTED
    return result
