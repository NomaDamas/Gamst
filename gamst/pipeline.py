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

A module added with ``queue=N`` runs in its own bounded lane (latest frames win, the oldest is
dropped and counted), so a slow model never blocks the input or the modules after it; this is how
an isolated worker (``gamst.crew.modules.WorkerModule``) runs next to per-frame math.
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
from dataclasses import dataclass, replace
import inspect
import time
from typing import Any

from .capabilities import SensorMetadata, identifier
from .plugins import InputFrame


@dataclass(frozen=True)
class Derived:
    """A module output that becomes a frame of another input (media stays out of context)."""
    input: str
    data: Any
    modality: str
    sample_rate: int | None = None
    timestamp_ms: int | None = None   # capture time of this item; default: the frame it came from


async def _call(function, *args):
    value = function(*args)
    return await value if inspect.isawaitable(value) else value


class Pipeline:
    def __init__(self, human, name='pipeline'):
        identifier(name, 'pipeline name')
        self.human, self.name = human, name
        self.inputs: dict[str, dict] = {}
        self.modules: dict[str, dict] = {}
        self.references: dict[str, dict] = {}
        self._sequence: dict[str, int] = {}
        self._closed = False
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
                                     derived_from=derived_from, frames=0, last_ms=None)
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

    def add_module(self, name, module, *, consumes, spec=None, enabled=True, every_ms=None, emits=(), queue=None):
        identifier(name, 'module name')
        if queue is not None and (type(queue) is not int or not 1 <= queue <= 64):
            raise ValueError('queue must be an integer from 1 to 64')
        if name in self.modules or name in self.inputs:
            raise ValueError(f'{name!r} is already an input or module')
        if not callable(getattr(module, 'process', None)):
            raise TypeError('a module needs process(frame)')
        consumes = [consumes] if isinstance(consumes, str) else list(consumes)
        if not consumes:
            raise ValueError('a module consumes at least one input or module')
        links = [tuple(item.split(':', 1)) if ':' in item and item not in self.inputs else (item, None)
                 for item in consumes]
        if spec is not None:
            self.human.register_capability(spec)
        for derived in emits:
            self.add_input(derived, derived_from=name)
        bind = getattr(module, 'bind', None)
        if callable(bind):
            bind(self)
        self.modules[name] = dict(module=module, consumes=consumes, links=links, spec=spec, enabled=bool(enabled),
                                  every_ms=every_ms, last_run_ms=None, processed=0, errors=0,
                                  last_error=None, last_latency_ms=None, needs_reset=False, emits=tuple(emits),
                                  last_admit_ms=None,
                                  lane=None if queue is None else dict(frames=deque(), capacity=queue, task=None,
                                                                       dropped=0))
        return self

    def enable(self, name):
        self._module(name)['enabled'] = True
        return self.status()['modules'][name]

    def disable(self, name):
        entry = self._module(name)
        entry['enabled'] = False
        entry['needs_reset'] = True  # what it saw before the gap must not join what it sees after
        if entry['lane'] is not None:
            entry['lane']['frames'].clear()
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
            lane = entry['lane']
            if lane is not None:
                lane['frames'].clear()
                if lane['task'] is not None and not lane['task'].done():
                    lane['task'].cancel()
                    tasks.append(lane['task'])
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
            sensor = SensorMetadata(source_id=self.name, sensor_id=input_id, sequence=sequence)
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

    async def feed(self, source, frame, *, output=None):
        """Route one frame of ``source`` (an input, or a module's ``output``) to the modules consuming it."""
        if self._closed or self.human.paused or self.human.closed or not self.human.consent:
            return []
        if source in self.inputs:
            entry = self.inputs[source]
            entry['frames'] += 1
            entry['last_ms'] = frame.timestamp_ms
        records = []
        for name, entry in self.modules.items():
            if not entry['enabled'] or not any(src == source and wanted in (None, output)
                                               for src, wanted in entry['links']):
                continue
            last = entry['last_admit_ms'] if entry['lane'] is not None else entry['last_run_ms']
            if entry['every_ms'] and last is not None and frame.timestamp_ms - last < entry['every_ms']:
                continue
            if entry['lane'] is not None:
                self._enqueue(name, entry, frame)
                continue
            records += await self._run_module(name, entry, frame)
        return records

    def _enqueue(self, name, entry, frame):
        lane = entry['lane']
        entry['last_admit_ms'] = frame.timestamp_ms
        if len(lane['frames']) >= lane['capacity']:
            lane['frames'].popleft()
            lane['dropped'] += 1
            if frame.modality == 'audio':
                entry['needs_reset'] = True  # never join audio across a dropped interval
        lane['frames'].append(frame)
        if lane['task'] is None or lane['task'].done():
            lane['task'] = asyncio.create_task(self._drain_lane(name, entry))

    async def _drain_lane(self, name, entry):
        lane = entry['lane']
        while lane['frames'] and not self._closed:
            frame = lane['frames'].popleft()
            if not entry['enabled'] or self.human.paused or self.human.closed:
                continue
            try:
                await self._run_module(name, entry, frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # counted on the module; the lane keeps running
                errors = getattr(self.human, 'plugin_errors', None)
                if isinstance(errors, list):
                    errors.append(dict(plugin=f'{self.name}:{name}', error=str(exc), timestamp_ms=frame.timestamp_ms))
                    del errors[:-100]

    async def drain(self):
        """Wait until every queued module has processed what it was given."""
        while True:
            tasks = [entry['lane']['task'] for entry in self.modules.values()
                     if entry['lane'] is not None and entry['lane']['task'] is not None
                     and not entry['lane']['task'].done()]
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_module(self, name, entry, frame):
        module = entry['module']
        if entry['needs_reset']:
            entry['needs_reset'] = False
            if callable(getattr(module, 'reset', None)):
                await _call(module.reset)
        started = time.monotonic()
        try:
            output = await _call(module.process, frame)
        except Exception as exc:
            entry['errors'] += 1
            entry['last_error'] = f'{type(exc).__name__}: {exc}'[:300]
            entry['needs_reset'] = True
            raise
        entry['last_run_ms'] = frame.timestamp_ms
        entry['processed'] += 1
        entry['last_latency_ms'] = round((time.monotonic() - started) * 1000, 3)
        return await self._commit(name, entry, frame, output)

    async def _commit(self, name, entry, frame, output):
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
                    entry['spec'].id, item, sensor=sensor, subject_id=frame.subject_id,
                    timestamp_ms=frame.timestamp_ms))
        else:
            signals = []
            for item in output:
                (derived if isinstance(item, Derived) else signals).append(item)
            if signals:
                records += [record for record in
                            await self._ingest(signals, frame, entry['module'], name) if record]
        for record in list(records):
            observation = record.get('observation') or {}
            # A record keeps its own capture time: a module running in parallel (its own process)
            # commits earlier frames' records while later frames are already flowing.
            link = self._frame(name, observation.get('value'), observation.get('modality') or frame.modality,
                               record.get('timestamp_ms', frame.timestamp_ms), subject_id=frame.subject_id,
                               sensor=frame.sensor if record.get('timestamp_ms') == frame.timestamp_ms else None)
            records += await self.feed(name, link, output=observation.get('name'))
        for item in derived:
            if item.input not in entry['emits']:
                raise ValueError(f'{name} emitted {item.input!r} but declares emits={entry["emits"]}')
            derived_frame = self._frame(item.input, item.data, item.modality,
                                        frame.timestamp_ms if item.timestamp_ms is None else item.timestamp_ms,
                                        subject_id=frame.subject_id, sample_rate=item.sample_rate,
                                        sensor=frame.sensor)
            records += await self.feed(item.input, derived_frame)
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
        False stops the run.
        """
        stop = stop or asyncio.Event()

        async def pump(input_id, source):
            async for frame in source.frames():
                if stop.is_set():
                    break
                frame = self._frame(input_id, frame.data, frame.modality, frame.timestamp_ms,
                                    subject_id=frame.subject_id, sample_rate=frame.sample_rate,
                                    sensor=frame.sensor)
                records = await self.feed(input_id, frame)
                if on_frame is not None and await _call(on_frame, input_id, frame, records) is False:
                    stop.set()
                await asyncio.sleep(0)

        tasks = [asyncio.create_task(pump(input_id, entry['source']))
                 for input_id, entry in self.inputs.items() if entry['source'] is not None]
        try:
            if tasks:
                await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
        return await self.finish()

    async def finish(self):
        """End of input: commit what stateful modules still hold (e.g. a count at the last frame)."""
        await self.drain()
        records = []
        for name, entry in self.modules.items():
            finish = getattr(entry['module'], 'finish', None)
            if not entry['enabled'] or not callable(finish):
                continue
            output = await _call(finish)
            last = entry['last_run_ms']
            if output is None or last is None or (entry['spec'] is None and not isinstance(output, list)):
                continue  # a plugin that commits its own records returns its final value only
            frame = self._frame(name, None, 'record', last)
            records += await self._commit(name, entry, frame, output)
        return records

    async def close(self):
        self._closed = True
        lanes = [entry['lane']['task'] for entry in self.modules.values()
                 if entry['lane'] is not None and entry['lane']['task'] is not None]
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

    def status(self):
        return dict(
            name=self.name,
            inputs={key: dict(modality=value['modality'], frames=value['frames'], last_ms=value['last_ms'],
                              derived_from=value['derived_from'], host_fed=value['source'] is None)
                    for key, value in self.inputs.items()},
            references={key: dict(kind=value['kind'], label=value['label']) for key, value in self.references.items()},
            modules={key: dict(enabled=value['enabled'], consumes=value['consumes'], emits=list(value['emits']),
                               capability=value['spec'].id if value['spec'] is not None else None,
                               every_ms=value['every_ms'], processed=value['processed'], errors=value['errors'],
                               last_error=value['last_error'], last_latency_ms=value['last_latency_ms'],
                               **({} if value['lane'] is None else dict(
                                   queue=value['lane']['capacity'], queued=len(value['lane']['frames']),
                                   dropped=value['lane']['dropped'])),
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
