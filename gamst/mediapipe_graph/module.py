"""A user-composed MediaPipe CalculatorGraph as one GAMST Pipeline module, in its own process.

The graph runs in ``gamst_graph_runner`` (C++, built by ``build.sh`` from a pinned MediaPipe
revision). MediaPipe schedules the graph's calculators in parallel on its own threads, so the graph
works alongside the Pipeline's PyTorch modules instead of inside their turn:

- ``process(frame)`` writes the frame to the runner and returns at once with the records of the
  frames the graph has finished so far (``wait=False``, the default), each stamped with *its own*
  capture time. ``wait=True`` returns a frame's records before the next frame is fed (offline files,
  tests, deterministic counts).
- GAMST controls the Framework: graph file, side packets (applied with a fresh graph run), branch
  switches (``enable_<branch>`` input streams, per frame, no restart), flow limiting and the
  process lifecycle. Over MCP: ``configure_pipeline_module(name, enabled, options)``.
- Timestamps: a packet's timestamp is the GAMST capture time in microseconds
  (``timestamp_ms * 1000``; ties inside one millisecond are separated by 1 us and map back to the
  exact original millisecond).
- Flow control: at most ``max_pending`` frames are inside the runner; with ``wait=False`` a frame
  beyond that is dropped here (``dropped_by='gamst'``), and the graph's FlowLimiterCalculator drops
  frames that arrive while ``max_in_flight`` frames are inside the models (``dropped_by='graph'``).
  Both appear in the ``vision.mediapipe_graph`` record, so a count can say what it did not see.
- Records: landmarks become ``scene.poses`` (the same shape as :class:`MediaPipePosePlugin`, so
  ``pose_exemplar.PoseChannels`` and the ``person-action-count`` recipe consume it unchanged),
  ``scene.hands`` and ``scene.face_landmarks``; images never leave the runner.

The runner speaks protocol v1 (``runner/gamst_graph_runner.cc``; reference in
``docs/mediapipe-graph.md``). The runner process is owned like a ProcessProvider worker: no shell,
no implicit restart after a failure, native error text kept out of context.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..process_provider import stop_child
from ..provider_wire import HEADER_LIMIT, parse_header
from ..providers.pose import MediaPipePosePlugin

HERE = Path(__file__).resolve().parent
PIN = json.loads((HERE / 'pin.json').read_text())
GRAPHS = HERE / 'graphs'
HOLISTIC_CPU = GRAPHS / 'holistic_cpu.pbtxt'
HOLISTIC_GPU = GRAPHS / 'holistic_gpu.pbtxt'
PROTOCOL = 1

HOLISTIC_SIDE_PACKETS = dict(model_complexity=1, smooth_landmarks=True, refine_face_landmarks=False,
                             use_prev_landmarks=True, max_in_flight=1)
HOLISTIC_BRANCHES = dict(pose=True, hands=True, face=True)
# record group -> the branch that produces it and the graph streams it is built from
HOLISTIC_OUTPUTS = dict(
    poses=dict(branch='pose', landmarks='pose_landmarks', world='pose_world_landmarks'),
    hands=dict(branch='hands', left='left_hand_landmarks', right='right_hand_landmarks'),
    face=dict(branch='face', landmarks='face_landmarks'),
)


class RunnerFailed(RuntimeError):
    """The runner reported a fatal error or broke the protocol; restart it explicitly."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _side_line(name, value):
    identifier(name, 'side packet name')
    if isinstance(value, tuple) and len(value) == 2:  # explicit (type, value)
        kind, value = value
    elif type(value) is bool:
        kind = 'bool'
    elif type(value) is int:
        kind = 'int'
    elif type(value) is float:
        kind = 'float'
    elif type(value) is str:
        kind = 'string'
    else:
        raise ValueError(f'side packet {name!r} must be bool, int, float, str or (type, value)')
    if kind not in ('bool', 'int', 'float', 'double', 'string'):
        raise ValueError(f'unsupported side packet type {kind!r}')
    text = ('1' if value else '0') if kind == 'bool' else str(value)
    if '\n' in text or '\r' in text or (kind != 'string' and ' ' in text) or not text:
        raise ValueError(f'side packet {name!r} value cannot be sent on one line')
    return f'SIDE {name} {kind} {text}\n'.encode()


def _point(row, digits):
    """[x, y, z, visibility|None, presence|None] -> the GAMST landmark dict."""
    x, y, z = (round(float(v), digits) if v is not None else 0. for v in row[:3])
    point = dict(x=x, y=y, z=z)
    for key, value in zip(('visibility', 'presence'), row[3:5]):
        if value is not None:
            point[key] = max(0., min(1., round(float(value), 4)))
    return point


def _points(packet, digits=6, *, scores=True):
    if packet is None or packet[0] not in ('landmarks', 'world_landmarks') or not isinstance(packet[1], list):
        return None
    points = [_point(row, digits) for row in packet[1] if isinstance(row, list) and len(row) == 5]
    if not scores:
        points = [dict(x=p['x'], y=p['y'], z=p['z']) for p in points]
    return points or None


def _box(points):
    xs, ys = [p['x'] for p in points], [p['y'] for p in points]
    x0, y0 = max(0., min(1., min(xs))), max(0., min(1., min(ys)))
    x1, y1 = max(0., min(1., max(xs))), max(0., min(1., max(ys)))
    return dict(x=x0, y=y0, width=x1 - x0, height=y1 - y0) if x1 > x0 and y1 > y0 else None


class MediaPipeGraphModule:
    """Run a MediaPipe graph in a ``gamst_graph_runner`` process; commit its outputs as records."""

    name = 'mediapipe_graph'
    modalities = ('video',)

    def __init__(self, runner, graph, *, side_packets=None, branches=None, outputs=None, streams=None,
                 wait=False, max_pending=2, resource_root=None, runner_sha256=None,
                 capability_id='vision.mediapipe_graph', devices=('cpu',), startup_timeout=120.,
                 frame_timeout=10., close_timeout=5., env=None):
        self.argv = [str(runner)] if isinstance(runner, (str, Path)) else [str(item) for item in runner]
        if not self.argv or not os.path.isfile(self.argv[0]) or not os.access(self.argv[0], os.X_OK):
            raise ValueError('runner must be an existing executable (the built gamst_graph_runner)')
        if runner_sha256 is not None and sha256_file(self.argv[0]) != runner_sha256:
            raise ValueError('runner does not match its recorded SHA-256; rebuild or update the manifest')
        self.graph = Path(graph).expanduser().resolve()
        if not self.graph.is_file():
            raise ValueError(f'graph file not found: {self.graph}')
        self.graph_sha256 = sha256_file(self.graph)
        for value in (startup_timeout, frame_timeout, close_timeout):
            if type(value) not in (int, float) or not .01 <= value <= 600:
                raise ValueError('timeouts must be 0.01..600 seconds')
        if type(max_pending) is not int or not 1 <= max_pending <= 64:
            raise ValueError('max_pending must be 1..64 frames')
        self.side_packets = dict(side_packets or {})
        for key, value in self.side_packets.items():
            _side_line(key, value)  # validate now, not at first frame
        self.branches = {identifier(k, 'branch'): bool(v) for k, v in (branches or {}).items()}
        self.outputs = dict(outputs or {})
        self.streams = {identifier(k, 'stream'): identifier(v, 'record name') for k, v in (streams or {}).items()}
        self.wait, self.max_pending = bool(wait), max_pending
        self.resource_root = Path(resource_root).expanduser().resolve() if resource_root else None
        self.startup_timeout, self.frame_timeout, self.close_timeout = startup_timeout, frame_timeout, close_timeout
        self.env = env
        self.graph_name = identifier(self.graph.stem, 'graph name')
        self.model_id = f'mediapipe.graph.{self.graph_name}'
        self.spec = CapabilitySpec(
            capability_id, 'vision.mediapipe_graph.frame', 'video', 'mediapipe.graph',
            f'{PIN["tag"]}.{self.graph_name}.sha256.{self.graph_sha256[:16]}',
            (OutputField('processed', 'boolean'), OutputField('run', 'integer'),
             OutputField('dropped_by', 'string', required=False),
             OutputField('latency_ms', 'number', unit='ms', required=False),
             OutputField('dropped', 'integer'), OutputField('branches', 'object'), OutputField('counts', 'object')),
            'frame', freshness_ms=1000, interpretation='measurement', devices=tuple(devices),
            sensor_requirements=('uint8-bgr-camera',), license='Apache-2.0')
        self._process = self._reader = None
        self._lock = asyncio.Lock()
        self._control: asyncio.Queue | None = None
        self._clear_run_state()
        self.hello, self.run, self.failed = None, 0, None
        self.dropped = dict(gamst=0, graph=0)
        self.processed = 0
        self.last_latency_ms = None
        self.frame_errors = 0

    # ------------------------------------------------------------------ constructors

    @classmethod
    def holistic(cls, runner, *, gpu=False, graph=None, side_packets=None, branches=None, **options):
        """The bundled holistic graph (pose + hands + face, smoothing, FlowLimiter)."""
        return cls(runner, graph or (HOLISTIC_GPU if gpu else HOLISTIC_CPU),
                   side_packets={**HOLISTIC_SIDE_PACKETS, **(side_packets or {})},
                   branches={**HOLISTIC_BRANCHES, **(branches or {})}, outputs=HOLISTIC_OUTPUTS,
                   devices=('gpu',) if gpu else ('cpu',), **options)

    @classmethod
    def from_manifest(cls, manifest, *, graph=None, verify=True, **options):
        """A build staged by ``build.sh``: runner, graphs and model files checked against their SHA-256."""
        manifest = Path(manifest).expanduser().resolve()
        data = json.loads(manifest.read_text())
        if data.get('kind') != 'gamst.mediapipe_graph.runner':
            raise ValueError(f'{manifest} is not a gamst_graph_runner manifest')
        root = manifest.parent
        if verify:
            for relative, expected in data['files'].items():
                if sha256_file(root / relative) != expected:
                    raise ValueError(f'{relative} does not match the manifest; rebuild with build.sh')
        gpu = data['variant'] == 'gpu'
        graph = root / 'graphs' / (graph or ('holistic_gpu.pbtxt' if gpu else 'holistic_cpu.pbtxt'))
        options.setdefault('resource_root', root / data['resource_root'])
        options['runner_sha256'] = data['runner_sha256'] if verify else None
        if graph.name.startswith('holistic_'):
            return cls.holistic(root / data['runner'], gpu=gpu, graph=graph, **options)
        return cls(root / data['runner'], graph, devices=('gpu',) if gpu else ('cpu',), **options)

    # ------------------------------------------------------------------ pipeline hooks

    def bind(self, pipeline):
        pipeline.human.register_capability(self.spec)
        self._source_id = pipeline.name

    async def process(self, frame):
        if self.failed:
            raise RunnerFailed(f'mediapipe graph runner failed ({self.failed}); '
                               "restart it with configure({'restart': True})")
        image = frame.data
        if getattr(image, 'dtype', None) is None or str(image.dtype) != 'uint8' or getattr(image, 'ndim', 0) != 3 \
                or image.shape[2] != 3:
            raise ValueError('MediaPipeGraphModule takes uint8 BGR frames of shape (height, width, 3)')
        if self._process is None:
            await self.start()
        ready = self._take()
        base_us = int(frame.timestamp_ms) * 1000
        us = base_us if self._last_us is None or base_us > self._last_us else self._last_us + 1
        if us - base_us >= 1000:
            raise ValueError('more than 1000 frames inside one millisecond')
        if len(self._pending) >= self.max_pending:
            if not self.wait:
                self.dropped['gamst'] += 1
                return ready + [self._frame_record(self._entry(frame), processed=False, dropped_by='gamst')]
            await self._wait_for(next(iter(self._pending.values()))['future'])
        entry = self._entry(frame)
        entry['future'] = asyncio.get_running_loop().create_future()
        height, width = image.shape[:2]
        import numpy as np
        payload = np.ascontiguousarray(image).tobytes()
        async with self._lock:
            self._check()
            entry['sent'] = time.monotonic()
            self._pending[us] = entry
            self._last_us = us
            await self._write(f'FRAME {us} {width} {height}\n'.encode() + payload)
        if self.wait:
            await self._wait_for(entry['future'])
        return ready + self._take()

    async def finish(self):
        """End of input: wait for the frames still inside the graph, return their records."""
        if self._process is not None and not self.failed:
            for entry in list(self._pending.values()):
                await self._wait_for(entry['future'])
        return self._take()

    async def reset(self):
        """A fresh graph run (Pipeline calls this after the module was switched off and on)."""
        if self._process is None or self.failed:
            return
        async with self._lock:
            await self._restart_run()

    async def configure(self, options):
        """``branches``: {name: bool} (next frame); ``side_packets``: {name: value} (fresh run);
        ``max_pending``, ``wait``; ``restart``: True starts a new runner after a failure."""
        options = dict(options or {})
        unknown = set(options) - {'branches', 'side_packets', 'max_pending', 'wait', 'restart'}
        if unknown:
            raise ValueError(f'unknown mediapipe_graph options: {sorted(unknown)}')
        if options.get('restart'):
            await self.close()
            self.failed = None
            self._closed = False
        for name, enabled in (options.get('branches') or {}).items():
            if type(enabled) is not bool:
                raise ValueError('branch switches are booleans')
            if self.hello is not None and name not in self.hello['branches']:
                raise ValueError(f'graph has no branch {name!r}; branches: {self.hello["branches"]}')
            identifier(name, 'branch')
        sides = options.get('side_packets') or {}
        lines = [_side_line(name, value) for name, value in sides.items()]
        if 'max_pending' in options:
            if type(options['max_pending']) is not int or not 1 <= options['max_pending'] <= 64:
                raise ValueError('max_pending must be 1..64 frames')
            self.max_pending = options['max_pending']
        if 'wait' in options:
            self.wait = bool(options['wait'])
        self.side_packets.update(sides)
        self.branches.update(options.get('branches') or {})
        if self._process is not None and not self.failed:
            async with self._lock:
                for name, enabled in (options.get('branches') or {}).items():
                    await self._write(f'BRANCH {name} {int(enabled)}\n'.encode())
                    await self._expect('ack')
                for line in lines:
                    await self._write(line)
                    await self._expect('ack')
                if lines:
                    await self._restart_run()
        return self.status()

    async def set_branch(self, name, enabled):
        return await self.configure({'branches': {name: bool(enabled)}})

    def status(self):
        return dict(graph=self.graph_name, graph_sha256=self.graph_sha256, running=self._process is not None,
                    run=self.run, failed=self.failed, pending=len(self._pending), processed=self.processed,
                    dropped=dict(self.dropped), frame_errors=self.frame_errors, branches=dict(self.branches),
                    side_packets={k: v if not isinstance(v, tuple) else v[1] for k, v in self.side_packets.items()},
                    max_pending=self.max_pending, wait=self.wait, last_latency_ms=self.last_latency_ms,
                    mediapipe=(self.hello or {}).get('mediapipe'), gpu=(self.hello or {}).get('gpu'))

    # ------------------------------------------------------------------ process lifecycle

    async def start(self):
        if self._process is not None:
            return self
        if self.failed:
            raise RunnerFailed(f'mediapipe graph runner failed ({self.failed})')
        argv = self.argv + [f'--graph={self.graph}']
        if self.resource_root is not None:
            argv.append(f'--resource_root_dir={self.resource_root}')
        self._control = asyncio.Queue()
        self._closed = False
        self._process = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, close_fds=True, limit=4 * HEADER_LIMIT,
            cwd=str(self.resource_root) if self.resource_root else None,
            env=None if self.env is None else {**os.environ, **self.env})
        self._reader = asyncio.create_task(self._read(self._process))
        try:
            hello = await self._expect('hello', timeout=self.startup_timeout)
            if hello.get('protocol') != PROTOCOL:
                raise RunnerFailed('protocol')
            missing = set(self.branches) - set(hello.get('branches', []))
            if missing:
                raise ValueError(f'graph has no branches {sorted(missing)}; it declares {hello.get("branches")}')
            self.hello = hello
            async with self._lock:
                for name, value in self.side_packets.items():
                    await self._write(_side_line(name, value))
                    await self._expect('ack')
                for name, enabled in self.branches.items():
                    await self._write(f'BRANCH {name} {int(enabled)}\n'.encode())
                    await self._expect('ack')
                await self._write(b'START\n')
                await self._ready(timeout=self.startup_timeout)
        except BaseException:
            await self.close()
            raise
        return self

    async def close(self):
        process, self._process = self._process, None
        self._closed = True
        if process is None:
            return
        try:
            if not self.failed and process.returncode is None:
                try:
                    process.stdin.write(b'CLOSE\n')
                    await asyncio.wait_for(process.stdin.drain(), self.close_timeout)
                    await self._expect('closed', timeout=self.close_timeout)
                except (OSError, asyncio.TimeoutError, RunnerFailed, ConnectionError):
                    pass
        finally:
            await stop_child(process)
            if self._reader is not None:
                self._reader.cancel()
                await asyncio.gather(self._reader, return_exceptions=True)
                self._reader = None
            for entry in self._pending.values():
                if not entry['future'].done():
                    entry['future'].cancel()
            self._clear_run_state(keep_ready=True)

    # ------------------------------------------------------------------ internals

    def _clear_run_state(self, keep_ready=False):
        self._pending: dict[int, dict] = {}
        self._packets: dict[int, dict] = {}
        if not keep_ready:
            self._ready: list[dict] = []
        self._last_us = None

    def _check(self):
        if self.failed:
            raise RunnerFailed(f'mediapipe graph runner failed ({self.failed})')
        if self._process is None:
            raise RunnerFailed('runner is not running')

    async def _write(self, data):
        try:
            self._process.stdin.write(data)
            await asyncio.wait_for(self._process.stdin.drain(), self.frame_timeout)
        except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
            self._fail(f'write:{type(exc).__name__}')
            raise RunnerFailed(self.failed) from None

    async def _expect(self, kind, timeout=None):
        """Next control message of ``kind``; a non-fatal error answering a command raises."""
        deadline = time.monotonic() + (timeout or self.frame_timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._fail(f'timeout:{kind}')
                raise RunnerFailed(self.failed)
            try:
                message = await asyncio.wait_for(self._control.get(), remaining)
            except asyncio.TimeoutError:
                continue
            if message.get('type') == 'eof':
                raise RunnerFailed(self.failed or 'exited')
            if message.get('type') == 'error':
                raise ValueError(f'runner rejected the command ({message.get("stage")}: {message.get("code")})')
            if message.get('type') == kind:
                return message

    async def _ready(self, timeout=None):
        message = await self._expect('ready', timeout=timeout)
        self.run = int(message['run'])
        self._last_us = None
        return message

    async def _restart_run(self):
        self._check()
        await self._write(b'RESET\n')
        await self._ready(timeout=self.startup_timeout)

    async def _wait_for(self, future):
        try:
            await asyncio.wait_for(asyncio.shield(future), self.frame_timeout)
        except asyncio.TimeoutError:
            self._fail('timeout:frame')
            raise RunnerFailed(self.failed) from None
        except asyncio.CancelledError:
            if not future.cancelled():
                raise
        if self.failed:
            raise RunnerFailed(self.failed)

    def _fail(self, code):
        if self.failed is None:
            self.failed = str(code)[:64]
        for entry in self._pending.values():
            if not entry['future'].done():
                entry['future'].set_result(None)

    async def _read(self, process):
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    raise ValueError('truncated line')
                message = parse_header(line)
                if type(message) is not dict or not isinstance(message.get('type'), str):
                    raise ValueError('not a protocol message')
                self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(f'protocol:{type(exc).__name__}')
        finally:
            if not self._closed and self.failed is None and self._process is process:
                self._fail('exited')
            if self._control is not None:
                self._control.put_nowait(dict(type='eof'))

    def _dispatch(self, message):
        kind = message['type']
        if kind == 'packet':
            if message.get('run') == self.run and type(message.get('ts_us')) is int:
                stream = message.get('stream')
                if isinstance(stream, str):
                    self._packets.setdefault(message['ts_us'], {})[stream] = (message.get('kind'), message.get('value'))
        elif kind == 'done':
            if message.get('run') == self.run and type(message.get('ts_us')) is int:
                self._done(message['ts_us'], message.get('processed') is True)
        elif kind == 'error':
            if message.get('fatal'):
                self._fail(f'{message.get("stage")}:{message.get("code")}')
            elif message.get('stage') == 'frame':
                self.frame_errors += 1
            else:
                self._control.put_nowait(message)
        else:
            self._control.put_nowait(message)

    def _done(self, ts_us, processed):
        """Every frame at or before ``ts_us`` is finished; only the frame at ``ts_us`` can have outputs."""
        for us in [key for key in self._pending if key <= ts_us]:
            entry = self._pending.pop(us)
            packets = self._packets.pop(us, {})
            done = processed and us == ts_us
            if done:
                self.processed += 1
                self.last_latency_ms = round((time.monotonic() - entry['sent']) * 1000, 3)
                self._ready += self._records(entry, packets)
            else:
                self.dropped['graph'] += 1
                self._ready.append(self._frame_record(entry, processed=False, dropped_by='graph'))
            if not entry['future'].done():
                entry['future'].set_result(None)
        for stale in [key for key in self._packets if key <= ts_us]:
            del self._packets[stale]

    def _take(self):
        ready, self._ready = self._ready, []
        return ready

    def _entry(self, frame):
        sensor = frame.sensor
        if not isinstance(sensor, SensorMetadata):
            sensor = SensorMetadata(source_id=getattr(self, '_source_id', 'mediapipe_graph'),
                                    sensor_id='camera', sequence=self.processed)
        return dict(ms=int(frame.timestamp_ms), subject_id=frame.subject_id, sensor=sensor,
                    branches=dict(self.branches), sent=time.monotonic(), future=None)

    def _signal(self, entry, name, value):
        return dict(name=name, value=value, modality='video', timestamp_ms=entry['ms'], subject_id=entry['subject_id'],
                    model_id=self.model_id, model_version=self.spec.version, quality='sufficient')

    def _frame_record(self, entry, *, processed, dropped_by=None, counts=None, latency_ms=None):
        value = dict(processed=processed, run=self.run, dropped=self.dropped['gamst'] + self.dropped['graph'],
                     branches=entry['branches'], counts=counts or {})
        if dropped_by:
            value['dropped_by'] = dropped_by
        if latency_ms is not None:
            value['latency_ms'] = latency_ms
        return dict(name=self.spec.output_name, value=value, modality='video', timestamp_ms=entry['ms'],
                    subject_id=entry['subject_id'], model_id=self.spec.provider, model_version=self.spec.version,
                    capability_id=self.spec.id, sensor=entry['sensor'])

    def _records(self, entry, packets):
        """One finished frame -> GAMST records (a switched-off branch commits nothing)."""
        signals, counts, on = [], {}, entry['branches']
        poses_out = self.outputs.get('poses')
        if poses_out and on.get(poses_out['branch'], True):
            poses = []
            points = _points(packets.get(poses_out['landmarks']))
            if points is not None and len(points) == 33:
                for point in points:
                    point.setdefault('visibility', 0.)
                    point.setdefault('presence', 0.)
                pose = dict(frame_pose_index=0, landmarks=points, bbox=MediaPipePosePlugin._bbox(points),
                            geometry=MediaPipePosePlugin._gestures(points), track_id=None,
                            association_quality='unavailable', identity_verified=False,
                            speaker_association='unavailable')
                world = _points(packets.get(poses_out.get('world')), scores=False)
                if world is not None:
                    pose['world_landmarks'] = world  # meters, hip-centred
                poses.append(pose)
            counts['poses'] = len(poses)
            signals += [self._signal(entry, 'scene.poses', poses), self._signal(entry, 'scene.pose_count', len(poses))]
        hands_out = self.outputs.get('hands')
        if hands_out and on.get(hands_out['branch'], True):
            hands = []
            for side in ('left', 'right'):
                points = _points(packets.get(hands_out[side]), 5)
                if points is not None:
                    hands.append(dict(handedness=side, landmarks=points, bbox=_box(points)))
            counts['hands'] = len(hands)
            signals += [self._signal(entry, 'scene.hands', hands), self._signal(entry, 'scene.hand_count', len(hands))]
        face_out = self.outputs.get('face')
        if face_out and on.get(face_out['branch'], True):
            faces = []
            points = _points(packets.get(face_out['landmarks']), 5, scores=False)
            if points is not None:
                faces.append(dict(frame_face_index=0, landmarks=points, bbox=_box(points)))
            counts['faces'] = len(faces)
            signals += [self._signal(entry, 'scene.face_landmarks', faces),
                        self._signal(entry, 'scene.face_count', len(faces))]
        for stream, record in self.streams.items():
            if stream in packets:
                kind, value = packets[stream]
                if kind in ('landmarks', 'world_landmarks'):
                    value = _points((kind, value))
                elif kind == 'landmarks_list':
                    value = [_points(('landmarks', item)) for item in value]
                elif kind in ('image', 'opaque'):
                    continue  # never media, never an untyped blob
                signals.append(self._signal(entry, record, value))
        return [self._frame_record(entry, processed=True, counts=counts, latency_ms=self.last_latency_ms)] + signals
