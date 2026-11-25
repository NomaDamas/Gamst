"""A user-composed MediaPipe CalculatorGraph as one GAMST Pipeline module, in its own process.

The graph runs in ``gamst_graph_runner`` (C++, built by ``build.sh`` from a pinned MediaPipe
revision). Give the module its own bounded Pipeline lane (``add_module(..., queue=N)``) and it runs
in parallel with, and independent of, the other modules. The lane is the only GAMST-side queue:

- at most ``max_in_flight`` frames (the graph's FlowLimiter side packet) are inside the runner.
  ``process`` sends a frame only when there is room, so a slow graph builds its backlog in
  the lane (``backlog`` in the Pipeline status), not somewhere unseen. With ``max_in_flight=1`` a frame's
  records come back from the call that fed it; with more, calls return the records of frames that
  finished meanwhile, each with its own capture time.
- timestamps: a packet's timestamp is the GAMST capture time in microseconds (``timestamp_ms *
  1000``; ties inside one millisecond are separated by 1 us and map back to the exact millisecond).
- control: graph file, side packets (a fresh graph run), branch switches (``enable_<branch>`` input
  streams, from the next frame, no restart), process start/reset/stop. Over MCP:
  ``configure_pipeline_module(name, enabled, options)``, limited to what the owner declared here.
- records: ``scene.poses``, ``scene.hands`` and ``scene.face_landmarks`` with boxes, geometry and how many points
  were found, never where; of a custom landmark stream only the number of points. The points themselves stay in
  this module (for the local screen) and leave it only on the derived inputs named in ``derived``
  (:class:`gamst.private_payload.PrivatePoses`, the shape ``pose_exemplar.PoseChannels`` reads), which a Pipeline
  never commits. Images never leave the runner.

The runner is owned like a ProcessProvider worker: no shell, no implicit restart after a failure,
native error text kept out of context. Protocol reference: docs/mediapipe-graph.md.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import time

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..pipeline import Derived
from ..plugins import InputFrame
from ..private_payload import PrivatePoses, public_pose
from ..process_provider import stop_child
from ..provider_wire import HEADER_LIMIT, parse_header
from ..providers.pose import MediaPipePosePlugin
from ..streams import _finish
from ..visualization import drawing_of

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


def _side_kind(name, value):
    """The (type, value) a side packet is sent as; a (type, value) tuple names it explicitly."""
    if isinstance(value, tuple) and len(value) == 2:
        kind, value = value
    else:
        kind = {bool: 'bool', int: 'int', float: 'float', str: 'string'}.get(type(value))
    expected = {'bool': bool, 'int': int, 'float': (int, float), 'double': (int, float), 'string': str}.get(kind)
    if expected is None or not isinstance(value, expected) or (kind != 'bool' and type(value) is bool):
        raise ValueError(f'side packet {name!r} must be bool, int, float, str or (type, value)')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'side packet {name!r} must be finite')
    return kind, value


def _side_line(name, value):
    identifier(name, 'side packet name')
    kind, value = _side_kind(name, value)
    text = str(int(value)) if kind == 'bool' else str(value)
    if not text or '\n' in text or '\r' in text:
        raise ValueError(f'side packet {name!r} value cannot be sent on one line')
    return f'SIDE {name} {kind} {text}\n'.encode()


def _point(row, digits):
    """[x, y, z, visibility|None, presence|None] -> the GAMST landmark dict."""
    x, y, z = (round(float(v), digits) for v in row[:3])
    point = dict(x=x, y=y, z=z)
    for key, value in zip(('visibility', 'presence'), row[3:5]):
        if value is not None:
            point[key] = max(0., min(1., round(float(value), 4)))
    return point


def _points(packet, digits=6, *, scores=True):
    if packet is None or packet[0] not in ('landmarks', 'world_landmarks') or not isinstance(packet[1], list):
        return None
    if any(not isinstance(row, list) or len(row) != 5 or None in row[:3] for row in packet[1]):
        return None  # a non-finite coordinate is sent as null: no landmarks rather than invented ones
    points = [_point(row, digits) for row in packet[1]]
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
    owns_process = True  # a module in its own process belongs in a bounded lane (queue=N)

    def __init__(self, runner, graph, *, side_packets=None, branches=None, outputs=None, streams=None,
                 resource_root=None, runner_sha256=None, revision=None, capability_id='vision.mediapipe_graph',
                 derived=None, devices=('cpu',), startup_timeout=120., frame_timeout=10., close_timeout=5., env=None):
        self.argv = [str(runner)] if isinstance(runner, (str, Path)) else [str(item) for item in runner]
        if not self.argv or not os.path.isfile(self.argv[0]) or not os.access(self.argv[0], os.X_OK):
            raise ValueError('runner must be an existing executable (the built gamst_graph_runner)')
        if runner_sha256 is not None:
            from ..crew.setup import verified_file
            verified_file(self.argv[0], runner_sha256)
        self.graph = Path(graph).expanduser().resolve()
        if not self.graph.is_file():
            raise ValueError(f'graph file not found: {self.graph}')
        for value in (startup_timeout, frame_timeout, close_timeout):
            if type(value) not in (int, float) or not .01 <= value <= 600:
                raise ValueError('timeouts must be 0.01..600 seconds')
        self.side_packets = dict(side_packets or {})
        for key, value in self.side_packets.items():
            _side_line(key, value)  # validate now, not at the first frame
        self._check_in_flight(self.side_packets)
        self.branches = {identifier(k, 'branch'): bool(v) for k, v in (branches or {}).items()}
        self.outputs = dict(outputs or {})
        # {'poses' | 'hands' | 'face' | a landmark stream: derived input}
        self.derived = {identifier(k, 'output'): identifier(v, 'derived input') for k, v in (derived or {}).items()}
        self.streams = {identifier(k, 'stream'): identifier(v, 'record name') for k, v in (streams or {}).items()}
        self.resource_root = Path(resource_root).expanduser().resolve() if resource_root else None
        self.startup_timeout, self.frame_timeout, self.close_timeout = startup_timeout, frame_timeout, close_timeout
        self.env, self.revision = env, revision
        self.graph_name = identifier(self.graph.stem, 'graph name')
        self.graph_sha256 = hashlib.sha256(self.graph.read_bytes()).hexdigest()
        self.model_id = f'mediapipe.graph.{self.graph_name}'
        source = PIN['tag'] if revision == PIN['commit'] else 'unverified-build'
        self.spec = CapabilitySpec(
            capability_id, 'vision.mediapipe_graph.frame', 'video', 'mediapipe.graph',
            f'{source}.{self.graph_name}.sha256.{self.graph_sha256[:16]}',
            (OutputField('processed', 'boolean'), OutputField('run', 'integer'),
             OutputField('latency_ms', 'number', unit='ms', required=False), OutputField('dropped', 'integer'),
             OutputField('branches', 'object'), OutputField('counts', 'object')),
            'frame', freshness_ms=1000, interpretation='measurement', devices=tuple(devices),
            sensor_requirements=('uint8-bgr-camera',), license='Apache-2.0')
        self._process = self._reader = self._control = None
        self._lock = asyncio.Lock()
        self._records_ready: list[dict] = []
        self._pending: dict[int, dict] = {}   # capture us -> frame inside the runner
        self._packets: dict[int, dict] = {}   # capture us -> {stream: (kind, value)}
        self._last_us = None
        self._closed = False
        self.hello, self.run, self.failed = None, 0, None
        self.processed = self.dropped = 0     # dropped: by the graph's FlowLimiter
        self.last_latency_ms = None

    # ------------------------------------------------------------------ constructors

    @classmethod
    def holistic(cls, runner, *, gpu=False, graph=None, side_packets=None, branches=None, **options):
        """The bundled holistic graph (pose + hands + face, smoothing, FlowLimiter)."""
        return cls(runner, graph or (HOLISTIC_GPU if gpu else HOLISTIC_CPU),
                   side_packets={**HOLISTIC_SIDE_PACKETS, **(side_packets or {})},
                   branches={**HOLISTIC_BRANCHES, **(branches or {})}, outputs=HOLISTIC_OUTPUTS,
                   devices=('gpu',) if gpu else ('cpu',), **options)

    @classmethod
    def from_manifest(cls, manifest, *, graph=None, **options):
        """A build staged by ``build.sh``: revision, runner, graphs and model files are all checked."""
        from ..crew.setup import verified_file
        manifest = Path(manifest).expanduser().resolve()
        data = json.loads(manifest.read_text())
        if data.get('kind') != 'gamst.mediapipe_graph.runner':
            raise ValueError(f'{manifest} is not a gamst_graph_runner manifest')
        if data.get('mediapipe_commit') != PIN['commit']:
            raise ValueError('the runner was built from another MediaPipe revision than pin.json; rebuild it')
        root, gpu = manifest.parent, data['variant'] == 'gpu'
        for relative, expected in data['files'].items():
            verified_file(root / relative, expected)
        name = graph or ('holistic_gpu.pbtxt' if gpu else 'holistic_cpu.pbtxt')
        if f'graphs/{name}' not in data['files']:
            raise ValueError(f'the manifest has no graph {name!r}')
        options.update(resource_root=root / data['resource_root'], runner_sha256=data['runner_sha256'],
                       revision=data['mediapipe_commit'])
        build = cls.holistic if name.startswith('holistic_') else cls
        if build is cls:
            options.setdefault('devices', ('gpu',) if gpu else ('cpu',))
        else:
            options['gpu'] = gpu
        return build(root / data['runner'], graph=root / 'graphs' / name, **options)

    # ------------------------------------------------------------------ pipeline hooks

    def bind(self, pipeline):
        pipeline.human.register_capability(self.spec)

    async def process(self, frame):
        self._check()
        image = frame.data
        if str(getattr(image, 'dtype', '')) != 'uint8' or getattr(image, 'ndim', 0) != 3 or image.shape[2] != 3:
            raise ValueError('MediaPipeGraphModule takes uint8 BGR frames of shape (height, width, 3)')
        if self._process is None:
            await self.start()
        await self._make_room()
        height, width = image.shape[:2]
        async with self._lock:
            self._check()
            base_us = int(frame.timestamp_ms) * 1000
            us = base_us if self._last_us is None or base_us > self._last_us else self._last_us + 1
            if not 0 <= us - base_us < 1000:
                raise ValueError('frames must arrive in capture order')
            sensor = frame.sensor if isinstance(frame.sensor, SensorMetadata) else \
                SensorMetadata(source_id='mediapipe_graph', sensor_id='camera', sequence=self.processed)
            self._last_us = us
            self._pending[us] = dict(ms=int(frame.timestamp_ms), subject_id=frame.subject_id, sensor=sensor,
                                     branches=dict(self.branches), sent=time.monotonic(),
                                     future=asyncio.get_running_loop().create_future())
            # One write call: a cancelled drain can never leave half a frame on the pipe.
            await self._write(f'FRAME {us} {width} {height}\n'.encode() + image.tobytes(order='C'))
        await self._make_room()
        return self._take()

    @property
    def pending_since_ms(self):
        """The oldest frame still inside the graph (the Pipeline does not call it complete yet)."""
        return min((entry['ms'] for entry in self._pending.values()), default=None)

    async def finish(self):
        """End of input: wait for the frames still inside the graph, return their records."""
        if self._process is not None and not self.failed:
            for entry in list(self._pending.values()):
                await self._wait_for(entry['future'])
        return self._take()

    async def reset(self):
        """A fresh graph run after a capture gap; what was seen before it is not committed after it."""
        if self._process is not None and not self.failed:
            await self._uninterrupted(self._restart_run())
        self._records_ready.clear()
        drawing_of(self).clear()

    async def configure(self, options):
        """``branches``: {name: bool} (next frame); ``side_packets``: {name: value} (fresh run);
        ``restart``: True starts a new runner after a failure. Only branches and side packets declared
        at construction can change, and a side packet keeps its type (an agent over MCP cannot
        introduce a new one, such as a file path)."""
        options = dict(options or {})
        unknown = set(options) - {'branches', 'side_packets', 'restart'}
        if unknown:
            raise ValueError(f'unknown mediapipe_graph options: {sorted(unknown)}')
        branches, sides = dict(options.get('branches') or {}), dict(options.get('side_packets') or {})
        for name, enabled in branches.items():
            if name not in self.branches or type(enabled) is not bool:
                raise ValueError(f'branch switches are booleans for {sorted(self.branches)}')
        for name, value in sides.items():
            if name not in self.side_packets:
                raise ValueError(f'side packets declared here: {sorted(self.side_packets)}')
            kind, _ = _side_kind(name, self.side_packets[name])
            if kind == 'string':
                raise ValueError('text side packets are fixed at construction')
            sides[name] = (kind, value)
            _side_line(name, sides[name])
        self._check_in_flight(sides)
        if options.get('restart'):
            await self.close()
            self.failed = None
        await self._uninterrupted(self._apply(branches, sides))
        return self.status()

    async def _apply(self, branches, sides):
        async with self._lock:
            running = self._process is not None and self.hello is not None and not self.failed
            for name, enabled in branches.items():
                if running:
                    await self._write(f'BRANCH {name} {int(enabled)}\n'.encode())
                    await self._expect('ack')
                self.branches[name] = enabled
            for name, value in sides.items():
                if running:
                    await self._write(_side_line(name, value))
                    await self._expect('ack')
                self.side_packets[name] = value
            if sides and running:
                await self._restart_locked()

    def status(self):
        def shown(name, value):  # numbers and switches; text (it may be a local path) stays private
            kind, value = _side_kind(name, value)
            return '<text>' if kind == 'string' else value
        return dict(graph=self.graph_name, graph_sha256=self.graph_sha256, running=self._process is not None,
                    run=self.run, failed=self.failed, in_flight=len(self._pending), processed=self.processed,
                    dropped_by_graph=self.dropped, branches=dict(self.branches),
                    side_packets={k: shown(k, v) for k, v in self.side_packets.items()},
                    last_latency_ms=self.last_latency_ms, mediapipe=(self.hello or {}).get('mediapipe'),
                    gpu=(self.hello or {}).get('gpu'))

    # ------------------------------------------------------------------ process lifecycle

    async def start(self):
        if self._process is not None:
            return self
        self._check()
        argv = self.argv + [f'--graph={self.graph}']
        if self.resource_root is not None:
            argv.append(f'--resource_root_dir={self.resource_root}')
        self._control, self._closed = asyncio.Queue(), False
        self._last_us = None
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, close_fds=True, limit=4 * HEADER_LIMIT,
            cwd=str(self.resource_root) if self.resource_root else None,
            env=None if self.env is None else {**os.environ, **self.env}))
        try:
            self._process = await asyncio.shield(spawn)
            self._reader = asyncio.create_task(self._read(self._process))
            hello = await self._expect('hello', timeout=self.startup_timeout)
            if hello.get('protocol') != PROTOCOL:
                raise RunnerFailed('the runner speaks another protocol version')
            if self.revision is not None and hello.get('mediapipe') != self.revision:
                raise RunnerFailed('the runner was built from another MediaPipe revision than its manifest')
            missing = set(self.branches) - set(hello.get('branches') or ())
            missing |= set(self.side_packets) - set(hello.get('side_packets') or ())
            if missing:
                raise ValueError(f'the graph does not declare {sorted(missing)}')
            async with self._lock:
                for name, value in self.side_packets.items():
                    await self._write(_side_line(name, value))
                    await self._expect('ack')
                for name, enabled in self.branches.items():
                    await self._write(f'BRANCH {name} {int(enabled)}\n'.encode())
                    await self._expect('ack')
                await self._write(b'START\n')
                self.run = int((await self._expect('ready', timeout=self.startup_timeout))['run'])
                self.hello = hello
        except BaseException:
            if self._process is None:
                try:
                    self._process = await _finish(spawn)
                except Exception:
                    pass
            await self.close()
            raise
        return self

    async def close(self):
        process, self._process = self._process, None
        self._closed, self.hello = True, None
        if process is None:
            return
        try:
            if not self.failed and process.returncode is None:
                try:
                    process.stdin.write(b'CLOSE\n')
                    await asyncio.wait_for(process.wait(), self.close_timeout)
                except (OSError, asyncio.TimeoutError):
                    pass
        finally:
            await _finish(asyncio.create_task(stop_child(process)))
            reader, self._reader = self._reader, None
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            for entry in self._pending.values():
                if not entry['future'].done():
                    entry['future'].set_result(None)
            self._pending.clear()
            self._packets.clear()
            drawing_of(self).clear()

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _check_in_flight(sides):
        if 'max_in_flight' in sides:
            kind, value = _side_kind('max_in_flight', sides['max_in_flight'])
            if kind != 'int' or not 1 <= value <= 16:
                raise ValueError('max_in_flight must be 1..16 frames')

    @property
    def max_in_flight(self):
        return _side_kind('max_in_flight', self.side_packets.get('max_in_flight', 1))[1]

    @staticmethod
    async def _uninterrupted(work):
        """Finish a command/answer exchange even if its caller is cancelled, then pass the cancel on."""
        task = asyncio.ensure_future(work)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task, return_exceptions=True)
            raise

    def _check(self):
        if self.failed:
            raise RunnerFailed(f"mediapipe graph runner failed ({self.failed}); restart it with "
                               "configure({'restart': True})")

    async def _make_room(self):
        while len(self._pending) >= self.max_in_flight:
            await self._wait_for(next(iter(self._pending.values()))['future'])

    async def _write(self, data):
        process = self._process
        if process is None:
            raise RunnerFailed('the runner is not running')
        try:
            process.stdin.write(data)
            await asyncio.wait_for(process.stdin.drain(), self.frame_timeout)
        except (OSError, asyncio.TimeoutError) as exc:
            self._fail(f'write:{type(exc).__name__}')
            self._check()

    async def _expect(self, kind, timeout=None):
        """The next answer of ``kind``; a non-fatal error answering a command raises ValueError."""
        deadline = time.monotonic() + (timeout or self.frame_timeout)
        while True:
            try:
                message = await asyncio.wait_for(self._control.get(), max(0, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                self._fail(f'timeout:{kind}')
                self._check()
            if message['type'] == 'eof':
                self._check()
                raise RunnerFailed('the runner was closed')
            if message['type'] == 'error':
                raise ValueError(f'the runner rejected a command ({message.get("stage")}: {message.get("code")})')
            if message['type'] == kind:
                return message

    async def _restart_run(self):
        async with self._lock:
            await self._restart_locked()

    async def _restart_locked(self):
        self._check()
        await self._write(b'RESET\n')
        # The runner finishes the old run (every done line) before it announces the new one.
        self.run = int((await self._expect('ready', timeout=self.startup_timeout))['run'])
        self._last_us = None

    async def _wait_for(self, future):
        done, _ = await asyncio.wait({future}, timeout=self.frame_timeout)
        if not done:
            self._fail('timeout:frame')
        self._check()

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
        except Exception as exc:  # LimitOverrunError, bad JSON, bad fields
            self._fail(f'protocol:{type(exc).__name__}')
        finally:
            if not self._closed:
                self._fail('exited')
            self._control.put_nowait(dict(type='eof'))

    def _dispatch(self, message):
        kind = message['type']
        if kind in ('packet', 'done'):
            run, us = message['run'], message['ts_us']
            if type(run) is not int or type(us) is not int:
                raise ValueError('run and ts_us are integers')
            if run != self.run:
                return
            if kind == 'done':
                self._done(us, message['processed'] is True)
            elif us in self._pending:  # only frames GAMST sent
                self._packets.setdefault(us, {})[str(message['stream'])] = (message['kind'], message['value'])
        elif kind == 'error' and (message.get('fatal') or message.get('stage') == 'frame'):
            self._fail(f'{message.get("stage")}:{message.get("code")}')
        else:
            self._control.put_nowait(message)

    def _done(self, ts_us, processed):
        """Every frame at or before ``ts_us`` is finished; only the frame at ``ts_us`` can have outputs."""
        for us in [key for key in self._pending if key <= ts_us]:
            entry = self._pending.pop(us)
            packets = self._packets.pop(us, {})
            if processed and us == ts_us:
                self.processed += 1
                self.last_latency_ms = round((time.monotonic() - entry['sent']) * 1000, 3)
                try:
                    self._records_ready += self._records(entry, packets)
                finally:
                    if not entry['future'].done():
                        entry['future'].set_result(None)
                continue
            self.dropped += 1
            self._records_ready.append(self._frame_record(entry, processed=False))
            if not entry['future'].done():
                entry['future'].set_result(None)

    def _take(self):
        ready, self._records_ready = self._records_ready, []
        return ready

    def _signal(self, entry, name, value):
        return dict(name=name, value=value, modality='video', timestamp_ms=entry['ms'], subject_id=entry['subject_id'],
                    model_id=self.model_id, model_version=self.spec.version, quality='sufficient')

    def _frame_record(self, entry, *, processed, counts=None, latency_ms=None):
        value = dict(processed=processed, run=self.run, dropped=self.dropped, branches=entry['branches'],
                     counts=counts or {})
        if latency_ms is not None:
            value['latency_ms'] = latency_ms
        return dict(name=self.spec.output_name, value=value, modality='video', timestamp_ms=entry['ms'],
                    subject_id=entry['subject_id'], model_id=self.spec.provider, model_version=self.spec.version,
                    capability_id=self.spec.id, sensor=entry['sensor'])

    def _records(self, entry, packets):
        """One finished frame -> GAMST records (a switched-off branch commits nothing)."""
        signals, counts, on = [], {}, entry['branches']
        frame = InputFrame(None, 'video', entry['ms'], subject_id=entry['subject_id'])

        def public(group, name, items):
            """The record of ``items`` without their points, which go to the screen and the derived input."""
            if name == 'scene.poses':       # the only one of them a screen draws
                drawing_of(self).put(name, frame, [item['landmarks'] for item in items])
            if group in self.derived:
                signals.append(Derived(self.derived[group], PrivatePoses(items), 'video', timestamp_ms=entry['ms']))
            return self._signal(entry, name, [dict(public_pose(item), landmark_count=len(item['landmarks'])) for item in items])

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
            signals += [public('poses', 'scene.poses', poses), self._signal(entry, 'scene.pose_count', len(poses))]
        hands_out = self.outputs.get('hands')
        if hands_out and on.get(hands_out['branch'], True):
            hands = []
            for side in ('left', 'right'):
                points = _points(packets.get(hands_out[side]), 5)
                if points is not None:
                    hands.append(dict(handedness=side, landmarks=points, bbox=_box(points)))
            counts['hands'] = len(hands)
            signals += [public('hands', 'scene.hands', hands), self._signal(entry, 'scene.hand_count', len(hands))]
        face_out = self.outputs.get('face')
        if face_out and on.get(face_out['branch'], True):
            faces = []
            points = _points(packets.get(face_out['landmarks']), 5, scores=False)
            if points is not None:
                faces.append(dict(frame_face_index=0, landmarks=points, bbox=_box(points)))
            counts['faces'] = len(faces)
            signals += [public('face', 'scene.face_landmarks', faces),
                        self._signal(entry, 'scene.face_count', len(faces))]
        for stream, record in self.streams.items():
            if stream in packets:
                kind, value = packets[stream]
                if kind in ('landmarks', 'world_landmarks', 'landmarks_list'):
                    lists = value if kind == 'landmarks_list' else [value]
                    found = [_points(('landmarks', item)) or [] for item in (lists if isinstance(lists, list) else [])]
                    if stream in self.derived:
                        signals.append(Derived(self.derived[stream], PrivatePoses([dict(landmarks=item) for item in found]),
                                               'video', timestamp_ms=entry['ms']))
                    value = dict(landmark_counts=[len(item) for item in found])
                elif kind in ('image', 'opaque'):
                    continue  # never media, never an untyped blob
                signals.append(self._signal(entry, record, value))
        return [self._frame_record(entry, processed=True, counts=counts, latency_ms=self.last_latency_ms)] + signals


def holistic_plugin(human, *, manifest, **options):
    """Registry entry point (``gamst.crew.registry`` ``in_process``): the staged build named by a profile."""
    return MediaPipeGraphModule.from_manifest(manifest, **options)
