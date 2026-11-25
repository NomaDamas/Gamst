"""MediaPipeGraphModule against a Python stand-in for the runner. The C++ runner itself is NOT tested
here: it has not been compiled (docs/mediapipe-graph.md)."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from gamst import Gamst
from gamst.crew.registry import ROLES
from gamst.crew.setup import SetupError, apply_plan, make_plan
from gamst.mcp import MCPServer
from gamst.mediapipe_graph import HOLISTIC_CPU, HOLISTIC_GPU, PIN, MediaPipeGraphModule, RunnerFailed, holistic_plugin
from gamst.pipeline import Derived, Pipeline
from gamst.plugins import InputFrame
from gamst.private_payload import poses_of
from gamst.profile import ProfileRuntime
from gamst.providers.pose_exemplar import ExemplarEvents, PoseChannels, pose_channels

FAKE = Path(__file__).with_name('fake_mediapipe_runner.py')
PACKAGE = HOLISTIC_CPU.parent.parent


def now():
    return time.time_ns() // 1_000_000


def image(angle=170, *, up=False, right_hand=False, person=True):
    frame = np.zeros((4, 6, 3), np.uint8)
    frame[0, 0] = (angle if person else 255, 255 if up else 0, 0)
    frame[0, 1, 1] = 255 if right_hand else 0
    return frame


def module(**options):
    env = options.pop('env', {})
    options.setdefault('frame_timeout', 5)
    return MediaPipeGraphModule.holistic([sys.executable, str(FAKE)], env=env, startup_timeout=20, **options)


def frame(ms, **options):
    return InputFrame(image(**options), 'video', ms)


def named(signals, name):
    return [signal for signal in signals if isinstance(signal, dict) and signal['name'] == name]


def private(signals, name):
    """The items of a derived input: where the points are, outside every record."""
    return [poses_of(signal.data) for signal in signals if isinstance(signal, Derived) and signal.input == name]


DERIVED = dict(poses='joints', hands='hand_points', face='face_points')


class ModuleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.log = Path(self.directory.name) / 'log'
        self.modules = []

    async def asyncTearDown(self):
        for item in self.modules:
            await item.close()
        self.directory.cleanup()

    def module(self, **options):
        options['env'] = {'FAKE_LOG': str(self.log), **options.get('env', {})}
        item = module(**options)
        self.modules.append(item)
        return item

    def commands(self):
        return self.log.read_text().splitlines()

    async def test_handshake_frames_and_records(self):
        graph = self.module(side_packets=dict(model_complexity=2), derived=DERIVED)
        signals = await graph.process(frame(1000, up=True, right_hand=True))
        self.assertEqual(self.commands(), [
            'SIDE model_complexity int 2', 'SIDE smooth_landmarks bool 1', 'SIDE refine_face_landmarks bool 0',
            'SIDE use_prev_landmarks bool 1', 'SIDE max_in_flight int 1', 'BRANCH pose 1', 'BRANCH hands 1',
            'BRANCH face 1', 'START', 'FRAME 1000000 6 4'])
        public, = named(signals, 'scene.poses')[0]['value']
        (pose,), = private(signals, 'joints')
        self.assertEqual(public, {**{key: pose[key] for key in pose if 'landmarks' not in key}, 'landmark_count': 33})
        self.assertNotIn('"landmarks"', json.dumps([signal for signal in signals if isinstance(signal, dict)], default=str))
        self.assertEqual(len(pose['landmarks']), 33)
        self.assertEqual(set(pose['landmarks'][0]), {'x', 'y', 'z', 'visibility', 'presence'})
        self.assertEqual(len(pose['world_landmarks']), 33)
        self.assertTrue(pose['geometry']['right_wrist_above_shoulder'])
        self.assertIsNotNone(pose['bbox'])
        self.assertEqual([hand['handedness'] for hand in named(signals, 'scene.hands')[0]['value']], ['left', 'right'])
        self.assertEqual(named(signals, 'scene.face_landmarks')[0]['value'][0]['landmark_count'], 468)
        self.assertEqual(len(private(signals, 'face_points')[0][0]['landmarks']), 468)
        stats = named(signals, 'vision.mediapipe_graph.frame')[0]
        graph.spec.validate(stats['value'])
        self.assertEqual(stats['value']['counts'], dict(poses=1, hands=2, faces=1))
        self.assertTrue(stats['value']['processed'])
        self.assertTrue(all((signal['timestamp_ms'] if isinstance(signal, dict) else signal.timestamp_ms) == 1000 for signal in signals))
        self.assertEqual([len(item['landmarks']) for item in private(signals, 'hand_points')[0]], [21, 21])
        self.assertIn('unverified-build', graph.spec.version)  # a bare runner proves no revision

    async def test_bgr_bytes_are_sent_unchanged_even_from_a_view(self):
        graph = self.module()
        big = np.zeros((8, 12, 3), np.uint8)
        big[::2, ::2] = image(45, right_hand=True)
        signals = await graph.process(InputFrame(big[::2, ::2], 'video', 5))
        self.assertEqual(len(named(signals, 'scene.hands')[0]['value']), 2)
        for bad in (np.zeros((4, 6), np.uint8), np.zeros((4, 6, 3), np.float32), b'jpeg', None):
            with self.assertRaises(ValueError):
                await graph.process(InputFrame(bad, 'video', 6))

    async def test_no_person_is_an_empty_record(self):
        signals = await self.module().process(frame(7, person=False))
        self.assertEqual(named(signals, 'scene.poses')[0]['value'], [])
        self.assertEqual(named(signals, 'scene.pose_count')[0]['value'], 0)

    async def test_timestamps_are_capture_microseconds_and_map_back(self):
        graph = self.module()
        seen = []
        for ms in (1000, 1000, 1000, 1004):
            seen += [s['timestamp_ms'] for s in named(await graph.process(frame(ms)), 'scene.poses')]
        self.assertEqual([line.split()[1] for line in self.commands() if line.startswith('FRAME')],
                         ['1000000', '1000001', '1000002', '1004000'])
        self.assertEqual(seen, [1000, 1000, 1000, 1004])
        with self.assertRaises(ValueError):
            await graph.process(frame(900))

    async def test_branch_switches_apply_from_the_next_frame_without_a_new_run(self):
        graph = self.module(branches=dict(face=False))
        signals = await graph.process(frame(1))
        self.assertEqual(named(signals, 'scene.face_landmarks'), [])  # off: no record, not an empty one
        await graph.configure(dict(branches=dict(face=True, hands=False)))
        signals = await graph.process(frame(2))
        self.assertEqual(len(named(signals, 'scene.face_landmarks')), 1)
        self.assertEqual(named(signals, 'scene.hands'), [])
        self.assertEqual(graph.run, 1)
        self.assertNotIn('RESET', self.commands())

    async def test_side_packets_start_a_fresh_run(self):
        graph = self.module()
        await graph.process(frame(1))
        status = await graph.configure(dict(side_packets=dict(refine_face_landmarks=True)))
        self.assertEqual((status['run'], status['side_packets']['refine_face_landmarks']), (2, True))
        signals = await graph.process(frame(2))
        self.assertEqual(named(signals, 'scene.face_landmarks')[0]['value'][0]['landmark_count'], 478)

    async def test_configure_is_limited_to_what_the_owner_declared(self):
        graph = self.module(side_packets=dict(max_in_flight=1))
        graph.side_packets['model_path'] = '/private/model'
        for options in (dict(branches=dict(audio=True)), dict(branches=dict(face=1)),
                        dict(side_packets=dict(other=1)), dict(side_packets=dict(model_complexity='x')),
                        dict(side_packets=dict(model_complexity=True)),
                        dict(side_packets=dict(model_path='/etc/passwd')),
                        dict(side_packets=dict(max_in_flight=0)), dict(graph='x')):
            with self.subTest(options), self.assertRaises(ValueError):
                await graph.configure(options)
        self.assertEqual(graph.status()['side_packets']['model_path'], '<text>')
        with self.assertRaises(ValueError):
            MediaPipeGraphModule(sys.executable, HOLISTIC_CPU, side_packets=dict(x='a\nCLOSE'))

    async def test_undeclared_side_packet_or_branch_fails_the_start(self):
        for options in (dict(branches=dict(audio=True)), dict(side_packets=dict(other=1))):
            graph = self.module(**options)
            with self.assertRaisesRegex(ValueError, 'does not declare'):
                await graph.process(frame(1))
            self.assertIsNone(graph._process)

    async def test_in_flight_frames_overlap_and_finish_returns_the_rest(self):
        graph = self.module(side_packets=dict(max_in_flight=3), env={'FAKE_DELAY_MS': '60'})
        await graph.start()
        first = await graph.process(frame(1))
        second = await graph.process(frame(2))
        self.assertEqual((first, second), ([], []))  # sent, still inside the graph
        self.assertEqual(len(graph._pending), 2)
        rest = await graph.process(frame(3)) + await graph.finish()
        self.assertEqual([s['timestamp_ms'] for s in named(rest, 'scene.pose_count')], [1, 2, 3])
        self.assertEqual(graph.dropped, 0)

    async def test_graph_flow_limiter_drops_are_recorded(self):
        graph = self.module(env={'FAKE_DROP_EVERY': '2'})
        signals = []
        for ms in range(1, 5):
            signals += await graph.process(frame(ms))
        stats = [s['value']['processed'] for s in named(signals, 'vision.mediapipe_graph.frame')]
        self.assertEqual(stats, [True, False, True, False])
        self.assertEqual(len(named(signals, 'scene.poses')), 2)
        self.assertEqual(graph.status()['dropped_by_graph'], 2)

    async def test_a_done_past_several_frames_finishes_them_and_unknown_packets_are_ignored(self):
        graph = self.module()
        for us in (1000, 2000, 3000):
            graph._pending[us] = dict(ms=us // 1000, subject_id='a', sensor=None, branches={}, sent=time.monotonic(),
                                      future=asyncio.get_running_loop().create_future())
        graph._dispatch(dict(type='packet', run=0, ts_us=1500, stream='x', kind='number', value=1))
        graph._dispatch(dict(type='done', run=7, ts_us=3000, processed=True))  # another run
        self.assertEqual((len(graph._pending), graph._packets), (3, {}))
        graph._dispatch(dict(type='done', run=0, ts_us=2000, processed=True))
        graph._dispatch(dict(type='done', run=0, ts_us=2999, processed=False))
        self.assertEqual((list(graph._pending), graph.processed, graph.dropped), ([3000], 1, 1))

    async def test_reset_is_a_fresh_run_and_discards_uncommitted_records(self):
        graph = self.module(side_packets=dict(max_in_flight=2), env={'FAKE_DELAY_MS': '30'})
        await graph.process(frame(1))
        await graph.reset()
        self.assertEqual((graph.run, len(graph._pending)), (2, 0))
        signals = await graph.process(frame(2)) + await graph.finish()
        self.assertEqual({signal['timestamp_ms'] for signal in signals}, {2})

    async def test_cancelled_reset_keeps_the_protocol_in_step(self):
        graph = self.module(env={'FAKE_DELAY_MS': '50'})
        await graph.process(frame(1))
        task = asyncio.create_task(graph.reset())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(named(await graph.process(frame(2)), 'scene.poses')), 1)
        self.assertEqual(graph.run, 2)

    async def test_fatal_error_needs_an_explicit_restart(self):
        graph = self.module(env={'FAKE_FAIL_AFTER': '1'})
        await graph.process(frame(1))
        with self.assertRaisesRegex(RunnerFailed, 'frame:INTERNAL'):
            await graph.process(frame(2))
        await graph.reset()  # a Pipeline reset is not a restart
        with self.assertRaises(RunnerFailed):
            await graph.process(frame(3))
        await graph.configure(dict(restart=True))
        self.assertEqual(len(named(await graph.process(frame(4)), 'scene.poses')), 1)

    async def test_stalled_dead_and_garbage_runners_fail_bounded(self):
        stalled = self.module(env={'FAKE_DELAY_MS': '3000'}, frame_timeout=.3, close_timeout=.2)
        await stalled.start()
        started = time.monotonic()
        with self.assertRaisesRegex(RunnerFailed, 'timeout:frame'):
            await stalled.process(frame(1))
        await stalled.close()
        self.assertLess(time.monotonic() - started, 3)

        killed = self.module()
        await killed.process(frame(1))
        killed._process.kill()
        with self.assertRaises(RunnerFailed):
            await killed.process(frame(2))

        for text in ('not json', '{"type":"hello","protocol":2}', '{"type":"hello","protocol":NaN}', 'x' * 900000, ''):
            script = Path(self.directory.name) / 'bad.py'
            script.write_text(f'import sys,time\nprint({text!r}, flush=True)\n'
                              + ('' if not text else 'sys.stdin.read()\n'))
            bad = MediaPipeGraphModule([sys.executable, str(script)], HOLISTIC_CPU, startup_timeout=10)
            with self.subTest(text[:20]), self.assertRaises(RunnerFailed):
                await bad.process(frame(1))
            self.assertIsNone(bad._process)

    async def test_close_ends_the_process(self):
        graph = self.module()
        await graph.process(frame(1))
        process = graph._process
        await graph.close()
        await graph.close()
        self.assertEqual(process.returncode, 0)  # CLOSE, not a signal
        self.assertEqual(self.commands()[-1], 'CLOSE')

    async def test_extra_streams_and_bad_landmarks(self):
        graph = self.module()
        graph.streams = dict(hits='custom.hits', picture='custom.picture', marks='custom.marks')
        entry = dict(ms=1, subject_id='a', sensor=None, branches={})
        signals = graph._records(entry, dict(
            hits=('number', 3), picture=('image', dict(width=4, height=4)), marks=('landmarks', [[.1, .2, .3, None, None]]),
            pose_landmarks=('landmarks', [[None, .2, .3, 1, 1]] * 33)))
        self.assertEqual(named(signals, 'custom.hits')[0]['value'], 3)
        self.assertEqual(named(signals, 'custom.picture'), [])
        self.assertEqual(named(signals, 'custom.marks')[0]['value'], dict(landmark_counts=[1]))     # how many, never where
        graph.derived = dict(marks='mark_points')
        signals = graph._records(entry, dict(marks=('landmarks', [[.1, .2, .3, None, None]])))
        self.assertEqual(private(signals, 'mark_points'), [[dict(landmarks=[dict(x=.1, y=.2, z=.3)])]])
        self.assertEqual(named(signals, 'scene.poses')[0]['value'], [])  # non-finite: not invented


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_lane_runs_in_parallel_with_other_modules_and_counts_drops(self):
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'parallel')
            pipeline.add_input('camera', modality='video')
            graph = module(env={'FAKE_DELAY_MS': '150'})
            await graph.start()
            pipeline.add_module('graph', graph, consumes='camera', queue=1)
            seen = []

            class Fast:
                name = 'fast'

                def process(self, item):
                    seen.append(item.timestamp_ms)
                    return []
            pipeline.add_module('fast', Fast(), consumes='camera')
            start, began = now() - 1000, time.monotonic()
            for index in range(6):
                await pipeline.push('camera', image(), timestamp_ms=start + index)
                await asyncio.sleep(.005)
            self.assertEqual(len(seen), 6)
            self.assertLess(time.monotonic() - began, .15)  # never waited for the slow graph
            await pipeline.finish()
            status = pipeline.status()['modules']['graph']
            self.assertGreater(status['dropped'], 0)
            self.assertEqual(status['processed'] + status['dropped'], 6)
            self.assertEqual(status['state']['processed'], status['processed'])
            self.assertEqual(status['state']['dropped_by_graph'], 0)
            process = graph._process
            await pipeline.close()
            self.assertIsNotNone(process.returncode)

    async def test_pose_records_drive_the_exemplar_count_with_frames_in_flight(self):
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'pose')
            graph = module(branches=dict(hands=False, face=False), side_packets=dict(max_in_flight=3), derived=dict(poses='joints'))
            signals = await graph.process(frame(1, angle=30, up=True)) + await graph.finish()
            values, mask = pose_channels(private(signals, 'joints')[0][0]['landmarks'])
            await graph.reset()
            pipeline.add_input('camera', modality='video')
            pipeline.add_reference('shotgun', 'exemplar', dict(channels=values, mask=mask))
            pipeline.add_module('graph', graph, consumes='camera', queue=8, emits=['joints'])
            channels = PoseChannels()
            pipeline.add_module('channels', channels, consumes='joints', spec=channels.spec)
            matcher = ExemplarEvents(reference='shotgun', mode='hold', min_hold_ms=60)
            pipeline.add_module('exemplar', matcher, consumes='channels', spec=matcher.spec)
            start = now() - 2000
            poses = ([dict(angle=170)] * 5 + [dict(angle=30, up=True)] * 5) * 2 + [dict(angle=170)] * 5
            for index, pose in enumerate(poses):
                await pipeline.push('camera', image(**pose), timestamp_ms=start + 33 * index)
                await asyncio.sleep(.005)
            await pipeline.finish()
            self.assertEqual(pipeline.status()['modules']['graph']['dropped'], 0)
            counted = await human.count_events('temporal.exemplar_events', event='exemplar.matched')
            self.assertEqual(counted['count'], 2)
            # Every linked record kept the capture time of its own frame.
            self.assertEqual([e['event']['start_ms'] - start for e in counted['evidence']], [33 * 5, 33 * 15])
            stats = human.store.query('local-user', kind='observation.signal', name='vision.mediapipe_graph.frame')
            self.assertEqual([r['timestamp_ms'] - start for r in stats], [33 * i for i in range(len(poses))])
            self.assertEqual(stats[0]['observation']['capability']['id'], 'vision.mediapipe_graph')
            for answer in (await human.get_state(), await human.inspect_interval(0, now())):
                self.assertNotIn('"landmarks"', json.dumps(answer))
            context = json.dumps(await human.get_context())
            self.assertIn('vision.mediapipe_graph', context)
            self.assertNotIn('"landmarks"', context)
            self.assertNotIn('"world_landmarks"', context)
            await pipeline.close()

    async def test_mcp_switches_branches_and_module(self):
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'mcp')
            pipeline.add_input('camera', modality='video')
            graph = module()
            pipeline.add_module('graph', graph, consumes='camera')
            server, start = MCPServer(human), now() - 1000
            await pipeline.push('camera', image(), timestamp_ms=start)
            result = await server.call_tool('configure_pipeline_module', {
                'name': 'graph', 'enabled': True, 'options': {'branches': {'face': False}}})
            self.assertFalse(result['state']['branches']['face'])
            with self.assertRaises(ValueError):
                await server.call_tool('configure_pipeline_module', {
                    'name': 'graph', 'enabled': True, 'options': {'side_packets': {'path': '/etc'}}})
            records = await pipeline.push('camera', image(), timestamp_ms=start + 33)
            self.assertNotIn('scene.face_landmarks', [r['observation']['name'] for r in records])
            await server.call_tool('configure_pipeline_module', {'name': 'graph', 'enabled': False})
            self.assertEqual(await pipeline.push('camera', image(), timestamp_ms=start + 66), [])
            await server.call_tool('configure_pipeline_module', {'name': 'graph', 'enabled': True})
            await pipeline.push('camera', image(), timestamp_ms=start + 99)
            self.assertEqual(graph.run, 2)  # off and on again is a fresh graph run
            json.dumps(await server.call_tool('get_pipeline', {}))
            await pipeline.close()
            self.assertIsNone(graph._process)


class InstallTest(unittest.IsolatedAsyncioTestCase):
    def build(self, root, *, commit=PIN['commit']):
        """A staged build as build.sh lays it out, with the stand-in as its runner."""
        for name in ('bin', 'graphs', 'resources'):
            (root / name).mkdir()
        runner = root / 'bin/gamst_graph_runner'
        runner.write_text(f'#!{sys.executable}\n' + FAKE.read_text())
        runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
        (root / 'graphs/holistic_cpu.pbtxt').write_text(HOLISTIC_CPU.read_text())
        (root / 'resources/model.tflite').write_bytes(b'model')
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps(dict(
            kind='gamst.mediapipe_graph.runner', variant='cpu', mediapipe_commit=commit, runner='bin/gamst_graph_runner',
            runner_sha256=digest(runner), resource_root='resources',
            files={name: digest(root / name) for name in ('graphs/holistic_cpu.pbtxt', 'resources/model.tflite')})))
        return manifest

    async def test_manifest_checks_revision_and_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest = self.build(root)
            graph = MediaPipeGraphModule.from_manifest(manifest, env={'FAKE_REVISION': PIN['commit']})
            self.assertTrue(graph.spec.version.startswith(PIN['tag'] + '.holistic_cpu'))
            self.assertEqual(len(named(await graph.process(frame(1)), 'scene.poses')), 1)
            await graph.close()
            other = MediaPipeGraphModule.from_manifest(manifest, env={'FAKE_REVISION': 'other'})
            with self.assertRaisesRegex(RunnerFailed, 'another MediaPipe revision'):
                await other.process(frame(1))
            for name in ('resources/model.tflite', 'bin/gamst_graph_runner'):
                original = (root / name).read_bytes()
                (root / name).write_bytes(original + b'#')
                with self.subTest(name), self.assertRaisesRegex(SetupError, 'sha256 mismatch'):
                    MediaPipeGraphModule.from_manifest(manifest)
                (root / name).write_bytes(original)
            data = json.loads(manifest.read_text())
            manifest.write_text(json.dumps({**data, 'mediapipe_commit': '0' * 40}))
            with self.assertRaisesRegex(ValueError, 'another MediaPipe revision'):
                MediaPipeGraphModule.from_manifest(manifest)

    async def test_profile_role_runs_in_a_lane_and_setup_refuses_an_unbuilt_one(self):
        candidate = ROLES['vision.mediapipe_graph'].candidates['graph.holistic_cpu']
        self.assertFalse(candidate.automatic)
        with self.assertRaisesRegex(SetupError, 'manual preparation'):
            apply_plan(make_plan(roles={'vision.mediapipe_graph': 'graph.holistic_cpu'}))
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, FAKE_REVISION=PIN['commit']):
            root = Path(directory).resolve()
            profile = root / 'profile.json'
            (root / 'build').mkdir()
            profile.write_text(json.dumps(dict(version=1, roles={'vision.mediapipe_graph': dict(
                model='graph.holistic_cpu', provider_profile=None, python=None,
                config=dict(manifest=str(self.build(root / 'build'))))})))
            async with Gamst(consent=True) as human:
                runtime = await ProfileRuntime.open(human, profile)
                status = runtime.pipeline.status()['modules']['vision.mediapipe_graph']
                self.assertEqual(status['lane'], 'vision.mediapipe_graph')
                await runtime.pipeline.push('camera', image(), timestamp_ms=now())
                await runtime.pipeline.drain()
                status = runtime.pipeline.status()['modules']['vision.mediapipe_graph']
                self.assertEqual((status['processed'], status['errors']), (1, 0))
                await runtime.close()


class SourceTest(unittest.TestCase):
    """Text checks only; nothing here compiles C++ or parses a graph with MediaPipe."""

    def test_pin_is_one_revision_everywhere(self):
        self.assertRegex(PIN['commit'], '^[0-9a-f]{40}$')
        self.assertIn(PIN['commit'], (PACKAGE / 'WORKSPACE.snippet').read_text())
        self.assertNotRegex((PACKAGE / 'build.sh').read_text(), '[0-9a-f]{40}')  # read from pin.json

    def test_graph_streams_are_connected_and_bundled(self):
        for path in (HOLISTIC_CPU, HOLISTIC_GPU):
            text = re.sub(r'#.*', '', path.read_text())
            self.assertEqual(text.count('{'), text.count('}'))
            produced = {item.rsplit(':', 1)[-1] for item in re.findall(r'(?m)^  output_stream:\s*"([^"]+)"', text)}
            top = set(re.findall(r'(?m)^input_stream:\s*"([^"]+)"', text))
            consumed = {item.rsplit(':', 1)[-1] for item in re.findall(r'(?m)^  input_stream:\s*"([^"]+)"', text)}
            with self.subTest(path.name):
                self.assertEqual(consumed - produced - top, set())
                self.assertEqual(top - consumed, set())
                bundled = re.findall(r'"STREAM:\d+:([^"]+)"', text)
                self.assertEqual(set(bundled), {'pose_landmarks', 'pose_world_landmarks', 'left_hand_landmarks',
                                                'right_hand_landmarks', 'face_landmarks'})
                sides = set(re.findall(r'(?m)^input_side_packet:\s*"([^"]+)"', text))
                used = {item.rsplit(':', 1)[-1] for item in re.findall(r'(?m)^  input_side_packet:\s*"([^"]+)"', text)}
                self.assertEqual(sides, used)

    def test_runner_source_and_module_name_the_same_protocol(self):
        source = (PACKAGE / 'runner/gamst_graph_runner.cc').read_text()
        self.assertIn('constexpr int kProtocol = 1;', source)
        for command in ('FRAME', 'SIDE', 'BRANCH', 'START', 'RESET', 'CLOSE'):
            self.assertIn(f'op == "{command}"', source)

    def test_holistic_plugin_needs_a_manifest(self):
        with self.assertRaises(FileNotFoundError):
            holistic_plugin(None, manifest='/nonexistent/manifest.json')


if __name__ == '__main__':
    unittest.main()
