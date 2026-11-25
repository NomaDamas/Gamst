"""What fails does not end the rest: real inputs and modules with stand-in devices and models."""
import asyncio
from types import SimpleNamespace
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from gamst import Gamst, ProcessProvider
from gamst.capabilities import CapabilitySpec, OutputField
from gamst.inputs import CameraInput, MicrophoneInput
from gamst.pipeline import FAILURES, Pipeline
from gamst.providers.prosody import AudioProsody
from gamst.redact import public_error, public_text
from gamst.schema import timestamp
from tests.test_multi_input import Recorded, Recorder
from tests.test_pose import fixture_plugin, points


class Device:
    """A camera: gives frames, fails, and may come back."""
    plans = []

    def __init__(self, *_):
        self.left = Device.plans.pop(0) if Device.plans else None

    def isOpened(self):
        return self.left is not None

    def read(self):
        import time
        time.sleep(.004)
        self.left -= 1
        return self.left >= 0, np.zeros((36, 64, 3), np.uint8)

    def release(self):
        pass


def camera(*plans):
    Device.plans = list(plans)
    return mock.patch.dict(sys.modules, cv2=SimpleNamespace(VideoCapture=Device, CAP_AVFOUNDATION=1, CAP_ANY=0))


class Stream:
    """A sound device that gives 20 ms blocks for ever, or fails."""
    latency, read_available = 0., 0

    def __init__(self, blocks=None):
        self.blocks = blocks

    def start(self):
        pass

    stop = close = start

    def read(self, count):
        import time
        time.sleep(.02)
        if self.blocks is not None:
            self.blocks -= 1
            if self.blocks < 0:
                raise OSError('the sound device went away')
        return np.full((count, 1), .1, np.float32), False


def microphone(blocks=None):
    return MicrophoneInput(chunk_ms=20, stream_factory=lambda **options: Stream(blocks))


class Failures(unittest.IsolatedAsyncioTestCase):
    async def start(self, human, eye=None, sound=None, waits=(.05, .05)):
        pipeline = Pipeline(human, 'two')
        pipeline.reopen_s = waits
        pipeline.add_input('camera', CameraInput(0)).add_input('microphone', sound or microphone())
        self.eye, self.ear = eye or fixture_plugin(lambda _: SimpleNamespace(pose_landmarks=[points()])), AudioProsody()
        pipeline.add_module('pose', self.eye, consumes=['camera'])
        pipeline.add_module('prosody', self.ear, consumes=['microphone'], spec=AudioProsody.capability())
        return pipeline

    async def until(self, pipeline, done, seconds=8):
        stop = asyncio.Event()
        run = asyncio.create_task(pipeline.run(stop=stop))
        for _ in range(int(seconds / .01)):
            await asyncio.sleep(.01)
            if run.done() or done(pipeline):
                break
        stop.set()
        await asyncio.wait_for(run, 10)

    async def test_a_live_input_is_reopened_the_outage_is_a_gap_and_the_other_goes_on(self):
        async with Gamst(consent=True) as human:
            with camera(10, None, 10_000):
                pipeline = await self.start(human)
                shown, heard = [], []

                def done(pipeline):
                    status = pipeline.status()
                    if status['inputs']['camera'].get('down_since_ms'):
                        shown.append(status)
                        heard.append(status['modules']['prosody']['processed'])
                    return status['modules']['pose']['processed'] > 20
                await self.until(pipeline, done)
                status = pipeline.status()
            down = shown[-1]
            self.assertFalse(down['complete'])
            self.assertGreater(heard[-1], heard[0])
            self.assertIn('cannot open camera 0', down['inputs']['camera']['error'] + str(shown))
            gap = down['modules']['pose']['gap_intervals'][-1]
            self.assertEqual((gap['reason'], gap['source']), ('input_failed', 'camera'))
            self.assertEqual(down['modules']['pose']['complete_until_ms'], gap['end_ms'])    # declared, not pending
            outage, = status['inputs']['camera']['outages']
            self.assertIn('stopped giving frames', outage['error'])
            self.assertGreater(outage['resumed_ms'], outage['from_ms'])
            self.assertNotIn('error', status['inputs']['camera'])
            self.assertEqual([item['input'] for item in status['failures']], ['camera'])
            self.assertEqual(status['modules']['pose']['gaps'], 1)
            gap, = status['modules']['pose']['gap_intervals']
            self.assertEqual((gap['start_ms'], gap['end_ms']), (outage['from_ms'], outage['resumed_ms'] - 1))
            self.assertEqual(status['modules']['prosody']['gaps'], 0)
            stored = [event for record in human.store.query('local-user', capability_id='pipeline.gaps', registered_only=True)
                      for event in record['observation']['value']['events']]
            self.assertEqual({tuple(event['reasons']) for event in stored}, {('input_failed',)})
            self.assertEqual((stored[0]['start_ms'], stored[-1]['end_ms']), (outage['from_ms'], outage['resumed_ms'] - 1))
            answer = await human.get_timeline(start_ms=outage['from_ms'] - 100, end_ms=outage['resumed_ms'] + 100, step_ms=20,
                                              capabilities=['pose'])
            cells = [row['cells']['pose'] for row in answer['rows']] if 'pose' in answer['rows'][0]['cells'] else [
                next(iter(row['cells'].values())) for row in answer['rows']]
            self.assertIn('input_failed', [cell.get('reason') for cell in cells])
            await pipeline.close()

    async def test_it_is_given_up_after_a_bounded_number_of_times_and_the_run_is_incomplete(self):
        async with Gamst(consent=True) as human:
            with camera(5):
                pipeline = await self.start(human)
                await self.until(pipeline, lambda p: p.inputs['camera']['ended'] and p.modules['prosody']['processed'] > 30)
            status = pipeline.status()
            self.assertTrue(status['failures'][0]['given_up'])
            self.assertGreater(status['modules']['prosody']['processed'], 30)
            self.assertFalse(status['complete'])
            self.assertGreater(pipeline.lost()['gaps'], 0)
            await pipeline.close()

    async def test_the_run_ends_when_nothing_is_left(self):
        async with Gamst(consent=True) as human:
            with camera(5):
                pipeline = await self.start(human, sound=microphone(3), waits=(.02,))
                await asyncio.wait_for(pipeline.run(), 8)
            self.assertEqual({item['input'] for item in pipeline.status()['failures']}, {'camera', 'microphone'})
            await pipeline.close()

    async def test_a_module_that_always_fails_is_given_up_and_can_be_switched_on_again(self):
        def broken(_):
            raise RuntimeError('the model is gone')

        async with Gamst(consent=True) as human:
            with camera(10_000):
                pipeline = await self.start(human, eye=fixture_plugin(broken))
                await self.until(pipeline, lambda p: p.modules['pose']['failed'] and p.modules['prosody']['processed'] > 20)
            status = pipeline.status()
            pose = status['modules']['pose']
            self.assertEqual(pose['errors'], FAILURES)
            self.assertIn('the model is gone', pose['failed'])
            self.assertEqual(pose['gap_intervals'][-1]['reason'], 'error')
            self.assertEqual(pose['gap_frames'], status['inputs']['camera']['frames'])    # every frame is accounted for
            self.assertEqual(pose['complete_until_ms'], status['inputs']['camera']['last_ms'])
            self.assertEqual(status['modules']['prosody']['errors'], 0)
            self.assertIn(dict(module='pose', error=pose['failed'], given_up=True), status['failures'])
            self.assertNotIn('failed', pipeline.enable('pose'))
            await pipeline.close()

    async def test_a_run_of_failed_modules_only_ends(self):
        def broken(_):
            raise RuntimeError('the model is gone')

        async with Gamst(consent=True) as human:
            with camera(10_000):
                pipeline = Pipeline(human, 'one')
                pipeline.add_input('camera', CameraInput(0))
                pipeline.add_module('pose', fixture_plugin(broken), consumes=['camera'])
                await asyncio.wait_for(pipeline.run(), 8)
            self.assertEqual(pipeline.status()['modules']['pose']['errors'], FAILURES)
            await pipeline.close()

    async def test_one_recording_that_breaks_does_not_end_the_other(self):
        base = timestamp() - 60000
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'files')
            pipeline.add_input('video', Recorded('video', [base + 40 * i for i in range(50)], fail_at=10))
            pipeline.add_input('sound', Recorded('audio', [base + 100 * i for i in range(1, 20)]))
            eye, ear = Recorder('test.eye', 'video'), Recorder('test.ear', 'audio')
            pipeline.add_module('eye', eye, consumes=['video'], spec=eye.spec)
            pipeline.add_module('ear', ear, consumes=['sound'], spec=ear.spec)
            await pipeline.run()
            status = pipeline.status()
            self.assertEqual((len(eye.seen), len(ear.seen)), (10, 19))
            gap, = status['modules']['eye']['gap_intervals']
            self.assertEqual((gap['start_ms'], gap['end_ms'], gap['reason']), (base + 361, base + 1900, 'input_failed'))
            self.assertEqual(status['modules']['eye']['complete_until_ms'], base + 1900)
            counted = await human.count_events('test.eye', window='5m')
            self.assertFalse(counted['complete'])
            await pipeline.close()


class Slow(unittest.IsolatedAsyncioTestCase):
    async def test_a_lane_slower_than_its_camera_is_named_with_what_helps(self):
        class Pose:
            owns_process = True
            spec = CapabilitySpec('test.pose', 'test.pose', 'video', 'test', '1', (OutputField('n', 'integer'),), 'frame')

            async def process(self, frame):
                await asyncio.sleep(.012)
                return dict(n=1)

        for options, wanted in ({}, '--camera-max-side PX'), (dict(max_side=160), 'a smaller --camera-max-side PX'):
            async with Gamst(consent=True) as human:
                with camera(120):
                    pipeline = Pipeline(human, 'slow')
                    pipeline.reopen_s = ()
                    pipeline.add_input('camera', CameraInput(0, **options))
                    pipeline.add_module('pose', Pose(), consumes='camera', spec=Pose.spec, queue='auto')
                    await pipeline.run()
                warning, = pipeline.status()['warnings']
                self.assertEqual((warning['lane'], warning['input'], warning['frame_size']), ('pose', 'camera', [64, 36]))
                self.assertGreater(warning['ms_per_frame'], warning['frame_interval_ms'])
                self.assertIn(wanted, warning['text'])
                await pipeline.close()


class StartErrors(unittest.IsolatedAsyncioTestCase):
    async def opened(self, worker, **environment):
        with tempfile.TemporaryDirectory() as folder:
            launcher = os.path.join(folder, 'python')       # a stand-in interpreter: the same wire, another worker
            with open(launcher, 'w') as out:
                out.write(f"#!{sys.executable}\n{worker}")
            os.chmod(launcher, 0o755)
            with mock.patch.dict(os.environ, environment), self.assertRaises(RuntimeError) as caught:
                await ProcessProvider.open(launcher, 'audio.prosody', {}, startup_timeout=20)
            return str(caught.exception)

    async def test_the_message_of_a_failed_start_is_reported_without_secrets(self):
        message = await self.opened("""
import os
from gamst import provider_worker
def construct(profile, config):
    raise AttributeError("type object 'FasterWhisperPlugin' has no attribute 'version' " + os.environ['MY_API_KEY'])
provider_worker._construct = construct
provider_worker.main()
""", MY_API_KEY='super-secret-value', PYTHONPATH=os.getcwd())
        self.assertIn("AttributeError: type object 'FasterWhisperPlugin' has no attribute 'version' ***", message)
        self.assertNotIn('super-secret', message)

    async def test_a_worker_that_never_answered_says_how_it_ended(self):
        message = await self.opened("raise ModuleNotFoundError(\"No module named 'gamst'\")\n")
        self.assertIn('ended while starting (exit code 1)', message)
        self.assertIn("No module named 'gamst'", message)

    async def test_what_a_started_worker_writes_never_blocks_or_reaches_anyone(self):
        with tempfile.TemporaryDirectory() as folder:
            launcher = os.path.join(folder, 'python')
            with open(launcher, 'w') as out:
                out.write(f"""#!{sys.executable}
import sys
from gamst import provider_worker
original = provider_worker.capability
class Noisy:
    def process(self, frame): return None
    def finish(self):
        for _ in range(20000): sys.stderr.write('x' * 100)
        raise RuntimeError('what it saw')
    def reset(self): pass
    def close(self): pass
provider_worker._construct = lambda profile, config: (Noisy(), original(profile, config))
provider_worker.main()
""")
            os.chmod(launcher, 0o755)
            with mock.patch.dict(os.environ, PYTHONPATH=os.getcwd()):
                worker = await ProcessProvider.open(launcher, 'audio.prosody', {}, startup_timeout=20)
            with self.assertRaisesRegex(RuntimeError, r'worker failed \(RuntimeError\)') as caught:
                await asyncio.wait_for(worker.finish(), 10)
            self.assertNotIn('what it saw', str(caught.exception))

    def test_what_is_hidden(self):
        home = os.path.expanduser('~')
        for text, shown in (('no file ' + home + '/models/a.onnx', 'no file ~/models/a.onnx'),
                            ('GET https://user:pw@host/a?token=abc failed', 'GET https://***@host/a?*** failed'),
                            ('Authorization: Bearer abcdefghijkl', 'Authorization: Bearer ***'),
                            ('api_key=abc123 rejected', 'api_key=*** rejected'),
                            ('bad sk-abcdefghijklmnop', 'bad ***')):
            self.assertEqual(public_text(text, environ={}), shown)
        self.assertEqual(len(public_text('x' * 5000)), 300)
        error = RuntimeError('Install gamst-sdk[asr]')
        error.__cause__ = ImportError("No module named 'faster_whisper'")
        self.assertEqual(public_error(error), "RuntimeError: Install gamst-sdk[asr] (from ImportError: No module named 'faster_whisper')")


if __name__ == '__main__':
    unittest.main()
