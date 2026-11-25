"""GAMST 240 on the Pipeline: the iPhone input, the take recorder and the after-take counter (refine mocked)."""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from unittest.mock import patch

from gamst.inputs import (FULL, RECORDING, IPhone240Input, PhoneEvent, PhoneFrame, TakeInput, parse_header,
                          read_recording)

HAS_CV = all(importlib.util.find_spec(name) for name in ('cv2', 'numpy'))
HAS_HUD = HAS_CV and importlib.util.find_spec('PIL') is not None


def gms3(sequence, seconds, flags, payload, *, width=48, height=64, zoom=10, angle=0, crop=(0, 0, 65535, 65535)):
    """One frame as the app sends it: "GMS3" header + JPEG."""
    return b'GMS3' + struct.pack('<IdHHBBH4HI', sequence, seconds, width, height, flags, zoom, angle, *crop,
                                 len(payload)) + payload


def jpeg(value):
    if not HAS_CV:
        return bytes([value % 256]) * 16
    import cv2
    import numpy as np
    ok, data = cv2.imencode('.jpg', np.full((64, 48, 3), value % 256, np.uint8))
    return data.tobytes()


def recording(takes=((20, 60), (80, 110)), frames=120, fps=240.):
    """A replayable stream: standby frames with ``takes`` (start, stop) frame ranges recorded."""
    return [gms3(index, 100 + index / fps, RECORDING if any(a <= index < b for a, b in takes) else 0, jpeg(index))
            for index in range(frames)]


def refined_results(take_dir, records):
    """What refine_take writes for a take: one throw per call (the models are not run here)."""
    frames = read_recording(Path(take_dir) / 'raw.gmst')
    stamp = frames[len(frames) // 2][0].seconds * 1000
    return dict(label='metal coin', detections=3, timing={}, results=[dict(
        type='rotation.ended', track_id=f'metal-coin-{len(records) + 1}', label='metal coin', timestamp_ms=stamp,
        reason='landed', status='counting', turns=2.5, rev_s=12.5, duration_ms=200, coverage=1.,
        trajectory=dict(kind='free_fall'), flight_frames=[1, 9], frames=[0, 10], shapes=[])])


class FakeRefine:
    """Stands in for TakeFlips._launch: writes refined.json at once, like a finished refine process."""

    def __init__(self):
        self.calls = []

    @property
    def launch(self):
        return lambda counter, directory: self(counter, directory)

    def __call__(self, counter, directory):
        from gamst.takes import _Finished
        self.calls.append(dict(dir=Path(directory), args=list(counter.refine_args)))
        (Path(directory) / 'refined.json').write_text(json.dumps(refined_results(directory, self.calls[:-1])))
        return _Finished()


class WireFormatTest(unittest.TestCase):
    def test_headers_of_every_app_build_parse(self):
        info = parse_header(gms3(7, 12.5, RECORDING | 2, b'x' * 9, zoom=20, angle=450, crop=(0, 0, 32767, 65535))[:36])
        self.assertEqual((info.sequence, info.seconds, info.flags, info.zoom, info.angle, info.size), (7, 12.5, 3, 2., 90, 9))
        self.assertAlmostEqual(info.crop[2], .5, places=4)
        upright = parse_header(b'GMS2' + struct.pack('<IdHHBB2xI', 1, 2., 10, 20, 0, 0, 5))
        self.assertEqual((upright.zoom, upright.angle, upright.crop, upright.flags), (1., 0, FULL, 0))
        first = parse_header(b'GMST' + struct.pack('<IdHHI', 1, 2., 10, 20, 5))
        self.assertEqual(first.flags, RECORDING)  # the first build had no flags: one continuous take

    def test_read_recording_keeps_header_bytes_and_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = recording(frames=10)
            path = Path(tmp) / 'raw.gmst'
            path.write_bytes(b''.join(stream))
            frames = read_recording(path)
            self.assertEqual([header + payload for _, header, payload in frames], stream)
            self.assertAlmostEqual(TakeInput(tmp).fps(), 240., places=6)


class IPhone240InputTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.tmp = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def replay(self, **options):
        path = self.tmp / 'raw.gmst'
        path.write_bytes(b''.join(recording()))
        return IPhone240Input(replay=path, **options)

    def frames(self, source):
        async def collect():
            return [frame async for frame in source.frames()]
        return asyncio.run(collect())

    def test_replay_keeps_capture_spacing_and_take_marks(self):
        source = self.replay()
        frames = self.frames(source)
        self.assertEqual(len(frames), 120)
        self.assertTrue(all(isinstance(frame.data, PhoneFrame) for frame in frames))
        self.assertEqual([index for index, frame in enumerate(frames) if frame.data.recording],
                         list(range(20, 60)) + list(range(80, 110)))
        stamps = [frame.timestamp_ms for frame in frames]
        self.assertEqual(stamps[-1] - stamps[0], round(1000 * 119 / 240))
        self.assertEqual(source.clock(frames[5].data.info.seconds * 1000), stamps[5])
        self.assertEqual(frames[3].sensor.capture_sequence, 3)
        self.assertEqual(source.state()['received'], 120)

    @unittest.skipUnless(HAS_CV, 'needs opencv and numpy')
    def test_decode_yields_upright_images_for_video_modules(self):
        frames = self.frames(self.replay(decode=True, max_frames=4))
        self.assertEqual([frame.modality for frame in frames], ['video'] * 4)
        self.assertEqual(frames[0].data.shape, (64, 48, 3))

    def test_live_stream_keeps_every_take_frame_and_reports_the_disconnect(self):
        async def scenario():
            source = IPhone240Input(0, host='127.0.0.1')
            received = []

            async def pump():
                async for frame in source.frames():
                    received.append(frame)
                    if isinstance(frame.data, PhoneEvent):
                        source.stop()

            task = asyncio.create_task(pump())
            while source._listener is None:
                await asyncio.sleep(.01)
            phone = socket.create_connection(('127.0.0.1', source.port))
            phone.sendall(b''.join(gms3(index, 5 + index / 240, RECORDING if 2 <= index < 6 else 0, b'j' * 4)
                                   for index in range(8)))
            while source.received < 8:
                await asyncio.sleep(.01)
            await asyncio.sleep(.05)
            sent = source.command('crop', around=(parse_header(gms3(0, 0, 0, b'')[:36]), .5, .5))
            command = json.loads(phone.recv(4096).decode().splitlines()[0])
            phone.close()
            await asyncio.wait_for(task, 5)
            return source, received, sent, command

        source, received, sent, command = asyncio.run(scenario())
        frames = [frame.data for frame in received if isinstance(frame.data, PhoneFrame)]
        self.assertEqual([frame.info.sequence for frame in frames if frame.recording], [2, 3, 4, 5])
        self.assertIn(6, [frame.info.sequence for frame in frames])  # the end of the take is never skipped
        self.assertEqual(received[-1].data, PhoneEvent('disconnected', 'stream ended'))
        self.assertEqual(command, sent)
        self.assertEqual(command, dict(cmd='crop', rect=[.3, 0., .4, 1.]))
        self.assertIsNone(source.command('record'))  # no phone connected any more


class CoinFlipPipelineTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.tmp = Path(self.directory.name)
        self.raw = self.tmp / 'in' / 'raw.gmst'
        self.raw.parent.mkdir()
        self.stream = recording()
        self.raw.write_bytes(b''.join(self.stream))

    def tearDown(self):
        self.directory.cleanup()

    def args(self, **options):
        values = dict(take=str(self.raw), record_dir=str(self.tmp / 'takes'), label=None, max_rev_s=None,
                      device='cpu', models=str(self.tmp / 'models'), store=':memory:', headless=True, report=None,
                      fast=False)
        values.update(options)
        return argparse.Namespace(**values)

    def run_pipeline(self, args, modules=None):
        from gamst import Gamst, recipes

        async def scenario():
            async with Gamst(consent=True) as human:
                recipe = recipes.load('coin-flip')
                pipeline = await recipes.build_pipeline(recipe, args, human)
                try:
                    await pipeline.run()
                    rows = await human.get_observations(capability_id='vision.rotation.refined', window='1h')
                    return pipeline.status(), pipeline.modules['record']['module'].saved, rows['items']
                finally:
                    await pipeline.close()
        return asyncio.run(scenario())

    def test_takes_are_recorded_and_counted_after_the_input_ends(self):
        fake = FakeRefine()
        with patch('gamst.takes.TakeFlips._launch', fake.launch):
            status, saved, records = self.run_pipeline(self.args())
        self.assertEqual(list(status['modules']), ['record', 'count'])  # find and follow run inside count
        self.assertEqual(status['inputs']['take']['derived_from'], 'record')
        self.assertEqual(len(saved), 2)
        # raw.gmst of each take is the phone stream as received
        recorded = [frame for frame in self.stream if parse_header(frame[:36]).flags & RECORDING]
        self.assertEqual((saved[0] / 'raw.gmst').read_bytes() + (saved[1] / 'raw.gmst').read_bytes(), b''.join(recorded))
        self.assertEqual(json.loads((saved[0] / 'summary.json').read_text())['frames'], 40)
        self.assertFalse((saved[0] / 'hud.mp4').exists())  # no view: nothing draws the screen
        # a replay counts its last take, once, with the recipe's parameters
        self.assertEqual([call['dir'] for call in fake.calls], [saved[1]])
        args = fake.calls[0]['args']
        self.assertEqual(args[args.index('--label') + 1], 'metal coin')
        self.assertEqual(args[args.index('--expected-max-rev-s') + 1], '40.0')
        self.assertNotIn('--fast', args)
        [record] = records
        attributes = record['observation']['value']['objects'][0]['attributes']
        self.assertEqual((attributes['flips'], attributes['trajectory']), (5., 'free_fall'))
        event = record['observation']['value']['events'][0]
        # the throw lands on the runtime clock at the moment it was captured
        self.assertEqual(record['timestamp_ms'] - attributes['started_ms'], 200)
        self.assertEqual(event['capture_timestamp_ms'], refined_results(saved[1], [])['results'][0]['timestamp_ms'])

    def test_counting_on_stop_counts_every_take(self):
        fake = FakeRefine()
        with patch('gamst.takes.TakeFlips._launch', fake.launch):
            _, saved, records = self.run_pipeline(self.args(count_on='stop'))
        self.assertEqual([call['dir'] for call in fake.calls], saved)
        self.assertEqual(len(records), 2)

    def test_switched_off_counter_records_takes_only(self):
        from gamst import Gamst, recipes
        fake = FakeRefine()

        async def scenario():
            async with Gamst(consent=True) as human:
                pipeline = await recipes.build_pipeline(recipes.load('coin-flip'), self.args(count_on='stop'), human)
                pipeline.disable('count')
                await pipeline.run()
                await pipeline.close()
        with patch('gamst.takes.TakeFlips._launch', fake.launch):
            asyncio.run(scenario())
        self.assertEqual(fake.calls, [])

    @unittest.skipUnless(HAS_HUD, 'needs opencv, numpy and Pillow (the HUD)')
    def test_recipe_run_builds_the_pipeline_with_the_hud(self):
        from gamst import recipes
        fake = FakeRefine()
        (self.raw.parent / 'summary.json').write_text(json.dumps(dict(frames=120, capture_fps=239.6)))
        report_path = self.tmp / 'report.json'
        with patch('gamst.takes.TakeFlips._launch', fake.launch), patch('gamst.takes.TakeFlips.open', lambda self: self):
            report = recipes.main(['run', 'coin-flip', '--take', str(self.raw.parent), '--label', 'gold coin',
                                   '--device', 'cpu', '--headless', '--record-dir', str(self.tmp / 'takes'),
                                   '--report', str(report_path)])
        self.assertEqual(report['frames'], 120)
        self.assertEqual(report['counts']['vision.rotation.refined'], {'rotation.ended': 1})
        self.assertEqual(report['pipeline']['modules']['count']['processed'], 2)  # both takes reach it
        args = fake.calls[0]['args']
        self.assertEqual(args[args.index('--label') + 1], 'gold coin')
        self.assertEqual(args[args.index('--device') + 1], 'cpu')
        self.assertIn('--fast', args)  # the recipe counts on ANALYZE with the models kept loaded
        takes = sorted((self.tmp / 'takes').glob('take-*'))
        self.assertEqual(len(takes), 2)
        self.assertGreater((takes[0] / 'hud.mp4').stat().st_size, 0)  # the HUD records its screen

    def test_rotation_demo_runs_the_same_pipeline(self):
        from gamst.demos import rotation
        fake = FakeRefine()
        with patch('gamst.takes.TakeFlips._launch', fake.launch), patch('gamst.recipes._view_class') as view:
            view.return_value = lambda pipeline, **_: _NoView()
            rotation.main(['--iphone-replay', str(self.raw), '--headless', '--record-dir', str(self.tmp / 'takes'),
                           '--refine', 'metal coin', '--efficienttam-source', str(self.tmp)])
        # the demo's defaults: count each take on Stop, every throw, one process per take
        self.assertEqual(len(fake.calls), 2)
        self.assertNotIn('--fast', fake.calls[0]['args'])


class _NoView:
    async def update(self, *_):
        return True

    def close(self):
        pass


if __name__ == '__main__':
    unittest.main()
