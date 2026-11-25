"""Several inputs in one run. Fake modules and synthetic media: this checks plumbing and times, no model's accuracy."""
import argparse
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np

from gamst import Gamst, recipes
from gamst.capabilities import CapabilitySpec, OutputField
from gamst.inputs import AudioFileInput, RecordedClock, SensorFileInput, SoundTrackInput, media_streams
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.recipes.parts import PARTS, Part
from gamst.schema import timestamp

try:
    import cv2  # noqa: F401
    HAS_CV = True
except ImportError:
    HAS_CV = False
HAS_FFMPEG = bool(shutil.which('ffmpeg') and shutil.which('ffprobe'))
FRAME_MS = 1000 / 30


def ffmpeg(*arguments):
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', *map(str, arguments)], check=True, timeout=60,
                   stdin=subprocess.DEVNULL, capture_output=True)


def media(path, *, video_late=0., audio_late=0., sound=True, audio_filter=None):
    """4 s, 30 fps: one white frame at 2.0 s of the pictures, one click at 2.0 s of the sound."""
    sources = ['-itsoffset', video_late, '-f', 'lavfi', '-i', 'color=black:s=96x64:r=30:d=4']
    if sound:
        sources += ['-itsoffset', audio_late, '-f', 'lavfi', '-i',
                    "aevalsrc='if(between(t,2,2.005),0.9*sin(2*PI*1000*t),0)':s=48000:d=4"]
    codec = ['-c:a', 'aac'] if sound else []
    if audio_filter:
        codec = ['-af', audio_filter, '-c:a', 'pcm_s16le']
    ffmpeg(*sources, '-vf', "drawbox=c=white:t=fill:enable='between(n,60,60)'", '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
           *codec, path)
    return str(path)


def wav(path, samples, rate=16000):
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1), out.setsampwidth(2), out.setframerate(rate)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
    return str(path)


def spec(name, modality):
    return CapabilitySpec(name, name, modality, 'test', '1', (OutputField('count', 'integer'), OutputField('events', 'array')),
                          'native')


class Flash:
    spec = spec('test.flash', 'video')

    def process(self, frame):
        if frame.data.mean() < 100:
            return None
        return dict(count=1, events=[dict(event='flash', start_ms=frame.timestamp_ms, end_ms=frame.timestamp_ms)])


class Click:
    spec = spec('test.click', 'audio')

    def process(self, frame):
        loud = np.flatnonzero(np.abs(frame.data) > .3)
        if not len(loud):
            return None
        at = frame.timestamp_ms - round(1000 * (len(frame.data) - int(loud[0])) / frame.sample_rate)
        return dict(count=1, events=[dict(event='click', start_ms=at, end_ms=at)])


def fake(module):
    async def build(options, context):
        built = module()
        return dict(module=built, spec=built.spec)
    return Part('module', build=build)


CAMERA = dict(id='camera', use='camera', sources=['camera', 'video'])
MICROPHONE = dict(id='microphone', use='microphone', sources=['microphone', 'audio', 'video'])
PAIR = dict(name='pair', summary='s', inputs=[CAMERA, MICROPHONE], view='count',
            modules=[dict(name='flash', use='t-flash', consumes=['camera'], optional='on'),
                     dict(name='click', use='t-click', consumes=['microphone'], optional='on')],
            records=[dict(id='test.flash', **{'for': 'flash'}), dict(id='test.click', **{'for': 'click'})])


class Recipes(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patcher = mock.patch.dict(PARTS, {'t-flash': fake(Flash), 't-click': fake(Click)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def recipe(self, value=PAIR):
        path = self.tmp / f"{value['name']}.json"
        path.write_text(json.dumps(value))
        return str(path)

    def run_recipe(self, *arguments, recipe=PAIR):
        return recipes.main(['run', self.recipe(recipe), '--headless', *map(str, arguments)])

    def refused(self, text, *arguments, recipe=PAIR):
        with self.assertRaisesRegex(SystemExit, text):
            self.run_recipe(*arguments, recipe=recipe)


@unittest.skipUnless(HAS_CV and HAS_FFMPEG, 'needs opencv and ffmpeg')
class OneFile(Recipes):
    def times(self, report):
        (flash,), (click,) = report['events']['test.flash'], report['events']['test.click']
        return flash['start_ms'] - report['start_ms'], click['start_ms'] - report['start_ms']

    def test_flash_and_click_of_one_file_are_within_a_frame(self):
        report = self.run_recipe('--video', media(self.tmp / 'a.mp4'))
        flash, click = self.times(report)
        self.assertLessEqual(abs(flash - 2000), FRAME_MS)
        self.assertLessEqual(abs(click - flash), FRAME_MS)
        self.assertTrue(report['complete'])
        self.assertEqual(report['input_frames']['camera'], 120)
        self.assertEqual(report['frames'], 120)             # of the first input
        self.assertTrue(all(item['ended'] for item in report['pipeline']['inputs'].values()))

    def test_stream_start_offsets_of_the_container_are_kept(self):
        for kind, name, video, audio in (('late sound', 'b.mp4', 0., .5), ('late pictures', 'c.mp4', .5, 0.),
                                         ('late sound', 'd.mkv', 0., .5)):
            with self.subTest(name):
                path = media(self.tmp / name, video_late=video, audio_late=audio)
                flash, click = self.times(self.run_recipe('--video', path))
                self.assertLessEqual(abs(click - flash - 1000 * (audio - video)), FRAME_MS, kind)

    def test_a_stretch_without_sound_does_not_shift_what_follows(self):
        path = media(self.tmp / 'gap.mkv', audio_filter="aselect='not(between(t,0.5,1.5))'")
        flash, click = self.times(self.run_recipe('--video', path))
        self.assertLessEqual(abs(click - flash), FRAME_MS)

    def test_an_offset_is_applied_and_reported(self):
        report = self.run_recipe('--video', media(self.tmp / 'a.mp4'), '--offset-ms', 'camera=-80')
        (flash,), (click,) = report['events']['test.flash'], report['events']['test.click']
        self.assertLessEqual(abs(click['start_ms'] - flash['start_ms'] - 80), FRAME_MS)
        self.assertEqual(report['pipeline']['inputs']['camera']['offset_ms'], -80)
        self.assertEqual(report['pipeline']['inputs']['microphone']['offset_ms'], 0)

    def test_both_are_in_the_same_timeline_row(self):
        async def scenario():
            recipe = recipes.load(self.recipe())
            args = argparse.Namespace(video=media(self.tmp / 'a.mp4'), capture_fps=None)
            async with Gamst(consent=True) as human:
                pipeline = await recipes.build_pipeline(recipe, args, human)
                ahead = []

                def on_frame(input_id, frame, records):
                    last = {key: value['last_ms'] for key, value in pipeline.inputs.items()}
                    if None not in last.values():
                        ahead.append(abs(last['camera'] - last['microphone']))

                await pipeline.run(on_frame=on_frame)
                start = pipeline.inputs['camera']['first_ms']
                answer = await human.get_timeline(start_ms=start + 1000, end_ms=start + 3000, step_ms=100,
                                                  capabilities=['test.flash', 'test.click'])
                await pipeline.close()
            return answer, max(ahead)

        answer, ahead = asyncio.run(scenario())
        rows = [row for row in answer['rows'] if any(cell['status'] == 'exact' for cell in row['cells'].values())]
        self.assertLessEqual(ahead, 100 + FRAME_MS)       # neither runs further ahead than one block of sound
        seen = {item['capability']: item['start_ms'] for item in answer['evidence'].values() if item['kind'] == 'events'}
        self.assertEqual(set(seen), {'test.flash', 'test.click'})
        self.assertLessEqual(abs(seen['test.flash'] - seen['test.click']), FRAME_MS)
        # the same moment is the same row (or, at a row's edge, the next one); the click's record is one block later
        self.assertLessEqual(rows[-1]['t_ms'] - rows[0]['t_ms'], 200)
        self.assertTrue(all(row['complete'] for row in rows))

    def test_what_is_not_read_is_not_opened(self):
        path = media(self.tmp / 'silent.mp4', sound=False)
        self.refused('no sound track.*--without click', '--video', path)
        report = self.run_recipe('--video', path, '--without', 'click')
        self.assertEqual((report['frames'], list(report['counts'])), (120, ['test.flash']))
        self.assertNotIn('input_frames', report)
        with mock.patch('gamst.inputs.shutil.which', return_value=None):
            self.refused('ffprobe.*not on PATH', '--video', media(self.tmp / 'a.mp4'))
            self.run_recipe('--video', path, '--without', 'click')

    def test_frames_stops_the_whole_run(self):
        report = self.run_recipe('--video', media(self.tmp / 'a.mp4'), '--frames', 30)
        self.assertEqual(report['input_frames']['camera'], 30)
        self.assertLessEqual(report['input_frames']['microphone'], 11)

    def test_truth_is_scored_from_the_start_of_the_recording(self):
        path = media(self.tmp / 'b.mp4', video_late=.5)
        truth = self.tmp / 'truth.json'
        truth.write_text(json.dumps(dict(capability='test.flash', event='flash', count=1,
                                         intervals=[dict(start_s=2.45, end_s=2.55)])))
        report = self.run_recipe('--video', path, '--truth', truth)
        self.assertEqual(report['score']['count_error'], 0)

    def test_the_stream_is_read_in_chunks_and_stops_with_its_reader(self):
        path = media(self.tmp / 'a.mp4')

        async def scenario():
            source = SoundTrackInput(path, start_s=1.95, end_s=2.1)
            frames = [frame async for frame in source.frames()]
            broken = SoundTrackInput(self.tmp / 'a.mp4')
            stream = broken.frames()
            await stream.__anext__()
            process = broken._process
            await stream.aclose()
            return source, frames, process.returncode

        source, frames, code = asyncio.run(scenario())
        self.assertEqual(sum(len(frame.data) for frame in frames), 2400)
        self.assertEqual(frames[-1].timestamp_ms - source.origin_ms, 2100)
        self.assertIsNotNone(code)

    def test_only_local_files_are_handed_to_ffmpeg(self):
        command = SoundTrackInput('-http://x/y.mp4').command()
        self.assertTrue(command[command.index('-i') + 1].startswith('file:/'))
        self.assertEqual(command[command.index('-protocol_whitelist') + 1], 'file')
        with self.assertRaisesRegex(RuntimeError, 'cannot open'):
            media_streams('http://example.invalid/a.mp4')
        (self.tmp / 'bad.mp4').write_bytes(b'not a video')
        self.refused('bad.mp4', '--video', self.tmp / 'bad.mp4')


class Choosing(Recipes):
    def test_every_input_says_where_it_is_read_from(self):
        self.refused('source for camera.*--camera.*--video.*--without flash', '--microphone')
        self.refused('source for microphone.*--without click', '--camera', 0)
        self.refused('no common capture clock', '--camera', 0, '--audio', 'x.wav')
        self.refused('2', '--audio', 'x.wav', '--camera', 0, '--without', 'click')     # one input left, one source
        self.refused('reads nothing from --sensor', '--camera', 0, '--microphone', '--sensor', 'x.csv')
        self.refused('no input nobody', '--input', 'nobody=camera:0')
        self.refused('different recordings.*--offset-ms microphone', '--video', 'a.mp4', '--audio', 'b.wav')
        self.refused('off by at most', '--camera', 0, '--microphone', '--offset-ms', 'camera=-9000')
        twice = dict(PAIR, inputs=[CAMERA, dict(CAMERA, id='other')], modules=[
            dict(name='flash', use='t-flash', consumes=['camera']), dict(name='second', use='t-flash', consumes=['other'])])
        self.refused('two inputs would read the same', '--camera', 0, recipe=twice)

    def test_one_input_recipes_are_as_before(self):
        with self.assertRaises(SystemExit) as caught, mock.patch('sys.stderr'):
            recipes.main(['run', 'person-action-count', '--camera', '0', '--video', 'x.mp4', '--headless'])
        self.assertEqual(caught.exception.code, 2)
        with self.assertRaisesRegex(SystemExit, 'reads from camera, video, take, not microphone'):
            recipes._source(recipes.load('person-action-count'), argparse.Namespace(microphone=True))
        for name in recipes.names():
            recipe = recipes.load(name)
            if name != 'scene':
                self.assertEqual(len(recipe['inputs']), 1, name)

    def test_recordings_of_different_files_relate_as_the_owner_says(self):
        samples = np.zeros(16000, np.float32)
        first = dict(PAIR, inputs=[MICROPHONE, dict(MICROPHONE, id='other')], modules=[
            dict(name='click', use='t-click', consumes=['microphone']),
            dict(name='second', use='t-click2', consumes=['other'])], records=['test.click', 'test.click2'])

        class Click2(Click):
            spec = spec('test.click2', 'audio')

        samples[8000:8080] = .9
        a, b = wav(self.tmp / 'a.wav', samples), wav(self.tmp / 'b.wav', samples)
        with mock.patch.dict(PARTS, {'t-click2': fake(Click2)}):
            report = self.run_recipe('--input', f'microphone=audio:{a}', '--input', f'other=audio:{b}',
                                     '--offset-ms', 'other=250', recipe=first)
        (one,), (two,) = report['events']['test.click'], report['events']['test.click2']
        self.assertEqual(two['start_ms'] - one['start_ms'], 250)

    def test_sampling_limits_are_of_the_module_s_own_input(self):
        value = dict(PAIR, modules=[dict(name='a', use='t-rate', consumes=['camera']),
                                    dict(name='b', use='t-rate', consumes=['microphone'])])
        seen = []

        def rule(options, fps, **_):
            seen.append((options['name'], fps))
            return dict(required_fps=10, ok=None)

        with mock.patch.dict(PARTS, {'t-rate': Part('module', build=fake(Flash).build, sampling=rule)}):
            recipes.check(recipes.load(self.recipe(dict(value, records=[]))), fps=30)
        self.assertEqual(seen, [('a', 30), ('b', None)])

    def test_the_vlm_s_output_is_never_counted(self):
        with self.assertRaisesRegex(ValueError, 'never counted'):
            recipes.load(self.recipe(dict(PAIR, records=['vision.streaming_semantic'])))


class Live:
    """A live source with given capture times."""
    live = True

    def __init__(self, modality, stamps, *, gap_at=None, fail_at=None, hold=False):
        self.modality, self.stamps, self.gap_at, self.fail_at, self.hold = modality, stamps, gap_at, fail_at, hold
        self.missed, self.closed = 0, False

    async def frames(self):
        for index, stamp in enumerate(self.stamps):
            if index == self.fail_at:
                raise OSError('the device went away')
            if index == self.gap_at:
                self.missed += 3
            data = np.zeros(1600, np.float32) if self.modality == 'audio' else np.zeros((4, 4, 3), np.uint8)
            yield InputFrame(data, self.modality, stamp, sample_rate=16000 if self.modality == 'audio' else None)
            await asyncio.sleep(0)
        if self.hold:
            await asyncio.Event().wait()

    def close(self):
        self.closed = True


class Recorded(Live):
    live = False


class Recorder:
    def __init__(self, name, modality):
        self.spec, self.seen = spec(name, modality), []

    def process(self, frame):
        self.seen.append(frame.timestamp_ms)
        return dict(count=len(self.seen), events=[])


class Runs(unittest.IsolatedAsyncioTestCase):
    async def start(self, human, camera, microphone, **options):
        pipeline = Pipeline(human, 'two')
        pipeline.add_input('camera', camera, **options).add_input('microphone', microphone)
        self.eye, self.ear = Recorder('test.eye', 'video'), Recorder('test.ear', 'audio')
        pipeline.add_module('eye', self.eye, consumes=['camera'], spec=self.eye.spec)
        pipeline.add_module('ear', self.ear, consumes=['microphone'], spec=self.ear.spec)
        return pipeline

    async def test_recorded_inputs_are_read_in_capture_order_and_end_together(self):
        base = timestamp() - 60000
        order = []
        async with Gamst(consent=True) as human:
            pipeline = await self.start(human, Recorded('video', [base + 40 * i for i in range(20)]),
                                        Recorded('audio', [base + 100 * i for i in range(1, 30)]))
            await pipeline.run(on_frame=lambda key, frame, _: order.append((frame.timestamp_ms, key)))
            status = pipeline.status()
            await pipeline.close()
        self.assertEqual([stamp for stamp, _ in order], sorted(stamp for stamp, _ in order))
        self.assertEqual((len(self.eye.seen), len(self.ear.seen)), (20, 29))   # the longer one ran to its end
        self.assertTrue(status['complete'])
        # the ended input holds back nobody: nothing was captured on it afterwards
        self.assertEqual(status['modules']['eye']['complete_until_ms'], base + 2900)

    async def test_the_order_follows_the_offsets(self):
        base = timestamp() - 60000
        order = []
        async with Gamst(consent=True) as human:
            pipeline = await self.start(human, Recorded('video', [base + 40 * i for i in range(5)]),
                                        Recorded('audio', [base + 100]), offset_ms=-150)
            await pipeline.run(on_frame=lambda key, frame, _: order.append(key))
            await pipeline.close()
        self.assertEqual(self.eye.seen, [base + 40 * i - 150 for i in range(5)])
        self.assertEqual(order, ['camera'] * 5 + ['microphone'])
        async with Gamst(consent=True) as human:
            with self.assertRaisesRegex(ValueError, 'offset_ms'):
                Pipeline(human, 'x').add_input('camera', offset_ms=1.5)

    async def test_live_and_recorded_do_not_mix(self):
        async with Gamst(consent=True) as human:
            pipeline = await self.start(human, Live('video', []), Recorded('audio', []))
            with self.assertRaisesRegex(ValueError, 'no common capture clock'):
                await pipeline.run()
            await pipeline.close()

    async def test_a_gap_and_a_pause_belong_to_their_own_input(self):
        base = timestamp() - 20000
        async with Gamst(consent=True) as human:
            pipeline = await self.start(human, Live('video', [base + 40 * i for i in range(30)], gap_at=10),
                                        Live('audio', [base + 100 * i for i in range(1, 12)]))
            await pipeline.run()
            status = pipeline.status()
            self.assertEqual((status['modules']['eye']['gaps'], status['modules']['ear']['gaps']), (1, 0))
            self.assertEqual((status['inputs']['camera']['missed'], status['inputs']['microphone']['missed']), (3, 0))
            await human.pause()
            for key in ('camera', 'microphone'):
                await pipeline.push(key, np.zeros(1600, np.float32) if key == 'microphone' else np.zeros((4, 4, 3), np.uint8),
                                    timestamp_ms=base + 5000, modality='audio' if key == 'microphone' else 'video')
            await human.resume()
            await pipeline.push('microphone', np.zeros(1600, np.float32), timestamp_ms=base + 5100, modality='audio',
                                sample_rate=16000)
            reasons = {key: [gap['reason'] for gap in value['gap_intervals']]
                       for key, value in pipeline.status()['modules'].items()}
            self.assertEqual(reasons, dict(eye=['input'], ear=['capture_paused']))
            await pipeline.close()

    async def test_one_failing_input_stops_the_run_and_is_named(self):
        base = timestamp() - 20000
        for first, second, failed in ((None, 3, 'microphone'), (2, None, 'camera')):
            with self.subTest(failed):
                async with Gamst(consent=True) as human:
                    pipeline = await self.start(human, Live('video', [base + 40 * i for i in range(5)], fail_at=first, hold=True),
                                                Live('audio', [base + 100 * i for i in range(1, 5)], fail_at=second, hold=True))
                    with self.assertRaisesRegex(OSError, 'went away'):
                        await asyncio.wait_for(pipeline.run(), 5)
                    inputs = pipeline.status()['inputs']
                    self.assertIn('went away', inputs[failed]['error'])
                    self.assertEqual(sum('error' in value for value in inputs.values()), 1)
                    await pipeline.close()

    async def test_a_silent_live_input_does_not_keep_a_stopped_run(self):
        base = timestamp() - 20000
        async with Gamst(consent=True) as human:
            pipeline = await self.start(human, Live('video', [base + 40 * i for i in range(50)]), Live('audio', [], hold=True))
            await asyncio.wait_for(pipeline.run(on_frame=lambda key, frame, _: len(self.eye.seen) < 10), 5)
            self.assertEqual(len(self.eye.seen), 10)
            await pipeline.close()


class Sensors(Recipes):
    ITEM = dict(id='imu', use='sensor', sources=['sensor'], modality='imu', time=dict(column='t', unit='s'),
                columns=[dict(name='ax', unit='m/s2', scale=10), dict(name='ay', unit='m/s2')])

    def csv(self, name='s.csv', seconds=20, rate=50, wave=lambda t: 10 * np.sin(2 * np.pi * t), hole=None):
        lines = ['t,ay,ax,temp'] + [f"{5 + i / rate},{'' if hole == i else 1.0},{wave(i / rate):.6f},20" for i in range(seconds * rate)]
        path = self.tmp / name
        path.write_text('\n'.join(lines))
        return str(path)

    def frames(self, path, **options):
        async def scenario():
            source = SensorFileInput(path, **{**{key: self.ITEM[key] for key in ('columns', 'time')}, **options})
            return source, [frame async for frame in source.frames()]
        return asyncio.run(scenario())

    def test_declared_columns_units_and_time(self):
        source, frames = self.frames(self.csv(seconds=2, hole=3))
        self.assertEqual([frame.timestamp_ms - source.origin_ms for frame in frames[:3]], [0, 20, 40])
        self.assertEqual(frames[0].data, dict(channels=[0., 1.], mask=[True, True]))
        self.assertEqual(frames[3].data['mask'], [True, False])        # a missing value is not a number
        self.assertEqual(source.contract, 'sensor.imu:ax[m/s2]/10,ay[m/s2]')
        self.assertLess(frames[-1].timestamp_ms, timestamp())
        lines = self.tmp / 's.jsonl'
        lines.write_text('{"t": 1000, "ax": 5, "ay": null}\n{"t": 1010, "ax": 5, "ay": 2}\n')
        source, frames = self.frames(lines, time=dict(column='t', unit='ms'))
        self.assertEqual([frame.data['mask'] for frame in frames], [[True, False], [True, True]])
        self.assertEqual(frames[1].timestamp_ms - frames[0].timestamp_ms, 10)

    def test_refusals(self):
        (self.tmp / 'back.csv').write_text('t,ax,ay\n1,0,0\n1,0,0\n')
        (self.tmp / 'nan.csv').write_text('t,ax,ay\nnan,0,0\n')
        (self.tmp / 'short.csv').write_text('t,ax\n1,0\n')
        (self.tmp / 'fast.csv').write_text('t,ax,ay\n' + ''.join(f'{i / 5000},0,0\n' for i in range(100)))
        for name, text in (('back.csv', 'not later'), ('nan.csv', 'no time'), ('short.csv', 'no column ay'),
                           ('fast.csv', '1000 samples')):
            with self.assertRaisesRegex(ValueError, text):
                self.frames(self.tmp / name)
        for options in (dict(modality='video'), dict(columns=[dict(name='a')]), dict(time=dict(column='t', unit='h'))):
            with self.assertRaises(ValueError):
                self.frames('x.csv', **options)

    def test_repetition_rhythm_and_exemplar_read_a_sensor(self):
        recipe = dict(name='shake', summary='s', inputs=[self.ITEM], view='count', references=[dict(name='exemplar', kind='exemplar')],
                      modules=[dict(name='repeats', use='repetition', consumes=['imu'], max_reps_s=3),
                               dict(name='rhythm', use='rhythm', consumes=['imu'], channel='ax', config=dict(band=[.5, 3.])),
                               dict(name='match', use='exemplar', mode='sequence', consumes=['imu'])],
                      records=['temporal.repetition_count', 'temporal.exemplar_events'])
        path = self.csv(seconds=40, wave=lambda t: 10 * np.sin(np.pi * (t % 2)) ** 2 if t % 2 < 1 else 0.)   # every 2 s
        out = self.tmp / 'move.json'
        recipes.main(['enroll', self.recipe(recipe), '--sensor', f'{path}@4-5', '--out', str(out)])
        saved = json.loads(out.read_text())
        self.assertEqual(saved['channels'], 'sensor.imu:ax[m/s2]/10,ay[m/s2]')
        report = self.run_recipe('--sensor', path, '--exemplar', out, recipe=recipe)
        self.assertAlmostEqual(report['counts']['temporal.repetition_count']['repetition'], 20, delta=2)
        self.assertEqual(report['counts']['temporal.exemplar_events']['exemplar.matched'], 20)
        self.assertEqual(report['pipeline']['inputs']['imu']['state']['columns'][0]['unit'], 'm/s2')
        other = dict(recipe, inputs=[dict(self.ITEM, columns=[dict(name='ax', unit='g'), dict(name='ay', unit='g')])])
        with self.assertRaisesRegex(ValueError, 'specified on sensor.imu'):
            self.run_recipe('--sensor', path, '--exemplar', out, recipe=other)
        found = recipes.main(['find', self.recipe(recipe), '--sensor', path, '--exemplar', str(out)])
        self.assertEqual(len(found[0]), 20)

    def test_a_sensor_shares_the_clock_of_a_recording(self):
        recipe = dict(PAIR, inputs=[MICROPHONE, self.ITEM], records=['test.click'], modules=[
            dict(name='click', use='t-click', consumes=['microphone']),
            dict(name='repeats', use='repetition', consumes=['imu'])])
        samples = np.zeros(16000 * 5, np.float32)
        samples[16000:16080] = .9
        report = self.run_recipe('--audio', wav(self.tmp / 'a.wav', samples), '--sensor', self.csv(seconds=5),
                                 '--offset-ms', 'imu=0', recipe=recipe)
        inputs = report['pipeline']['inputs']
        self.assertEqual(inputs['microphone']['last_ms'] - inputs['imu']['last_ms'], 5000 - 4980)

    def test_the_host_pushes_a_live_sensor(self):
        async def scenario():
            recipe = recipes.load(self.recipe(dict(name='live', summary='s', inputs=[self.ITEM], view='count',
                                                   modules=[dict(name='repeats', use='repetition', consumes=['imu'])])))
            async with Gamst(consent=True) as human:
                pipeline = await recipes.build_pipeline(recipe, argparse.Namespace(), human, sources=dict(imu=None))
                now = timestamp() - 10000
                for i in range(300):
                    await pipeline.push('imu', dict(channels=[float(np.sin(2 * np.pi * i / 50)), 0.], mask=[True, True]),
                                        timestamp_ms=now + 20 * i)
                status = pipeline.status()['modules']['repeats']
                await pipeline.close()
                return status
        status = asyncio.run(scenario())
        self.assertEqual((status['processed'], status['errors']), (300, 0))


class Clock(unittest.TestCase):
    def test_the_last_recording_ends_one_second_ago(self):
        clock = RecordedClock()
        clock.add(3.)
        clock.add(10.)
        self.assertAlmostEqual(timestamp() - clock.base_ms, 11000, delta=200)
        with self.assertRaises(RuntimeError):
            clock.add(1.)

    def test_a_part_of_a_wav(self):
        async def scenario(path):
            source = AudioFileInput(path, start_s=.25, end_s=.6)
            return source, [frame async for frame in source.frames()]
        with tempfile.TemporaryDirectory() as tmp:
            source, frames = asyncio.run(scenario(wav(Path(tmp) / 'a.wav', np.arange(16000) / 16000)))
        samples = np.concatenate([frame.data for frame in frames])
        self.assertEqual(len(samples), round(.35 * 16000))
        self.assertAlmostEqual(float(samples[0]), .25, places=3)
        self.assertEqual(frames[-1].timestamp_ms - source.origin_ms, 600)


if __name__ == '__main__':
    unittest.main()
