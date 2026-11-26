"""The run's voice sample from the same input (recipes/voice_sample.py), with stand-ins for the microphone and model."""
import argparse
import asyncio
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from gamst.recipes import enrollment, voice_sample

RECIPE = dict(name='person-term-count', references=[dict(name='voice', kind='voice')],
              modules=[dict(name='speaker', use='speaker-verify')])


def args(**given):
    base = dict(voice=None, exclude=[], record_voice_first=None, voice_range=None, voice_wav=None, voice_consent='owner test',
                voice_out=None, main_speaker_seconds=15., microphone=False, camera=None, iphone=None, video=None,
                audio=None, allow_remote_diarization=False)
    base.update(given)
    return argparse.Namespace(**base)


class Fake:
    def __init__(self):
        self.calls = []

    async def __call__(self, enroll_args):
        self.calls.append(enroll_args)
        if callable(getattr(enroll_args, 'meter', None)):
            enroll_args.meter(1., -20.)
        origin = dict(kind='main_speaker', segments_s=[[1., 4.]], window_s=[0., 15.]) if enroll_args.main_speaker \
            else dict(kind='range', from_s=enroll_args.start, to_s=enroll_args.start + enroll_args.seconds)
        return dict(enrolled_from=origin)


class PrepareTest(unittest.TestCase):
    def run_prepare(self, given, **extra):
        fake, out = Fake(), io.StringIO()
        sample = asyncio.run(voice_sample.prepare(RECIPE, given, enroll_voice=fake, out=out, **extra))
        return sample, fake, out.getvalue()

    def test_nothing_changes_without_the_options(self):
        given = args(voice_consent=None)
        self.assertIsNone(asyncio.run(voice_sample.prepare(RECIPE, given, enroll_voice=Fake())))
        self.assertIsNone(asyncio.run(voice_sample.prepare(RECIPE, args(voice='v.json'), enroll_voice=Fake())))

    def test_live_input_records_first_on_the_same_input_with_countdown_meter_and_re_record(self):
        answers = iter([True, False])
        given = args(microphone=True, record_voice_first=8.)
        sample, fake, said = self.run_prepare(given, ask=lambda question: next(answers))
        self.assertEqual(len(fake.calls), 2)        # recorded again once
        self.assertTrue(fake.calls[0].microphone and fake.calls[0].seconds == 8. and fake.calls[0].consent == 'owner test')
        self.assertIn('세려는 사람만 8초 동안 말해 주세요', said)
        self.assertIn('consent: owner test', said)
        self.assertIn('[####', said)
        self.assertEqual(sample['kind'], 'recorded_first')
        self.assertTrue(given.voice)
        voice_sample.cleanup(given)

    def test_a_live_run_without_a_recording_step_is_refused(self):
        with self.assertRaises(SystemExit):
            self.run_prepare(args(microphone=True))

    def test_a_recording_uses_the_range_then_a_wav_then_the_main_speaker(self):
        sample, fake, _ = self.run_prepare(args(video='a.mp4', voice_range='31-38.5'))
        self.assertEqual((sample['kind'], sample['from_s'], sample['to_s'], fake.calls[0].video), ('range', 31., 38.5, 'a.mp4'))
        sample, fake, _ = self.run_prepare(args(video='a.mp4', voice_wav='me.wav'))
        self.assertEqual((sample['kind'], fake.calls[0].audio, fake.calls[0].video), ('file', 'me.wav', None))
        with self.assertRaises(SystemExit):        # the main speaker is hosted: its own consent
            self.run_prepare(args(video='a.mp4'))
        sample, fake, _ = self.run_prepare(args(video='a.mp4', allow_remote_diarization=True))
        self.assertEqual((sample['kind'], sample['segments_s'], fake.calls[0].seconds, fake.calls[0].main_speaker),
                         ('main_speaker', [[1., 4.]], 15., True))

    def test_consent_is_required(self):
        with self.assertRaises(SystemExit):
            self.run_prepare(args(video='a.mp4', voice_range='1-3', voice_consent=None))


class EnrollFromSameInputTest(unittest.TestCase):
    def test_the_window_is_cut_exactly_and_the_level_is_shown(self):
        class Source:
            async def frames(self):
                for index in range(10):      # 10 frames of 0.5 s
                    yield argparse.Namespace(data=np.full(8000, .1 * (index + 1), dtype=np.float32), sample_rate=16000)

        class Embedder:
            model_id = 'stand-in'

            def __init__(self):
                self.lengths = []

            def embed(self, samples, rate):
                self.lengths.append(len(samples))
                return [1., 0., 0.]

        embedder, levels = Embedder(), []

        async def stand_in(item, context):
            return embedder
        with tempfile.TemporaryDirectory() as folder, mock.patch('gamst.recipes.perception_parts.speaker_embedder', stand_in):
            given = argparse.Namespace(consent='owner test', start=1., seconds=3., out=os.path.join(folder, 'v.json'),
                                       main_speaker=False, meter=lambda at, dbfs: levels.append((at, dbfs)))
            saved = asyncio.run(enrollment.enroll_voice(RECIPE, given, source=Source(), context={}))
            self.assertEqual(saved['enrolled_from'], dict(kind='range', from_s=1., to_s=4.))
            self.assertEqual(sum(embedder.lengths), 3 * 16000)
            self.assertEqual(oct(os.stat(given.out).st_mode & 0o777), '0o600')
            self.assertEqual(json.load(open(given.out))['seconds'], 3.)
            self.assertTrue(levels and levels[-1][0] == 3.)


if __name__ == '__main__':
    unittest.main()
