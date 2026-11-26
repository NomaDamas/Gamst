"""Audio-visual speaker evidence: the owner's rule (voice first, lips only in the band), the face gate, the records.

The rule and the module plumbing run with stand-in workers; ``RealLightASDTest`` loads the real Light-ASD weights
when ``GAMST_LIGHTASD_SOURCE`` and ``GAMST_LIGHTASD_CHECKPOINT`` name them (skipped otherwise).
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import unittest

import numpy as np

from gamst.capabilities import is_raw_key
from gamst.plugins import InputFrame
from gamst.private_payload import CONTRACTS
from gamst.providers.active_speaker import ActiveSpeaker, check_weights, face_geometry, pack_window, unpack_window
from gamst.providers.speaker_evidence import SpeakerEvidence
from gamst.providers.term_events import TermEvents

HAS_CV2 = importlib.util.find_spec('cv2') is not None


def lips_record(start_ms, faces):
    slots = max(len(face['speaking']) for face in faces) if faces else 25
    return dict(start_ms=start_ms, end_ms=start_ms + 40 * slots, slot_ms=40, faces=faces, model={})


def utterance(words, end_ms=None):
    return dict(kind='utterance', reference='voice', start_ms=words[0]['start_ms'], end_ms=end_ms or words[-1]['end_ms'],
                match=False, state='other', utterance_id='u1', status='final', text=' '.join(w['text'] for w in words),
                words=words, speaker=dict(reference='voice', match=False, model='m'), model=dict(id='m'))


def word(start, end, score, text='뽀또'):
    return dict(text=text, start_ms=start, end_ms=end, match=score >= .35, speaker_score=score)


def frame(value, ts=0):
    return InputFrame(value, 'record', ts)


def decide(record_faces, words):
    module = SpeakerEvidence(threshold=.35)
    module.process(frame(lips_record(0, record_faces)))
    return module.process(frame(utterance(words)))[0]['words']


class RuleTest(unittest.TestCase):
    """Band [0.15, 0.45) around threshold 0.35."""

    def test_a_clear_voice_is_never_overruled(self):
        speaking_other = [dict(track='face-2', identity=.02, speaking=[3.] * 25)]
        high, low = decide(speaking_other, [word(100, 400, .55), word(500, 800, .10)])
        self.assertEqual((high['match'], high['decided_by']), (True, 'voice'))
        self.assertEqual((low['match'], low['decided_by']), (False, 'voice'))
        self.assertNotIn('lips', high)

    def test_the_enrolled_face_speaking_counts_a_word_the_voice_missed(self):
        (item,) = decide([dict(track='face-1', identity=.55, speaking=[1.] * 25)], [word(100, 400, .25)])
        self.assertEqual((item['match'], item['voice_match'], item['decided_by']), (True, False, 'lips'))
        self.assertEqual(item['lips']['verdict'], 'for')

    def test_another_face_speaking_excludes_and_agreement_says_both(self):
        other = [dict(track='face-2', identity=.03, speaking=[2.] * 25)]
        against, agrees = decide(other, [word(100, 400, .40), word(500, 800, .30)])
        self.assertEqual((against['match'], against['decided_by']), (False, 'lips'))
        self.assertEqual((agrees['match'], agrees['decided_by']), (False, 'both'))

    def test_an_unreadable_or_silent_enrolled_face_is_no_evidence_against(self):
        unseen = [dict(track='face-1', identity=.55, speaking=[None] * 25)]
        silent = [dict(track='face-1', identity=.55, speaking=[-2.] * 25)]
        for faces in (unseen, silent, []):
            (item,) = decide(faces, [word(100, 400, .40)])
            self.assertEqual((item['match'], item['decided_by'], item['lips']['verdict']), (True, 'voice', 'unknown'))

    def test_both_speaking_or_unclear_identity_is_unknown(self):
        both = [dict(track='face-1', identity=.55, speaking=[1.] * 25), dict(track='face-2', identity=.0, speaking=[1.] * 25)]
        unclear = [dict(track='face-3', identity=.25, speaking=[3.] * 25)]
        for faces in (both, unclear):
            (item,) = decide(faces, [word(100, 400, .30)])
            self.assertEqual((item['match'], item['decided_by']), (False, 'voice'))

    def test_too_few_readable_slots_are_unknown(self):
        (item,) = decide([dict(track='face-1', identity=.55, speaking=[None] * 20 + [1.] * 5)], [word(100, 300, .30)])
        self.assertEqual(item['lips']['verdict'], 'unknown')

    def test_a_word_waits_for_its_lips_and_the_end_releases_it(self):
        module = SpeakerEvidence(threshold=.35)
        self.assertIsNone(module.process(frame(utterance([word(900, 1200, .3)]))))
        self.assertIsNone(module.process(frame(lips_record(0, [dict(track='face-1', identity=.6, speaking=[1.] * 25)]))))
        (released,) = module.process(frame(lips_record(1000, [dict(track='face-1', identity=.6, speaking=[1.] * 25)])))
        self.assertEqual(released['words'][0]['decided_by'], 'lips')
        module.process(frame(utterance([word(5000, 5300, .3)])))
        (late,) = module.finish()
        self.assertEqual(late['words'][0]['lips']['verdict'], 'unknown')

    def test_band_edges_are_parameters(self):
        with self.assertRaises(ValueError):
            SpeakerEvidence(threshold=.35, low=.4)
        module = SpeakerEvidence(threshold=.5)
        self.assertEqual((module.low, module.high), (.3, .6))

    def test_terms_count_by_the_decision_and_say_what_decided(self):
        module = SpeakerEvidence(threshold=.35)
        module.process(frame(lips_record(0, [dict(track='face-1', identity=.6, speaking=[1.] * 25)])))
        (value,) = module.process(frame(utterance([word(100, 400, .25)])))
        terms = TermEvents(terms=['뽀또'], report_others=True)
        result = terms.process(frame(value, 1000))
        (event,) = result['events']
        self.assertEqual((result['count'], event['event'], event['decided_by']), (1, 'term.spoken', 'lips'))
        self.assertEqual(event['speaker_score'], .25)


class GeometryTest(unittest.TestCase):
    def rows(self, nose_x):
        # eyes, nose, mouth corners, inner lips, extent (normalized, 100 x 100 picture)
        return [(.40, .40), (.60, .40), (nose_x, .50), (.43, .60), (.57, .60), (.50, .58), (.50, .62),
                (.30, .25), (.70, .75)]

    def test_frontal_and_turned(self):
        frontal, turned = face_geometry(self.rows(.50), 100, 100), face_geometry(self.rows(.65), 100, 100)
        self.assertAlmostEqual(frontal['yaw'], 0)
        self.assertAlmostEqual(turned['yaw'], .75)
        self.assertEqual(frontal['size'], 50)
        self.assertTrue(frontal['mouth_inside'])

    def test_mouth_outside(self):
        rows = self.rows(.5)
        rows[4] = (1.02, .6)
        self.assertFalse(face_geometry(rows, 100, 100)['mouth_inside'])

    def test_face_points_contract(self):
        contract = CONTRACTS['mediapipe.face.video']
        self.assertEqual(contract.check_shape([2, 9, 2], None), 2 * 9 * 2 * 4)
        contract.check_public(dict(faces=[{}, {}]), [2, 9, 2])
        with self.assertRaises(ValueError):
            contract.check_public(dict(faces=[{}]), [2, 9, 2])


class WindowTest(unittest.TestCase):
    def test_pack_round_trip_and_weights(self):
        faces = np.arange(3 * 112 * 112, dtype=np.uint8).reshape(3, 112, 112)
        pcm = np.arange(3 * 640 + 240, dtype=np.int16)
        back_faces, back_pcm = unpack_window(pack_window(faces, pcm), 3)
        self.assertTrue((back_faces == faces).all() and (back_pcm == pcm).all())
        with self.assertRaises(ValueError):
            unpack_window(pack_window(faces, pcm), 4)
        check_weights('d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317')
        with self.assertRaises(PermissionError):
            check_weights('efc375833887eefa9d209dc92810e18519b04c3c73ea35a549f2a7f40b7d94d5')
        with self.assertRaises(ValueError):
            check_weights('0' * 64)


class _Faces:
    model_id = 'faces:test'

    def __init__(self, rows):
        self.rows = rows

    async def points(self, image, timestamp_ms):
        return np.asarray(self.rows(timestamp_ms), np.float32).reshape(-1, 9, 2)


class _Lips:
    model_id = 'lips:test'

    def __init__(self):
        self.calls = []

    async def scores(self, faces, pcm):
        self.calls.append((len(faces), len(pcm)))
        return [1.5] * len(faces)


class _Identity:
    model_id = 'lvface:test'

    async def embed(self, face):
        return [1.] + [0.] * 15


@unittest.skipUnless(HAS_CV2, 'needs OpenCV')
class ModuleTest(unittest.TestCase):
    def run_module(self, rows, *, enrolled=True, seconds=3.):
        lips = _Lips()
        module = ActiveSpeaker(_Faces(rows), lips, _Identity(), hop_ms=1000, lookahead_ms=400, context_ms=400,
                               min_sharpness=0.)
        if enrolled:
            module.set_reference(dict(face_model='lvface:test', face=[[1.] + [0.] * 15]))
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        records = []

        events = sorted([(40 + round(index * 1000 / 30), 'video') for index in range(int(seconds * 30))] +
                        [(20 * (index + 1), 'audio') for index in range(int(seconds * 50))])

        async def go():
            for ts, kind in events:
                item = InputFrame(np.zeros(320, np.float32), 'audio', ts, 'u', 16000) if kind == 'audio' else \
                    InputFrame(image, 'video', ts)
                records.extend(await module.process(item) or [])
            records.extend(await module.finish() or [])
        asyncio.run(go())
        return records, lips

    def test_records_hold_numbers_only(self):
        frontal = [(.40, .40), (.60, .40), (.50, .50), (.43, .60), (.57, .60), (.50, .58), (.50, .62), (.30, .25), (.70, .75)]
        records, lips = self.run_module(lambda ts: [frontal])
        self.assertTrue(lips.calls and all(pcm == frames * 640 + 240 for frames, pcm in lips.calls))
        self.assertGreaterEqual(len(records), 2)
        faces = [face for record in records for face in record['faces']]
        self.assertTrue(faces and all(face['identity'] == 1. for face in faces))
        self.assertIn(1.5, faces[0]['speaking'])

        def keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)
        found = set(keys(records))
        self.assertFalse({key for key in found if is_raw_key(key)})
        self.assertFalse(found & {'box', 'bbox', 'points', 'points5', 'x', 'y', 'crop'})
        json.dumps(records)

    def test_a_turned_or_small_face_is_unread(self):
        turned = [(.40, .40), (.60, .40), (.65, .50), (.43, .60), (.57, .60), (.50, .58), (.50, .62), (.30, .25), (.70, .75)]
        small = [(.49, .49), (.51, .49), (.50, .50), (.49, .51), (.51, .51), (.50, .505), (.50, .51), (.48, .48), (.52, .52)]
        for rows, reason in ((turned, 'turned'), (small, 'small')):
            records, _ = self.run_module(lambda ts: [rows], enrolled=False)
            faces = [face for record in records for face in record['faces']]
            self.assertTrue(faces)
            self.assertTrue(all(value is None for face in faces for value in face['speaking']))
            self.assertTrue(all(face['identity'] is None for face in faces))
            self.assertIn(reason, faces[0]['unread'])


class RecipeTest(unittest.TestCase):
    def test_lips_is_off_by_default_and_adds_the_camera(self):
        from gamst.recipes import load, roles
        plain = load('person-term-count')
        self.assertEqual([item['id'] for item in plain['inputs']], ['microphone'])
        self.assertEqual(next(item for item in plain['modules'] if item['name'] == 'terms')['consumes'], ['speaker_check'])
        lips = load('person-term-count', include=['lips'])
        self.assertEqual([item['id'] for item in lips['inputs']], ['microphone', 'camera'])
        self.assertEqual(next(item for item in lips['modules'] if item['name'] == 'terms')['consumes'], ['speaker_lips'])
        self.assertEqual({key: roles(lips)[key] for key in ('vision.face', 'vision.face_identity', 'vision.active_speaker')},
                         {'vision.face': 'mediapipe.face', 'vision.face_identity': 'lvface.b',
                          'vision.active_speaker': 'lightasd.ava'})
        self.assertIn('person', [item['name'] for item in lips['references']])
        self.assertNotIn('person', [item['name'] for item in plain['references']])


@unittest.skipUnless(os.environ.get('GAMST_LIGHTASD_SOURCE') and os.environ.get('GAMST_LIGHTASD_CHECKPOINT'),
                     'set GAMST_LIGHTASD_SOURCE and GAMST_LIGHTASD_CHECKPOINT to run the real model')
class RealLightASDTest(unittest.TestCase):
    def test_real_weights_score_every_frame(self):
        from gamst.providers.active_speaker import LightASD
        checkpoint = Path(os.environ['GAMST_LIGHTASD_CHECKPOINT'])
        import hashlib
        model = LightASD.from_local(os.environ['GAMST_LIGHTASD_SOURCE'], checkpoint,
                                    hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        rng = np.random.default_rng(1)
        faces = rng.integers(0, 255, (50, 112, 112), dtype=np.uint8)
        pcm = (rng.standard_normal(50 * 640 + 240) * 3000).astype(np.int16)
        scores = model.scores(faces, pcm)
        self.assertEqual(scores.shape, (50,))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == '__main__':
    unittest.main()
