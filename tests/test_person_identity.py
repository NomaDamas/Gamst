"""Person identity: few-shot enforcement, the reference check, per-track binding (retroactive), per-track counting,
who in count_events / get_timeline / set_instruction, MCP enroll_person and photo enrollment. Test doubles only."""
import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gamst.person_reference import HOW, check, hud_line, refusal, require, session_of, with_voice
from gamst.providers.person_binding import TrackBindings, intervals_of, resolve, seen_intervals

try:
    import numpy as np
    import cv2  # noqa: F401
    HAS_CV = True
except ImportError:
    HAS_CV = False

CONSENT = dict(statement='P consented to face and body matching in this test', recorded_ms=1)


def reference(face=3, body=0, *, spread=0., session=None, voice=None, persons=1, consent=CONSENT, name='P'):
    faces = [[1., spread * (i % 2), 0.] for i in range(face)] if not spread else \
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]][:face]
    value = dict(kind='person', name=name, consent=consent, face_model='face', body_model='body', face=faces,
                 body=[[0., 1., .01 * i] for i in range(body)], session=session,
                 samples=[dict(source='image', persons=persons, face=True, body=bool(body)) for _ in range(max(face, body, 1))])
    if voice is not None:
        value['voice'] = voice
    return value


class ReferenceCheckTest(unittest.TestCase):
    def test_three_agreeing_faces_pass_and_say_their_basis(self):
        verdict = check(reference())
        self.assertTrue(verdict['ok'], verdict)
        self.assertEqual(verdict['basis'], ['face'])
        self.assertEqual(verdict['face']['n'], 3)

    def test_what_is_missing_is_said(self):
        cases = [(reference(consent=None), 'consent'), (reference(persons=2), 'exactly one person'),
                 (reference(face=2), 'at least 3'), (reference(spread=1.), 'disagree'),
                 (reference(face=0), 'no face, body or voice'), ({'kind': 'voice'}, 'not a person reference')]
        for value, words in cases:
            verdict = check(value)
            self.assertFalse(verdict['ok'])
            self.assertIn(words, ' '.join(verdict['problems']))

    def test_appearance_alone_holds_only_in_its_own_video(self):
        value = reference(face=0, body=3, session='file:abc')
        self.assertTrue(check(value, session='file:abc')['ok'])
        self.assertTrue(check(value, session='file:abc')['session_only'])
        other = check(value, session='file:other')
        self.assertFalse(other['ok'])
        self.assertIn('trusted only in the video it came from', other['problems'][0])
        self.assertEqual(check(reference(body=3, session='file:abc'), session='file:abc')['basis'], ['face', 'body'])
        self.assertEqual(check(reference(body=3, session='file:abc'), session=None)['basis'], ['face'])

    def test_a_voice_identifies_without_a_face_and_a_mixed_voice_is_refused(self):
        voice = dict(model='campplus', embeddings=[[1., 0.], [.9, .1]])
        self.assertEqual(check(reference(face=0, voice=voice))['basis'], ['voice'])
        mixed = check(reference(face=0, voice=dict(model='campplus', embeddings=[[1., 0.], [0., 1.]])))
        self.assertFalse(mixed['ok'])
        self.assertIn('voice samples disagree', mixed['problems'][0])
        self.assertEqual(with_voice(reference(), dict(kind='voice', model='m', embeddings=[[1., 0.]]))['voice']['model'], 'm')

    def test_require_refuses_in_one_wording(self):
        with self.assertRaises(ValueError) as missing:
            require('P', {})
        self.assertEqual(str(missing.exception), refusal('P'))
        self.assertIn(HOW, str(missing.exception))
        with self.assertRaisesRegex(ValueError, "who='P' refused before start: only 1 usable face"):
            require('P', dict(person=dict(kind='person', value=reference(face=1))))
        key, _, verdict = require('P', dict(person=dict(kind='person', value=reference())))
        self.assertEqual((key, verdict['basis']), ('person', ['face']))
        self.assertTrue(hud_line('P', ['x']).startswith('WHO P REFUSED: x'))
        self.assertTrue(hud_line('감스트').isascii())

    def test_a_few_badly_aligned_crops_are_dropped_by_the_majority_and_two_people_are_not(self):
        from gamst.person_reference import cleaned
        same = [[1., .1 * i, 0.] for i in range(4)]
        kept, dropped = cleaned(same + [[0., 0., 1.], [0., 0., -1.]], .5)
        self.assertEqual((kept[:4], len(kept), dropped), (same, 5, 1))     # stops once the rest agree (voice rule)
        two = [[1., 0., 0.]] * 2 + [[0., 1., 0.]] * 2
        self.assertEqual(cleaned(two, .5), (two, 0))      # no majority of one person: kept, and the check refuses it
        self.assertFalse(check(reference(face=0) | dict(face=two))['ok'])

    def test_session_of_names_one_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder, 'a.mp4'), Path(folder, 'b.mp4')
            a.write_bytes(b'x' * 1000)
            b.write_bytes(b'y' * 1000)
            self.assertEqual(session_of(a), session_of(a))
            self.assertNotEqual(session_of(a), session_of(b))


class BindingDataTest(unittest.TestCase):
    def test_the_first_binding_reaches_back_to_the_track_start_and_a_later_one_does_not(self):
        bindings = TrackBindings()
        bindings.seen('t1', 1000)
        bindings.seen('t1', 4000)
        first = bindings.bind('t1', 'P', 5000, 'face', face_score=.8)
        self.assertEqual((first['start_ms'], first['end_ms'], first['retroactive']), (1000, 5000, True))
        bindings.seen('t1', 8000)
        released = bindings.unbind('t1', 8000, 'face_contradicts')
        again = bindings.bind('t1', 'Q', 9000, 'face')
        self.assertEqual((again['start_ms'], again['retroactive']), (9000, False))
        intervals = intervals_of([first, released, again])
        self.assertEqual(intervals['t1'], [(1000, 8000, 'P'), (9000, None, 'Q')])
        self.assertEqual(resolve(dict(track_id='t1', end_ms=2000, who='unknown'), intervals), 'P')     # retroactive
        self.assertEqual(resolve(dict(track_id='t1', end_ms=8500), intervals), 'unknown')
        self.assertEqual(resolve(dict(track_id='t1', end_ms=9500), intervals), 'Q')
        self.assertEqual(resolve(dict(track_id='t2', end_ms=2000, who='R'), intervals), 'R')
        self.assertEqual(seen_intervals(intervals, 'P', 2000, 10000), [dict(track_id='t1', start_ms=2000, end_ms=8000)])

    def test_a_person_is_on_one_track_at_a_time_also_in_the_past(self):
        bindings = TrackBindings()
        bindings.seen('t1', 0)
        bindings.bind('t1', 'P', 500, 'face')
        bindings.seen('t2', 1000)                     # a second track of the same person starts while t1 still holds
        bindings.seen('t1', 3000)
        released = bindings.unbind('t1', 4000, 'seen_on_another_track')    # t1 was last seen at 3000
        moved = bindings.bind('t2', 'P', 4000, 'face')
        self.assertEqual((released['end_ms'], moved['start_ms'], moved['retroactive']), (3000, 3001, True))  # not both at 3000

    def test_appearance_thresholds_are_per_model_and_provisional(self):
        from gamst.providers.person_binding import APPEARANCE, appearance_defaults
        model = lambda name: argparse.Namespace(model_id=f'{name}:0123456789abcdef')
        self.assertEqual(appearance_defaults(model('personvit-b')), APPEARANCE['personvit-b'])
        self.assertEqual(appearance_defaults(model('osnet-ain'))['body_threshold'], .8)
        self.assertEqual(appearance_defaults(None), APPEARANCE['osnet-ain'])

    def test_a_lost_track_ends_its_binding_at_its_last_frame(self):
        bindings = TrackBindings()
        bindings.seen('t', 0)
        bindings.bind('t', 'P', 100, 'face')
        bindings.seen('t', 900)
        event = bindings.forget('t')
        self.assertEqual((event['event'], event['end_ms'], event['reason']), ('person.unbound', 900, 'track_lost'))


# ---------------------------------------------------------------- modules with test doubles

class Color:
    """Test double embedder: mean color of the crop (a person is a color in these frames)."""

    def __init__(self, model_id):
        self.model_id = model_id

    def embed(self, image):
        return [float(v) + 1e-6 for v in image.reshape(-1, 3).mean(0)]


def person(x, track, *, face=True, wrist_up=False):
    """A MediaPipe-33 pose at x; the left wrist above the shoulder when ``wrist_up``."""
    landmarks = [dict(x=x, y=.5, visibility=0., presence=1.) for _ in range(33)]
    for index, dx, y in ((11, -.05, .35), (12, .05, .35), (13, -.06, .45), (14, .06, .45), (23, -.04, .6), (24, .04, .6),
                         (15, -.06, .2 if wrist_up else .55), (16, .06, .55)):
        landmarks[index] = dict(x=x + dx, y=y, visibility=1., presence=1.)
    if face:
        for index, dx, dy in ((5, -.04, 0), (2, .04, 0), (0, 0, .04), (10, -.03, .08), (9, .03, .08)):
            landmarks[index] = dict(x=x + dx, y=.2 + dy, visibility=1., presence=1.)
    return dict(landmarks=landmarks, bbox=dict(x=x - .1, y=.1, width=.2, height=.8), track_id=track)


def frame_of(left, right):
    image = np.zeros((400, 400, 3), np.uint8)
    image[:, :200], image[:, 200:] = left, right
    return image


RED, BLUE = (0, 0, 255), (255, 0, 0)
RAISE = {'rule': {'steps': [{'when': {'measure': {'height': ['left_wrist', 'left_shoulder']}, 'op': 'gt', 'value': .5},
                             'hold_ms': 0}]}, 'label': 'raise'}


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class PersonBindingTest(unittest.IsolatedAsyncioTestCase):
    def module(self, **options):
        from gamst.providers.person_binding import PersonBinding
        options = dict(dict(anonymous=False, scene_cuts=False), **options)
        module = PersonBinding(Color('face'), Color('body'), face_threshold=.9, face_reject=.5, body_threshold=.95,
                               face_every_ms=0, body_every_ms=0, session='file:s', **options)
        module.set_references(dict(person=reference(face=3, name='P') | dict(face=[[0., 0., 1.]] * 3)))
        return module

    async def step(self, module, ts, poses, image):
        """A frame, then (faces are scored beside the lane) the next frame 1 ms later, where its scores are decided on."""
        from gamst.plugins import InputFrame
        from gamst.private_payload import PrivatePoses
        await module.process(InputFrame(PrivatePoses(poses), 'video', ts))
        first = await module.process(InputFrame(image, 'video', ts))
        await module.settled()
        await module.process(InputFrame(PrivatePoses(poses), 'video', ts + 1))
        second = await module.process(InputFrame(image, 'video', ts + 1))
        await module.settled()
        return dict(second, events=first['events'] + second['events'])

    async def test_a_face_binds_its_track_from_the_track_start_and_another_face_releases_it(self):
        module = self.module(confirm_every_ms=0)     # a bound face looked at on every frame here
        first = await self.step(module, 1000, [person(.25, 'a', face=False), person(.75, 'b', face=False)], frame_of(BLUE, RED))
        self.assertEqual(first['events'], [])
        self.assertEqual({item['who'] for item in first['tracks']}, {'unknown'})
        bound = await self.step(module, 2000, [person(.25, 'a'), person(.75, 'b')], frame_of(BLUE, RED))
        event = bound['events'][0]
        self.assertEqual((event['event'], event['track_id'], event['who'], event['start_ms'], event['by'], event['retroactive']),
                         ('person.bound', 'b', 'P', 1000, 'face', True))
        self.assertEqual(bound['bound'], {'P': 'b'})
        switched = await self.step(module, 3000, [person(.25, 'a'), person(.75, 'b')], frame_of(RED, BLUE))
        names = [(item['event'], item['track_id']) for item in switched['events']]
        self.assertIn(('person.unbound', 'b'), names)
        self.assertIn(('person.bound', 'a'), names)
        self.assertEqual(switched['bound'], {'P': 'a'})

    async def test_appearance_of_this_session_binds_without_a_face_and_l2_forks_hand_over(self):
        module = self.module()
        await self.step(module, 1000, [person(.75, 'b')], frame_of(BLUE, RED))       # the face confirms; body is kept
        later = await self.step(module, 9000, [person(.75, 'c', face=False)], frame_of(BLUE, RED))
        self.assertEqual([(item['event'], item.get('by')) for item in later['events'] if item['event'] == 'person.bound'],
                         [('person.bound', 'body')])
        snapshot = module.snapshot()
        twin = module.fork()
        twin.restore(snapshot)
        handed = await self.step(twin, 9100, [person(.75, 'person-1', face=False)], frame_of(BLUE, BLUE))
        bound = [item for item in handed['events'] if item['event'] == 'person.bound']
        self.assertEqual((bound[0]['by'], bound[0]['track_id']), ('handover', 'person-1@9100'))

    async def test_frames_without_a_person_are_empty_answers_and_keep_the_bindings(self):
        from gamst.plugins import InputFrame
        from gamst.private_payload import PrivatePoses
        module = self.module()
        bound = await self.step(module, 1000, [person(.75, 'b')], frame_of(BLUE, RED))
        self.assertEqual(bound['bound'], {'P': 'b'})
        black = np.zeros((400, 400, 3), np.uint8)
        zero_box = dict(person(.5, 'z'), bbox=dict(x=.5, y=.5, width=0., height=0.))
        for ts, poses in ((1100, []), (1133, [dict(person(.5, None), bbox=None)]), (1166, [zero_box]), (1200, [])):
            for first, second in ((PrivatePoses(poses), black), (black, PrivatePoses(poses))):     # either order
                outputs = [await module.process(InputFrame(item, 'video', ts)) for item in (first, second)]
                await module.settled()
                answer = next(item for item in outputs if item is not None)
                self.assertEqual(answer['bound'], {'P': 'b'})       # nobody in view: the binding stands
                self.assertEqual(answer['events'], [])
                ts += 1
        later = await self.step(module, 1500, [person(.75, 'b')], frame_of(BLUE, RED))
        self.assertEqual(later['bound'], {'P': 'b'})
        await module.process(InputFrame(PrivatePoses([]), 'video', 9000))   # nobody for longer than lost_ms
        gone = await module.process(InputFrame(black, 'video', 9000))
        self.assertEqual([(item['event'], item['reason']) for item in gone['events']], [('person.unbound', 'track_lost')])

    async def test_camera_seconds_collect_the_one_person_in_view(self):
        module = self.module()

        async def feed():
            await asyncio.sleep(0)
            for ts in range(0, 1400, 200):
                poses = [person(.75, 'b')] if ts != 400 else [person(.25, 'a'), person(.75, 'b')]
                await self.step(module, 10000 + ts, poses, frame_of(BLUE, RED))
        collected, _ = await asyncio.gather(module.collect(1.), feed())
        self.assertEqual(collected['skipped'], 2)      # the crowded frame and the one after it (step: two frames)
        self.assertEqual(len(collected['face']), 5)
        self.assertTrue(all(sample['persons'] == 1 for sample in collected['samples']))

    async def test_in_a_lane_of_its_own_the_camera_frame_comes_first_and_is_kept_only_when_faces_are_due(self):
        from gamst.plugins import InputFrame
        from gamst.private_payload import PrivatePoses
        module = self.module(confirm_every_ms=1000)
        image = frame_of(BLUE, RED)
        found = []
        for ts in range(0, 1400, 100):
            await module.process(InputFrame(image, 'video', ts))                       # camera first
            found.append(await module.process(InputFrame(PrivatePoses([person(.75, 'b')]), 'video', ts)))
            await module.settled()
        self.assertTrue(all(value is not None for value in found))                    # every frame decided
        events = [event for value in found for event in value['events']]
        self.assertEqual([(event['event'], event['who'], event['start_ms'], event['end_ms']) for event in events],
                         [('person.bound', 'P', 0, 100)])         # scored beside the lane, decided on the next frame
        self.assertEqual(module.counts['face_embeddings'], 2)        # bound at 0, confirmed again at 1000, not in between
        self.assertEqual(module.counts['unpaired_images'], 0)
        self.assertLessEqual(len(module._images), 1)
        self.assertEqual(module.status()['decision_delay_ms']['max'], 100)

    async def test_nobody_enrolled_binds_nobody_and_a_person_enrolled_later_reaches_back_over_live_tracks(self):
        from gamst.providers.person_binding import PersonBinding
        module = PersonBinding(Color('face'), Color('body'), face_threshold=.9, face_reject=.5, session='file:s',
                               anonymous=False, scene_cuts=False)        # nobody: no persons of its own either
        module.bind(argparse.Namespace(references={}))
        first = await self.step(module, 1000, [person(.75, 'b')], frame_of(BLUE, RED))
        self.assertEqual((first['tracks'][0]['who'], first['bound'], module.counts['face_embeddings']), ('unknown', {}, 0))
        await self.step(module, 2000, [person(.75, 'b')], frame_of(BLUE, RED))
        module.set_references(dict(person_1=reference(name='P') | dict(face=[[0., 0., 1.]] * 3)))     # enroll_person
        later = await self.step(module, 3000, [person(.75, 'b')], frame_of(BLUE, RED))
        event = later['events'][0]
        self.assertEqual((event['who'], event['start_ms'], event['retroactive']), ('P', 1000, True))

    async def test_a_new_track_where_the_person_just_was_continues_the_person(self):
        module = self.module()
        await self.step(module, 1000, [person(.75, 'b')], frame_of(BLUE, RED))
        swapped = await self.step(module, 1033, [person(.74, 'c', face=False)], frame_of(BLUE, RED))
        moved = [(item['event'], item['track_id'], item.get('by') or item.get('reason')) for item in swapped['events']]
        self.assertEqual(moved, [('person.unbound', 'b', 'track_replaced'), ('person.bound', 'c', 'continuation')])
        other = self.module()
        await self.step(other, 1000, [person(.75, 'b')], frame_of(BLUE, RED))
        elsewhere = await self.step(other, 1033, [person(.25, 'c', face=False)], frame_of(BLUE, RED))
        self.assertEqual(elsewhere['events'], [])           # no overlap: the face (or appearance) has to say it
        late = self.module()
        await self.step(late, 1000, [person(.75, 'b')], frame_of(BLUE, RED))
        gone = await self.step(late, 3000, [person(.75, 'c', face=False)], frame_of(RED, RED))
        self.assertNotIn('continuation', [item.get('by') for item in gone['events']])   # too long after

    async def test_at_the_end_of_input_every_open_binding_ends_at_its_last_frame(self):
        module = self.module()
        await self.step(module, 1000, [person(.75, 'b')], frame_of(BLUE, RED))
        await self.step(module, 2000, [person(.75, 'b')], frame_of(BLUE, RED))
        ended = module.finish()['events']
        self.assertEqual([(item['event'], item['reason'], item['end_ms']) for item in ended],
                         [('person.unbound', 'end_of_input', 2001)])
        self.assertIsNone(module.finish())

    async def test_closing_waits_for_the_scorer_and_tolerates_a_worker_that_already_ended(self):
        class Slow(Color):
            async def embed(self, image):
                await asyncio.sleep(.05)
                return super().embed(image)

            def close(self):
                raise RuntimeError('provider worker is closed; explicitly reopen its stream')

        from gamst.plugins import InputFrame
        from gamst.private_payload import PrivatePoses
        from gamst.providers.person_binding import PersonBinding
        module = PersonBinding(Slow('face'), Slow('body'), face_threshold=.9, face_reject=.5, session='file:s')
        module.set_references(dict(person=reference(name='P') | dict(face=[[0., 0., 1.]] * 3)))
        await module.process(InputFrame(PrivatePoses([person(.75, 'b')]), 'video', 0))
        await module.process(InputFrame(frame_of(BLUE, RED), 'video', 0))
        await asyncio.sleep(.01)                  # a job is in flight
        await module.close()
        self.assertTrue(module.scorer.task.done())
        self.assertEqual(module.counts['face_embeddings'], 1)

    async def test_a_reference_below_the_check_is_refused_at_bind(self):
        from gamst.providers.person_binding import PersonBinding
        module = PersonBinding(Color('face'), Color('body'))
        pipeline = argparse.Namespace(references=dict(person=dict(kind='person', value=reference(face=1))))
        with self.assertRaisesRegex(ValueError, "who='P' refused before start"):
            module.bind(pipeline)


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class TrackExemplarTest(unittest.TestCase):
    def counter(self, value=RAISE):
        from gamst.providers.track_exemplar import TrackExemplar
        module = TrackExemplar(reference='exemplar', min_hold_ms=0)
        module.bind(argparse.Namespace(references=dict(exemplar=dict(kind='exemplar', value=value, label='raise'))))
        return module

    def push(self, module, ts, poses):
        from gamst.plugins import InputFrame
        from gamst.private_payload import PrivatePoses
        return module.process(InputFrame(PrivatePoses(poses), 'video', ts))

    def bind(self, module, ts, events):
        from gamst.plugins import InputFrame
        return module.process(InputFrame(dict(tracks=[], bound={}, events=events), 'video', ts))

    def test_every_track_counts_on_its_own_and_who_is_bound_afterwards(self):
        module = self.counter()
        events = []
        for ts, a_up, b_up in ((0, False, False), (100, True, False), (200, False, False), (300, False, True),
                               (400, True, True), (500, False, False)):
            value = self.push(module, ts, [person(.25, 'a', wrist_up=a_up), person(.75, 'b', wrist_up=b_up)])
            events += [item for item in value['events'] if item['event'] == 'exemplar.matched']
        self.assertEqual([(item['track_id'], item['who'], item['index']) for item in events],
                         [('a', 'unknown', 1), ('b', 'unknown', 2), ('a', 'unknown', 3)])
        self.assertEqual(value['count'], 3)
        self.assertEqual(value['by_who'], {'unknown': 3})
        bound = dict(event='person.bound', track_id='a', who='P', start_ms=0, end_ms=600, by='face', retroactive=True)
        self.bind(module, 600, [bound])
        later = self.push(module, 700, [person(.25, 'a', wrist_up=True), person(.75, 'b')])
        matched = [item for item in later['events'] if item['event'] == 'exemplar.matched']
        self.assertEqual([(item['track_id'], item['who']) for item in matched], [('a', 'P')])
        self.assertEqual(later['by_who'], {'P': 3, 'unknown': 1})       # a's earlier two are P's too (retroactive)

    def test_an_instruction_for_one_person_counts_that_person(self):
        module = self.counter(dict(RAISE, who='P'))
        self.bind(module, 0, [dict(event='person.bound', track_id='b', who='P', start_ms=0, end_ms=0, by='face')])
        value = None
        for ts, up in ((10, False), (100, True), (200, False), (300, True)):
            value = self.push(module, ts, [person(.25, 'a', wrist_up=up), person(.75, 'b', wrist_up=up and ts == 300)])
        self.assertEqual((value['who'], value['count'], value['count_all']), ('P', 1, 3))

    def test_frames_without_a_person_count_nothing_and_break_nothing(self):
        module, events = self.counter(), []
        frames = [[]] * 9 + [[person(.5, 'a', wrist_up=True)]] * 2 + [[]] * 4      # nobody, a raise, nobody again
        for index, poses in enumerate(frames):
            events += (self.push(module, 1000 + 33 * index, poses) or {}).get('events', [])
        self.assertEqual([event['track_id'] for event in events if event['event'] == 'exemplar.matched'], ['a'])

    def test_a_pose_without_a_track_continues_the_track_it_overlaps(self):
        module = self.counter()
        untracked = lambda x, up=False: dict(person(x, None, wrist_up=up), track_id=None)
        self.push(module, 0, [person(.25, 'a'), untracked(.75)])
        self.push(module, 100, [untracked(.25, True), untracked(.75)])
        self.assertEqual(sorted(module._tracks), ['a', 'untracked'])
        self.assertEqual(module.by_who(), {'unknown': 1})

    def test_a_fork_names_its_tracks_by_its_first_frame(self):
        module = self.counter()
        twin = module.fork()
        value = twin.process(__import__('gamst.plugins', fromlist=['InputFrame']).InputFrame(
            __import__('gamst.private_payload', fromlist=['PrivatePoses']).PrivatePoses([person(.25, 'a')]), 'video', 500))
        self.assertEqual(value['tracks'], 1)
        self.assertEqual(list(twin._tracks), ['a@500'])


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class OwnLaneTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_spilling_lane_reading_derived_joints_gets_them_as_they_are_and_pairs_them_with_the_frame(self):
        from gamst import Gamst
        from gamst.pipeline import Derived, Pipeline
        from gamst.private_payload import PrivatePoses
        from gamst.providers.person_binding import PersonBinding
        from tests.test_speaker_verify import now

        class Pose:
            def process(self, frame):
                return [Derived('joints', PrivatePoses([person(.75, 'b')]), 'video')]

        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'lanes')
            pipeline.add_input('camera')
            pipeline.add_module('pose', Pose(), consumes='camera', emits=('joints',), backlog='spill')
            binding = PersonBinding(Color('face'), Color('body'), face_threshold=.9, face_reject=.5, session='file:s')
            pipeline.add_reference('person', 'person', reference(name='P') | dict(face=[[0., 0., 1.]] * 3), consent=CONSENT)
            pipeline.add_module('person_binding', binding, consumes=['camera', 'joints'], spec=binding.spec, backlog='spill')
            pipeline.video_memory(memory_bytes=1 << 24, disk_bytes=0, disk_floor_bytes=0)
            start = now() - 5000
            for index in range(6):
                await pipeline.push('camera', frame_of(BLUE, RED), timestamp_ms=start + 100 * index, modality='video')
                await pipeline.drain()
                await binding.settled()
            rows = human.store.query('local-user', kind='observation.signal', registered_only=True,
                                     capability_id='vision.person_binding')
            await pipeline.close()
        self.assertEqual(binding.counts['frames'], 6, binding.counts)
        self.assertEqual(binding.counts['face_embeddings'], 1, binding.counts)
        events = [event for row in rows for event in row['observation']['value']['events']]
        self.assertEqual([(event['event'], event['who']) for event in events], [('person.bound', 'P')])


# ---------------------------------------------------------------- command line, store and MCP

class CommandLineTest(unittest.TestCase):
    def test_a_run_with_a_person_below_the_check_or_an_unknown_who_is_refused_before_start(self):
        from gamst.recipes import _reference_value, load, with_who
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'p.json')
            path.write_text(json.dumps(reference(face=1)))
            args = argparse.Namespace(person=str(path), video=None)
            with self.assertRaises(SystemExit) as low:
                _reference_value(dict(kind='person', name='person'), args)
            self.assertIn("who='P' refused before start: only 1 usable face crop", str(low.exception))
            path.write_text(json.dumps(reference()))
            value, consent = _reference_value(dict(kind='person', name='person'), args)
        recipe = load('person-action-count')
        given = [(dict(kind='person', name='person'), value, consent), (dict(kind='exemplar', name='exemplar'), dict(RAISE), None)]
        with self.assertRaisesRegex(SystemExit, "who='Q' refused before start: no person reference"):
            with_who(recipe, argparse.Namespace(who='Q', video=None), given, recipe['modules'])
        chosen = with_who(recipe, argparse.Namespace(who='P', video=None), given, recipe['modules'])
        self.assertEqual(chosen[1][1]['who'], 'P')
        without = load('person-action-count', (), ('person_binding',))
        with self.assertRaisesRegex(SystemExit, 'no person-binding module'):
            with_who(without, argparse.Namespace(who='P', video=None), given, without['modules'])

    def test_without_a_person_the_binding_still_runs_so_a_person_can_be_enrolled_later(self):
        from gamst.recipes import not_given, read_recipe, reference_values, load
        raw = read_recipe('person-action-count')[1]
        self.assertEqual(not_given(raw, argparse.Namespace(person=None)), [])
        recipe = load('person-action-count')
        args = argparse.Namespace(person=None, exemplar=None, terms=None, voice=None)
        with self.assertRaisesRegex(SystemExit, 'exemplar') as error:     # the movement is still needed; the person is not
            reference_values(recipe, args)
        self.assertNotIn('person', str(error.exception).split('exemplar')[0])

    def test_the_recipe_counts_every_track_and_keeps_the_single_target_choice(self):
        from gamst.recipes import load
        from gamst.recipes.parts import per_track
        default, single = load('person-action-count'), load('person-action-count', ('target_person',), ())
        context = lambda recipe: dict(modules=recipe['modules'])
        exemplar = lambda recipe: next(item for item in recipe['modules'] if item['name'] == 'exemplar')
        self.assertTrue(per_track(exemplar(default), context(default)))
        self.assertFalse(per_track(exemplar(single), context(single)))
        self.assertNotIn('person_binding', [item['name'] for item in single['modules']])


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class StoreWhoTest(unittest.IsolatedAsyncioTestCase):
    async def test_count_events_and_get_timeline_resolve_who_retroactively(self):
        from gamst import Gamst
        from gamst.capabilities import CapabilitySpec, OutputField
        from gamst.providers.person_binding import CAPABILITY
        from tests.test_speaker_verify import now
        async with Gamst(consent=True) as human:
            start = now() - 5000
            from gamst.capabilities import SensorMetadata
            events = CapabilitySpec('temporal.exemplar_events', 'temporal.exemplar_events', 'video', 'test', '1',
                                    (OutputField('count', 'integer'), OutputField('events', 'array')), 'native')
            binding = CapabilitySpec(CAPABILITY, CAPABILITY, 'video', 'test', '1', (OutputField('events', 'array'),), 'frame')
            human.register_capability(events)
            human.register_capability(binding)
            matched = lambda track, ts: dict(event='exemplar.matched', track_id=track, who='unknown', start_ms=ts - 50,
                                             end_ms=ts, label='raise')
            await human.push_observation('temporal.exemplar_events', dict(count=2, events=[
                matched('a', start + 100), matched('b', start + 200)]), sensor=SensorMetadata('t', 'camera', 1),
                timestamp_ms=start + 200)
            await human.push_observation(CAPABILITY, dict(events=[dict(event='person.bound', track_id='a', who='P',
                                                                       start_ms=start, end_ms=start + 1000, by='face')]),
                                         sensor=SensorMetadata('t', 'camera', 2), timestamp_ms=start + 1000)
            mine = await human.count_events('temporal.exemplar_events', who='P')
            self.assertEqual(mine['count'], 1)
            self.assertEqual(mine['evidence'][0]['event']['track_id'], 'a')
            self.assertEqual(mine['evidence'][0]['event']['who_when_counted'], 'unknown')
            self.assertEqual((await human.count_events('temporal.exemplar_events', who='unknown'))['count'], 1)
            self.assertEqual((await human.count_events('temporal.exemplar_events'))['count'], 2)
            timeline = await human.get_timeline(start_ms=start, end_ms=start + 2000, step_ms=500,
                                                capabilities=['temporal.exemplar_events'], who='P')
            shown = [entry['value'] for entry in timeline['evidence'].values() if entry['kind'] == 'events']
            self.assertEqual([(item['track_id'], item['who']) for item in shown], [('a', 'P')])
            self.assertEqual(timeline['who']['seen'][0]['track_id'], 'a')

    async def test_count_events_counts_one_action_of_a_person_on_two_tracks_once(self):
        from gamst import Gamst
        from gamst.capabilities import CapabilitySpec, OutputField, SensorMetadata
        from gamst.providers.person_binding import CAPABILITY
        from tests.test_speaker_verify import now
        async with Gamst(consent=True) as human:
            start = now() - 5000
            human.register_capability(CapabilitySpec('temporal.exemplar_events', 'temporal.exemplar_events', 'video',
                                                     'test', '1', (OutputField('count', 'integer'),
                                                                   OutputField('events', 'array')), 'native'))
            human.register_capability(CapabilitySpec(CAPABILITY, CAPABILITY, 'video', 'test', '1',
                                                     (OutputField('events', 'array'),), 'frame'))
            matched = lambda track, ts: dict(event='exemplar.matched', track_id=track, who='unknown', start_ms=ts - 50,
                                             end_ms=ts, label='raise')
            await human.push_observation('temporal.exemplar_events', dict(count=3, events=[
                matched('a', start + 100), matched('b', start + 160), matched('b', start + 900)]),
                sensor=SensorMetadata('t', 'camera', 1), timestamp_ms=start + 900)
            await human.push_observation(CAPABILITY, dict(events=[
                dict(event='person.bound', track_id='a', who='P', start_ms=start, end_ms=start + 120, by='face'),
                dict(event='person.unbound', track_id='a', who='P', start_ms=start, end_ms=start + 120, reason='x'),
                dict(event='person.bound', track_id='b', who='P', start_ms=start + 121, end_ms=start + 1000, by='face')]),
                sensor=SensorMetadata('t', 'camera', 2), timestamp_ms=start + 1000)
            mine = await human.count_events('temporal.exemplar_events', who='P')
            self.assertEqual(mine['count'], 2)          # a at 100 and b at 160 are one action of P; b at 900 another


class MCPToolsTest(unittest.TestCase):
    def test_enroll_person_and_who_are_tools(self):
        from gamst.mcp import tool_definitions
        tools = {tool['name']: tool for tool in tool_definitions()}
        enroll = tools['enroll_person']['inputSchema']
        self.assertEqual(set(enroll['required']), {'name', 'consent'})
        self.assertTrue({'images', 'video', 'camera_seconds'} <= set(enroll['properties']))
        for name in ('set_instruction', 'count_events', 'get_timeline'):
            self.assertIn('who', tools[name]['inputSchema']['properties'])


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class InstructionWhoTest(unittest.IsolatedAsyncioTestCase):
    def instructions(self, references, binding=True, folders=()):
        from gamst.recipes.instructions import Instructions
        from gamst.providers.person_binding import PersonBinding
        module = PersonBinding(Color('face'), Color('body'), session='file:s')
        good = {key: item['value'] for key, item in references.items() if check(item['value'])['ok']}
        if good:
            module.set_references(good)
        modules = dict(person_binding=dict(module=module)) if binding else {}
        pipeline = argparse.Namespace(references=dict(references), modules=modules, notes={},
                                      _reading=lambda name: {}, recipe_context=dict(modules=[]))
        self_ = Instructions.__new__(Instructions)
        self_.pipeline, self_.folders, self_._busy, self_.recipe = pipeline, [Path(item) for item in folders], None, {'name': 'r'}
        return self_, module

    async def test_who_is_refused_without_a_checked_reference_and_the_hud_says_so(self):
        instructions, _ = self.instructions(dict(person=dict(kind='person', value=reference(face=1))))
        with self.assertRaisesRegex(ValueError, "who='P' refused before start: only 1 usable face"):
            instructions._who('P', 'exemplar')
        self.assertTrue(instructions.pipeline.notes['who_refused'].startswith('WHO P REFUSED'))
        missing, _ = self.instructions({})
        with self.assertRaisesRegex(ValueError, "who='P' refused before start: no person reference"):
            missing._who('P', 'exemplar')
        unbound, _ = self.instructions(dict(person=dict(kind='person', value=reference())), binding=False)
        with self.assertRaisesRegex(ValueError, 'no person_binding module'):
            unbound._who('P', 'exemplar')
        words, _ = self.instructions(dict(person=dict(kind='person', value=reference())))
        with self.assertRaisesRegex(ValueError, 'enrolled voice'):
            words._who('P', 'terms')
        good, _ = self.instructions(dict(person=dict(kind='person', value=reference())))
        good._who('P', 'exemplar')
        self.assertNotIn('who_refused', good.pipeline.notes)

    async def test_enroll_person_needs_consent_and_one_source_and_adds_the_person_from_camera_seconds(self):
        with tempfile.TemporaryDirectory() as folder:
            instructions, module = self.instructions({}, folders=[folder])
            instructions.pipeline.add_reference = lambda name, kind, value, **about: instructions.pipeline.references.update(
                {name: dict(kind=kind, value=value, **about)})
            with self.assertRaisesRegex(ValueError, 'no consent record'):
                await instructions.enroll_person(name='P', consent=' ', camera_seconds=2)
            with self.assertRaisesRegex(ValueError, 'exactly one of'):
                await instructions.enroll_person(name='P', consent='ok', camera_seconds=2, images=['a.jpg'])
            few = dict(face=[[0., 0., 1.]], body=[], samples=[dict(persons=1)], skipped=0)
            with mock.patch.object(module, 'collect', mock.AsyncMock(return_value=few)):
                with self.assertRaisesRegex(ValueError, "who='P' refused before start: only 1 usable face"):
                    await instructions.enroll_person(name='P', consent='ok', camera_seconds=2)
            enough = dict(face=[[0., 0., 1.]] * 3, body=[[0., 1., 0.]] * 3, samples=[dict(persons=1)] * 3, skipped=2)
            with mock.patch.object(module, 'collect', mock.AsyncMock(return_value=enough)):
                result = await instructions.enroll_person(name='P', consent='ok', camera_seconds=2)
            self.assertEqual(result['basis'], ['face', 'body'])       # appearance of this run's own session
            saved = Path(folder, 'P.person.json')
            self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
            self.assertNotIn('image', json.loads(saved.read_text()))
            self.assertEqual([person.name for person in module.persons], ['P'])
            instructions._who('P', 'exemplar')


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class PhotoEnrollmentTest(unittest.IsolatedAsyncioTestCase):
    async def enroll(self, folder, people):
        """``gamst recipe enroll --kind person --image ...`` with a test pose module and colour embedders."""
        from gamst.pipeline import Derived, Pipeline
        from gamst.private_payload import PrivatePoses
        from gamst.recipes import enrollment, load
        paths = []
        for index, count in enumerate(people):
            path = Path(folder, f'{index}.png')
            cv2.imwrite(str(path), frame_of(BLUE, RED))
            paths.append(str(path))

        class Pose:
            def __init__(self):
                self.index = 0

            def process(self, frame):
                count = people[self.index]
                self.index += 1
                return [Derived('pose_joints', PrivatePoses([person(.75, 'b'), person(.25, 'a')][:count]), 'video')]

        async def build(recipe, args, human, *, modules, lanes, sources):
            pipeline = Pipeline(human, 'enroll')
            pipeline.add_input('camera', sources['camera'], modality='video')
            pipeline.add_module('pose', Pose(), consumes=['camera'], emits=('pose_joints',))
            return pipeline

        async def embedder(item, context):
            return Color('face')

        async def body(item, context):
            return Color('body')

        args = argparse.Namespace(consent='P consented', image=paths, out=str(Path(folder, 'P.json')), name='P',
                                  voice_file=None, camera=None, start=0., seconds=1.5, video=None)
        with mock.patch('gamst.recipes.perception_parts.face_embedder', embedder), \
                mock.patch('gamst.recipes.perception_parts.body_embedder', body):
            return await enrollment.enroll_person(load('person-action-count'), args, build_pipeline=build, context={})

    async def test_photos_of_one_person_are_enrolled_and_a_crowded_photo_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = await self.enroll(folder, [1, 1, 1])
            self.assertEqual((len(saved['face']), len(saved['body']), saved['quality']['basis']), (3, 3, ['face']))
            self.assertEqual(saved['enrolled_from']['kind'], 'images')
            self.assertIsNone(saved['session'])       # photos: their clothes are not trusted in any video
            self.assertEqual(Path(folder, 'P.json').stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(SystemExit, 'every photo must show exactly one person: 2.png: 2 people'):
                await self.enroll(folder, [1, 1, 2])
            with self.assertRaisesRegex(SystemExit, 'only 2 usable face'):
                await self.enroll(folder, [1, 1])


# ---------------------------------------------------------------- one person, one track at any time

def at_once(events, who, times):
    """The instants (of ``times``) at which two or more tracks resolve to ``who``."""
    intervals = intervals_of([item for item in events if item])
    return [ts for ts in times
            if sum(1 for track in intervals if resolve(dict(track_id=track, end_ms=ts), intervals) == who) > 1]


def stripes(*parts):
    """An image of vertical stripes: (from column, to column, BGR) on black."""
    image = np.zeros((400, 400, 3), np.uint8)
    for low, high, color in parts:
        image[:, low:high] = color
    return image


class OneTrackDataTest(unittest.TestCase):
    def test_a_handover_in_one_frame_leaves_no_instant_with_two_tracks(self):
        bindings = TrackBindings()
        events = []
        for ts in (0, 33, 66):
            bindings.seen('a', ts)
        events.append(bindings.bind('a', 'P', 66, 'face'))
        for ts in (100, 133):
            bindings.seen('a', ts)
            bindings.seen('b', ts)
        events += bindings.bind('b', 'P', 133, 'face')      # takes P from a in a frame where both are seen
        unbound = next(item for item in events if item['event'] == 'person.unbound')
        bound = events[-1]
        self.assertEqual((unbound['track_id'], unbound['end_ms'], unbound['reason']), ('a', 100, 'another_track'))
        self.assertEqual((bound['start_ms'], bound['retroactive']), (101, True))
        self.assertEqual(at_once(events, 'P', range(0, 140)), [])
        self.assertIsNone(bindings.holder('Q'))
        self.assertEqual(bindings.holder('P'), 'b')

    def test_a_late_retroactive_binding_never_reaches_over_an_earlier_one(self):
        bindings, events = TrackBindings(), []
        bindings.seen('b', 0)                    # b is seen from the start, its face only much later
        bindings.seen('a', 500)
        events.append(bindings.bind('a', 'P', 500, 'face'))
        for ts in (1000, 2000, 3000):
            bindings.seen('a', ts)
            bindings.seen('b', ts)
        events.append(bindings.unbind('a', 3000, 'face_contradicts', before=True))      # a ends at 2000
        bindings.seen('b', 4000)
        events.append(bindings.bind('b', 'P', 4000, 'face'))      # retroactive, but only after a's binding
        self.assertEqual(events[-1]['start_ms'], 2001)
        self.assertEqual(at_once(events, 'P', range(0, 4001, 10)), [])
        bindings.seen('c', 100)                  # an even older track, bound later still
        bindings.seen('c', 5000)
        bindings.seen('b', 5000)
        events += bindings.bind('c', 'P', 5000, 'face')
        self.assertGreater(events[-1]['start_ms'], 4000)
        self.assertEqual(at_once(events, 'P', range(0, 5001, 10)), [])

    def test_an_action_on_two_tracks_of_one_person_counts_once(self):
        from gamst.providers.person_binding import PersonOnce
        once = PersonOnce()
        self.assertTrue(once.fresh('P', 'a', 1000))
        self.assertFalse(once.fresh('P', 'b', 1060))         # the same action seen on a second track
        self.assertTrue(once.fresh('P', 'a', 1060))          # one track counts what it counts
        self.assertTrue(once.fresh('P', 'b', 1200))
        self.assertTrue(once.fresh('unknown', 'c', 1000) and once.fresh('unknown', 'd', 1000))   # unknown is anybody
        self.assertEqual(once.dropped, 1)

    def test_duplicate_poses_need_both_box_and_joints(self):
        from gamst.providers.person_match import duplicates, same_body
        one, again = person(.5, 'a'), person(.5, 'b')
        self.assertTrue(same_body(one, again))
        close = person(.55, 'c')                 # two people side by side: boxes overlap (IoU .6), joints .05 apart
        self.assertFalse(same_body(one, close))
        shifted = dict(person(.6, 'd'), bbox=dict(one['bbox']))     # one box, joints of another place: two people
        self.assertFalse(same_body(one, shifted))
        self.assertEqual(duplicates(dict(a=one, b=again), dict(a=0, b=100)), dict(b='a'))     # the first seen stays
        self.assertEqual(duplicates(dict(a=one, b=again), dict(a=100, b=0)), dict(a='b'))
        self.assertEqual(duplicates(dict(a=one, d=shifted), dict(a=0, d=0)), {})


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class OneTrackModuleTest(PersonBindingTest):
    def binding(self, **options):
        from gamst.providers.person_binding import PersonBinding
        module = PersonBinding(Color('face'), Color('body'), face_every_ms=0, body_every_ms=0, confirm_every_ms=0,
                               session='file:s', **dict(dict(face_threshold=.9, face_reject=.5, body_threshold=.95,
                                                             anonymous=False, scene_cuts=False), **options))
        module.set_references(dict(person=reference(name='P') | dict(face=[[0., 0., 1.]] * 3)))
        return module

    async def test_two_tracks_that_both_show_the_face_bind_one_and_the_better_face_takes_over(self):
        module, events, times = self.binding(), [], []
        left, right = (80, 160), (240, 320)
        for ts, a, b in ((1000, RED, RED), (2000, (0, 80, 255), RED), (3000, (0, 150, 255), RED)):
            value = await self.step(module, ts, [person(.3, 'a'), person(.7, 'b')], stripes((*left, a), (*right, b)))
            events += value['events']
            times += [ts, ts + 1]
            if ts == 1000:      # both faces score 1: one track holds P (the first seen, then by id)
                self.assertEqual([(item['event'], item['track_id']) for item in value['events']], [('person.bound', 'a')])
            if ts == 2000:      # a's face .95, b's 1: within the margin, the holder keeps P (no flip)
                self.assertEqual(value['events'], [])
            if ts == 3000:      # a's face .86, b's 1: b's face is better by more than the margin
                self.assertEqual([(item['event'], item['track_id'], item.get('reason') or item.get('by'))
                                  for item in value['events']],
                                 [('person.unbound', 'a', 'seen_on_another_track'), ('person.bound', 'b', 'face')])
            self.assertEqual(sum(1 for item in value['tracks'] if item['who'] == 'P'), 1)
        self.assertEqual(at_once(events, 'P', times), [])
        self.assertGreaterEqual(module.counts['contested'], 3)      # frames where both claimed P

    async def test_the_track_that_loses_a_person_goes_to_its_own_best_other_person(self):
        module = self.binding(face_threshold=.5, face_reject=.1)
        module.set_references(dict(p=reference(name='P') | dict(face=[[0., 0., 1.]] * 3),
                                   q=reference(name='Q') | dict(face=[[0., 1., 0.]] * 3)))
        # a: pure red (P 1.0); b: P .79, Q .62: b claims P first, loses it to a, then takes Q
        value = await self.step(module, 1000, [person(.3, 'a'), person(.7, 'b')],
                                stripes((80, 160, RED), (240, 320, (0, 200, 255))))
        self.assertEqual(value['bound'], {'P': 'a', 'Q': 'b'})

    async def test_a_duplicate_box_is_neither_bound_nor_counted_and_hands_its_person_to_the_kept_track(self):
        module = self.binding()
        await self.step(module, 1000, [person(.3, 'a', face=False)], stripes((0, 400, BLUE)))       # a is seen first
        bound = await self.step(module, 1100, [person(.3, 'a', face=False), person(.75, 'b')],
                                stripes((0, 200, BLUE), (200, 400, RED)))
        self.assertEqual(bound['bound'], {'P': 'b'})
        same = await self.step(module, 1200, [person(.75, 'a', face=False), person(.75, 'b')], stripes((200, 400, RED)))
        moved = [(item['event'], item['track_id'], item.get('reason') or item.get('by')) for item in same['events']]
        self.assertEqual(moved, [('person.unbound', 'b', 'duplicate'), ('person.bound', 'a', 'duplicate')])
        self.assertEqual(same['bound'], {'P': 'a'})
        self.assertIn(dict(track_id='b', who='unknown', person_id=None, duplicate_of='a'), same['tracks'])
        self.assertGreaterEqual(module.counts['duplicates'], 2)
        events = bound['events'] + same['events']
        self.assertEqual(at_once(events, 'P', range(1000, 1202)), [])

    async def test_at_a_scene_cut_the_person_moves_to_the_new_track_without_an_instant_on_both(self):
        module = self.binding()
        events = []
        for ts in (1000, 1500):         # scene 1: P on the right, track a
            events += (await self.step(module, ts, [person(.75, 'a')], stripes((200, 400, RED))))['events']
        # the cut: the tracker keeps track a on the person now on the right (someone else), P is a new track on the left
        cut = await self.step(module, 2000, [person(.75, 'a'), person(.25, 'b')], stripes((0, 200, RED), (200, 400, BLUE)))
        events += cut['events']
        self.assertEqual(cut['bound'], {'P': 'b'})
        self.assertEqual([(item['event'], item['track_id'], item.get('reason') or item.get('by')) for item in cut['events']],
                         [('person.unbound', 'a', 'face_contradicts'), ('track.split', 'a', 'face'),
                          ('person.bound', 'b', 'face')])
        ended = next(item for item in cut['events'] if item['event'] == 'person.unbound')
        self.assertLess(ended['end_ms'], cut['events'][-1]['start_ms'])
        self.assertEqual(at_once(events, 'P', [1000, 1001, 1500, 1501, 2000, 2001]), [])


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class OneCountTest(TrackExemplarTest):
    def test_two_tracks_on_one_body_count_one_action(self):
        module = self.counter()
        events = []
        self.push(module, 0, [person(.5, 'a')])                 # a is seen first
        for ts, up in ((100, False), (200, True), (300, False), (400, True), (500, False)):
            value = self.push(module, ts, [person(.5, 'a', wrist_up=up), person(.5, 'b', wrist_up=up)])
            events += [item for item in value['events'] if item['event'] == 'exemplar.matched']
        self.assertEqual([item['track_id'] for item in events], ['a', 'a'])
        self.assertEqual((value['count'], module.status()['duplicates']), (2, 5))

    def test_one_action_of_a_person_on_two_tracks_within_100_ms_counts_once(self):
        module = self.counter()
        for ts, a_up, b_up in ((0, False, False), (100, True, False), (150, False, True), (250, False, False)):
            self.push(module, ts, [person(.25, 'a', wrist_up=a_up), person(.75, 'b', wrist_up=b_up)])
        self.bind(module, 300, [dict(event='person.bound', track_id='a', who='P', start_ms=0, end_ms=120, by='face'),
                                dict(event='person.unbound', track_id='a', who='P', start_ms=0, end_ms=120, reason='x'),
                                dict(event='person.bound', track_id='b', who='P', start_ms=121, end_ms=300, by='face')])
        self.assertEqual(module.by_who(), {'P': 1})         # matched at 100 on a and at 150 on b: one action of P


# ---------------------------------------------------------------- everyone, named or not; scene cuts

GREEN = (0, 255, 0)


class SceneCutTest(unittest.TestCase):
    @unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
    def test_a_hard_cut_and_a_dissolve_are_found_and_motion_is_not(self):
        from gamst.providers.scene_cut import SceneCuts
        cuts, found = SceneCuts(), []
        room = np.full((180, 320, 3), 40, np.uint8)
        other = np.full((180, 320, 3), 200, np.uint8)
        for index in range(90):          # 3 s at 30 fps: a person (a block) moving in one room, then a cut
            ts = index * 33
            image = (room if index < 45 else other).copy()
            image[60:120, (index * 3) % 260:(index * 3) % 260 + 60] = 255 if index < 45 else 0
            found.append(cuts.feed(ts, image))
        self.assertEqual([item for item in found if item is not None], [45 * 33])
        fade, found = SceneCuts(), []
        for index in range(120):         # a one-second cross-fade from one room to another
            ts = index * 33
            weight = min(1., max(0., (index - 45) / 30))
            image = (room * (1 - weight) + other * weight).astype(np.uint8)
            found.append(fade.feed(ts, image))
        cut = [item for item in found if item is not None]
        self.assertEqual(len(cut), 1)
        self.assertLess(abs(cut[0] - 60 * 33), 400)      # the middle of the fade


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class EveryoneTest(PersonBindingTest):
    def everyone(self, **options):
        from gamst.providers.person_binding import PersonBinding
        return PersonBinding(Color('face'), Color('body'), face_threshold=.9, face_reject=.5, body_threshold=.95,
                             face_every_ms=0, body_every_ms=0, confirm_every_ms=0, session='file:s',
                             **dict(dict(scene_cuts=False), **options))

    async def test_everyone_gets_a_person_id_and_keeps_it_when_they_come_back(self):
        module, events = self.everyone(), []
        for ts in (1000, 1100):          # nobody enrolled: two faces, two people
            events += (await self.step(module, ts, [person(.25, 'a'), person(.75, 'b')], frame_of(RED, GREEN)))['events']
        value = await self.step(module, 1200, [person(.25, 'a'), person(.75, 'b')], frame_of(RED, GREEN))
        self.assertEqual(sorted((item['track_id'], item['person_id']) for item in value['tracks']),
                         [('a', 'P1'), ('b', 'P2')])
        self.assertEqual({item['who'] for item in value['tracks']}, {'unknown'})        # not named
        self.assertEqual(value['persons'], {'P1': 'a', 'P2': 'b'})
        first = next(item for item in events if item['event'] == 'person.bound' and item['track_id'] == 'a')
        self.assertEqual((first['person_id'], first['start_ms'], first['retroactive']), ('P1', 1000, True))
        back = await self.step(module, 9000, [person(.75, 'c')], frame_of(BLUE, RED))    # P1 returns on a new track
        self.assertEqual(back['persons']['P1'], 'c')
        self.assertEqual(module.counts['new_persons'], 2)

    async def test_a_person_can_be_named_later_and_the_name_reaches_back(self):
        from gamst.providers.person_binding import persons_of
        module, events = self.everyone(), []
        for ts in (1000, 1100, 1200):
            events += (await self.step(module, ts, [person(.25, 'a')], frame_of(RED, BLUE)))['events']
        module.name_person('P1', 'P')
        named = await self.step(module, 1300, [person(.25, 'a')], frame_of(RED, BLUE))
        events += named['events']
        self.assertIn(dict(event='person.named', person_id='P1', who='P', end_ms=1300), events)
        self.assertEqual(resolve(dict(track_id='a', end_ms=1050), intervals_of(events)), 'P')    # before the naming
        self.assertEqual(resolve(dict(track_id='a', end_ms=1050), intervals_of(events, key='person_id'), 'person_id'),
                         'P1')
        persons = persons_of(events, start_ms=1000)
        self.assertEqual([(item['person_id'], item['name'], item['segments'][0]['start']) for item in persons],
                         [('P1', 'P', 0.)])
        later = persons_of(events, start_ms=1000, names={'P1': 'Q'})     # the owner's name in the label tool
        self.assertEqual(later[0]['name'], 'Q')
        collected = module.collected_of('P1')
        self.assertGreaterEqual(len(collected['face']), 2)

    async def test_a_person_enrolled_from_their_person_id_keeps_it(self):
        module = self.everyone()
        for ts in (1000, 1100):
            await self.step(module, ts, [person(.25, 'a')], frame_of(RED, BLUE))
        value = reference(name='P') | dict(face=[[0., 0., 1.]] * 3, enrolled_from=dict(kind='person', person_id='P1'))
        module.set_references(dict(person_1=value))
        later = await self.step(module, 1200, [person(.25, 'a')], frame_of(RED, BLUE))
        self.assertEqual(later['bound'], {'P': 'a'})
        self.assertEqual(later['persons'], {'P1': 'a'})
        self.assertIn(dict(event='person.named', person_id='P1', who='P', end_ms=1200), later['events'])

    async def test_a_face_that_contradicts_withdraws_the_binding_back_to_the_last_confirmation(self):
        module = self.everyone(anonymous=False)
        module.set_references(dict(person=reference(name='P') | dict(face=[[0., 0., 1.]] * 3)))
        events = []
        events += (await self.step(module, 1000, [person(.75, 'b')], frame_of(BLUE, RED)))['events']      # confirmed
        events += (await self.step(module, 2000, [person(.75, 'b', face=False)], frame_of(BLUE, RED)))['events']
        late = await self.step(module, 3000, [person(.75, 'b')], frame_of(RED, BLUE))     # b's face: not P
        events += late['events']
        ended = next(item for item in late['events'] if item['event'] == 'person.unbound')
        self.assertEqual((ended['reason'], ended['end_ms']), ('face_contradicts', 1001))     # last face that said P
        intervals = intervals_of(events)
        self.assertEqual(resolve(dict(track_id='b', end_ms=2500), intervals), 'unknown')     # withdrawn, not only ended
        self.assertEqual(resolve(dict(track_id='b', end_ms=1000), intervals), 'P')

    async def frames(self, module, items):
        """One frame at a time (joints, then the camera frame); the scorer settles after each."""
        from gamst.plugins import InputFrame
        from gamst.private_payload import PrivatePoses
        events = []
        for ts, poses, image in items:
            await module.process(InputFrame(PrivatePoses(poses), 'video', ts))
            value = await module.process(InputFrame(image, 'video', ts))
            await module.settled()
            events += (value or {}).get('events', [])
        return events

    async def test_two_people_crossing_with_swapped_boxes_split_the_tracks_where_they_swapped(self):
        module = self.everyone(anonymous=False)
        module.set_references(dict(person=reference(name='P') | dict(face=[[0., 0., 1.]] * 3)))
        people = frame_of(RED, GREEN)       # P (red) on the left, someone in green on the right, all along
        events = await self.frames(module, [(ts, [person(.25, 'a'), person(.75, 'b')], people)
                                            for ts in range(1000, 1400, 100)])
        self.assertEqual(module.bindings.who('a'), module._ids['P'])
        # they cross; the tracker swaps the boxes: a now follows the green person, b the red one (faces turned away)
        events += await self.frames(module, [(ts, [person(.25, 'b', face=False), person(.75, 'a', face=False)], people)
                                             for ts in range(1400, 2200, 100)])
        splits = [(item['track_id'], item['start_ms'], item['reason']) for item in events if item['event'] == 'track.split']
        self.assertIn(('a', 1400, 'appearance'), splits)
        intervals = intervals_of(events)
        self.assertEqual(resolve(dict(track_id='a', end_ms=1300), intervals), 'P')
        self.assertEqual(resolve(dict(track_id='a', end_ms=1800), intervals), 'unknown')    # not P after the swap
        self.assertEqual(module.bindings.who('b'), module._ids['P'])        # re-linked by appearance in this scene
        self.assertEqual(at_once(events, 'P', range(1000, 2200, 10)), [])

    async def test_one_odd_look_does_not_split_a_track(self):
        module = self.everyone(anonymous=False)
        module.set_references(dict(person=reference(name='P') | dict(face=[[0., 0., 1.]] * 3)))
        seen, hidden = frame_of(BLUE, RED), frame_of(BLUE, (0, 0, 0))      # one frame: P hidden behind something
        items = [(ts, [person(.75, 'b')], hidden if ts == 1500 else seen) for ts in range(1000, 2200, 100)]
        events = await self.frames(module, items)
        self.assertEqual([item for item in events if item['event'] in ('track.split', 'person.unbound')], [])
        self.assertEqual(module.bindings.who('b'), module._ids['P'])

    async def test_a_track_carried_over_a_scene_cut_onto_someone_else_is_split_there(self):
        module, events = self.everyone(scene_cuts=True), []
        for ts in (1000, 1100, 1200):    # scene 1: red face on track a
            events += (await self.step(module, ts, [person(.25, 'a')], frame_of(RED, BLUE)))['events']
        for ts in (1300, 1400, 1500, 1600):      # the cut: the tracker keeps a, now on a green face, colours swapped
            events += (await self.step(module, ts, [person(.25, 'a')], frame_of(GREEN, RED)))['events']
        cut = [item for item in events if item['event'] == 'scene.cut']
        self.assertEqual([item['start_ms'] for item in cut], [1300])
        persons = intervals_of(events, key='person_id')['a']
        self.assertEqual([(low < 1300, high is not None and high < 1300, person) for low, high, person in persons],
                         [(True, True, 'P1'), (False, False, 'P2')])
        self.assertEqual(at_once(events, 'P1', range(1000, 1700, 10)), [])

    async def test_two_tracks_never_hold_one_person_at_once(self):
        module, events, times = self.everyone(), [], []
        for ts in range(1000, 3000, 100):        # two tracks with the same face all along
            events += (await self.step(module, ts, [person(.3, 'a'), person(.7, 'b')],
                                       stripes((80, 160, RED), (240, 320, RED))))['events']
            times += [ts, ts + 1]
        spans = intervals_of(events, key='person_id')
        self.assertEqual([at for at in times
                          if len({person for track in spans for low, high, person in spans[track]
                                  if person and low <= at and (high is None or at <= high)} ) <
                          sum(1 for track in spans for low, high, person in spans[track]
                              if person and low <= at and (high is None or at <= high))], [])


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class PersonCountTest(TrackExemplarTest):
    def test_events_carry_the_person_id_and_by_person_counts_unnamed_people(self):
        module = self.counter()
        self.bind(module, 0, [dict(event='person.bound', track_id='a', who='unknown', person_id='P3', start_ms=0,
                                   end_ms=0, by='face')])
        value = None
        for ts, up in ((10, False), (100, True), (200, False), (300, True)):
            value = self.push(module, ts, [person(.25, 'a', wrist_up=up)])
        matched = [item for item in value['events'] if item['event'] == 'exemplar.matched']
        self.assertEqual([(item['person_id'], item['who']) for item in matched], [('P3', 'unknown')])
        self.assertEqual(value['by_person'], {'P3': 2})


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')      # tests.test_speaker_verify imports numpy
class StorePersonTest(unittest.IsolatedAsyncioTestCase):
    async def test_count_events_and_get_timeline_take_a_person_id(self):
        from gamst import Gamst
        from gamst.capabilities import CapabilitySpec, OutputField, SensorMetadata
        from gamst.providers.person_binding import CAPABILITY
        from tests.test_speaker_verify import now
        async with Gamst(consent=True) as human:
            start = now() - 5000
            human.register_capability(CapabilitySpec('temporal.exemplar_events', 'temporal.exemplar_events', 'video',
                                                     'test', '1', (OutputField('count', 'integer'),
                                                                   OutputField('events', 'array')), 'native'))
            human.register_capability(CapabilitySpec(CAPABILITY, CAPABILITY, 'video', 'test', '1',
                                                     (OutputField('events', 'array'),), 'frame'))
            matched = lambda track, ts: dict(event='exemplar.matched', track_id=track, who='unknown', start_ms=ts - 50,
                                             end_ms=ts, label='raise')
            await human.push_observation('temporal.exemplar_events', dict(count=3, events=[
                matched('a', start + 100), matched('b', start + 300), matched('a', start + 600)]),
                sensor=SensorMetadata('t', 'camera', 1), timestamp_ms=start + 600)
            await human.push_observation(CAPABILITY, dict(events=[
                dict(event='person.bound', track_id='a', who='unknown', person_id='P3', start_ms=start, end_ms=start + 50,
                     by='face'),
                dict(event='person.bound', track_id='b', who='unknown', person_id='P4', start_ms=start, end_ms=start + 50,
                     by='face')]), sensor=SensorMetadata('t', 'camera', 2), timestamp_ms=start + 700)
            self.assertEqual((await human.count_events('temporal.exemplar_events', person_id='P3'))['count'], 2)
            timeline = await human.get_timeline(start_ms=start, end_ms=start + 2000, step_ms=500,
                                                capabilities=['temporal.exemplar_events'], person_id='P4')
            shown = [entry['value'] for entry in timeline['evidence'].values() if entry['kind'] == 'events']
            self.assertEqual([(item['track_id'], item['person_id']) for item in shown], [('b', 'P4')])
            await human.push_observation(CAPABILITY, dict(events=[dict(event='person.named', person_id='P3', who='Q',
                                                                       end_ms=start + 800)]),
                                         sensor=SensorMetadata('t', 'camera', 2), timestamp_ms=start + 800)
            self.assertEqual((await human.count_events('temporal.exemplar_events', who='Q'))['count'], 2)   # retroactive


if __name__ == '__main__':
    unittest.main()
