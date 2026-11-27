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
        module = PersonBinding(Color('face'), Color('body'), face_threshold=.9, face_reject=.5, body_threshold=.95,
                               face_every_ms=0, body_every_ms=0, session='file:s', **options)
        module.set_references(dict(person=reference(face=3, name='P') | dict(face=[[0., 0., 1.]] * 3)))
        return module

    async def step(self, module, ts, poses, image):
        from gamst.plugins import InputFrame
        from gamst.private_payload import PrivatePoses
        await module.process(InputFrame(PrivatePoses(poses), 'video', ts))
        return await module.process(InputFrame(image, 'video', ts))

    async def test_a_face_binds_its_track_from_the_track_start_and_another_face_releases_it(self):
        module = self.module()
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

    async def test_camera_seconds_collect_the_one_person_in_view(self):
        module = self.module()

        async def feed():
            await asyncio.sleep(0)
            for ts in range(0, 1400, 200):
                poses = [person(.75, 'b')] if ts != 400 else [person(.25, 'a'), person(.75, 'b')]
                await self.step(module, 10000 + ts, poses, frame_of(BLUE, RED))
        collected, _ = await asyncio.gather(module.collect(1.), feed())
        self.assertEqual(collected['skipped'], 1)
        self.assertEqual(len(collected['face']), 5)
        self.assertTrue(all(sample['persons'] == 1 for sample in collected['samples']))

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

    def test_without_a_person_the_binding_stays_out_unless_asked_for(self):
        from gamst.recipes import not_given, read_recipe
        raw = read_recipe('person-action-count')[1]
        self.assertEqual(not_given(raw, argparse.Namespace(person=None)), ['person_binding'])
        self.assertEqual(not_given(raw, argparse.Namespace(person='p.json')), [])
        self.assertEqual(not_given(raw, argparse.Namespace(person=None), ['person_binding']), [])

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


if __name__ == '__main__':
    unittest.main()
