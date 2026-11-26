import asyncio
import math
import time
import unittest

from gamst import Gamst
from gamst.capabilities import CapabilitySpec, OutputField
from gamst.mcp import MCPServer
from gamst.pipeline import Derived, Pipeline
from gamst.plugins import InputFrame
from gamst.providers.pose_exemplar import ExemplarEvents, PoseChannels, pose_channels, subsequence_dtw
from gamst.providers.term_events import TermEvents, normalize


def now():
    return time.time_ns() // 1_000_000


class Doubler:
    """A provider module: value in, observation out."""
    spec = CapabilitySpec('test.double', 'test.double', 'video', 'test.doubler', '1',
                          (OutputField('value', 'number'), OutputField('events', 'array')), 'frame')

    def __init__(self):
        self.resets = 0

    def process(self, frame):
        value = frame.data * 2
        return self.spec.validate(dict(value=value, events=[dict(event='big')] if value > 5 else []))

    def reset(self):
        self.resets += 1


class Separator:
    """A plugin module that derives a new input and commits nothing itself."""
    name = 'separator'

    def process(self, frame):
        return [Derived('target', frame.data + 100, 'video')]


class Sink:
    spec = CapabilitySpec('test.sink', 'test.sink', 'video', 'test.sink', '1',
                          (OutputField('seen', 'number'),), 'frame')

    def __init__(self):
        self.frames = []

    def process(self, frame):
        self.frames.append(frame)
        return self.spec.validate(dict(seen=frame.data if not isinstance(frame.data, dict) else frame.data['value']))


class Source:
    modality = 'video'

    def __init__(self, values, start):
        self.values, self.start = values, start

    async def frames(self):
        for index, value in enumerate(self.values):
            yield InputFrame(value, 'video', self.start + 33 * index)


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_routes_records_links_derived_inputs_and_switches(self):
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'test')
            pipeline.add_input('camera', Source([1, 2, 3, 4], now() - 2000))
            doubler, after, separated = Doubler(), Sink(), Sink()
            pipeline.add_module('double', doubler, consumes='camera', spec=Doubler.spec)
            pipeline.add_module('after', after, consumes='double', spec=CapabilitySpec(
                'test.after', 'test.after', 'video', 'test.after', '1', (OutputField('seen', 'number'),), 'frame'))
            pipeline.add_module('separator', Separator(), consumes='camera', emits=('target',))
            pipeline.add_module('separated', separated, consumes='target', spec=Sink.spec)
            await pipeline.run()
            self.assertEqual([frame.data['value'] for frame in after.frames], [2, 4, 6, 8])
            self.assertEqual([frame.data for frame in separated.frames], [101, 102, 103, 104])
            # Derived frames keep the original capture time and carry their own input id.
            self.assertEqual(separated.frames[0].sensor.sensor_id, 'target')
            # A video frame says its boxes are image coordinates (YOLOE refuses one that does not).
            self.assertEqual(separated.frames[0].sensor.coordinate_frame, 'image')
            pipeline.add_input('microphone', modality='audio')
            self.assertIsNone(pipeline._frame('microphone', [0.], 'audio', now()).sensor.coordinate_frame)
            status = pipeline.status()
            self.assertEqual(status['inputs']['target']['derived_from'], 'separator')
            self.assertEqual(status['modules']['double']['processed'], 4)
            counted = await human.count_events('test.double', event='big')
            self.assertEqual(counted['count'], 2)
            self.assertEqual(len(counted['evidence']), 2)
            # Switching off skips the module; switching on resets it before the next frame.
            pipeline.disable('double')
            await pipeline.push('camera', 9, timestamp_ms=now())
            self.assertEqual(pipeline.status()['modules']['double']['processed'], 4)
            self.assertEqual(pipeline.status()['inputs']['camera']['frames'], 5)
            pipeline.enable('double')
            await pipeline.push('camera', 9, timestamp_ms=now())
            self.assertEqual(doubler.resets, 1)
            self.assertEqual((await human.count_events('test.double', event='big'))['count'], 3)
            await pipeline.close()
            self.assertIsNone(human.pipeline)

    async def test_mcp_pipeline_and_count_tools(self):
        async with Gamst(consent=True) as human:
            server = MCPServer(human)
            with self.assertRaises(ValueError):
                await server.call_tool('get_pipeline', {})
            pipeline = Pipeline(human, 'mcp')
            pipeline.add_input('camera')
            pipeline.add_module('double', Doubler(), consumes='camera', spec=Doubler.spec)
            await pipeline.push('camera', 4, timestamp_ms=now())
            status = await server.call_tool('get_pipeline', {})
            self.assertTrue(status['modules']['double']['enabled'])
            result = await server.call_tool('configure_pipeline_module', {'name': 'double', 'enabled': False})
            self.assertFalse(result['enabled'])
            counted = await server.call_tool('count_events', {'capability_id': 'test.double', 'event': 'big'})
            self.assertEqual(counted['count'], 1)

    async def test_person_reference_needs_consent(self):
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human)
            with self.assertRaises(ValueError):
                pipeline.add_reference('a', 'voice', [0.] * 10)
            pipeline.add_reference('a', 'voice', [0.] * 10, consent=dict(subject='a', basis='owner-attested'))
            self.assertEqual(pipeline.status()['references']['a']['kind'], 'voice')


def body(left_elbow_angle, *, right_wrist_up=False):
    """33 MediaPipe-style landmarks: shoulders level, left arm bent by ``left_elbow_angle`` degrees."""
    points = [dict(x=.5, y=.5, visibility=0.) for _ in range(33)]
    def put(index, x, y):
        points[index] = dict(x=x, y=y, visibility=1.)
    put(11, .45, .4); put(12, .55, .4)          # shoulders
    put(13, .40, .5); put(14, .60, .5)          # elbows
    angle = math.radians(left_elbow_angle)
    put(15, .40 + .1 * math.sin(angle), .5 - .1 * math.cos(angle))  # left wrist around the elbow
    put(16, .60, .3 if right_wrist_up else .6)
    put(23, .46, .7); put(24, .54, .7); put(25, .46, .85); put(26, .54, .85); put(27, .46, 1.); put(28, .54, 1.)
    return points


class PoseExemplarTest(unittest.IsolatedAsyncioTestCase):
    def test_features_do_not_depend_on_position_or_size(self):
        base, mask = pose_channels(body(90))
        moved = [dict(p, x=p['x'] * 2 + .1, y=p['y'] * 2 + .2) for p in body(90)]
        shifted, _ = pose_channels(moved)
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(base, shifted)))
        self.assertGreaterEqual(sum(mask), 12)

    async def test_counts_demonstrated_hold_pose_through_pipeline(self):
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'pose')
            exemplar_values, exemplar_mask = pose_channels(body(30, right_wrist_up=True))
            pipeline.add_reference('shotgun', 'exemplar', dict(channels=exemplar_values, mask=exemplar_mask))
            pipeline.add_input('poses', modality='video')
            channels = PoseChannels()
            pipeline.add_module('pose_channels', channels, consumes='poses', spec=channels.spec)
            matcher = ExemplarEvents(reference='shotgun', mode='hold', min_hold_ms=60)
            pipeline.add_module('shotgun', matcher, consumes='pose_channels', spec=matcher.spec)
            # neutral -> pose (held 5 frames) -> neutral -> pose -> neutral: two holds.
            sequence = ([body(170)] * 5 + [body(30, right_wrist_up=True)] * 5) * 2 + [body(170)] * 5
            start = now() - 2000
            for index, landmarks in enumerate(sequence):
                await pipeline.push('poses', [dict(landmarks=landmarks)], timestamp_ms=start + 33 * index)
            counted = await human.count_events('temporal.exemplar_events', event='exemplar.matched')
            self.assertEqual(counted['count'], 2)
            self.assertEqual((await human.count_events('temporal.exemplar_events', event='exemplar.ended'))['count'], 2)

    def test_sequence_dtw_finds_a_slower_copy_of_the_movement(self):
        template = [([v, 0, 0, 0], [True] * 4) for v in (0, .2, .4, .6, .4, .2, 0)]
        slower = [([v, 0, 0, 0], [True] * 4) for v in (0, .1, .2, .3, .4, .5, .6, .5, .4, .3, .2, .1, 0)]
        unrelated = [([.9, .9, .9, .9], [True] * 4)] * 13
        self.assertLess(subsequence_dtw(template, slower), .06)
        self.assertGreater(subsequence_dtw(template, unrelated), .5)


class TermEventsTest(unittest.IsolatedAsyncioTestCase):
    def test_normalize_drops_spacing_and_punctuation(self):
        self.assertEqual(normalize(' 대 박!! '), '대박')

    async def test_counts_final_transcripts_once_with_particles_and_variants(self):
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'terms')
            pipeline.add_reference('words', 'terms', {'대박': ['대박', '대-박']})
            pipeline.add_input('transcript', modality='audio')
            terms = TermEvents(reference='words')
            pipeline.add_module('terms', terms, consumes='transcript', spec=terms.spec)
            start = now() - 5000
            updates = [dict(utterance_id='u1', status='partial', text='대박', start_ms=0, end_ms=500),
                       dict(utterance_id='u1', status='final', text='와 대박이다 진짜 대 박', start_ms=0, end_ms=900,
                            words=[dict(text='와', start_ms=0, end_ms=100), dict(text='대박이다', start_ms=150, end_ms=500),
                                   dict(text='진짜', start_ms=520, end_ms=700), dict(text='대', start_ms=720, end_ms=800),
                                   dict(text='박', start_ms=800, end_ms=900)]),
                       dict(utterance_id='u1', status='final', text='대박', start_ms=0, end_ms=900),  # repeat
                       dict(utterance_id='u2', status='final', text='대-박 사건', start_ms=1000, end_ms=1500)]
            for index, update in enumerate(updates):
                await pipeline.push('transcript', update, timestamp_ms=start + 100 * index)
            counted = await human.count_events('speech.term_events', event='term.spoken', group_by='term')
            self.assertEqual(counted['count'], 3)
            self.assertEqual(counted['groups'], {'대박': 3})
            timed = [item['event'] for item in counted['evidence'] if item['event']['word_timed']]
            self.assertEqual([(event['start_ms'], event['end_ms']) for event in timed], [(150, 500), (720, 900)])


if __name__ == '__main__':
    unittest.main()
