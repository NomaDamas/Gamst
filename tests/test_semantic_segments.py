import unittest

from gamst import recipes
from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.streaming_semantic import StreamingSemanticContext


class Backend:
    model_id, model_version = 'm', 'r'

    def __init__(self, events):
        self.events, self.calls = events, []

    def predict(self, frames, timestamps_ms, previous_state):
        self.calls.append(list(timestamps_ms))
        return dict(state={}, events=self.events, summary='a person stands up and leaves', evidence_ids=[])

    def close(self):
        pass


def run(events, count=4, window=4):
    provider = StreamingSemanticContext(Backend(events), window_frames=window)
    out = [provider.process(InputFrame(b'', 'video', 1000 + 500 * i, sensor=SensorMetadata('c', 'v', i)))
           for i in range(count)]
    return provider, [value for value in out if value]


class SegmentTest(unittest.TestCase):
    def test_one_meaning_per_segment_with_events_placed_by_frame(self):
        _, (value,) = run([dict(type='stands up', first_frame=1, last_frame=2)])
        self.assertEqual((value['window']['start_ms'], value['window']['end_ms']), (1000, 2500))
        self.assertEqual(value['events'], [dict(type='stands up', start_ms=1500, end_ms=2000)])

    def test_times_the_model_made_up_are_dropped(self):
        _, (value,) = run([dict(type='a', start_ms=5, end_ms=9e12), dict(type='b', first_frame=3, last_frame=99),
                           dict(type='c', start_ms=2000, end_ms=1500), dict(type='d', first_frame=True, last_frame=2)])
        self.assertEqual(value['events'], [dict(type='a'), dict(type='b'), dict(type='c'), dict(type='d')])

    def test_the_open_segment_is_described_at_the_end_once(self):
        provider, done = run([], count=6)
        self.assertEqual(len(done), 1)
        value = provider.finish()
        self.assertEqual((value['window']['start_ms'], value['window']['frames']), (3000, 2))
        self.assertIsNone(provider.finish())

    def test_the_vlm_is_off_unless_asked_for(self):
        for name in ('person-action-count', 'rotation'):
            self.assertNotIn('semantic', [item['name'] for item in recipes.load(name)['modules']])
            self.assertNotIn('vision.semantic_scene', recipes.roles(recipes.load(name)))
            self.assertIn('semantic', [item['name'] for item in recipes.load(name, ['semantic'])['modules']])


if __name__ == '__main__':
    unittest.main()
