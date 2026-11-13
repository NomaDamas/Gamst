from types import SimpleNamespace
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from humansense import HumanSense, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.providers.detection import RFDETRObjects
from humansense.schema import timestamp


class Backend:
    rows = ((10, 10, 40, 40, .9, 1, 'cup'),)

    def predict(self, data, **kwargs):
        self.received, self.options = data, kwargs
        return SimpleNamespace(xyxy=np.array([r[:4] for r in self.rows]),
            confidence=np.array([r[4] for r in self.rows]),
            class_id=np.array([r[5] for r in self.rows], dtype=int),
            data={'class_name': np.array([r[6] for r in self.rows], dtype=object)})


def frame(ts=1000, sequence=0, size=100):
    return InputFrame(np.full((size, size, 3), (10, 20, 30), dtype=np.uint8),
        'video', ts, sensor=SensorMetadata('cam', 'sensor', sequence,
                                          coordinate_frame='camera.normalized'))


@unittest.skipIf(np is None, 'optional numpy not installed')
class Detection(unittest.TestCase):
    def test_rgb_units_provenance_and_disappearance(self):
        backend = Backend()
        provider = RFDETRObjects(backend, version='test')
        first = provider.process(frame())
        self.assertEqual(backend.received[0, 0].tolist(), [30, 20, 10])
        self.assertFalse(backend.options['include_source_image'])
        item = first['objects'][0]
        self.assertEqual(item['location']['bbox']['x'], .1)
        second = provider.process(frame(1010))
        self.assertEqual(item['track_id'], second['objects'][0]['track_id'])
        self.assertEqual(second['objects'][0]['attributes']['association'], 'matched')
        backend.rows = ()
        self.assertEqual(provider.process(frame(1020))['objects'], [])
        backend.rows = Backend.rows
        self.assertEqual(provider.process(frame(1030))['objects'][0]['track_id'], item['track_id'])
        self.assertNotEqual(provider.process(frame(3000))['objects'][0]['track_id'], item['track_id'])
        provider.capability('test').validate(first)

    def test_ambiguous_crossing_retires_history(self):
        backend = Backend()
        backend.rows = ((10, 10, 50, 50, .9, 1, 'cup'), (30, 10, 70, 50, .9, 1, 'cup'))
        provider = RFDETRObjects(backend, version='test')
        original = {o['track_id'] for o in provider.process(frame())['objects']}
        backend.rows = ((20, 10, 60, 50, .9, 1, 'cup'),)
        result = provider.process(frame(1010))['objects'][0]
        self.assertEqual(result['attributes']['association'], 'uncertain_new')
        self.assertNotIn(result['track_id'], original)
        self.assertFalse(original & provider.tracks.keys())

    def test_classes_reset_resize_and_expiry_do_not_reuse_ids(self):
        backend = Backend()
        provider = RFDETRObjects(backend, version='test')
        ids = [provider.process(frame())['objects'][0]['track_id']]
        backend.rows = ((10, 10, 40, 40, .9, 2, 'bowl'),)
        ids.append(provider.process(frame(1010))['objects'][0]['track_id'])
        provider.reset()
        ids.append(provider.process(frame(1020))['objects'][0]['track_id'])
        ids.append(provider.process(frame(1030, size=120))['objects'][0]['track_id'])
        self.assertEqual(len(set(ids)), 4)

    def test_bounded_tracks_truncation_and_invalid_output(self):
        backend = Backend()
        provider = RFDETRObjects(backend, version='test', max_objects=1)
        backend.rows = ((10, 10, 40, 40, .9, 1, 'cup'), (50, 50, 90, 90, .8, 2, 'bowl'))
        result = provider.process(frame())
        self.assertTrue(result['summary']['truncated'])
        self.assertFalse(result['summary']['complete_snapshot'])
        self.assertEqual(len(provider.tracks), 1)
        backend.rows = ((10, 10, float('nan'), 40, .9, 1, 'cup'),)
        with self.assertRaises(ValueError):
            provider.process(frame(1010))
        self.assertEqual(provider.last_ts, 1000)

    def test_bad_input_and_local_loading_fail_before_import(self):
        provider = RFDETRObjects(Backend(), version='test')
        provider.process(frame())
        with self.assertRaises(ValueError):
            provider.process(frame())
        other = frame(1010)
        other = replace(other, sensor=SensorMetadata('different', 'sensor', 0, coordinate_frame='camera.normalized'))
        with self.assertRaises(ValueError):
            provider.process(other)
        provider.close()
        with self.assertRaises(RuntimeError):
            provider.process(frame(1020))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'weights.pth'
            path.write_bytes(b'not a checkpoint')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                RFDETRObjects.from_local(path, sha256='0'*64)

    def test_changing_classes_never_accumulate_unbounded_missing_tracks(self):
        backend = Backend()
        provider = RFDETRObjects(backend, version='test', max_objects=4)
        for index in range(100):
            backend.rows = ((10, 10, 40, 40, .9, index, 'class-'+str(index)),)
            provider.process(frame(1000+index))
            self.assertLessEqual(len(provider.tracks), 4)


@unittest.skipIf(np is None, 'optional numpy not installed')
class DetectionRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_observations_reach_memory_and_survive_disappearance(self):
        backend = Backend()
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            spec = RFDETRObjects.capability('test')
            streams.register(spec, lambda: RFDETRObjects(backend, version='test'))
            await streams.open('cam', 'sensor', [spec.id])
            now = timestamp()-100
            for seq in range(2):
                await streams.submit(frame(now+seq, sequence=seq))
                await streams.drain()
                record = human.store.latest_signals('local-user', 0, timestamp())[0]
                value = record['observation']['value']
                await human.remember_objects(record['id'], complete_snapshot=value['summary']['complete_snapshot'])
                backend.rows = ()
            found = await human.find_objects(text='cup')
            self.assertEqual(len(found), 1)
            self.assertFalse(found[0]['currently_visible'])
            self.assertIsNone(found[0]['current_location'])
