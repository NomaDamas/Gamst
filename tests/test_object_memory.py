from copy import deepcopy
import unittest

from humansense import HumanSense, CapabilitySpec, OutputField, SensorMetadata, SensorStreams
from humansense.memory import ObjectMemory
from humansense.plugins import InputFrame
from humansense.schema import timestamp


def evidence(objects=None, ts=1000, seq=1, epoch='run1'):
    return {'id': f'evidence-{seq}', 'type': 'observation.signal', 'subject_id': 'local-user',
        'timestamp_ms': ts, 'quality': {'status': 'sufficient'}, 'models': [{'id': 'detector', 'version': '1'}],
        'observation': {'capability': {'id': 'objects'}, 'processing_epoch': epoch,
            'sensor': {'source_id': 'cam', 'sensor_id': 'camera', 'sequence': seq,
                       'coordinate_frame': 'camera-pixels'},
            'value': {'objects': objects if objects is not None else [
                {'track_id': 'cup1', 'label': 'white cup', 'location': {'x': 100},
                 'attributes': {'text': 'hello'}, 'relations': ['right of laptop']}]}}}


class Memory(unittest.TestCase):
    def test_disappearance_retains_evidence_not_current_location(self):
        memory = ObjectMemory()
        memory.observe(evidence(), now_ms=1000)
        self.assertTrue(memory.find(now_ms=1000)[0]['currently_visible'])
        memory.observe(evidence([], ts=1100, seq=2), now_ms=1100)
        found = memory.find(text='hello', now_ms=1100)[0]
        self.assertFalse(found['currently_visible'])
        self.assertIsNone(found['current_location'])
        self.assertEqual(found['last_seen_ms'], 1000)
        self.assertEqual(found['last_observation']['object']['location'], {'x': 100})
        self.assertEqual(found['last_observation']['record_id'], 'evidence-1')
        self.assertEqual(memory.find(text='never seen', now_ms=1100), [])

    def test_epoch_reuse_does_not_assert_identity(self):
        memory = ObjectMemory()
        memory.observe(evidence(), now_ms=1000)
        memory.observe(evidence(epoch='run2', ts=1001), now_ms=1001)
        found = memory.find(now_ms=1001)
        self.assertEqual(len(found), 2)
        self.assertNotEqual(found[0]['entity_id'], found[1]['entity_id'])

    def test_atomic_validation_and_out_of_order_snapshots(self):
        memory = ObjectMemory()
        memory.observe(evidence(), now_ms=1000)
        invalid = evidence([{'track_id': 'bad', 'label': 'bad'}, {'oops': 1}], ts=1100, seq=2)
        with self.assertRaises(ValueError):
            memory.observe(invalid, now_ms=1100)
        self.assertTrue(memory.find(now_ms=1100)[0]['currently_visible'])
        memory.observe(evidence([], seq=3), now_ms=1100)
        with self.assertRaises(ValueError):
            memory.observe(evidence(seq=2), now_ms=1100)

    def test_bounds_expiry_and_defensive_copy(self):
        memory = ObjectMemory(max_entities=1, history_size=2, retention_ms=100, visibility_ms=10)
        memory.observe(evidence(), now_ms=1000)
        for n in (2, 3):
            memory.observe(evidence(ts=1000+n, seq=n), now_ms=1000+n)
        result = memory.find(now_ms=1015)[0]
        self.assertEqual(len(result['history']), 2)
        self.assertFalse(result['currently_visible'])
        result['label'] = 'mutated'
        self.assertEqual(memory.find(now_ms=1015)[0]['label'], 'white cup')
        memory.observe(evidence([{'track_id': 'plate', 'label': 'plate'}], ts=1020, seq=4), now_ms=1020)
        self.assertEqual(memory.find(now_ms=1020)[0]['label'], 'plate')
        self.assertEqual(memory.find(now_ms=1121), [])

    def test_rejects_unframed_location_and_insufficient_evidence(self):
        memory = ObjectMemory()
        bad = evidence()
        del bad['observation']['sensor']['coordinate_frame']
        with self.assertRaises(ValueError):
            memory.observe(bad, now_ms=1000)
        bad = evidence()
        bad['quality']['status'] = 'insufficient'
        with self.assertRaises(ValueError):
            memory.observe(bad, now_ms=1000)


class MemoryRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_reconnect_changes_track_epoch_and_ends_visibility(self):
        class Detector:
            def process(self, frame):
                return {'objects': [{'track_id': '1', 'label': 'cup'}]}
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            streams.register(CapabilitySpec('objects', 'scene.objects', 'video', 'detector', '1',
                (OutputField('objects', 'array'),), 'frame'), Detector)
            for _ in range(2):
                await streams.open('cam', 'sensor', ['objects'])
                await streams.submit(InputFrame({}, 'video', timestamp(), sensor=SensorMetadata('cam', 'sensor', 0)))
                await streams.drain()
                record = human.store.latest_signals('local-user', 0, timestamp())[0]
                await human.remember_objects(record['id'])
                await streams.clear()
                await human.remember_objects(record['id'])
            found = await human.find_objects()
            self.assertEqual(len(found), 2)
            self.assertTrue(all(not e['currently_visible'] for e in found))

    async def test_memory_survives_record_eviction_and_is_deleted(self):
        async with HumanSense(consent=True, max_records=5) as human:
            human.register_capability(CapabilitySpec('objects', 'scene.objects', 'video', 'detector', '1',
                (OutputField('objects', 'array'),), 'frame'))
            record = await human.push_observation('objects', {'objects': [{'track_id': 'cup1', 'label': 'cup'}]},
                sensor=SensorMetadata('camera', 'sensor', 1))
            await human.remember_objects(record['id'])
            for i in range(20):
                await human.push_signal('audio.rms', .1, 'audio')
            self.assertEqual(human.store.get_many([record['id']]), [])
            self.assertEqual(len(await human.find_objects(text='cup')), 1)
            await human.pause()
            self.assertFalse((await human.find_objects())[0]['currently_visible'])
            await human.delete_data()
            self.assertEqual(await human.find_objects(), [])

    async def test_old_evidence_after_reset_does_not_become_visible(self):
        async with HumanSense(consent=True) as human:
            human.register_capability(CapabilitySpec('objects', 'scene.objects', 'video', 'detector', '1',
                (OutputField('objects', 'array'),), 'frame'))
            record = await human.push_observation('objects', {'objects': [{'track_id': 'cup1', 'label': 'cup'}]},
                sensor=SensorMetadata('camera', 'sensor', 1))
            await human.remember_objects(record['id'])
            await human.reset_processing()
            await human.remember_objects(record['id'])
            found = await human.find_objects()
            self.assertEqual(len(found), 1)
            self.assertFalse(found[0]['currently_visible'])
            with self.assertRaises(ValueError):
                await human.remember_objects('not-real')
