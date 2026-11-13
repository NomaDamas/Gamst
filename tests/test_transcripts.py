import unittest

from humansense import HumanSense, SensorMetadata, SensorStreams, transcript_capability
from humansense.plugins import InputFrame
from humansense.schema import timestamp
from humansense.transcripts import TranscriptMemory, TRANSCRIPT_OUTPUT


def update(text='안녕하세요', revision=0, status='partial', utterance='u1', start=1000, end=1100):
    return dict(utterance_id=utterance, revision=revision, text=text, status=status,
                start_ms=start, end_ms=end, language='ko-KR')


def record(value=None, seq=0, ts=1200, epoch='epoch1', subject='user'):
    return dict(id='r'+str(seq), type='observation.signal', subject_id=subject, timestamp_ms=ts,
        quality={'status': 'sufficient'}, models=[{'id': 'asr', 'version': 'test'}],
        observation=dict(name=TRANSCRIPT_OUTPUT, value=value or update(), modality='audio',
            sensor=dict(source_id='mic', sensor_id='a', sequence=seq, stream_epoch=epoch),
            capability=dict(id='asr', freshness_ms=100), processing_epoch='runtime'))


class TranscriptReducer(unittest.TestCase):
    def put(self, memory, r, now=2000):
        memory.commit(memory.prepare(r, now_ms=now))

    def test_replacements_final_empty_retraction_and_retry(self):
        memory = TranscriptMemory()
        self.put(memory, record())
        self.put(memory, record(update('안녕', 1), 1))
        self.put(memory, record(update('', 2, 'final'), 2))
        self.put(memory, record(update('', 2, 'final'), 3))
        output = memory.snapshot('user', now_ms=2000)
        self.assertEqual(len(output['utterances']), 1)
        self.assertEqual(output['utterances'][0]['update']['text'], '')
        self.assertEqual(output['utterances'][0]['status'], 'final')
        self.assertEqual(output['utterances'][0]['evidence_id'], 'r2')
        with self.assertRaises(ValueError):
            self.put(memory, record(update('invented', 3, 'final'), 4))

    def test_rejected_updates_do_not_mutate(self):
        memory = TranscriptMemory()
        self.put(memory, record())
        before = memory.snapshot('user', now_ms=2000)
        invalid = [record(seq=0), record(update('different', 0), 1), record(update('skip', 2), 1),
                   record(update('bad time', 1, start=1001), 1), record(update('bad end', 1, end=1099), 1),
                   record(update('x'*4097, 1), 1), record(update('future', 1, end=1300), 1),
                   record(update('bad status', 1, 'interrupted'), 1)]
        for r in invalid:
            with self.assertRaises(ValueError):
                self.put(memory, r)
            self.assertEqual(memory.snapshot('user', now_ms=2000), before)

    def test_epochs_sources_subjects_and_interrupt(self):
        memory = TranscriptMemory()
        self.put(memory, record())
        self.put(memory, record(epoch='epoch2'))
        self.put(memory, record(subject='other'))
        self.assertEqual(len(memory.snapshot('user', now_ms=2000)['utterances']), 2)
        memory.interrupt('mic', 'a')
        self.assertTrue(all(v['status'] == 'interrupted' and not v['fresh'] for v in memory.snapshot('user', now_ms=2000)['utterances']))
        with self.assertRaises(ValueError):
            self.put(memory, record(update('late final', 1, 'final'), 1))
        memory.clear('other')
        self.assertEqual(memory.snapshot('other', now_ms=2000)['utterances'], [])
        self.assertEqual(len(memory.snapshot('user', now_ms=2000)['utterances']), 2)

    def test_bounds_eviction_watermark_and_retention(self):
        memory = TranscriptMemory(max_entries=1, max_streams=1, retention_ms=1000)
        self.put(memory, record(), now=1200)
        self.put(memory, record(update(utterance='u2'), 1), now=1200)
        self.assertEqual(len(memory.entries), 1)
        with self.assertRaises(ValueError):
            self.put(memory, record(), now=1200)  # Eviction does not erase source ordering.
        with self.assertRaises(ValueError):
            self.put(memory, record(epoch='epoch2'), now=1200)
        self.assertEqual(memory.snapshot('user', now_ms=2201)['utterances'], [])
        self.assertEqual(memory.streams, {})

    def test_byte_bounds_and_snapshot_truncation(self):
        memory = TranscriptMemory(max_bytes=20000)
        self.put(memory, record(update('가'*1000)))
        self.put(memory, record(update('나'*1000, utterance='u2'), 1))
        result = memory.snapshot('user', now_ms=2000, max_bytes=7000)
        self.assertEqual(len(result['utterances']), 1)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['utterances'][0]['update']['text'], '나'*1000)
        with self.assertRaises(ValueError):
            self.put(TranscriptMemory(max_bytes=10), record())


class TranscriptRuntime(unittest.IsolatedAsyncioTestCase):
    def spec(self):
        return transcript_capability('asr.test', 'fixture', '1', streaming_mode='native')

    async def test_context_replaces_partial_and_keeps_previous_final(self):
        async with HumanSense(consent=True) as human:
            spec = self.spec()
            human.register_capability(spec)
            now = timestamp()-1000
            for sequence, value in enumerate((update('안녕', start=now, end=now+100),
                    update('안녕하세요', 1, 'final', start=now, end=now+200),
                    update('다음', utterance='u2', start=now+300, end=now+400))):
                await human.push_observation(spec.id, value, sensor=SensorMetadata('mic', 'a', sequence), timestamp_ms=now+500+sequence)
            context = await human.get_context()
            entries = context['transcripts']['utterances']
            self.assertEqual([e['update']['text'] for e in entries], ['안녕하세요', '다음'])
            self.assertFalse(any(v.get('name') == TRANSCRIPT_OUTPUT for v in context['observations'].values()))
            before = human.store.count('local-user', 0, timestamp())
            with self.assertRaises(ValueError):
                await human.push_observation(spec.id, update('invalid', 4, start=now, end=now+200),
                    sensor=SensorMetadata('mic', 'a', 3), timestamp_ms=now+503)
            self.assertEqual(human.store.count('local-user', 0, timestamp()), before)
            await human.pause()
            entries = (await human.get_context())['transcripts']['utterances']
            self.assertEqual([e['status'] for e in entries], ['final', 'interrupted'])
            await human.delete_data()
            self.assertEqual((await human.get_context())['transcripts']['utterances'], [])

    async def test_gap_new_epoch_and_clear_interrupt_partial(self):
        class Backend:
            def process(self, frame):
                return update(start=frame.timestamp_ms-10, end=frame.timestamp_ms)
            def reset(self):
                pass
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            spec = self.spec()
            streams.register(spec, Backend)
            await streams.open('mic', 'a', [spec.id])
            now = timestamp()-1000
            for sequence in (0, 2):
                await streams.submit(InputFrame(bytes(320), 'audio', now+sequence*10,
                    sample_rate=16000, sensor=SensorMetadata('mic', 'a', sequence)))
                await streams.drain()
            entries = (await human.get_context())['transcripts']['utterances']
            self.assertEqual([e['status'] for e in entries], ['interrupted', 'partial'])
            self.assertNotEqual(entries[0]['epoch'], entries[1]['epoch'])
            await streams.clear()
            self.assertTrue(all(e['status'] == 'interrupted' for e in (await human.get_context())['transcripts']['utterances']))

    async def test_failed_persistence_does_not_commit_reducer(self):
        async with HumanSense(consent=True) as human:
            spec = self.spec()
            human.register_capability(spec)
            original = human.store.add
            def fail(record):
                raise RuntimeError('storage unavailable')
            human.store.add = fail
            now = timestamp()-100
            with self.assertRaises(RuntimeError):
                await human.push_observation(spec.id, update(start=now-100, end=now),
                    sensor=SensorMetadata('mic', 'a', 0), timestamp_ms=now)
            self.assertEqual(human.transcript_memory.entries, {})
            human.store.add = original
