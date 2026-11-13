import unittest
from types import SimpleNamespace
from dataclasses import replace
try:
    import numpy as np
except ImportError:
    np = None
from humansense.providers.qwen_asr import QwenStreamingASR
from humansense import HumanSense, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.schema import timestamp


class EngineDouble:
    """Lifecycle/API double; never recognition evidence."""
    backend = 'vllm'
    sampling_params = SimpleNamespace(max_tokens=32)
    def __init__(self): self.calls = []; self.flushes = 0
    def init_streaming_state(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text='', language='', count=0)
    def streaming_transcribe(self, pcm, state):
        self.calls.append(len(pcm)); state.count += len(pcm)
        if state.count >= 1600:
            state.text = '안녕' if state.count < 3200 else '안녕하세요'
            state.language = 'Korean'
    def finish_streaming_transcribe(self, state):
        self.flushes += 1; state.text += '!'; state.language = 'Korean'


def provider(engine=None, **kwargs):
    return QwenStreamingASR(engine or EngineDouble(), model_version='fixture.not_inference',
        chunk_ms=100, max_segment_ms=300, **kwargs)


def frame(i, start=1000):
    return InputFrame(np.zeros(320, dtype=np.float32), 'audio', start+(i+1)*20,
        sample_rate=16000, sensor=SensorMetadata('fixture', 'mic', i, channels=('mono',)))


@unittest.skipIf(np is None, 'numpy not installed')
class QwenLifecycle(unittest.TestCase):
    def test_partial_replacements_tail_final_and_no_duplicate_finish(self):
        engine = EngineDouble(); p = provider(engine)
        updates = [p.process(frame(i)) for i in range(11)]
        updates = [u for u in updates if u is not None]
        self.assertEqual([u['text'] for u in updates], ['안녕', '안녕하세요'])
        self.assertEqual([u['revision'] for u in updates], [0,1])
        self.assertEqual([u['end_ms'] for u in updates], [1100,1200])
        final = p.finish()
        self.assertEqual((final['status'],final['end_ms'],final['revision']), ('final',1220,2))
        self.assertEqual(final['finalization_reason'],'input_end')
        self.assertTrue(all('finalization_reason' not in u for u in updates))
        self.assertIsNone(p.finish()); self.assertEqual(engine.flushes,1)
        self.assertEqual(engine.calls[0]['chunk_size_sec'], .1)

    def test_limit_rollover_and_input_copy(self):
        engine = EngineDouble(); p = provider(engine)
        for i in range(15): result = p.process(frame(i))
        self.assertEqual(result['status'], 'final'); self.assertEqual(result['end_ms'],1300)
        self.assertEqual(result['finalization_reason'],'segment_limit')
        old = result['utterance_id']
        for i in range(15,20): result = p.process(frame(i))
        self.assertNotEqual(result['utterance_id'],old); self.assertEqual(result['revision'],0)
        self.assertLessEqual(p.elapsed_ms,300)
        def mutate(pcm,state): pcm[:] = .5
        p.reset(); engine.streaming_transcribe=mutate
        f=frame(0); p.process(f); self.assertEqual(float(f.data.sum()),0)

    def test_errors_gaps_and_overflow_discard_state(self):
        for bad in (replace(frame(1), data=np.zeros(10)), replace(frame(1),sample_rate=8000),
                    replace(frame(1),timestamp_ms=5000), frame(3)):
            p=provider(); p.process(frame(0))
            with self.assertRaises(ValueError): p.process(bad)
            self.assertIsNone(p.state)
        p=provider(); p.process(frame(0)); p.state.text='x'*4097
        with self.assertRaises(ValueError): p.finish()
        self.assertIsNone(p.state)
        p.close()
        with self.assertRaises(RuntimeError): p.process(frame(0))
        with self.assertRaises(ValueError): provider(language='unsupported')


@unittest.skipIf(np is None, 'numpy not installed')
class QwenPublication(unittest.IsolatedAsyncioTestCase):
    async def test_manual_output_sequence_allows_final_without_new_capture(self):
        async with HumanSense(consent=True) as human:
            p=provider(); human.register_capability(p.spec)
            start=timestamp()-1000; output_sequence=0
            for i in range(11):
                f=frame(i,start); update=p.process(f)
                if update is not None:
                    await human.push_observation(p.spec.id,update,
                        sensor=replace(f.sensor,sequence=output_sequence,stream_epoch='run'),timestamp_ms=f.timestamp_ms)
                    output_sequence+=1
            await human.push_observation(p.spec.id,p.finish(),
                sensor=replace(f.sensor,sequence=output_sequence,stream_epoch='run'),timestamp_ms=f.timestamp_ms)
            snapshot=await human.get_transcripts()
            self.assertEqual(snapshot['utterances'][0]['status'],'final')
            self.assertEqual(snapshot['utterances'][0]['update']['text'],'안녕하세요!')

    async def test_manual_endpoint_publication_uses_distinct_output_sequences(self):
        from humansense.mcp import MCPServer
        async with HumanSense(consent=True) as human:
            p=provider(); human.register_capability(p.spec)
            start=timestamp()-1000; output_sequence=0
            async def publish(value,f):
                nonlocal output_sequence
                if value is not None:
                    await human.push_observation(p.spec.id,value,
                        sensor=replace(f.sensor,sequence=output_sequence,stream_epoch='manual-run'),
                        timestamp_ms=f.timestamp_ms)
                    output_sequence+=1
            for i in range(6):
                f=frame(i,start); await publish(p.process(f),f)
            await publish(p.endpoint(),f)
            for i in range(6,12):
                f=frame(i,start); await publish(p.process(f),f)
            await publish(p.finish(),f)
            entries=(await MCPServer(human).call_tool('get_transcripts'))['utterances']
            self.assertEqual(len(entries),2)
            self.assertTrue(all(e['status']=='final' for e in entries))
            self.assertEqual({e['update']['finalization_reason'] for e in entries},{'manual_endpoint','input_end'})
            self.assertEqual(len({e['sequence'] for e in entries}),2)
            self.assertEqual(sorted(e['update']['start_ms'] for e in entries),[start,start+120])

    async def test_empty_partial_retracts_previously_published_text(self):
        class RetractionEngine(EngineDouble):
            def streaming_transcribe(self, pcm, state):
                state.count += len(pcm)
                if state.count >= 1600:
                    state.text = 'wrong' if state.count < 3200 else ''
                    state.language = 'English'
        async with HumanSense(consent=True) as human:
            p=provider(RetractionEngine()); human.register_capability(p.spec)
            start=timestamp()-1000; emitted=[]
            for i in range(11):
                f=frame(i,start); update=p.process(f)
                if update is not None:
                    emitted.append(update)
                    await human.push_observation(p.spec.id,update,
                        sensor=replace(f.sensor,sequence=len(emitted)-1,stream_epoch='run'),timestamp_ms=f.timestamp_ms)
            self.assertEqual([u['text'] for u in emitted], ['wrong',''])
            self.assertEqual([u['revision'] for u in emitted], [0,1])
            snapshot=await human.get_transcripts()
            self.assertEqual(snapshot['utterances'][0]['update']['text'],'')
            self.assertEqual(snapshot['utterances'][0]['status'],'partial')
            final=p.finish()
            await human.push_observation(p.spec.id,final,
                sensor=replace(f.sensor,sequence=2,stream_epoch='run'),timestamp_ms=f.timestamp_ms)
            snapshot=await human.get_transcripts()
            self.assertEqual(snapshot['utterances'][0]['update']['text'],'!')
            self.assertEqual(snapshot['utterances'][0]['status'],'final')

    async def test_forced_boundary_and_tail_remain_distinct_in_mcp_context(self):
        from humansense.mcp import MCPServer
        async with HumanSense(consent=True) as human:
            p=provider(); streams=SensorStreams(human)
            streams.register(p.spec,lambda:p)
            await streams.open('fixture','mic',[p.spec.id])
            start=timestamp()-1000
            for i in range(20):
                await streams.submit(frame(i,start)); await streams.drain()
            await streams.finish('fixture','mic')
            snapshot=await MCPServer(human).call_tool('get_transcripts')
            entries=sorted(snapshot['utterances'],key=lambda e:e['update']['start_ms'])
            self.assertEqual(len(entries),2)
            self.assertEqual([e['update']['finalization_reason'] for e in entries],['segment_limit','input_end'])
            self.assertEqual(entries[0]['update']['end_ms'],entries[1]['update']['start_ms'])
            self.assertNotEqual(entries[0]['update']['utterance_id'],entries[1]['update']['utterance_id'])
            context=await human.get_context()
            self.assertEqual({e['update']['finalization_reason'] for e in context['transcripts']['utterances']},
                             {'segment_limit','input_end'})

    async def test_managed_stream_finish_flushes_qwen_tail(self):
        async with HumanSense(consent=True) as human:
            engine=EngineDouble(); p=provider(engine)
            streams=SensorStreams(human)
            streams.register(p.spec,lambda:p)
            await streams.open('fixture','mic',[p.spec.id])
            start=timestamp()-1000
            for i in range(11):
                await streams.submit(frame(i,start)); await streams.drain()
            result=await streams.finish('fixture','mic')
            self.assertEqual(result['published'],[p.spec.id])
            entry=(await human.get_transcripts())['utterances'][0]
            self.assertEqual(entry['status'],'final')
            self.assertEqual(entry['update']['text'],'안녕하세요!')
            self.assertEqual(entry['update']['finalization_reason'],'input_end')
            self.assertEqual(entry['sequence'],11)
            self.assertEqual(entry['update']['end_ms'],start+220)
            self.assertEqual(engine.flushes,1)
            self.assertTrue(p.closed)
