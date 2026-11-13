import unittest
from dataclasses import replace
from humansense import HumanSense, SensorStreams
from humansense.providers.endpoint_asr import SileroEndpointASR
from humansense.providers.vad import SileroActivity
from humansense.schema import timestamp
from test_qwen_asr import frame, np, provider


class ActivityModel:
    def __init__(self, scores):
        self.scores=list(scores)
        self.resets=0
        self.inputs=[]
    def reset_states(self): self.resets+=1
    def score(self, values):
        self.inputs.extend(values)
        return self.scores.pop(0) if self.scores else .1


def pipeline(scores, **kwargs):
    model=ActivityModel(scores)
    vad=SileroActivity(model,model.score,version='fixture')
    return SileroEndpointASR(provider(),vad,minimum_speech_ms=64,silence_ms=64,**kwargs)


@unittest.skipIf(np is None,'numpy not installed')
class EndpointPolicy(unittest.TestCase):
    def test_alignment_and_every_pcm_sample_reaches_asr(self):
        p=pipeline([.9,.9,.1,.1]*2)
        updates=[p.process(frame(i)) for i in range(14)]
        finals=[v for v in updates if v and v['status']=='final']
        self.assertEqual([v['end_ms'] for v in finals],[1140,1280])
        self.assertEqual([v['finalization_reason'] for v in finals],['vad_endpoint']*2)
        self.assertEqual(finals[0]['end_ms'],finals[1]['start_ms'])
        self.assertNotEqual(finals[0]['utterance_id'],finals[1]['utterance_id'])
        self.assertEqual(sum(v for v in p.asr.engine.calls if type(v) is int),14*320)
        self.assertIsNone(p.finish())

    def test_pcm16_and_float_have_identical_scale_and_preserve_owned_input(self):
        a,b=pipeline([.9,.9,.1,.1]),pipeline([.9,.9,.1,.1])
        captured=[[],[]]
        for p,samples in zip((a,b),captured):
            original=p.asr.engine.streaming_transcribe
            def observe(pcm,state,original=original,samples=samples):
                samples.extend(pcm.tolist())
                original(pcm,state)
                pcm[:]=1  # Native ASR mutation cannot affect caller or VAD input.
            p.asr.engine.streaming_transcribe=observe
        for i in range(7):
            raw=np.resize(np.array([-32768,-1000,0,1,1000,32767],dtype=np.int16),320)
            owned=raw.copy()
            x=a.process(replace(frame(i),data=raw))
            y=b.process(replace(frame(i),data=raw.astype(np.float32)/32768.))
            np.testing.assert_array_equal(raw,owned)
            if x is not None:
                self.assertEqual({k:v for k,v in x.items() if k!='utterance_id'},
                                 {k:v for k,v in y.items() if k!='utterance_id'})
            else: self.assertIsNone(y)
        self.assertEqual(captured[0],captured[1])
        self.assertEqual(a.vad.model.inputs,b.vad.model.inputs)
        self.assertEqual(a.vad.model.inputs,captured[0][:len(a.vad.model.inputs)])
        self.assertEqual(len(captured[0]),7*320)

    def test_short_pause_noise_and_silence_do_not_endpoint(self):
        for scores in ([.1]*9, [.9,.1]*4, [.9,.9,.1,.9,.1,.9,.1,.9]):
            p=pipeline(scores)
            outputs=[p.process(frame(i)) for i in range(14)]
            self.assertFalse(any(o and o['status']=='final' for o in outputs))
            self.assertEqual(p.finish()['finalization_reason'],'input_end')

    def test_manual_and_cap_priority_have_no_missing_revision(self):
        p=pipeline([])
        outputs=[p.process(frame(i),manual_endpoint=i==5) for i in range(6)]
        self.assertEqual([v['revision'] for v in outputs if v],[0,1])
        self.assertEqual(outputs[-1]['finalization_reason'],'manual_endpoint')
        p=pipeline([])
        outputs=[p.process(frame(i),manual_endpoint=i==14) for i in range(15)]
        self.assertEqual(outputs[-1]['finalization_reason'],'segment_limit')
        self.assertIsNone(p.finish())

    def test_invalid_gap_and_source_reset_both_states(self):
        for change in (lambda f: replace(f,timestamp_ms=f.timestamp_ms+1),
                       lambda f: replace(f,sensor=replace(f.sensor,source_id='other')),
                       lambda f: replace(f,sensor=replace(f.sensor,sequence=99))):
            p=pipeline([.9]*20)
            for i in range(4): p.process(frame(i))
            with self.assertRaises(ValueError): p.process(change(frame(4)))
            self.assertIsNone(p.asr.state)
            self.assertIsNone(p.vad.owner)
            self.assertFalse(p.armed)
            p.process(frame(0))
        p=pipeline([])
        with self.assertRaises(ValueError): p.process(frame(0),manual_endpoint='yes')
        p.close(); p.close()
        self.assertTrue(p.asr.closed and p.vad.closed)
        with self.assertRaises(RuntimeError): p.process(frame(0))

    def test_backend_error_resets_both_states(self):
        for failing in ('vad','asr'):
            p=pipeline([.9]*10)
            for i in range(4): p.process(frame(i))
            def fail(*args): raise RuntimeError('fixture failure')
            if failing=='vad': p.vad.score=fail
            else: p.asr.engine.finish_streaming_transcribe=fail
            with self.assertRaises(RuntimeError): p.process(frame(4),manual_endpoint=True)
            self.assertIsNone(p.asr.state)
            self.assertIsNone(p.vad.owner)
            self.assertFalse(p.armed)

    def test_policy_validation_and_provenance(self):
        p=pipeline([])
        self.assertEqual(p.spec,pipeline([]).spec)
        for value in (True,0,33,6000):
            with self.assertRaises(ValueError):
                SileroEndpointASR(provider(),SileroActivity(ActivityModel([]),lambda v:.1,version='fixture'),
                                  silence_ms=value)
        other=SileroEndpointASR(provider(),SileroActivity(ActivityModel([]),lambda v:.1,version='fixture'),
                               minimum_speech_ms=64,silence_ms=96)
        self.assertNotEqual(p.spec.version,other.spec.version)


@unittest.skipIf(np is None,'numpy not installed')
class EndpointPublication(unittest.IsolatedAsyncioTestCase):
    async def test_nonterminal_endpoints_keep_stream_open_and_finals_immutable(self):
        from humansense.mcp import MCPServer
        async with HumanSense(consent=True) as human:
            p=pipeline([.9,.9,.1,.1]*2)
            streams=SensorStreams(human)
            streams.register(p.spec,lambda:p)
            await streams.open('fixture','mic',[p.spec.id])
            start=timestamp()-1000
            for i in range(14):
                await streams.submit(frame(i,start)); await streams.drain()
            self.assertIn(('fixture','mic'),streams.streams)
            finals=(await MCPServer(human).call_tool('get_transcripts'))['utterances']
            self.assertEqual(len(finals),2)
            self.assertTrue(all(v['status']=='final' and v['update']['finalization_reason']=='vad_endpoint' for v in finals))
            self.assertEqual(streams.status()['streams'][0]['providers'][0]['errors'],0)
            await streams.submit(frame(14,start)); await streams.drain()
            result=await streams.finish('fixture','mic')
            self.assertEqual(result['published'],[p.spec.id])
            entries=(await human.get_transcripts())['utterances']
            self.assertEqual(len(entries),3)
            self.assertEqual(sum(e['update']['finalization_reason']=='input_end' for e in entries),1)
            self.assertTrue(p.closed)
