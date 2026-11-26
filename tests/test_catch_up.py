"""Catch-up lanes (L2) and the verification lane (L4): no audio is lost, results land at their capture times, and a
word is settled once, after its stretch is complete."""
import asyncio
import time
import unittest

from gamst import CapabilitySpec, Gamst, OutputField
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.providers.audio_memory import AudioMemory
from gamst.providers.streaming_asr import StreamingWords
from gamst.providers.word_verification import WordVerification, exclusive, match_occurrences
from gamst.providers.term_events import TermEvents

try:
    import numpy as np
except ImportError:
    np = None

FRAME_MS = 50
RATE = 16000
WORDS = spec = CapabilitySpec('speech.transcript_words', 'speech.transcript_words', 'audio', 'test', '1', (
    OutputField('utterance_id', 'string'), OutputField('text', 'string'), OutputField('status', 'string'),
    OutputField('start_ms', 'integer'), OutputField('end_ms', 'integer'), OutputField('words', 'array'),
    OutputField('trimmed', 'boolean', required=False)), 'native', stateful=True)


def pattern(index):
    """Speech in 400 ms stretches with 150 ms pauses; the frame's first sample says which frame it is."""
    return index % 11 < 8


class Paced:
    """A live microphone stand-in: 50 ms frames, each when its last sample is captured, stamped then."""
    modality, live, missed = 'audio', True, 0

    def __init__(self, frames):
        self.count, self.base = frames, None

    async def frames(self):
        self.base = time.time_ns() // 1_000_000
        for index in range(self.count):
            due = self.base + (index + 1) * FRAME_MS
            wait = due / 1000 - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            samples = np.full(RATE * FRAME_MS // 1000, .2 if pattern(index) else 0., np.float32)
            samples[0] = index / 10000
            yield InputFrame(samples, 'audio', due, sample_rate=RATE)


class Hearer:
    """A recogniser stand-in that cannot copy its state: one word per frame of speech, committed at each pause."""
    owns_process = True

    def __init__(self):
        self.spec, self.open, self.pause_ms, self.number, self.forks = WORDS, [], None, 0, []

    def _word(self, frame):
        index = int(round(float(frame.data[0]) * 10000))
        return dict(text=f'w{index}', start_ms=frame.timestamp_ms - FRAME_MS, end_ms=frame.timestamp_ms)

    def _commit(self):
        if not self.open:
            return None
        words, self.open = self.open, []
        self.number += 1
        return self.spec.validate(dict(utterance_id=f'h{id(self)}:{self.number}', status='final',
                                       text=' '.join(word['text'] for word in words), start_ms=words[0]['start_ms'],
                                       end_ms=words[-1]['end_ms'], words=words))

    def process(self, frame):
        if float(np.abs(frame.data[1:]).max()) > 0:
            if len(frame.data) == RATE * FRAME_MS // 1000:      # a frame cut by a worker's start is not a whole word
                self.open.append(self._word(frame))
            return None
        self.pause_ms = frame.timestamp_ms
        return self._commit()

    @property
    def pending_since_ms(self):
        return self.open[0]['start_ms'] if self.open else None

    def snapshot(self):
        return dict(pause_ms=self.pause_ms)

    def fork(self):
        other = Hearer()
        self.forks.append(other)
        return other

    def skipped(self, start_ms, end_ms):
        return self._commit()

    def finish(self):
        return self._commit()

    trim = staticmethod(StreamingWords.trim)


class Collect:
    def __init__(self):
        self.words = []

    def process(self, frame):
        if isinstance(frame.data, dict):
            self.words += [word['text'] for word in frame.data.get('words') or ()]
        return None


async def run(slowdown=None, frames=60):
    async with Gamst(consent=True) as human:
        pipeline = Pipeline(human, 'catch')
        pipeline.add_input('microphone', Paced(frames), modality='audio')
        hearer, collect = Hearer(), Collect()
        pipeline.add_module('asr', hearer, consumes='microphone', spec=WORDS, queue=500,
                            catch_up=dict(lag_ms=150, max_lookback_ms=2000, tail_ms=200, workers=1), slowdown=slowdown)
        pipeline.add_module('words', collect, consumes='asr')
        await pipeline.run()
        status = pipeline.status()
        await pipeline.close()
        return collect.words, status


@unittest.skipIf(np is None, 'numpy is needed')
class CatchUpTest(unittest.TestCase):
    def test_a_lane_that_falls_behind_skips_to_now_and_a_worker_hears_what_it_skipped(self):
        steady, _ = asyncio.run(run())
        slowed, status = asyncio.run(run(dict(ms=120, spans_s=[[.4, .9], [1.6, 2.1]])))
        asr = status['modules']['asr']
        self.assertEqual(sorted(steady, key=lambda word: int(word[1:])), [f'w{index}' for index in range(60) if pattern(index)])
        self.assertGreaterEqual(asr['catch_up']['skips'], 2)          # it fell behind more than once
        self.assertEqual(asr['catch_up']['done'], asr['catch_up']['skips'])
        self.assertEqual(asr['gaps'], 0)                              # no audio was given up
        self.assertEqual(status['audio_memory']['microphone']['dropped'], 0)
        self.assertEqual(sorted(slowed, key=lambda word: int(word[1:])), sorted(steady, key=lambda word: int(word[1:])))
        self.assertEqual(len(slowed), len(set(slowed)))               # nothing twice

    def test_memory_spills_to_disk_and_keeps_what_is_held(self):
        memory = AudioMemory(memory_bytes=8000, keep_ms=100)
        memory.hold('worker', 0)
        for index in range(20):
            memory.append(np.full(1600, index, np.float32), (index + 1) * 100, RATE)
        self.assertGreater(memory.spilled, 0)
        self.assertLessEqual(memory.in_memory, 8000 + 6400)
        self.assertEqual(float(memory.span(250, 260)[0]), 2.)          # read back from the spill file
        self.assertEqual(len(memory.span(0, 2000)), 20 * 1600)
        memory.hold('worker', None)
        self.assertGreater(memory.start_ms, 1000)                      # released once nobody holds it
        self.assertEqual(memory.dropped, 0)
        memory.close()


def occurrence(term, end, **extra):
    return dict(term=term, end_ms=end, start_ms=end - 300, **extra)


class RuleTest(unittest.TestCase):
    def test_occurrences_pair_one_to_one_nearest_first(self):
        first = [occurrence('a', 1000), occurrence('a', 1200), occurrence('b', 5000)]
        second = [occurrence('a', 1150), occurrence('b', 7000)]
        self.assertEqual(match_occurrences(first, second, 1000), {1: 0})

    def test_exclusive_sound_of_each_speaker(self):
        turns = [(0, 1000, 'A'), (800, 2000, 'B')]
        self.assertEqual(exclusive(turns, 0, 2000), {'A': [(0, 800)], 'B': [(1000, 2000)]})


class FakeEmbedder:
    """Speaker 'embedding' read from the sound itself: the first sample says who it is."""
    model_id = 'fake'

    def embed(self, samples, sample_rate=RATE):
        who = float(np.median(samples))
        return [1., 0.] if who > .15 else [0., 1.]


class FakeDiarizer:
    name = 'fake'

    def __init__(self):
        self.calls = []

    def diarize(self, samples):
        """Speaker by level: .2 is the person, .1 another voice."""
        self.calls.append(len(samples))
        out, step = [], RATE // 10
        for first in range(0, len(samples), step):
            level = float(np.median(samples[first:first + step]))
            if level > 0:
                out.append((first / RATE, (first + step) / RATE, 'P' if level > .15 else 'O'))
        return out


class FakeRecogniser:
    """A second hearing that writes the term wherever the sound is .2 or .1 and the frame index is a multiple of 4."""
    name = 'second'

    def __init__(self, invent=()):
        self.out, self.pending = [], None

    async def process(self, frame):
        index = int(round(float(frame.data[0]) * 10000))
        if index % 4 == 0 and float(np.median(frame.data)) > 0:
            self.out.append(dict(utterance_id=f's{index}', status='final', text='bbo', start_ms=frame.timestamp_ms - 100,
                                 end_ms=frame.timestamp_ms, words=[dict(text='bbo', start_ms=frame.timestamp_ms - 100,
                                                                        end_ms=frame.timestamp_ms)]))
        out, self.out = self.out, []
        return out

    async def finish(self):
        return None

    pending_since_ms = None


@unittest.skipIf(np is None, 'numpy is needed')
class VerificationTest(unittest.TestCase):
    def test_words_are_confirmed_revoked_or_added_once_their_stretch_is_complete(self):
        async def main():
            async with Gamst(consent=True) as human:
                pipeline = Pipeline(human, 'verify')
                pipeline.add_input('microphone', None, modality='audio')
                pipeline.add_input('first', None, modality='text')
                pipeline.add_reference('voice', 'voice', dict(model='fake', embeddings=[[1., 0.]]), consent='test')
                pipeline.add_reference('terms', 'terms', ['bbo'])
                diarizer = FakeDiarizer()
                verify = WordVerification(FakeRecogniser(), diarizer, FakeEmbedder(), audio='microphone',
                                          rule=dict(near_ms=150), windows=dict(step_ms=1000, before_ms=500, after_ms=200))
                pipeline.add_module('verification', verify, consumes=['microphone', 'first'], spec=verify.spec, queue=500)
                terms = TermEvents(report_others=True)
                pipeline.add_module('terms', terms, consumes=['first', 'verification'], spec=terms.spec)
                base = time.time_ns() // 1_000_000 - 60000
                levels = []
                for index in range(40):       # 100 ms frames: person 0-1.5 s, another voice 2-3 s, silence after
                    level = .2 if index < 15 else .1 if 20 <= index < 30 else 0.
                    levels.append(level)
                    samples = np.full(1600, level, np.float32)
                    samples[0] = index / 10000
                    await pipeline.push('microphone', samples, timestamp_ms=base + (index + 1) * 100, sample_rate=RATE)
                # First hearing: 'bbo' at frames 4 (person, also heard twice), 6 (person, heard once: revoked),
                # 24 (other voice, heard twice), 35 (silence: invented). Frames 8 and 12 only the second hearing writes.
                first = []
                for index in (4, 6, 24, 35):
                    end = base + (index + 1) * 100
                    first.append(dict(text='bbo', start_ms=end - 100, end_ms=end, word_id=f'u{index}#0',
                                      state='provisional', match=index < 20, speaker_score=.9 if index < 20 else .1))
                for word in first:
                    value = dict(kind='utterance', reference='voice', utterance_id=word['word_id'][:-2], status='final',
                                 text='bbo', start_ms=word['start_ms'], end_ms=word['end_ms'], words=[word],
                                 speaker=dict(match=word['match']))
                    await pipeline.push('first', value, timestamp_ms=word['end_ms'])
                    await asyncio.sleep(.3)
                await pipeline.drain()
                records = await pipeline.finish()
                rows = [record for record in human.store.query('local-user', kind='observation.signal', limit=500,
                                                               registered_only=True, capability_id='speech.term_events')]
                from gamst.counting import count_events
                counted = count_events(rows, event='term.spoken')
                verdicts = {item['word_id']: item for record in human.store.query(
                    'local-user', kind='observation.signal', limit=500, registered_only=True,
                    capability_id='speech.word_verdicts')
                    for item in record['observation']['value'].get('verdicts') or ()}
                await pipeline.close()
                return counted, verdicts, verify.stats
        counted, verdicts, stats = asyncio.run(main())
        self.assertTrue(verdicts['u4#0']['match'])                       # heard twice, the person's speaker
        self.assertEqual(verdicts['u6#0']['reason'], 'not_heard')
        self.assertEqual(verdicts['u24#0']['reason'], 'other_speaker')
        self.assertEqual(verdicts['u35#0']['reason'], 'no_speech')
        self.assertEqual(stats['added'], 5)          # 0, 8, 12 (the person) and 20, 28 (another voice): heard the second time only
        self.assertEqual(counted['count'], 1 + 3)    # 4, and the added 0, 8, 12
        self.assertEqual(counted['states'], dict(provisional=0, final=4, revoked=1))   # 6 revoked: not heard twice
        self.assertEqual(stats['late'], 0)


if __name__ == '__main__':
    unittest.main()


try:
    import websockets
except ImportError:
    websockets = None


@unittest.skipIf(np is None or websockets is None, 'numpy and websockets')
class StreamingForkTest(unittest.TestCase):
    def test_a_hosted_session_says_what_is_pending_where_it_paused_and_forks_with_the_text_as_context(self):
        import base64
        import json
        from gamst.providers.streaming_hosted import session
        from tests.test_streaming_asr import BASE, _Stand, frames
        firsts = []

        async def scribe(ws):
            first, sent = True, 0
            async for raw in ws:
                message = json.loads(raw)
                if first:
                    firsts.append(message.get('previous_text'))
                    first = False
                sent += len(base64.b64decode(message['audio_base_64'])) // 2
                if message['commit']:
                    await asyncio.sleep(.2)
                    end = sent / 16000
                    words = [dict(text='뽀또', start=end - .5, end=end - .2, type='word')]
                    await ws.send(json.dumps(dict(message_type='committed_transcript_with_timestamps', text='뽀또',
                                                  words=words)))

        async def run():
            async with _Stand(scribe) as stand:
                chosen = session(dict(api='elevenlabs', endpoint=stand.url, commit_strategy='manual'))
                module = StreamingWords(chosen, language='ko', endpointer=None)
                module.endpointer = None
                out = []
                for frame in frames(2):
                    out += await module.process(frame) or []
                self.assertEqual(module.pending_since_ms, BASE)           # nothing is out yet
                await module.skipped(BASE + 2000, BASE + 3000)            # the open turn ends here
                self.assertEqual(module.pending_since_ms, BASE)           # ... and waits for its answer
                for _ in range(100):
                    await asyncio.sleep(.02)
                    if module.pending_since_ms is None:
                        break
                out += module._take() or []
                self.assertIsNone(module.pending_since_ms)
                state = module.snapshot()
                self.assertEqual(state['pause_ms'], BASE + 2000)
                self.assertEqual(state['context'], '뽀또')
                other = module.fork()
                other.restore(state)
                await other.process(frames(1)[0])
                await other.close()
                await module.close()
                kept = StreamingWords.trim(out[0], lambda start, end: end < BASE + 1000)
                return out, kept
        out, kept = asyncio.run(run())
        self.assertEqual([(word['start_ms'], word['end_ms']) for word in out[0]['words']], [(BASE + 1500, BASE + 1800)])
        self.assertIsNone(kept)
        self.assertEqual(firsts, [None, '뽀또'])
