"""Offline tests: answer parsing, scoring against the owner's truth, cost formulas, and one full fake Live run.

    python -m unittest discover -s try-it/testbed/gemini_live/tests -v

No network, no SDK: the Live session is a fake that replays ``fake_live_man_rt.json``.
"""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import live_eval as le  # noqa: E402
import harness  # noqa: E402

G = le.TARGET


def ev(start, end, who=G, said_at=None):
    return dict(start=start, end=end, who=who, said_at=said_at)


class ParseTest(unittest.TestCase):
    def test_json_lines_fences_and_none(self):
        found, bad = le.parse_answer('```json\n{"who":"감스트","start":1,"end":2}\n{"who":"다른 사람","start":3,"end":4}\n```')
        self.assertEqual([(f['start'], f['end'], f['who']) for f in found], [(1.0, 2.0, G), (3.0, 4.0, le.OTHER)])
        self.assertEqual(bad, [])
        for none in ('없음', ' none. ', 'None', ''):
            self.assertEqual(le.parse_answer(none), ([], []))

    def test_points_strings_swapped_and_rejected(self):
        found, bad = le.parse_answer('{"who":"감스트","t":"2.5s"} {"who":"Gamst","start":5,"end":4} {"who":"x"} {bad}')
        self.assertEqual([(f['start'], f['end'], f['who']) for f in found], [(2.5, 2.5, G), (4.0, 5.0, G)])
        self.assertEqual(len(bad), 2)

    def test_who_defaults(self):
        self.assertEqual(le.normalize_who('감스트 (빨간 옷)'), G)
        self.assertEqual(le.normalize_who('other'), le.OTHER)
        self.assertEqual(le.normalize_who(None), G)      # the realtime format names only 감스트

    def test_prompts_never_hold_the_truth(self):
        truth = le.load_truth('man')
        for lang in ('ko', 'en'):
            for mode in ('rt', 'post'):
                text = json.dumps(le.prompt_texts('man', lang, mode, window=(0, 67.255)), ensure_ascii=False)
                for occ in truth['occurrences']:
                    self.assertNotIn(f"{occ['start']}", text)
        self.assertIn('양옆으로', le.prompt_texts('gwan', 'ko', 'rt')['system'])


class ScoreTest(unittest.TestCase):
    def test_truth_copy_scores_one(self):
        truth = le.load_truth('man')['occurrences']
        result = le.score('man', [dict(o) for o in truth])
        self.assertEqual(result[G]['matched'], 16)
        self.assertEqual(result[G]['f1'], 1.0)
        self.assertEqual(result[G]['start_err_s'], 0.0)

    def test_tolerance_and_empty(self):
        self.assertEqual(le.score('man', [ev(58.0, 58.3)])[G]['matched'], 1)    # gap 0.425 s <= 0.5
        self.assertEqual(le.score('man', [ev(57.0, 58.1)])[G]['matched'], 0)    # gap 0.625 s
        empty = le.score('man', [])[G]
        self.assertEqual((empty['said'], empty['recall'], empty['f1']), (0, 0.0, None))

    def test_gwan_examples_are_excluded_and_others_are_not_gamst(self):
        on_example = le.score('gwan', [ev(62.2, 63.8)])[G]
        self.assertEqual((on_example['truth'], on_example['said']), (39, 0))
        whole = le.score('gwan', [ev(62.2, 63.8)], exclude_examples=False)[G]
        self.assertEqual((whole['truth'], whole['matched']), (46, 1))
        other = le.score('gwan', [ev(62.2, 63.8, le.OTHER)], exclude_examples=False)
        self.assertEqual(other[G]['said'], 0)
        self.assertEqual(other[le.mp.EVERYONE]['matched'], 1)

    def test_slam_points_and_latency(self):
        points = le.load_truth('shot')['points']
        result = le.score('shot', [ev(2.1, 2.1, said_at=4.0), ev(3.0, 3.0, said_at=4.0), ev(3.5, 3.5)])[G]
        self.assertEqual((result['truth'], result['matched'], result['said']), (16, 3, 3))
        self.assertEqual(result['latency']['n'], 2)
        self.assertAlmostEqual(result['latency']['max_s'], round(4.0 - points[0], 2))
        self.assertEqual(le.score('shot', [ev(3.0, 3.0)], tol=0.0)[G]['matched'], 0)


class CostTest(unittest.TestCase):
    def test_post_formula_by_hand(self):
        est = le.gemini_estimate('shot', 'post', expected_items=16)
        p = le.GEMINI_PRICE
        tokens_in = 450 + 5 * 258 + 11.9 * (258 + 8 + 25) + 90
        usd = (450 * p['text_in'] + 5 * 258 * p['image_in'] + 11.9 * (258 * p['video_in'] + 8 * p['text_in'] +
               25 * p['audio_in']) + 90 * p['text_in'] + (25 * 16 + 5) * p['text_out']) / 1e6
        self.assertEqual(est['tokens_in'], int(tokens_in))
        self.assertAlmostEqual(est['usd'], round(usd, 4), places=4)
        self.assertEqual((est['sessions'], est['turns']), (1, 1))

    def test_rt_bills_the_context_every_turn(self):
        rt, post = le.gemini_estimate('gwan', 'rt'), le.gemini_estimate('gwan', 'post')
        self.assertEqual(rt['sessions'], 5)                    # 522 s in sessions of <= 110 s
        self.assertEqual(rt['turns'], 5 + sum(-(-min(110, 522.436 - 110 * k) // 5) for k in range(5)))
        self.assertGreater(rt['usd'], 10 * post['usd'])

    def test_usage_cost(self):
        rows = [dict(prompt=dict(TEXT=1_000_000, VIDEO=1_000_000, AUDIO=1_000_000), response=1_000_000),
                dict(prompt=dict(DOCUMENT=10), response=0, response_modality='AUDIO')]
        spent = le.usage_cost(rows)
        self.assertAlmostEqual(spent['usd'], 0.75 + 1.0 + 3.0 + 4.5 + 10 * 0.75 / 1e6, places=6)
        self.assertEqual(spent['unknown'], ['DOCUMENT'])

    def test_claude_estimate_grows_with_fps(self):
        self.assertLess(le.claude_estimate('man', 2)['usd'], le.claude_estimate('man', 4)['usd'])


class ClaudeArmTest(unittest.TestCase):
    def test_score_answer_and_price(self):
        import claude_arm
        answer = ('{"who":"감스트","start":1.0,"end":1.25}\n{"who":"감스트","start":1.0,"end":1.25}\n'
                  '{"who":"다른 사람","start":20,"end":21}\n{"who":"감스트","start":59.0,"end":59.5}')
        usage = dict(input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=0, cache_write_tokens=0)
        result = claude_arm.score_answer('man', 4, 1, answer, usage)
        self.assertEqual(len(result['items']), 3)                 # the exact repeat is dropped
        self.assertEqual(result['score'][G]['matched'], 2)
        self.assertAlmostEqual(result['usd'], 4.0 + 2.0)
        self.assertEqual(result['fps'], 4)
        self.assertNotIn('1.001', claude_arm.prompt('man', 4, 4, Path('/x'), 17))


class FakeSession:
    """Replays the recorded fake: answers the n-th closed activity with turn n of the fixture."""

    def __init__(self, turns, sent):
        self.turns, self.sent, self.queue = turns, sent, asyncio.Queue()
        self.closed = 0

    async def send_realtime_input(self, **kwargs):
        self.sent.append(kwargs)
        if 'activity_end' in kwargs:
            turn = self.turns[self.closed] if self.closed < len(self.turns) else dict(text=['없음'], usage={})
            self.closed += 1
            for k, chunk in enumerate(turn['text']):
                last = k == len(turn['text']) - 1
                usage = None
                if last:
                    usage = SimpleNamespace(
                        prompt_token_count=sum(turn['usage'].values()), response_token_count=turn['response'],
                        thoughts_token_count=0, total_token_count=None,
                        prompt_tokens_details=[SimpleNamespace(modality=SimpleNamespace(name=m), token_count=n)
                                               for m, n in turn['usage'].items()])
                await self.queue.put(SimpleNamespace(text=chunk, usage_metadata=usage, go_away=None,
                                                     server_content=SimpleNamespace(turn_complete=last)))

    async def receive(self):
        while True:
            msg = await self.queue.get()
            yield msg
            if msg.server_content.turn_complete:
                return


class FakeConnect:
    def __init__(self, turns):
        self.turns, self.sent, self.configs = turns, [], []

    def __call__(self, *, model, config):
        self.configs.append((model, config))
        outer = self

        class Ctx:
            async def __aenter__(self):
                return FakeSession(outer.turns, outer.sent)

            async def __aexit__(self, *exc):
                return False
        return Ctx()


class FakeRunTest(unittest.TestCase):
    def test_man_rt_end_to_end(self):
        turns = json.loads((HERE / 'fake_live_man_rt.json').read_text())['turns']
        connect = FakeConnect(turns)
        frames = [(float(t), b'jpeg') for t in range(68)]
        pcm = b'\0\0' * 16000 * 68
        args = SimpleNamespace(clip='man', mode='rt', lang='ko', run=0, audio=True, model='gemini-3.8-live',
                               location='us-central1', session_s=110.0, poll_s=None, pace=400.0, modality='TEXT')
        with tempfile.TemporaryDirectory() as tmp:
            result = asyncio.run(harness.run_condition(args, connect=connect, photos=[b'p'] * 5, frames=frames,
                                                       pcm=pcm, out_dir=tmp))
            self.assertTrue((Path(tmp) / 'result.json').exists())
            self.assertGreater(len((Path(tmp) / 'messages.jsonl').read_text().splitlines()), 15)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['frames_sent'], 68)
        self.assertEqual(len(result['turns']), 15)
        self.assertEqual([t['kind'] for t in result['turns']].count('final'), 1)
        sent_text = [s['text'] for s in connect.sent if 'text' in s]
        self.assertEqual(sum('[보고]' in t for t in sent_text), 14)
        self.assertEqual(sum('[전체 목록]' in t for t in sent_text), 1)
        self.assertEqual(sum(1 for s in connect.sent if 'video' in s), 68 + 5)
        model, config = connect.configs[0]
        self.assertEqual(model, 'gemini-3.8-live')
        self.assertTrue(config['realtime_input_config']['automatic_activity_detection']['disabled'])
        rt = result['realtime']
        self.assertEqual(len(rt['items']), 5)                     # the broken object is rejected
        self.assertEqual(len(result['rejected']), 1)
        self.assertTrue(all(item['said_at'] is not None and item['said_at'] >= item['end'] for item in rt['items']))
        score = rt['score'][G]
        self.assertEqual((score['said'], score['matched']), (5, 4))   # 12-13 s is extra
        self.assertEqual(score['truth'], 16)
        self.assertIsNotNone(score['latency'])
        after = result['after']['score']
        self.assertEqual((after[G]['said'], after[G]['matched']), (4, 4))
        self.assertEqual(after[le.mp.EVERYONE]['said'], 5)
        self.assertEqual(result['spent']['turns'], 15)
        self.assertGreater(result['spent']['usd'], 0)
        self.assertEqual(result['frames_per_clip_s'], round(68 / 67.255, 3))


if __name__ == '__main__':
    unittest.main()
