"""Instructions while a recipe runs. The recipe, its workers, the Pipeline, the Store and the MCP server are the real ones;
only the two model objects (the recognizer, the pose detector) are stand-ins, inside the real worker processes."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import numpy as np

from gamst import Gamst, recipes
from gamst.inputs import CameraInput, MicrophoneInput
from gamst.mcp import MCPServer
from gamst.movement_proposals import Proposals, video_identity
from gamst.pipeline import Pipeline
from gamst.providers.pose_exemplar import ExemplarEvents
from gamst.providers.term_events import TermEvents
from gamst.recipes.instructions import Instructions
from gamst.schema import timestamp
from tests.test_movement_spec import condition, grey_video, movement, entry, frame, repetition, vector
from tests.test_multi_input import HAS_CV, HAS_FFMPEG, Recipes, ffmpeg, wav

ROOT = str(Path(recipes.__file__).parents[2])
WORKER = f'''#!{sys.executable}
import sys, types
import numpy as np

class Whisper:
    """Writes down a word for each loud tenth of a second: the word is how loud it is."""
    WORDS = {{2: '대박', 4: '사과', 6: '바나나'}}
    def __init__(self, *a, **k): pass
    def transcribe(self, samples, **k):
        words = []
        for index in range(len(samples) // 1600):
            level = round(10 * float(np.abs(samples[1600 * index:1600 * (index + 1)]).max()))
            if level in self.WORDS:
                words.append(types.SimpleNamespace(word=' ' + self.WORDS[level], start=index / 10, end=index / 10 + .1, probability=.5))
        text = ''.join(word.word for word in words).strip()
        return ([types.SimpleNamespace(text=text, words=words, avg_logprob=-.1, no_speech_prob=.1, compression_ratio=1.)] if words else []), None

class Detector:
    """One upright body; the grey level of the picture is how high its right wrist is."""
    def detect_for_video(self, image, stamp):
        from tests.test_movement_spec import landmarks
        points = [types.SimpleNamespace(x=p['x'], y=p['y'], z=0., visibility=p['visibility'], presence=p['visibility'])
                  for p in landmarks(.6 - .5 * float(image.mean()) / 255)]
        return types.SimpleNamespace(pose_landmarks=[points])
    def close(self): pass

sys.modules['faster_whisper'] = types.SimpleNamespace(WhisperModel=Whisper)
from gamst.providers.landmarker_video import MediaPipeVideoContext
def from_local(kind, path, sha256, **options):
    api = types.SimpleNamespace(Image=lambda **kw: kw['data'], ImageFormat=types.SimpleNamespace(SRGB=1))
    return MediaPipeVideoContext(kind, Detector, api, sha256=sha256, **options)
MediaPipeVideoContext.from_local = staticmethod(from_local)
from gamst.provider_worker import main
main()
'''
LEAVE = ['--without', 'person', '--without', 'voice']


def speech(seconds, **words):
    """Sound with a loud tenth of a second per word: ``{word: [seconds]}``."""
    level = dict(big=.2, apple=.4, banana=.6)
    samples = np.zeros(16000 * seconds, np.float32)
    for word, times in words.items():
        for at in times:
            samples[round(16000 * at):round(16000 * at) + 1600] = level[word]
    return samples


class Case(Recipes):
    def setUp(self):
        super().setUp()
        self.tmp = self.tmp.resolve()
        self.python = self.tmp / 'python'
        self.python.write_text(WORKER)
        self.python.chmod(0o755)
        (self.tmp / 'whisper').mkdir()
        (self.tmp / 'whisper' / 'model.bin').write_bytes(b'x')
        self.clips = self.tmp / 'clips'
        self.clips.mkdir()
        self.profile = self.tmp / 'profile.json'
        self.write_profile('audio.asr', 'vision.pose')
        patcher = mock.patch.dict(os.environ, PYTHONPATH=ROOT)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_profile(self, *roles):
        entries = {'audio.asr': dict(model='faster_whisper.small', provider_profile='faster_whisper.words', python=str(self.python),
                                     config=dict(model_path=str(self.tmp / 'whisper'))),
                   'vision.pose': dict(model='mediapipe.pose_full', provider_profile='mediapipe.body.video', python=str(self.python),
                                       config=dict(path='/m/pose.task', sha256='1' * 64))}
        self.profile.write_text(json.dumps(dict(version=1, roles={key: entries[key] for key in roles})))

    def scene(self, *arguments, at=None):
        """Run the recipe; ``at``: {position in seconds of the first input: async hook(call, pipeline)}."""
        waiting = sorted((at or {}).items())

        async def update(view, input_id, frame, records):
            view.start = getattr(view, 'start', frame.timestamp_ms)
            while waiting and frame.timestamp_ms - view.start >= 1000 * waiting[0][0]:
                server = MCPServer(view.pipeline.human)
                await waiting.pop(0)[1](server.call_tool, view.pipeline)
            return True

        with mock.patch('gamst.views.CountView.update', update):
            report = recipes.main(['run', 'scene', '--headless', '--profile', str(self.profile), *map(str, arguments)])
        self.assertEqual(waiting, [])
        return report


class Words(Case):
    def test_words_are_replaced_from_a_stated_time_and_never_mixed(self):
        said = speech(12, big=[.5, 1.5, 7.5], apple=[1.0, 8.0, 10.5], banana=[5.5])
        seen = {}

        async def change(call, pipeline):
            seen['before'] = await call('list_instructions')
            seen['set'] = await call('set_instruction', dict(kind='terms', actor='jev', terms=['사과']))
            seen['after'] = await call('list_instructions')

        async def ask(call, pipeline):
            base = seen['start'] = min(value['first_ms'] for value in pipeline.inputs.values())
            seen['count'] = await call('count_events', dict(capability_id='speech.term_events', event='term.spoken'))
            seen['old'] = await call('count_events', dict(capability_id='speech.term_events', event='term.spoken',
                                                          where=dict(reference_revision=1)))
            seen['past'] = await call('search_retained', dict(reference='terms', window='10m'))
            seen['timeline'] = await call('get_timeline', dict(start_ms=base, end_ms=base + 9000, step_ms=500,
                                                               capabilities=['speech.term_events'], max_bytes=262144))
            seen['stored'] = pipeline.human.store.query('local-user', limit=1000, capability_id='speech.term_events',
                                                        registered_only=True)

        report = self.scene('--audio', wav(self.tmp / 'a.wav', said), '--without', 'movement', '--terms', '대박', *LEAVE,
                            at={4.0: change, 11.5: ask})
        change = seen['set']
        self.assertEqual((change['reference'], change['reference_revision'], change['set_by'], change['confirmed']),
                         ('terms', 2, 'jev', False))
        self.assertNotEqual(change['reference_id'], change['previous_id'])
        self.assertEqual(seen['before']['instructions'][1]['set_by'], 'owner')
        now = next(item for item in seen['after']['instructions'] if item['reference'] == 'terms')
        self.assertEqual((now['state'], now['terms'], now['effective_from_ms'], now['complete_from_ms']),
                         ('set', {'사과': ['사과']}, change['effective_from_ms'], change['effective_from_ms']))
        start = seen['start'] - 100
        self.assertEqual(change['effective_from_ms'], start + 4101)     # right after the last sound taken in
        # 사과 at 8.0 only (10.5 is in a window not transcribed yet); 대박 at 7.5 was not asked for any more
        self.assertEqual((seen['count']['count'], seen['old']['count']), (1, 2))
        self.assertEqual([(item['reference_revision'], item['count']) for item in seen['count']['other_instructions']], [(1, 2)])
        self.assertEqual(seen['count']['counted_from_ms'], dict(terms=change['effective_from_ms']))
        event = seen['count']['evidence'][0]['event']
        self.assertEqual((event['reference_id'], event['reference_revision'], event['term'], event['confirmed'], event['method']),
                         (change['reference_id'], 2, '사과', False, 'text'))
        for record in seen['stored']:       # every count carries whose it is
            value = record['observation']['value']
            self.assertEqual({item['reference_revision'] for item in value['events']}, {value['reference_revision']})
        # what was said before the change: found after the fact, apart from the live count
        past = seen['past']
        self.assertEqual((past['count'], past['retrospective'], past['in_live_counts']), (1, True, False))
        self.assertTrue(past['events'][0]['retrospective'])
        self.assertAlmostEqual(past['events'][0]['start_ms'] - start, 1000, delta=1)
        self.assertEqual((past['end_ms'], past['searched']), (change['effective_from_ms'] - 1, ['speech.transcript_words']))
        # the timeline says what was asked for and from when
        old, new = seen['timeline']['instructions']
        self.assertEqual((old['reference_revision'], old['until_ms'], new['reference_revision'], new['from_ms'], new['set_by']),
                         (1, change['effective_from_ms'], 2, change['effective_from_ms'], 'jev'))
        self.assertNotIn('pipeline.instructions', [column['capability'] for column in seen['timeline']['columns']]
                         if 'capability' in seen['timeline']['columns'][0] else json.dumps(seen['timeline']['columns']))
        # the report of the run
        self.assertEqual(report['counts']['speech.term_events'], {'term.spoken': 2})
        self.assertEqual(report['other_instructions']['speech.term_events'][0]['count'], 2)
        self.assertEqual({event['term'] for event in report['events']['speech.term_events']}, {'사과'})
        self.assertTrue(report['complete'])

    def test_a_word_belongs_to_what_was_asked_when_it_was_said(self):
        """One recognizer window spans the change: its words are told apart by their own times."""
        said = speech(6, big=[.5, 2.5], apple=[.8, 2.2])

        async def change(call, pipeline):
            await call('set_instruction', dict(kind='terms', actor='user', terms=dict(과일=['사과', '바나나'])))

        report = self.scene('--audio', wav(self.tmp / 'a.wav', said), '--without', 'movement', '--terms', '대박', *LEAVE, at={1.5: change})
        self.assertEqual([event['term'] for event in report['events']['speech.term_events']], ['과일'])
        self.assertEqual(report['other_instructions']['speech.term_events'][0]['count'], 1)

    def test_removed_words_are_not_looked_for_until_they_are_set_again(self):
        said, seen = speech(18, big=[.5, 7.0, 13.0]), {}

        async def remove(call, pipeline):
            seen['removed'] = await call('remove_instruction', dict(reference='terms', actor='user'))

        async def again(call, pipeline):
            with self.assertRaisesRegex(ValueError, 'not set'):
                await call('search_retained', dict(reference='terms'))
            seen['again'] = await call('set_instruction', dict(kind='terms', actor='user', terms=['대박']))

        async def ask(call, pipeline):
            base = min(value['first_ms'] for value in pipeline.inputs.values())
            rows = (await call('get_timeline', dict(start_ms=base + 5000, end_ms=base + 9000, step_ms=1000,
                                                    capabilities=['speech.term_events'])))['rows']
            seen['cells'] = [(row['complete'], next(iter(row['cells'].values()))) for row in rows]

        report = self.scene('--audio', wav(self.tmp / 'a.wav', said), '--without', 'movement', '--terms', '대박', *LEAVE,
                            at={3.0: remove, 12.2: again, 17.0: ask})
        self.assertEqual((seen['removed']['event'], seen['again']['reference_revision']), ('instruction.removed', 3))
        self.assertEqual({(complete, cell['status'], cell['reason']) for complete, cell in seen['cells']}, {(True, 'off', 'not_instructed')})
        self.assertEqual(report['counts']['speech.term_events'], {'term.spoken': 1})
        self.assertEqual([item['count'] for item in report['other_instructions']['speech.term_events']], [1])

    def test_what_is_refused(self):
        seen = {}

        async def refuse(call, pipeline):
            (self.clips / 'voice.json').write_text(json.dumps(dict(kind='voice', embeddings=[[1.]], consent=dict(statement='x'))))
            (self.clips / 'bad.mp4').write_text('#EXTM3U\nhttp://example.invalid/a.ts\n')
            (self.tmp / 'secret.mp4').write_bytes(b'\0\0\0\x18ftypmp42')
            (self.clips / 'link.mp4').symlink_to(self.tmp / 'secret.mp4')
            (self.clips / 'notes.txt').write_text('x')
            terms = lambda **options: call('set_instruction', dict(dict(kind='terms', actor='jev'), **options))
            clips = lambda *files: call('set_instruction', dict(kind='exemplar', actor='jev', clips=[dict(file=file) for file in files]))
            for problem, action in (
                    ('unsupported value', lambda: call('set_instruction', dict(kind='voice', actor='user', file='voice.json'))),
                    ('actor', lambda: call('set_instruction', dict(kind='terms', terms=['a']))),
                    ('exactly one', lambda: terms()),
                    ('exactly one', lambda: terms(terms=['a'], file='a.json')),
                    ('exactly one', lambda: terms(rule=dict())),
                    ('1..64', lambda: terms(terms=[])),
                    ('1..64', lambda: terms(terms=[f'w{i}' for i in range(65)])),
                    ('not a pattern', lambda: terms(terms=['re:(a+)+$'])),
                    ('nothing to match', lambda: terms(terms=['!!'])),
                    ('not an enrolled terms file', lambda: terms(file='voice.json')),
                    ('no such file', lambda: terms(file='../profile.json')),
                    ('no such file', lambda: terms(file=str(self.profile))),
                    ('path inside', lambda: terms(file='http://example.invalid/a.json')),
                    ('no such file', lambda: clips('link.mp4')),
                    ('no such file', lambda: clips('../secret.mp4')),
                    ('not a media file', lambda: clips('bad.mp4')),
                    ('give one of', lambda: clips('notes.txt')),
                    ('start_s', lambda: call('set_instruction', dict(kind='exemplar', actor='user', clips=[dict(file='bad.mp4', start_s=1.)]))),
                    ('no proposals file', lambda: terms(proposals=['abcdef'])),
                    ('camera is not open.*--camera or --video', lambda: call('set_instruction', dict(
                        kind='exemplar', actor='user', text='arm up', rule=dict(steps=[dict(when=condition())])))),
                    ('no reference', lambda: call('remove_instruction', dict(reference='nothing', actor='user'))),
                    ('since the start', lambda: call('search_retained', dict(reference='terms')))):
                with self.subTest(problem), self.assertRaisesRegex((ValueError, PermissionError), problem):
                    await action()
            seen['human'], seen['list'] = pipeline.human, await call('list_instructions')
            with self.assertRaisesRegex(PermissionError, 'before the run'):     # the application's calls are held to the same
                await pipeline.human.set_instruction('voice', file='voice.json')
            with self.assertRaisesRegex(PermissionError, 'identifies a person'):
                await pipeline.set_reference('me', 'person', {})

        report = self.scene('--audio', wav(self.tmp / 'a.wav', speech(4, big=[.5])), '--without', 'movement', '--terms', '대박',
                            '--clips', self.clips, *LEAVE, at={1.0: refuse})
        self.assertEqual(report['counts']['speech.term_events'], {'term.spoken': 1})        # nothing was changed
        listed = seen['list']
        self.assertEqual(listed['owner_only'], ['person', 'voice'])
        self.assertEqual(sorted(item['file'] for item in listed['clips']['files']), ['bad.mp4', 'link.mp4', 'voice.json'])
        left_out = next(item for item in listed['instructions'] if item['kind'] == 'exemplar')
        self.assertEqual((left_out['state'], left_out['needs_input']), ('left_out', ['camera']))
        self.assertNotIn('changes', [key for key in listed if not listed[key] and key != 'changes'])

    def test_words_confirmed_by_the_owner_and_a_words_file(self):
        store, seen = self.tmp / 'proposals.json', {}
        proposals = Proposals(store)
        item = proposals.propose_term(recipe='scene', spellings=['사과', '바나나'], proposer='jev')
        (self.clips / 'words.json').write_text(json.dumps(dict(kind='terms', format=1, terms={'바나나': ['바나나']}, method='examples',
                                                               confirmed=True)))

        async def change(call, pipeline):
            with self.assertRaisesRegex(ValueError, 'confirmed'):
                await call('set_instruction', dict(kind='terms', actor='jev', proposals=[item['id']]))
            await call('resolve_movement_proposal', dict(proposal_id=item['id'], actor='user', decision='confirm',
                                                         expected_version=item['version']))
            seen['set'] = await call('set_instruction', dict(kind='terms', actor='jev', proposals=[item['id']]))

        async def file(call, pipeline):
            seen['file'] = await call('set_instruction', dict(kind='terms', actor='jev', file='words.json'))

        said = speech(12, apple=[4.5], banana=[4.8, 10.0])
        with mock.patch('gamst.mcp.MCPServer.__init__', _with_proposals(str(store))):
            report = self.scene('--audio', wav(self.tmp / 'a.wav', said), '--without', 'movement', '--terms', '대박', '--clips',
                                self.clips, '--proposals', store, *LEAVE, at={3.0: change, 9.0: file})
        self.assertEqual((seen['set']['confirmed'], seen['file']['confirmed'], seen['file']['reference_revision']), (True, True, 3))
        self.assertEqual(report['counts']['speech.term_events'], {'term.spoken': 1})
        self.assertEqual([(item['reference_revision'], item['count']) for item in report['other_instructions']['speech.term_events']], [(2, 2)])


def _with_proposals(path):
    original = MCPServer.__init__

    def init(self, human, **options):
        original(self, human, movement_proposals=path)
    return init


@unittest.skipUnless(HAS_CV and HAS_FFMPEG, 'needs opencv and ffmpeg')
class Movements(Case):
    def film(self, levels, seconds):
        silent = grey_video(self.tmp / 'long.avi', levels)
        sound, film = wav(self.tmp / 's.wav', speech(seconds, big=[.5])), self.tmp / 'film.mkv'
        ffmpeg('-i', silent, '-i', sound, '-map', '0:v', '-map', '1:a', '-c', 'copy', film)
        return film

    def enrolled(self, name='move.json', peaks=(230, 250)):
        out = self.clips / name
        recipes.main(['enroll', 'scene', '--out', str(out), '--mode', 'sequence', '--profile', str(self.profile), *[
            value for peak in peaks for value in ('--video', grey_video(self.clips / f'ok{peak}.avi', repetition(peak)))]])
        return out

    def test_a_movement_is_replaced_by_clips_a_rule_and_a_proposal_while_it_runs(self):
        rest, seen = [20] * 15, {}
        film = self.film(rest + (repetition() + rest) * 10, 18)      # 17.5 s: ten times
        move = self.enrolled()
        grey_video(self.clips / 'half.avi', [20 + (level - 20) / 3 for level in repetition()])
        store = self.tmp / 'p.json'
        proposal = Proposals(store).propose_intervals(recipe='scene', text='arm up', video=video_identity(self.clips / 'ok250.avi'),
                                                      candidates=[dict(start_s=0., end_s=.95)], duration_s=1.)[0]
        outside = grey_video(self.tmp / 'outside.avi', repetition())
        foreign = Proposals(store).propose_intervals(recipe='scene', text='arm out', video=video_identity(outside),
                                                     candidates=[dict(start_s=0., end_s=.95)], duration_s=1.)[0]
        for item in (proposal, foreign):
            Proposals(store).resolve(proposal_id=item['id'], actor='user', decision='confirm', expected_version=item['version'])

        async def clips(call, pipeline):
            seen['worker'] = pipeline.modules['pose']['module'].worker
            seen['clips'] = await call('set_instruction', dict(
                kind='exemplar', actor='user', text='arm up', mode='sequence', clips=[dict(file='ok230.avi'), dict(file='ok250.avi', start_s=0, end_s=.95)],
                negatives=[dict(file='half.avi')]))
            seen['listed'] = await call('list_instructions')

        async def rule(call, pipeline):
            seen['rule'] = await call('set_instruction', dict(kind='exemplar', actor='jev', text='wrist high', rule=dict(
                steps=[dict(when=dict(measure=dict(height=['right_wrist', 'right_shoulder']), op='gt', value=.5), hold_ms=50)], margin=.1)))
            with self.assertRaisesRegex(ValueError, 'joint positions are never kept'):
                await call('search_retained', dict(reference='exemplar'))

        async def confirmed(call, pipeline):
            with self.assertRaisesRegex(ValueError, 'no such file'):
                await call('set_instruction', dict(kind='exemplar', actor='jev', proposals=[foreign['id']], mode='sequence'))
            seen['proposal'] = await call('set_instruction', dict(kind='exemplar', actor='jev', proposals=[proposal['id']], mode='sequence'))
            seen['past'] = await call('search_retained', dict(reference='exemplar', window='1h'))
            self.assertIs(pipeline.modules['pose']['module'].worker, seen['worker'])        # nothing was restarted

        report = self.scene('--video', film, '--terms', '대박', '--exemplar', move, '--clips', self.clips, '--proposals', store, *LEAVE,
                            '--store', self.tmp / 'run.db', at={4.0: clips, 8.0: rule, 12.0: confirmed})
        listed = next(item for item in seen['listed']['instructions'] if item['kind'] == 'exemplar')
        self.assertEqual((listed['reference_revision'], listed['set_by'], listed['source'], listed['confirmed'], listed['mode']),
                         (2, 'user', 'clips', True, 'sequence'))
        self.assertEqual((seen['clips']['confirmed'], seen['rule']['confirmed'], seen['proposal']['confirmed']), (True, False, True))
        self.assertEqual([seen[key]['reference_revision'] for key in ('clips', 'rule', 'proposal')], [2, 3, 4])
        by_revision = {item['reference_revision']: item['count'] for item in report['other_instructions']['temporal.exemplar_events']}
        events = report['events']['temporal.exemplar_events']
        self.assertEqual({event['reference_revision'] for event in events}, {4})
        self.assertGreaterEqual(min(event['start_ms'] for event in events), seen['proposal']['effective_from_ms'])
        self.assertEqual({event['reference_id'] for event in events}, {seen['proposal']['reference_id']})
        self.assertEqual(set(by_revision), {1, 2, 3})
        matched = [event for event in events if event['event'] == 'exemplar.matched']
        # ten repetitions; one that a change fell into may be nobody's: it is not whole under either instruction
        self.assertIn(len(matched), (2, 3))
        rows = recipes.main(['export', '--store', str(self.tmp / 'run.db'), '--event', 'exemplar.matched', '--out', str(self.tmp / 'e.jsonl')])
        whole = {revision: sum(row['reference_revision'] == revision for row in rows) for revision in (1, 2, 3, 4)}
        self.assertTrue(8 <= sum(whole.values()) <= 10 and all(whole[key] >= 2 for key in (1, 2)), whole)
        for row in rows:
            start = {1: 0, **{seen[key]['reference_revision']: seen[key]['effective_from_ms'] for key in ('clips', 'rule', 'proposal')}}
            until = {1: seen['clips'], 2: seen['rule'], 3: seen['proposal']}.get(row['reference_revision'])
            self.assertGreaterEqual(row['start_ms'], start[row['reference_revision']])
            if until is not None:
                self.assertLess(row['end_ms'], until['effective_from_ms'])
        past = seen['past']
        self.assertEqual((past['retrospective'], past['searched']), (True, ['vision.pose_channels']))
        self.assertTrue(5 <= past['count'] <= 7, past['count'])       # seven had begun by then, whoever had counted them
        self.assertLess(past['end_ms'], seen['proposal']['effective_from_ms'])
        self.assertEqual(report['pipeline']['modules']['exemplar']['errors'], 0)
        self.assertTrue(report['complete'])


class Device:
    """A live camera: an arm goes up for a second, every other second."""
    fps = 25

    def __init__(self, *_):
        self.start = time.monotonic()
        self.index = 0

    def isOpened(self):
        return True

    def read(self):
        time.sleep(max(0., self.start + self.index / self.fps - time.monotonic()))
        phase = (self.index / self.fps) % 2
        self.index += 1
        return True, np.full((48, 64, 3), 20 if phase < 1 else int(20 + 220 * np.sin(np.pi * (phase - 1))), np.uint8)

    def release(self):
        pass


class Stream:
    latency, read_available = 0., 0

    def __init__(self, **options):
        self.start_time = None

    def start(self):
        self.start_time, self.done = time.monotonic(), 0

    stop = close = lambda self: None

    def read(self, count):
        self.done += count
        time.sleep(max(0., self.start_time + self.done / 16000 - time.monotonic()))
        return np.zeros((count, 1), np.float32), False


@unittest.skipUnless(HAS_CV, 'needs opencv')
class Live(Case):
    def start(self, human, *, profile=True):
        args = __import__('argparse').Namespace(recipe='scene', exclude=['movement', 'voice'], include=[], profile=str(self.profile),
                                                terms='대박', exemplar=None, person=None, voice=None, camera=0, microphone=True)
        recipe = recipes.load('scene', exclude=['movement', 'voice'], keep=lambda item: True)
        self.camera = CameraInput(0)
        sources = dict(camera=self.camera, microphone=MicrophoneInput(stream_factory=Stream))
        return recipe, args, sources

    def test_a_movement_is_added_to_a_run_of_words_without_delaying_capture(self):
        import cv2
        for peak in (230, 250):
            grey_video(self.clips / f'ok{peak}.avi', repetition(peak, 25), fps=25)
        original = cv2.VideoCapture

        async def scenario():
            beats, capabilities = [], []

            async def heart():
                while True:
                    before = time.monotonic()
                    await asyncio.sleep(.005)
                    beats.append(time.monotonic() - before)

            async with Gamst(consent=True, sensor_inference_interval_ms=1000) as human:
                recipe, args, sources = self.start(human)
                pipeline = await recipes.build_pipeline(recipe, args, human, sources=sources)
                human.instructions = Instructions(recipe, args, pipeline, folders=[self.clips], on_capability=capabilities.append)
                stop = asyncio.Event()
                run = asyncio.create_task(pipeline.run(stop=stop))
                await asyncio.sleep(1.)
                server = MCPServer(human)
                listed = next(item for item in (await server.call_tool('list_instructions'))['instructions'] if item['kind'] == 'exemplar')
                self.assertEqual(listed, dict(reference='exemplar', kind='exemplar', state='left_out'))
                pulse = asyncio.create_task(heart())
                working = asyncio.create_task(server.call_tool('set_instruction', dict(
                    kind='exemplar', actor='user', mode='sequence', text='arm up', clips=[dict(file='ok230.avi'), dict(file='ok250.avi')])))
                await asyncio.sleep(.05)
                with self.assertRaisesRegex(ValueError, 'being prepared'):
                    await server.call_tool('set_instruction', dict(kind='terms', actor='user', terms=['x']))
                self.assertIn('modules', await server.call_tool('get_pipeline'))        # other tools do not wait for it
                done = await working
                pulse.cancel()
                await asyncio.sleep(7.)
                counted = await human.count_events('temporal.exemplar_events', event='exemplar.matched')
                timeline = await human.get_timeline(start_ms=done['effective_from_ms'] - 1000, end_ms=done['effective_from_ms'] + 5000,
                                                    step_ms=250, capabilities=['exemplar', 'temporal.exemplar_events'])
                status = pipeline.status()
                stop.set()
                await run
                await pipeline.close()
            return done, counted, timeline, status, beats, capabilities

        with mock.patch('cv2.VideoCapture', lambda source, *rest: Device() if type(source) is int else original(source, *rest)):
            done, counted, timeline, status, beats, capabilities = asyncio.run(scenario())
        self.assertEqual(done['added_modules'], ['exemplar', 'pose', 'pose_channels'])
        self.assertEqual(done['not_limited_to'], ['person'])            # no enrolled person: said, not assumed
        self.assertEqual(capabilities, ['temporal.exemplar_events'])
        self.assertIn(counted['count'], (2, 3, 4))
        for item in counted['evidence']:
            self.assertGreaterEqual(item['event']['start_ms'], done['effective_from_ms'])
            self.assertEqual(item['event']['reference_id'], done['reference_id'])
        module = status['modules']['exemplar']
        self.assertEqual((module['complete_from_ms'], module['gaps'], status['modules']['pose']['gaps']), (done['effective_from_ms'], 0, 0))
        self.assertEqual((status['inputs']['camera']['missed'], self.camera.missed, status['inputs']['microphone']['missed']), (0, 0, 0))
        self.assertEqual(status['modules']['asr']['gaps'], 0)
        statuses = [next(iter(row['cells'].values())) for row in timeline['rows']]
        self.assertEqual((statuses[0]['status'], statuses[0]['reason']), ('off', 'not_instructed'))
        self.assertIn('exact', [cell['status'] for cell in statuses])
        self.assertLess(sorted(beats)[int(.99 * len(beats))], .1)      # the loop goes on while clips are read

    def test_a_part_whose_worker_is_not_set_up_names_the_command(self):
        self.write_profile('audio.asr')

        async def scenario():
            async with Gamst(consent=True) as human:
                recipe, args, sources = self.start(human)
                pipeline = await recipes.build_pipeline(recipe, args, human, sources=sources)
                human.instructions = Instructions(recipe, args, pipeline)
                before = set(pipeline.modules)
                with self.assertRaisesRegex(ValueError, 'vision.pose mediapipe.pose_full is not set up; run: gamst recipe setup scene'):
                    await human.set_instruction('exemplar', text='arm up', rule=dict(steps=[dict(when=condition())]))
                self.assertEqual(set(pipeline.modules), before)
                self.assertNotIn('exemplar', pipeline.references)
                await pipeline.close()
        asyncio.run(scenario())

    def test_the_application_sets_a_rule_with_the_same_call(self):
        async def scenario():
            async with Gamst(consent=True) as human:
                recipe, args, sources = self.start(human)
                pipeline = await recipes.build_pipeline(recipe, args, human, sources=sources)
                human.instructions = Instructions(recipe, args, pipeline)
                done = await human.set_instruction('exemplar', text='arm up', rule=dict(steps=[dict(when=condition())]))
                self.assertTrue(done['confirmed'])
                self.assertEqual(done['set_by'], 'user')
                worker = pipeline.modules['pose']['module'].worker
                await pipeline.close()
                self.assertTrue(worker.closed)          # added workers end with the run
            with self.assertRaisesRegex(ValueError, 'no recipe is running'):
                async with Gamst(consent=True) as human:
                    await human.list_instructions()
        asyncio.run(scenario())


class Switching(unittest.TestCase):
    """The counters themselves: the old one until the stated time, the new one from it."""

    def test_a_match_in_progress_ends_at_the_change_and_the_count_starts_again(self):
        module = ExemplarEvents(min_hold_ms=100)
        first, second = movement([entry(vector(1, 0))]), dict(movement([entry(vector(0, 1))]), id='other')
        module._adopt(module.prepare(dict(value=first, label='first', id='abc', revision=1)), 0)
        stream = [module.process(frame(vector(1, 0), 1000 + 50 * i)) for i in range(4)]
        self.assertEqual((stream[-1]['count'], stream[-1]['reference_revision']), (1, 1))
        module.schedule(module.prepare(dict(value=second, label='second', id='other', revision=2)), 1200)
        self.assertEqual(module.process(frame(vector(1, 0), 1199))['reference_revision'], 1)
        closing, opening = module.process(frame(vector(0, 1), 1200))
        self.assertEqual((closing['events'][0]['cut'], closing['reference_revision'], closing['events'][0]['end_ms']),
                         ('instruction_changed', 1, 1199))
        self.assertEqual((opening['count'], opening['reference_id'], opening['reference_revision']), (0, 'other', 2))
        last = [module.process(frame(vector(0, 1), 1250 + 50 * i)) for i in range(3)][-1]
        self.assertEqual((last['count'], module.status()['label']), (1, 'second'))
        module.schedule(None, 1400)
        ended = module.process(frame(vector(0, 1), 1400))
        self.assertEqual(ended[0]['events'][0]['event'], 'exemplar.ended')
        self.assertIsNone(module.process(frame(vector(0, 1), 1450)))

    def test_an_unusable_movement_changes_nothing(self):
        module = ExemplarEvents()
        module.set_exemplar(movement([entry(vector(1, 0))]))
        before = module.status()
        for value in (dict(rule=dict(steps=[])), dict(positives=[], mode='hold'), dict(positives=[entry(vector(1))], mode='nope')):
            with self.assertRaises(ValueError):
                module.prepare(dict(value=value, label='x'))
        self.assertEqual(module.status(), before)

    def test_changes_before_a_new_frame_replace_each_other(self):
        module = TermEvents(terms=['a'])
        for revision, word in ((2, 'b'), (3, 'c')):
            module.schedule(module.prepare(dict(value=[word], id=word, revision=revision)), 500)
        self.assertEqual([list(item.terms) for item in module._words], [['a'], ['c']])

    def test_an_utterance_without_word_times_across_the_change_is_marked(self):
        module = TermEvents(terms=['a'])
        module.schedule(module.prepare(dict(value=['b'], id='b', revision=2)), 500)
        from gamst.plugins import InputFrame
        found = module.process(InputFrame(dict(utterance_id='u', status='final', text='a b', start_ms=400, end_ms=600), 'audio', 600))
        self.assertEqual([(value['events'][0]['term'], value['events'][0]['across_change']) for value in found], [('a', True), ('b', True)])

    def test_the_pipeline_rolls_back_and_bounds(self):
        async def scenario():
            async with Gamst(consent=True) as human:
                pipeline = Pipeline(human, 'x')
                pipeline.add_input('text', modality='audio')
                pipeline.add_reference('terms', 'terms', ['a'])
                terms = TermEvents()
                pipeline.add_module('terms', terms, consumes='text', spec=terms.spec)
                with self.assertRaises(ValueError):
                    await pipeline.set_reference('terms', 'terms', [])
                self.assertEqual(pipeline.references['terms']['revision'], 1)
                start = timestamp() - 60000
                for index in range(40):
                    await pipeline.push('text', None, timestamp_ms=start + 10 * index)
                    await pipeline.set_reference('terms', 'terms', [f'w{index}'])
                self.assertLessEqual(len(terms._words), 16)
                self.assertLessEqual(len(pipeline.status()['modules']['terms']['asked']), 16)
                await human.pause()
                with self.assertRaisesRegex(PermissionError, 'paused'):
                    await pipeline.set_reference('terms', 'terms', ['z'])
                await human.resume()
                answer = await human.get_timeline(start_ms=start, end_ms=start + 400)
                self.assertEqual((len(answer['instructions']), answer['instructions_omitted'] > 0), (8, True))
                await pipeline.close()
        asyncio.run(scenario())


if __name__ == '__main__':
    unittest.main()
