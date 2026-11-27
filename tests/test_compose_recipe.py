"""compose_recipe / gamst recipe compose (owner decision 46, issue #369): a structured request becomes a validated recipe
with a reason for every choice, or one sentence per missing piece. Test files only; nothing is run."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from gamst import Gamst
from gamst import recipes
from gamst.mcp import MCPServer
from gamst.recipes.compose import compose

CONSENT = dict(statement='P consented to face matching in this test', recorded_ms=1)
SHOTGUN = {"channels": "vision.pose_channels", "steps": [
    {"when": {"measure": {"height": ["right_wrist", "right_shoulder"]}, "op": "gt", "value": 0.0}},
    {"when": {"measure": {"rate": {"height": ["right_wrist", "right_shoulder"]}}, "op": "lt", "value": -10},
     "within_ms": 300},
    {"when": {"all": [{"measure": {"height": ["right_wrist", "right_shoulder"]}, "op": "lt", "value": -0.8},
                      {"measure": {"rate": {"height": ["right_wrist", "right_shoulder"]}}, "op": "gt", "value": -3}]},
     "within_ms": 150}], "margin": 0}


def person(name='감스트', faces=3):
    return dict(kind='person', name=name, consent=CONSENT, face_model='face', body_model='body',
                face=[[1., 0., 0.] for _ in range(faces)], body=[], session=None,
                samples=[dict(source='image', persons=1, face=True, body=False) for _ in range(max(faces, 1))])


class Files:
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(__import__('shutil').rmtree, self.folder)
        (self.folder / 'b.mp4').write_bytes(b'\x00\x00\x00\x18ftypmp42' + bytes(64))
        (self.folder / 'a.wav').write_bytes(b'RIFF' + bytes(64))
        (self.folder / 'gamst.json').write_text(json.dumps(person()))
        (self.folder / 'one-face.json').write_text(json.dumps(person(faces=1)))
        (self.folder / 'voice.json').write_text(json.dumps(dict(kind='voice', consent=CONSENT, embeddings=[[1., 0.]])))
        (self.folder / 'move.json').write_text(json.dumps(dict(kind='exemplar', format=2, label='slam', mode='rule',
                                                               rule=SHOTGUN, text='shotgun')))
        (self.folder / 'imu.csv').write_text('t,az\n0,1\n')

    def path(self, name):
        return str(self.folder / name)

    def shotgun(self, **changes):
        return dict(dict(inputs=dict(video=self.path('b.mp4')), who=dict(name='감스트', person=self.path('gamst.json')),
                         what=dict(movement=dict(exemplar=self.path('move.json'))), mode='count', timing='post'),
                    **changes)


class Compose(Files, unittest.TestCase):
    def chosen(self, result):
        return result['modules']

    def test_a_movement_of_a_person_is_the_hand_assembled_recipe(self):
        result = compose(self.shotgun())
        self.assertTrue(result['ok'], result)
        hand = recipes.load('person-action-count')
        self.assertEqual(result['modules'], [item['name'] for item in hand['modules']])     # issue #369's acceptance
        self.assertEqual(result['records'], hand['records'])
        self.assertEqual(result['run'][:4], ['gamst', 'recipe', 'run', f"{result['name']}.json"])
        for option, value in (('--video', self.path('b.mp4')), ('--exemplar', self.path('move.json')),
                              ('--person', self.path('gamst.json')), ('--who', '감스트'), ('--timing', 'post')):
            self.assertEqual(result['run'][result['run'].index(option) + 1], value)
        why = {item['capability']: item for item in result['reasons']}
        self.assertEqual(why['pose model']['chose'], 'rtmw.l_384x288')
        self.assertIn('post', why['pose model']['why'])
        self.assertIn('identified by face', why['who']['why'])
        self.assertEqual(why['semantic']['chose'], 'off')
        self.assertIn('semantic', result['left_out'])          # OFF in every composed recipe (decision 47)
        realtime = compose(self.shotgun(timing='realtime'))
        self.assertIn('benchmark', {item['capability']: item for item in realtime['reasons']}['pose model']['why'])
        self.assertEqual(realtime['run'][realtime['run'].index('--timing') + 1], 'realtime')

    def test_words_are_the_word_lanes_unchanged(self):
        anyone = compose(dict(inputs=dict(audio=self.path('a.wav')), what=dict(words=['뽀또'])))
        self.assertTrue(anyone['ok'], anyone)
        self.assertEqual(anyone['modules'], [item['name'] for item in recipes.load('person-term-count', (), ('voice',))['modules']])
        self.assertIn('--without', anyone['run'])
        self.assertTrue(any('consent.remote_asr' in sentence for sentence in anyone['before_run']))
        said = compose(dict(inputs=dict(video=self.path('b.mp4')), what=dict(words=['뽀또']),
                            who=dict(name='감스트', person=self.path('gamst.json'), voice=self.path('voice.json')),
                            consent=dict(remote_asr=True)))
        self.assertTrue(said['ok'], said)
        self.assertEqual(said['modules'], [item['name'] for item in recipes.load('person-term-count')['modules']])
        self.assertIn('--allow-remote-asr', said['run'])
        self.assertEqual(said['run'][said['run'].index('--voice') + 1], self.path('voice.json'))
        self.assertEqual(said['inputs'], ['microphone'])       # the lips part that would read the camera is off

    def test_a_person_a_movement_and_words_at_once(self):
        result = compose(dict(inputs=dict(video=self.path('b.mp4')), what=dict(movement=dict(exemplar=self.path('move.json')),
                                                                               words=['뽀또']),
                              who=dict(name='감스트', person=self.path('gamst.json'), voice=self.path('voice.json'))))
        self.assertTrue(result['ok'], result)
        self.assertEqual([entry['id'] for entry in result['recipe']['inputs']], ['camera', 'microphone'])
        self.assertTrue({'exemplar', 'person_binding', 'asr', 'speaker_check', 'terms'} <= set(result['modules']))
        self.assertEqual(set(result['records']), {'temporal.exemplar_events', 'temporal.repetition_count',
                                                  'vision.person_binding', 'speech.term_events'})

    def test_who_alone_is_when_the_person_is_on_screen(self):
        result = compose(dict(inputs=dict(video=self.path('b.mp4')), who=dict(name='감스트', person=self.path('gamst.json'))))
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['modules'], ['pose', 'person_binding'])
        self.assertEqual(result['records'], ['vision.person_binding'])

    def test_every_missing_piece_is_one_sentence(self):
        cases = [
            (dict(inputs=dict(video=self.path('b.mp4')), what=dict(movement=dict(text='shotgun'))), 'no hidden LLM'),
            (self.shotgun(who=dict(name='감스트')), "who='감스트' refused before start: no person reference"),
            (self.shotgun(who=dict(name='감스트', person=self.path('one-face.json'))), 'only 1 usable face crop'),
            (self.shotgun(who=dict(name='Q', person=self.path('gamst.json'))), "enrolls '감스트', not 'Q'"),
            (dict(what=dict(words=['뽀또'])), 'at least one input'),
            (dict(inputs=dict(camera=0), what=dict(words=['뽀또'])), 'words are heard'),
            (self.shotgun(inputs=dict(camera=0), timing='post'), 'a camera or microphone is realtime'),
            (dict(inputs=dict(audio=self.path('a.wav')), what=dict(words=['뽀또']), who=dict(name='감스트', person=self.path('gamst.json'))),
             'who.voice'),
            (dict(inputs=dict(video=self.path('b.mp4'))), 'say what to look for'),
            (self.shotgun(what=dict(movement=dict(clips=[dict(file='x.mp4')]))), 'enrolled first'),
            (self.shotgun(what=dict(movement=dict(rule=SHOTGUN))), 'needs the text'),
            (self.shotgun(inputs=dict(video=self.path('none.mp4'))), 'no file'),
            (self.shotgun(when='now'), 'unknown when'),
            (self.shotgun(inputs=dict(video=self.path('b.mp4'), sensors=[dict(id='imu', file=self.path('imu.csv'),
                                                                            columns=[dict(name='az', unit='g')],
                                                                            time=dict(column='t', unit='s'))])),
             'nothing in this request reads imu'),
        ]
        for request, sentence in cases:
            with self.subTest(sentence):
                result = compose(request)
                self.assertFalse(result['ok'], result)
                self.assertTrue(any(sentence in item for item in result['missing']), result['missing'])
                self.assertTrue(all(item.endswith('.') or 'Few-shot identity is required' in item     # refusal()'s own
                                    for item in result['missing']), result['missing'])

    def test_a_rule_the_caller_wrote_is_saved_and_the_recipe_runs_again(self):
        out = self.folder / 'out' / 'shotgun.json'
        request = dict(name='shotgun', inputs=dict(camera=0), who=dict(name='감스트', person=self.path('gamst.json')),
                       what=dict(movement=dict(text='주먹을 내리친다', rule=SHOTGUN)), consent=dict(capture=True))
        result = compose(request, out=out)
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['before_run'], [])
        saved = json.loads(out.read_text())
        self.assertEqual(saved['composed']['run'], result['run'])
        movement = json.loads(out.with_suffix('.movement.json').read_text())
        self.assertEqual((movement['kind'], movement['method'], movement['confirmed']), ('exemplar', 'rule', True))
        recipe, args = recipes.prepare_run(result['run'][3:], out=open('/dev/null', 'w'))
        self.assertEqual(recipe['name'], 'shotgun')
        self.assertEqual([item['name'] for item in recipe['modules']], result['modules'])
        self.assertEqual((args.who, args.timing, args.camera), ('감스트', 'realtime', 0))
        by_jev = compose(dict(request, name='by-jev'), out=self.folder / 'out' / 'by-jev.json', actor='jev')
        self.assertFalse(json.loads((self.folder / 'out' / 'by-jev.movement.json').read_text())['confirmed'])
        self.assertTrue(by_jev['ok'])
        live = compose(dict(request, consent={}))
        self.assertTrue(any('consent.capture' in sentence for sentence in live['before_run']))

    def test_a_built_in_recipe_is_used_as_it_is_or_customised(self):
        same = compose(dict(recipe='person-action-count', inputs=dict(video=self.path('b.mp4')),
                            what=dict(movement=dict(exemplar=self.path('move.json')))))
        self.assertTrue(same['ok'], same)
        self.assertEqual(same['name'], 'person-action-count')
        self.assertEqual(same['modules'], [item['name'] for item in recipes.load('person-action-count')['modules']])
        changed = compose(dict(recipe='person-action-count', inputs=dict(video=self.path('b.mp4')),
                               what=dict(movement=dict(exemplar=self.path('move.json'))), without=['person_binding'],
                               pose='mediapipe.pose_full', set=['pose.max_poses=2']))
        self.assertTrue(changed['ok'], changed)
        self.assertNotIn('person_binding', changed['modules'])
        pose = next(item for item in changed['recipe']['modules'] if item['name'] == 'pose')
        self.assertEqual((pose['use'], pose['candidate'], pose['max_poses']), ('mediapipe-pose', 'mediapipe.pose_full', 2))
        self.assertIn('customised', changed['reasons'][1]['why'])
        lacking = compose(dict(recipe='person-term-count', inputs=dict(video=self.path('b.mp4')),
                               what=dict(movement=dict(exemplar=self.path('move.json')))))
        self.assertFalse(lacking['ok'])
        self.assertTrue(any('give what.words' in sentence for sentence in lacking['missing']), lacking)
        unknown = compose(dict(recipe='person-action-count', inputs=dict(video=self.path('b.mp4')), with_=1))
        self.assertFalse(unknown['ok'])

    def test_a_sensor_rhythm(self):
        result = compose(dict(inputs=dict(sensors=[dict(id='imu', file=self.path('imu.csv'), modality='imu',
                                                        columns=[dict(name='az', unit='g')], time=dict(column='t', unit='s'))]),
                              what=dict(rhythm=dict(input='imu', channel='az', band='tremor'))))
        self.assertTrue(result['ok'], result)
        self.assertEqual((result['modules'], result['records']), (['rhythm_imu'], ['temporal.rhythm.tremor']))
        self.assertEqual(result['run'][result['run'].index('--sensor') + 1], self.path('imu.csv'))


class OverMCP(Files, unittest.IsolatedAsyncioTestCase):
    async def test_files_come_from_the_owner_folders_and_the_recipe_is_saved_there(self):
        async with Gamst(consent=True) as human:
            server = MCPServer(human, clips=[str(self.folder)])
            result = await server.call_tool('compose_recipe', dict(
                name='shotgun', inputs=dict(video='b.mp4'), who=dict(name='감스트', person='gamst.json'),
                what=dict(movement=dict(text='주먹을 내리친다', rule=SHOTGUN)), timing='post'))
            self.assertTrue(result['ok'], result)
            self.assertEqual(Path(result['saved']['recipe']), self.folder.resolve() / 'recipes' / 'shotgun.json')
            self.assertTrue((self.folder / 'recipes' / 'shotgun.movement.json').is_file())
            outside = await server.call_tool('compose_recipe', dict(inputs=dict(video='/etc/hosts.mp4'),
                                                                    what=dict(words=['a'])))
            self.assertFalse(outside['ok'])
            self.assertIn('--clips folder', outside['missing'][0])
            refused = await server.call_tool('compose_recipe', dict(inputs=dict(camera=0), start=True,
                                                                    what=dict(movement=dict(text='x', rule=SHOTGUN))))
            self.assertEqual((refused['ok'], refused['started']), (False, False))
            self.assertIn('consent.capture', refused['missing'][0])
            self.assertIn('compose_recipe', [tool['name'] for tool in __import__('gamst.mcp', fromlist=['x']).tool_definitions()])
        async with Gamst(consent=True) as human:
            with self.assertRaisesRegex(ValueError, 'gamst mcp --clips DIR'):
                await MCPServer(human).call_tool('compose_recipe', dict(inputs=dict(video='b.mp4')))

    async def test_a_started_run_is_read_in_the_same_session(self):
        """start=true runs the recipe on this session's runtime; a stand-in replaces the run itself."""
        async with Gamst(consent=True) as human:
            server = MCPServer(human, clips=[str(self.folder)])
            seen = {}

            async def run_pipeline(recipe, args, *, human=None, out=None):
                seen.update(recipe=recipe['name'], human=human, out=out, who=args.who, clips=args.clips)
                await asyncio.sleep(0)
                return dict(recipe=recipe['name'], timing='post', frames=3, start_ms=1, counts={'x': {'y': 1}},
                            complete=True)

            from unittest import mock
            with mock.patch('gamst.recipes.run_pipeline', run_pipeline):
                result = await server.call_tool('compose_recipe', dict(
                    name='shotgun', inputs=dict(camera=0), who=dict(name='감스트', person='gamst.json'),
                    what=dict(movement=dict(exemplar='move.json')), consent=dict(capture=True), start=True, wait=True))
            self.assertTrue(result['ok'], result)
            self.assertEqual((result['started'], result['finished'], result['report']['counts']), (True, True, {'x': {'y': 1}}))
            self.assertIs(seen['human'], human)
            self.assertEqual((seen['who'], seen['clips'], seen['recipe']), ('감스트', [str(self.folder.resolve())], 'shotgun'))
            self.assertIsNot(seen['out'], None)       # never the protocol's stdout
            status = await server.call_tool('get_pipeline', {})
            self.assertEqual((status['running'], status['last_run']['frames']), (False, 3))


if __name__ == '__main__':
    unittest.main()
