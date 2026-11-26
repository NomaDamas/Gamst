"""User-made modules and inputs in recipes, the hosted VLM's separate consent, and a camera that is only shown."""
import asyncio
import json
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from gamst import Gamst, ProcessProvider, recipes
from gamst.capabilities import CapabilitySpec, OutputField
from gamst.plugins import InputFrame, PerceptionPlugin, _signal
from gamst.recipes.perception_parts import _semantic_scene
from gamst.schema import timestamp
from tests.test_multi_input import HAS_CV, HAS_FFMPEG, Recipes, media


class Brightness(PerceptionPlugin):
    """A user's plugin: signals, no contract."""
    name, version, modalities = 'brightness', '7', ('video',)

    def __init__(self, scale=1):
        self.scale = scale

    async def process(self, frame):
        return [_signal(frame, 'user.brightness', self.scale * float(frame.data.mean()), self.name, self.version)]


class Peaks:
    """A user's provider: a contract, values."""
    modalities = ('imu',)
    owns_process = True     # gets its own lane, like any worker
    spec = CapabilitySpec('user.peaks', 'user.peaks', 'imu', 'user', '1', (OutputField('count', 'integer'), OutputField('events', 'array')),
                          'native', stateful=True)

    def __init__(self):
        self.count = 0

    async def process(self, frame):
        if frame.data['channels'][0] < 1:
            return None
        self.count += 1
        return dict(count=self.count, events=[dict(event='peak', start_ms=frame.timestamp_ms, end_ms=frame.timestamp_ms)])


class Leaky(Brightness):
    async def process(self, frame):
        return [_signal(frame, 'user.pose', dict(landmarks=[[.1, .2]]), 'leaky', '1')]


class Gyro:
    """A user's input."""
    modality, live = 'imu', False

    def __init__(self, samples=20, bad=None, ahead=0):
        self.samples, self.bad, self.ahead, self.closed = samples, bad, ahead, False

    async def frames(self):
        base = timestamp() - 30000
        for index in range(self.samples):
            data = dict(channels=[float(index % 5 == 0)], mask=[True])
            yield 'not a frame' if index == self.bad else InputFrame(data, self.modality, base + 100 * index + self.ahead)

    def close(self):
        self.closed = True


def entries(**factories):
    """Installed entry points (what ``importlib.metadata`` finds), by group."""
    def entry_points(*, group):
        return [SimpleNamespace(name=name, value='user:x', dist=None, load=lambda factory=factory: factory)
                for name, factory in factories.get(group.split('.')[1], {}).items()]
    return mock.patch('gamst.extensions.entry_points', entry_points)


INSTALLED = dict(plugins=dict(brightness=Brightness, peaks=Peaks, leaky=Leaky, wrong=lambda: object()), inputs=dict(gyro=Gyro, odd=Brightness))
SHOWN = dict(name='mine', summary='s', view='count', records=[], inputs=[dict(id='camera', use='camera', sources=['camera', 'video'])],
             modules=[dict(name='bright', use='plugin', plugin='brightness', consumes=['camera'], options=dict(scale=2))])
MOVED = dict(name='moved', summary='s', view='count', records=['user.peaks'],
             inputs=[dict(id='gyro', use='plugin', plugin='gyro', modality='imu')],
             modules=[dict(name='peaks', use='plugin', plugin='peaks', consumes=['gyro'])])


class Plugins(Recipes):
    def setUp(self):
        super().setUp()
        patcher = entries(**INSTALLED)
        patcher.start()
        self.addCleanup(patcher.stop)

    @unittest.skipUnless(HAS_CV and HAS_FFMPEG, 'needs opencv and ffmpeg')
    def test_a_users_module_runs_like_any_other_when_the_owner_allows_it(self):
        film = media(self.tmp / 'a.mp4', sound=False)
        self.refused('trusted code.*--plugin brightness', '--video', film, recipe=SHOWN)
        store = self.tmp / 'run.db'
        report = self.run_recipe('--video', film, '--plugin', 'brightness', '--store', store, recipe=SHOWN)
        module = report['pipeline']['modules']['bright']
        self.assertEqual((module['processed'], module['errors'], module['complete']), (120, 0, True))

        async def kept():
            async with Gamst(consent=True, store=str(store)) as human:
                row = human.store.query('local-user', limit=1, name='user.brightness')[0]
                answer = await human.get_timeline(start_ms=row['timestamp_ms'] - 500, end_ms=row['timestamp_ms'] + 1, step_ms=100)
                return row, json.dumps(answer)
        row, answer = asyncio.run(kept())
        self.assertEqual(row['models'][0], dict(id='brightness', version='7', backend='plugin'))
        self.assertIn('user.brightness', answer)

    def test_a_users_input_and_provider(self):
        report = self.run_recipe('--plugin', 'gyro', '--plugin', 'peaks', recipe=MOVED)
        self.assertEqual((report['frames'], report['counts']['user.peaks'], report['complete']), (20, {'peak': 4}, True))
        module = report['pipeline']['modules']['peaks']
        self.assertEqual((module['lane'], module['capability']), ('peaks', 'user.peaks'))

    def test_each_is_allowed_by_name_and_refusals_say_what_to_do(self):
        self.refused('--plugin gyro', recipe=MOVED)
        self.refused('--plugin peaks', '--plugin', 'gyro', recipe=MOVED)
        missing = dict(MOVED, inputs=[dict(MOVED['inputs'][0], plugin='nope')])
        self.refused("0 gamst.inputs named 'nope'.*installed: gyro, odd.*install the package", '--plugin', 'nope', recipe=missing)
        self.refused('must define modality and frames', '--plugin', 'odd', recipe=dict(MOVED, inputs=[dict(MOVED['inputs'][0], plugin='odd')]))
        self.refused('gives imu, the recipe declares thermal', '--plugin', 'gyro', recipe=dict(
            MOVED, inputs=[dict(MOVED['inputs'][0], modality='thermal')]))
        self.refused('reads video; gyro is imu', '--plugin', 'gyro', '--plugin', 'brightness', recipe=dict(
            MOVED, modules=[dict(MOVED['modules'][0], plugin='brightness')]))
        self.refused('modalities and process', '--plugin', 'gyro', '--plugin', 'wrong', recipe=dict(
            MOVED, modules=[dict(MOVED['modules'][0], plugin='wrong')]))
        with self.assertRaisesRegex(ValueError, 'declares its modality'):
            self.run_recipe(recipe=dict(MOVED, inputs=[dict(id='gyro', use='plugin', plugin='gyro')]))

    def test_the_same_guards_hold(self):
        for options, problem in (dict(bad=5), 'a frame is an InputFrame'), (dict(ahead=60000), 'not in the future'):
            recipe = dict(MOVED, inputs=[dict(MOVED['inputs'][0], options=options)])
            if options.get('ahead'):
                self.refused(problem, '--plugin', 'gyro', '--plugin', 'peaks', recipe=recipe)
                continue
            report = self.run_recipe('--plugin', 'gyro', '--plugin', 'peaks', recipe=recipe)     # what came before it stays
            self.assertEqual((report['frames'], report['complete']), (5, False))
            self.assertIn(problem, report['failures'][0]['error'])

    @unittest.skipUnless(HAS_CV and HAS_FFMPEG, 'needs opencv and ffmpeg')
    def test_a_plugin_cannot_put_positions_into_a_record(self):
        report = self.run_recipe('--video', media(self.tmp / 'a.mp4', sound=False), '--plugin', 'leaky', recipe=dict(
            SHOWN, modules=[dict(name='bright', use='plugin', plugin='leaky', consumes=['camera'])]))
        self.assertIn('private field', report['errors']['bright'])
        self.assertTrue(report['failures'][0]['given_up'])


class HostedContext(unittest.IsolatedAsyncioTestCase):
    CONFIG = dict(endpoint='https://vlm.example.invalid/v1', model='m', revision='r', api_key_env='TEST_VLM_KEY', allow_remote=True)

    async def build(self, config=None, **context):
        opened = []

        async def fake_open(python, profile, config, **options):
            opened.append(config)
            return SimpleNamespace(spec=SimpleNamespace(provider='x', version='1', modality='video'), private=False)

        recipe = recipes.load('scene', ['semantic'], ['voice'])
        item = dict(next(item for item in recipe['modules'] if item['use'] == 'semantic-scene'), config=dict(config or self.CONFIG))
        with mock.patch.object(ProcessProvider, 'open', fake_open):
            built = await _semantic_scene(item, dict(dict(recipe=recipe, modules=recipe['modules'], human=object()), **context))
        return built['module'], opened[0]

    async def test_a_hosted_endpoint_gets_frames_only_unless_the_owner_consents_to_the_rest(self):
        module, config = await self.build()
        self.assertEqual((module.declared, 'context' in config), ((), False))
        with self.assertRaises(ValueError):
            module.configure(dict(transcript=True))         # an agent cannot switch it on either
        with self.assertRaisesRegex(SystemExit, '--vlm-remote-context'):
            await self.build(vlm_context='transcript')
        module, config = await self.build(remote_context=True)
        self.assertEqual((module.declared, config['context'], config['allow_remote_context']), (('transcript', 'counts'), True, True))
        module, config = await self.build(remote_context=True, vlm_context='counts')
        self.assertEqual(module.declared, ('counts',))
        module, config = await self.build(config=dict(self.CONFIG, allow_remote_context=True))
        self.assertEqual(module.declared, ('transcript', 'counts'))

    async def test_a_local_endpoint_is_unchanged(self):
        local = dict(endpoint='http://127.0.0.1:9/v1', model='m', revision='r')
        for config in (local, dict(local, allow_remote=True)):
            module, sent = await self.build(config=config)
            self.assertEqual((module.declared, sent['context']), (('transcript', 'counts'), True))
            self.assertNotIn('allow_remote_context', sent)

    async def test_the_worker_itself_refuses_without_it(self):
        with mock.patch.dict('os.environ', TEST_VLM_KEY='k' * 8):
            with self.assertRaisesRegex(RuntimeError, 'PermissionError: .*allow_remote_context'):
                await ProcessProvider.open(sys.executable, 'streamingvlm.local', dict(self.CONFIG, context=True), startup_timeout=30)
            worker = await ProcessProvider.open(sys.executable, 'streamingvlm.local', dict(
                self.CONFIG, context=True, allow_remote_context=True), startup_timeout=30)     # opens no connection
            await worker.close()


class OnlyShown(unittest.TestCase):
    def load(self, *arguments):
        """The recipe ``run`` would start, as its command line says."""
        seen = {}

        async def stop(recipe, args):
            seen.update(recipe=recipe, args=args)

        with mock.patch('gamst.recipes.run_pipeline', stop):
            recipes.main(['run', 'scene', *arguments])
        recipes._sources(seen['recipe'], seen['args'], seen['recipe']['inputs'])
        return [item['id'] for item in seen['recipe']['inputs']]

    def test_a_camera_nothing_reads_is_opened_for_the_window_or_for_an_agent_only(self):
        words = ['--microphone', '--without', 'movement', '--without', 'voice', '--terms', 'a']
        self.assertEqual(self.load('--camera', '0', *words), ['camera', 'microphone'])
        self.assertEqual(self.load('--camera', '0', *words, '--headless', '--serve'), ['camera', 'microphone'])
        self.assertEqual(self.load('--input', 'camera=camera:1', *words), ['camera', 'microphone'])
        self.assertEqual(self.load(*words), ['microphone'])
        with self.assertRaisesRegex(SystemExit, 'nothing from --camera here: drop it .*--headless.*--serve'):
            self.load('--camera', '0', *words, '--headless')
        with self.assertRaisesRegex(SystemExit, 'nothing from --microphone here: drop it .*--serve'):
            self.load('--camera', '0', '--microphone', '--without', 'speech', '--exemplar', 'x')
        # an agent may add what reads them later
        self.assertEqual(self.load('--camera', '0', '--microphone', '--without', 'speech', '--without', 'movement', '--serve'),
                         ['camera', 'microphone'])


if __name__ == '__main__':
    unittest.main()
