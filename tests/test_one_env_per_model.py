"""One environment per model, one weight root. Fakes only: nothing is installed, downloaded or run."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gamst import Gamst, recipes
from gamst.crew import registry, setup
from gamst.private_payload import PrivatePoses
from gamst.providers.pose_exemplar import PoseChannels, pose_channels

try:
    import numpy as np
except ImportError:
    np = None
HAS_CV = importlib.util.find_spec('cv2') is not None and np is not None
HOST = dict(tier='apple-mps', accelerator='mps', os='darwin', arch='arm64', python=platform.python_version(),
            ram_bytes=64 << 30, vram_bytes=0, disk_free_bytes=512 << 30)
DETECT, SEGMENT = ('vision.detect', 'yoloe.26s'), ('vision.video_segmentation', 'efficienttam.ti_512')


class Temporary(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.tmp = Path(self.directory.name).resolve()
        self.addCleanup(self.directory.cleanup)


class Environments(Temporary):
    def plan(self, **options):
        return setup.make_plan(roles=dict([DETECT, SEGMENT, ('vision.rotation', 'gamst.rotation'),
                                           ('input.video', 'opencv')]), dependencies=False, hardware=HOST, **options)

    def test_models_get_environments_and_the_pipeline_process_keeps_its_own_imports(self):
        for plan in (self.plan(), setup.make_plan('rotation', hardware=HOST)):
            entries = {entry['environment']: entry for entry in plan['install']}
            self.assertIn('vision.rotation', entries['current']['roles'])
            self.assertNotIn('vision.detect', entries['current']['roles'])
            self.assertEqual(entries['current']['python'], platform.python_version())
        self.assertEqual(set(entries), {'current', 'rfdetr-nano'})
        explicit = self.plan(isolation='current')
        self.assertEqual([entry['environment'] for entry in explicit['install']], ['current'])

    def test_the_registry_not_the_plan_decides_where_packages_go(self):
        moved = self.plan()
        model = next(entry for entry in moved['install'] if entry['environment'] != 'current')
        next(entry for entry in moved['install'] if entry['environment'] == 'current')['packages'] += model['packages']
        with self.assertRaisesRegex(setup.SetupError, 'environments differ'):
            setup._validate_plan(moved)
        for key, value in (('environment', 'elsewhere'), ('in_process', 'os:system'), ('python', '3.0'),
                           ('source', {'url': 'https://example.invalid/x', 'revision': '0' * 40})):
            tampered = self.plan()
            tampered['roles']['vision.detect'][key] = value
            with self.assertRaisesRegex(setup.SetupError, f'planned {key} differs'):
                setup._validate_plan(tampered)

    def test_apply_uses_each_environment_and_records_its_python(self):
        commands = []

        def run(args):
            commands.append([str(arg) for arg in args])
            if '-m' in commands[-1] and 'venv' in commands[-1]:
                python = Path(commands[-1][-1]) / 'bin' / 'python'
                python.parent.mkdir(parents=True)
                python.write_text('')
        with patch.object(setup, '_run', run), patch.object(setup, '_install_source', return_value='/source'), \
                patch.object(setup, 'download_weight', lambda weight, path: dict(
                    name=weight['name'], path=str(path), sha256=weight['sha256'], bytes=1, cached=False)):
            result = setup.apply_plan(self.plan(), prefix=self.tmp / 'prefix', profile_path=self.tmp / 'p.json',
                                      models_dir=self.tmp / 'models')
        shared = str(self.tmp / 'prefix' / 'venvs' / 'yoloe-efficienttam' / 'bin' / 'python')
        profile = json.loads((self.tmp / 'p.json').read_text())['roles']
        self.assertEqual(profile['vision.detect']['python'], shared)
        self.assertEqual(profile['vision.video_segmentation']['python'], shared)  # they load in one process
        self.assertEqual(profile['vision.rotation']['python'], sys.executable)
        self.assertEqual(profile['input.video']['python'], sys.executable)
        self.assertEqual(Path(profile['vision.detect']['config']['checkpoint_path']),
                         self.tmp / 'models' / 'yoloe.26s' / 'yoloe-26s-seg.pt')
        self.assertEqual(len([c for c in commands if 'venv' in c]), 1)
        installs = {command[0]: command for command in commands if 'pip' in command}
        self.assertEqual(set(installs), {sys.executable, shared})
        self.assertNotIn('-e', installs[sys.executable])
        self.assertFalse(any('torch' in item for item in installs[sys.executable]))
        self.assertEqual(result['profile'], str(self.tmp / 'p.json'))

    def test_no_environment_holds_two_cv2_distributions(self):
        """mediapipe==0.10.35 depends on opencv-contrib-python, ultralytics on opencv-python."""
        environments = {}
        for role in registry.ROLES.values():
            for model, candidate in role.candidates.items():
                name = setup._environment(role.id, model, 'venv')
                environments.setdefault(name, []).extend(candidate.packages)
        self.assertGreater(len(environments), 10)
        for name, packages in environments.items():
            self.assertLessEqual(len(registry.cv2_distributions(packages)), 1, name)
        self.assertEqual(registry.cv2_distributions(['mediapipe==0.10.35']), {'opencv-contrib-python'})
        self.assertEqual(registry.cv2_distributions(['ultralytics==8.4.212', 'opencv-python>=4.8,<5']), {'opencv-python'})
        merged = recipes.plan(recipes.load('person-action-count', exclude=['target_person']), hardware=HOST, isolation='current')
        self.assertTrue(any('overwrite each other' in warning for warning in merged['warnings']))
        apart = recipes.plan(recipes.load('person-action-count'), hardware=HOST)
        self.assertFalse(any('overwrite each other' in warning for warning in apart['warnings']))

    def test_a_shared_environment_has_no_conflicting_pins(self):
        pins = {}
        for role, model in (DETECT, SEGMENT):
            for package in registry.candidate(role, model).packages:
                name, _, version = package.partition('==')
                if version:
                    self.assertEqual(pins.setdefault(name, version), version)
        self.assertEqual(pins['torch'], '2.8.0')


class WorkerGamst(Temporary):
    def test_a_checkout_is_installed_in_place_with_its_commit(self):
        found = setup.gamst_requirement()
        self.assertEqual(found['kind'], 'checkout')
        self.assertEqual(Path(found['path']), Path(recipes.__file__).resolve().parents[2])
        self.assertTrue(found['commit'] is None or len(found['commit']) == 40)
        entry = dict(environment='silero-vad', python='3.10', gamst=found, packages=['torch>=2.2,<3'])
        commands = []
        python = self.tmp / 'venvs' / 'silero-vad' / 'bin' / 'python'
        python.parent.mkdir(parents=True)
        python.write_text('')
        with patch.object(setup, '_run', commands.append):
            setup._install_environment(entry, self.tmp)
            self.assertEqual(setup._install_environment(entry, self.tmp)[1], 'cached')
        [command] = commands
        self.assertEqual([str(item) for item in command][-3:], ['-e', found['path'], 'torch>=2.2,<3'])

    def test_an_installed_release_is_pinned(self):
        with patch.object(setup, '__file__', str(self.tmp / 'site-packages' / 'gamst' / 'crew' / 'setup.py')):
            found = setup.gamst_requirement()
        self.assertEqual(found, {'kind': 'release', 'requirement': f'gamst-sdk=={setup.__version__}'})

    def test_a_git_install_is_pinned_to_its_commit(self):
        origin = {'url': 'https://github.com/NomaDamas/Gamst', 'vcs_info': {'vcs': 'git', 'commit_id': 'a' * 40}}
        fake = SimpleNamespace(version=setup.__version__, read_text=lambda name: json.dumps(origin))
        with patch.object(setup, '__file__', str(self.tmp / 'lib' / 'gamst' / 'crew' / 'setup.py')), \
                patch('importlib.metadata.distribution', return_value=fake):
            self.assertEqual(setup.gamst_requirement()['requirement'],
                             f"gamst-sdk @ git+https://github.com/NomaDamas/Gamst@{'a' * 40}")

    def test_a_plan_for_another_checkout_or_commit_is_refused(self):
        plan = setup.make_plan('speech', ['audio.asr=faster_whisper.small'], hardware=HOST)
        entry = next(item for item in plan['install'] if item['environment'] != 'current')
        entry['gamst'] = dict(entry['gamst'], path='/somewhere/else')
        with self.assertRaisesRegex(setup.SetupError, 'another GAMST'):
            setup.apply_plan(plan, prefix='/unused', profile_path='/unused/p.json')


class WeightRoot(Temporary):
    def write(self, path, data=b'x'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_every_entry_point_uses_one_root(self):
        from gamst.demos import rotation
        with patch.dict(os.environ, GAMST_MODELS=str(self.tmp)):
            self.assertEqual(setup.models_root(), self.tmp)
            self.assertEqual(recipes.plan(recipes.load('rotation'), hardware=HOST)['models_dir'], str(self.tmp))
            self.assertEqual(rotation.detector_directory(), self.tmp / 'yoloe.26s')
            checkpoint, digest = rotation.refiner_checkpoint()
            self.assertEqual(checkpoint, self.tmp / 'efficienttam.ti_512' / 'efficienttam_ti_512x512.pt')
            self.assertEqual(len(digest), 64)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('GAMST_MODELS', None)
            self.assertEqual(setup.models_root(), Path.home() / '.cache' / 'gamst' / 'models')

    def test_an_existing_flat_directory_is_still_found(self):
        from gamst.demos import rotation
        for name in (*rotation.MODEL_ASSETS, *rotation.REFINE_ASSETS):
            self.write(self.tmp / name)
        self.assertEqual(rotation.detector_directory(str(self.tmp)), self.tmp)
        self.assertEqual(rotation.refiner_checkpoint(self.tmp)[0].parent, self.tmp)
        config = rotation.detector_config(str(self.tmp), ['coin'], 'cpu')
        self.assertEqual(config['checkpoint_path'], str(self.tmp / 'yoloe-26s-seg.pt'))
        self.assertEqual(config['runtime_root'], str(self.tmp.parent / 'runtime'))
        for name in rotation.MODEL_ASSETS:  # its own directory wins once it is complete
            self.write(self.tmp / 'yoloe.26s' / name)
        self.assertEqual(rotation.detector_directory(self.tmp), self.tmp / 'yoloe.26s')

    def test_same_named_files_of_two_models_never_collide(self):
        weights = {}
        for model, data in (('a.one', b'first'), ('b.two', b'second')):
            source = self.write(self.tmp / 'source' / model, data)
            weights[model] = dict(name='config.json', url=source.as_uri(), sha256=hashlib.sha256(data).hexdigest())
        flat = self.write(self.tmp / 'models' / 'config.json', b'first')
        for model, weight in weights.items():
            directory = setup.weights_directory(model, [weight], self.tmp / 'models')
            setup.download_weight(weight, directory / weight['name'])
        self.assertEqual(setup.weights_directory('a.one', [weights['a.one']], self.tmp / 'models'), flat.parent)
        self.assertEqual((self.tmp / 'models' / 'b.two' / 'config.json').read_bytes(), b'second')
        self.assertEqual(flat.read_bytes(), b'first')  # untouched


    def test_a_bundle_gets_its_own_folder_even_next_to_flat_files(self):
        """FasterWhisperPlugin hashes its whole directory: nothing else may be in it."""
        candidate = registry.candidate('audio.asr', 'faster_whisper.small')
        self.assertTrue(candidate.bundle)
        weights = [dict(name=item.name, url='unused', sha256=hashlib.sha256(item.name.encode()).hexdigest())
                   for item in candidate.weights]
        for weight in weights:  # a flat directory that even holds the bundle's verified files
            self.write(self.tmp / 'models' / weight['name'], weight['name'].encode())
        self.write(self.tmp / 'models' / 'model.safetensors')
        directory = setup.weights_directory('faster_whisper.small', weights, self.tmp / 'models', candidate.bundle)
        self.assertEqual(directory, self.tmp / 'models' / 'faster_whisper.small')
        fetched = [dict(name=item.name, path=str(directory / item.name)) for item in candidate.weights]
        config = setup._role_config(dict(model='faster_whisper.small', config={}), fetched, None)
        self.assertEqual(config['model_path'], str(directory))
        from gamst.recipes.perception_parts import _whisper_words
        seen = {}

        async def opened(client, entry):
            seen.update(entry[0])
            return SimpleNamespace(spec=None)
        with patch('gamst.recipes.perception_parts._open', opened):
            asyncio.run(_whisper_words({}, dict(installed=lambda item: dict(config=config, python=sys.executable))))
        self.assertEqual(seen['model_path'], str(directory))


class CoinFlipEnvironment(Temporary):
    def profile(self, detect, segment):
        path = self.tmp / 'profile.json'
        path.write_text(json.dumps({'version': 1, 'hardware': {}, 'roles': {
            DETECT[0]: {'model': DETECT[1], 'provider_profile': 'yoloe.local', 'python': detect,
                        'config': {'checkpoint_path': '/m/yoloe.26s/yoloe-26s-seg.pt'}},
            SEGMENT[0]: {'model': SEGMENT[1], 'provider_profile': 'efficienttam.tiny512', 'python': segment,
                         'config': {'source_dir': '/src', 'checkpoint_path': '/m/efficienttam.ti_512/e.pt'}}}}))
        return str(path)

    def counter(self, profile, **options):
        recipe = recipes.load('coin-flip')
        count = next(item for item in recipe['modules'] if item['use'] == 'flip')
        context = dict(human=None, installed=lambda item: recipes.installed(recipe, item, profile), device='cpu',
                       stages={item['name']: item for item in recipe['modules'] if item.get('in')}, fast=False,
                       **options)
        return asyncio.run(recipes.PARTS['flip'].build(count, context))['module']

    def test_the_refine_process_runs_in_the_environment_both_models_share(self):
        python = self.tmp / 'python'
        python.write_text('')
        counter = self.counter(self.profile(str(python), str(python)))
        self.assertEqual(counter.python, str(python))
        args = counter.refine_args
        self.assertEqual(args[args.index('--worker-python') + 1], str(python))
        self.assertEqual(args[args.index('--models') + 1], '/m/yoloe.26s')
        self.assertEqual(args[args.index('--efficienttam-checkpoint') + 1], '/m/efficienttam.ti_512/e.pt')
        (self.tmp / 'take').mkdir()
        with patch('gamst.takes.subprocess.Popen') as process:
            counter._launch(self.tmp / 'take')
            self.assertEqual(process.call_args.args[0][:5], [str(python), '-m', 'gamst.takes', 'refine', str(self.tmp / 'take')])
            counter.fast, counter.log_dir = True, self.tmp
            counter.open()
            self.assertEqual(process.call_args.args[0][:4], [str(python), '-m', 'gamst.takes', 'refine-server'])

    def test_separate_or_missing_environments_ask_for_setup(self):
        with self.assertRaisesRegex(SystemExit, 'different environments'):
            self.counter(self.profile('/a/python', '/b/python'))
        with self.assertRaisesRegex(SystemExit, 'does not exist'):
            self.counter(self.profile('/a/python', '/a/python'))

    def test_rotation_demo_keeps_an_explicit_worker_python_for_the_detector_only(self):
        counter = self.counter(None, efficienttam_source='/src', worker_python='/x/python')
        self.assertEqual(counter.python, sys.executable)
        self.assertEqual(counter.refine_args[counter.refine_args.index('--worker-python') + 1], '/x/python')
        with patch('gamst.profile.role_python', return_value='/env/python'):
            self.assertEqual(self.counter(None, efficienttam_source='/src').python, '/env/python')
        with patch('gamst.profile.role_python', side_effect=['/a', '/b']):
            self.assertEqual(self.counter(None, efficienttam_source='/src').python, sys.executable)


def body(width=1., height=1.):
    """33 keypoints of one pose given in pixels-like units, normalized per axis by width and height."""
    pixel = {11: (40, 30), 12: (60, 30), 13: (40, 50), 14: (75, 45), 15: (60, 50), 16: (75, 25),
             23: (42, 70), 24: (58, 70), 25: (42, 90), 26: (62, 88), 27: (42, 110), 28: (66, 106),
             0: (50, 24), 2: (70, 8), 5: (30, 8), 9: (62, 40), 10: (38, 40)}  # a face large enough to align
    return [(pixel.get(i, (50, 20))[0] / width, pixel.get(i, (50, 20))[1] / height) for i in range(33)]


def landmarks(width, height):
    return [dict(x=x, y=y, visibility=.9) for x, y in body(width, height)]


class Aspect(unittest.TestCase):
    def test_the_same_body_gives_the_same_channels_whatever_the_image_shape(self):
        truth, _ = pose_channels(landmarks(1, 1))
        self.assertAlmostEqual(truth[0], .5)  # a right-angle left elbow
        distorted, _ = pose_channels(landmarks(1920, 1080))
        self.assertGreater(max(abs(a - b) for a, b in zip(truth, distorted)), .05)  # the defect
        module = PoseChannels()
        for width, height in ((1920, 1080), (1080, 1920), (500, 500)):
            size = dict(image_width=width, image_height=height)
            plugin = module.process(SimpleNamespace(data=[dict(landmarks=landmarks(width, height), **size)]))
            joints = module.process(SimpleNamespace(data=PrivatePoses([dict(landmarks=landmarks(width, height), **size)])))
            self.assertEqual(plugin['channels'], joints['channels'])
            for got, expected in zip(plugin['channels'], truth):
                self.assertAlmostEqual(got, expected, places=4)
            self.assertAlmostEqual(plugin['image_aspect'], width / height, places=4)

    def test_an_unreported_image_size_is_visible_in_the_record(self):
        value = PoseChannels().process(SimpleNamespace(data=[dict(landmarks=landmarks(1, 1))]))
        self.assertNotIn('image_aspect', value)
        self.assertTrue(PoseChannels().spec.version.startswith('2.'))

    def test_exemplars_enrolled_before_the_fix_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'exemplar.json'
            saved = dict(kind='exemplar', value=dict(channels=[0.] * 13, mask=[True] * 13))
            reference, args = dict(kind='exemplar', name='exemplar'), argparse.Namespace(exemplar=str(path))
            for old in ({}, {'channels': 'vision.pose_channels@1.mediapipe_pose'}):
                path.write_text(json.dumps(dict(saved, **old)))
                with self.assertRaisesRegex(SystemExit, 'enroll it again'):
                    recipes._reference_value(reference, args)
            for current in ('vision.pose_channels@2.mediapipe_pose', 'audio.voice_channels@1.x'):
                path.write_text(json.dumps(dict(saved, channels=current)))
                self.assertEqual(recipes._reference_value(reference, args), (saved['value'], None))


class VideoDetector:
    """Stands in for MediaPipe's VIDEO PoseLandmarker; records the relative timestamps it was given."""
    made = []

    def __init__(self):
        self.stamps, self.closed = [], False
        VideoDetector.made.append(self)

    def detect_for_video(self, image, stamp):
        self.stamps.append(stamp)
        height, width = image.shape[:2]
        points = [SimpleNamespace(x=x, y=y, z=0., visibility=.9, presence=.8) for x, y in body(width, height)]
        return SimpleNamespace(pose_landmarks=[points])

    def close(self):
        self.closed = True


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class PoseWorker(Temporary):
    """The mediapipe-pose part with the worker process replaced by its own provider on a fake detector."""

    async def open(self, python, profile, config, **_):
        from gamst.private_payload import CONTRACTS, Private
        from gamst.providers.landmarker_video import MediaPipeVideoContext
        self.opened = (python, profile, config)
        api = SimpleNamespace(Image=lambda **kw: kw['data'], ImageFormat=SimpleNamespace(SRGB=1))
        provider = MediaPipeVideoContext('body', VideoDetector, api, sha256=config['sha256'],
                                         max_entities=config['max_entities'])

        class Worker:
            spec, private = provider.spec, True

            async def query(self, frame, request=None):
                public, joints = provider.query(frame, request)
                CONTRACTS[profile].check_shape(list(joints.shape), None)
                return public, Private('joints', joints)

            async def reset(self):
                provider.reset()

            async def close(self):
                provider.close()
        return Worker()

    def run_frames(self, frames, *, between=None, target=False):
        VideoDetector.made = []
        recipe = recipes.load('person-action-count', exclude=[] if target else ['target_person'])
        entry = dict(model='mediapipe.pose_full', provider_profile='mediapipe.body.video', python='/env/python',
                     config=dict(path='/m/pose.task', sha256='1' * 64))

        class Face:
            model_id = 'face'

            def embed(self, crop):
                return [1., 0., 0.]

        async def embedders(client, config):
            return Face()

        async def scenario():
            from gamst.mcp import MCPServer
            from gamst.pipeline import Pipeline
            from gamst.schema import timestamp
            async with Gamst(consent=True) as human:
                pipeline = Pipeline(human, 'poses')
                pipeline.add_input('camera', modality='video')
                pipeline.add_reference('person', 'person', dict(face_model='face', face=[[1., 0., 0.]], body_model=None, body=[]),
                                       consent=dict(statement='owner'))
                context = dict(installed=lambda item: entry)
                for item in recipe['modules'][:3 if target else 2]:
                    built = await recipes.PARTS[item['use']].build(item, context)
                    pipeline.add_module(item['name'], built['module'], consumes=item['consumes'], spec=built.get('spec'),
                                        emits=tuple(item.get('emits', ())))
                start, records = timestamp() - 2000, []
                for index, frame in enumerate(frames):
                    if between is not None:
                        await between(pipeline, index)
                    records += await pipeline.push('camera', frame, timestamp_ms=start + 33 * index)
                server = MCPServer(human)
                seen = [await server.call_tool(name, arguments) for name, arguments in (
                    ('get_state', {}), ('get_context', {}), ('get_pipeline', {}), ('get_observations', {}),
                    ('inspect_interval', dict(start_ms=start - 1000, end_ms=start + 9000)))]
                everything = json.dumps([records, seen, human.store.query('local-user')], default=str)
                await server.close()
                await pipeline.close()
                return records, everything
        with patch('gamst.crew.modules.open_private_worker', self.open), \
                patch('gamst.recipes.perception_parts._open', embedders):
            return asyncio.run(scenario())

    def test_video_tracking_survives_the_pipeline_and_feeds_pose_channels(self):
        records, everything = self.run_frames([np.zeros((720, 1280, 3), np.uint8)] * 5)
        self.assertEqual(self.opened[:2], ('/env/python', 'mediapipe.body.video'))
        self.assertNotIn('landmarks', self.opened[2])  # private, not a configured output
        [detector] = VideoDetector.made  # one native tracker for the whole run: never reset
        self.assertEqual(detector.stamps, [0, 33, 66, 99, 132])
        self.assertTrue(detector.closed)
        names = [record['observation']['name'] for record in records]
        self.assertEqual(names.count('scene.poses'), 5)
        channels = [record['observation']['value'] for record in records
                    if (record['observation'].get('capability') or {}).get('id') == 'vision.pose_channels']
        self.assertEqual(len(channels), 5)
        truth, _ = pose_channels(landmarks(1, 1))
        for got, expected in zip(channels[0]['channels'], truth):
            self.assertAlmostEqual(got, expected, places=3)
        poses = next(record['observation']['value'] for record in records if record['observation']['name'] == 'scene.poses')
        self.assertEqual((poses[0]['image_width'], poses[0]['image_height']), (1280, 720))
        self.assertIsNotNone(poses[0]['track_id'])
        # no joint position in the Store or any MCP answer (33 x 4 numbers per pose, under any name)
        for word in ('keypoints', 'landmarks'):
            self.assertNotIn(word, everything)
        x, y = body(1280, 720)[12]  # not an extreme point: box edges are public
        self.assertNotIn(str(np.float32(x).item())[:9], everything)
        self.assertNotIn(str(np.float32(y).item())[:9], everything)

    def test_the_default_recipe_counts_the_enrolled_person_from_the_frames_own_size(self):
        """camera -> pose -> target_person -> pose_channels, the recipe's own links."""
        records, everything = self.run_frames([np.zeros((720, 1280, 3), np.uint8)] * 4, target=True)
        [detector] = VideoDetector.made
        self.assertEqual(detector.stamps, [0, 33, 66, 99])  # tracking is on the full frames, never reset
        target = [r['observation']['value'] for r in records if r['observation']['name'] == 'vision.target_person']
        self.assertEqual([value['state'] for value in target], ['face'] * 4)
        channels = [r['observation']['value'] for r in records
                    if (r['observation'].get('capability') or {}).get('id') == 'vision.pose_channels']
        self.assertEqual(len(channels), 4)
        truth, _ = pose_channels(landmarks(1, 1))
        for got, expected in zip(channels[-1]['channels'], truth):
            self.assertAlmostEqual(got, expected, places=3)
        self.assertNotIn('landmarks', everything)

    def test_poses_found_in_a_changing_crop_use_each_crops_own_size(self):
        """A derived crop changes size; the same body must give the same channels in every crop."""
        records, _ = self.run_frames([np.zeros((h, w, 3), np.uint8) for h, w in ((400, 200), (300, 260), (200, 400))])
        channels = [r['observation']['value'] for r in records
                    if (r['observation'].get('capability') or {}).get('id') == 'vision.pose_channels']
        self.assertEqual([value['image_aspect'] for value in channels], [.5, round(260 / 300, 5), 2.])
        for value in channels[1:]:
            for got, expected in zip(value['channels'], channels[0]['channels']):
                self.assertAlmostEqual(got, expected, places=3)
        self.assertEqual(len(VideoDetector.made), 3)  # a size change is a continuity boundary of VIDEO mode

    def test_a_large_frame_is_scaled_for_the_worker_and_keeps_its_proportions(self):
        records, _ = self.run_frames([np.zeros((2160, 3840, 3), np.uint8)])
        channels = next(record['observation']['value'] for record in records
                        if (record['observation'].get('capability') or {}).get('id') == 'vision.pose_channels')
        truth, _ = pose_channels(landmarks(1, 1))
        for got, expected in zip(channels['channels'], truth):
            self.assertAlmostEqual(got, expected, places=2)

    def test_switching_the_module_off_and_on_starts_a_new_track_segment(self):
        async def between(pipeline, index):
            if index == 2:
                pipeline.disable('pose')
                pipeline.enable('pose')
        self.run_frames([np.zeros((100, 200, 3), np.uint8)] * 4, between=between)
        self.assertEqual([item.stamps for item in VideoDetector.made], [[0, 33], [0, 33]])

    def test_enrollment_records_the_channel_version(self):
        class Clip:
            modality = 'video'

            async def frames(self):
                from gamst.plugins import InputFrame
                from gamst.schema import timestamp
                start = timestamp() - 3000
                for index in range(6):
                    yield InputFrame(np.zeros((100, 200, 3), np.uint8), 'video', start + 33 * index)
        profile = self.tmp / 'profile.json'
        profile.write_text(json.dumps({'version': 1, 'hardware': {}, 'roles': {'vision.pose': dict(
            model='mediapipe.pose_full', provider_profile='mediapipe.body.video', python='/env/python',
            config=dict(path='/m/pose.task', sha256='1' * 64))}}))
        out = self.tmp / 'exemplar.json'
        with patch('gamst.crew.modules.open_private_worker', self.open), patch('gamst.inputs.open_source', lambda *a, **k: Clip()):
            recipes.main(['enroll', 'person-action-count', '--video', 'x.mp4', '--out', str(out), '--profile', str(profile)])
        saved = json.loads(out.read_text())
        self.assertTrue(saved['channels'].startswith('vision.pose_channels@2.'))
        self.assertEqual(recipes._reference_value(dict(kind='exemplar', name='exemplar'),
                                                  argparse.Namespace(exemplar=str(out))), (saved['value'], None))
        self.assertTrue(all(math.isfinite(value) for value in saved['value']['channels']))


if __name__ == '__main__':
    unittest.main()
