import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from gamst import Gamst
from gamst.profile import ProfileRuntime, read_profile


def write(directory, roles):
    path = Path(directory) / 'profile.json'
    path.write_text(json.dumps({'version': 1, 'hardware': {}, 'roles': roles}))
    return path


class ProfileRuntimeTest(unittest.TestCase):
    def test_rotation_runs_and_workerless_roles_are_reported_not_dropped(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                path = write(directory, {
                    'audio.asr': {'model': 'faster_whisper.small', 'provider_profile': None, 'python': None, 'config': {}},
                    'vision.rotation': {'model': 'gamst.rotation', 'provider_profile': 'vision.rotation',
                                        'python': None, 'config': {'region_source': 'motion'}}})
                async with Gamst(consent=True) as human:
                    runtime = await ProfileRuntime.open(human, path)
                    status = dict(runtime.status)
                    registered = [plugin.name for plugin in human.plugins]
                    await runtime.close()
                    return status, registered, [plugin.name for plugin in human.plugins]
        status, registered, after = asyncio.run(run())
        self.assertEqual(status['vision.rotation']['state'], 'running')
        self.assertEqual(status['audio.asr']['state'], 'skipped')
        self.assertIn('gamst.rotation', registered)
        self.assertNotIn('gamst.rotation', after)

    def test_rejects_unversioned_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            path.write_text(json.dumps({'roles': {}}))
            with self.assertRaises(ValueError):
                read_profile(path)


if __name__ == '__main__':
    unittest.main()
