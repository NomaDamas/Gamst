import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from humansense.plugins import InputFrame
from humansense.providers.sam3 import LocalSAM3Plugin, validate_segments

SEGMENT = {'bbox': {'x': .1, 'y': .2, 'width': .3, 'height': .4}, 'score': .9,
           'polygons': [[[.1,.2],[.4,.2],[.4,.6]]]}


class GeometryTests(unittest.TestCase):
    def test_normalized_geometry_preserved(self):
        self.assertEqual(validate_segments([SEGMENT]), [SEGMENT])

    def test_invalid_geometry_and_budgets(self):
        for key, value in [('score', float('nan')), ('polygons', [[[0,0]]]),
                           ('bbox', {'x': .9, 'y': 0, 'width': .5, 'height': .2})]:
            invalid = copy.deepcopy(SEGMENT)
            invalid[key] = value
            with self.assertRaises(ValueError): validate_segments([invalid])
        with self.assertRaises(ValueError): validate_segments([SEGMENT]*17)
        invalid = copy.deepcopy(SEGMENT)
        invalid['polygons'] = [[[0,0]]*129]
        with self.assertRaises(ValueError): validate_segments([invalid])


class FakeInput:
    def __init__(self): self.requests = []
    def write(self, value): self.requests.append(json.loads(value))
    async def drain(self): pass


class FakeWorker:
    def __init__(self, reply):
        self.returncode = None
        self.stdin = FakeInput()
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(json.dumps(reply).encode()+b'\n')
        self.terminated = False
    def terminate(self): self.terminated = True; self.returncode = 0
    async def wait(self): return self.returncode


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.tmp.name)
        self.python = self.root / 'python'
        self.python.touch()
        self.plugin = LocalSAM3Plugin(self.root, self.python, 'commit123', timeout=1)
        self.frame = InputFrame({'sam3_jpeg': 'dummy'}, 'video', 123)

    async def asyncTearDown(self):
        await self.plugin.close()
        self.tmp.cleanup()

    async def test_worker_contract_and_loaded(self):
        worker = FakeWorker({'id':1, 'segments':[SEGMENT]})
        self.plugin._process = worker
        with patch.object(self.plugin, '_start_worker', AsyncMock()):
            signals = await self.plugin.process(self.frame)
        self.assertTrue(self.plugin.loaded)
        self.assertEqual(signals[0]['name'], 'scene.segments')
        self.assertEqual(signals[0]['value'][0]['prompt'], 'person')
        self.assertEqual(signals[1]['value'], 1)
        self.assertEqual(signals[0]['model_version'], 'commit123')
        self.assertEqual(worker.stdin.requests[0]['jpeg'], 'dummy')
        await self.plugin.reset()
        self.assertTrue(worker.terminated)
        self.assertFalse(self.plugin.loaded)

    async def test_wrong_sequence_closes_owned_child(self):
        worker = FakeWorker({'id':999, 'segments':[]})
        self.plugin._process = worker
        with patch.object(self.plugin, '_start_worker', AsyncMock()):
            with self.assertRaisesRegex(RuntimeError, 'sequence'):
                await self.plugin.process(self.frame)
        self.assertTrue(worker.terminated)
        self.assertIn('sequence', self.plugin.last_error)
        self.assertFalse(self.plugin.loaded)

    async def test_stale_result_discarded(self):
        worker = FakeWorker({'id':1, 'segments':[SEGMENT]})
        self.plugin._process = worker
        async def changed_generation(): self.plugin._generation += 1
        with patch.object(self.plugin, '_start_worker', changed_generation):
            self.assertEqual(await self.plugin.process(self.frame), [])
        self.assertFalse(self.plugin.loaded)

    async def test_timeout_terminates_worker(self):
        worker = FakeWorker({'id':1,'segments':[]})
        worker.stdout = asyncio.StreamReader()
        self.plugin._process = worker
        with patch.object(self.plugin, '_start_worker', AsyncMock()):
            with self.assertRaises(asyncio.TimeoutError): await self.plugin.process(self.frame)
        self.assertTrue(worker.terminated)

    async def test_closed_plugin_emits_nothing(self):
        await self.plugin.close()
        self.assertEqual(await self.plugin.process(self.frame), [])


if __name__ == '__main__': unittest.main()
