import asyncio
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gamst import recipes
from gamst.plugins import InputFrame

HAS_WS = importlib.util.find_spec('websockets') is not None
HAS_CV = all(importlib.util.find_spec(name) for name in ('cv2', 'numpy'))


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


@unittest.skipUnless(HAS_WS and HAS_CV, 'needs websockets and opencv')
class ServeTest(unittest.TestCase):
    def test_an_agent_reads_and_switches_a_running_recipe_over_mcp(self):
        import cv2
        from tests.test_polar_rotation import fan_frames
        port, answers = free_port(), {}
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                'GAMST_WS_TOKEN_FILE': str(Path(folder) / 'token')}):
            path = Path(folder) / 'fan.avi'
            frames = fan_frames(12)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, frames[0].shape[1::-1])
            for frame in frames:
                writer.write(frame)
            writer.release()

            def ask():   # the MCP stdio bridge an agent starts, in its own process
                def call(identifier, name, arguments=None):
                    return json.dumps(dict(jsonrpc='2.0', id=identifier, method='tools/call', params=dict(
                        name=name, arguments=arguments or {})))
                process = subprocess.Popen([sys.executable, '-m', 'gamst.mcp_live', '--url', f'ws://127.0.0.1:{port}'],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                           env=dict(os.environ, PYTHONPATH=str(Path(recipes.__file__).parents[2])))
                process.stdin.write('\n'.join([
                    call(1, 'get_pipeline'), call(2, 'get_timeline', dict(window='1m')),
                    call(3, 'count_events', dict(capability_id='vision.rotation')),
                    call(4, 'configure_pipeline_module', dict(name='rotation', enabled=False)),
                    call(5, 'start_capture', dict(consent=True))]) + '\n')
                process.stdin.flush()
                try:
                    for line in process.stdout:
                        message = json.loads(line)
                        if message.get('id') in range(1, 6):
                            answers[message['id']] = message
                        if len(answers) == 5:
                            break
                finally:
                    process.kill()
                    process.communicate()

            state = {}

            async def update(self, input_id, frame, records):
                state['n'] = state.get('n', 0) + 1
                if state['n'] == 20:
                    await asyncio.wait_for(asyncio.to_thread(ask), 30)
                return True

            with patch('gamst.views.RotationView.update', update):
                report = recipes.main(['run', 'rotation', '--video', str(path), '--capture-fps', '240', '--roi',
                                       '.02,.02,.96,.96', '--symmetry', '3', '--max-rev-s', '20', '--headless',
                                       '--serve', str(port)])
        text = lambda n: json.dumps(answers[n])
        self.assertIn('rotation', text(1))
        self.assertIn('complete_until_ms', text(2))
        self.assertIn('count', text(3))
        self.assertNotIn('"isError": true', text(4))
        self.assertTrue('Unknown tool' in text(5) or 'error' in text(5))   # it cannot open another device
        self.assertFalse(report['pipeline']['modules']['rotation']['enabled'])
        self.assertFalse(report['complete'])                               # and the run says it was switched off
        with socket.socket() as probe:
            self.assertNotEqual(probe.connect_ex(('127.0.0.1', port)), 0)   # closed with the run

    def test_a_connection_without_the_token_is_refused(self):
        async def scenario():
            from gamst import Gamst
            from websockets.asyncio.client import connect
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                    'GAMST_WS_TOKEN_FILE': str(Path(folder) / 'token')}):
                async with Gamst(consent=True) as human:
                    live = recipes._LiveMCP(human, free_port())
                    await live.start()
                    try:
                        with self.assertRaises(Exception) as refused:
                            async with connect(f'ws://127.0.0.1:{live.port}'):
                                pass
                        self.assertIn('401', str(refused.exception))
                    finally:
                        await live.close()
        asyncio.run(scenario())


if __name__ == '__main__':
    unittest.main()
