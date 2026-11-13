import io
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from humansense.plugins import InputFrame
from humansense.realtime import LocalScenePlugin, QueuedPlugin, scene_status
from humansense.vlm_metrics import VLMCallMetrics, reported_usage


class Metrics(unittest.TestCase):
    def test_missing_partial_invalid_and_cache_counts_are_not_invented(self):
        self.assertEqual(reported_usage(None), (None, 'missing'))
        values, state = reported_usage({'prompt_tokens': 100, 'completion_tokens': 5})
        self.assertEqual(state, 'partial')
        self.assertIsNone(values['cached_prompt_tokens'])
        self.assertIsNone(values['cache_write_tokens'])
        for payload in ([], {'prompt_tokens': True}, {'completion_tokens': -1},
                        {'prompt_tokens_details': []}, {'prompt_tokens': 3,
                         'prompt_tokens_details': {'cached_tokens': 4}}):
            self.assertEqual(reported_usage(payload), (None, 'invalid'))
        values, state = reported_usage({'prompt_tokens': 100, 'completion_tokens': 5,
            'prompt_tokens_details': {'cached_tokens': 30, 'cache_write_tokens': 20}})
        self.assertEqual(state, 'reported')
        self.assertEqual(values['prompt_tokens'], 100)

    def test_concurrent_lifetime_counts_and_bounded_copied_snapshots(self):
        metrics = VLMCallMetrics()
        def call(_):
            token = metrics.start()
            metrics.finish(token, success=True, usage={'prompt_tokens': 2})
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(call, range(200)))
        result = metrics.snapshot()
        self.assertEqual((result['completed'], result['in_flight']), (200, 0))
        self.assertEqual(result['reported_token_totals']['prompt_tokens'], 400)
        self.assertEqual(result['token_observation_counts']['completion_tokens'], 0)
        self.assertEqual((len(result['recent_calls']), result['omitted_calls']), (16, 184))
        result['recent_calls'][0]['usage']['prompt_tokens'] = 999
        self.assertEqual(metrics.snapshot()['recent_calls'][0]['usage']['prompt_tokens'], 2)

    def test_scene_call_integration_preserves_outputs_and_redacts_metrics(self):
        plugin = LocalScenePlugin('http://127.0.0.1:8080/v1/chat/completions', 'fixture')
        response = {'choices': [{'message': {'content': 'PRIVATE DESCRIPTION'}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}
        plugin.transport._opener = lambda *a, **k: io.BytesIO(json.dumps(response).encode())
        frame = InputFrame({'frames': [{'timestamp_ms': 0, 'jpeg': 'PRIVATE IMAGE'}],
                            'settings': plugin.settings()}, 'video', 0)
        output = plugin._describe(frame)
        self.assertEqual(output[0]['value'], 'PRIVATE DESCRIPTION')
        human = SimpleNamespace(plugins=[QueuedPlugin(None, plugin)])
        status = scene_status(human)
        self.assertEqual(status['call_metrics']['completed'], 1)
        self.assertNotIn('PRIVATE', json.dumps(status))
        self.assertNotIn('call_metrics', plugin.settings())
        self.assertIsNone(status['call_metrics']['task_answer_ready_latency_ms'])
        response['choices'] = []
        with self.assertRaises(IndexError):
            plugin._describe(frame)
        self.assertEqual(plugin.call_metrics.snapshot()['errors'], 1)
        # Usage returned with a malformed answer still records reported work.
        self.assertEqual(plugin.call_metrics.snapshot()['reported_token_totals']['prompt_tokens'], 200)
        plugin.reset()
        self.assertEqual(plugin.call_metrics.snapshot()['completed'], 2)

    def test_actual_loopback_http_usage_and_preflight_failure(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.server.request_model = body['model']
                payload = json.dumps({'choices': [{'message': {'content': 'fixture scene'}}],
                    'usage': {'prompt_tokens': 40, 'completion_tokens': 5,
                        'prompt_tokens_details': {'cached_tokens': 10, 'cache_write_tokens': 0}}}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            plugin = LocalScenePlugin(f'http://127.0.0.1:{server.server_port}/v1/chat/completions', 'fixture')
            plugin._describe(InputFrame({'frames': [{'timestamp_ms': 0, 'jpeg': 'eA=='}],
                'settings': {}}, 'video', 0))
            self.assertEqual(server.request_model, 'fixture')
            with self.assertRaises(KeyError):
                plugin._describe(InputFrame({}, 'video', 0))
            status = plugin.call_metrics.snapshot()
            self.assertEqual((status['started'], status['http_attempts']), (2, 1))
            self.assertEqual(status['reported_token_totals']['cached_prompt_tokens'], 10)
            self.assertFalse(status['recent_calls'][-1]['http_request_attempted'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_running_http_request_is_visible_without_recording_body_or_error(self):
        plugin = LocalScenePlugin('http://127.0.0.1:8080/v1/chat/completions', 'fixture')
        entered, release = threading.Event(), threading.Event()
        def fail(*args, **kwargs):
            entered.set()
            release.wait(5)
            raise RuntimeError('PRIVATE RESPONSE ERROR')
        plugin.transport._opener = fail
        frame = InputFrame({'frames': [{'timestamp_ms': 0, 'jpeg': 'x'}], 'settings': {}}, 'video', 0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(plugin._describe, frame)
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(plugin.call_metrics.snapshot()['in_flight'], 1)
            finally:
                release.set()
            with self.assertRaises(RuntimeError):
                future.result()
        status = plugin.call_metrics.snapshot()
        self.assertEqual((status['in_flight'], status['errors']), (0, 1))
        self.assertNotIn('PRIVATE', json.dumps(status))
        self.assertEqual(status['usage_states']['missing'], 1)
