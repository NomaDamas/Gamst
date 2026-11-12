"""Loopback-only browser demo. Raw browser media is processed, never persisted."""
import argparse
import asyncio
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Dashboard:
    def __init__(self, human, loop):
        self.human, self.loop = human, loop
        self.token = secrets.token_urlsafe(32)
        self.errors = []
        self.capture = False
        self.companion = None
        self._lock = asyncio.Lock()
        self._vlm_test_lock = asyncio.Lock()
        from .media import media_router
        self.router = media_router(human)

    def asr_plugin(self):
        from .plugins import FasterWhisperPlugin
        for worker in self.human.plugins:
            plugin = getattr(worker, "plugin", worker)
            if isinstance(plugin, FasterWhisperPlugin):
                return plugin
        return None

    def asr_settings(self):
        plugin = self.asr_plugin()
        return {"available": plugin is not None,
                "language": (plugin.language or "auto") if plugin else "auto"}

    async def dispatch(self, body):
        # Model calls must not hold a transport-wide lock: stop/delete need to
        # invalidate in-flight work immediately, not after a provider timeout.
        return await self._dispatch(body)

    async def _dispatch(self, body):
        from .realtime import scene_status, configure_scene
        op = body.get("op")
        if op == "capabilities":
            from .mcp import tool_definitions
            from .extensions import describe_plugin
            return {"tools": tool_definitions(), "plugins": [describe_plugin(p) for p in self.human.plugins],
                    "transport": "WebSocket / MCP", "scope": "shared local runtime",
                    "observations": sorted((await self.human.get_context()).get("observations", {}))}
        if op == "configure_feature":
            from .settings import configure_feature
            return await configure_feature(self.human, body.get("name"), body.get("enabled"))
        if op == "test_vlm":
            from .realtime import scene_worker
            worker = scene_worker(self.human)
            if worker is None:
                raise ValueError("No local VLM configured")
            if self._vlm_test_lock.locked():
                raise ValueError("VLM test already running")
            async with self._vlm_test_lock:
                # A synthetic image exercises the real vision endpoint without devices
                # or writing a test observation into the user's context.
                import base64
                import cv2
                import numpy as np
                from .plugins import InputFrame
                sample = np.full((192, 192, 3), 255, dtype=np.uint8)
                sample[48:144, 48:144] = (255, 0, 0)
                ok, encoded = cv2.imencode('.jpg', sample)
                if not ok:
                    raise ValueError("Unable to construct VLM test frame")
                started = time.monotonic()
                frame = InputFrame({"frames": [{"timestamp_ms": 0, "measurements": {},
                    "jpeg": base64.b64encode(encoded).decode()}], "settings": worker.plugin.settings()},
                    "video", 0)
                output = await asyncio.to_thread(worker.plugin._describe, frame)
                return {"ok": True, "latency_ms": round((time.monotonic()-started)*1000),
                        "description": output[0]["value"], "input": "synthetic blue square"}
        if op == "configure_asr":
            language = body.get("language")
            if language not in ("auto", "en", "zh", "ko"):
                raise ValueError("Choose auto, en, zh or ko")
            if not self.human.paused:
                raise ValueError("Stop capture before changing speech language")
            plugin = self.asr_plugin()
            if plugin is None:
                raise ValueError("Speech recognition model is not installed")
            plugin.reset()
            plugin.language = None if language == "auto" else language
            return self.asr_settings()
        if op == "configure_vlm":
            return await configure_scene(self.human, {k:v for k,v in body.items() if k != "op"})
        if op == "state":
            from .settings import feature_status
            return {"features": feature_status(self.human), "text_companion_enabled": getattr(self.human, "text_companion_enabled", True), "asr": self.asr_settings(), "routing": self.router.status(), "vlm": scene_status(self.human), "state": await self.human.get_state(),
                    "context": await self.human.get_context(),
                    "capture": self.capture, "adapter_errors": self.errors,
                    "companion_available": self.companion is not None and self.companion.backend is not None,
                    "analysis_available": self.human.analysis_backend is not None,
                    "plugins": [p.name for p in self.human.plugins],
                    "workers": [p.status() for p in self.human.plugins if callable(getattr(p, 'status', None))],
                    "latest_inference": (await self.human.get_history(type="human.inference", limit=1)),
                    "events": await self.human.get_history(type="observation.event", limit=20)}
        if op == "start":
            if body.get("consent") is not True:
                raise ValueError("Explicit consent is required")
            await self.human.resume()
            self.capture = True
            return {"capture": True}
        if op == "stop":
            self.capture = False
            await self.human.pause()
            return {"capture": False}
        if op == "delete":
            await self.human.delete_data()
            return {"deleted": True}
        if not self.capture:
            raise ValueError("Start with consent before submitting observations")
        if op in ("chat", "text", "claim") and not getattr(self.human, "text_companion_enabled", True):
            raise ValueError("Text and Companion are disabled in settings")
        if op == "chat":
            if self.companion is None:
                raise ValueError("No companion backend configured")
            return await self.companion.reply(body["text"])
        if op == "analyze":
            return await self.human.analyze()
        if op == "feedback":
            return await self.human.record_feedback(body["inference_id"], body["outcome"], prediction=body.get("prediction"))
        if op == "text":
            return await self.human.push_text(str(body["text"])[:16000], role=body.get("role", "user"))
        if op == "claim":
            return await self.human.push_claim(str(body["key"]), body["value"], scope=body.get("scope", "session"))
        if op in ('audio', 'video'):
            from .media import decode_media
            modality, data, rate = decode_media(body)
            return await self.router.submit(modality, data, source='dashboard', sample_rate=rate)
        raise ValueError("Unknown operation")


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No text or media in access logs.

        def respond(self, status, payload, content_type="application/json"):
            raw = payload.encode() if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(raw)

        def allowed_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if self.allowed_host() and self.path in ("/observability.js", "/i18n.js"):
                return self.respond(200, Path(__file__).with_name(self.path[1:]).read_text(), "application/javascript")
            if not self.allowed_host() or self.path != "/":
                return self.respond(404, {"error": "Not found"})
            page = Path(__file__).with_name("dashboard.html").read_text()
            self.respond(200, page.replace("__SESSION_TOKEN__", app.token), "text/html")

        def do_POST(self):
            if not self.allowed_host() or self.path != "/api":
                return self.respond(404, {"error": "Not found"})
            if not secrets.compare_digest(self.headers.get("X-HumanSense-Token", ""), app.token):
                return self.respond(403, {"error": "Invalid session token"})
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://localhost:{self.server.server_port}", f"http://127.0.0.1:{self.server.server_port}"}:
                return self.respond(403, {"error": "Invalid origin"})
            future = None
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2_000_000:
                    return self.respond(413, {"error": "Invalid request size"})
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                if body.get('op') == 'stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/x-ndjson')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.end_headers()
                    try:
                        while not app.human.closed:
                            future = asyncio.run_coroutine_threadsafe(app.dispatch({'op': 'state'}), app.loop)
                            data = future.result(timeout=5)
                            self.wfile.write((json.dumps(data, ensure_ascii=False) + '\n').encode())
                            self.wfile.flush()
                            time.sleep(.25)
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        return
                    return
                future = asyncio.run_coroutine_threadsafe(app.dispatch(body), app.loop)
                self.respond(200, future.result(timeout=90 if body.get("op") in ("chat", "analyze", "test_vlm") else 30))
            except Exception as exc:
                if future is not None and not future.done():
                    future.cancel()
                self.respond(400, {"error": str(exc)})
    return Handler


async def serve(port, store, face_model=None, asr_model=None, vad_model=None, scene_url=None, scene_model=None, websocket_port=None, asr_language=None, plugins_config=None, pose_model=None):
    from . import HumanSense
    from .plugins import AudioLevelPlugin, OpenCVPersonPlugin
    async with HumanSense(consent=True, store=store, retention_hours=1/6, sensor_inference_interval_ms=250) as human:
        from .reasoning import backend_from_env
        backend = backend_from_env()
        if backend is not None:
            human.set_analysis_backend(backend)
        app = Dashboard(human, asyncio.get_running_loop())
        from .companion import Companion, companion_backend_from_env
        app.companion = Companion(human, companion_backend_from_env())
        for factory in (AudioLevelPlugin, OpenCVPersonPlugin):
            try:
                human.register(factory())
            except (ImportError, RuntimeError) as exc:
                app.errors.append(str(exc))
        from .plugins import MediaPipeFacePlugin, FasterWhisperPlugin, SileroVADPlugin
        from .realtime import QueuedPlugin, LocalScenePlugin
        for path, factory in ((face_model, MediaPipeFacePlugin), (asr_model, FasterWhisperPlugin), (vad_model, SileroVADPlugin)):
            if path:
                plugin = factory(path)
                if factory is FasterWhisperPlugin:
                    plugin.language = asr_language
                    plugin = QueuedPlugin(human, plugin, capacity=4)
                human.register(plugin)
        if pose_model:
            from .providers.pose import MediaPipePosePlugin
            human.register(QueuedPlugin(human, MediaPipePosePlugin(pose_model), capacity=1))
        if plugins_config:
            from .extensions import register_config
            register_config(human, json.loads(Path(plugins_config).read_text()))
        if scene_url:
            human.register(QueuedPlugin(human, LocalScenePlugin(scene_url, scene_model, human=human), capacity=1))
        websocket_task = None
        if websocket_port:
            from .server import serve_websocket
            from .mcp import MCPServer
            adapter = MCPServer(human)
            adapter.companion = app.companion
            websocket_task = asyncio.create_task(serve_websocket(adapter, websocket_port))
        await human.pause()
        server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"HumanSense: http://127.0.0.1:{server.server_port} — consent required in browser", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            if websocket_task:
                websocket_task.cancel()
                await asyncio.gather(websocket_task, return_exceptions=True)
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


def main():
    parser = argparse.ArgumentParser(description="HumanSense local browser dashboard")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--store", default=":memory:")
    parser.add_argument("--face-model", help="Local MediaPipe task file")
    parser.add_argument("--asr-model", help="Local faster-whisper model directory")
    parser.add_argument("--vad-model", help="Local Silero JIT file")
    parser.add_argument('--scene-url', help='Opt-in loopback VLM chat-completions URL; sends camera frames locally')
    parser.add_argument('--scene-model', default='local-vlm')
    parser.add_argument('--websocket-port', type=int, help='Expose the SAME runtime to local agent WebSocket clients')
    parser.add_argument('--pose-model', help='Local MediaPipe pose model path')
    parser.add_argument('--asr-language', help='Whisper language code, e.g. ko or en; omit for automatic detection')
    parser.add_argument("--plugins-config", help="Explicit JSON list of installed plugin factories and local model options")
    args = parser.parse_args()
    try:
        asyncio.run(serve(args.port, args.store, args.face_model, args.asr_model, args.vad_model, args.scene_url, args.scene_model, args.websocket_port, args.asr_language, args.plugins_config, args.pose_model))
    except KeyboardInterrupt:
        pass
    except PermissionError as exc:
        parser.exit(1, f"Cannot open local dashboard socket: {exc}. Run outside a socket-restricted sandbox.\n")


if __name__ == "__main__":
    main()
