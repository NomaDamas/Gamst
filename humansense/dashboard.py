"""Loopback-only browser demo. Raw browser media is processed, never persisted."""
import argparse
import asyncio
import base64
import json
import secrets
import threading
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

    async def dispatch(self, body):
        # Model calls must not hold a transport-wide lock: stop/delete need to
        # invalidate in-flight work immediately, not after a provider timeout.
        return await self._dispatch(body)

    async def _dispatch(self, body):
        op = body.get("op")
        if op == "state":
            return {"state": await self.human.get_state(),
                    "context": await self.human.get_context(),
                    "capture": self.capture, "adapter_errors": self.errors,
                    "companion_available": self.companion is not None and self.companion.backend is not None,
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
        if op == "audio":
            samples = body.get("samples", [])
            if not isinstance(samples, list) or len(samples) > 96000:
                raise ValueError("Audio block exceeds 96000 samples")
            rate = int(body.get("sample_rate", 16000))
            if rate < 8000 or rate > 96000:
                raise ValueError("Invalid sample rate")
            return await self.human.push_audio(samples, sample_rate=rate)
        if op == "video":
            encoded = body.get("jpeg", "")
            if not isinstance(encoded, str) or len(encoded) > 1_000_000:
                raise ValueError("Invalid video payload")
            try:
                import cv2
                import numpy as np
            except ImportError as exc:
                raise ValueError('Install "humansense-sdk[vision]" for video analysis') from exc
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[0] > 1080 or frame.shape[1] > 1920:
                raise ValueError("Invalid video dimensions")
            return await self.human.push_video(frame)
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
                future = asyncio.run_coroutine_threadsafe(app.dispatch(body), app.loop)
                self.respond(200, future.result(timeout=90 if body.get("op") in ("chat", "analyze") else 30))
            except Exception as exc:
                if future is not None and not future.done():
                    future.cancel()
                self.respond(400, {"error": str(exc)})
    return Handler


async def serve(port, store, face_model=None, asr_model=None, vad_model=None):
    from . import HumanSense
    from .plugins import AudioLevelPlugin, OpenCVPersonPlugin
    async with HumanSense(consent=True, store=store) as human:
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
        for path, factory in ((face_model, MediaPipeFacePlugin), (asr_model, FasterWhisperPlugin), (vad_model, SileroVADPlugin)):
            if path:
                human.register(factory(path))
        await human.pause()
        server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"HumanSense: http://127.0.0.1:{server.server_port} — consent required in browser", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
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
    args = parser.parse_args()
    try:
        asyncio.run(serve(args.port, args.store, args.face_model, args.asr_model, args.vad_model))
    except KeyboardInterrupt:
        pass
    except PermissionError as exc:
        parser.exit(1, f"Cannot open local dashboard socket: {exc}. Run outside a socket-restricted sandbox.\n")


if __name__ == "__main__":
    main()
