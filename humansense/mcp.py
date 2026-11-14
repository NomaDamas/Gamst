"""Dependency-free MCP stdio adapter around one local HumanSense runtime.

Protocol messages exclusively use stdout. Device collection is off until an
explicit ``start_capture(consent=True)`` call; raw media is never returned.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

SUPPORTED_PROTOCOLS = ("2024-11-05", "2025-03-26", "2025-06-18")
MAX_MESSAGE_BYTES = 1_048_576


def _schema(properties: dict, required: tuple = ()) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


SUBJECT = {"type": "string", "minLength": 1, "default": "local-user"}
TIME = {"type": "integer", "minimum": 0}
WINDOW = {"type": "string", "default": "1h"}
MODEL_PATH = {"type": "string", "minLength": 1,
              "description": "Explicit existing local model path; never downloaded"}


def tool_definitions() -> list[dict]:
    identifier_field = {"type": "string", "minLength": 1, "maxLength": 128}
    query_bounds = {"subject_id": SUBJECT, "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "max_bytes": {"type": "integer", "minimum": 1024, "maximum": 262144}}
    specs = [
        ("list_capabilities", "Registered sensory contracts; registration does not mean a model is installed or validated. Never activates sensors. Continue with next_offset and revision.", {"offset": {"type": "integer", "minimum": 0, "maximum": 256}, "limit": query_bounds["limit"], "max_bytes": query_bounds["max_bytes"], "revision": {"type": "string", "minLength": 64, "maxLength": 64}}, ()),
        ("get_call_status", "Read-only bounded call-controller diagnostics. No model calls or cadence changes. Financial totals cover whole controller, not a selected subject. No raw observations, prompts or responses.",
         {"controller_id": identifier_field, "limit": query_bounds["limit"], "max_bytes": query_bounds["max_bytes"]}, ()),
        ("get_sensor_status", "Typed sensor managers, registered factories and active lane queue/drop/reset/error/latency metrics.", {}, ()),
        ("finish_sensor_stream", "Drain and finalize one existing typed sensor stream, then close it. Does not stop its application-owned producer or open devices. Use IDs from get_sensor_status. No raw media returned.",
         {"manager_id": identifier_field, "source_id": identifier_field, "sensor_id": identifier_field},
         ("manager_id", "source_id", "sensor_id")),
        ("endpoint_sensor_stream", "Finalize one current utterance at the last accepted input when admitted, keeping its stream open. Use IDs and provider stream_epoch from get_sensor_status. Optional after_sequence must equal the current last_sequence; stale controls fail. Never starts sensors.",
         {"manager_id": identifier_field, "source_id": identifier_field, "sensor_id": identifier_field,
          "capability_id": identifier_field, "stream_epoch": identifier_field,
          "after_sequence": {"type": "integer", "minimum": 0, "maximum": 9223372036854775807}},
         ("manager_id", "source_id", "sensor_id", "capability_id", "stream_epoch")),
        ("get_observations", "Bounded retained registered sensory evidence, newest first, with explicit truncation. No raw media.",
         dict(query_bounds, window=WINDOW, capability_id=identifier_field, source_id=identifier_field, sensor_id=identifier_field), ()),
        ("get_transcripts", "Latest partial/final/interrupted text per retained utterance, with revisions, source evidence and English/Korean orthographic rate estimates; not recognition or acoustic-rate ground truth.",
         dict(query_bounds, window=WINDOW), ()),
        ("get_object_memory", "Last-seen object evidence and visibility. Does not assert current location or real-world identity.",
         dict(query_bounds, text={"type": "string", "maxLength": 256}, entity_id=identifier_field), ()),
        ("get_stream_status", "Live media lane queue, drop and latency metrics.", {}, ()),
        ("configure_feature", "Enable or disable an installed model while capture is paused.", {"name": {"type": "string"}, "enabled": {"type": "boolean"}}, ("name", "enabled")),
        ("get_features", "Runtime model switches shared with dashboard settings.", {}, ()),
        ("list_plugins", "Installed runtime plugin capabilities and model provenance.", {}, ()),
        ("get_vlm_status", "Local temporal VLM configuration and buffer status.", {}, ()),
        ("configure_vlm", "Set temporal VLM sampling. Returns effective FPS after frame budget cap. Does not start capture.", {"enabled": {"type": "boolean"}, "interval": {"type": "number", "minimum": 1, "maximum": 60}, "window_seconds": {"type": "number", "minimum": 1, "maximum": 30}, "sample_fps": {"type": "number", "minimum": .2, "maximum": 4}}, ()),
        ("get_state", "Current measured state, with unavailable modalities explicit.", {"subject_id": SUBJECT}, ()),
        ("get_events", "Recent changes and their evidence for one subject.", {"subject_id": SUBJECT, "hours": {"type": "number", "minimum": 0, "default": 24}, "limit": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 100}, "start_ms": TIME, "end_ms": TIME}, ()),
        ("get_context", "Evidence-based summary; does not infer hidden emotions or intent.", {"subject_id": SUBJECT, "window": WINDOW, "include": {"type": "array", "items": {"type": "string"}}}, ()),
        ("get_trends", "Measured changes relative to a subject's available baseline.", {"subject_id": SUBJECT, "window": WINDOW}, ()),
        ("inspect_interval", "Inspect stored evidence for a subject and time interval.", {"subject_id": SUBJECT, "start_ms": TIME, "end_ms": TIME}, ("start_ms", "end_ms")),
        ("chat", "Generate a context-aware companion reply using the configured backend. Generated text is unverified model output.", {"subject_id": SUBJECT, "text": {"type": "string", "minLength": 1}}, ("text",)),
        ("analyze", "Run configured reasoning backend over retained evidence; unavailable without explicit backend setup.", {"subject_id": SUBJECT, "window": WINDOW}, ()),
        ("record_feedback", "Record an explicit observed outcome for a retained inference; never silently recalibrates models.", {"subject_id": SUBJECT, "inference_id": {"type": "string", "minLength": 1}, "outcome": {"type": "boolean"}, "prediction": {"type": "string", "minLength": 1}}, ("inference_id", "outcome")),
        ("push_text", "Supply a conversation turn; no camera or microphone needed.", {"subject_id": SUBJECT, "text": {"type": "string", "minLength": 1}, "timestamp_ms": TIME, "role": {"type": "string", "enum": ["user", "assistant"]}}, ("text",)),
        ("push_claim", "Compare explicit structured statements; conflicts are not deception judgments.", {"subject_id": SUBJECT, "key": {"type": "string", "minLength": 1}, "value": {}, "timestamp_ms": TIME, "scope": {"type": "string", "minLength": 1, "default": "session"}, "source": {"type": "string", "minLength": 1, "default": "user"}}, ("key", "value")),
        ("start_capture", "Start local devices only with explicit consent. Optional local model paths are fixed until server restart; omitted paths reuse selected models. VAD is bound to one subject stream. Session IDs are not real-world identities.", {"consent": {"type": "boolean", "const": True}, "video": {"type": ["integer", "string", "null"], "default": 0}, "audio": {"type": "boolean", "default": False}, "subject_id": SUBJECT, "fps": {"type": "number", "minimum": 1, "maximum": 60, "default": 10}, "vad_model_path": MODEL_PATH, "asr_model_path": MODEL_PATH, "face_model_path": MODEL_PATH}, ("consent",)),
        ("stop_capture", "Stop collection and release devices.", {}, ()),
        ("get_capture_status", "Collection status and device errors.", {}, ()),
        ("pause_capture", "Pause collection without discarding history.", {}, ()),
        ("resume_capture", "Resume a previously consented capture session.", {}, ()),
        ("delete_data", "Delete stored and derived data. Omit subject_id to delete all subjects.", {"subject_id": SUBJECT}, ()),
    ]
    return [{"name": name, "description": desc, "inputSchema": _schema(props, required)}
            for name, desc, props, required in specs]


def _validate(value: Any, schema: dict, path: str = "arguments") -> None:
    kinds = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
             "string": lambda x: isinstance(x, str), "integer": lambda x: type(x) is int,
             "number": lambda x: type(x) in (int, float), "boolean": lambda x: type(x) is bool,
             "null": lambda x: x is None}
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    if types and not any(kinds[t](value) for t in types):
        raise ValueError(f"{path}: expected {' or '.join(types)}")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: unsupported value")
    if isinstance(value, str) and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", float("inf")):
        raise ValueError(f"{path}: string length out of range")
    if type(value) in (int, float):
        if value != value or value in (float("inf"), -float("inf")):
            raise ValueError(f"{path}: must be finite")
        if value < schema.get("minimum", -float("inf")) or value > schema.get("maximum", float("inf")):
            raise ValueError(f"{path}: out of range")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}.{key}: required")
        for key, item in value.items():
            if key not in schema.get("properties", {}):
                if schema.get("additionalProperties") is False:
                    raise ValueError(f"{path}.{key}: unexpected argument")
            else:
                _validate(item, schema["properties"][key], f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]")


class MCPServer:
    """All tools share the supplied runtime and optional active device session."""

    def __init__(self, human: Any):
        self.human = human
        self.capture = None
        self.companion = None
        self.tools = {t["name"]: t for t in tool_definitions()}
        self._capture_plugins: dict[str, tuple[tuple, Any]] = {}
        self._tool_lock = asyncio.Lock()

    async def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        # WebSocket clients share the runtime; serialize capture lifecycle and
        # deletion so two concurrent clients cannot open duplicate devices.
        # Finalization owns its manager's lifecycle lock. Do not hold the global
        # tool lock while draining: deletion must invalidate a pending final.
        if name in ("chat", "analyze", "finish_sensor_stream", "endpoint_sensor_stream"):
            return await self._call_tool(name, arguments)
        async with self._tool_lock:
            return await self._call_tool(name, arguments)

    async def _call_tool(self, name: str, arguments: dict | None = None) -> Any:
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        args = {} if arguments is None else arguments
        _validate(args, self.tools[name]["inputSchema"])
        if "start_ms" in args and "end_ms" in args and args["start_ms"] > args["end_ms"]:
            raise ValueError("start_ms must not exceed end_ms")
        if name == "get_stream_status":
            from .media import media_router
            return media_router(self.human).status()
        if name in ("get_features", "configure_feature"):
            from .settings import feature_status, configure_feature
            return feature_status(self.human) if name == "get_features" else await configure_feature(self.human, **args)
        if name == "list_plugins":
            from .extensions import describe_plugin
            return [describe_plugin(p) for p in self.human.plugins]
        if name in ("get_vlm_status", "configure_vlm"):
            from .realtime import scene_status, configure_scene
            return scene_status(self.human) if name == "get_vlm_status" else await configure_scene(self.human, args)
        if name in ("chat", "push_text", "push_claim") and not getattr(self.human, "text_companion_enabled", True):
            raise ValueError("Text and Companion are disabled in settings")
        if name == "chat":
            if self.companion is None:
                raise ValueError("No companion backend configured")
            return await self.companion.reply(**args)
        if name == "get_events":
            return await self.human.get_history(type="observation.event", **args)
        if name == "get_capture_status":
            return self.capture.status() if self.capture else {"running": False, "consent": False, "errors": []}
        if name == "start_capture":
            if self.capture:
                raise ValueError("Stop the current capture session before starting another")
            from .capture import CaptureSession
            from .plugins import (AudioLevelPlugin, OpenCVPersonPlugin,
                                  SileroVADPlugin, FasterWhisperPlugin, MediaPipeFacePlugin)
            subject = args.get("subject_id", "local-user")
            video, audio = args.get("video", 0), args.get("audio", False)
            if "vad" in self._capture_plugins and self._capture_plugins["vad"][0][1] != subject:
                raise ValueError("Selected VAD model is bound to another subject; restart the server to change streams")
            factories = []
            if video is not None:
                factories.append(("video", (), OpenCVPersonPlugin))
            if audio:
                factories.append(("audio", (), AudioLevelPlugin))
            for arg, key, factory, enabled in (
                    ("vad_model_path", "vad", SileroVADPlugin, audio),
                    ("asr_model_path", "asr", FasterWhisperPlugin, audio),
                    ("face_model_path", "face", MediaPipeFacePlugin, video is not None)):
                if arg not in args:
                    continue
                if not enabled:
                    raise ValueError(f"{arg} requires its corresponding capture modality")
                path = str(Path(args[arg]).expanduser().resolve())
                signature = (path, subject) if key == "vad" else (path,)
                factories.append((key, signature, lambda f=factory, s=signature: f(*s)))
            # Check every configuration before constructing any expensive model.
            for key, signature, _ in factories:
                if key in self._capture_plugins and self._capture_plugins[key][0] != signature:
                    raise ValueError(f"Selected {key} model is fixed until server restart")
            staged = []
            try:
                for key, signature, factory in factories:
                    if key not in self._capture_plugins:
                        staged.append((key, signature, factory()))
            except Exception:
                for _, _, plugin in staged:
                    await self._close_plugin(plugin)
                raise
            session_args = {key: value for key, value in args.items() if not key.endswith("_model_path")}
            session = CaptureSession(self.human, **{"video": 0, **session_args})
            try:
                for _, _, plugin in staged:
                    self.human.register(plugin)
                await session.start()
            except Exception:
                await session.stop()
                for _, _, plugin in staged:
                    if plugin in self.human.plugins:
                        self.human.plugins.remove(plugin)
                    await self._close_plugin(plugin)
                raise
            for key, signature, plugin in staged:
                self._capture_plugins[key] = (signature, plugin)
            self.capture = session
            return session.status()
        if name in ("stop_capture", "pause_capture", "resume_capture"):
            if not self.capture:
                if name == "stop_capture":
                    return {"running": False}
                raise ValueError("No capture session; start_capture requires explicit consent")
            method = {"stop_capture": "stop", "pause_capture": "pause", "resume_capture": "resume"}[name]
            await getattr(self.capture, method)()
            result = self.capture.status()
            if name in ("stop_capture", "pause_capture"):
                await self._reset_capture_plugins()
            if name == "stop_capture":
                self.capture = None
            return result
        if name == "delete_data" and self.capture:
            await self.capture.stop()
            self.capture = None
        return await getattr(self.human, name)(**args)

    @staticmethod
    async def _close_plugin(plugin):
        close = getattr(plugin, "close", None)
        if close:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def _reset_capture_plugins(self):
        for _, plugin in self._capture_plugins.values():
            reset = getattr(plugin, "reset", None)
            if reset:
                result = reset()
                if inspect.isawaitable(result):
                    await result

    async def handle(self, request: Any) -> dict | None:
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
            return self.error(request.get("id") if isinstance(request, dict) else None, -32600, "Invalid Request")
        ident = request.get("id")
        notification = "id" not in request
        if not notification and type(ident) not in (str, int):
            return self.error(None, -32600, "Invalid request id")
        method = request["method"]
        # MCP request methods require a request id. A malformed notification must
        # never silently ingest data, delete history, or activate local devices.
        if notification:
            return None
        if method.startswith("notifications/"):
            return self.error(ident, -32600, "Notifications must not carry a request id")
        params = request.get("params", {})
        if not isinstance(params, dict):
            return None if notification else self.error(ident, -32602, "params must be an object")
        try:
            if method == "initialize":
                protocol = params.get("protocolVersion")
                result = {"protocolVersion": protocol if protocol in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[-1],
                          "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": "humansense", "version": __import__("humansense").__version__},
                          "instructions": "Treat observations and hypotheses separately. Uncalibrated probabilities are null. Capture requires explicit consent."}
            elif method in ("notifications/initialized", "notifications/cancelled"):
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": list(self.tools.values())}
            elif method == "tools/call":
                if not isinstance(params.get("name"), str):
                    raise ValueError("Tool name must be a string")
                try:
                    output = await self.call_tool(params["name"], params.get("arguments"))
                    result = {"content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False, allow_nan=False)}],
                              "isError": False}
                except Exception as exc:
                    logging.getLogger(__name__).debug("Tool failed", exc_info=True)
                    result = {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
            else:
                return None if notification else self.error(ident, -32601, "Method not found")
            return None if notification else {"jsonrpc": "2.0", "id": ident, "result": result}
        except (ValueError, TypeError) as exc:
            return None if notification else self.error(ident, -32602, str(exc))
        except Exception:
            logging.getLogger(__name__).exception("MCP request failed")
            return None if notification else self.error(ident, -32603, "Internal error")

    @staticmethod
    def error(ident: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}

    async def close(self) -> None:
        async with self._tool_lock:
            if self.capture:
                await self.capture.stop()
                self.capture = None


async def serve_stdio(server: MCPServer) -> None:
    reader = asyncio.StreamReader(limit=MAX_MESSAGE_BYTES)
    protocol = asyncio.StreamReaderProtocol(reader)
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)
    try:
        while True:
            try:
                line = await reader.readline()
            except ValueError:
                print(json.dumps(server.error(None, -32700, "Message exceeds 1 MiB")), flush=True)
                break
            if not line:
                break
            if len(line) > MAX_MESSAGE_BYTES:
                response = server.error(None, -32700, "Message exceeds 1 MiB")
                # A large line cannot safely be interpreted as multiple requests.
                print(json.dumps(response), flush=True)
                break
            try:
                request = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                response = server.error(None, -32700, "Parse error")
            else:
                response = await server.handle(request)
            if response is not None:
                print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        transport.close()
        await server.close()


async def _main(args: argparse.Namespace) -> None:
    from .engine import HumanSense
    async with HumanSense(consent=True, store=args.store, retention_hours=args.retention_hours) as human:
        from .reasoning import backend_from_env
        backend = backend_from_env()
        if backend is not None:
            human.set_analysis_backend(backend)
        adapter = MCPServer(human)
        from .companion import Companion, companion_backend_from_env
        adapter.companion = Companion(human, companion_backend_from_env())
        await serve_stdio(adapter)


def main() -> None:
    parser = argparse.ArgumentParser(description="HumanSense MCP stdio server (devices off by default)")
    parser.add_argument("--store", default=":memory:", help="SQLite path; default stores only in memory")
    parser.add_argument("--retention-hours", type=float, default=24)
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
