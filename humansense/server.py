"""Optional loopback WebSocket transport for the same MCP/runtime contract.

Each connection can call the MCP JSON-RPC tools and receives
``notifications/humansense/event`` messages. Browser origins are rejected unless
explicitly supplied with --origin. Install humansense-sdk[websocket] first.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging

from .mcp import MAX_MESSAGE_BYTES, MCPServer


async def handle_connection(server, socket):
    """One WebSocket connection; custom subscriptions are separate from MCP."""
    from .media import media_router, decode_media
    import uuid
    router = media_router(server.human)
    source = "ws-" + uuid.uuid4().hex
    streaming = False
    lock = asyncio.Lock()
    publishers = []
    allowed = {"event": "events", "signal": "signals", "inference": "observe"}

    async def send(message):
        async with lock:
            await socket.send(json.dumps(message, ensure_ascii=False, allow_nan=False))

    async def publish(kind, context_options):
        if kind == 'context':
            async for context in server.human.contexts(**context_options):
                await send({'jsonrpc': '2.0', 'method': 'notifications/humansense/context',
                            'params': context})
            return
        async for record in getattr(server.human, allowed[kind])():
            await send({"jsonrpc": "2.0", "method": f"notifications/humansense/{kind}", "params": record})

    async def subscribe(kinds, context_options=None):
        context_options = context_options or {}
        for task in publishers:
            task.cancel()
        if publishers:
            await asyncio.gather(*publishers, return_exceptions=True)
        publishers.clear()
        publishers.extend(asyncio.create_task(publish(kind, context_options)) for kind in kinds)
        await asyncio.sleep(0)  # Register runtime iterators before acknowledging.

    await subscribe(["event"])
    try:
        async for raw in socket:
            try:
                request = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                response = server.error(None, -32700, "Parse error")
            else:
                if isinstance(request, dict) and request.get("method") == "humansense/subscribe":
                    ident = request.get("id")
                    if request.get("jsonrpc") != "2.0" or type(ident) not in (str, int):
                        response = server.error(None, -32600, "Subscription requires a request ID")
                    else:
                        params = request.get("params", {})
                        kinds = params.get("types") if isinstance(params, dict) else None
                        if not isinstance(kinds, list) or len(kinds) > 4 or any(type(kind) is not str or kind not in (*allowed, 'context') for kind in kinds) or len(set(kinds)) != len(kinds):
                            response = server.error(ident, -32602, "types must select unique event, signal, inference, context values")
                        else:
                            try:
                                from .engine import _window_ms
                                allowed_params = {'types', 'interval_ms', 'subject_id', 'window'}
                                if set(params)-allowed_params:
                                    raise ValueError('unknown subscription parameter')
                                interval_ms = params.get('interval_ms', 250)
                                subject_id = params.get('subject_id', 'local-user')
                                window = params.get('window', '5m')
                                if type(interval_ms) is not int or not 50 <= interval_ms <= 60000:
                                    raise ValueError('interval_ms must be 50..60000')
                                if not isinstance(subject_id, str) or not 1 <= len(subject_id) <= 128 or not subject_id.strip():
                                    raise ValueError('subject_id must be 1..128 nonblank characters')
                                if not isinstance(window, str) or len(window) > 32 or _window_ms(window) > 86400000:
                                    raise ValueError('window must be a positive duration at most one day')
                                if 'context' not in kinds and set(params)-{'types'}:
                                    raise ValueError('cadence/subject/window options require a context subscription')
                                await subscribe(kinds, dict(interval=interval_ms/1000, subject_id=subject_id, window=window))
                                result = {"types": kinds}
                                if 'context' in kinds:
                                    result['context'] = dict(interval_ms=interval_ms, subject_id=subject_id, window=window)
                                response = {"jsonrpc": "2.0", "id": ident, "result": result}
                            except (ValueError, TypeError, OverflowError) as exc:
                                response = server.error(ident, -32602, str(exc))
                elif isinstance(request, dict) and request.get('method') in (
                        'humansense/stream/start', 'humansense/stream/stop', 'humansense/stream/status', 'humansense/media'):
                    ident = request.get('id')
                    try:
                        if request.get('jsonrpc') != '2.0' or type(ident) not in (str,int):
                            raise ValueError('Media operations require a JSON-RPC request ID')
                        params = request.get('params', {})
                        if not isinstance(params, dict):
                            raise ValueError('params must be an object')
                        method = request['method']
                        if method == 'humansense/stream/start':
                            if params.get('consent') is not True:
                                raise ValueError('Explicit consent required')
                            await server.human.resume()
                            streaming = True
                            result = {'streaming': True, 'source': source}
                        elif method == 'humansense/stream/stop':
                            streaming = False
                            await router.clear(source)
                            result = {'streaming': False}
                        elif method == 'humansense/stream/status':
                            result = {'streaming': streaming, 'routing': router.status()}
                        else:
                            if not streaming:
                                raise ValueError('Start this media stream with consent first')
                            modality, data, rate = decode_media(params)
                            result = await router.submit(modality, data, source=source,
                                subject_id=params.get('subject_id','local-user'),
                                sequence=params.get('sequence'), timestamp_ms=params.get('timestamp_ms'), sample_rate=rate)
                        response = {'jsonrpc':'2.0', 'id':ident, 'result':result}
                    except (ValueError, TypeError, RuntimeError, PermissionError, ImportError) as exc:
                        response = server.error(ident, -32602, str(exc))
                else:
                    response = await server.handle(request)
            if response is not None:
                await send(response)
    finally:
        try:
            await router.clear(source)
        finally:
            for task in publishers:
                task.cancel()
            await asyncio.gather(*publishers, return_exceptions=True)


async def serve_websocket(server: MCPServer, port: int = 8765,
                          origins: list[str] | None = None) -> None:
    try:
        from websockets.asyncio.server import serve
    except ImportError as exc:
        raise RuntimeError("WebSocket transport requires pip install 'humansense-sdk[websocket]'") from exc

    # Bind only to loopback. Browsers must opt in with an exact trusted origin.
    async with serve(lambda socket: handle_connection(server, socket), "127.0.0.1", port, origins=[None, *(origins or [])],
                     max_size=MAX_MESSAGE_BYTES, max_queue=16):
        logging.getLogger(__name__).warning("HumanSense WebSocket listening on 127.0.0.1:%s", port)
        await asyncio.Future()


async def _main(args):
    from .engine import HumanSense
    async with HumanSense(consent=True, store=args.store, retention_hours=args.retention_hours) as human:
        adapter = MCPServer(human)
        try:
            from .reasoning import backend_from_env
            backend = backend_from_env()
            if backend is not None:
                human.set_analysis_backend(backend)
            from .companion import Companion, companion_backend_from_env
            adapter.companion = Companion(human, companion_backend_from_env())
            await serve_websocket(adapter, args.port, args.origin)
        finally:
            await adapter.close()


def main():
    parser = argparse.ArgumentParser(description="HumanSense local WebSocket server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--origin", action="append", default=[], help="Allow an exact trusted browser origin")
    parser.add_argument("--store", default=":memory:")
    parser.add_argument("--retention-hours", type=float, default=24)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
