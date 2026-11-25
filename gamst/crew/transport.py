from __future__ import annotations
import asyncio
import json
import math
import os
import tempfile
import uuid
from collections.abc import Mapping
from .packet import MAX_PACKET_BYTES


MAX_CORRELATION_ID = 128
MIN_LINE_BYTES = 256
MAX_TIMEOUT_S = 600.0


def _timeout(value, name):
    if type(value) not in (int, float):
        raise ValueError(
            f"{name} must be finite and between 0 and {MAX_TIMEOUT_S:g} seconds"
        )
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"{name} must be finite and between 0 and {MAX_TIMEOUT_S:g} seconds"
        ) from exc
    if not math.isfinite(converted) or not 0 < converted <= MAX_TIMEOUT_S:
        raise ValueError(f"{name} must be finite and between 0 and {MAX_TIMEOUT_S:g} seconds")
    return converted


def _correlation_id(value):
    """Validate the bounded request/response correlation token."""
    if (not isinstance(value, str) or not 1 <= len(value) <= MAX_CORRELATION_ID
            or value.strip() != value or "\n" in value or "\r" in value):
        raise ValueError("request_id must be a bounded string")
    return value


def _decode(line):
    """Decode one JSON object without duplicate keys or non-finite values."""
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(
        line,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("non-finite JSON value")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError("JSON request must be an object")
    return value


def _error_bytes(message, request_id, max_line):
    """Encode a protocol error without exceeding the negotiated line bound."""
    for text in (str(message), "transport protocol error"):
        value = {"error": text}
        if request_id is not None:
            value["request_id"] = request_id
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode() + b"\n"
        if len(raw) <= max_line:
            return raw
    # A bound smaller than the smallest JSON error cannot carry a response;
    # closing the connection is the only bounded behavior.
    return b""


class JsonlUnixServer:
    def __init__(self, handler, path=None, max_line=MAX_PACKET_BYTES,
                 handler_timeout_s=30.0):
        if type(max_line) is not int or not MIN_LINE_BYTES <= max_line <= MAX_PACKET_BYTES:
            raise ValueError(
                f"max_line must be an integer from {MIN_LINE_BYTES} to {MAX_PACKET_BYTES}"
            )
        self.handler, self.path, self.max_line, self.server = (
            handler,
            path or tempfile.mktemp(prefix="gamst-crew-", suffix=".sock"),
            max_line,
            None,
        )
        self.handler_timeout_s = _timeout(handler_timeout_s, "handler_timeout_s")

    async def start(self):
        if os.path.exists(self.path):
            os.unlink(self.path)
        # StreamReader's line limit bounds buffering before a delimiter arrives.
        self.server = await asyncio.start_unix_server(
            self._client, self.path, limit=self.max_line
        )

    async def _client(self, reader, writer):
        try:
            while True:
                # Never carry a prior request's ID into a malformed next-line
                # diagnostic; the connection is closed after a protocol error.
                request_id = None
                try:
                    line = await reader.readuntil(b"\n")
                except asyncio.LimitOverrunError as exc:
                    raise ValueError("line exceeds transport bound") from exc
                except asyncio.IncompleteReadError as exc:
                    if not exc.partial:
                        break
                    raise ValueError("truncated transport line") from exc
                if not line:
                    break
                if len(line) > self.max_line or not line.endswith(b"\n"):
                    raise ValueError("line exceeds transport bound")
                request = _decode(line)
                request_id = _correlation_id(request.get("request_id"))
                response = await asyncio.wait_for(
                    self.handler(request), timeout=self.handler_timeout_s
                )
                if not isinstance(response, Mapping):
                    raise ValueError("handler response must be an object")
                response = dict(response)
                returned_id = response.get("request_id", request_id)
                if _correlation_id(returned_id) != request_id:
                    raise ValueError("response request_id does not match request")
                response["request_id"] = request_id
                raw = (
                    json.dumps(
                        response,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
                if len(raw) > self.max_line:
                    raise ValueError("response exceeds transport bound")
                writer.write(raw)
                await writer.drain()
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError,
                TypeError, asyncio.TimeoutError) as exc:
            raw = _error_bytes(exc, request_id, self.max_line)
            if raw:
                writer.write(raw)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


async def jsonl_call(path, request, max_line=MAX_PACKET_BYTES, timeout_s=30.0):
    if type(max_line) is not int or not MIN_LINE_BYTES <= max_line <= MAX_PACKET_BYTES:
        raise ValueError(
            f"max_line must be an integer from {MIN_LINE_BYTES} to {MAX_PACKET_BYTES}"
        )
    timeout_s = _timeout(timeout_s, "timeout_s")
    if not isinstance(request, Mapping):
        raise TypeError("request must be an object")
    request = dict(request)
    request_id = request.get("request_id")
    if request_id is None:
        request_id = uuid.uuid4().hex
        request["request_id"] = request_id
    _correlation_id(request_id)
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(path, limit=max_line), timeout=timeout_s
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError("Unix transport connection timed out") from exc
    try:
        raw = json.dumps(request, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode() + b"\n"
        if len(raw) > max_line:
            raise ValueError("request exceeds transport bound")
        writer.write(raw)
        await asyncio.wait_for(writer.drain(), timeout=timeout_s)
        try:
            response = await asyncio.wait_for(reader.readuntil(b"\n"), timeout=timeout_s)
        except asyncio.LimitOverrunError as exc:
            raise ValueError("response exceeds transport bound") from exc
        if len(response) > max_line:
            raise ValueError("response exceeds transport bound")
        result = _decode(response)
        # A malformed or oversized request can fail before its request ID is
        # decoded. Preserve that bounded transport error for callers; valid
        # responses and handler errors must still echo the request ID.
        if "request_id" not in result and "error" in result:
            return result
        if _correlation_id(result.get("request_id")) != request_id:
            raise ValueError("response request_id does not match request")
        return result
    finally:
        writer.close()
        await writer.wait_closed()
