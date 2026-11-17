"""Optional bridge from GAMST to a running GamstCrew Unix packet runtime."""
from __future__ import annotations
import asyncio, json, uuid
from .temporal import TemporalEvent

class CrewBridge:
    def __init__(self, socket_path: str, *, max_line: int = 131072,
                 timeout_s: float = 30.0):
        if type(max_line) is not int or max_line < 1:
            raise ValueError('max_line must be a positive integer')
        if type(timeout_s) not in (int, float) or timeout_s <= 0:
            raise ValueError('timeout_s must be positive')
        self.socket_path, self.max_line, self.timeout_s = socket_path, max_line, float(timeout_s)
    async def submit(self, event: TemporalEvent):
        request_id = uuid.uuid4().hex
        request = {'op': 'submit', 'packet': event.as_dict(), 'request_id': request_id}
        raw = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode() + b'\n'
        if len(raw) > self.max_line: raise ValueError('Crew request exceeds byte budget')
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.socket_path, limit=self.max_line),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError('Crew transport connection timed out') from exc
        try:
            writer.write(raw); await asyncio.wait_for(writer.drain(), timeout=self.timeout_s)
            try:
                response = await asyncio.wait_for(
                    reader.readuntil(b'\n'), timeout=self.timeout_s
                )
            except asyncio.IncompleteReadError as exc:
                raise ValueError('truncated Crew response') from exc
            except asyncio.LimitOverrunError as exc:
                raise ValueError('Crew response exceeds byte budget') from exc
            if len(response) > self.max_line: raise ValueError('Crew response exceeds byte budget')
            result = json.loads(
                response,
                object_pairs_hook=lambda pairs: _unique_object(pairs),
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-finite JSON value')),
            )
            if not isinstance(result, dict):
                raise ValueError('Crew response must be an object')
            if result.get('request_id') != request_id:
                raise RuntimeError('Crew response request_id does not match request')
            if result.get('error'): raise RuntimeError(result['error'])
            result.pop('request_id', None)
            return result
        finally:
            writer.close(); await writer.wait_closed()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON field')
        result[key] = value
    return result
