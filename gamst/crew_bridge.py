"""Optional bridge from GAMST to a running GamstCrew Unix packet runtime."""
from __future__ import annotations
import asyncio, json
from .temporal import TemporalEvent

class CrewBridge:
    def __init__(self, socket_path: str, *, max_line: int = 131072):
        self.socket_path, self.max_line = socket_path, max_line
    async def submit(self, event: TemporalEvent):
        request = {'op': 'submit', 'packet': event.as_dict()}
        raw = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode() + b'\n'
        if len(raw) > self.max_line: raise ValueError('Crew request exceeds byte budget')
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        try:
            writer.write(raw); await writer.drain()
            response = await reader.readline()
            if len(response) > self.max_line: raise ValueError('Crew response exceeds byte budget')
            result = json.loads(response)
            if result.get('error'): raise RuntimeError(result['error'])
            return result
        finally:
            writer.close(); await writer.wait_closed()
