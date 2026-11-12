import asyncio
import json
import unittest
from humansense import HumanSense
from humansense.mcp import MCPServer
from humansense.server import handle_connection


class Socket:
    def __init__(self): self.incoming = asyncio.Queue(); self.outgoing = asyncio.Queue()
    def __aiter__(self): return self
    async def __anext__(self):
        value = await self.incoming.get()
        if value is None: raise StopAsyncIteration
        return json.dumps(value)
    async def send(self, raw): await self.outgoing.put(json.loads(raw))


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_subscription_streams_actual_runtime_records_and_cleans_up(self):
        async with HumanSense(consent=True) as human:
            server, socket = MCPServer(human), Socket()
            task = asyncio.create_task(handle_connection(server, socket))
            await socket.incoming.put({'jsonrpc': '2.0', 'id': 1, 'method': 'humansense/subscribe', 'params': {'types': ['inference','signal']}})
            ack = await asyncio.wait_for(socket.outgoing.get(), 1)
            self.assertEqual(ack['result']['types'], ['inference','signal'])
            result = await human.push_text('hello')
            messages = [await asyncio.wait_for(socket.outgoing.get(), 1) for _ in range(2)]
            inferred = next(m for m in messages if m['method'].endswith('/inference'))
            self.assertEqual(inferred['params']['id'], result['id'])
            self.assertTrue(any(m['method'].endswith('/signal') for m in messages))
            await socket.incoming.put(None)
            await task
            self.assertEqual(len(human._subscribers), 0)
            await server.close()

    async def test_invalid_subscriptions_and_default_events(self):
        async with HumanSense(consent=True) as human:
            server, socket = MCPServer(human), Socket()
            task = asyncio.create_task(handle_connection(server, socket))
            await asyncio.sleep(0); await asyncio.sleep(0)
            for kinds in (['unknown'], ['event','event'], 'event'):
                await socket.incoming.put({'jsonrpc':'2.0','id':2,'method':'humansense/subscribe','params':{'types':kinds}})
                response = await asyncio.wait_for(socket.outgoing.get(),1)
                self.assertEqual(response['error']['code'],-32602)
            await human.push_text('help me')
            event = await asyncio.wait_for(socket.outgoing.get(),1)
            self.assertEqual(event['method'],'notifications/humansense/event')
            await socket.incoming.put(None); await task; await server.close()
