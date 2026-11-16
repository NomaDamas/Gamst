import asyncio
import unittest
from gamst.crew_bridge import CrewBridge
from gamst.temporal import TemporalEvent
from gamstcrew.transport import JsonlUnixServer

def event():
    return TemporalEvent('e','s','u','vision','event','wall','clock-1','epoch-1',1,1,1,1,'fixture','1',{'x':1})

class Bridge(unittest.IsolatedAsyncioTestCase):
    async def test_submit_roundtrip(self):
        async def handler(request):
            self.assertEqual(request['op'], 'submit')
            return {'accepted': True, 'event_id': request['packet']['event_id']}
        server = JsonlUnixServer(handler)
        await server.start()
        try:
            result = await CrewBridge(server.path).submit(event())
            self.assertEqual(result, {'accepted': True, 'event_id': 'e'})
        finally:
            await server.close()

if __name__ == '__main__': unittest.main()
