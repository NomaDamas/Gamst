import asyncio
import importlib.util
import unittest
from gamst.crew_bridge import CrewBridge
from gamst.temporal import TemporalEvent
if importlib.util.find_spec('gamstcrew') is not None:
    from gamstcrew.transport import JsonlUnixServer
else:
    JsonlUnixServer = None

def event():
    return TemporalEvent('e','s','u','vision','event','wall','clock-1','epoch-1',1,1,1,1,'fixture','1',{'x':1})


class BridgeValidation(unittest.TestCase):
    def test_timeout_is_finite_and_bounded(self):
        for value in (0, -1, float('nan'), float('inf'), 601):
            with self.assertRaises(ValueError):
                CrewBridge('/tmp/unused', timeout_s=value)

@unittest.skipIf(JsonlUnixServer is None, 'optional GamstCrew integration is not installed')
class Bridge(unittest.IsolatedAsyncioTestCase):
    def test_bridge_line_bound_matches_crew_server(self):
        with self.assertRaises(ValueError):
            CrewBridge('/tmp/gamst-test.sock', max_line=255)

    async def test_submit_roundtrip(self):
        async def handler(request):
            self.assertEqual(request['op'], 'submit')
            return {'accepted': True, 'event_id': request['packet']['event_id'],
                    'request_id': request['request_id']}
        server = JsonlUnixServer(handler)
        await server.start()
        try:
            result = await CrewBridge(server.path).submit(event())
            self.assertEqual(result, {'accepted': True, 'event_id': 'e'})
        finally:
            await server.close()

if __name__ == '__main__': unittest.main()
