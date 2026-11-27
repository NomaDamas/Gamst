"""The semantic module as a causal state chain (owner decision 47), with a stand-in VLM: no endpoint is called."""
import time
import unittest

from gamst import Gamst
from gamst.capabilities import CapabilitySpec, OutputField, SensorMetadata
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.providers.streaming_semantic import StreamingSemanticContext
from gamst.recipes.perception_parts import SegmentModule
from gamst.state_chain import attributed, diff, link, walk


class StandIn:
    """Stands in for the VLM: the states it says, in order."""
    model_id, model_version = 'stand-in', '0'

    def __init__(self, *states):
        self.states, self.calls = list(states), 0

    def predict(self, frames, timestamps_ms, previous_state, context=None):
        state = self.states[min(self.calls, len(self.states) - 1)]
        self.calls += 1
        return dict(state=dict(state), events=[], summary=f'state {self.calls}', evidence_ids=[])

    def close(self):
        pass


def frame(stamp):
    return InputFrame(b'', 'video', stamp, sensor=SensorMetadata('c', 'v', stamp))


class Chain(unittest.TestCase):
    def test_diff_says_what_was_set_changed_and_removed(self):
        changed = diff({'phase': 'sit', 'who': {'name': 'A', 'pose': 'down'}, 'old': 1},
                       {'phase': 'stand', 'who': {'name': 'A', 'pose': 'up'}, 'new': 2})
        self.assertEqual(changed['changed'], {'phase': {'from': 'sit', 'to': 'stand'},
                                              'who.pose': {'from': 'down', 'to': 'up'}})
        self.assertEqual((changed['set'], changed['removed'], changed['truncated']), ({'new': 2}, ['old'], False))
        self.assertEqual(diff(None, {'a': 1})['set'], {'a': 1})
        many = diff({}, {f'k{i}': i for i in range(40)})
        self.assertEqual((len(many['set']), many['truncated']), (32, True))

    def test_causes_are_records_that_ended_since_the_previous_state(self):
        candidates = [dict(record_id='r1', end_ms=900), dict(record_id='r2', end_ms=1500, who='A'),
                      dict(record_id='r3', end_ms=2100), dict(record_id='r4', start_ms=1200, end_ms=2000), {'end_ms': 1}]
        self.assertEqual([item['record_id'] for item in attributed(candidates, 1000, 2000)], ['r2', 'r4'])
        self.assertEqual([item['record_id'] for item in attributed(candidates, None, 2000, start_ms=1000)], ['r2', 'r4'])
        first = link(None, {'p': 1}, identity='m', start_ms=1000, end_ms=2000, candidates=candidates)
        self.assertNotIn('previous_state_id', first)
        self.assertEqual(first['attribution'], 'interval')
        same = link(dict(state_id=first['state_id'], state={'p': 1}, end_ms=2000), {'p': 1}, identity='m',
                    start_ms=2500, end_ms=3000, candidates=[dict(record_id='r5', end_ms=2600)])
        self.assertEqual((same['previous_state_id'], same['caused_by'], same['attribution']), (first['state_id'], [], 'none'))

    def test_the_provider_links_each_segment_to_the_one_before(self):
        provider = StreamingSemanticContext(StandIn({'phase': 'sit'}, {'phase': 'sit'}, {'phase': 'stand'}),
                                            window_frames=2, max_gap_ms=2000)
        values = [provider.process(frame(1000 + 500 * i)) for i in range(6)]
        first, second, third = [value for value in values if value]
        self.assertNotIn('previous_state_id', first['chain'])
        self.assertEqual(second['chain']['previous_state_id'], first['chain']['state_id'])
        self.assertEqual(third['chain']['previous_state_id'], second['chain']['state_id'])
        self.assertFalse(second['chain']['changed']['changed'])
        self.assertEqual(third['chain']['changed']['changed'], {'phase': {'from': 'sit', 'to': 'stand'}})
        self.assertEqual(third['chain']['since_ms'], second['window']['end_ms'])
        StreamingSemanticContext.capability().validate(third)
        self.assertIsNone(provider.process(frame(4000)))
        self.assertIsNone(provider.process(frame(20000)))     # a gap in a segment: the model starts again, so does the chain
        self.assertNotIn('previous_state_id', provider.process(frame(20500))['chain'])

    def test_walk_is_bounded_and_says_why_it_stopped(self):
        states, previous = [], None
        for index in range(12):
            chain = link(previous, {'n': index}, identity='m', start_ms=1000 * index, end_ms=1000 * index + 500)
            states.append(dict(end_ms=1000 * index + 500, state={'n': index}, chain=chain))
            previous = dict(state_id=chain['state_id'], state={'n': index}, end_ms=1000 * index + 500)
        found = walk(states, steps=4)
        self.assertEqual(([item['state']['n'] for item in found['states']], found['stopped']), ([11, 10, 9, 8], 'steps'))
        self.assertEqual([item['state']['n'] for item in walk(states, at_ms=3600, steps=32)['states']], [3, 2, 1, 0])
        self.assertEqual(walk(states, at_ms=3600, steps=32)['stopped'], 'start')
        self.assertEqual(walk(states[5:], state_id=states[6]['chain']['state_id'])['stopped'], 'not_retained')
        self.assertEqual(walk(states, max_bytes=1024, steps=32)['stopped'], 'bytes')
        with self.assertRaises(ValueError):
            walk(states, steps=0)


class Worker:
    """A StreamingSemanticContext in this process in place of its worker process (no endpoint at all)."""
    private = False

    def __init__(self, provider):
        self.provider, self.spec = provider, StreamingSemanticContext.capability(window_ms=1000)

    async def process(self, item):
        return self.provider.process(item)

    async def finish(self):
        return self.provider.finish()

    async def reset(self):
        self.provider.reset()

    async def close(self):
        self.provider.close()


class Shotguns:
    """Stands in for a movement counter: one event of 감스트's track at 1.2 s."""
    spec = CapabilitySpec('test.movement', 'test.movement', 'video', 'test', '1', (OutputField('events', 'array'),),
                          'frame', freshness_ms=60000)

    def __init__(self, at_ms):
        self.at_ms, self.done = at_ms, False

    def process(self, item):
        if self.done or item.timestamp_ms < self.at_ms:
            return None
        self.done = True
        return dict(events=[dict(event='exemplar.matched', label='shotgun', start_ms=self.at_ms - 200, end_ms=self.at_ms,
                                 track_id='t1', who='감스트')])


class Lane(unittest.IsolatedAsyncioTestCase):
    async def test_states_name_their_causes_and_walk_back_over_mcp(self):
        from gamst.mcp import MCPServer
        start = int(time.time() * 1000) - 10000
        async with Gamst(consent=True) as human:
            provider = StreamingSemanticContext(StandIn({'action': 'none'}, {'action': 'shotgun'}), window_frames=2)
            semantic = SegmentModule(Worker(provider), human=human, capabilities=('test.movement',))
            pipeline = Pipeline(human, 'chain').add_input('camera', modality='video')
            pipeline.add_module('movement', Shotguns(start + 1200), consumes=['camera'], spec=Shotguns.spec)
            pipeline.add_module('semantic', semantic, consumes=['camera'], spec=semantic.spec)
            for index in range(5):
                await pipeline.push('camera', b'', timestamp_ms=start + 500 * index)
                await pipeline.drain()
            await pipeline.close()
            server = MCPServer(human)
            found = await server.call_tool('explain_state', {'steps': 8})
            newest, oldest = found['states']
            self.assertEqual(found['stopped'], 'start')
            self.assertEqual(newest['chain']['previous_state_id'], oldest['chain']['state_id'])
            self.assertEqual(newest['chain']['changed']['changed'], {'action': {'from': 'none', 'to': 'shotgun'}})
            (cause,) = newest['chain']['caused_by']
            self.assertEqual((cause['capability'], cause['event'], cause['who'], cause['end_ms']),
                             ('test.movement', 'exemplar.matched', '감스트', start + 1200))
            self.assertTrue(cause['record_id'])
            self.assertEqual(oldest['chain']['caused_by'], [])        # the first state: nothing changed before it ended
            again = await server.call_tool('explain_state', {'state_id': oldest['chain']['state_id']})
            self.assertEqual(again['count'], 1)

    async def test_the_short_term_chain_links_too(self):
        async with Gamst(consent=True) as human:
            now = int(time.time() * 1000)
            first = await human.push_semantic_state({'phase': 'ready'}, timestamp_ms=now - 1000)
            second = await human.push_semantic_state({'phase': 'go'}, timestamp_ms=now,
                                                     caused_by=[dict(record_id='r1', end_ms=now - 10)])
            self.assertEqual(second['chain']['previous_state_id'], first['chain']['state_id'])
            self.assertEqual(second['chain']['changed']['changed'], {'phase': {'from': 'ready', 'to': 'go'}})
            self.assertEqual((second['chain']['attribution'], second['chain']['caused_by'][0]['record_id']), ('caller', 'r1'))
            chain = await human.explain_state()
            self.assertEqual([state['state']['phase'] for state in chain['states']], ['go', 'ready'])
            self.assertFalse(chain['states'][0]['chain']['caused_by'][0].get('retained', True))   # r1 is not a stored record


if __name__ == '__main__':
    unittest.main()
