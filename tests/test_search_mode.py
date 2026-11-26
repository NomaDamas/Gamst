"""Search mode (owner decision 33: never miss the person, over-count rather than miss), with stand-ins: the live lane,
the person's identity, the run's command line and the switch over MCP. The verification lane is tested in
test_catch_up.SearchModeTest."""
import argparse
import asyncio
import contextlib
import io
import unittest

from gamst import Gamst
from gamst.counting import SEARCH, search_settings
from gamst.mcp import MCPServer
from gamst.pipeline import Pipeline

try:
    import numpy as np
except ImportError:
    np = None


class Embedder:
    model_id = 'stand-in'

    def __init__(self, vector):
        self.vector = vector

    def embed(self, samples, rate=16000):
        return self.vector


def word(vector, **options):
    from gamst.providers.speaker_verify import SpeakerVerify
    found = SpeakerVerify(Embedder(vector), words=True, context_ms=1500, segments=False, threshold=.5, **options)
    found.bind(type('P', (), dict(references={'voice': dict(kind='voice', value=dict(
        model='stand-in', embeddings=[[1., 0., 0.]]))}))())
    found._covers = lambda first, last: True
    found._span = lambda first, last: np.full(16000, .1, dtype=np.float32)
    return found, asyncio.run(found._word(dict(text='뽀또', start_ms=500, end_ms=800)))


@unittest.skipIf(np is None, 'numpy is needed')
class LiveLaneTest(unittest.TestCase):
    def test_without_a_later_lane_a_word_under_the_threshold_counts_as_a_candidate(self):
        _, count = word([.3, .95, 0.])                       # 0.3 < 0.5: another voice's in count mode
        self.assertFalse(count['match'])
        _, search = word([.3, .95, 0.], mode='search')
        self.assertEqual((search['match'], search['candidate'], search['reason']), (True, True, 'below_threshold'))
        _, sure = word([1., 0., 0.], mode='search')          # the person's: no candidate
        self.assertTrue(sure['match'])
        self.assertNotIn('candidate', sure)
        _, other = word([-.2, .98, 0.], mode='search')       # under the minimal floor (0): still another voice's
        self.assertFalse(other['match'])

    def test_with_the_instant_policy_only_the_minimal_floor_excludes(self):
        instant = dict(provisional=True, l1=dict(policy='instant', clear_other=.2))
        _, count = word([.1, .99, 0.], **instant)
        self.assertFalse(count['match'])                     # 0.1 < clear_other
        found, search = word([.1, .99, 0.], mode='search', **instant)
        self.assertTrue(search['match'])
        self.assertEqual(found.status(), dict(mode='search', l1_floor=0.))
        found.set_mode('count')
        self.assertEqual(found.status(), {})


class IdentityTest(unittest.TestCase):
    def test_a_face_that_does_not_contradict_selects_the_person(self):
        from gamst.providers.person_match import TargetPerson
        person = TargetPerson(Embedder([1., 0.]), face_threshold=.35, face_reject=.15, body_threshold=.6)
        self.assertEqual((person.face_threshold, person.body_threshold), (.35, .6))
        person.set_mode('search')
        self.assertEqual((person.face_threshold, person.body_threshold), (.15, SEARCH['body_threshold']))
        person.set_mode('count')
        self.assertEqual((person.face_threshold, person.body_threshold), (.35, .6))
        with self.assertRaises(ValueError):
            person.set_mode('loose')


class CommandLineTest(unittest.TestCase):
    def test_mode_and_its_parameters(self):
        from gamst.recipes import _mode
        self.assertEqual(_mode(argparse.Namespace()), ('count', None))
        mode, search = _mode(argparse.Namespace(mode='search', search_revoke_margin=.5, l1_clear_other=None))
        self.assertEqual((mode, search['revoke_margin'], search['l1_clear_other']), ('search', .5, 0.))
        self.assertEqual(_mode(argparse.Namespace(mode='search', l1_clear_other=.05))[1]['l1_clear_other'], .05)
        with self.assertRaises(SystemExit):
            _mode(argparse.Namespace(mode='search', search_revoke_margin=-1))
        with self.assertRaises(ValueError):
            search_settings(dict(threshold=.1))

    def test_both_counting_recipes_take_the_mode(self):
        from gamst.recipes import main
        for recipe in ('person-term-count', 'person-action-count'):
            with self.assertRaises(SystemExit) as stopped, contextlib.redirect_stderr(io.StringIO()):
                main(['run', recipe, '--mode', 'find'])
            self.assertEqual(stopped.exception.code, 2)      # argparse: not one of count, search


class LiveHudTest(unittest.TestCase):
    """The live window of a sound-only run draws a panel of its own with the revision states and the corrections."""

    def test_sound_only_panel_shows_states_corrections_mode_and_voice_warning(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('opencv is needed')
        from types import SimpleNamespace
        from unittest import mock
        from gamst.views import CountView, event_line
        pipeline = SimpleNamespace(name='person-term-count', mode='search', inputs={'microphone': dict(modality='audio')},
                                   voice_quality=dict(low=True, quality=dict(mean=.42), used='replaced'),
                                   lost=lambda capability: dict(gaps=0, frames=0),
                                   status=lambda: dict(modules={'asr': dict(enabled=True, last_latency_ms=3)}))
        view = CountView(pipeline, capabilities=('speech.term_events',))

        def record(index, event):
            return dict(id=f'r{index}', timestamp_ms=1000 * index, observation=dict(
                capability=dict(id='speech.term_events'), value=dict(count=1, events=[event])))
        records = [record(1, dict(event='term.spoken', state='heard', term='bbo', event_id='a')),
                   record(2, dict(event='term.spoken', state='provisional', term='bbo', event_id='b')),
                   record(3, dict(event='event.revision', state='final', revises='a', term='bbo', candidate=True,
                                  reason='uncertain_overlap')),
                   record(4, dict(event='event.revision', state='revoked', revises='b', term='bbo',
                                  reason='other_speaker'))]
        shown, texts = [], []
        real = view._put
        view._put = lambda panel, line, *rest: (texts.append(line), real(panel, line, *rest))[1]
        with mock.patch.object(cv2, 'imshow', lambda title, image: shown.append(image.shape)), \
                mock.patch.object(cv2, 'waitKey', lambda ms: 0), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(asyncio.run(view.update('microphone', SimpleNamespace(
                modality='audio', timestamp_ms=5000, data=None), records)))
            asyncio.run(view.update('microphone', SimpleNamespace(modality='audio', timestamp_ms=5050, data=None), []))
        self.assertEqual(len(shown), 1)          # at most every 100 ms
        for line in ('final 1', 'revoked 1', 'heard 0', 'uncertain 0', '-> final  bbo  candidate  (uncertain_overlap)',
                     '-> revoked  bbo  (other_speaker)'):
            self.assertTrue(any(line in text for text in texts), line)
        self.assertTrue(any('mode: search' in text for text in texts))
        self.assertTrue(any('VOICE REFERENCE QUALITY LOW' in text for text in texts))
        self.assertEqual(event_line(dict(event='term.other_voice', term='뽀또', speaker_score=.1)), '  other voice  뽀또  voice 0.10')


class Switchable:
    spec = None

    def __init__(self):
        self.modes = []

    def set_mode(self, mode, search=None):
        self.modes.append((mode, search and search['revoke_margin']))

    def process(self, frame):
        return None


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_the_mode_switches_over_mcp_and_the_pipeline_says_it(self):
        async with Gamst(consent=True) as human:
            server = MCPServer(human)
            pipeline = Pipeline(human, 'mode')
            pipeline.add_input('microphone', None, modality='audio')
            module = Switchable()
            pipeline.add_module('verification', module, consumes='microphone')
            self.assertEqual(pipeline.status()['mode'], 'count')
            pipeline.search = dict(revoke_margin=.4)
            switched = await server.call_tool('set_count_mode', dict(mode='search', actor='user'))
            self.assertEqual((switched['mode'], switched['modules'], switched['search']['revoke_margin']),
                             ('search', ['verification'], .4))
            status = await server.call_tool('get_pipeline', {})
            self.assertEqual((status['mode'], status['search']['revoke_margin'], len(status['mode_changes'])),
                             ('search', .4, 1))
            await server.call_tool('set_count_mode', dict(mode='count', actor='jev'))
            self.assertEqual(module.modes, [('search', .4), ('count', None)])
            self.assertEqual(pipeline.status()['mode'], 'count')
            with self.assertRaises(ValueError):
                await server.call_tool('set_count_mode', dict(mode='find', actor='user'))
            await pipeline.close()


if __name__ == '__main__':
    unittest.main()
