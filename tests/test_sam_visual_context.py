import unittest
from dataclasses import replace
import test_sam_visual_window as windows
import test_sam_audio_context as audio
from humansense.sam_visual_packet import visual_policy,unpack_visual_packet
from humansense.providers.sam_audio_context import SAMAudioContext


class Backend(audio.FixtureBackend):
    def visual_parameters(self):return dict(image_size=224,batch_size=16)
    def predict(self,window,*,visual=None):
        self.visual=visual
        return super().predict(window)


@unittest.skipIf(windows.fixtures.np is None,'optional numpy missing')
class VisualContext(unittest.TestCase):
    def setUp(self):
        fixture=windows.Window();fixture.setUp()
        self.packet=fixture.fixture.packet()
        policy=visual_policy(unpack_visual_packet(self.packet.data)[0])
        self.config=dict(audio.CONFIG,prompt='',visual_policy=policy)
        self.backend=Backend()
        self.context=SAMAudioContext(self.backend,self.config)

    def tearDown(self):self.context.close()

    def test_actual_context_consumes_aligned_visual_and_empty_description(self):
        pair=self.context.process(self.packet)
        self.assertEqual(self.backend.last.prompt,'')
        self.assertEqual(len(self.backend.visual.indices),25)
        self.assertEqual(pair.observation['prompt']['utf8_bytes'],0)
        self.assertEqual(pair.observation['prompt']['visual']['mask_polarity'],'zero-kept')
        self.assertFalse(pair.observation['prompt']['visual']['clock_verified'])
        self.assertEqual(len(pair.target),48000*4)

    def test_bad_capture_never_reaches_backend(self):
        with self.assertRaises(ValueError):
            self.context.process(replace(self.packet,timestamp_ms=2001))
        self.assertFalse(hasattr(self.backend,'visual'))

    def test_reset_clears_both_histories(self):
        self.context.process(self.packet)
        self.context.reset()
        pair=self.context.process(self.packet)
        self.assertEqual(pair.observation['window']['discontinuity_reason'],'reset')
