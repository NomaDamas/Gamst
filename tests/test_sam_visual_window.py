import copy
import unittest
import test_sam_visual_packet as fixtures
from humansense.sam_visual_packet import visual_frame,unpack_visual_packet,visual_policy,validate_visual_policy
from humansense.sam_visual_window import prepare_visual


@unittest.skipIf(fixtures.np is None,'optional numpy missing')
class Window(unittest.TestCase):
    def setUp(self):
        f=fixtures.VisualPacketTests(); f.setUp(); self.fixture=f
        f.rgb=fixtures.np.zeros((10,3,16,16),dtype=fixtures.np.uint8)
        f.masks=fixtures.np.zeros((10,1,16,16),dtype=fixtures.np.uint8)
        f.kw['intervals_us']=[[t,t] for t in range(1000000,2000000,100000)]
        f.kw['sequences']=list(range(10,20))
        self.packet=f.packet().data
        self.policy=visual_policy(unpack_visual_packet(self.packet)[0])
        self.args=dict(policy=self.policy,sample_count=48000,hop_length=1920,image_size=224,batch_size=16)

    def test_consumer_admission_keeps_original_pixels_and_causal_indices(self):
        result=prepare_visual(self.packet,**self.args)
        self.assertEqual(result.shape,(10,3,16,16))
        self.assertEqual(result.indices,tuple((k*40000)//100000 for k in range(25)))
        rgb,masks=result.arrays(); rgb[:]=1; masks[:]=1
        self.assertEqual(result.rgb,self.fixture.rgb.tobytes())
        self.assertEqual(result.masks,self.fixture.masks.tobytes())
        self.assertEqual(len(result.mapping_sha256),64)

    def test_actual_model_dimensions_and_source_samples_are_admitted(self):
        for change in (dict(sample_count=48001),dict(image_size=1024),dict(batch_size=0),
                       dict(batch_size=True),dict(hop_length=1)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                prepare_visual(self.packet,**dict(self.args,**change))

    def test_strict_policy_and_clock_coverage(self):
        for change in (lambda p:p['clock'].update(clock_error_us=True),
                       lambda p:p.update(width=16.0),
                       lambda p:p.update(mask_polarity='one-kept')):
            bad=copy.deepcopy(self.policy);change(bad)
            with self.assertRaises(ValueError):validate_visual_policy(bad)
        self.fixture.kw['clock']['clock_error_us']=1
        uncertain=self.fixture.packet().data
        policy=visual_policy(unpack_visual_packet(uncertain)[0])
        # First exposure ends at first feature, so nonzero uncertainty cannot
        # conservatively guarantee that it is a past observation.
        with self.assertRaises(ValueError):
            prepare_visual(uncertain,**dict(self.args,policy=policy))
