import unittest
from fractions import Fraction

from humansense.sam_audio_visual import align_visual_frames, visual_allocation_bounds


class VisualAlignmentTests(unittest.TestCase):
    def align(self, frames, **overrides):
        args = dict(audio_end_us=2000000, sample_count=48000, hop_length=4800,
                    camera_to_audio_offset_us=0, clock_error_us=0, max_age_us=150000)
        args.update(overrides)
        return align_visual_frames(frames, **args)

    def test_irregular_causal_selection_preserves_original_indices(self):
        frames = [(t,t) for t in (1000000,1080000,1190000,1300000,1390000,
                                  1500000,1570000,1690000,1800000,1890000)]
        result = self.align(frames)
        self.assertEqual(result.indices, tuple(range(10)))
        for index, time, age in zip(result.indices, result.feature_times_sixth_us,
                                    result.ages_sixth_us):
            self.assertEqual(age, time-frames[index][1]*6)
            self.assertGreaterEqual(age, 0)

    def test_exact_fractional_sample_times_and_offset(self):
        # A nonintegral-microsecond audio start cannot be rounded to the future.
        frames = [(t,t) for t in range(899900,1900001,40000)]
        result = self.align(frames, sample_count=48001, hop_length=1920,
                            camera_to_audio_offset_us=100000, max_age_us=50000)
        start = Fraction(2000000)-Fraction(48001*1000000,48000)
        for k,(idx,ts) in enumerate(zip(result.indices,result.feature_times_sixth_us)):
            actual = start+Fraction(k*1920*1000000,48000)
            self.assertEqual(Fraction(ts,6),actual)
            self.assertLessEqual(frames[idx][1]+100000, actual)
            if idx+1 < len(frames):
                self.assertGreater(frames[idx+1][1]+100000, actual)

    def test_uncertainty_requires_past_capture_and_bounded_worst_age(self):
        frames = [(t,t) for t in range(990000,1990001,100000)]
        result = self.align(frames, clock_error_us=10000)
        self.assertEqual(result.ages_sixth_us[0],60000)
        with self.assertRaises(ValueError):
            self.align([(1000000,1000000)]+frames[1:],clock_error_us=1)
        with self.assertRaises(ValueError):
            self.align(frames,clock_error_us=10000,max_age_us=15000)

    def test_missing_future_stale_and_regressing_are_rejected(self):
        for frames in ([(1000001,1000001)], [(900000,900000)],
                       [(1000000,1000000),(999999,999999)],
                       [(1000000,1000000),(1000000,1000000)],
                       [(1000000,1000000),(2000001,2000001)]):
            with self.subTest(frames=frames),self.assertRaises(ValueError):
                self.align(frames)

    def test_strict_types_and_feature_bound(self):
        for override in (dict(sample_count=True),dict(hop_length=1),
                         dict(audio_end_us=float('nan')),dict(clock_error_us=-1),
                         dict(camera_to_audio_offset_us=True),dict(max_age_us=250001)):
            with self.subTest(override=override),self.assertRaises(ValueError):
                self.align([(1000000,1000000)],**override)

    def test_expansion_accounts_for_upstream_float_resize(self):
        sizes=visual_allocation_bounds(frame_count=10,height=64,width=64,
                                      feature_count=25,image_size=224)
        self.assertEqual(sizes['preprocessing_tensor_bytes'],
                         10*4*64*64+2*25*3*64*64+25*3*224*224+4*25*3*(64*64+3*224*224))
        with self.assertRaises(ValueError):
            visual_allocation_bounds(frame_count=1,height=16,width=16,
                                     feature_count=96,image_size=336)
        # Small original frames do not make the full resized sequence cheap.
        with self.assertRaises(ValueError):
            visual_allocation_bounds(frame_count=1,height=16,width=16,
                                     feature_count=512,image_size=448)
        with self.assertRaises(ValueError):
            visual_allocation_bounds(frame_count=128,height=480,width=640,
                                     feature_count=1,image_size=16)

    def test_irregular_mapping_matches_independent_fraction_oracle(self):
        import random
        rng=random.Random(202)
        for _ in range(100):
            count=rng.randint(48000,60000)
            start=Fraction(2000000)-Fraction(count*1000000,48000)
            first=int(start)-1000
            ends=[first]
            while ends[-1]+50000 < 2000000:
                ends.append(ends[-1]+rng.randint(10000,50000))
            result=self.align([(t,t) for t in ends],sample_count=count,
                              hop_length=1920,clock_error_us=500,max_age_us=100000)
            expected=[]
            for k in range((count+1919)//1920):
                moment=start+Fraction(k*1920*1000000,48000)
                admitted=[i for i,t in enumerate(ends)
                          if t+500<=moment and moment-(t-500)<=100000]
                self.assertTrue(admitted)
                expected.append(max(admitted))
            self.assertEqual(result.indices,tuple(expected))


class PinnedProcessorVisualOracle(unittest.TestCase):
    def test_actual_pinned_tensor_selection_and_mask_polarity(self):
        import os
        source_path=os.environ.get('GAMST_SAM_PROCESSOR_SOURCE')
        if not source_path:
            self.skipTest('set GAMST_SAM_PROCESSOR_SOURCE for pinned source oracle')
        import ast
        import hashlib
        from pathlib import Path
        import torch
        source=Path(source_path).read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         '0d4af472dcbf05b9c9c2550ab4488d29a224d4b21556b3bbcd577137fcaff307')
        tree=ast.parse(source)
        load=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_video')
        processor=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Processor')
        mask=next(n for n in processor.body if isinstance(n,ast.FunctionDef) and n.name=='mask_videos')
        # Only type annotations are removed; function bodies are source-exact.
        for fn in (load,mask):
            fn.returns=None
            for arg in fn.args.args: arg.annotation=None
        namespace={'torch':torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[load,mask],type_ignores=[])),
                     'pinned_sam_processor_visual','exec'),namespace)
        for m in range(1,513):
            frames=torch.arange(m).reshape(m,1,1,1).expand(m,3,1,1)
            actual=namespace['load_video'](torch.tensor([m]),[frames],lambda k:k*1920,48000)[0]
            self.assertTrue(torch.equal(frames,actual),m)
        rgb=torch.tensor([[[[11,12]],[[21,22]],[[31,32]]]],dtype=torch.uint8)
        binary=torch.tensor([[[[0,1]]]],dtype=torch.uint8)
        actual=namespace['mask_videos'](None,[rgb],[binary])[0]
        self.assertEqual(actual.tolist(),[[[[11,0]],[[21,0]],[[31,0]]]])
