import unittest

try:
    import numpy as np
    import torch
    from PIL import Image
except ImportError:
    np = torch = Image = None

from humansense.providers.efficienttam_stream import EfficientTAMStream, _normalize


class Predictor:
    image_size = 512
    num_maskmem = 7
    memory_temporal_stride_for_eval = 1
    max_obj_ptrs_in_encoder = 16
    training = False
    only_obj_ptrs_in_the_past_for_eval = True
    add_tpos_enc_to_obj_ptrs = False
    clear_non_cond_mem_around_input = False
    add_all_frames_to_correct_as_cond = False
    fill_hole_area = 0
    non_overlap_masks = False

    def __init__(self):
        self.device = torch.device('cpu')
        self.failure = None
        self.finished = 0
        self.preceding = []

    def _get_image_feature(self, state, frame_idx, batch_size):
        assert torch.is_inference_mode_enabled()
        state['cached_features'] = {frame_idx: state['images'][frame_idx]}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box, normalize_coords):
        assert frame_idx == 0 and normalize_coords is True and box.shape == (4,)
        index = len(state['obj_ids'])
        state['obj_ids'].append(obj_id)
        state['obj_id_to_idx'][obj_id] = index
        state['obj_idx_to_id'][index] = obj_id
        state['point_inputs_per_obj'][index] = {0: box.copy()}
        state['mask_inputs_per_obj'][index] = {}
        state['output_dict_per_obj'][index] = {'cond_frame_outputs': {0: object()}, 'non_cond_frame_outputs': {}}
        state['temp_output_dict_per_obj'][index] = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        state['frames_tracked_per_obj'][index] = {}

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track, reverse):
        t = start_frame_idx
        assert max_frame_num_to_track == 0 and reverse is False
        assert state['num_frames'] == t+1 and set(state['images']) == {t}
        try:
            self._get_image_feature(state, t, 1)
            for i, output in state['output_dict_per_obj'].items():
                # Independent source-policy oracle: every prior memory address
                # requested by mask or pointer attention must still be present.
                selected = set(range(max(1, t-15), t))
                assert selected <= set(output['non_cond_frame_outputs'])
                assert set(output['cond_frame_outputs']) == {0}
                self.preceding.append((t, len(output['non_cond_frame_outputs'])))
                if t:
                    output['non_cond_frame_outputs'][t] = object()
                state['frames_tracked_per_obj'][i][t] = {'reverse': False}
            masks = torch.ones((len(state['obj_ids']), 1, state['video_height'], state['video_width']))
            masks[..., :5] = -1
            if self.failure == 'nan': masks[0, 0, 0, 0] = float('nan')
            yield t, state['obj_ids'], masks
            if self.failure == 'extra': yield t+1, state['obj_ids'], masks
            if self.failure == 'late': raise RuntimeError('late generator failure')
        finally:
            self.finished += 1


@unittest.skipIf(torch is None or Image is None, 'optional torch/numpy/Pillow unavailable')
class EfficientTAMStreaming(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor()
        self.stream = EfficientTAMStream(self.predictor)
        self.image = np.zeros((32, 45, 3), dtype=np.uint8)
        self.boxes = [[0, 0, 20, 30], [20, 0, 45, 32]]

    def tearDown(self):
        self.stream.close()

    def test_causal_retention_and_owned_masks(self):
        # Patch only image resizing to keep a long state-machine test inexpensive.
        from unittest.mock import patch
        with patch('humansense.providers.efficienttam_stream._normalize', return_value=torch.zeros(3, 2, 2)):
            for t in range(1000):
                masks = self.stream.step(self.image, boxes=self.boxes if t == 0 else None)
                self.assertEqual(masks.shape, (2, 32, 45))
                self.assertEqual(masks.dtype, np.bool_)
                self.assertFalse(masks[:, :, :5].any())
                self.assertTrue(masks[:, :, 5:].all())
                state = self.stream.state
                self.assertEqual(state['images'], {})
                self.assertEqual(state['cached_features'], {})
                for i, output in state['output_dict_per_obj'].items():
                    self.assertLessEqual(len(output['non_cond_frame_outputs']), 15)
                    self.assertEqual(set(state['frames_tracked_per_obj'][i]), {t})
                    self.assertEqual(set(state['point_inputs_per_obj'][i]), {0})
        self.assertEqual(self.predictor.finished, 1000)
        self.assertFalse(self.image.any())

    def test_preprocess_matches_upstream_arithmetic(self):
        rng = np.random.default_rng(5)
        image = rng.integers(0, 256, (33, 47, 3), dtype=np.uint8)
        resized = np.array(Image.fromarray(image[:, :, ::-1]).convert('RGB').resize((512, 512)))
        batch = torch.zeros(1, 3, 512, 512, dtype=torch.float32)
        batch[0] = torch.from_numpy(resized/255.0).permute(2, 0, 1)
        batch -= torch.tensor([.485, .456, .406], dtype=torch.float32)[:, None, None]
        batch /= torch.tensor([.229, .224, .225], dtype=torch.float32)[:, None, None]
        self.assertTrue(torch.equal(_normalize(image), batch[0]))
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            self.assertTrue(torch.equal(_normalize(image), batch[0]))
        finally:
            torch.set_default_dtype(original_dtype)

    def test_failures_drop_all_state_and_close_generator(self):
        for failure in ('nan', 'extra', 'late'):
            self.predictor.failure = failure
            with self.assertRaises((ValueError, RuntimeError)):
                self.stream.step(self.image, boxes=self.boxes)
            self.assertIsNone(self.stream.state)
            self.assertEqual(self.stream.index, 0)
        self.assertEqual(self.predictor.finished, 3)
        self.predictor.failure = None
        with self.assertRaises(ValueError): self.stream.step(self.image)
        self.stream.step(self.image, boxes=self.boxes)

    def test_seed_validation_and_no_stale_reseed(self):
        invalid = (None, [], self.boxes*3, [[0, 0, 46, 32]], [[True, 0, 20, 30]],
                   [[0, 0, float('nan'), 30]], [[0, 0, .5, 20]])
        for boxes in invalid:
            with self.assertRaises(ValueError): self.stream.step(self.image, boxes=boxes)
            self.assertIsNone(self.stream.state)
        self.stream.step(self.image, boxes=self.boxes)
        with self.assertRaises(ValueError): self.stream.step(self.image, boxes=self.boxes)
        with self.assertRaises(ValueError): self.stream.step(self.image)
        self.stream.step(self.image, boxes=self.boxes)
        with self.assertRaises(ValueError): self.stream.step(np.zeros((33, 45, 3), dtype=np.uint8))
        self.assertIsNone(self.stream.state)

    def test_session_limit_reset_and_terminal_close(self):
        stream = EfficientTAMStream(Predictor(), max_frames=1)
        stream.step(self.image, boxes=self.boxes)
        with self.assertRaises(ValueError): stream.step(self.image)
        stream.step(self.image, boxes=self.boxes)
        stream.reset()
        self.assertIsNone(stream.state)
        with self.assertRaises(ValueError): stream.step(self.image)
        stream.close(); stream.close()
        with self.assertRaises(RuntimeError): stream.reset()
        with self.assertRaises(RuntimeError): stream.step(self.image, boxes=self.boxes)

    def test_policy_drift_and_input_bounds(self):
        for name, value in (('training', True), ('memory_temporal_stride_for_eval', 2),
                ('max_obj_ptrs_in_encoder', 17), ('add_tpos_enc_to_obj_ptrs', True),
                ('clear_non_cond_mem_around_input', True), ('image_size', 1024)):
            predictor = Predictor(); setattr(predictor, name, value)
            with self.assertRaises(ValueError): EfficientTAMStream(predictor)
        for image in (None, np.zeros((32, 45), dtype=np.uint8), self.image.astype(np.float32),
                      np.zeros((2049, 32, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError): self.stream.step(image, boxes=self.boxes)
        self.stream.step(self.image, boxes=self.boxes)
        self.predictor.training = True
        with self.assertRaises(ValueError): self.stream.step(self.image)
        self.assertIsNone(self.stream.state)


if __name__ == '__main__':
    unittest.main()
