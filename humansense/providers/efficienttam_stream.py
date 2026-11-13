"""Bounded causal bridge to the pinned EfficientTAM video predictor.

This is an internal runtime component, not a verified model loader or registered
context provider. The caller owns capture continuity and exact-frame prompting.
"""
from collections import OrderedDict
import threading

UPSTREAM_REVISION = 'abcd061ebd3cc6e7527d152d75b890126aaa53f6'
RETAIN_NONCOND = 15


def _policy(predictor):
    expected = dict(image_size=512, num_maskmem=7,
        memory_temporal_stride_for_eval=1, max_obj_ptrs_in_encoder=16,
        training=False, only_obj_ptrs_in_the_past_for_eval=True,
        add_tpos_enc_to_obj_ptrs=False, clear_non_cond_mem_around_input=False,
        add_all_frames_to_correct_as_cond=False, fill_hole_area=0,
        non_overlap_masks=False)
    for name, value in expected.items():
        actual = getattr(predictor, name, None)
        if type(actual) is not type(value) or actual != value:
            raise ValueError('unsupported EfficientTAM policy: '+name)
    for name in ('_get_image_feature', 'add_new_points_or_box', 'propagate_in_video'):
        if not callable(getattr(predictor, name, None)):
            raise ValueError('missing EfficientTAM predictor method: '+name)


def _image(image):
    import numpy as np
    if (type(image) is not np.ndarray or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32 or
            max(image.shape[:2]) > 2048 or image.shape[0]*image.shape[1] > 1048576):
        raise ValueError('bounded uint8 BGR image required (32..2048 axes, at most 1 MP)')
    return image.copy(order='C')


def _normalize(image):
    """Match upstream JPEG-loader arithmetic without a lossy JPEG roundtrip."""
    import numpy as np
    import torch
    from PIL import Image
    rgb = Image.fromarray(image[:, :, ::-1]).resize((512, 512), Image.Resampling.BICUBIC)
    # Upstream divides uint8 in float64, then assigns into a float32 batch before
    # normalization. Preserve this order, including its rounding behavior.
    tensor = torch.from_numpy(np.array(rgb)/255.0).permute(2, 0, 1).float()
    tensor.sub_(torch.tensor([.485, .456, .406])[:, None, None])
    tensor.div_(torch.tensor([.229, .224, .225])[:, None, None])
    return tensor


class EfficientTAMStream:
    """One seed capture, 1..4 box prompts, forward-only consecutive frames.

    Owns predictor reference and inference state. No file/network/device capture
    operation is performed here. Pixel boxes are supplied only for the first
    frame; corrections/reseeding require reset and a fresh capture-bound prompt.
    """
    def __init__(self, predictor, *, max_frames=1000000):
        _policy(predictor)
        if type(max_frames) is not int or not 1 <= max_frames <= 1000000:
            raise ValueError('max_frames must be in 1..1000000')
        self.predictor = predictor
        self.max_frames = max_frames
        self.lock = threading.RLock()
        self.closed = False
        self.state = None
        self.index = 0
        self.shape = None
        self.ids = ()

    def _clear(self):
        self.state = None
        self.index = 0
        self.shape = None
        self.ids = ()

    def reset(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('EfficientTAM stream closed')
            self._clear()

    def close(self):
        with self.lock:
            self.closed = True
            self._clear()
            self.predictor = None

    def _start(self, image, boxes):
        import numpy as np
        h, w = image.shape[:2]
        if type(boxes) not in (tuple, list) or not 1 <= len(boxes) <= 4:
            raise ValueError('initial frame requires 1..4 pixel xyxy boxes')
        validated = []
        for box in boxes:
            if (type(box) not in (tuple, list) or len(box) != 4 or
                    any(type(x) not in (int, float) or not np.isfinite(x) for x in box)):
                raise ValueError('finite pixel xyxy box required')
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and x1-x0 >= 1 and y1-y0 >= 1):
                raise ValueError('box must lie inside initial capture and span at least one pixel')
            validated.append(np.asarray(box, dtype=np.float32))
        device = self.predictor.device
        self.state = dict(images={0: _normalize(image)}, num_frames=1,
            offload_video_to_cpu=True, offload_state_to_cpu=False,
            video_height=h, video_width=w, device=device, storage_device=device,
            point_inputs_per_obj={}, mask_inputs_per_obj={}, cached_features={}, constants={},
            obj_id_to_idx=OrderedDict(), obj_idx_to_id=OrderedDict(), obj_ids=[],
            output_dict_per_obj={}, temp_output_dict_per_obj={}, frames_tracked_per_obj={})
        self.shape = image.shape
        self.ids = tuple(range(1, len(validated)+1))
        self.predictor._get_image_feature(self.state, frame_idx=0, batch_size=1)
        for obj_id, box in zip(self.ids, validated):
            self.predictor.add_new_points_or_box(self.state, frame_idx=0, obj_id=obj_id,
                box=box, normalize_coords=True)

    def _prune(self):
        state = self.state
        # At frame t+1 the object-pointer path reads t..t-14 and the mask path
        # reads t..t-5. Retain their union plus the sole conditioning frame 0.
        cutoff = max(1, self.index-RETAIN_NONCOND+1)
        for output in state['output_dict_per_obj'].values():
            if set(output['cond_frame_outputs']) != {0}:
                raise ValueError('unexpected conditioning frame; reset required')
            for key in list(output['non_cond_frame_outputs']):
                if key < cutoff:
                    del output['non_cond_frame_outputs'][key]
        for tracked in state['frames_tracked_per_obj'].values():
            for key in list(tracked):
                if key != self.index:
                    del tracked[key]
        # Prompt inputs are needed only to construct the initial conditioning
        # output. Their small frame-zero records remain; no correction API.
        if any(any(outputs.values()) for outputs in state['temp_output_dict_per_obj'].values()):
            raise ValueError('unconsumed temporary outputs')
        state['images'].clear()
        state['cached_features'].clear()

    def step(self, image, *, boxes=None):
        """Return owned boolean N×H×W masks; indices are session-local IDs 1..N.

        Any error invalidates state. The caller must provide fresh seed boxes on
        the next call. This component cannot detect capture gaps on its own.
        """
        with self.lock:
            if self.closed:
                raise RuntimeError('EfficientTAM stream closed')
            try:
                import torch
                image = _image(image)
                _policy(self.predictor)
                if self.index >= self.max_frames:
                    raise ValueError('session frame limit reached; fresh prompt required')
                with torch.inference_mode():
                    if self.state is None:
                        self._start(image, boxes)
                    else:
                        if boxes is not None or image.shape != self.shape:
                            raise ValueError('correction or resized capture requires fresh session')
                        self.state['images'] = {self.index: _normalize(image)}
                        self.state['num_frames'] = self.index+1
                    iterator = self.predictor.propagate_in_video(self.state,
                        start_frame_idx=self.index, max_frame_num_to_track=0, reverse=False)
                    try:
                        item = next(iterator)
                        # Exhaust the one-frame generator before pruning its state.
                        exhausted = object()
                        if next(iterator, exhausted) is not exhausted:
                            raise ValueError('predictor returned more than the current frame')
                    finally:
                        iterator.close()
                    frame_idx, ids, masks = item
                    if (type(frame_idx) is not int or frame_idx != self.index or
                            tuple(ids) != self.ids or any(type(x) is not int for x in ids) or
                            not isinstance(masks, torch.Tensor) or
                            tuple(masks.shape) != (len(self.ids), 1, *self.shape[:2]) or
                            not masks.is_floating_point() or not torch.isfinite(masks).all().item()):
                        raise ValueError('invalid current-frame mask output')
                    result = (masks[:, 0] > 0).cpu().numpy().copy()
                    self._prune()
                    self.index += 1
                    return result
            except BaseException:
                self._clear()
                raise
