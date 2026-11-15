"""Optional bounded SAM3.1 Object Multiplex video-masklet provider.

Official source: SAM3.1 revision f664253c90122f5d968a17c3f75a62893b4e15e7,
custom SAM License; gated Hugging Face model facebook/sam3.1. The adapter uses
bounded native clip sessions, not sampled SAM3 images or EfficientTAM defaults.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading

from ..capabilities import CapabilitySpec, OutputField
from .yoloe import _image, mask_context, prompt_labels

UPSTREAM_REVISION = 'f664253c90122f5d968a17c3f75a62893b4e15e7'
MODEL_ID = 'facebook/sam3.1'


def _hash(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('lowercase SAM3.1 checkpoint SHA256 required')
    return value


def configuration(config):
    required = {'source_dir', 'path', 'sha256'}
    optional = {'prompt', 'max_frames', 'max_gap_ms', 'max_objects', 'max_track_gap', 'device'}
    if (type(config) is not dict or not required <= set(config) or set(config)-required-optional or
            any(type(config[key]) is not str or not Path(config[key]).is_absolute()
                for key in ('source_dir', 'path'))):
        raise ValueError('SAM3.1 requires absolute local source/checkpoint paths and SHA256')
    _hash(config['sha256'])
    prompt = prompt_labels([config.get('prompt', 'person')])[0]
    device = config.get('device', 'cuda')
    if device != 'cuda':
        raise ValueError('official SAM3.1 multiplex runtime requires explicit cuda')
    args = dict(checkpoint_sha256=config['sha256'], prompt=prompt, device=device,
        max_frames=config.get('max_frames', 8), max_gap_ms=config.get('max_gap_ms', 1000),
        max_objects=config.get('max_objects', 4), max_track_gap=config.get('max_track_gap'))
    SAM31VideoContext.capability(**args)
    return args


def _verify_checkpoint(path, digest):
    _hash(digest)
    path = Path(path).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError('absolute existing local SAM3.1 checkpoint required')
    size = path.stat().st_size
    if not 0 < size <= 8*1024**3:
        raise ValueError('SAM3.1 checkpoint exceeds the 8GiB local-artifact bound')
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        while True:
            block = source.read(8*1024*1024)
            if not block:
                break
            hasher.update(block)
    if hasher.hexdigest() != digest:
        raise ValueError('SAM3.1 checkpoint SHA256 mismatch')
    return path


class SAM31VideoContext:
    """Bounded per-stream clip state and synchronous native backpressure.

    A result is emitted after each full chunk (or ``finish``). Earlier frames in
    that chunk necessarily use later frames during native propagation; outputs
    are therefore timestamped masklets, not causal live identities.
    """
    def __init__(self, backend, *, checkpoint_sha256, prompt='person', device='cuda',
                 max_frames=8, max_gap_ms=1000, max_objects=4, max_track_gap=None,
                 artifact_verified=False):
        if max_track_gap is None:
            max_track_gap = min(2, max_frames-1)
        self.spec = self.capability(checkpoint_sha256=checkpoint_sha256, prompt=prompt,
            device=device, max_frames=max_frames, max_gap_ms=max_gap_ms,
            max_objects=max_objects, max_track_gap=max_track_gap)
        if any(not callable(getattr(backend, name, None)) for name in ('infer', 'reset', 'close')):
            raise ValueError('owned SAM3.1 backend with infer/reset/close required')
        if type(artifact_verified) is not bool:
            raise ValueError('artifact_verified must be boolean')
        self.backend, self.verified = backend, artifact_verified
        self.prompt, self.max_frames, self.max_gap_ms = prompt, max_frames, max_gap_ms
        self.max_objects, self.max_track_gap = max_objects, max_track_gap
        self.lock = threading.RLock()
        self.closed = False
        self.owner = self.last = None
        self.buffer = []
        self.processed = self.dropped = self.chunks = 0

    @staticmethod
    def capability(*, checkpoint_sha256, prompt='person', device='cuda', max_frames=8,
                   max_gap_ms=1000, max_objects=4, max_track_gap=None):
        _hash(checkpoint_sha256)
        prompt = prompt_labels([prompt])[0]
        if device != 'cuda':
            raise ValueError('SAM3.1 multiplex provider supports only explicit cuda')
        if type(max_frames) is not int or not 1 <= max_frames <= 16:
            raise ValueError('SAM3.1 max_frames must be 1..16')
        if max_track_gap is None:
            max_track_gap = min(2, max_frames-1)
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 60000:
            raise ValueError('SAM3.1 max_gap_ms must be 1..60000')
        if type(max_objects) is not int or not 1 <= max_objects <= 4:
            raise ValueError('SAM3.1 max_objects must be 1..4 for bounded context')
        if type(max_track_gap) is not int or not 0 <= max_track_gap < max_frames:
            raise ValueError('SAM3.1 max_track_gap must be in 0..max_frames-1')
        version = hashlib.sha256(json.dumps([UPSTREAM_REVISION, checkpoint_sha256,
            prompt, device, max_frames, max_gap_ms, max_objects, max_track_gap]).encode()).hexdigest()
        return CapabilitySpec('vision.sam3_1.video_masklets', 'vision.mask_tracks',
            'video', 'sam3.1.multiplex.video', '1.'+version,
            (OutputField('objects', 'array'), OutputField('summary', 'object')),
            'window', window_ms=max_frames*max_gap_ms, stateful=True, reset_on_gap=True,
            freshness_ms=60000, interpretation='estimate', devices=('cuda',),
            sensor_requirements=('uint8-bgr-camera', 'ordered-continuous-capture', 'text-prompt'),
            license='adapter:Apache-2.0;SAM3.1-code-and-model:custom-SAM-License;gated-HF-asset')

    def _clear_chunk(self):
        self.buffer = []
        self.owner = self.last = None

    def reset(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('SAM3.1 provider is closed')
            self.dropped += len(self.buffer)
            self._clear_chunk()
            self.backend.reset()

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.dropped += len(self.buffer)
                self._clear_chunk()
                backend, self.backend = self.backend, None
                backend.close()

    def _summary(self, status, **values):
        result = dict(status=status, prompt=self.prompt, processed_frames=self.processed,
            dropped_buffered_frames=self.dropped, completed_chunks=self.chunks,
            buffered_frames=len(self.buffer), backpressure='synchronous_bounded_chunk_inference',
            causal=False, complete_snapshot=False, identity_verified=False,
            track_ids_chunk_local=True, artifact_integrity_verified=self.verified,
            model_quality_verified=False, source_revision=UPSTREAM_REVISION,
            model_id=MODEL_ID, native_frame_bound=self.max_frames,
            native_object_bound=self.max_objects)
        result.update(values)
        return result

    def process(self, frame):
        with self.lock:
            if self.closed:
                raise RuntimeError('SAM3.1 provider is closed')
            image = _image(frame.data)
            if frame.modality != 'video' or type(frame.timestamp_ms) is not int or frame.timestamp_ms < 0:
                raise ValueError('SAM3.1 requires timestamped video frames')
            sensor = frame.sensor
            sequence = None if sensor is None else (
                sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence)
            owner = (frame.subject_id, None if sensor is None else sensor.source_id,
                None if sensor is None else sensor.sensor_id,
                None if sensor is None else sensor.stream_epoch,
                None if sensor is None else sensor.coordinate_frame,
                None if sensor is None else sensor.calibration_id, image.shape)
            if self.last is not None and owner == self.owner and frame.timestamp_ms <= self.last[0]:
                raise ValueError('SAM3.1 capture timestamps must increase')
            discontinued = bool(self.last is not None and (owner != self.owner or
                frame.timestamp_ms-self.last[0] > self.max_gap_ms or
                (sequence is not None and sequence != self.last[1]+1)))
            reset = discontinued
            if discontinued:
                self.dropped += len(self.buffer)
                self._clear_chunk()
            if sequence is None:
                sequence = 0 if self.last is None else self.last[1]+1
            self.buffer.append((image, frame.timestamp_ms, sequence))
            self.owner, self.last = owner, (frame.timestamp_ms, sequence)
            self.processed += 1
            if len(self.buffer) < self.max_frames:
                return self.spec.validate(dict(objects=[], summary=self._summary('buffering',
                    stream_reset=reset, capture_timestamp_ms=frame.timestamp_ms,
                    capture_sequence=sequence)))
            return self._run_chunk(stream_reset=reset)

    def finish(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('SAM3.1 provider is closed')
            if not self.buffer:
                return None
            return self._run_chunk(stream_reset=False)

    def _run_chunk(self, *, stream_reset):
        records, self.buffer = self.buffer, []
        chunk_id = self.chunks+1
        prefix = f'sam31-chunk-{chunk_id}:'
        try:
            outputs = self.backend.infer([record[0] for record in records],
                prompt=self.prompt, max_objects=self.max_objects)
            import numpy as np
            if type(outputs) is not list or len(outputs) != len(records):
                raise ValueError('SAM3.1 must return one output per buffered frame')
            active = OrderedDict()
            retired = OrderedDict()
            expired = []
            created = 0
            for index, (output, record) in enumerate(zip(outputs, records)):
                if type(output) is not dict or set(output) != {'obj_ids', 'masks'}:
                    raise ValueError('SAM3.1 output requires obj_ids and masks')
                ids, masks = output['obj_ids'], output['masks']
                if (type(ids) is not list or len(ids) > self.max_objects or
                        any(type(obj_id) is not int or obj_id < 0 for obj_id in ids) or
                        len(set(ids)) != len(ids) or not isinstance(masks, np.ndarray) or
                        masks.dtype != np.bool_ or masks.shape != (len(ids), *record[0].shape[:2])):
                    raise ValueError('SAM3.1 returned invalid bounded boolean masks')
                current = set(ids)
                for obj_id, track in list(active.items()):
                    if obj_id not in current:
                        track['missing'] += 1
                        if track['missing'] > self.max_track_gap:
                            track['status'] = 'expired'
                            track['eviction_timestamp_ms'] = record[1]
                            expired.append(track['track_id'])
                            retired[obj_id] = active.pop(obj_id)
                for position, obj_id in enumerate(ids):
                    if obj_id in retired or (obj_id not in active and created >= self.max_objects):
                        continue
                    if obj_id not in active:
                        created += 1
                        active[obj_id] = dict(track_id=prefix+str(obj_id), label=self.prompt,
                            native_object_id=obj_id, status='tracking', frames=[], missing=0,
                            first_timestamp_ms=record[1], last_timestamp_ms=record[1])
                    track = active[obj_id]
                    track['missing'] = 0
                    track['last_timestamp_ms'] = record[1]
                    mask = masks[position]
                    if not mask.any():
                        continue
                    ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
                    height, width = mask.shape
                    track['frames'].append(dict(timestamp_ms=record[1], capture_sequence=record[2],
                        mask=mask_context(mask), location=dict(unit='normalized_image', bbox=dict(
                            x=int(xs[0])/width, y=int(ys[0])/height,
                            width=int(xs[-1]-xs[0]+1)/width, height=int(ys[-1]-ys[0]+1)/height))))
            objects = []
            for track in [*active.values(), *retired.values()]:
                if not track['frames']:
                    continue
                track['status'] = 'session_evicted' if track['status'] == 'tracking' else track['status']
                track.pop('missing', None)
                objects.append(track)
            self.chunks += 1
            start, end = records[0][1], records[-1][1]
            result = self.spec.validate(dict(objects=objects, summary=self._summary('estimated',
                stream_reset=stream_reset, chunk_id=chunk_id, chunk_start_timestamp_ms=start,
                chunk_end_timestamp_ms=end, capture_timestamp_ms=records[-1][1],
                capture_sequence=records[-1][2], lookahead_ms=end-start,
                expired_track_ids=expired, session_evicted_track_ids=[track['track_id'] for track in objects],
                native_session_evicted=True)))
            self._clear_chunk()
            return result
        except BaseException:
            self._clear_chunk()
            self.backend.reset()
            raise


class _NativeSAM31:
    def __init__(self, predictor):
        self.predictor = predictor
        self.session_id = None

    @staticmethod
    def _array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return value

    def infer(self, images, *, prompt, max_objects):
        import numpy as np
        from PIL import Image
        if self.predictor is None or self.session_id is not None:
            raise RuntimeError('SAM3.1 native session state is invalid')
        frames = [Image.fromarray(image[:, :, ::-1], mode='RGB') for image in images]
        response = self.predictor.handle_request(request=dict(type='start_session',
            resource_path=frames, offload_video_to_cpu=True))
        self.session_id = response.get('session_id')
        try:
            if type(self.session_id) is not str:
                raise ValueError('SAM3.1 native session ID missing')
            self.predictor.handle_request(request=dict(type='add_prompt',
                session_id=self.session_id, frame_index=0, text=prompt))
            responses = {}
            for response in self.predictor.handle_stream_request(request=dict(
                    type='propagate_in_video', session_id=self.session_id,
                    propagation_direction='forward', start_frame_index=0,
                    max_frame_num_to_track=len(frames)-1)):
                index = response.get('frame_index')
                if type(index) is not int or index in responses or not 0 <= index < len(frames):
                    raise ValueError('SAM3.1 returned duplicate/out-of-range frame output')
                responses[index] = response.get('outputs')
            if set(responses) != set(range(len(frames))):
                raise ValueError('SAM3.1 propagation did not cover the bounded chunk')
            result = []
            for index in range(len(frames)):
                output = responses[index]
                if type(output) is not dict or 'out_obj_ids' not in output or 'out_binary_masks' not in output:
                    raise ValueError('SAM3.1 native masklet fields missing')
                ids = self._array(output['out_obj_ids'])
                masks = self._array(output['out_binary_masks'])
                if not isinstance(ids, np.ndarray) or ids.ndim != 1 or len(ids) > max_objects:
                    raise ValueError('SAM3.1 native object IDs invalid')
                if not isinstance(masks, np.ndarray) or masks.dtype != np.bool_:
                    raise ValueError('SAM3.1 native binary masks required')
                result.append(dict(obj_ids=[int(value) for value in ids], masks=masks))
            return result
        finally:
            self.reset()

    def reset(self):
        predictor, session_id = self.predictor, self.session_id
        self.session_id = None
        if predictor is not None and session_id is not None:
            predictor.handle_request(request=dict(type='close_session',
                session_id=session_id, run_gc_collect=False))

    def close(self):
        self.reset()
        predictor, self.predictor = self.predictor, None
        if predictor is not None:
            predictor.shutdown()


def load_worker(config):
    args = configuration(config)
    if (os.environ.get('GAMST_PROVIDER_WORKER') != '1' or
            any(key == 'sam3' or key.startswith('sam3.') for key in sys.modules)):
        raise RuntimeError('SAM3.1 requires a fresh owned ProcessProvider worker')
    if sys.version_info < (3, 12):
        raise RuntimeError('official SAM3.1 runtime requires Python 3.12+')
    source_dir = Path(config['source_dir']).expanduser().resolve(strict=True)
    if not source_dir.is_dir() or not (source_dir/'sam3'/'model_builder.py').is_file():
        raise ValueError('local pinned SAM3.1 source checkout required')
    checkpoint = _verify_checkpoint(config['path'], config['sha256'])
    from importlib.metadata import PackageNotFoundError, version
    try:
        if version('torch').split('+')[0] != '2.10.0':
            raise RuntimeError('SAM3.1 runtime requires torch==2.10.0')
        for package in ('torchvision', 'timm', 'ftfy', 'iopath', 'huggingface-hub', 'Pillow'):
            version(package)
    except PackageNotFoundError as exc:
        raise RuntimeError('select the documented isolated SAM3.1 runtime') from exc
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(source_dir))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('selected SAM3.1 CUDA device unavailable')
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(checkpoint),
        max_num_objects=args['max_objects'], multiplex_count=args['max_objects'],
        use_fa3=False, use_rope_real=True, compile=False, warm_up=False,
        session_expiration_sec=0, async_loading_frames=False)
    return SAM31VideoContext(_NativeSAM31(predictor), artifact_verified=True, **args)
