"""Optional Video Depth Anything streaming temporal-depth worker (Small or Large).

Pinned upstream: CVPR 2025 Video Depth Anything, revision
4f5ae23172ba60fd7bc11ef671cca678842c7072. Small code/weights are Apache-2.0.
Large weights are CC-BY-NC-4.0: selectable only as the explicit ``large``
variant, and the capability license says so (research/non-commercial use).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import threading

from ..capabilities import CapabilitySpec, OutputField

UPSTREAM_REVISION = '4f5ae23172ba60fd7bc11ef671cca678842c7072'
PAPER = 'Chen et al., Video Depth Anything, CVPR 2025, arXiv:2501.12375'
# Official model configs (upstream run.py) and checkpoint size bounds.
VARIANTS = {
    'small': dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384], limit=256*1024*1024,
                  license='adapter:Apache-2.0;VDA-S-code-and-model:Apache-2.0'),
    'large': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], limit=2048*1024*1024,
                  license='adapter:Apache-2.0;VDA-L-model:CC-BY-NC-4.0(non-commercial research only)'),
}
# Blackwell (sm_120) needs CUDA 12.8 builds; torch 2.1.1 cannot run there.
RUNTIME = (('torch', '2.8.0'), ('torchvision', '0.23.0'))


def _hash(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('lowercase checkpoint SHA256 required')
    return value


def configuration(config):
    required = {'source_dir', 'path', 'sha256'}
    optional = {'device', 'grid_size', 'max_gap_ms', 'max_frames', 'variant', 'fp32'}
    if (type(config) is not dict or not required <= set(config) or set(config)-required-optional or
            any(type(config[key]) is not str or not Path(config[key]).is_absolute()
                for key in ('source_dir', 'path'))):
        raise ValueError('VDA requires absolute local source/checkpoint paths and SHA256')
    _hash(config['sha256'])
    device = config.get('device', 'cpu')
    if type(device) is not str or device not in ('cpu', 'cuda'):
        raise ValueError('VDA device must be explicit cpu or cuda')
    args = dict(checkpoint_sha256=config['sha256'], device=device,
        grid_size=config.get('grid_size', 3), max_gap_ms=config.get('max_gap_ms', 1000),
        max_frames=config.get('max_frames', 1000000), variant=config.get('variant', 'small'),
        fp32=config.get('fp32', True))
    VideoDepthAnythingContext.capability(**args)
    return args


def _read(path, digest, limit):
    _hash(digest)
    path = Path(path).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError('absolute existing local VDA-S checkpoint required')
    with path.open('rb') as source:
        payload = source.read(limit+1)
    if not 0 < len(payload) <= limit or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('VDA-S checkpoint size/SHA256 mismatch')
    return payload


class VideoDepthAnythingContext:
    """One forward-only native VDA cache; no internal inference queue."""
    def __init__(self, predictor, *, checkpoint_sha256, device='cpu', grid_size=3,
                 max_gap_ms=1000, max_frames=1000000, variant='small', fp32=True, artifact_verified=False):
        self.spec = self.capability(checkpoint_sha256=checkpoint_sha256, device=device,
            grid_size=grid_size, max_gap_ms=max_gap_ms, max_frames=max_frames, variant=variant, fp32=fp32)
        if not callable(predictor) or not callable(getattr(predictor, 'reset', None)):
            raise ValueError('owned streaming VDA predictor with reset required')
        if type(artifact_verified) is not bool:
            raise ValueError('artifact_verified must be boolean')
        self.predictor = predictor
        self.grid_size, self.max_gap_ms, self.max_frames = grid_size, max_gap_ms, max_frames
        self.verified = artifact_verified
        self.lock = threading.RLock()
        self.closed = False
        self._clear()
        self.resets = 0

    @staticmethod
    def capability(*, checkpoint_sha256, device='cpu', grid_size=3,
                   max_gap_ms=1000, max_frames=1000000, variant='small', fp32=True):
        _hash(checkpoint_sha256)
        if variant not in VARIANTS:
            raise ValueError('VDA variant must be small or large')
        if type(fp32) is not bool:
            raise ValueError('fp32 must be boolean')
        if type(device) is not str or device not in ('cpu', 'cuda'):
            raise ValueError('VDA device must be cpu or cuda')
        if type(grid_size) is not int or not 1 <= grid_size <= 8:
            raise ValueError('depth grid_size must be 1..8')
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 60000:
            raise ValueError('max_gap_ms must be 1..60000')
        if type(max_frames) is not int or not 1 <= max_frames <= 1000000:
            raise ValueError('max_frames must be 1..1000000')
        identity = [UPSTREAM_REVISION, checkpoint_sha256, device, grid_size, max_gap_ms, max_frames]
        if (variant, fp32) != ('small', True):
            identity += [variant, fp32]
        version = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        return CapabilitySpec('vision.video_depth_anything.'+variant, 'vision.relative_depth',
            'video', 'video_depth_anything.'+variant+'.streaming', '1.'+version,
            (OutputField('grid', 'object'), OutputField('summary', 'object')),
            'native', stateful=True, reset_on_gap=True, freshness_ms=1000,
            interpretation='estimate', devices=(device,),
            sensor_requirements=('uint8-bgr-camera', 'ordered-continuous-capture'),
            license=VARIANTS[variant]['license'])

    def _clear(self):
        self.owner = self.last = None
        self.frames = 0

    def _reset(self):
        self._clear()
        self.predictor.reset()
        self.resets += 1

    def reset(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('VDA-S provider is closed')
            self._reset()

    def process(self, frame):
        with self.lock:
            if self.closed:
                raise RuntimeError('VDA-S provider is closed')
            import numpy as np
            data = frame.data
            # Wider-than-16:9 frames are valid: upstream infer_video_depth_one shrinks
            # input_size itself for ratio > 1.78 (memory recommendation only).
            if (frame.modality != 'video' or not isinstance(data, np.ndarray) or
                    data.dtype != np.uint8 or data.ndim != 3 or data.shape[2] != 3 or
                    not 32 <= data.shape[0] <= 2160 or not 32 <= data.shape[1] <= 3840 or
                    data.shape[0]*data.shape[1] > 1920*1080 or
                    max(data.shape[:2])/min(data.shape[:2]) > 4):
                raise ValueError('VDA requires uint8 BGR, 32px minimum, at most 2MP and 4:1 aspect ratio')
            if type(frame.timestamp_ms) is not int or frame.timestamp_ms < 0:
                raise ValueError('VDA-S timestamp must be a nonnegative integer')
            sensor = frame.sensor
            sequence = None
            if sensor is not None:
                sequence = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
            owner = (frame.subject_id, None if sensor is None else sensor.source_id,
                None if sensor is None else sensor.sensor_id,
                None if sensor is None else sensor.stream_epoch,
                None if sensor is None else sensor.coordinate_frame,
                None if sensor is None else sensor.calibration_id, data.shape)
            if self.last is not None and owner == self.owner and frame.timestamp_ms <= self.last[0]:
                raise ValueError('VDA-S capture timestamps must increase')
            discontinued = bool(self.last is not None and (owner != self.owner or
                frame.timestamp_ms-self.last[0] > self.max_gap_ms or
                (sequence is not None and sequence != self.last[1]+1)))
            if self.frames >= self.max_frames:
                discontinued = True
            reset = discontinued
            if discontinued:
                self._reset()
            try:
                depth = np.asarray(self.predictor(data[:, :, ::-1].copy()))
                if (depth.ndim != 2 or depth.dtype.kind not in 'fi' or
                        tuple(depth.shape) != tuple(data.shape[:2]) or
                        not np.isfinite(depth).all() or (depth < 0).any() or (depth > 1e6).any()):
                    raise ValueError('VDA-S must return an original-size finite nonnegative scalar map')
                depth = depth.astype(np.float64)
                low, high = np.percentile(depth, [5, 95])
                usable = high-low > max(1e-6, abs(high)*1e-6)
                scores = []
                rows = np.linspace(0, depth.shape[0], self.grid_size+1, dtype=int)
                columns = np.linspace(0, depth.shape[1], self.grid_size+1, dtype=int)
                if usable:
                    normalized = np.clip((depth-low)/(high-low), 0, 1)
                    for row in range(self.grid_size):
                        for column in range(self.grid_size):
                            scores.append(float(np.median(normalized[rows[row]:rows[row+1], columns[column]:columns[column+1]])))
                else:
                    scores = [None]*(self.grid_size**2)
                self.frames += 1
                if sequence is None:
                    sequence = self.frames-1 if self.last is None else self.last[1]+1
                result = self.spec.validate(dict(grid=dict(rows=self.grid_size,
                    columns=self.grid_size, order='row_major_top_left', relative_nearness=scores,
                    x_edges=(columns/depth.shape[1]).tolist(), y_edges=(rows/depth.shape[0]).tolist()),
                    summary=dict(status='estimated' if usable else 'unavailable_low_spread',
                    semantics='larger_is_relatively_nearer', units='dimensionless',
                    normalization='per_frame_p05_p95_clipped', aggregation='cell_median',
                    confidence='unavailable', metric_depth=False, temporal_comparability=False,
                    temporal_tracking=True, native_streaming_cache=True,
                    cache_frames=int(getattr(self.predictor, 'cache_size', 0)),
                    cache_frame_bound=32, processed_frames=self.frames, stream_reset=reset,
                    reset_count=self.resets, capture_timestamp_ms=frame.timestamp_ms,
                    capture_sequence=sequence, coordinate_frame='normalized_image_xy',
                    image_width=int(data.shape[1]), image_height=int(data.shape[0]),
                    calibration='unavailable', source_revision=UPSTREAM_REVISION,
                    artifact_integrity_verified=self.verified, model_quality_verified=False,
                    backpressure='synchronous_no_internal_queue')))
                self.owner, self.last = owner, (frame.timestamp_ms, sequence)
                return result
            except BaseException:
                self._reset()
                raise

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self._clear()
                predictor, self.predictor = self.predictor, None
                close = getattr(predictor, 'close', None)
                if callable(close):
                    close()


class _NativeVDA:
    def __init__(self, model, device, fp32=True):
        self.model, self.device, self.fp32 = model, device, fp32

    @property
    def cache_size(self):
        return len(self.model.frame_cache_list) if self.model is not None else 0

    def __call__(self, rgb):
        if self.model is None:
            raise RuntimeError('VDA native runtime closed')
        import torch
        with torch.inference_mode():
            return self.model.infer_video_depth_one(rgb, input_size=518,
                device=self.device, fp32=self.fp32)

    def reset(self):
        model = self.model
        if model is None:
            return
        model.transform = None
        model.frame_id_list = []
        model.frame_cache_list = []
        model.id = -1
        for name in ('frame_height', 'frame_width'):
            if hasattr(model, name):
                delattr(model, name)

    def close(self):
        self.reset()
        self.model = None


def load_worker(config):
    """Load verified local bytes only in an explicitly owned fresh worker."""
    args = configuration(config)
    if (os.environ.get('GAMST_PROVIDER_WORKER') != '1' or
            any(key == 'video_depth_anything' or key.startswith('video_depth_anything.')
                for key in sys.modules)):
        raise RuntimeError('VDA-S requires a fresh owned ProcessProvider worker')
    source_dir = Path(config['source_dir']).expanduser().resolve(strict=True)
    if not source_dir.is_dir() or not (source_dir/'video_depth_anything'/'video_depth_stream.py').is_file():
        raise ValueError('local pinned Video-Depth-Anything checkout required')
    variant = VARIANTS[args['variant']]
    payload = _read(config['path'], config['sha256'], variant['limit'])
    from importlib.metadata import PackageNotFoundError, version
    try:
        for package, wanted in RUNTIME:
            if version(package).split('+')[0] != wanted:
                raise RuntimeError(f'VDA runtime requires {package}=={wanted}')
    except PackageNotFoundError as exc:
        raise RuntimeError('select the documented isolated VDA runtime') from exc
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(source_dir))
    import torch
    from video_depth_anything.video_depth_stream import VideoDepthAnything
    if args['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('selected VDA CUDA device unavailable')
    with torch.device('cpu'):
        model = VideoDepthAnything(encoder=variant['encoder'], features=variant['features'],
            out_channels=variant['out_channels'])
    weights = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    del payload
    if type(weights) is not dict or set(weights) != set(model.state_dict()):
        raise ValueError('VDA checkpoint must contain exactly the selected variant state dictionary')
    for name, tensor in weights.items():
        expected = model.state_dict()[name]
        if (type(tensor) is not torch.Tensor or tensor.shape != expected.shape or
                tensor.dtype != expected.dtype or not torch.isfinite(tensor).all().item()):
            raise ValueError('VDA checkpoint tensor shape/dtype/finite mismatch: '+name)
    model.load_state_dict(weights, strict=True)
    model = model.to(args['device']).float().eval()
    return VideoDepthAnythingContext(_NativeVDA(model, args['device'], args['fp32']),
        artifact_verified=True, **args)
