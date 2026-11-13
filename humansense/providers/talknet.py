"""Pinned local TalkNet window inference. No implicit model/source downloads."""
# Retained in installed Python distributions for adapted preprocessing math.
# The MIT License (MIT)
# 
# Copyright (c) 2013 James Lyons (MFCC reference)
# Copyright (c) 2021 Tao Ruijie (face crop reference)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import sys
import threading
import types
import uuid

from ..audiovisual import decode_talknet_window
from ..capabilities import CapabilitySpec, OutputField, SensorMetadata
from ..plugins import InputFrame
from .vocal_expression import _read, _digest

REVISION = '6d6821479af485e251c4991487e40573b42181b4'
SOURCES = {
    'audioEncoder.py': (3851, 'd4334100ba4e3921dc2011eb56e4d0eb1fd8b69cce16235cb14c47a05d89c6c3'),
    'visualEncoder.py': (6193, '5a33ee41194666eeccb48590f72ea1e59b52f40e19dca31390dd7b67d9f22d15'),
    'attentionLayer.py': (1356, '5c0451f2dfd5a52c086ba3e5957d57987962ea979ccfae42360be2b2668aeaa7'),
    'talkNetModel.py': (2349, 'df788d96833f67178821f31abdb789fd3aa2a5144fd258d5437c7fbc43a69611'),
}


def mfcc(pcm):
    """Fixed TalkNet geometry; algorithm follows python_speech_features0.6/MIT.

    See docs/licenses/python-speech-features-MIT.txt. No normalization/padding.
    """
    import numpy as np
    from scipy.fftpack import dct
    if (type(pcm) is not np.ndarray or pcm.dtype != np.dtype('int16') or pcm.ndim != 1 or
            len(pcm) not in tuple(n*640+240 for n in (25, 50, 75, 100, 125, 150))):
        raise ValueError('exact bounded PCM16 audiovisual window required')
    signal = pcm.astype(np.float64)
    emphasized = np.r_[signal[0], signal[1:]-.97*signal[:-1]]
    frames = np.lib.stride_tricks.sliding_window_view(emphasized, 400)[::160]
    power = (np.abs(np.fft.rfft(frames, 512))**2)/512
    energy = power.sum(axis=1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    mel = np.linspace(0, 2595*np.log10(1+8000/700), 28)
    bins = np.floor(513*(700*(10**(mel/2595)-1))/16000).astype(int)
    bank = np.zeros((26, 257), np.float64)
    for i in range(26):
        left, center, right = bins[i:i+3]
        bank[i, left:center] = (np.arange(left, center)-left)/(center-left)
        bank[i, center:right] = (right-np.arange(center, right))/(right-center)
    features = power @ bank.T
    features = np.where(features == 0, np.finfo(float).eps, features)
    features = dct(np.log(features), type=2, axis=1, norm='ortho')[:, :13]
    features *= 1+11*np.sin(np.pi*np.arange(13)/22)
    features[:, 0] = np.log(energy)
    return features.astype(np.float32)


def _classes(blobs):
    # Only exact hash-verified upstream bytes reach this function. Rewrite the
    # three absolute model imports into a unique temporary package namespace.
    package = '_gamst_talknet_'+uuid.uuid4().hex
    root = types.ModuleType(package)
    root.__path__ = []
    modules = {package: root}
    try:
        sys.modules[package] = root
        for filename, blob in blobs.items():
            name = package+'.'+filename[:-3]
            module = types.ModuleType(name)
            module.__package__ = package
            modules[name] = module
            sys.modules[name] = module
            source = blob.decode('utf-8').replace('from model.', 'from '+package+'.')
            exec(compile(source, filename, 'exec'), module.__dict__)
        return modules[package+'.talkNetModel'].talkNetModel
    finally:
        for name, module in modules.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]


class _Native:
    def __init__(self, model, head, torch, device):
        self.model, self.head, self.torch, self.device = model, head, torch, device

    def __call__(self, features, faces):
        torch, model = self.torch, self.model
        with torch.inference_mode():
            a = torch.from_numpy(features.copy()).unsqueeze(0).to(self.device)
            v = torch.from_numpy(faces.astype('float32')).unsqueeze(0).to(self.device)
            a, v = model.forward_audio_frontend(a), model.forward_visual_frontend(v)
            a, v = model.forward_cross_attention(a, v)
            logits = self.head(model.forward_audio_visual_backend(a, v))
            if (type(logits) is not torch.Tensor or logits.dtype != torch.float32 or
                    tuple(logits.shape) != (len(faces), 2) or not torch.isfinite(logits).all().item()):
                raise ValueError('invalid native TalkNet logits')
            return logits.cpu().numpy().copy()


class TalkNetContext:
    def __init__(self, predictor, *, checkpoint_sha256, window_frames=25, device='cpu'):
        self.spec = self.capability(checkpoint_sha256=checkpoint_sha256, window_frames=window_frames, device=device)
        if not callable(predictor):
            raise ValueError('explicit TalkNet predictor required')
        self.predictor, self.window_frames = predictor, window_frames
        self.checkpoint_sha256 = checkpoint_sha256
        self.artifact_hashes_verified = False
        self.closed = False
        self._lock = threading.RLock()
        self._last = None

    @staticmethod
    def capability(*, checkpoint_sha256, window_frames=25, device='cpu'):
        _digest(checkpoint_sha256)
        if type(window_frames) is not int or window_frames not in (25, 50, 75, 100, 125, 150):
            raise ValueError('window must be1..6seconds at25fps')
        if type(device) is not str or device not in ('cpu', 'cuda'):
            raise ValueError('select cpu or cuda')
        digest = hashlib.sha256(json.dumps([REVISION, checkpoint_sha256, window_frames, device]).encode()).hexdigest()
        return CapabilitySpec('video.talknet.active_speaker', 'video.active_speaker', 'video',
            'talknet.local', '1.'+digest, (OutputField('association', 'object'),),
            'window', window_ms=window_frames*40, lookahead_ms=15, freshness_ms=2000,
            stateful=True, reset_on_gap=True, interpretation='classification', devices=(device,),
            sensor_requirements=('gtav001-paired-input', 'declared-same-clock', '25fps-face-crops', '16khz-mono-pcm16'),
            license='TalkNet-code:MIT;weights:unverified;adapter:Apache-2.0;MFCC:MIT-derived')

    @classmethod
    def from_local(cls, source_dir, checkpoint_path, *, checkpoint_sha256, window_frames=25, device='cpu'):
        cls.capability(checkpoint_sha256=checkpoint_sha256, window_frames=window_frames, device=device)
        if version('torch').split('+')[0] != '2.8.0':
            raise RuntimeError('selected TalkNet runtime requires torch==2.8.0')
        blobs = {name: _read(Path(source_dir)/name, limit, digest) for name, (limit, digest) in SOURCES.items()}
        checkpoint = _read(checkpoint_path, 128*1024*1024, checkpoint_sha256)
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('selected CUDA device unavailable')
        model_class = _classes(blobs)
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            model = model_class().float().cpu().eval()
            head = torch.nn.Linear(256, 2).float().cpu().eval()
        weights = torch.load(io.BytesIO(checkpoint), map_location='cpu', weights_only=True)
        del checkpoint
        expected = {'model.'+k: v for k, v in model.state_dict().items()}
        expected.update({'lossAV.FC.'+k: v for k, v in head.state_dict().items()})
        for lane in ('lossA', 'lossV'):
            expected[lane+'.FC.weight'] = torch.empty(2, 128, dtype=torch.float32, device='cpu')
            expected[lane+'.FC.bias'] = torch.empty(2, dtype=torch.float32, device='cpu')
        if not isinstance(weights, dict) or set(weights) != set(expected):
            raise ValueError('TalkNet checkpoint must contain every exact model/head tensor')
        for name, tensor in weights.items():
            if (type(tensor) is not torch.Tensor or tensor.shape != expected[name].shape or
                    tensor.dtype != expected[name].dtype or tensor.device.type != 'cpu' or
                    not torch.isfinite(tensor).all().item()):
                raise ValueError('TalkNet checkpoint tensor shape/dtype/finite mismatch')
        model.load_state_dict({k[6:]: v for k, v in weights.items() if k.startswith('model.')}, strict=True)
        head.load_state_dict({k[10:]: v for k, v in weights.items() if k.startswith('lossAV.FC.')}, strict=True)
        result = cls(_Native(model.to(device), head.to(device), torch, device),
                     checkpoint_sha256=checkpoint_sha256, window_frames=window_frames, device=device)
        result.artifact_hashes_verified = True
        return result

    def reset(self):
        with self._lock:
            self._last = None

    def close(self):
        with self._lock:
            self._last = None
            self.predictor = None
            self.closed = True

    def process(self, frame):
        with self._lock:
            if self.closed:
                raise RuntimeError('TalkNet provider is closed')
            try:
                import numpy as np
                if (not isinstance(frame, InputFrame) or frame.modality != 'video' or
                        type(frame.sample_rate) is not int or frame.sample_rate != 16000 or
                        not isinstance(frame.sensor, SensorMetadata)):
                    raise ValueError('explicit paired audiovisual InputFrame required')
                meta, pcm, faces = decode_talknet_window(frame.data)
                sensor = frame.sensor
                if (meta['frames'] != self.window_frames or frame.subject_id != meta['subject_id'] or
                        type(frame.timestamp_ms) is not int or frame.timestamp_ms != meta['audio_end_ms'] or
                        sensor.channels != ('mono',)):
                    raise ValueError('paired window disagrees with outer frame provenance/configuration')
                owner = json.dumps([meta['session_id'], meta['track_id'], meta['clock_id'],
                    meta['subject_id'], meta['audio_sensor'], meta['video_sensor'], sensor.source_id,
                    sensor.sensor_id, sensor.stream_epoch, sensor.calibration_id, sensor.coordinate_frame,
                    sensor.capture_sequence is not None], sort_keys=True)
                seq = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
                if self._last and self._last[0] == owner:
                    if seq <= self._last[1] or frame.timestamp_ms <= self._last[2]:
                        raise ValueError('paired windows must advance capture time and sequence')
                logits = self.predictor(mfcc(pcm), faces)
                if (type(logits) is not np.ndarray or logits.dtype != np.float32 or
                        logits.shape != (self.window_frames, 2) or not np.isfinite(logits).all()):
                    raise ValueError('invalid TalkNet class logits')
                scores = logits.astype(np.float64)
                probabilities = np.exp(scores-scores.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                values = [dict(start_ms=meta['start_ms']+i*40, end_ms=meta['start_ms']+(i+1)*40,
                    non_speaking_logit=float(scores[i, 0]), speaking_logit=float(scores[i, 1]),
                    speaking_probability=float(probabilities[i, 1])) for i in range(self.window_frames)]
                output = self.spec.validate({'association': dict(window=meta, scores=values,
                    source_revision=REVISION, checkpoint_sha256=self.checkpoint_sha256,
                    artifact_hashes_verified=self.artifact_hashes_verified, checkpoint_authenticity_verified=False,
                    quality_verified=False, identity_verified=False, synchronization_verified=False,
                    probability_calibrated=False, diarization_slot=None, asr_attribution_verified=False,
                    window_ensemble=False, streaming_mode='window')})
                self._last = (owner, seq, frame.timestamp_ms)
                return output
            except Exception:
                self._last = None
                raise


def prepare_talknet_face(bgr, bbox):
    """Instantaneous upstream crop geometry; no temporal smoothing/identity.

    bbox is explicit pixel xyxy from a detector/tracker. This codec-free path
    does not reproduce the offline demo's13-frame median or XVID artifacts.
    """
    import math
    import numpy as np
    import cv2
    if (type(bgr) is not np.ndarray or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or
            not 32 <= bgr.shape[0] <= 2048 or not 32 <= bgr.shape[1] <= 2048 or bgr.shape[0]*bgr.shape[1] > 2097152):
        raise ValueError('bounded uint8 BGR frame required')
    if (type(bbox) not in (tuple, list) or len(bbox) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)):
        raise ValueError('finite pixel xyxy box required')
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= bgr.shape[1] and 0 <= y0 < y1 <= bgr.shape[0] and x1-x0 >= 8 and y1-y0 >= 8):
        raise ValueError('face box must be inside frame and at least8pixels per side')
    size = max(x1-x0, y1-y0)/2
    cx, cy = (x0+x1)/2, (y0+y1)/2
    pad = int(size*1.8)
    left, right = int(cx+pad-size*1.4)-pad, int(cx+pad+size*1.4)-pad
    top, bottom = int(cy+pad-size)-pad, int(cy+pad+size*1.8)-pad
    crop = np.full((bottom-top, right-left, 3), 110, np.uint8)
    sx0, sy0, sx1, sy1 = max(0,left), max(0,top), min(bgr.shape[1],right), min(bgr.shape[0],bottom)
    crop[sy0-top:sy1-top, sx0-left:sx1-left] = bgr[sy0:sy1,sx0:sx1]
    resized = cv2.resize(crop, (224,224), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray[56:168,56:168].copy()
