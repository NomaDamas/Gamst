"""FactorizePhys (NeurIPS 2024) camera pulse estimate using the official rPPG-Toolbox pipeline.

Official pieces executed from a pinned local rPPG-Toolbox checkout (no copies):
model ``neural_methods/model/FactorizePhys/FactorizePhys.py`` with the official
inference config (``configs/infer_configs/*FactorizePhys_FSAM_Res.yaml``:
FRAME_NUM 160, MD_FSAM False, MD_TYPE NMF, MD_INFERENCE True, MD_RESIDUAL True,
Raw frames, 72x72 face crop with 1.5x box, dynamic detection every 30 frames)
and ``evaluation/post_process.py`` (detrend, 0.6-3.3 Hz bandpass, FFT peak).

Deviation from the official preprocessing: face boxes come from the toolbox's
own "HC" (Haar cascade) backend, not "Y5F" (YOLO5Face weights are not shipped).
Output is a pulse-rate estimate from video only; it is not a medical measurement.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
from importlib.metadata import version
from pathlib import Path

from ..capabilities import CapabilitySpec, OutputField

TOOLBOX_REVISION = 'b7500b848f84ad7f86e277b4612563b69f4f88f9'
MD_CONFIG = {'FRAME_NUM': 160, 'MD_TYPE': 'NMF', 'MD_FSAM': False, 'MD_TRANSFORM': 'T_KAB',
             'MD_S': 1, 'MD_R': 1, 'MD_STEPS': 4, 'MD_INFERENCE': True, 'MD_RESIDUAL': True}
CHUNK, SIZE, FS, DETECT_EVERY, BOX_COEF = 160, 72, 30, 30, 1.5


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class FactorizePhysContext:
    def __init__(self, model, post, cascade, *, checkpoint_sha256, device, history_s=30, torch=None):
        self.model, self.post, self.cascade = model, post, cascade
        self.device, self.torch, self.history_s = device, torch, history_s
        self.spec = self.capability(checkpoint_sha256=checkpoint_sha256, device=device, history_s=history_s)
        self.lock = threading.RLock()
        self.closed = False
        self.reset()

    @staticmethod
    def capability(*, checkpoint_sha256, device='cuda', history_s=30):
        if type(history_s) is not int or not 6 <= history_s <= 120:
            raise ValueError('history_s must be 6..120')
        identity = json.dumps([TOOLBOX_REVISION, checkpoint_sha256, device, history_s, MD_CONFIG])
        return CapabilitySpec('physio.factorizephys.pulse', 'physio.pulse_rate', 'video',
            'factorizephys.rppg_toolbox', '1.'+hashlib.sha256(identity.encode()).hexdigest(),
            (OutputField('pulse_bpm', 'number', unit='bpm'), OutputField('window', 'object'),
             OutputField('face', 'object')), 'window', window_ms=CHUNK*1000//FS, freshness_ms=12000,
            stateful=True, reset_on_gap=True, interpretation='estimate', devices=(device,),
            sensor_requirements=('uint8-bgr-camera', '30fps-ordered-capture', 'single-visible-face'),
            license='rPPG-Toolbox:Responsible-AI-License;adapter:Apache-2.0')

    @classmethod
    def from_local(cls, source_dir, checkpoint_path, *, checkpoint_sha256, device='cuda', history_s=30):
        root = Path(source_dir).expanduser().resolve(strict=True)
        model_file = root/'neural_methods'/'model'/'FactorizePhys'/'FactorizePhys.py'
        cascade_file = root/'dataset'/'haarcascade_frontalface_default.xml'
        if not model_file.is_file() or not cascade_file.is_file():
            raise ValueError('pinned rPPG-Toolbox checkout required')
        if _sha256(checkpoint_path) != checkpoint_sha256:
            raise ValueError('FactorizePhys checkpoint SHA256 mismatch')
        if version('torch').split('+')[0] != '2.8.0':
            raise RuntimeError('selected FactorizePhys runtime requires torch==2.8.0')
        import cv2
        import torch
        sys.path.insert(0, str(root))
        from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys
        from evaluation import post_process
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('selected CUDA device unavailable')
        model = FactorizePhys(frames=CHUNK, md_config=dict(MD_CONFIG), in_channels=3, dropout=0.1,
                              device=torch.device(device))
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if all(key.startswith('module.') for key in state):
            state = {key[7:]: value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        cascade = cv2.CascadeClassifier(str(cascade_file))
        if cascade.empty():
            raise RuntimeError('Haar cascade failed to load')
        return cls(model, post_process, cascade, checkpoint_sha256=checkpoint_sha256, device=device,
                   history_s=history_s, torch=torch)

    def reset(self):
        self.frames, self.box, self.count, self.last_ts = [], None, 0, None
        self.bvp, self.detections, self.misses = [], 0, 0
        self.first_ts = None

    def close(self):
        with self.lock:
            self.closed = True
            self.model = None

    def _detect(self, rgb):
        import numpy as np
        zones = self.cascade.detectMultiScale(rgb.astype(np.uint8))
        self.detections += 1
        if len(zones) < 1:
            self.misses += 1
            box = [0, 0, rgb.shape[0], rgb.shape[1]]  # official fallback (whole frame)
        else:
            box = [float(v) for v in zones[int(np.argmax(zones[:, 2]))]]
        box[0] = max(0, box[0] - (BOX_COEF - 1.0) / 2 * box[2])
        box[1] = max(0, box[1] - (BOX_COEF - 1.0) / 2 * box[3])
        box[2], box[3] = BOX_COEF * box[2], BOX_COEF * box[3]
        return [int(v) for v in box]

    def process(self, frame):
        import cv2
        import numpy as np
        with self.lock:
            if self.closed:
                raise RuntimeError('FactorizePhys provider closed')
            if frame.modality != 'video' or frame.data.dtype != np.uint8 or frame.data.ndim != 3:
                raise ValueError('uint8 BGR video frame required')
            if self.last_ts is not None and frame.timestamp_ms <= self.last_ts:
                raise ValueError('frames must be strictly chronological')
            if self.last_ts is not None and frame.timestamp_ms - self.last_ts > 1000:
                self.reset()
            self.last_ts = frame.timestamp_ms
            self.first_ts = self.first_ts if self.first_ts is not None else frame.timestamp_ms
            rgb = cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB)
            if self.count % DETECT_EVERY == 0:
                self.box = self._detect(rgb)
            x, y, w, h = self.box
            crop = rgb[max(y, 0):min(y + h, rgb.shape[0]), max(x, 0):min(x + w, rgb.shape[1])]
            self.frames.append(cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32))
            self.count += 1
            if len(self.frames) < CHUNK:
                return None
            torch = self.torch
            chunk = np.stack(self.frames)  # T,H,W,C raw values (official "Raw")
            self.frames = []
            data = torch.from_numpy(chunk.transpose(3, 0, 1, 2)[None]).to(self.device)  # N,C,D,H,W
            data = torch.cat((data, data[:, :, -1:]), 2)  # official last-frame duplication
            with torch.inference_mode():
                out = self.model(data)
                pred = out[0] if isinstance(out, (tuple, list)) else out
                pred = (pred - torch.mean(pred)) / torch.std(pred)
            self.bvp.extend(pred.reshape(-1).float().cpu().numpy().tolist())
            keep = self.history_s * FS
            self.bvp = self.bvp[-keep:]
            signal = self.post._detrend(np.asarray(self.bvp, dtype=np.float64), 100)
            from scipy.signal import butter, filtfilt
            b, a = butter(1, [0.6 / FS * 2, 3.3 / FS * 2], btype='bandpass')
            signal = filtfilt(b, a, signal)
            bpm = float(self.post._calculate_fft_hr(signal, fs=FS))
            if not math.isfinite(bpm):
                return None
            return dict(pulse_bpm=round(bpm, 2),
                        window=dict(seconds=round(len(self.bvp) / FS, 2), chunks=len(self.bvp) // CHUNK,
                                    fs=FS, end_ms=frame.timestamp_ms),
                        face=dict(box_xywh=list(self.box), detections=self.detections, misses=self.misses))
