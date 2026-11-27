"""TrackNetV3 ball heatmaps, one frame at a time (badminton shuttlecock, tennis ball): the SOTA stack's ball part.

TrackNetV3 (Chen and Wang, "TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory
Rectification", MMAsia 2023; code and checkpoints MIT, https://github.com/qaz812345/TrackNetV3) reads ``seq_len``
consecutive frames plus the background (the median frame) at 512 x 288 and returns one heatmap per frame. Its
authors report on the Shuttlecock Trajectory test set accuracy 97.51 %, precision 97.79 %, recall 99.33 %, F1 98.56 %
(README; not measured by GAMST). The network below is that repository's ``model.TrackNet`` (MIT, Copyright (c) 2023
qaz812345), rewritten here so the checkpoint loads without its training code; the pre-processing follows its
``dataset.Video_IterableDataset`` (RGB, resized to 512 x 288, scaled to [0, 1], the median frame first for
``bg_mode="concat"``) and the location its ``test.predict_location`` (threshold 0.5, the largest blob's box centre).

Realtime and post differ here (docs/sports.md):

- ``ensemble: "causal"`` (realtime): each new frame closes a window ending at it; the window's last heatmap is the
  answer for that frame, at once.
- ``ensemble: "weight"`` (post, the authors' default ``eval_mode=weight``): every frame is in ``seq_len`` windows;
  their heatmaps are averaged with the authors' positional weights. The answer for a frame comes ``seq_len - 1``
  frames later (``frame_ms`` says which frame it is). InpaintNet trajectory rectification (also post-only) is not
  wired yet.

The background is the median of frames sampled every ``median_every_ms`` (up to ``median_frames``), recomputed as
samples come in; the authors compute it over the whole video before the first frame, which a live camera cannot.
"""
from __future__ import annotations

import hashlib
import re
from collections import deque
from pathlib import Path

from ..capabilities import CapabilitySpec, OutputField

WIDTH, HEIGHT = 512, 288
ENSEMBLES = ('causal', 'weight')
BG_MODES = ('', 'subtract', 'subtract_concat', 'concat')


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def network(in_dim, out_dim):
    """TrackNetV3's ``TrackNet`` U-Net (MIT, qaz812345/TrackNetV3 ``model.py``), same layer names as its checkpoints."""
    import torch
    import torch.nn as nn

    class Conv2DBlock(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv = nn.Conv2d(i, o, kernel_size=3, padding='same', bias=False)
            self.bn = nn.BatchNorm2d(o)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Double2DConv(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv_1, self.conv_2 = Conv2DBlock(i, o), Conv2DBlock(o, o)

        def forward(self, x):
            return self.conv_2(self.conv_1(x))

    class Triple2DConv(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv_1, self.conv_2, self.conv_3 = Conv2DBlock(i, o), Conv2DBlock(o, o), Conv2DBlock(o, o)

        def forward(self, x):
            return self.conv_3(self.conv_2(self.conv_1(x)))

    class TrackNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_block_1 = Double2DConv(in_dim, 64)
            self.down_block_2 = Double2DConv(64, 128)
            self.down_block_3 = Triple2DConv(128, 256)
            self.bottleneck = Triple2DConv(256, 512)
            self.up_block_1 = Triple2DConv(768, 256)
            self.up_block_2 = Double2DConv(384, 128)
            self.up_block_3 = Double2DConv(192, 64)
            self.predictor = nn.Conv2d(64, out_dim, (1, 1))
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            pool, up = nn.MaxPool2d((2, 2), stride=(2, 2)), nn.Upsample(scale_factor=2)
            x1 = self.down_block_1(x)
            x2 = self.down_block_2(pool(x1))
            x3 = self.down_block_3(pool(x2))
            x = self.bottleneck(pool(x3))
            x = self.up_block_1(torch.cat([up(x), x3], dim=1))
            x = self.up_block_2(torch.cat([up(x), x2], dim=1))
            x = self.up_block_3(torch.cat([up(x), x1], dim=1))
            return self.sigmoid(self.predictor(x))

    return TrackNet()


def input_channels(seq_len, bg_mode):
    """TrackNetV3 ``utils.general.get_model``: input channels for a sequence length and background mode."""
    return {'subtract': seq_len, 'subtract_concat': seq_len * 4, 'concat': (seq_len + 1) * 3}.get(bg_mode, seq_len * 3)


def ensemble_weight(seq_len):
    """The authors' ``get_ensemble_weight(seq_len, "weight")``: 1, 2, ... towards the middle of the window."""
    weight = [1.] * seq_len
    for i in range((seq_len + 1) // 2):
        weight[i] = weight[seq_len - i - 1] = float(i + 1)
    total = sum(weight)
    return [value / total for value in weight]


def locate(heatmap, threshold=.5):
    """Heatmap (H, W) in [0, 1] -> (x, y, peak) of the largest blob's box centre in heatmap pixels, or None."""
    import numpy as np
    mask = (np.asarray(heatmap) > threshold).astype(np.uint8)
    if not mask.any():
        return None
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contour) for contour in contours]
    x, y, w, h = max(rects, key=lambda rect: rect[2] * rect[3])
    # the authors' centre: int(x + w / 2), int(y + h / 2) (predict.py)
    return int(x + w / 2), int(y + h / 2), float(np.asarray(heatmap)[y:y + h, x:x + w].max())


def check_config(config):
    known = {'path', 'sha256', 'device', 'ensemble', 'threshold', 'median_frames', 'median_every_ms'}
    if set(config) - known or not {'path', 'sha256'} <= set(config):
        raise ValueError(f'tracknet.ball needs path and sha256; optional {sorted(known - {"path", "sha256"})}')
    if not isinstance(config['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', config['sha256']):
        raise ValueError('a lowercase checkpoint SHA256 is required')
    if config.get('ensemble', 'causal') not in ENSEMBLES:
        raise ValueError(f'ensemble is one of {ENSEMBLES}')
    if config.get('device', 'cpu') not in ('cpu', 'mps', 'cuda'):
        raise ValueError('device is cpu, mps or cuda')
    return config


class TrackNetBall:
    """One stream per instance; frames must come in capture order."""

    @staticmethod
    def capability(config):
        config = check_config(dict(config))
        version = '1.' + config['sha256'][:16] + '.' + config.get('ensemble', 'causal')
        return CapabilitySpec('vision.tracknet.ball', 'vision.balls', 'video', 'tracknetv3', version,
                              (OutputField('frames', 'array'), OutputField('model', 'string')), 'frame',
                              stateful=True, reset_on_gap=True, freshness_ms=1000, devices=('cpu', 'mps', 'cuda'),
                              sensor_requirements=('uint8-bgr-camera',), license='adapter:Apache-2.0;TrackNetV3:MIT',
                              interpretation='estimate')

    @classmethod
    def from_local(cls, **config):
        import torch
        config = check_config(dict(config))
        path = Path(config['path']).expanduser().resolve()
        if not path.is_file() or _sha256(path) != config['sha256']:
            raise ValueError('existing TrackNetV3 checkpoint with the pinned SHA256 required')
        device = config.get('device', 'cpu')
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
        params = checkpoint['param_dict']
        seq_len, bg_mode = int(params['seq_len']), params.get('bg_mode') or ''
        model = network(input_channels(seq_len, bg_mode), seq_len)
        model.load_state_dict(checkpoint['model'])
        model.eval().to(device)
        return cls(model, seq_len=seq_len, bg_mode=bg_mode, device=device,
                   **{key: config[key] for key in ('ensemble', 'threshold', 'median_frames', 'median_every_ms') if key in config},
                   version=config['sha256'][:16])

    def __init__(self, model, *, seq_len, bg_mode='concat', device='cpu', ensemble='causal', threshold=.5,
                 median_frames=60, median_every_ms=500, version='custom', infer=None):
        if bg_mode not in BG_MODES:
            raise ValueError(f'bg_mode is one of {BG_MODES}')
        self.model, self.seq_len, self.bg_mode, self.device = model, int(seq_len), bg_mode, device
        self.ensemble, self.threshold, self.version = ensemble, float(threshold), version
        self.median_frames, self.median_every_ms = int(median_frames), int(median_every_ms)
        self.weight = ensemble_weight(self.seq_len)
        self._infer = infer or self._torch_infer
        self.spec = self.capability(dict(path='-', sha256='0' * 64, ensemble=ensemble))
        self.reset()

    def reset(self):
        self._frames = deque(maxlen=self.seq_len)       # (ms, RGB uint8 288x512x3)
        self._samples, self._last_sample_ms, self._median = [], None, None
        self._pending = {}                              # ms -> [weighted heatmap sum, weight sum]
        self._order = deque()
        self._size = None

    def close(self):
        self.model = None

    def _torch_infer(self, stacked):
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(stacked[None]).float().to(self.device)
            return self.model(tensor)[0].detach().cpu().numpy()

    def _background(self, ms, image):
        import numpy as np
        if self._last_sample_ms is None or ms - self._last_sample_ms >= self.median_every_ms:
            if len(self._samples) < self.median_frames:
                self._samples.append(image)
                self._median = np.median(np.stack(self._samples), 0).astype(np.uint8)
            self._last_sample_ms = ms
        return self._median

    def _stacked(self, median):
        import numpy as np
        frames = [image for _, image in self._frames]
        frames = [frames[0]] * (self.seq_len - len(frames)) + frames    # pad a short start with the first frame
        if self.bg_mode == 'subtract':
            channels = [np.abs(f.astype(np.int16) - median).sum(2, keepdims=True).clip(0, 255) for f in frames]
        elif self.bg_mode == 'subtract_concat':
            channels = [np.concatenate([f, np.abs(f.astype(np.int16) - median).sum(2, keepdims=True).clip(0, 255)], 2)
                        for f in frames]
        else:
            channels = list(frames)
        array = np.concatenate([np.moveaxis(np.asarray(item, dtype=np.float32), -1, 0) for item in channels], 0)
        if self.bg_mode == 'concat':
            array = np.concatenate([np.moveaxis(median.astype(np.float32), -1, 0), array], 0)
        return array / 255.

    def _answer(self, ms, heatmap):
        found = locate(heatmap, self.threshold)
        balls = [] if found is None else [dict(x=round(found[0] / WIDTH, 5), y=round(found[1] / HEIGHT, 5),
                                               score=round(found[2], 4))]
        return dict(frame_ms=int(ms), balls=balls)

    def process(self, frame):
        import numpy as np
        from PIL import Image
        data = frame.data
        if frame.modality != 'video' or not isinstance(data, np.ndarray) or data.dtype != np.uint8 or data.ndim != 3:
            raise ValueError('TrackNetV3 needs a uint8 BGR image')
        if self._size is not None and self._size != data.shape[:2]:
            raise ValueError('the image size changed: reset before another stream')
        self._size = data.shape[:2]
        rgb = np.asarray(Image.fromarray(data[:, :, ::-1]).resize((WIDTH, HEIGHT)))
        ms = frame.timestamp_ms
        self._frames.append((ms, rgb))
        median = self._background(ms, rgb)
        heatmaps = self._infer(self._stacked(median))           # (seq_len, H, W): one per frame of the window
        if self.ensemble == 'causal':
            answers = [self._answer(ms, heatmaps[-1])]
        else:
            window = [item[0] for item in self._frames]
            offset = self.seq_len - len(window)
            for position, at in enumerate(window):
                weight = self.weight[position + offset]
                total = self._pending.get(at)
                if total is None:
                    self._pending[at] = [heatmaps[position + offset].astype(np.float32) * weight, weight]
                    self._order.append(at)
                else:
                    total[0] += heatmaps[position + offset] * weight
                    total[1] += weight
            answers = []
            while self._order and (len(self._order) > self.seq_len - 1):
                at = self._order.popleft()
                total = self._pending.pop(at)
                answers.append(self._answer(at, total[0] / total[1]))
        return self.spec.validate(dict(frames=answers, model=f'tracknetv3:{self.version}:{self.ensemble}'))

    def finish(self):
        """Post: the last ``seq_len - 1`` frames, from the windows that reached them."""
        answers = []
        while self._order:
            at = self._order.popleft()
            total = self._pending.pop(at)
            answers.append(self._answer(at, total[0] / total[1]))
        return self.spec.validate(dict(frames=answers, model=f'tracknetv3:{self.version}:{self.ensemble}')) \
            if answers else None
