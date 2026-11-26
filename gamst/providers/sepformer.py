"""SpeechBrain SepFormer (``speechbrain/sepformer-whamr16k``): two sources from a mixture, local files only.

Trained on WHAMR! (two English speakers, noise and reverberation, synthetic mixtures) at 16 kHz;
Apache-2.0 code and model card. It always returns two sources and does not say when a speaker is
silent, so a silent speaker's source holds leakage of the other; activity is decided from each
source's level by its reader (:class:`gamst.providers.speaker_verify.SpeakerVerify`).

The model directory holds ``hyperparams.yaml``, ``encoder.ckpt``, ``decoder.ckpt`` and ``masknet.ckpt``
of revision ``21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5``; ``sha256`` pins ``masknet.ckpt``.
"""
from __future__ import annotations

from pathlib import Path

from .pixit import _verify

SAMPLE_RATE = 16000
SPEECHBRAIN_VERSION = '1.1.2'


class SepFormerSeparator:
    sample_rate = SAMPLE_RATE

    def __init__(self, model, model_id):
        self.model, self.model_id = model, model_id

    @classmethod
    def from_local(cls, directory, *, sha256=None, device='cpu'):
        from importlib.metadata import version
        if version('speechbrain') != SPEECHBRAIN_VERSION:
            raise RuntimeError(f'SepFormer needs speechbrain {SPEECHBRAIN_VERSION}')
        from speechbrain.inference.separation import SepformerSeparation
        directory = Path(directory)
        for name in ('hyperparams.yaml', 'encoder.ckpt', 'decoder.ckpt', 'masknet.ckpt'):
            if not (directory / name).is_file():
                raise ValueError(f'{directory} needs {name} (a SepFormer model directory)')
        digest = _verify(directory / 'masknet.ckpt', sha256)
        model = SepformerSeparation.from_hparams(source=str(directory), savedir=str(directory),
                                                 run_opts={'device': device})
        model.eval()
        return cls(model, f'sepformer:{digest[:16]}')

    def separate(self, window):
        import numpy as np
        import torch
        samples = np.asarray(window, dtype=np.float32).reshape(-1)
        with torch.inference_mode():
            estimate = self.model.separate_batch(torch.from_numpy(samples).unsqueeze(0))[0].T.cpu().numpy()
        out = np.zeros((estimate.shape[0], len(samples)), np.float32)
        size = min(len(samples), estimate.shape[1])
        out[:, :size] = estimate[:, :size]
        return out
