"""pyannote PixIT (``pyannote/speech-separation-ami-1.0``): joint diarization and separation, local files only.

PixIT (Kalda et al., Odyssey 2024) trains one model (ToTaToNet: an encoder/decoder with a DPRNN mask net
and WavLM features) to diarize and to separate at once, on real meetings (AMI). The pipeline
``pyannote.audio.pipelines.SpeechSeparation`` (pyannote.audio 3.3.2) slides the model over a window,
clusters local speakers with a speaker embedding, and returns ``(diarization, sources)``: one source per
speaker. With ``leakage_removal`` (the published setting) a source is zero wherever its speaker is
inactive, so "which sources are active at this word" is read off the sources themselves.

Files (gated on Hugging Face; the owner accepts the conditions of both repositories):

- ``config.yaml`` of ``pyannote/speech-separation-ami-1.0``: the pipeline's hyper-parameters,
- ``pytorch_model.bin`` of ``pyannote/separation-ami-1.0``: the ToTaToNet checkpoint,
- the clustering embedding the config names: ``speechbrain/spkrec-ecapa-voxceleb`` at revision
  ``5c0be3875fda05e81f3c004ed8c7c06be308de1e`` (not gated; Apache-2.0), a local directory with its
  ``hyperparams.yaml``, ``embedding_model.ckpt``, ``mean_var_norm_emb.ckpt``, ``classifier.ckpt`` and
  ``label_encoder.txt``; its hub path in ``hyperparams.yaml`` is answered from that directory,
- ``config.json`` of ``microsoft/wavlm-large`` (2 KB): ToTaToNet builds WavLM by that hub name when it
  loads. This loader builds it from the config alone; its weights are inside the PixIT checkpoint
  (1.28 GB, loaded strictly, so a checkpoint without them is refused). Nothing is downloaded.

Licences (Hugging Face model cards): pyannote.audio and PixIT MIT, SpeechBrain ECAPA Apache-2.0.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

SAMPLE_RATE = 16000
PYANNOTE_VERSION = '3.3.2'
WAVLM = 'microsoft/wavlm-large'
PIPELINE_ARGUMENTS = ('segmentation_step', 'embedding_exclude_overlap', 'clustering', 'embedding_batch_size',
                      'segmentation_batch_size')


def _verify(path, expected):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    value = digest.hexdigest()
    if expected is not None and value != expected:
        raise ValueError(f'{path} does not match its pinned sha256 {expected}')
    return value


@contextlib.contextmanager
def _local_wavlm(directory):
    """ToTaToNet calls ``AutoModel.from_pretrained('microsoft/wavlm-large')``: build it from the local config
    (random weights, replaced by the checkpoint's own)."""
    import transformers
    original = transformers.AutoModel.from_pretrained

    def local(name, *args, **kwargs):
        if name != WAVLM:
            raise ValueError(f'PixIT asked for {name!r}; only {WAVLM} is provided locally')
        if directory is None or not (Path(directory) / 'config.json').is_file():
            raise ValueError(f'this checkpoint uses WavLM: give "wavlm", a directory with {WAVLM} config.json')
        config = transformers.AutoConfig.from_pretrained(str(directory), local_files_only=True)
        return transformers.AutoModel.from_config(config)

    transformers.AutoModel.from_pretrained = local
    try:
        yield
    finally:
        transformers.AutoModel.from_pretrained = original


@contextlib.contextmanager
def _local_speechbrain(directory):
    """pyannote builds the SpeechBrain embedding from its hub name: answer it from ``directory`` instead."""
    from pyannote.audio.pipelines import speaker_verification
    if directory is None or not (Path(directory) / 'hyperparams.yaml').is_file():
        raise ValueError('the PixIT config clusters with a SpeechBrain embedding: give its local directory')
    original = speaker_verification.SpeechBrain_EncoderClassifier

    class Local:
        @staticmethod
        def from_hparams(*, source=None, savedir=None, run_opts=None, use_auth_token=None, revision=None, **_):
            collected = Path(directory) / 'collected'
            run_opts = {key: str(value) for key, value in (run_opts or {}).items()}   # speechbrain 1.1 wants 'cpu'
            return original.from_hparams(source=str(directory), savedir=str(collected), run_opts=run_opts,
                                         overrides={'pretrained_path': str(directory)})

    speaker_verification.SpeechBrain_EncoderClassifier = Local
    try:
        yield
    finally:
        speaker_verification.SpeechBrain_EncoderClassifier = original


def _load_checkpoint(path):
    """pyannote 3.3.2 checkpoints hold pickled objects (``torch.load`` weights-only refuses them). The bytes
    were checked against their pinned sha256 before this runs."""
    import torch
    from pyannote.audio import Model
    original = torch.load

    def full(*args, **kwargs):
        kwargs['weights_only'] = False
        return original(*args, **kwargs)

    torch.load = full
    try:
        return Model.from_pretrained(str(path))
    finally:
        torch.load = original


class PixITSeparator:
    """``separate(window) -> (speakers, samples)`` with the published PixIT pipeline."""

    sample_rate = SAMPLE_RATE

    def __init__(self, pipeline, model_id, device):
        self.pipeline, self.model_id, self.device = pipeline, model_id, device

    @classmethod
    def from_local(cls, config, separation, embedding, *, wavlm=None, sha256=None, embedding_sha256=None,
                   config_sha256=None, wavlm_sha256=None, device='cpu', pipeline=None):
        from importlib.metadata import version
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        if version('pyannote.audio') != PYANNOTE_VERSION:
            raise RuntimeError(f'PixIT needs pyannote.audio {PYANNOTE_VERSION}')
        import torch
        import yaml
        from pyannote.audio.pipelines import SpeechSeparation
        _verify(config, config_sha256)
        digest = _verify(separation, sha256)
        if wavlm is not None:
            _verify(Path(wavlm) / 'config.json', wavlm_sha256)
        settings = yaml.safe_load(Path(config).read_text()) or {}
        arguments = (settings.get('pipeline') or {}).get('params') or {}
        with _local_wavlm(wavlm):
            segmentation = _load_checkpoint(separation)
        named = str(arguments.get('embedding') or '')
        if 'speechbrain' in named:
            _verify(Path(embedding) / 'embedding_model.ckpt', embedding_sha256)
            with _local_speechbrain(embedding):    # ``to(device)`` rebuilds the SpeechBrain model too
                built = SpeechSeparation(segmentation=segmentation, embedding=named,
                                         **{key: arguments[key] for key in PIPELINE_ARGUMENTS if key in arguments})
                built.to(torch.device(device))
        else:
            _verify(embedding, embedding_sha256)
            built = SpeechSeparation(segmentation=segmentation, embedding=_load_checkpoint(embedding),
                                     **{key: arguments[key] for key in PIPELINE_ARGUMENTS if key in arguments})
            built.to(torch.device(device))
        built.instantiate(pipeline if pipeline is not None else settings.get('params') or {})
        return cls(built, f'pyannote-pixit:{digest[:16]}', device)

    def separate(self, window):
        """Sources of one window, each zero outside its speaker's turns (+- ``asr_collar``).

        pyannote.audio 3.3.2's own leakage removal fails on a speaker that never pauses in the window and on
        one with no turn at all (IndexError / empty concatenate), so it is switched off and the same masking
        is applied here from the per-speaker activity the pipeline computes (its columns are the sources).
        """
        import numpy as np
        import torch
        from pyannote.core import SlidingWindowFeature
        samples = np.asarray(window, dtype=np.float32).reshape(-1)
        file = {'waveform': torch.from_numpy(samples).unsqueeze(0), 'sample_rate': SAMPLE_RATE}
        captured, original = {}, self.pipeline.to_annotation

        def capture(discrete, *args, **kwargs):
            captured['activity'] = discrete
            return original(discrete, *args, **kwargs)

        leakage, collar = bool(self.pipeline.separation.leakage_removal), float(self.pipeline.separation.asr_collar)
        self.pipeline.to_annotation, self.pipeline.separation.leakage_removal = capture, False
        try:
            with torch.inference_mode():
                _, sources = self.pipeline(file)
        finally:
            self.pipeline.separation.leakage_removal = leakage
            del self.pipeline.to_annotation
        if sources is None:
            return np.zeros((0, len(samples)), np.float32)
        data = np.asarray(sources.data, np.float32)
        if leakage and 'activity' in captured:
            activity = captured['activity']
            mask = dilate(np.asarray(activity.data) > 0, int(round(collar / activity.sliding_window.step)))
            aligned = SlidingWindowFeature(mask.astype(np.float32), activity.sliding_window).align(sources).data
            data = data * aligned[:, :data.shape[1]]
        out = np.zeros((data.shape[1], len(samples)), np.float32)
        size = min(len(samples), data.shape[0])
        out[:, :size] = data[:size].T
        return out


def dilate(active, frames):
    """(frames, speakers) booleans, each run widened by ``frames`` on both sides."""
    import numpy as np
    active = np.asarray(active, bool)
    if frames <= 0 or not active.size:
        return active
    out = active.copy()
    for shift in range(1, frames + 1):
        out[shift:] |= active[:-shift]
        out[:-shift] |= active[shift:]
    return out
