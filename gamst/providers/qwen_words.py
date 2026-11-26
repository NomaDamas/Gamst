"""Qwen3-ASR-1.7B with Qwen3-ForcedAligner-0.6B word times, on CPU or Apple MPS (transformers backend).

The official ``qwen_asr`` package's transformers backend (not vLLM), so it runs where there is no CUDA. It is a
decoder of :class:`gamst.providers.whisper_phrases.Phrases`: the same growing buffer and the same final phrases as
Whisper, so ``speech.transcript_words`` records (capture-time word times) are the same for every later module.

Qwen3-ASR gives one text for the decoded audio and no segments; the aligner gives each word's start and end. A
phrase ends where the words pause for ``pause_seconds`` or longer.

``context`` is Qwen3-ASR's context prompt (the system message; the Qwen3-ASR report calls it contextual biasing):
e.g. the words being counted. It can make the model write a word that was not said; measure with and without.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

from .asr_words import QWEN_LANGUAGES
from .term_events import normalize
from .whisper_phrases import Phrase

RATE = 16000
PINNED = (('qwen-asr', '0.0.6'), ('transformers', '4.57.6'), ('torch', '2.8.0'))


def check_versions():
    for package, expected in PINNED:
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f'install {package}=={expected} in this environment') from error
        if installed != expected:
            raise RuntimeError(f'qwen3 transformers backend is pinned to {package}=={expected}, found {installed}')


def verify_files(root, files):
    """``files``: {name: sha256} of the snapshot; every one must be there and match. Returns a short digest."""
    root = Path(root)
    if not isinstance(files, dict) or not files:
        raise ValueError('name the pinned files of the snapshot ({name: sha256})')
    for name, expected in sorted(files.items()):
        path = root / name
        if Path(name).name != name or not path.is_file():
            raise ValueError(f'{root} has no {name}')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 22), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'{path} does not match its pinned sha256')
    return hashlib.sha256(''.join(f'{name}={files[name]};' for name in sorted(files)).encode()).hexdigest()[:16]


def text_spans(text, words):
    """Where each aligned word is in ``text`` (character spans), or None when the two do not agree.

    The aligner drops punctuation and spaces; matching runs over normalized characters (term_events.normalize).
    """
    spans, position = [], 0
    for word in words:
        wanted = normalize(word)
        if not wanted:
            spans.append(None)
            continue
        start, got = None, ''
        while position < len(text) and len(got) < len(wanted):
            piece = normalize(text[position])
            if piece and start is None:
                start = position
            got += piece
            position += 1
        if got != wanted:
            return None
        spans.append((start, position))
    return spans


def split_phrases(text, items, pause_seconds):
    """Aligned ``items`` [(text, start_s, end_s)] of one decoded ``text`` -> phrases split at pauses."""
    items = [(str(word).strip(), float(start), float(end)) for word, start, end in items if str(word).strip()]
    if not items:
        return []
    groups, current = [], [0]
    for index in range(1, len(items)):
        if items[index][1] - items[current[-1]][2] >= pause_seconds:
            groups.append(current)
            current = []
        current.append(index)
    groups.append(current)
    spans = text_spans(text, [word for word, _, _ in items])
    phrases = []
    for group in groups:
        words = [items[index] for index in group]
        located = [spans[index] for index in group if spans[index]] if spans else []
        piece = text[located[0][0]:located[-1][1]].strip() if located else ' '.join(word for word, _, _ in words)
        phrases.append(Phrase(piece, words, max(end for _, _, end in words)))
    return phrases


class QwenDecoder:
    """Qwen3ASRModel (transformers) + Qwen3ForcedAligner: ``decode(samples)`` -> phrases with word times."""
    name = 'qwen3_asr.transformers'

    def __init__(self, model, *, language, model_id, context='', pause_seconds=.5):
        if language not in QWEN_LANGUAGES:
            raise ValueError(f"language must be one of {', '.join(sorted(QWEN_LANGUAGES))}")
        if type(context) is not str or len(context) > 2000:
            raise ValueError('context is a text of at most 2000 characters')
        if not 0 < float(pause_seconds) <= 5:
            raise ValueError('pause_seconds must be in (0, 5]')
        self.model, self.language, self.context = model, language, context
        self.pause_seconds = float(pause_seconds)
        bias = hashlib.sha256(context.encode()).hexdigest()[:8] if context else 'none'
        self.version = f'{model_id}:context.{bias}:pause.{self.pause_seconds:g}'

    @classmethod
    def from_local(cls, model_path, files, aligner_path, aligner_sha256, *, language='ko', device='cpu',
                   dtype='float32', context='', pause_seconds=.5, max_new_tokens=256):
        """Both snapshots are local and pinned; nothing is downloaded (HF offline is set by the worker)."""
        import os
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        check_versions()
        import torch
        from qwen_asr import Qwen3ASRModel
        if device not in ('cpu', 'mps', 'cuda') or dtype not in ('float32', 'float16', 'bfloat16'):
            raise ValueError('device is cpu, mps or cuda; dtype float32, float16 or bfloat16')
        asr_digest = verify_files(model_path, files)
        aligner = Path(aligner_path) / 'model.safetensors'
        verify_files(aligner_path, {'model.safetensors': aligner_sha256})
        kind = getattr(torch, dtype)
        model = Qwen3ASRModel.from_pretrained(
            str(model_path), dtype=kind, device_map=device, max_inference_batch_size=1, max_new_tokens=int(max_new_tokens),
            forced_aligner=str(aligner.parent), forced_aligner_kwargs=dict(dtype=kind, device_map=device))
        model_id = f'qwen3-asr-1.7b:{asr_digest}:aligner.{aligner_sha256[:16]}:{device}.{dtype}'
        return cls(model, language=language, model_id=model_id, context=context, pause_seconds=pause_seconds)

    def decode(self, samples):
        import numpy as np
        results = self.model.transcribe(audio=(np.asarray(samples, dtype=np.float32), RATE), context=self.context,
                                        language=QWEN_LANGUAGES[self.language], return_time_stamps=True)
        result = results[0]
        if not result.text.strip() or result.time_stamps is None:
            return []
        items = [(item.text, item.start_time, item.end_time) for item in result.time_stamps]
        return split_phrases(result.text, items, self.pause_seconds)
