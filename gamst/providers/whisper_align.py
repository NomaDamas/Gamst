"""Word times for a given transcript, from a local Whisper model (faster-whisper / CTranslate2 alignment).

For a recogniser that returns text without word times (OpenAI ``gpt-live-transcribe``): the text stays the
recogniser's; this only says where each of its words is in the audio. Whisper is not asked to transcribe. The text's
tokens are forced on the decoder (``ctranslate2.models.Whisper.align``) and the cross-attention alignment (DTW, the
same step faster-whisper uses for its own ``word_timestamps``) gives each word's start and end.

Words are split where the text has spaces (Korean, English ...); the words returned spell exactly the text given (the
tokenizer round-trips it), which :class:`gamst.providers.streaming_asr.StreamingWords` checks before using them.
A segment longer than 30 s is not aligned (Whisper's window); the streaming endpointer commits at most every 20 s.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

RATE = 16000


class WhisperAligner:
    def __init__(self, model, model_id):
        self.model, self.model_id = model, model_id
        self._tokenizers = {}

    @classmethod
    def from_local(cls, model_path, *, sha256=None, device='cpu', compute_type='int8', cpu_threads=4):
        from faster_whisper import WhisperModel
        root = Path(model_path)
        weights = root / 'model.bin'
        if not weights.is_file():
            raise ValueError(f'{root} is not a faster-whisper model directory (no model.bin)')
        digest = hashlib.sha256()
        with weights.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 22), b''):
                digest.update(block)
        digest = digest.hexdigest()
        if sha256 is not None and digest != sha256:
            raise ValueError(f'{weights} does not match its pinned sha256')
        model = WhisperModel(str(root), device=device, compute_type=compute_type, cpu_threads=int(cpu_threads))
        return cls(model, f'whisper-align:{digest[:16]}')

    def _tokenizer(self, language):
        if language not in self._tokenizers:
            from faster_whisper.tokenizer import Tokenizer
            self._tokenizers[language] = Tokenizer(self.model.hf_tokenizer, self.model.model.is_multilingual,
                                                   task='transcribe', language=language)
        return self._tokenizers[language]

    def align(self, samples, sample_rate, text, language):
        """Items ``{text, start_s, end_s}`` relative to the start of ``samples``, one per word of ``text``."""
        import numpy as np
        from faster_whisper.audio import pad_or_trim
        if int(sample_rate) != RATE:
            raise ValueError('the Whisper aligner needs 16 kHz audio')
        samples = np.asarray(samples, dtype=np.float32)
        text = ' '.join(str(text).split())
        if not text or len(samples) < RATE // 10:
            return []
        if len(samples) > 30 * RATE:
            raise ValueError('a segment longer than 30 s cannot be aligned in one Whisper window')
        tokenizer = self._tokenizer(language)
        tokens = tokenizer.encode(' ' + text)
        features = self.model.feature_extractor(samples)
        frames = min(features.shape[-1], self.model.feature_extractor.nb_max_frames)
        encoded = self.model.encode(pad_or_trim(features[:, :frames]))
        found = self.model.find_alignment(tokenizer, [tokens], encoded, frames)[0]
        from .term_events import normalize
        limit, items = len(samples) / RATE, []
        for item in found:
            word = item['word'].strip()
            if not word:
                continue
            if items and not normalize(word) and not item['word'].startswith(' '):
                items[-1]['text'] += word      # punctuation belongs to the word before it (it has no sound)
                continue
            items.append(dict(text=word, start_s=float(min(item['start'], limit)),
                              end_s=float(min(max(item['end'], item['start']), limit))))
        return items


def load(config):
    config = dict(config)
    return WhisperAligner.from_local(config.pop('model_path'), **config)
