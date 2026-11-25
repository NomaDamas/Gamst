"""Transcripts with word times: GAMST's streaming ASR plus a forced aligner on each final utterance.

This module does not reimplement ASR. It wraps what GAMST already has:

- CUDA: :class:`gamst.providers.qwen_asr.QwenStreamingASR` or
  :class:`gamst.providers.endpoint_asr.SileroEndpointASR` (Qwen3-ASR on vLLM), in process or as a
  ``ProcessProvider`` worker (profiles ``qwen3.asr`` / ``qwen3.asr.endpoint``). Qwen3-ASR streaming
  emits no timestamps, so each **final** utterance is aligned afterwards by
  :class:`QwenForcedAligner` (Qwen3-ForcedAligner-0.6B, Apache-2.0; the Qwen3-ASR report gives
  37.2 ms accumulated average shift on Korean MFA-labelled data).
- Mac / CPU: :class:`gamst.plugins.FasterWhisperPlugin` with ``word_timestamps=True`` emits
  ``speech.transcript_words`` itself; :class:`WhisperWords` only adds the last, shorter window at
  the end of an input. Its windows are fixed, so a word cut by a window edge can be missed.

:class:`AlignedASR` (a Pipeline provider module, capability ``speech.transcript_words``):

1. keeps the last ``ring_seconds`` of 16 kHz audio by capture time (:class:`AudioRing`);
2. re-cuts arbitrary input chunks into the engine's fixed hops on a gap-free sample clock
   (:class:`HopClock`) with the continuity metadata the Qwen engine checks (one mono channel,
   consecutive sequence numbers, +hop_ms end times); a capture gap finalizes the utterance and
   restarts the engine;
3. passes partial revisions through (``words: []``) when ``partials=True``;
4. for each final revision cuts that utterance's audio and aligns its text: ``words: [{text,
   start_ms, end_ms}]`` in capture time, the input :class:`gamst.providers.term_events.TermEvents`
   uses to time each counted word.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
from pathlib import Path
import uuid

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata
from ..plugins import InputFrame
from .audio_ring import AudioRing, HopClock

SAMPLE_RATE = 16000

# Qwen3-ASR language names (qwen_asr's ``language=`` values) for GAMST's two-letter codes.
QWEN_LANGUAGES = {'ko': 'Korean', 'en': 'English', 'zh': 'Chinese', 'ja': 'Japanese', 'de': 'German',
                  'fr': 'French', 'es': 'Spanish', 'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
                  'yue': 'Cantonese'}


async def _call(function, *args):
    value = function(*args)
    return await value if inspect.isawaitable(value) else value


def words_capability(version='1'):
    return CapabilitySpec(
        'speech.transcript_words', 'speech.transcript_words', 'audio', 'gamst.asr_words', version,
        (OutputField('utterance_id', 'string'), OutputField('revision', 'integer', required=False),
         OutputField('text', 'string'),
         OutputField('status', 'string'), OutputField('start_ms', 'integer'), OutputField('end_ms', 'integer'),
         OutputField('language', 'string', required=False), OutputField('words', 'array'),
         OutputField('aligned', 'boolean', required=False),
         OutputField('aligner', 'object', required=False), OutputField('finalization_reason', 'string', required=False)),
        'native', stateful=True, reset_on_gap=True, freshness_ms=30000, interpretation='semantic_description',
        devices=('cpu', 'cuda'), license='Apache-2.0')


def offset_words(items, start_ms):
    """Aligner items with times relative to the utterance start (seconds) -> capture-time words."""
    words = []
    for item in items:
        text = str(item['text']).strip()
        if not text:
            continue
        begin, end = float(item['start_s']), float(item['end_s'])
        if end < begin:
            begin, end = end, begin
        words.append(dict(text=text, start_ms=int(start_ms + round(1000 * begin)),
                          end_ms=int(start_ms + round(1000 * end))))
    return words


class QwenForcedAligner:
    """Qwen3-ForcedAligner-0.6B through the official ``qwen_asr`` package (transformers backend)."""

    def __init__(self, model, model_id):
        self.model, self.model_id = model, model_id

    @classmethod
    def from_local(cls, path, *, sha256=None, device='cuda:0', dtype='bfloat16'):
        """``path``: a local snapshot of ``Qwen/Qwen3-ForcedAligner-0.6B``; ``sha256`` pins model.safetensors."""
        import torch
        from qwen_asr import Qwen3ForcedAligner
        root = Path(path)
        weights = root / 'model.safetensors'
        if not weights.is_file():
            raise ValueError(f'{root} is not a Qwen3-ForcedAligner snapshot (no model.safetensors)')
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if sha256 is not None and digest != sha256:
            raise ValueError(f'{weights} does not match its pinned sha256 {sha256}')
        model = Qwen3ForcedAligner.from_pretrained(str(root), dtype=getattr(torch, dtype), device_map=device)
        return cls(model, f'qwen3-forced-aligner:{digest[:16]}')

    def align(self, samples, sample_rate, text, language):
        """Items ``{text, start_s, end_s}`` relative to the start of ``samples``."""
        name = QWEN_LANGUAGES.get(language, language)
        results = self.model.align(audio=(samples, int(sample_rate)), text=text, language=name)
        result = results[0] if isinstance(results, (list, tuple)) else results
        return [dict(text=getattr(item, 'text', None) if not isinstance(item, dict) else item['text'],
                     start_s=_seconds(item, 'start_time'), end_s=_seconds(item, 'end_time'))
                for item in _items(result)]


def _items(result):
    for key in ('items', 'words', 'segments'):
        value = getattr(result, key, None) if not isinstance(result, dict) else result.get(key)
        if value is not None:
            return value
    return result if isinstance(result, (list, tuple)) else []


def _seconds(item, key):
    return float(item[key] if isinstance(item, dict) else getattr(item, key))


class AlignedASR:
    """Pipeline module: 16 kHz audio in, ``speech.transcript_words`` (final utterances with word times) out."""
    owns_process = True    # the engine and the aligner run in their workers

    def __init__(self, asr, aligner=None, *, language=None, partials=False, ring_seconds=40.,
                 hop_ms=None, max_drift_ms=250):
        if not callable(getattr(asr, 'process', None)):
            raise TypeError('asr needs process(frame)')
        self.asr, self.aligner, self.language, self.partials = asr, aligner, language, bool(partials)
        hop_ms = hop_ms or getattr(asr, 'hop_ms', None) or getattr(getattr(asr, 'asr', None), 'hop_ms', None) or 20
        self.hop_ms = int(hop_ms)
        self.clock = HopClock(SAMPLE_RATE, self.hop_ms, max_drift_ms)
        self.ring = AudioRing(SAMPLE_RATE, ring_seconds)
        aligner_id = getattr(aligner, 'model_id', 'none')
        self.spec = words_capability(f"1.{aligner_id}".replace(':', '.'))
        self._restart()

    def _restart(self):
        self._epoch, self._sequence = uuid.uuid4().hex, 0

    async def reset(self):
        self.clock.reset()
        self.ring.reset()
        await self._restart_engine()

    async def _restart_engine(self):
        self._restart()
        reset = getattr(self.asr, 'reset', None)
        if callable(reset):
            await _call(reset)

    async def process(self, frame):
        if frame.modality != 'audio' or frame.data is None:
            return None
        if frame.sample_rate != SAMPLE_RATE:
            raise ValueError('asr_words needs 16 kHz mono audio')
        hops, discontinuity = self.clock.push(frame.data, int(frame.timestamp_ms))
        outputs = []
        if discontinuity:
            outputs += await self._flush('gap')
        for end_ms, samples in hops:
            self.ring.append(samples, end_ms, SAMPLE_RATE)   # same sample clock as the engine's utterance times
            sensor = SensorMetadata(source_id='gamst.asr_words', sensor_id='mono', sequence=self._sequence,
                                    channels=('mono',), stream_epoch=self._epoch)
            self._sequence += 1
            update = await _call(self.asr.process, InputFrame(samples, 'audio', int(end_ms), frame.subject_id,
                                                              SAMPLE_RATE, sensor))
            outputs += await self._emit(update)
        return outputs or None

    async def finish(self):
        outputs = await self._flush('input_end')
        self.clock.reset()
        return outputs or None

    async def _flush(self, reason):
        finish = getattr(self.asr, 'finish', None)
        update = await _call(finish) if callable(finish) else None
        outputs = await self._emit(update, reason=reason)
        await self._restart_engine()
        return outputs

    async def _emit(self, update, reason=None):
        if not isinstance(update, dict) or not update.get('text') and update.get('status') != 'final':
            return []
        final = update.get('status') == 'final'
        if not final and not self.partials:
            return []
        value = dict(utterance_id=str(update['utterance_id']), revision=int(update.get('revision', 0)),
                     text=str(update.get('text', '')), status='final' if final else 'partial',
                     start_ms=int(update['start_ms']), end_ms=int(update['end_ms']),
                     language=str(update.get('language') or self.language or 'und'), words=[], aligned=False)
        if update.get('finalization_reason') or reason:
            value['finalization_reason'] = str(update.get('finalization_reason') or reason)
        if final and value['text'] and self.aligner is not None:
            samples = self.ring.span(value['start_ms'], value['end_ms'])
            language = value['language'] if value['language'] != 'und' else self.language
            if len(samples) and language:
                arguments = (samples, SAMPLE_RATE, value['text'], language)
                items = await (self.aligner.align(*arguments) if inspect.iscoroutinefunction(self.aligner.align)
                               else asyncio.to_thread(self.aligner.align, *arguments))
                value['words'] = offset_words(items, value['start_ms'])
                value['aligned'] = True
                value['aligner'] = dict(id=self.aligner.model_id)
        return [self.spec.validate(value)]

    async def close(self):
        for owned in (self.asr, self.aligner):
            close = getattr(owned, 'close', None)
            if callable(close):
                await _call(close)


class WhisperWords:
    """FasterWhisperPlugin plus the end of the input: the last partial window is padded with silence."""

    def __init__(self, plugin):
        self.plugin = plugin
        self.name, self.version = plugin.name, plugin.version
        self._last = None

    async def process(self, frame):
        if frame.modality != 'audio' or frame.data is None:
            return []
        self._last = frame
        return await self.plugin.process(frame)

    async def finish(self):
        if self._last is None:
            return []
        import numpy as np
        size = int(self.plugin.window_seconds * SAMPLE_RATE)
        missing = size - len(self.plugin.buffers.get(self._last.subject_id, ()))
        if missing in (0, size):
            return []
        return await self.plugin.process(InputFrame(
            np.zeros(missing, np.float32), 'audio', self._last.timestamp_ms + round(1000 * missing / SAMPLE_RATE),
            self._last.subject_id, SAMPLE_RATE))

    def reset(self):
        self._last = None
        self.plugin.reset()
