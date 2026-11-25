"""Whisper on a growing buffer: each *phrase* is a final utterance, so a word is never cut by a window edge.

Audio since the last final phrase is kept and decoded again every ``hop_seconds``. A phrase (a Whisper
segment) is final once it ended ``margin_seconds`` before the newest audio; the buffer then starts at its end.
A phrase that never ends is committed at ``max_seconds``. One phrase is mostly one person's, so a speaker
check of an utterance is a check of a voice, not of whatever a fixed window held.
"""
from __future__ import annotations

from ..plugins import InputFrame, _pcm_floats, _signal

RATE = 16000


class WhisperPhrases:
    def __init__(self, plugin, *, hop_seconds=4., margin_seconds=3., max_seconds=25., beam_size=5, max_gap_ms=100):
        if not 0 < margin_seconds < hop_seconds + max_seconds or not hop_seconds < max_seconds <= 30:
            raise ValueError('need hop_seconds < max_seconds <= 30 and 0 < margin_seconds')
        self.plugin, self.beam_size, self.max_gap_ms = plugin, int(beam_size), int(max_gap_ms)
        self.hop_seconds, self.margin_seconds, self.max_seconds = hop_seconds, margin_seconds, max_seconds
        self.reset()

    @property
    def version(self):
        return f'{self.plugin.version}:phrases.{self.hop_seconds:g}-{self.margin_seconds:g}-{self.max_seconds:g}'

    def reset(self):
        import numpy as np
        self._buffer, self._end_ms, self._since, self._subject = np.zeros(0, dtype=np.float32), None, 0, None

    def process(self, frame):
        import numpy as np
        if frame.sample_rate != RATE:
            raise ValueError('WhisperPhrases requires 16 kHz mono input')
        values = np.asarray(_pcm_floats(frame.data), dtype=np.float32)
        start_ms, output = frame.timestamp_ms - 1000 * len(values) / RATE, []
        if self._end_ms is not None and abs(start_ms - self._end_ms) > self.max_gap_ms:
            output += self._decode(final=True)     # missing sound: what is held ends there
            self.reset()
        self._buffer = np.concatenate([self._buffer, values])
        self._end_ms, self._subject = frame.timestamp_ms, frame.subject_id
        self._since += len(values)
        if self._since >= self.hop_seconds * RATE:
            output += self._decode()
        return output

    def finish(self):
        return self._decode(final=True) if self._end_ms is not None else []

    def _decode(self, *, final=False):
        import numpy as np
        self._since = 0
        seconds = len(self._buffer) / RATE
        if seconds < .3 or float(np.abs(self._buffer).max()) < 1e-4:
            if seconds > self.margin_seconds:
                self._trim(seconds - self.margin_seconds)
            return []
        segments, _ = self.plugin.model.transcribe(
            self._buffer, language=self.plugin.language, vad_filter=False, beam_size=self.beam_size, temperature=0,
            condition_on_previous_text=False, word_timestamps=True)
        segments = list(segments)
        full = seconds >= self.max_seconds
        limit = seconds if final else seconds - self.margin_seconds
        output, cut = [], 0.
        for segment in segments:
            words = [word for word in segment.words or () if word.word.strip()]
            end = max([segment.end] + [word.end for word in words])
            if end > limit and not (full and segment is segments[0]):
                break
            cut = max(cut, min(end, seconds))
            if words and segment.avg_logprob >= -1.2 and segment.no_speech_prob <= .6 and \
                    segment.compression_ratio <= 2.4:
                output.append(self._utterance(words, segment))
        if not segments and not final:
            cut = max(0., seconds - self.margin_seconds - self.hop_seconds)    # nobody spoke: keep the recent part
        self._trim(cut)
        return output

    def _utterance(self, words, segment):
        zero = self._end_ms - 1000 * len(self._buffer) / RATE
        at = lambda seconds: round(zero + 1000 * seconds)
        start_ms, end_ms = at(words[0].start), max(at(words[-1].end), at(words[0].start) + 1)
        value = dict(utterance_id=f'phrase:{self._subject}:{start_ms}:{end_ms}', status='final',
                     text=segment.text.strip(), start_ms=start_ms, end_ms=end_ms,
                     words=[dict(text=word.word.strip(), start_ms=at(word.start), end_ms=at(word.end)) for word in words])
        return _signal(InputFrame(None, 'audio', end_ms, self._subject, RATE), 'speech.transcript_words', value,
                       self.plugin.name, self.version)

    def _trim(self, seconds):
        self._buffer = self._buffer[max(0, round(seconds * RATE)):]


class PhraseWorker:
    """What the worker of the windowed plugin expects: ``await process`` and ``await finish``."""

    def __init__(self, module):
        self.module = module
        self.version = module.version

    async def process(self, frame):
        return self.module.process(frame)

    async def finish(self):
        return self.module.finish()

    def reset(self):
        self.module.reset()
