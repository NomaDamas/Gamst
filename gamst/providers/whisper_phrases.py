"""ASR on a growing buffer: each *phrase* is a final utterance, so a word is never cut by a window edge.

Audio since the last final phrase is kept and decoded again every ``hop_seconds``. A phrase is final once it
ended ``margin_seconds`` before the newest audio; the buffer then starts at its end. A phrase that never ends is
committed at ``max_seconds``. One phrase is mostly one person's, so a speaker check of an utterance is a check of a
voice, not of whatever a fixed window held.

The decoder is the model: :class:`WhisperDecoder` (faster-whisper; a phrase is a Whisper segment) or any object
with ``name``, ``version``, ``language`` and ``decode(samples) -> [Phrase]`` (e.g.
:class:`gamst.providers.qwen_words.QwenDecoder`, Qwen3-ASR with Qwen3-ForcedAligner word times).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..plugins import InputFrame, _pcm_floats, _signal

RATE = 16000


@dataclass
class Phrase:
    """One decoded phrase; times are seconds from the start of the decoded samples."""
    text: str
    words: list           # [(text, start_s, end_s)]
    end_s: float          # where the phrase ends (may be later than its last word)
    keep: bool = True     # False: decoded but not trustworthy (it still moves the buffer on)


class WhisperDecoder:
    """faster-whisper segments as phrases; a segment below Whisper's own confidence limits is not kept."""

    def __init__(self, plugin, *, beam_size=5):
        self.plugin, self.beam_size = plugin, int(beam_size)
        self.name, self.language = plugin.name, plugin.language

    @property
    def version(self):
        return self.plugin.version

    def decode(self, samples):
        segments, _ = self.plugin.model.transcribe(
            samples, language=self.plugin.language, vad_filter=False, beam_size=self.beam_size, temperature=0,
            condition_on_previous_text=False, word_timestamps=True)
        phrases = []
        for segment in segments:
            words = [(word.word.strip(), word.start, word.end) for word in segment.words or () if word.word.strip()]
            phrases.append(Phrase(segment.text.strip(), words, max([segment.end] + [end for _, _, end in words]),
                                  keep=bool(words) and segment.avg_logprob >= -1.2 and segment.no_speech_prob <= .6
                                  and segment.compression_ratio <= 2.4))
        return phrases


class Phrases:
    def __init__(self, decoder, *, hop_seconds=4., margin_seconds=3., max_seconds=25., max_gap_ms=100):
        if not 0 < margin_seconds < hop_seconds + max_seconds or not hop_seconds < max_seconds <= 30:
            raise ValueError('need hop_seconds < max_seconds <= 30 and 0 < margin_seconds')
        self.decoder, self.max_gap_ms = decoder, int(max_gap_ms)
        self.hop_seconds, self.margin_seconds, self.max_seconds = hop_seconds, margin_seconds, max_seconds
        self.reset()

    @property
    def version(self):
        return f'{self.decoder.version}:phrases.{self.hop_seconds:g}-{self.margin_seconds:g}-{self.max_seconds:g}'

    def reset(self):
        import numpy as np
        self._buffer, self._end_ms, self._since, self._subject = np.zeros(0, dtype=np.float32), None, 0, None

    def process(self, frame):
        import numpy as np
        if frame.sample_rate != RATE:
            raise ValueError('phrase ASR requires 16 kHz mono input')
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
        phrases = list(self.decoder.decode(self._buffer))
        full = seconds >= self.max_seconds
        limit = seconds if final else seconds - self.margin_seconds
        output, cut = [], 0.
        for phrase in phrases:
            if phrase.end_s > limit and not (full and phrase is phrases[0]):
                break
            cut = max(cut, min(phrase.end_s, seconds))
            if phrase.words and phrase.keep:
                output.append(self._utterance(phrase))
        if not phrases and not final:
            cut = max(0., seconds - self.margin_seconds - self.hop_seconds)    # nobody spoke: keep the recent part
        self._trim(cut)
        return output

    def _utterance(self, phrase):
        zero = self._end_ms - 1000 * len(self._buffer) / RATE
        at = lambda seconds: round(zero + 1000 * seconds)
        words = phrase.words
        start_ms, end_ms = at(words[0][1]), max(at(words[-1][2]), at(words[0][1]) + 1)
        value = dict(utterance_id=f'phrase:{self._subject}:{start_ms}:{end_ms}', status='final',
                     text=phrase.text, start_ms=start_ms, end_ms=end_ms,
                     words=[dict(text=text, start_ms=at(start), end_ms=at(end)) for text, start, end in words])
        return _signal(InputFrame(None, 'audio', end_ms, self._subject, RATE), 'speech.transcript_words', value,
                       self.decoder.name, self.version)

    def _trim(self, seconds):
        self._buffer = self._buffer[max(0, round(seconds * RATE)):]


class WhisperPhrases(Phrases):
    """faster-whisper phrases (the plugin's loaded model)."""

    def __init__(self, plugin, *, hop_seconds=4., margin_seconds=3., max_seconds=25., beam_size=5, max_gap_ms=100):
        super().__init__(WhisperDecoder(plugin, beam_size=beam_size), hop_seconds=hop_seconds,
                         margin_seconds=margin_seconds, max_seconds=max_seconds, max_gap_ms=max_gap_ms)
        self.plugin = plugin


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
