"""Streaming speech recognition: words while the sound is still arriving, with capture-time word times.

A streaming recogniser (hosted over a WebSocket, or local in its own worker) is fed the microphone as it comes and
answers with partial text and committed (final) utterances. This module turns those answers into the same
``speech.transcript_words`` records as the phrase parts (:mod:`gamst.providers.whisper_phrases`), so ``terms``, the
speaker modules and scoring read them unchanged.

Capture time. A recogniser says where a word is in *the audio it was sent* (seconds from the start of its session or
of the committed segment). :class:`CaptureClock` remembers which capture times each sent sample had, so a word's time
is its capture time however late the answer arrives (network delay does not move it). A capture gap is glued out of
the stream: the clock maps each side to its own capture times.

Word times. Where the recogniser gives none (OpenAI ``gpt-live-transcribe``), the text of each committed segment is
force-aligned on the same audio by a local aligner in its own lane (``aligner.align(samples, rate, text, language)``,
by default Qwen3-ForcedAligner-0.6B, :class:`gamst.providers.asr_words.QwenForcedAligner`, in its worker;
:class:`gamst.providers.whisper_align.WhisperAligner` is a non-default baseline). The aligner only adds times:
the record's text is the recogniser's, and if the aligned words do not spell exactly that text the record keeps the
text without word times (``aligned: false``, ``align_error``) rather than a changed one.

Turns. A recogniser that waits for the client to end a turn (OpenAI; ElevenLabs with ``commit_strategy: manual``)
gets a commit from :class:`Endpointer`, an energy pause detector with an adaptive noise floor (no model; one rule,
not tuned on a clip): a pause of ``pause_ms`` after speech, a shorter one past ``soft_max_seconds``, always at
``max_seconds``.

Catch-up (:meth:`gamst.pipeline.Pipeline._skip`). A hosted session's state cannot be copied, so :meth:`StreamingWords.fork`
opens a new session of the same recogniser, and :meth:`StreamingWords.snapshot` says where this one last paused (the end
of its last committed turn, ``pause_ms``) and the text it committed last (``context``), which the fork gives the new
session as its prompt (OpenAI ``prompt``, ElevenLabs ``previous_text``). :meth:`StreamingWords.trim` keeps the words said
at capture times a worker owns. ``pending_since_ms`` is the capture time of the first sound whose words are not out yet
(the open turn, or a committed one whose answer has not arrived), so the Pipeline's ``complete_until_ms`` never passes
a word that is still coming.

Latency. ``final_after_ms`` on a final record is the wall time from when its last word's sound reached this module to
when the final answer did (a live microphone: from the moment it was said). A word's own latency is
``final_after_ms + (record end_ms - word end_ms)``. Partial records (``partials: true``) carry ``heard_to_ms``, the
capture time of the newest sound sent when the partial arrived.
"""
from __future__ import annotations

import asyncio
from collections import deque
import inspect
import time

from ..capabilities import CapabilitySpec, OutputField
from .asr_words import offset_words
from .audio_ring import AudioRing, mono_float, resample

RATE = 16000


def _wall_ms():
    return time.time_ns() // 1_000_000


def streaming_capability(version='1'):
    return CapabilitySpec(
        'speech.transcript_words', 'speech.transcript_words', 'audio', 'gamst.streaming_asr', version,
        (OutputField('utterance_id', 'string'), OutputField('revision', 'integer', required=False),
         OutputField('text', 'string'), OutputField('status', 'string'),
         OutputField('start_ms', 'integer'), OutputField('end_ms', 'integer'),
         OutputField('language', 'string', required=False), OutputField('words', 'array'),
         OutputField('aligned', 'boolean', required=False), OutputField('aligner', 'object', required=False),
         OutputField('align_error', 'string', required=False),
         OutputField('finalization_reason', 'string', required=False),
         OutputField('final_after_ms', 'integer', required=False), OutputField('heard_to_ms', 'integer', required=False),
         OutputField('trimmed', 'boolean', required=False)),
        'native', stateful=True, reset_on_gap=True, freshness_ms=30000, interpretation='semantic_description',
        devices=('cpu', 'mps', 'cuda'), license='Apache-2.0')


class CaptureClock:
    """Samples sent to a recogniser (its stream clock) -> capture time, and when each piece arrived here."""

    def __init__(self, rate=RATE, keep_seconds=900.):
        self.rate, self.keep = int(rate), int(keep_seconds * rate)
        self.reset()

    def reset(self):
        self.pieces = deque()   # (first stream sample, count, capture end ms of the piece, wall ms it arrived)
        self.sent = 0

    def push(self, count, capture_end_ms, wall_ms=None):
        """``count`` more samples whose last one was captured at ``capture_end_ms``; returns the first's index."""
        first = self.sent
        if count > 0:
            self.pieces.append((first, int(count), float(capture_end_ms), _wall_ms() if wall_ms is None else wall_ms))
            self.sent += int(count)
            while len(self.pieces) > 1 and self.pieces[1][0] < self.sent - self.keep:
                self.pieces.popleft()
        return first

    def capture_ms(self, sample):
        """Capture time of stream sample ``sample`` (a fraction is fine); outside what was sent, the nearest piece."""
        if not self.pieces:
            raise ValueError('nothing was sent yet')
        chosen = self.pieces[-1]
        for piece in self.pieces:
            if sample < piece[0] + piece[1]:
                chosen = piece
                break
        first, count, end_ms, _ = chosen
        return int(round(end_ms - 1000. * (first + count - sample) / self.rate))

    def seconds_ms(self, seconds, origin=0):
        """A recogniser's time (seconds from stream sample ``origin``) -> capture ms."""
        return self.capture_ms(origin + seconds * self.rate)

    def arrival_ms(self, capture_ms):
        """Wall time when the sound up to ``capture_ms`` had reached this module (None if not held)."""
        for _, _, end_ms, wall in self.pieces:
            if end_ms >= capture_ms - 1:
                return wall
        # a recogniser may place the last word a little after the last sample it was sent
        return self.pieces[-1][3] if self.pieces and capture_ms - self.pieces[-1][2] <= 1000 else None


class StreamResampler:
    """16 kHz -> another rate, chunk by chunk, without edge clicks: each output waits ``lookahead`` input samples."""

    def __init__(self, source, target, context=256, lookahead=64):
        from fractions import Fraction
        ratio = Fraction(int(target), int(source))
        self.source, self.target = int(source), int(target)
        self.up, self.down = ratio.numerator, ratio.denominator
        self.context, self.lookahead = context - context % self.down, lookahead - lookahead % self.down
        self.reset()

    def reset(self):
        import numpy as np
        self._held = np.zeros(0, np.float32)   # input not yet turned into output (at most lookahead samples)
        self._past = np.zeros(0, np.float32)   # input already turned into output, kept as filter context
        self._done = 0                          # output samples produced so far

    def push(self, samples, *, final=False):
        import numpy as np
        if self.source == self.target:
            return mono_float(samples)
        held = np.concatenate([self._held, mono_float(samples)])
        ready = len(held) if final else max(0, len(held) - self.lookahead)
        if not ready:
            self._held = held
            return np.zeros(0, np.float32)
        window = np.concatenate([self._past, held])
        out = resample(window, self.source, self.target)
        start = len(self._past)
        first, last = start * self.up // self.down, (start + ready) * self.up // self.down
        produced = out[first:last]
        self._past = window[:start + ready][-self.context:]
        self._held = held[ready:]
        return produced


def pcm16(samples):
    import numpy as np
    return (np.clip(np.asarray(samples, np.float32), -1, 1) * 32767).astype('<i2').tobytes()


class Endpointer:
    """Where a turn ends, for a recogniser that waits for the client's commit (energy pauses; no model)."""

    def __init__(self, rate=RATE, *, frame_ms=20, pause_ms=500, short_pause_ms=160, min_seconds=1.,
                 soft_max_seconds=12., max_seconds=20., margin_db=8., floor_seconds=8.):
        if not 0 < min_seconds < soft_max_seconds <= max_seconds <= 28:
            raise ValueError('need 0 < min_seconds < soft_max_seconds <= max_seconds <= 28')
        self.rate, self.frame = int(rate), int(rate * frame_ms / 1000)
        self.pause, self.short_pause = int(pause_ms / frame_ms), int(short_pause_ms / frame_ms)
        self.min_frames, self.soft_frames = int(min_seconds * 1000 / frame_ms), int(soft_max_seconds * 1000 / frame_ms)
        self.max_frames, self.margin_db = int(max_seconds * 1000 / frame_ms), float(margin_db)
        self.floor_frames = int(floor_seconds * 1000 / frame_ms)
        self.reset()

    def reset(self):
        import numpy as np
        self._rest = np.zeros(0, np.float32)
        self._levels = deque(maxlen=self.floor_frames)
        self._frames = self._quiet = 0
        self._speech = False

    def push(self, samples):
        """Samples in stream order -> how many of them end turns: a list of positions in ``samples`` (after which
        to commit)."""
        import numpy as np
        values = np.concatenate([self._rest, mono_float(samples)])
        cuts, used = [], 0
        offset = -len(self._rest)
        while used + self.frame <= len(values):
            frame = values[used:used + self.frame]
            used += self.frame
            level = 10 * np.log10(float(np.mean(np.square(frame, dtype=np.float64))) + 1e-10)
            self._levels.append(level)
            floor = float(np.percentile(self._levels, 15))
            loud = level > floor + self.margin_db
            self._frames += 1
            self._speech = self._speech or loud
            self._quiet = 0 if loud else self._quiet + 1
            pause = self.pause if self._frames < self.soft_frames else self.short_pause
            if self._frames >= self.max_frames or (
                    self._speech and self._frames >= self.min_frames and self._quiet >= pause):
                cuts.append(offset + used)
                self._frames = self._quiet = 0
                self._speech = False
        self._rest = values[used:]
        return cuts


async def _call(function, *args):
    value = function(*args)
    return await value if inspect.isawaitable(value) else value


class StreamingWords:
    """Pipeline module: 16 kHz audio in, ``speech.transcript_words`` out, from a streaming recogniser session.

    ``session``: an object with ``rate`` (Hz it is sent), ``manual_commit`` (it needs the client's commits),
    ``word_times`` (its finals carry word times), ``name``/``version``, and the coroutines ``open()``,
    ``send(pcm16_bytes)``, ``commit()``, ``end()`` (no more audio: answer what is held) and ``close()``; it puts
    answers on ``session.answers`` (an ``asyncio.Queue``) as dicts:

    - ``{"kind": "partial", "key": K, "text": T}``
    - ``{"kind": "committed", "key": K}``: the segment the next commit (in commit order) made has the id K
    - ``{"kind": "final", "key": K, "text": T, "words": [(text, start_s, end_s)] | None, "base": "session" |
      "segment"}``: times from the start of the session's audio or of the committed segment
    - ``{"kind": "error", "message": M}``

    The session runs while the Pipeline feeds frames; answers that arrived are returned by the next ``process``
    call (or by ``finish``, which waits for the answers to every commit, at most ``finish_timeout`` s).
    """
    owns_process = True

    def __init__(self, session, *, aligner=None, language='ko', partials=False, endpointer=None,
                 finish_timeout=30., ring_seconds=120., max_gap_ms=250):
        self.session, self.aligner, self.language, self.partials = session, aligner, language, bool(partials)
        self.endpointer = endpointer if endpointer is not None else (Endpointer() if session.manual_commit else None)
        self.finish_timeout, self.max_gap_ms = float(finish_timeout), int(max_gap_ms)
        self.clock = CaptureClock(RATE)
        self.ring = AudioRing(RATE, ring_seconds, max_drift_ms=max_gap_ms)
        self.resampler = StreamResampler(RATE, session.rate)
        aligner_id = getattr(aligner, 'model_id', None)
        self.spec = streaming_capability(f"1.{session.version}{'.' + aligner_id if aligner_id else ''}".replace(':', '.'))
        self.name, self.version = session.name, session.version
        self._opened, self._outbox, self._lane = False, [], None
        self._segments = deque()      # (first, last) stream samples of commits not yet named by the recogniser
        self._named = {}              # key -> (first, last) stream samples
        self._order, self._ready = deque(), {}     # committed keys in commit order; finals waiting for an earlier one
        self._waiting = 0             # commits whose final has not arrived
        self._started = 0             # stream sample where the open segment starts
        self._last_end_ms = None
        self._revision = {}
        self._pump = None
        self.errors = []
        self._commits = deque()       # (first, last) stream samples of commits whose words are not out yet
        self._settled = 0             # stream samples before this one: every word of them is out
        self._history = deque(maxlen=8)   # the last committed texts (a fork's context)
        self._owns_aligner = True
        self._options = dict(language=language, partials=partials, finish_timeout=finish_timeout,
                             ring_seconds=ring_seconds, max_gap_ms=max_gap_ms)

    async def _open(self):
        if not self._opened:
            await self.session.open()
            self._opened = True
            self._pump = asyncio.ensure_future(self._read())

    async def process(self, frame):
        if frame.modality != 'audio' or frame.data is None:
            return None
        if frame.sample_rate != RATE:
            raise ValueError('streaming ASR needs 16 kHz mono audio')
        await self._open()
        samples = mono_float(frame.data)
        start_ms = frame.timestamp_ms - 1000. * len(samples) / RATE
        if self._last_end_ms is not None and abs(start_ms - self._last_end_ms) > self.max_gap_ms:
            await self._commit()      # missing sound: the open turn ends where the sound stopped
        self._last_end_ms = frame.timestamp_ms
        self.ring.append(samples, frame.timestamp_ms, RATE)
        first = self.clock.push(len(samples), frame.timestamp_ms)
        await self.session.send(pcm16(self.resampler.push(samples)))
        for cut in (self.endpointer.push(samples) if self.endpointer is not None else ()):
            await self._commit(first + cut)
        return self._take()

    async def _commit(self, at=None):
        at = self.clock.sent if at is None else at
        if not self.session.manual_commit or at - self._started < RATE // 10:
            return
        self._segments.append((self._started, at))
        self._commits.append((self._started, at))
        self._started, self._waiting = at, self._waiting + 1
        await self.session.commit()

    async def finish(self):
        if not self._opened:
            return None
        await self.session.send(pcm16(self.resampler.push([], final=True)))
        await self._commit()
        await self.session.end()
        deadline = time.monotonic() + self.finish_timeout
        closes = getattr(self.session, 'closes_on_end', False)   # it answers what it holds, then closes
        while ((not self.session.ended) if closes else (
                self._waiting > 0 or not self.session.manual_commit and not self.session.ended)) \
                and time.monotonic() < deadline and not self._pump.done():
            await asyncio.sleep(.05)
        if self._lane is not None:
            await self._lane
        out = self._take()
        return out or None

    def _take(self):
        out, self._outbox = self._outbox, []
        return out or None

    async def _read(self):
        while True:
            answer = await self.session.answers.get()
            if answer is None:
                return
            try:
                await self._answer(answer)
            except Exception as error:      # one bad answer must not stop the stream
                self.errors.append(f'{type(error).__name__}: {error}')

    async def _answer(self, answer):
        kind = answer.get('kind')
        if kind == 'error':
            self.errors.append(str(answer.get('message')))
            return
        if kind == 'committed':
            if self._segments:
                self._named[answer['key']] = self._segments.popleft()
            self._order.append(answer['key'])
            return
        if kind == 'partial':
            if self.partials and str(answer.get('text', '')).strip():
                key = str(answer['key'])
                revision = self._revision[key] = self._revision.get(key, -1) + 1
                now_ms = self.clock.capture_ms(self.clock.sent) if self.clock.sent else 0
                span = self._named.get(answer['key']) or (self._started, self.clock.sent)
                start_ms = self.clock.capture_ms(span[0]) if self.clock.sent else 0
                self._outbox.append(self.spec.validate(dict(
                    utterance_id=f'{self.name}:{key}', revision=revision, text=str(answer['text']).strip(),
                    status='partial', start_ms=start_ms, end_ms=max(start_ms + 1, now_ms), words=[],
                    language=self.language or 'und', heard_to_ms=now_ms)))
            return
        if kind != 'final':
            return
        self._waiting = max(0, self._waiting - 1) if self.session.manual_commit else self._waiting
        span = self._named.pop(answer['key'], None)
        if span is None and self.session.manual_commit and self._segments:
            span = self._segments.popleft()      # a recogniser that does not say which commit it answers
        key = answer['key']
        text = str(answer.get('text') or '').strip()
        if not text:
            return self._deliver(key, None)
        received = _wall_ms()
        if answer.get('words'):
            origin = span[0] if answer.get('base') == 'segment' and span else 0
            words = [dict(text=word, start_ms=self.clock.seconds_ms(start, origin),
                          end_ms=max(self.clock.seconds_ms(start, origin) + 1, self.clock.seconds_ms(end, origin)))
                     for word, start, end in answer['words'] if str(word).strip()]
            if not words:
                return self._deliver(key, None)
            value = self._value(key, text, words[0]['start_ms'], words[-1]['end_ms'], words, received)
            return self._deliver(key, self.spec.validate(value))
        if span is None:
            self.errors.append('a final without word times and without a known segment was dropped')
            return self._deliver(key, None)
        start_ms, end_ms = self.clock.capture_ms(span[0]), self.clock.capture_ms(span[1])
        value = self._value(key, text, start_ms, max(start_ms + 1, end_ms), [], received)
        if self.aligner is None:
            return self._deliver(key, self.spec.validate(value))
        # The aligner works in its own lane: answers keep arriving while it runs.
        previous = self._lane
        self._lane = asyncio.ensure_future(self._align(key, value, self.ring.span(start_ms, end_ms), previous))

    def _deliver(self, key, value):
        """Records leave in commit order (a recogniser may answer turns out of order); None: nothing to say."""
        if key not in self._order:
            if value is not None:
                self._outbox.append(value)
                self._history.append(value['text'])
                if not self.session.manual_commit and value.get('words') and self.clock.sent:
                    self._settled = max(self._settled, self._sample_at(value['end_ms']))
            return
        self._ready[key] = value
        while self._order and self._order[0] in self._ready:
            done = self._ready.pop(self._order.popleft())
            if self._commits:
                self._settled = max(self._settled, self._commits.popleft()[1])
            if done is not None:
                self._outbox.append(done)
                self._history.append(done['text'])

    async def _align(self, key, value, samples, previous):
        if previous is not None:
            await previous
        value = dict(value)
        if len(samples):
            started = time.monotonic()
            items = await _call(self.aligner.align, samples, RATE, value['text'], self.language)
            took = round(1000 * (time.monotonic() - started))
            words = offset_words(items, value['start_ms'])
            if spelled(words) == spelled_text(value['text']):
                value.update(words=words, aligned=True, start_ms=words[0]['start_ms'] if words else value['start_ms'])
            else:
                value['align_error'] = 'aligned words do not spell the transcript; kept without word times'
            value['aligner'] = dict(id=getattr(self.aligner, 'model_id', 'aligner'), ms=took, seconds=round(len(samples) / RATE, 2))
            value['final_after_ms'] = max(0, _wall_ms() - (self.clock.arrival_ms(value['end_ms']) or _wall_ms()))
        self._deliver(key, self.spec.validate(value))

    def _value(self, key, text, start_ms, end_ms, words, received):
        arrived = self.clock.arrival_ms(end_ms)
        value = dict(utterance_id=f'{self.name}:{key}', revision=self._revision.get(str(key), -1) + 1, text=text,
                     status='final', start_ms=int(start_ms), end_ms=int(end_ms), words=words,
                     language=self.language or 'und', aligned=False)
        if arrived is not None:
            value['final_after_ms'] = max(0, received - arrived)
        return value

    async def reset(self):
        await self.close()
        self.clock.reset()
        self.ring.reset()
        self.resampler.reset()
        if self.endpointer is not None:
            self.endpointer.reset()
        self._segments.clear()
        self._named.clear()
        self._order.clear()
        self._ready.clear()
        self._commits.clear()
        self._waiting = self._started = self._settled = 0
        self._last_end_ms = None

    async def close(self):
        if self._opened:
            self._opened = False
            await self.session.close()
            if self._pump is not None:
                self._pump.cancel()
        close = getattr(self.aligner, 'close', None)
        if callable(close) and self._owns_aligner:
            await _call(close)

    # ------------------------------------------------------------------ catch-up

    def _sample_at(self, capture_ms):
        """The stream sample captured at ``capture_ms`` (the nearest piece's), for a recogniser that commits by itself."""
        for first, count, end_ms, _ in self.clock.pieces:
            start_ms = end_ms - 1000. * count / RATE
            if capture_ms <= end_ms:
                return first + max(0, min(count, int(round((capture_ms - start_ms) * RATE / 1000))))
        return self.clock.sent

    @property
    def pending_since_ms(self):
        """Capture time of the first sound whose words have not been given yet (None: all of it is out)."""
        if not self._opened or self.clock.sent <= self._settled or not self.clock.pieces:
            return None
        return self.clock.capture_ms(self._settled)

    @property
    def can_fork(self):
        return callable(getattr(self.session, 'fork', None))

    @property
    def max_forks(self):
        """Sessions of this recogniser that may run at once: the provider's (``concurrency``), else 1."""
        return int(getattr(self.session, 'concurrency', 1) or 1)

    def snapshot(self):
        """Where this session last paused (``pause_ms``) and what it committed last (``context``)."""
        pause = None
        if self.clock.sent and self.clock.pieces:
            pause = self.clock.capture_ms(self._started if self.session.manual_commit else self._settled)
        return dict(pause_ms=pause, context=' '.join(self._history)[-400:])

    def fork(self):
        """A new session of the same recogniser for a catch-up worker (the aligner is shared, not closed by it)."""
        import copy
        endpointer = None
        if self.endpointer is not None:
            endpointer = copy.copy(self.endpointer)
            endpointer.reset()
        other = StreamingWords(self.session.fork(), aligner=self.aligner, endpointer=endpointer, **self._options)
        other.partials = False
        other._owns_aligner = False
        self._forks = getattr(self, '_forks', 0) + 1
        other.name = f'{self.name}.catch-up{self._forks}'     # its turns are other utterances than this session's
        return other

    def restore(self, state):
        """A fork starts where ``snapshot`` was taken: the hosted session cannot be copied, so it gets the text as context."""
        context = (state or {}).get('context')
        if context and callable(getattr(self.session, 'set_context', None)):
            self.session.set_context(context)

    async def skipped(self, start_ms, end_ms):
        """The Pipeline skipped [start_ms, end_ms] (a worker hears it): the open turn ends where the sound stopped."""
        if self._opened:
            await self._commit()
        self._last_end_ms = None

    @staticmethod
    def trim(value, keep):
        """The words of ``value`` said where ``keep(start_ms, end_ms)`` holds (an utterance without word times: all
        or nothing, by its own times); None when nothing is left."""
        if not isinstance(value, dict):
            return value
        words = [word for word in value.get('words') or () if isinstance(word, dict)
                 and type(word.get('start_ms')) is int and type(word.get('end_ms')) is int]
        if not words:
            ends = (value.get('start_ms'), value.get('heard_to_ms', value.get('end_ms')))
            return value if not all(type(at) is int for at in ends) or keep(*ends) else None
        kept = [word for word in words if keep(word['start_ms'], word['end_ms'])]
        if len(kept) == len(words):
            return value
        if not kept:
            return None
        return dict(value, words=kept, text=' '.join(str(word['text']) for word in kept), start_ms=kept[0]['start_ms'],
                    end_ms=max(kept[-1]['end_ms'], kept[0]['start_ms'] + 1), trimmed=True)


def spelled(words):
    from .term_events import normalize
    return ''.join(normalize(word['text']) for word in words)


def spelled_text(text):
    from .term_events import normalize
    return normalize(text)


def pause_utterances(words, pause_seconds):
    """[(text, start_s, end_s)] -> groups split where the words pause (for recognisers with no turns)."""
    groups, current = [], []
    for word in words:
        if current and word[1] - current[-1][2] >= pause_seconds:
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups
