"""Overlapped speech, live: the enrolled person's voice extracted from the mixture, and the words heard in it.

Owner decision (2026-11-26, docs/realtime-tse.md). Where two voices are close together the per-word voice score on the
room audio embeds both, so it cannot say who said a word. On each such stretch this lane extracts the enrolled
person's voice with a target-speaker-extraction model conditioned on the owner's few-shot clip
(``audio.target_speech``), lets a fork of the live recogniser hear the extracted sound, scores the extracted sound
against the voice (the speaker model of ``speaker_check``) and decides the counted words said there. Everything else is
left as it was.

Three parts, each a Pipeline module in the existing contracts:

- **Where voices meet** (:class:`MultiVoice`, math): from the live diarization's slot activity
  (``audio.speaker_activity``; slot identities are not trusted, only how many slots are active). ``rule``:
  ``"near"`` (default) -- two or more slots active within ``context_ms`` (the per-word voice score's own context, so
  exactly where that score mixes voices), ``"overlap"`` -- two or more slots active at the same time.
- **The TSE lane** (:class:`TargetSpeechWords`): a module in a lane of its own that reads the microphone and the slot
  activity. For each closed stretch (with ``margin_ms`` of sound on each side, at most ``max_window_ms``) it extracts
  the person's voice, feeds it to its fork of the live recogniser (``asr``: the same recogniser, key terms and
  language; another session of it) and scores it. One record per stretch (``speech.target_words``, ``kind:
  "window"``): the stretch, the extracted span, the words heard in the extracted sound with times, the voice score of
  each counted word heard there (``word_context_ms`` around it) and of the whole extracted span, levels and timings.
  No audio, no embedding. It can ``fork()``: when it falls more than ``lag_ms`` behind, the Pipeline's catch-up
  (docs/catch-up-verification.md) gives the skipped audio to workers -- forks with their own extraction worker and
  recogniser session, sharing the slot activity this module has read -- and their records are committed at their
  capture times. ``max_forks`` = ``workers``.
- **The decision** (:class:`TSEDecision`, math): between ``speaker_check`` and ``terms`` / the verification lane.
  Each utterance waits until the TSE lane is complete past its last word (``Pipeline.complete_until``); a counted word
  inside a stretch is decided from that stretch's record (:data:`RULE`), every other word passes unchanged.

Decision per counted word in a stretch (``tse`` on the word: ``decision``, ``heard``, ``score``, ``window_score``):

- heard in the extracted sound (the same term ending within ``near_ms``) and its score there >= ``threshold`` ->
  **person** (``match`` true, ``decided_by: "tse"``);
- heard, score below ``threshold`` -> **uncertain** (``match`` false, ``reason: "uncertain_overlap"``);
- not heard, and the extracted sound is the person's (whole-span score >= ``threshold``) -> **not the person**
  (``match`` false, ``reason: "tse_not_heard"``);
- not heard, extracted sound not identified as the person -> **uncertain** (``uncertain_overlap``).

A counted word heard only in the extracted sound (scored >= ``threshold``) is added as a provisional utterance of its own
(``decided_by: "tse"``) when ``add_heard`` is on; the verification lane settles it like any other provisional word.
Nothing here is chosen on a clip; every number is a parameter.
"""
from __future__ import annotations

import asyncio
from collections import deque
import time

from ..capabilities import CapabilitySpec, OutputField
from .audio_ring import mono_float, rms_dbfs
from .speaker_verify import centroid, cosine, embed
from .term_events import parse_terms, term_occurrences

SAMPLE_RATE = 16000
FRAME_MS = 10
DETECT = dict(rule='near', context_ms=1500, min_slot_ms=100, keep_ms=120000)
WINDOW = dict(margin_ms=1500, max_window_ms=8000, word_context_ms=1500, min_dbfs=-45.)
RULE = dict(threshold=.5, near_ms=1000, add_heard=True)


def _settings(defaults, given, what):
    given = dict(given or {})
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f'unknown {what} settings {sorted(unknown)}; known: {sorted(defaults)}')
    return dict(defaults, **given)


async def _call(function, *args):
    value = function(*args)
    return await value if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future) else value


# ---------------------------------------------------------------- where voices meet (math)

class MultiVoice:
    """Slot activity (``audio.speaker_activity`` values) -> closed stretches where two or more voices meet.

    Shared by a TSE lane and its catch-up forks: the live module reads the records, every fork asks it.
    """

    def __init__(self, **settings):
        self.settings = _settings(DETECT, settings, 'detection')
        if self.settings['rule'] not in ('near', 'overlap'):
            raise ValueError('detection rule is "near" or "overlap"')
        self.reset()

    def reset(self):
        self.intervals = []          # (session, slot, start_ms, end_ms)
        self.known_from_ms = self.known_until_ms = None
        self.session = None
        self._emitted_until = None   # stretches ending before this were given out already

    def add(self, value):
        """One ``{"diarization": {...}}`` value (or its inner dict)."""
        value = value.get('diarization', value) if isinstance(value, dict) else None
        if not isinstance(value, dict):
            return
        session = value.get('session_id')
        for chunk in value.get('chunks') or ():
            start, end = int(chunk['start_ms']), int(chunk['end_ms'])
            if self.known_from_ms is None:
                self.known_from_ms = start
            self.known_until_ms = end if self.known_until_ms is None else max(self.known_until_ms, end)
            for item in chunk.get('activity') or ():
                for low, high in item.get('intervals_ms') or ():
                    if high > low:
                        self.intervals.append((session, int(item['slot']), int(low), int(high)))
        self.session = session
        horizon = (self.known_until_ms or 0) - self.settings['keep_ms']
        self.intervals = [item for item in self.intervals if item[3] > horizon]

    def _flags(self, start, end):
        """Per 10 ms frame of [start, end): do voices meet there (the rule)."""
        import numpy as np
        count = max(0, (end - start) // FRAME_MS)
        half = self.settings['context_ms'] // 2 if self.settings['rule'] == 'near' else 0
        reach = start - half, start + count * FRAME_MS + half
        slots = {}
        for session, slot, low, high in self.intervals:
            if high > reach[0] and low < reach[1]:
                row = slots.setdefault((session, slot), np.zeros(count + 2 * half // FRAME_MS + 1, bool))
                first = max(0, (low - reach[0]) // FRAME_MS)
                row[first:max(first, (high - reach[0]) // FRAME_MS)] = True
        if count == 0 or len(slots) < 2:
            return np.zeros(count, bool)
        rows = np.stack(list(slots.values()))
        if half == 0:
            return rows[:, :count].sum(axis=0) >= 2
        width = 2 * half // FRAME_MS
        need = max(1, self.settings['min_slot_ms'] // FRAME_MS)
        sums = np.concatenate([np.zeros((rows.shape[0], 1), int), np.cumsum(rows, axis=1)], axis=1)
        window = sums[:, width:width + count] - sums[:, :count]       # each slot's frames within context, per frame
        return (window >= need).sum(axis=0) >= 2

    @property
    def reach_ms(self):
        """How far later activity can still change whether voices meet at a frame."""
        return self.settings['context_ms'] if self.settings['rule'] == 'near' else self.settings['min_slot_ms']

    def scan(self, after_ms=None):
        """``(closed, open_from_ms, clean_until_ms)`` from ``after_ms`` on: closed stretches ``[(start_ms, end_ms)]``
        that no later activity can change, the start of a stretch that may still grow (or None), and up to where
        nothing more can be found when nothing is open (or None)."""
        if self.known_until_ms is None:
            return [], None, None
        reach, known = self.reach_ms, self.known_until_ms
        start = self.known_from_ms if after_ms is None else max(self.known_from_ms, int(after_ms))
        start = max(start, known - self.settings['keep_ms'])
        start -= (start - self.known_from_ms) % FRAME_MS       # on the 10 ms grid of the first chunk
        flags = self._flags(start, known)
        closed, opened, first = [], None, None
        for index, flag in enumerate(list(flags) + [False]):
            if flag and first is None:
                first = index
            elif not flag and first is not None:
                low, high = start + first * FRAME_MS, start + index * FRAME_MS
                first = None
                if index >= len(flags) or high + reach > known:
                    opened = low if opened is None else min(opened, low)
                elif self.settings['rule'] == 'near' or high - low >= self.settings['min_slot_ms']:
                    closed.append((low, high))
        return closed, opened, (known - reach if opened is None else None)

    def stretches(self, after_ms=None):
        return self.scan(after_ms)[0]


# ---------------------------------------------------------------- the TSE lane

def target_words_capability(capability_id='speech.target_words'):
    return CapabilitySpec(
        capability_id, capability_id, 'audio', 'gamst.overlap_tse', '1', (
            OutputField('kind', 'string'), OutputField('reference', 'string'), OutputField('start_ms', 'integer'),
            OutputField('end_ms', 'integer'), OutputField('extract_ms', 'array'), OutputField('words', 'array'),
            OutputField('terms', 'array'), OutputField('score', 'number', required=False),
            OutputField('level_db', 'number', required=False), OutputField('mixture_db', 'number', required=False),
            OutputField('took_ms', 'object'), OutputField('model', 'object')),
        'native', stateful=True, reset_on_gap=True, freshness_ms=60000, interpretation='estimate', devices=('cpu',),
        license='Apache-2.0 adapter; extraction model, recogniser and speaker model per their terms')


class TargetSpeechWords:
    """Pipeline module (own lane): microphone + slot activity in, one ``speech.target_words`` record per stretch.

    ``extractor``: ``extract(mixture, enrollment) -> samples`` (e.g. :class:`gamst.providers.model_workers.RemoteExtractor`
    of an ``audio.target_speech`` candidate); ``open_extractor``: an async factory of another one, for a catch-up fork
    (None: forks share ``extractor``). ``embedder``: the speaker model the voice reference was enrolled with. The
    recogniser is a fork of the Pipeline module ``asr`` (``recogniser``), made at the first stretch.
    """
    owns_process = True

    def __init__(self, extractor, embedder, *, reference='voice', terms='terms', audio='microphone', asr='asr',
                 detect=None, window=None, rule=None, workers=1, open_extractor=None, answer_timeout_s=15.,
                 capability_id='speech.target_words', detector=None, records=None):
        self.extractor, self.embedder, self.open_extractor = extractor, embedder, open_extractor
        self.reference_name, self.terms_name, self.audio, self.asr_name = reference, terms, audio, asr
        self.detector = detector if detector is not None else MultiVoice(**(detect or {}))
        self.window = _settings(WINDOW, window, 'window')
        self.rule = _settings(RULE, rule, 'rule')
        if not 1 <= int(workers) <= 16:
            raise ValueError('workers is 1..16')
        self.workers, self.answer_timeout_s = int(workers), float(answer_timeout_s)
        self.spec = target_words_capability(capability_id)
        self.name, self.version = 'target_speech_words', getattr(extractor, 'model_id', 'unspecified')
        self.pipeline = self.memory = self.enrollment = self.target = self.terms = None
        self._recogniser_name = 'recogniser'
        self._is_fork = detector is not None and records is not None     # a catch-up fork shares both
        self.records = records if records is not None else deque(maxlen=256)
        self._parent = None
        self.stats = dict(stretches=0, extract_ms=0, asr_ms=0, score_ms=0, audio_ms=0, errors=[])
        self.reset()

    # ------------------------------------------------------------------ binding

    def bind(self, pipeline):
        from .target_speech import enrollment_audio
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'voice':
            raise ValueError(f'the TSE lane needs a voice reference named {self.reference_name!r}')
        value = reference['value'] or {}
        if value.get('model') != self.embedder.model_id or 'embeddings' not in value:
            raise ValueError('the voice reference was enrolled with another speaker model than the TSE lane scores with')
        self.target = centroid(value['embeddings'])
        self.enrollment = enrollment_audio(value)       # the owner's clip: --voice-wav (kept in memory only)
        words = pipeline.references.get(self.terms_name)
        if words is None or words['kind'] != 'terms':
            raise ValueError(f'the TSE lane needs the terms reference {self.terms_name!r}')
        self.terms = parse_terms(words['value'])[0]
        if self.asr_name not in pipeline.modules or not callable(getattr(pipeline.modules[self.asr_name]['module'], 'fork', None)):
            raise ValueError(f'the TSE lane hears the extracted voice with a fork of {self.asr_name!r}, a streaming recogniser')
        self.pipeline, self.memory = pipeline, pipeline.keep_audio(self.audio)

    def reset(self):
        if not self._is_fork:
            self.detector.reset()
            self.records.clear()          # every stretch record of this lane and its forks (the decision reads it)
        self._done_until = None       # stretches ending at or before this capture time are done (or not this one's)
        self._heard_until = None      # capture time of the newest audio frame this instance was given
        self._heard_from = None       # ... and where its first one started
        self._entry = None
        self._outbox = []
        for task in list(getattr(self, '_inflight', {}).values()):
            task.cancel()
        self._inflight = {}           # (stretch start, end) -> its hearing task

    # ------------------------------------------------------------------ catch-up

    @property
    def max_forks(self):
        return self.workers

    def _pipeline_entry(self):
        if self._entry is None and self.pipeline is not None:
            self._entry = next((entry for entry in self.pipeline.modules.values() if entry['module'] is self), None)
        return self._entry

    def _claimed(self, low, high):
        """A catch-up worker owns this stretch (its middle lies in a range the Pipeline gave to one)."""
        entry = self._pipeline_entry()
        middle = (low + high) / 2
        return entry is not None and any(first <= middle <= last for first, last in entry['claims'])

    def _owns(self, low, high):
        middle = (low + high) / 2
        if self._is_fork:       # what it heard, up to the end of the range the Pipeline claimed for it (not its tail)
            if self._heard_from is None or not self._heard_from <= middle <= self._heard_until:
                return False
            entry = self._parent._pipeline_entry() if self._parent is not None else None
            ends = [last for first, last in (entry['claims'] if entry else ()) if first <= self._heard_from + 1 <= last + 1]
            return not ends or middle <= max(ends)
        return not self._claimed(low, high)

    def snapshot(self):
        """A fork starts ``margin_ms`` before the first stretch not done or still open (None: at the TODO's start)."""
        closed, opened, _ = self.detector.scan(self._done_until)
        starts = [low for low, high in closed if self._owns(low, high)][:1] + ([opened] if opened is not None else [])
        return dict(pause_ms=None if not starts else min(starts) - self.window['margin_ms'])

    async def fork(self):
        other = TargetSpeechWords(await self.open_extractor() if self.open_extractor else self.extractor, self.embedder,
                                  reference=self.reference_name, terms=self.terms_name, audio=self.audio,
                                  asr=self.asr_name, window=self.window, rule=self.rule, workers=1,
                                  answer_timeout_s=self.answer_timeout_s, capability_id=self.spec.id,
                                  detector=self.detector, records=self.records)
        other._owns_extractor, other._parent = self.open_extractor is not None, self
        other.spec, other.stats = self.spec, self.stats       # one account of the lane's work, forks included
        other.pipeline, other.memory, other.enrollment, other.target, other.terms = \
            self.pipeline, self.memory, self.enrollment, self.target, self.terms
        return other

    def restore(self, state):
        pass

    async def skipped(self, start_ms, end_ms):
        """The Pipeline gave [.., end_ms] to a worker (its claim): this module leaves the stretches there to it."""

    @property
    def pending_since_ms(self):
        """Sound whose stretch is not decided yet: an open stretch, or a closed one this module still owes."""
        if self._heard_until is None:
            return None
        if self.detector.known_until_ms is None:       # no slot activity yet: nothing heard can be called clear
            return int(self._heard_from)
        closed, opened, _ = self.detector.scan(self._done_until)
        bounds = [low for low, high in closed if self._owns(low, high)] + ([opened] if opened is not None else []) + \
            [low for low, _ in list(self._inflight)]
        # The diarization says who is active later than it says it heard the sound: what it has not covered (plus the
        # reach of the rule) may still turn out to be a stretch.
        unknown = self.detector.known_until_ms - self.detector.reach_ms
        held = min(bounds) - self.window['margin_ms'] if bounds else None
        return unknown if held is None else min(held, unknown)

    # ------------------------------------------------------------------ frames

    async def process(self, frame):
        value = frame.data
        if isinstance(value, dict):
            if isinstance(value.get('diarization'), dict) and not self._is_fork:
                self.detector.add(value)
        elif frame.modality == 'audio' and value is not None:
            start = frame.timestamp_ms - 1000. * len(value) / (frame.sample_rate or SAMPLE_RATE)
            self._heard_from = start if self._heard_from is None else self._heard_from
            self._heard_until = frame.timestamp_ms
        await self._work()
        return self._take()

    async def finish(self):
        if self._is_fork:     # the live diarization runs ahead of a worker: wait (bounded) until its stretches close
            deadline = time.monotonic() + 30
            while self._heard_until is not None and time.monotonic() < deadline and \
                    (self.detector.known_until_ms or 0) < self._heard_until + self.detector.reach_ms:
                await asyncio.sleep(.05)
        await self._work(final=True)
        while self._inflight:
            await asyncio.gather(*list(self._inflight.values()), return_exceptions=True)
        return self._take()

    def _take(self):
        out, self._outbox = self._outbox, []
        return out or None

    async def _work(self, final=False):
        if self._heard_until is None or self.memory is None:
            return
        closed, opened, clean = self.detector.scan(self._done_until)
        owed = False
        for low, high in closed:
            if not self._owns(low, high):
                if self._is_fork and (low + high) / 2 < self._heard_from:
                    self._done_until = high if self._done_until is None else max(self._done_until, high)
                continue
            if not final and (high + self.window['margin_ms'] > (self.memory.end_ms or 0)
                              or (low + high) / 2 > self._heard_until):
                owed = True
                break                 # its sound is not all here (or not heard by this instance) yet
            for first in range(low, high, self.window['max_window_ms']):
                await self._stretch(first, min(high, first + self.window['max_window_ms']))
            self._done_until = high if self._done_until is None else max(self._done_until, high)
        if not owed and clean is not None and not self._is_fork and not any(
                self._owns(low, high) and high > (self._done_until or float('-inf')) for low, high in closed):
            self._done_until = max(self._done_until or clean, clean)     # nothing there: scan from here on

    # ------------------------------------------------------------------ one stretch

    async def _stretch(self, low, high):
        """Extract the person (in the lane: one model at a time), then hear and score it in a task of its own, so the
        next stretch is extracted while the recogniser answers (``inflight``; it holds the lane's completeness)."""
        import numpy as np
        margin = self.window['margin_ms']
        first = max(int(self.memory.start_ms or low), low - margin)
        last = min(int(self.memory.end_ms or high), high + margin)
        mixture = self.memory.span(first, last, SAMPLE_RATE)
        began = time.monotonic()
        extracted = np.asarray(await _call(self.extractor.extract, mixture, self.enrollment), dtype=np.float32)
        extract_ms = 1000 * (time.monotonic() - began)
        self.stats['extract_ms'] += extract_ms
        self.stats['audio_ms'] += last - first
        key = (int(low), int(high))
        self._inflight[key] = asyncio.ensure_future(self._heard(key, first, last, mixture, extracted, extract_ms))

    async def _heard(self, key, first, last, mixture, extracted, extract_ms):
        import numpy as np
        low, high = key
        try:
            began = time.monotonic()
            heard = await self._hear(extracted, first)
            asr_ms = 1000 * (time.monotonic() - began)
            began = time.monotonic()
            words = [dict(text=str(word['text']), start_ms=int(word['start_ms']), end_ms=int(word['end_ms']))
                     for value in heard for word in value.get('words') or () if type(word.get('start_ms')) is int]
            found = []
            for value in heard:
                for item in term_occurrences(value, self.terms):
                    score = await self._score(extracted, first, item['start_ms'], item['end_ms'])
                    found.append(dict(term=item['term'], spoken=item['spoken'], start_ms=item['start_ms'],
                                      end_ms=item['end_ms'], **({} if score is None else {'score': score})))
            whole = await self._score(extracted, first, low, high, whole=True)
            score_ms = 1000 * (time.monotonic() - began)
            inner = slice(int((low - first) * SAMPLE_RATE / 1000), int((high - first) * SAMPLE_RATE / 1000))
            level, mixed = rms_dbfs(extracted[inner]), rms_dbfs(mixture[inner])
            self.stats['asr_ms'] += asr_ms
            self.stats['score_ms'] += score_ms
            self.stats['stretches'] += 1
            value = dict(kind='window', reference=self.reference_name, start_ms=int(low), end_ms=int(high),
                         extract_ms=[int(first), int(last)], words=words, terms=found,
                         took_ms=dict(extract=round(extract_ms), asr=round(asr_ms), score=round(score_ms)),
                         model=dict(extractor=getattr(self.extractor, 'model_id', 'extractor'),
                                    recogniser=self._recogniser_name, speaker=self.embedder.model_id,
                                    **{f'detect_{name}': item for name, item in self.detector.settings.items()},
                                    **self.window, **{f'rule_{name}': item for name, item in self.rule.items()}))
            if whole is not None:
                value['score'] = whole
            if np.isfinite(level):
                value['level_db'] = round(float(level), 1)
            if np.isfinite(mixed):
                value['mixture_db'] = round(float(mixed), 1)
            record = self.spec.validate(value)
            self.records.append(record)       # before it stops holding the lane's completeness
            self._outbox.append(record)
        except asyncio.CancelledError:
            raise
        except Exception as error:      # said in status; the words there keep their own decision
            self.stats['failed'] = self.stats.get('failed', 0) + 1
            self.stats['errors'] = (self.stats['errors'] + [f'{type(error).__name__}: {error}'[:300]])[-20:]
        finally:
            self._inflight.pop(key, None)

    async def _score(self, extracted, first, start, end, whole=False):
        """Voice score of the extracted sound: ``word_context_ms`` around a word, or the whole stretch."""
        if whole:
            low, high = start - self.window['margin_ms'], end + self.window['margin_ms']
        else:
            middle, half = (start + end) // 2, max(self.window['word_context_ms'], end - start) // 2
            low, high = middle - half, middle + half
        a = max(0, int((low - first) * SAMPLE_RATE / 1000))
        b = min(len(extracted), int((high - first) * SAMPLE_RATE / 1000))
        samples = extracted[a:b]
        if len(samples) < SAMPLE_RATE // 4 or rms_dbfs(samples) < self.window['min_dbfs']:
            return None
        try:
            return round(cosine(await embed(self.embedder, samples, SAMPLE_RATE), self.target), 5)
        except ValueError:
            return None

    async def _hear(self, samples, first_ms):
        """The extracted sound through a fork of the live recogniser (a session of its own per stretch, so stretches
        are heard side by side and no session idles between them); its final utterances. A session that fails is
        opened once more and the stretch heard again (``reopened`` in status)."""
        for attempt in (0, 1):
            recogniser = await _call(self.pipeline.modules[self.asr_name]['module'].fork)
            recogniser.name = self._recogniser_name = f"{getattr(recogniser, 'name', 'asr')}.tse"
            try:
                return await self._hear_once(recogniser, samples, first_ms)
            except Exception:
                if attempt:
                    raise
                self.stats['reopened'] = self.stats.get('reopened', 0) + 1
            finally:
                try:
                    await _call(recogniser.close)
                except Exception:
                    pass

    async def _hear_once(self, recogniser, samples, first_ms):
        from ..capabilities import SensorMetadata
        from ..plugins import InputFrame
        out, size = [], SAMPLE_RATE // 10
        for index in range(0, len(samples), size):
            chunk = samples[index:index + size]
            end_ms = int(round(first_ms + 1000 * (index + len(chunk)) / SAMPLE_RATE))
            frame = InputFrame(mono_float(chunk), 'audio', end_ms, 'local-user', SAMPLE_RATE, SensorMetadata(
                source_id='gamst.tse', sensor_id=self.audio, sequence=end_ms, channels=('mono',)))
            out += await recogniser.process(frame) or []
        commit = getattr(recogniser, '_commit', None)
        if callable(commit):
            await commit()
        deadline = time.monotonic() + self.answer_timeout_s
        while getattr(recogniser, 'pending_since_ms', None) is not None and time.monotonic() < deadline:
            await asyncio.sleep(.05)
            out += getattr(recogniser, '_take', lambda: None)() or []
        out += getattr(recogniser, '_take', lambda: None)() or []
        errors = getattr(recogniser, 'errors', None)
        if errors and not out and any('ConnectionClosed' in str(item) for item in errors[-3:]):
            raise ConnectionError(str(errors[-1])[:200])
        return [value for value in out if isinstance(value, dict) and value.get('status') == 'final']

    def status(self):
        return dict({key: (round(value) if isinstance(value, float) else value) for key, value in self.stats.items()
                     if key != 'errors'}, errors=self.stats['errors'][-5:], done_until_ms=self._done_until,
                    known_until_ms=self.detector.known_until_ms)

    async def close(self):
        for task in list(self._inflight.values()):
            task.cancel()
        if getattr(self, '_owns_extractor', False) and callable(getattr(self.extractor, 'close', None)):
            await _call(self.extractor.close)
        elif not self._is_fork and callable(getattr(self.extractor, 'close', None)):
            await _call(self.extractor.close)


# ---------------------------------------------------------------- the decision (math)

def decide(word_end_ms, term, record, rule):
    """(decision, match, reason, heard occurrence or None) of one counted word inside a TSE stretch (module doc)."""
    near, threshold = rule['near_ms'], rule['threshold']
    heard = [item for item in record.get('terms') or () if item['term'] == term and abs(item['end_ms'] - word_end_ms) <= near]
    best = min(heard, key=lambda item: abs(item['end_ms'] - word_end_ms)) if heard else None
    if best is not None:
        if best.get('score') is not None and best['score'] >= threshold:
            return 'person', True, None, best
        return 'uncertain', False, 'uncertain_overlap', best
    if record.get('score') is not None and record['score'] >= threshold:
        return 'other', False, 'tse_not_heard', None
    return 'uncertain', False, 'uncertain_overlap', None


class TSEDecision:
    """Pipeline module (math): ``speaker_check`` utterances (+ audio as ticks) in; the same utterances out, their counted
    words inside a stretch decided by the TSE lane (module doc). It reads the lane's records from the lane itself
    (``TargetSpeechWords.records``: its forks' too), and each utterance waits until ``Pipeline.complete_until`` of the
    lane has passed its last word, so a record that is still on its way can never be missed."""

    def __init__(self, *, tse='tse', reference='voice', terms='terms', rule=None,
                 capability_id='speech.speaker_check_tse'):
        self.tse_name, self.reference_name, self.terms_name = tse, reference, terms
        self.rule = _settings(RULE, rule, 'rule')
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'audio', 'gamst.overlap_tse', '1', (
                OutputField('kind', 'string'), OutputField('reference', 'string'), OutputField('start_ms', 'integer'),
                OutputField('end_ms', 'integer'), OutputField('score', 'number', required=False),
                OutputField('match', 'boolean'), OutputField('state', 'string'),
                OutputField('utterance_id', 'string', required=False), OutputField('status', 'string', required=False),
                OutputField('text', 'string', required=False), OutputField('words', 'array', required=False),
                OutputField('language', 'string', required=False), OutputField('speaker', 'object', required=False),
                OutputField('model', 'object')),
            'native', stateful=True, freshness_ms=60000, interpretation='estimate', devices=('cpu',), license='Apache-2.0')
        self.pipeline = self.terms = self.lane = None
        self.name, self.version = 'tse_decision', '1'
        self.reset()

    def bind(self, pipeline):
        words = pipeline.references.get(self.terms_name)
        if words is None or words['kind'] != 'terms':
            raise ValueError(f'the TSE decision needs the terms reference {self.terms_name!r}')
        self.terms = parse_terms(words['value'])[0]
        lane = (pipeline.modules.get(self.tse_name) or {}).get('module')
        if not isinstance(getattr(lane, 'records', None), deque):
            raise ValueError(f'the TSE decision reads the TSE lane {self.tse_name!r} (add it first)')
        self.pipeline, self.lane = pipeline, lane

    def reset(self):
        self._waiting, self._added = [], set()
        self.log = deque(maxlen=64)
        self.stats = dict(decided=0, person=0, other=0, uncertain=0, added=0, waited_ms=[])

    @property
    def _records(self):
        return list(self.lane.records) if self.lane is not None else []

    @property
    def pending_since_ms(self):
        return min((value['start_ms'] for value, _ in self._waiting), default=None)

    def _complete(self):
        value = self.pipeline.complete_until(self.tse_name) if self.pipeline is not None else None
        return float('-inf') if value is None else value

    def process(self, frame):
        value = frame.data
        if isinstance(value, dict) and value.get('kind') == 'utterance' and value.get('status') == 'final' \
                and 'extract_ms' not in value:
            self._waiting.append((value, time.monotonic()))
        return self._release(False)

    def finish(self):
        return self._release(True)

    def _release(self, final):
        out, complete = [], self._complete()
        while self._waiting:
            value, since = self._waiting[0]
            last = max([word.get('end_ms', value['end_ms']) for word in value.get('words') or () if isinstance(word, dict)]
                       + [value['end_ms']])
            if not final and complete < last and self._needs_lane(value):
                break
            self._waiting.pop(0)
            self.log.append(dict(end_ms=last, complete_ms=None if complete == float('-inf') else complete,
                                 records=len(self._records)))
            self.stats['waited_ms'] = (self.stats['waited_ms'] + [round(1000 * (time.monotonic() - since))])[-512:]
            out.append(self._decide(value))
        out += self._heard_only(complete if not final else float('inf'))
        return [self.spec.validate(value) for value in out] or None

    def _needs_lane(self, value):
        """Whether a counted word of ``value`` may lie in a stretch: else it passes at once, without waiting for the lane
        (words that are no counted term are never decided here; a stretch the detector closed does not change)."""
        detector = getattr(self.lane, 'detector', None)
        found = term_occurrences(value, self.terms)
        if not found:
            return False
        if detector is None or detector.known_until_ms is None:
            return True
        if any(item['end_ms'] + detector.reach_ms > detector.known_until_ms for item in found):
            return True
        closed, opened, _ = detector.scan(min(item['start_ms'] for item in found) - detector.reach_ms)
        spans = closed + ([(opened, float('inf'))] if opened is not None else [])
        return any(low < max(item['end_ms'], item['start_ms'] + 1) and high > item['start_ms']
                   for item in found for low, high in spans)

    def _record_at(self, start, end):
        over = [record for record in self._records if record['start_ms'] < end and record['end_ms'] > start]
        return max(over, key=lambda record: min(end, record['end_ms']) - max(start, record['start_ms'])) if over else None

    def _decide(self, value):
        words = list(value.get('words') or ())
        changed = False
        for item in term_occurrences(value, self.terms):
            record = self._record_at(item['start_ms'], max(item['end_ms'], item['start_ms'] + 1))
            if record is None:
                continue
            decision, match, reason, heard = decide(item['end_ms'], item['term'], record, self.rule)
            if heard is not None:
                self._added.add((record['start_ms'], heard['end_ms'], heard['term']))
            self.stats['decided'] += 1
            self.stats[decision] += 1
            evidence = dict(decision=decision, heard=heard is not None, window=[record['start_ms'], record['end_ms']],
                            **({'score': heard['score']} if heard is not None and heard.get('score') is not None else {}),
                            **({'window_score': record['score']} if record.get('score') is not None else {}))
            for index, word in enumerate(words):
                if isinstance(word, dict) and any(word is covered or word == covered for covered in item['words']):
                    updated = {key: val for key, val in word.items() if key != 'reason'}
                    updated.update(match=match, decided_by='tse', basis=f'tse_{decision}', tse=evidence,
                                   **({'reason': reason} if reason else {}))
                    if 'speaker_score' in word:
                        updated['mixture_score'] = word['speaker_score']
                    words[index] = updated
                    changed = True
        if not changed:
            return value
        return dict(value, words=words)

    def _heard_only(self, complete):
        """Counted words the extracted voice said (scored as the person) that the live recogniser did not write there."""
        if not self.rule['add_heard']:
            return []
        out = []
        for record in self._records:
            if record['end_ms'] > complete:
                continue
            for heard in record.get('terms') or ():
                key = (record['start_ms'], heard['end_ms'], heard['term'])
                if key in self._added or not (record['start_ms'] <= heard['end_ms'] <= record['end_ms'] + self.rule['near_ms']):
                    continue
                if any(abs(waiting_end - heard['end_ms']) <= self.rule['near_ms']
                       for value, _ in self._waiting for item in term_occurrences(value, self.terms)
                       for waiting_end in (item['end_ms'],)):
                    continue      # the live words there are not decided yet
                self._added.add(key)
                if heard.get('score') is None or heard['score'] < self.rule['threshold']:
                    continue
                self.stats['added'] += 1
                utterance_id = f"tse:{record['start_ms']}:{heard['end_ms']}"
                word = dict(text=heard['spoken'], start_ms=heard['start_ms'],
                            end_ms=max(heard['end_ms'], heard['start_ms'] + 1), match=True, decided_by='tse',
                            basis='tse_heard_only', speaker_score=heard['score'], word_id=f'{utterance_id}#0',
                            state='provisional', tse=dict(decision='person', heard=True, score=heard['score'],
                                                          window=[record['start_ms'], record['end_ms']]))
                out.append(dict(kind='utterance', reference=self.reference_name, start_ms=word['start_ms'],
                                end_ms=word['end_ms'], match=True, state='match', utterance_id=utterance_id,
                                status='final', text=heard['spoken'], words=[word],
                                speaker=dict(reference=self.reference_name, match=True, model='tse',
                                             score=heard['score']), model=dict(id='tse', threshold=self.rule['threshold'])))
        return out

    def status(self):
        waited = sorted(self.stats['waited_ms'])
        return dict({key: value for key, value in self.stats.items() if key != 'waited_ms'}, waiting=len(self._waiting),
                    released=list(self.log),
                    waited_ms_p50=waited[len(waited) // 2] if waited else None,
                    waited_ms_p95=waited[int(.95 * (len(waited) - 1))] if waited else None)
