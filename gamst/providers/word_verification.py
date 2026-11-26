"""The verification lane: a second hearing of every complete stretch, which settles the words counted at once.

Owner decision (docs/catch-up-verification.md). Words are counted at once (``provisional``) from the live recogniser
and the per-word voice score; this lane runs behind them, in capture order, and decides each word once, as
``final`` or ``revoked`` with a reason, through the same revision states as every other module
(:mod:`gamst.counting`, :class:`gamst.providers.term_events.TermEvents`).

It works on a stretch of capture time only after the Pipeline says that stretch is complete for the words it reads
(``Pipeline.inputs_complete_until``: no catch-up TODO and no unanswered turn is left there), so a catch-up worker can
never change a word after it was settled. For each stretch it has:

- **a second recogniser** (``recogniser``, a :class:`gamst.providers.streaming_asr.StreamingWords`: by default
  ``gpt-live-transcribe`` with the counted words as keyword hints and Qwen3-ForcedAligner word times). It is fed the
  kept audio of the input (``Pipeline.keep_audio``) as far as it is complete, as fast as it takes it;
- **diarization of a window around it** (``diarizer.diarize(samples) -> [(start_s, end_s, speaker)]``, by default
  pyannoteAI Precision-3), each speaker's own sound (where only that speaker is active) scored against the owner's
  few-shot voice with the speaker model (CAM++): speakers whose score reaches ``threshold`` are the person. Windows
  overlap (``before_ms`` of earlier sound, ``after_ms`` of later sound around each ``step_ms`` decided);
- **the word rule** between the two recognisers (:data:`RULE`): a provisional occurrence of a counted word is
  confirmed when the second recogniser heard the same word within ``near_ms`` of it (one to one, nearest first),
  revoked (``not_heard``) when it heard speech there but not the word, revoked (``no_speech``) when nobody spoke there;
  an occurrence only the second recogniser heard is added, final;
- **the voice rule** (:data:`VOICE`) for each word that stays: the person's speaker active over the word and no other
  speaker over ``overlap_share`` of it -> the person; another speaker dominant -> another voice; both over
  ``overlap_share`` (overlapped speech) -> the TSE lane's decision where the word carries one
  (:mod:`gamst.providers.overlap_tse`: the person's voice extracted from the mixture, heard and scored), else the
  per-word voice score decides, and below ``threshold`` the word is ``uncertain_overlap``: not counted, and said so. With no speaker active, or an unidentified one, the per-word score
  decides.

Outputs (capability ``speech.word_verdicts``): ``kind: "verdicts"`` (``[{word_id, match, provisional, changed, basis,
reason?, heard, speaker, speaker_score, word_score}]`` for every provisional word), ``kind: "utterance"`` (an
occurrence only the second recogniser heard: a final utterance of that word, ``decided_by: "verification"``) and
``kind: "window"`` (each window's speakers, their speech and score; no audio, no embedding). Nothing here is tuned on
a clip; every setting is a parameter.
"""
from __future__ import annotations

import asyncio
import time

from ..capabilities import CapabilitySpec, OutputField
from .audio_ring import rms_dbfs
from .speaker_verify import centroid, cosine, embed, l2_normalize
from .term_events import parse_terms, term_occurrences

SAMPLE_RATE = 16000
RULE = dict(near_ms=1000, revoke_not_heard=True, revoke_no_speech=True)
VOICE = dict(threshold=.5, min_speaker_ms=1000, piece_ms=2000, overlap_share=.3, active_share=.1, word_context_ms=1500,
             min_dbfs=-45.)
WINDOWS = dict(step_ms=15000, before_ms=15000, after_ms=5000, concurrency=2)


def _settings(defaults, given, what):
    given = dict(given or {})
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f'unknown {what} settings {sorted(unknown)}; known: {sorted(defaults)}')
    return dict(defaults, **given)


def _share(turns, start, end):
    """{speaker: share of [start, end) it is active}."""
    span = max(1, end - start)
    shares = {}
    for low, high, speaker in turns:
        shared = min(end, high) - max(start, low)
        if shared > 0:
            shares[speaker] = shares.get(speaker, 0) + shared
    return {speaker: min(1., value / span) for speaker, value in shares.items()}


def exclusive(turns, start, end):
    """{speaker: [(start, end)]} of [start, end) where that speaker alone is active."""
    edges = sorted({start, end, *(max(start, min(end, at)) for low, high, _ in turns for at in (low, high))})
    found = {}
    for low, high in zip(edges, edges[1:]):
        active = {speaker for first, last, speaker in turns if first < high and last > low}
        if len(active) == 1:
            speaker = active.pop()
            pieces = found.setdefault(speaker, [])
            if pieces and pieces[-1][1] == low:
                pieces[-1] = (pieces[-1][0], high)
            else:
                pieces.append((low, high))
    return found


def match_occurrences(first, second, near_ms):
    """One-to-one pairs of occurrences of the same term whose ends are within ``near_ms``, nearest first:
    ``{index in first: index in second}``."""
    pairs = sorted((abs(a['end_ms'] - b['end_ms']), i, j) for i, a in enumerate(first) for j, b in enumerate(second)
                   if a['term'] == b['term'] and abs(a['end_ms'] - b['end_ms']) <= near_ms)
    taken, used = {}, set()
    for _, i, j in pairs:
        if i not in taken and j not in used:
            taken[i] = j
            used.add(j)
    return taken


class WordVerification:
    """Pipeline module (its own lane): reads the audio input (as ticks) and the provisional words; see the module doc."""
    owns_process = True

    def __init__(self, recogniser, diarizer, embedder, *, reference='voice', terms='terms', audio='microphone',
                 rule=None, voice=None, windows=None, feed_speed=1.25, capability_id='speech.word_verdicts'):
        self.recogniser, self.diarizer, self.embedder = recogniser, diarizer, embedder
        self.reference_name, self.terms_name, self.audio = reference, terms, audio
        self.rule, self.voice = _settings(RULE, rule, 'rule'), _settings(VOICE, voice, 'voice')
        self.windows = _settings(WINDOWS, windows, 'window')
        # The second recogniser is a realtime API: it gets the complete audio at most ``feed_speed`` times real time,
        # evenly (a little faster than real time, so it catches up after a stall); None: as soon as it is complete
        self.feed_speed = None if feed_speed is None else float(feed_speed)
        if not 0 < self.windows['step_ms'] <= 120000 or self.windows['before_ms'] < 0 or self.windows['after_ms'] < 0:
            raise ValueError('windows need 0 < step_ms <= 120000 and non-negative before_ms, after_ms')
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'audio', 'gamst.word_verification', '1', (
                OutputField('kind', 'string'), OutputField('reference', 'string'), OutputField('start_ms', 'integer'),
                OutputField('end_ms', 'integer'), OutputField('verdicts', 'array', required=False),
                OutputField('utterance_id', 'string', required=False), OutputField('status', 'string', required=False),
                OutputField('text', 'string', required=False), OutputField('words', 'array', required=False),
                OutputField('speaker', 'object', required=False), OutputField('speakers', 'array', required=False),
                OutputField('model', 'object')),
            'native', stateful=True, reset_on_gap=True, freshness_ms=120000, interpretation='estimate',
            devices=('cpu',), license='Apache-2.0 adapter; recogniser, diarization and speaker model per their terms')
        self.pipeline = self.target = self.terms = self.memory = None
        self._name = None
        self._task = None
        self.reset()

    # ------------------------------------------------------------------ binding and state

    def bind(self, pipeline):
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'voice' or 'embeddings' not in (reference['value'] or {}):
            raise ValueError(f'verification needs an enrolled voice reference named {self.reference_name!r}')
        if reference['value'].get('model') != self.embedder.model_id:
            raise ValueError('the voice reference was enrolled with another speaker model than verification runs')
        self.target = centroid(reference['value']['embeddings'])
        words = pipeline.references.get(self.terms_name)
        if words is None or words['kind'] != 'terms':
            raise ValueError(f'verification needs the terms reference {self.terms_name!r}')
        self.terms = parse_terms(words['value'])[0]
        self.pipeline, self.memory = pipeline, pipeline.keep_audio(self.audio)

    def reset(self):
        if self._task is not None:
            self._task.cancel()
        self._task, self._final, self._done = None, False, False
        self._origin = None
        self._fed = None               # capture time up to which the second recogniser was given the audio
        self._utterances = []          # first-hearing utterances with provisional words, in arrival order
        self._open = {}                # word_id -> provisional word, until its verdict
        self._heard = []               # the second recogniser's final utterances
        self._windows = {}             # index -> {"turns", "speakers", "person"} | asyncio task
        self._decided = 0              # regions decided so far
        self._used = set()             # ids of second-hearing occurrences that confirmed or were added
        self._outbox, self._scores = [], {}
        self.stats = dict(regions=0, verdicts=0, added=0, late=0, windows=0, diarize_ms=[], decided_after_ms=[],
                          fed_ms=0, errors=[])

    @property
    def pending_since_ms(self):
        return min((word['start_ms'] for word in self._open.values()), default=None)

    def _model(self):
        return dict(recogniser=getattr(self.recogniser, 'name', 'recogniser'), speaker=self.embedder.model_id,
                    diarizer=getattr(self.diarizer, 'name', type(self.diarizer).__name__), **self.rule,
                    **{f'voice_{key}': value for key, value in self.voice.items()},
                    **{f'window_{key}': value for key, value in self.windows.items()})

    def status(self):
        return dict({key: value for key, value in self.stats.items() if not isinstance(value, list)},
                    errors=self.stats['errors'][-5:], open=len(self._open), heard=len(self._heard),
                    heard_terms=[dict(end_ms=item['end_ms'], timed=item['timed'], aligned=bool(value.get('aligned')))
                                 for value in self._heard for item in term_occurrences(value, self.terms or {})][-64:],
                    diarize_ms=self.stats['diarize_ms'][-16:],
                    recogniser_errors=[str(item)[:200] for item in getattr(self.recogniser, 'errors', ())][-8:],
                    heard_turns=[dict(start_ms=value['start_ms'], end_ms=value['end_ms'], words=len(value.get('words') or ()),
                                      aligned=bool(value.get('aligned'))) for value in self._heard][-64:],
                    fed_until_ms=self._fed, decided_until_ms=None if self._origin is None else
                    self._origin + self._decided * self.windows['step_ms'])

    # ------------------------------------------------------------------ frames

    def process(self, frame):
        value = frame.data
        if isinstance(value, dict):
            if value.get('kind') == 'utterance' and value.get('status') == 'final' and \
                    value.get('reference', self.reference_name) == self.reference_name:
                words = [word for word in value.get('words') or () if isinstance(word, dict)
                         and word.get('state') == 'provisional' and isinstance(word.get('word_id'), str)]
                if words:
                    self._utterances.append(value)
                    for word in words:
                        self._open[word['word_id']] = dict(word, utterance_id=value.get('utterance_id'))
                    self._late()
        elif frame.modality == 'audio' and self._origin is None and self.memory is not None and self.memory.start_ms:
            self._origin = int(self.memory.start_ms)
        if self._task is None and self._origin is not None and not self._done:
            self._task = asyncio.ensure_future(self._work())
        return self._take()

    async def finish(self):
        self._final = True
        if self._origin is None and self.memory is not None and self.memory.start_ms is not None:
            self._origin = int(self.memory.start_ms)
        if self._origin is not None:
            if self._task is None or self._task.done():
                self._task = asyncio.ensure_future(self._work())
            await self._task
        return self._take()

    def _take(self):
        out, self._outbox = self._outbox, []
        return out or None

    async def close(self):
        if self._task is not None:
            self._task.cancel()
        for window in self._windows.values():
            if isinstance(window, asyncio.Task):
                window.cancel()
        close = getattr(self.recogniser, 'close', None)
        if callable(close):
            value = close()
            if asyncio.iscoroutine(value):
                await value

    # ------------------------------------------------------------------ the lane's work, in capture order

    def _complete(self):
        if self._final:
            return float('inf')
        if self._name is None:
            self._name = next((key for key, entry in self.pipeline.modules.items() if entry['module'] is self), None)
        value = self.pipeline.inputs_complete_until(self._name) if self._name else None
        return float('-inf') if value is None else value

    def _region(self, index):
        step = self.windows['step_ms']
        return self._origin + index * step, self._origin + (index + 1) * step

    def _window(self, index):
        low, high = self._region(index)
        return max(self._origin, low - self.windows['before_ms']), high + self.windows['after_ms']

    async def _work(self):
        try:
            while True:
                complete = self._complete()
                progressed = await self._feed(complete)
                progressed = self._diarize(complete) or progressed
                progressed = await self._decide(complete) or progressed
                self._hold()
                if self._final and self._origin is not None and self.memory.end_ms is not None \
                        and self._region(self._decided)[0] > self.memory.end_ms:
                    self._late()          # anything still open is decided now, and counted as late
                if self._final and self._finished():
                    self._done = True
                    return
                if not progressed:
                    await asyncio.sleep(.1 if not self._final else .05)
        except asyncio.CancelledError:
            raise
        except Exception as error:      # said in status; what is open stays provisional
            self.stats['errors'] = (self.stats['errors'] + [f'{type(error).__name__}: {error}'[:300]])[-20:]
            self._task = None
            raise

    def _finished(self):
        end = self.memory.end_ms
        return end is not None and self._fed is not None and self._fed >= end and not self._open and \
            self._region(self._decided)[0] > end

    async def _feed(self, complete):
        """Give the second recogniser the kept audio as far as it is complete; take what it answered."""
        from ..plugins import InputFrame
        from ..capabilities import SensorMetadata
        progressed, position = False, self._origin if self._fed is None else self._fed
        while True:
            chunk = self.memory.after(position)
            if chunk is None or chunk[0] > complete:
                break
            if self.feed_speed and not self._final:
                if getattr(self, '_pace', None) is None:
                    self._pace = (time.monotonic(), position)
                due = self._pace[0] + (chunk[0] - self._pace[1]) / 1000 / self.feed_speed
                if time.monotonic() < due:
                    break
            elif self.feed_speed and self._final and getattr(self, '_pace', None) is not None:
                due = self._pace[0] + (chunk[0] - self._pace[1]) / 1000 / self.feed_speed
                if time.monotonic() < due:
                    await asyncio.sleep(min(.1, due - time.monotonic()))
                    continue
            end_ms, samples, rate, subject = chunk
            begin = end_ms - 1000. * len(samples) / rate
            if begin < position:
                samples = samples[int(round((position - begin) * rate / 1000)):]
            if len(samples):
                frame = InputFrame(samples, 'audio', end_ms, subject, rate, SensorMetadata(
                    source_id='gamst.verification', sensor_id=self.audio, sequence=int(end_ms), channels=('mono',)))
                self._heard += [item for item in (await self.recogniser.process(frame) or ()) if item.get('status') == 'final']
                self.stats['fed_ms'] += round(1000 * len(samples) / rate)
            position = self._fed = end_ms
            progressed = True
        if self._final and self.memory.end_ms is not None and self._fed is not None and self._fed >= self.memory.end_ms \
                and not getattr(self, '_flushed', False):
            self._flushed = True
            self._heard += [item for item in (await self.recogniser.finish() or ()) if item.get('status') == 'final']
            progressed = True
        taken = getattr(self.recogniser, '_take', None)
        if callable(taken):
            more = [item for item in (taken() or ()) if item.get('status') == 'final']
            self._heard += more
            progressed = progressed or bool(more)
        return progressed

    def _heard_until(self):
        """Up to where the second recogniser's words are all out."""
        if getattr(self, '_flushed', False):
            return float('inf')
        if self._fed is None:
            return float('-inf')
        pending = getattr(self.recogniser, 'pending_since_ms', None)
        return self._fed if pending is None else min(self._fed, pending)

    def _diarize(self, complete):
        """Start the diarization of every window whose sound is complete (at most ``concurrency`` at once)."""
        started, index = False, self._decided
        end = self.memory.end_ms
        while True:
            low, high = self._window(index)
            if end is None or low >= end:
                break
            if high > complete and not (self._final and complete == float('inf')):
                break
            if index not in self._windows:
                running = sum(isinstance(item, asyncio.Task) and not item.done() for item in self._windows.values())
                if running >= self.windows['concurrency']:
                    break
                self._windows[index] = asyncio.ensure_future(self._identify(index, low, min(high, end)))
                started = True
            index += 1
        return started

    async def _identify(self, index, low, high):
        """Diarization of one window, and which of its speakers is the person (each one's own sound vs the voice)."""
        samples = self.memory.span(low, high, SAMPLE_RATE)
        began = time.monotonic()
        segments = await asyncio.to_thread(self.diarizer.diarize, samples) if len(samples) >= SAMPLE_RATE else []
        self.stats['diarize_ms'].append(round(1000 * (time.monotonic() - began)))
        self.stats['windows'] += 1
        turns = [(low + int(round(1000 * start)), low + int(round(1000 * end)), str(speaker)[:40])
                 for start, end, speaker in segments if end > start]
        speakers = []
        for speaker, pieces in exclusive(turns, low, high).items():
            speech = sum(end - start for start, end in pieces)
            score = await self._speaker_score(pieces) if speech >= self.voice['min_speaker_ms'] else None
            speakers.append(dict(speaker=speaker, speech_ms=int(speech), score=score,
                                 person=score is not None and score >= self.voice['threshold']))
        for speaker in {turn[2] for turn in turns} - {item['speaker'] for item in speakers}:
            speakers.append(dict(speaker=speaker, speech_ms=0, score=None, person=False))
        result = dict(turns=turns, speakers=speakers, low=low, high=high,
                      person={item['speaker'] for item in speakers if item['person']},
                      unknown={item['speaker'] for item in speakers if item['score'] is None})
        self._outbox.append(self.spec.validate(dict(
            kind='window', reference=self.reference_name, start_ms=int(low), end_ms=int(high),
            speakers=sorted(speakers, key=lambda item: -item['speech_ms']), model=self._model())))
        return result

    async def _speaker_score(self, pieces):
        import numpy as np
        sound = [self.memory.span(start, end, SAMPLE_RATE) for start, end in pieces]
        sound = [part for part in sound if len(part)]
        if not sound:
            return None
        sound = np.concatenate(sound)
        size = SAMPLE_RATE * self.voice['piece_ms'] // 1000
        vectors = []
        for first in range(0, len(sound), size):
            piece = sound[first:first + size]
            if len(piece) < SAMPLE_RATE // 2 or rms_dbfs(piece) < self.voice['min_dbfs']:
                continue
            try:
                vectors.append((l2_normalize(await embed(self.embedder, piece, SAMPLE_RATE)), len(piece)))
            except ValueError:
                continue
        if not vectors:
            return None
        weight = sum(count for _, count in vectors)
        mean = [sum(vector[i] * count for vector, count in vectors) / weight for i in range(len(vectors[0][0]))]
        return round(cosine(mean, self.target), 5)

    async def _word_score(self, start, end):
        key = (start, end)
        if key not in self._scores:
            middle, half = (start + end) // 2, max(self.voice['word_context_ms'], end - start) // 2
            samples = self.memory.span(middle - half, middle + half, SAMPLE_RATE)
            score = None
            if len(samples) >= SAMPLE_RATE // 4 and rms_dbfs(samples) >= self.voice['min_dbfs']:
                try:
                    score = round(cosine(await embed(self.embedder, samples, SAMPLE_RATE), self.target), 5)
                except ValueError:
                    score = None
            self._scores[key] = score
            while len(self._scores) > 4096:
                self._scores.pop(next(iter(self._scores)))
        return self._scores[key]

    async def _voice_of(self, window, start, end, words=()):
        """(match, basis, reason or None, speaker, its score, per-word score) of [start, end) in one window.

        ``words``: the provisional words there; where they carry the TSE lane's decision (``tse``,
        :mod:`gamst.providers.overlap_tse`), that decides an overlapped word instead of the score on the mixture."""
        shares = _share(window['turns'], start, end)
        active = {speaker: share for speaker, share in shares.items() if share >= self.voice['active_share']}
        person = max((share for speaker, share in active.items() if speaker in window['person']), default=0.)
        other = max((share for speaker, share in active.items() if speaker not in window['person']), default=0.)
        dominant = max(active, key=active.get) if active else None
        scores = {item['speaker']: item['score'] for item in window['speakers']}
        limit = self.voice['overlap_share']
        if person >= limit and other >= limit:
            tse = next((word['tse'] for word in words if isinstance(word.get('tse'), dict)), None)
            if tse is not None:
                decision = tse.get('decision')
                reason = None if decision == 'person' else 'other_speaker' if decision == 'other' else 'uncertain_overlap'
                return decision == 'person', f'overlap_tse_{decision}', reason, dominant, scores.get(dominant), \
                    tse.get('score')
            score = await self._word_score(start, end)
            matched = score is not None and score >= self.voice['threshold']
            return matched, 'overlap_voice_score', None if matched else 'uncertain_overlap', dominant, \
                scores.get(dominant), score
        if active and person >= other:
            return True, 'person_speaker', None, dominant, scores.get(dominant), None
        if active and dominant not in window['unknown']:
            return False, 'other_speaker', 'other_speaker', dominant, scores.get(dominant), None
        score = await self._word_score(start, end)
        matched = score is not None and score >= self.voice['threshold']
        basis = 'speaker_unidentified' if active else 'no_speaker'
        return matched, f'{basis}_voice_score', None if matched else basis, dominant, scores.get(dominant), score

    def _speech_near(self, window, start, end):
        near = self.rule['near_ms']
        heard = any(type(word.get('start_ms')) is int and word['start_ms'] < end + near and word['end_ms'] > start - near
                    for item in self._heard for word in (item.get('words') or [dict(start_ms=item['start_ms'],
                                                                                      end_ms=item['end_ms'])]))
        return heard or bool(_share(window['turns'], start, end))

    def _first_occurrences(self, low, high):
        found = []
        for value in self._utterances:
            for item in term_occurrences(value, self.terms):
                ids = [word['word_id'] for word in item['words'] if word.get('word_id') in self._open]
                if ids and low <= item['end_ms'] < high:
                    found.append(dict(item, word_ids=ids))
        return found

    def _second_occurrences(self, low, high):
        found = []
        for number, value in enumerate(self._heard):
            for index, item in enumerate(term_occurrences(value, self.terms)):
                key = f'{number}:{index}'
                if key not in self._used and low <= item['end_ms'] < high:
                    found.append(dict(item, key=key))
        return found

    async def _decide(self, complete):
        """Decide the next regions whose words are all in, heard twice, and whose window is diarized (in order)."""
        decided, near = False, self.rule['near_ms']
        while True:
            low, high = self._region(self._decided)
            end = self.memory.end_ms
            task = self._windows.get(self._decided)
            if end is None or low > end or task is None or not task.done():
                return decided
            last = self._final and high >= end
            if last and self._heard_until() != float('inf'):
                return decided
            if not last and (complete < high + near or self._heard_until() < high + near):
                return decided
            window = task.result()
            await self._settle(window, low, float('inf') if last else high)
            self._windows[self._decided] = window     # only its result is kept
            for old in [key for key in self._windows if key < self._decided - 2]:
                del self._windows[old]
            self.stats['regions'] += 1
            self._decided += 1
            decided = True

    async def _settle(self, window, low, high):
        near = self.rule['near_ms']
        first = self._first_occurrences(low, high)
        second = self._second_occurrences(low - near, high + near)
        pairs = match_occurrences(first, second, near)
        verdicts, words_done = [], set()
        for index, item in enumerate(first):
            if index in pairs:
                heard = second[pairs[index]]
                self._used.add(heard['key'])
                match, basis, reason, speaker, speaker_score, score = await self._voice_of(
                    window, heard['start_ms'], heard['end_ms'], [self._open[key] for key in item['word_ids']])
                basis = 'heard_twice+' + basis
            else:
                speech = self._speech_near(window, item['start_ms'], item['end_ms'])
                revoke = self.rule['revoke_not_heard'] if speech else self.rule['revoke_no_speech']
                reason = 'not_heard' if speech else 'no_speech'
                if revoke:
                    match, basis, speaker, speaker_score, score = False, reason, None, None, None
                else:
                    match, basis, reason, speaker, speaker_score, score = await self._voice_of(
                        window, item['start_ms'], item['end_ms'], [self._open[key] for key in item['word_ids']])
                    basis = 'heard_once+' + basis
            for word_id in item['word_ids']:
                verdicts.append(self._verdict(word_id, match, basis, reason, speaker, speaker_score, score,
                                              heard=index in pairs))
                words_done.add(word_id)
        for word_id, word in list(self._open.items()):     # the other words: by the voice alone
            if word_id in words_done or not (low <= word['end_ms'] < high):
                continue
            match, basis, reason, speaker, speaker_score, score = await self._voice_of(
                window, word['start_ms'], word['end_ms'], [word])
            verdicts.append(self._verdict(word_id, match, basis, reason, speaker, speaker_score, score, heard=None))
        if verdicts:
            self.stats['verdicts'] += len(verdicts)
            self._outbox.append(self.spec.validate(dict(
                kind='verdicts', reference=self.reference_name, start_ms=min(item['start_ms'] for item in verdicts),
                end_ms=max(item['end_ms'] for item in verdicts), verdicts=verdicts, model=self._model())))
        for item in self._second_occurrences(low, high):     # heard only the second time: final at once
            self._used.add(item['key'])
            match, basis, reason, speaker, speaker_score, score = await self._voice_of(window, item['start_ms'], item['end_ms'])
            self.stats['added'] += 1
            word = dict(text=item['spoken'], start_ms=item['start_ms'], end_ms=max(item['end_ms'], item['start_ms'] + 1),
                        match=match, decided_by='verification', basis='heard_once_by_verification+' + basis,
                        **({'reason': reason} if reason else {}), **({'speaker_score': score} if score is not None else {}))
            self._outbox.append(self.spec.validate(dict(
                kind='utterance', reference=self.reference_name, utterance_id=f"verification:{item['key']}",
                status='final', text=item['spoken'], start_ms=word['start_ms'], end_ms=word['end_ms'], words=[word],
                speaker=dict(reference=self.reference_name, match=match, model=self.embedder.model_id),
                model=self._model())))

    def _verdict(self, word_id, match, basis, reason, speaker, speaker_score, score, *, heard):
        word = self._open.pop(word_id)
        provisional = bool(word.get('match'))
        verdict = dict(word_id=word_id, utterance_id=word.get('utterance_id'), start_ms=word['start_ms'],
                       end_ms=word['end_ms'], match=bool(match), provisional=provisional, changed=bool(match) != provisional,
                       settled_by='verification', basis=basis)
        if reason and not match:
            verdict['reason'] = reason
        for key, value in (('heard', heard), ('speaker', speaker), ('speaker_score', speaker_score), ('word_score', score)):
            if value is not None:
                verdict[key] = value
        if self.pipeline is not None:
            self.stats['decided_after_ms'].append(max(0, time.time_ns() // 1_000_000 - word['end_ms']))
        return verdict

    def _late(self):
        """Words that arrive for a region already decided (a watermark that was wrong): decided now, and counted."""
        if self._origin is None or not self._decided:
            return
        decided_until = self._region(self._decided)[0]
        late = [word_id for word_id, word in self._open.items() if word['end_ms'] < decided_until]
        if late:
            self.stats['late'] += len(late)
            for word_id in late:      # no window kept for them: by their own first decision, said as late
                word = self._open[word_id]
                self._outbox.append(self.spec.validate(dict(
                    kind='verdicts', reference=self.reference_name, start_ms=word['start_ms'], end_ms=word['end_ms'],
                    verdicts=[self._verdict(word_id, bool(word.get('match')), 'late', None if word.get('match')
                                            else 'late', None, None, word.get('speaker_score'), heard=None)],
                    model=self._model())))

    def _hold(self):
        """Keep the audio this lane still needs: the next window, the second recogniser's next sound, the open words."""
        if self.memory is None or self._origin is None:
            return
        low = self._window(self._decided)[0] - self.voice['word_context_ms']
        if self._fed is not None:
            low = min(low, self._fed)
        self.memory.hold('verification', low)
