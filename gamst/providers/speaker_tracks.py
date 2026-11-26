"""Who said each word: streaming speaker slots, each slot's own sound against the enrolled voice.

A streaming diarization module says which speaker *slot* is active every few tens of milliseconds; it does not
say who a slot is. :class:`SpeakerTracks` keeps, per slot, the sound where that slot alone was active (a bounded
store in memory, never in a record), embeds it piece by piece with the speaker model and decides the slot's
identity from all of it against the owner's few-shot voice (a weighted centroid of the pieces; old pieces leave
the store first). Each word of a transcript belongs to the slot active at its capture time; the word is passed on
``provisional``, with its slot and whether that slot is the enrolled person, so
:class:`gamst.providers.term_events.TermEvents` counts it at once. :class:`TrackVerification`, in its own lane,
settles those words later: when the slot's identity is stable, or a settle delay has passed in capture time, it
gives each word a verdict, which TermEvents turns into ``final`` or ``revoked`` revisions. Nothing is recomputed
from the start.

Inputs of :class:`SpeakerTracks` (``consumes``):

- the microphone: 16 kHz mono audio frames, kept in a ring of ``ring_seconds``;
- slot activity, the record value of an ``audio.speaker_activity`` module (:class:`gamst.providers.diarization.
  NemotronDiarization` is the contract)::

      {"diarization": {"session_id": str,            # a new session: its slots are other people
                       "chunks": [{"start_ms": int, "end_ms": int,    # activity is known for this whole span
                                   "activity": [{"slot": int, "intervals_ms": [[start, end], ...]}, ...]}]}}

  chunks of one session arrive in capture order and do not overlap;
- words, ``speech.transcript_words`` values of any ASR (``{"utterance_id", "status": "final", "text", "start_ms",
  "end_ms", "words": [{"text", "start_ms", "end_ms"}]}``). Each utterance ID once; a word is decided once the slot
  activity covers it (``margin_ms`` after its end), or ``wait_ms`` of later sound without it.

Outputs (capability ``speech.speaker_tracks``):

- ``kind: "slots"``: each slot's identity when it changed: ``slot``, ``speech_ms`` (sound it holds), ``score``
  (cosine to the enrolled voice), ``enrolled`` (``score >= threshold``), ``stable`` (``speech_ms >= stable_ms``),
  ``revision``; ``end_ms``: slot activity is known up to here;
- ``kind: "utterance"``: the transcript passed on, each word with ``word_id``, ``slot`` (``"s<session>:<slot>"`` or
  None), ``slot_score``, ``slot_speech_ms``, ``slots_active`` (slots active over the word; 2 or more: overlap),
  ``match``, ``decided_by`` (``slot``, ``no_slot``: no slot active at the word, ``slot_unknown``: its slot holds too
  little sound) and ``state: "provisional"``.

:class:`TrackVerification` reads those records and gives ``speech.word_verdicts`` (``{"kind": "verdicts",
"verdicts": [{"word_id", "match", "provisional", "changed", "reason", "slot", "slot_score"}]}``).

Scores are uncalibrated cosine similarities, the same as :mod:`gamst.providers.speaker_verify`; the threshold is
set on the owner's own clips.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import replace
import inspect
import math

from ..capabilities import CapabilitySpec, OutputField
from .audio_ring import AudioRing, mono_float, rms_dbfs
from .speaker_verify import centroid, cosine, embed, l2_normalize

SAMPLE_RATE = 16000
STEP_MS = 10            # the grid on which slot activity is compared


def _spec(capability_id, provider, version, fields):
    return CapabilitySpec(capability_id, capability_id, 'audio', provider, version, fields, 'native', stateful=True,
                          reset_on_gap=True, freshness_ms=5000, interpretation='estimate', devices=('cpu',),
                          license='Apache-2.0 adapter; model weights per backend')


class _Slot:
    """One slot's own sound (pieces not yet embedded) and the embeddings of what it said before."""
    __slots__ = ('key', 'pending', 'pending_ms', 'pieces', 'speech_ms', 'score', 'revision', 'temporary')

    def __init__(self, key):
        self.key, self.pending, self.pending_ms = key, [], 0
        self.pieces = deque()          # (normalised embedding, ms)
        self.speech_ms, self.score, self.revision, self.temporary = 0, None, 0, None


class SpeakerTracks:
    """Pipeline module: microphone + slot activity + words in, slot identities and attributed words out."""
    owns_process = True    # the embedding model runs in its worker

    def __init__(self, embedder, *, reference='voice', threshold=.5, piece_ms=2000, min_ms=800, max_slot_ms=60000,
                 stable_ms=8000, ring_seconds=60., margin_ms=200, wait_ms=5000, min_dbfs=-45., overlap=False,
                 capability_id='speech.speaker_tracks'):
        if not 500 <= piece_ms <= 10000 or not 200 <= min_ms <= piece_ms or not piece_ms <= max_slot_ms <= 600000:
            raise ValueError('require 500 <= piece_ms <= 10000, 200 <= min_ms <= piece_ms <= max_slot_ms <= 600000')
        if not 5 <= ring_seconds <= 300 or not 0 <= margin_ms <= 2000 or not 0 < wait_ms <= 60000:
            raise ValueError('require 5 <= ring_seconds <= 300, 0 <= margin_ms <= 2000, 0 < wait_ms <= 60000')
        self.embedder, self.reference_name, self.threshold = embedder, reference, float(threshold)
        self.piece_ms, self.min_ms, self.max_slot_ms, self.stable_ms = int(piece_ms), int(min_ms), int(max_slot_ms), int(stable_ms)
        self.margin_ms, self.wait_ms, self.min_dbfs = int(margin_ms), int(wait_ms), float(min_dbfs)
        self.overlap = bool(overlap)      # also keep sound where other slots spoke at the same time (mixed)
        self.ring = AudioRing(SAMPLE_RATE, ring_seconds)
        self.target = None
        self.spec = _spec(capability_id, 'gamst.speaker_tracks', f'1.{embedder.model_id}'.replace(':', '.'), (
            OutputField('kind', 'string'), OutputField('reference', 'string'), OutputField('start_ms', 'integer'),
            OutputField('end_ms', 'integer'), OutputField('slots', 'array', required=False),
            OutputField('session', 'integer', required=False), OutputField('match', 'boolean', required=False),
            OutputField('utterance_id', 'string', required=False), OutputField('status', 'string', required=False),
            OutputField('text', 'string', required=False), OutputField('words', 'array', required=False),
            OutputField('language', 'string', required=False), OutputField('speaker', 'object', required=False),
            OutputField('model', 'object')))
        self.reset()

    # ------------------------------------------------------------------ reference and state

    def bind(self, pipeline):
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'voice':
            raise ValueError(f'speaker_tracks needs a voice reference named {self.reference_name!r}')
        value = reference['value']
        if not isinstance(value, dict) or 'embeddings' not in value:
            raise ValueError('speaker_tracks needs an enrolled voice ({"model", "embeddings"}: gamst recipe enroll --kind voice)')
        if value.get('model') != self.embedder.model_id:
            raise ValueError(f"voice reference was enrolled with {value.get('model')!r}; this module runs "
                             f'{self.embedder.model_id!r} (re-enroll with the same model)')
        self.target = centroid(value['embeddings'])

    def reset(self):
        self.ring.reset()
        self._sessions, self._slots = {}, OrderedDict()
        self._activity = deque()        # (start_ms, end_ms, slot key) of every slot, last ring_seconds
        self._known_until = None        # slot activity is known up to here (current session)
        self._waiting = []              # utterances whose words the activity does not cover yet

    def _model(self):
        return dict(id=self.embedder.model_id, threshold=self.threshold, piece_ms=self.piece_ms,
                    max_slot_ms=self.max_slot_ms, stable_ms=self.stable_ms)

    @property
    def pending_since_ms(self):
        return min((item['start_ms'] for item in self._waiting), default=None)

    async def close(self):
        close = getattr(self.embedder, 'close', None)
        value = close() if callable(close) else None
        if inspect.isawaitable(value):
            await value

    # ------------------------------------------------------------------ frames

    async def process(self, frame):
        value = frame.data
        if isinstance(value, dict) and isinstance(value.get('diarization'), dict):
            return await self._activity_of(value['diarization'], frame.timestamp_ms)
        if isinstance(value, dict):
            if value.get('status') != 'final' or not isinstance(value.get('words'), list) or not value.get('text'):
                return None
            self._waiting.append(value)
            return await self._release()
        if frame.modality != 'audio' or value is None or not hasattr(value, '__len__'):
            return None
        if frame.sample_rate != SAMPLE_RATE:
            raise ValueError('speaker_tracks needs 16 kHz mono audio')
        self.ring.append(value, int(frame.timestamp_ms), SAMPLE_RATE)
        return await self._release()

    async def finish(self):
        return await self._release(final=True)

    # ------------------------------------------------------------------ slot activity -> each slot's own sound

    async def _activity_of(self, value, timestamp_ms):
        session = value.get('session_id')
        if not isinstance(session, str):
            raise ValueError('slot activity needs a session_id')
        if session not in self._sessions:
            self._sessions[session] = len(self._sessions) + 1
            self._known_until = None        # a new session: nothing of it is known yet
            while len(self._sessions) > 4:
                self._sessions.pop(next(iter(self._sessions)))
        number, changed = self._sessions[session], []
        for chunk in value.get('chunks') or ():
            start, end = chunk.get('start_ms'), chunk.get('end_ms')
            if type(start) is not int or type(end) is not int or end <= start:
                continue
            spans = {}
            for item in chunk.get('activity') or ():
                if type(item.get('slot')) is not int:
                    continue
                key = f's{number}:{item["slot"]}'
                for pair in item.get('intervals_ms') or ():
                    low, high = max(start, int(pair[0])), min(end, int(pair[1]))
                    if high > low:
                        spans.setdefault(key, []).append((low, high))
                        self._activity.append((low, high, key))
            self._known_until = max(end, self._known_until or end)
            for key in await self._accumulate(start, end, spans):
                if key not in changed:
                    changed.append(key)
        horizon = (self._known_until or 0) - int(self.ring.seconds * 1000)
        while self._activity and self._activity[0][1] < horizon:
            self._activity.popleft()
        outputs = []
        if changed:
            outputs.append(self.spec.validate(dict(
                kind='slots', reference=self.reference_name, session=number, start_ms=self._known_until,
                end_ms=self._known_until, slots=[self._identity(self._slots[key]) for key in self._slots
                                                 if key.startswith(f's{number}:')], model=self._model())))
        released = await self._release()
        return outputs + (released or []) or None

    async def _accumulate(self, start, end, spans):
        """Add each slot's own sound of [start, end) to its store; embed every full piece. Returns the changed slots."""
        import numpy as np
        steps = (end - start) // STEP_MS
        if steps <= 0 or not spans:
            return []
        active = {}
        for key, pairs in spans.items():
            mask = np.zeros(steps, bool)
            for low, high in pairs:
                mask[(low - start) // STEP_MS:max((low - start) // STEP_MS + 1, -(-(high - start) // STEP_MS))] = True
            active[key] = mask
        together = sum(mask.astype(int) for mask in active.values())
        changed = []
        for key, mask in active.items():
            own = mask if self.overlap else mask & (together == 1)
            edges = np.diff(np.r_[0, own.astype(int), 0])
            slot = self._slots.get(key)
            if slot is None:
                slot = self._slots[key] = _Slot(key)
                while len(self._slots) > 64:
                    self._slots.popitem(last=False)
            for first, last in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                low, high = start + int(first) * STEP_MS, start + int(last) * STEP_MS
                if not self.ring.covers(low, high):
                    continue
                samples = self.ring.span(low, high)
                if not len(samples) or rms_dbfs(samples) < self.min_dbfs:
                    continue
                slot.pending.append(samples)
                slot.pending_ms += high - low
                slot.temporary = None
                if slot.pending_ms >= self.piece_ms and await self._embed_piece(slot):
                    changed.append(key)
        return changed

    async def _embed_piece(self, slot):
        import numpy as np
        samples, ms = np.concatenate(slot.pending), slot.pending_ms
        slot.pending, slot.pending_ms, slot.temporary = [], 0, None
        try:
            vector = l2_normalize(await embed(self.embedder, samples, SAMPLE_RATE))
        except ValueError:
            return False     # too short or unclear for the model: no evidence either way
        slot.pieces.append((vector, ms))
        slot.speech_ms += ms
        while sum(item[1] for item in slot.pieces) - slot.pieces[0][1] >= self.max_slot_ms:
            slot.speech_ms -= slot.pieces.popleft()[1]
        slot.score = self._score([item for item in slot.pieces])
        slot.revision += 1
        return True

    def _score(self, pieces):
        if self.target is None:
            raise RuntimeError('no voice reference bound')
        weight = sum(ms for _, ms in pieces)
        mean = [sum(vector[index] * ms for vector, ms in pieces) / weight for index in range(len(pieces[0][0]))]
        if len(mean) != len(self.target):
            raise RuntimeError('the voice reference has another embedding size than this model')
        return round(cosine(mean, self.target), 5)

    async def _current(self, slot):
        """(score, speech_ms) of a slot now: its pieces, or, before its first piece, what it holds if that is enough."""
        if slot.pieces:
            return slot.score, slot.speech_ms
        if slot.pending_ms < self.min_ms:
            return None, slot.pending_ms
        if slot.temporary is None:
            import numpy as np
            try:
                vector = l2_normalize(await embed(self.embedder, np.concatenate(slot.pending), SAMPLE_RATE))
                slot.temporary = self._score([(vector, slot.pending_ms)])
            except ValueError:
                slot.temporary = math.nan
        return (None if math.isnan(slot.temporary) else slot.temporary), slot.pending_ms

    def _identity(self, slot):
        score = slot.score
        return dict(slot=slot.key, speech_ms=int(slot.speech_ms), score=score,
                    enrolled=score is not None and score >= self.threshold, stable=slot.speech_ms >= self.stable_ms,
                    revision=slot.revision)

    # ------------------------------------------------------------------ words -> slots

    async def _release(self, final=False):
        """Utterances whose words the slot activity covers now (all of them at the end, or after ``wait_ms``)."""
        outputs, newest = [], self.ring.end_ms
        while self._waiting:
            value = self._waiting[0]
            ends = [word['end_ms'] for word in value['words'] if isinstance(word, dict) and type(word.get('end_ms')) is int]
            needed = max(ends, default=value.get('end_ms') or 0) + self.margin_ms
            covered = self._known_until is not None and self._known_until >= needed
            late = newest is not None and newest >= needed + self.wait_ms
            if not (final or covered or late):
                break
            self._waiting.pop(0)
            outputs.append(await self._attribute(value, covered))
        return outputs or None

    def _overlaps(self, start, end):
        found = {}
        for low, high, key in self._activity:
            shared = min(end, high) - max(start, low)
            if shared > 0:
                found[key] = found.get(key, 0) + shared
        return found

    async def _attribute(self, value, covered):
        utterance = str(value.get('utterance_id') or f"utterance:{value.get('end_ms')}")
        words = []
        for index, word in enumerate(value['words']):
            if not isinstance(word, dict) or type(word.get('start_ms')) is not int or type(word.get('end_ms')) is not int:
                words.append(word)
                continue
            start, end = word['start_ms'], max(word['end_ms'], word['start_ms'] + 1)
            over = self._overlaps(start, end)
            near = over or self._overlaps(start - self.margin_ms, end + self.margin_ms)
            key = max(near, key=near.get) if near else None
            annotated = dict(word, word_id=f'{utterance}#{index}', slot=key, slots_active=len(over), state='provisional')
            if key is None:
                annotated.update(match=False, decided_by='no_slot')
            else:
                score, held = await self._current(self._slots[key]) if key in self._slots else (None, 0)
                annotated.update(slot_speech_ms=int(held), match=score is not None and score >= self.threshold,
                                 decided_by='slot' if score is not None else 'slot_unknown')
                if score is not None:
                    annotated.update(slot_score=score, speaker_score=score)
            if not covered:
                annotated['activity_complete'] = False
            words.append(annotated)
        match = any(isinstance(word, dict) and word.get('match') for word in words)
        result = dict(kind='utterance', reference=self.reference_name, utterance_id=utterance, status='final',
                      text=str(value['text']), start_ms=int(value['start_ms']), end_ms=int(value['end_ms']), words=words,
                      match=match, speaker=dict(reference=self.reference_name, match=match, model=self.embedder.model_id),
                      model=self._model())
        if isinstance(value.get('language'), str):
            result['language'] = value['language']
        return self.spec.validate(result)


class TrackVerification:
    """Pipeline module (its own lane): settles the provisional words of :class:`SpeakerTracks`.

    A word gets its verdict once its slot's identity is stable, or once slot activity is known ``settle_ms`` past
    the word's end, whichever comes first; it is decided by its slot's identity at that time. A word older than
    ``window_ms`` (in the same capture time) is decided then at the latest, so what is held stays bounded.
    """

    def __init__(self, *, reference='voice', settle_ms=10000, window_ms=60000, max_words=4096,
                 capability_id='speech.word_verdicts'):
        if not 0 <= settle_ms <= window_ms <= 600000:
            raise ValueError('require 0 <= settle_ms <= window_ms <= 600000')
        self.reference_name, self.settle_ms, self.window_ms, self.max_words = reference, int(settle_ms), int(window_ms), int(max_words)
        self.spec = _spec(capability_id, 'gamst.track_verification', '1', (
            OutputField('kind', 'string'), OutputField('reference', 'string'), OutputField('start_ms', 'integer'),
            OutputField('end_ms', 'integer'), OutputField('verdicts', 'array'), OutputField('model', 'object')))
        self.reset()

    def reset(self):
        self._words, self._slots, self._now = OrderedDict(), {}, None

    @property
    def pending_since_ms(self):
        return min((word['start_ms'] for word in self._words.values()), default=None)

    def _model(self):
        return dict(settle_ms=self.settle_ms, window_ms=self.window_ms)

    def process(self, frame):
        value = frame.data
        if not isinstance(value, dict) or value.get('reference') != self.reference_name:
            return None
        if value.get('kind') == 'slots':
            for slot in value.get('slots') or ():
                self._slots[slot['slot']] = slot
            self._now = max(value['end_ms'], self._now or value['end_ms'])
        elif value.get('kind') == 'utterance':
            for word in value.get('words') or ():
                if isinstance(word, dict) and word.get('state') == 'provisional' and isinstance(word.get('word_id'), str):
                    self._words[word['word_id']] = dict(
                        word_id=word['word_id'], utterance_id=value.get('utterance_id'), start_ms=word['start_ms'],
                        end_ms=word['end_ms'], slot=word.get('slot'), provisional=bool(word.get('match')),
                        score=word.get('slot_score'))
            while len(self._words) > self.max_words:
                self._words.popitem(last=False)
        else:
            return None
        return self._verdicts()

    def finish(self):
        return self._verdicts(final=True)

    def _verdicts(self, final=False):
        ready = []
        for key, word in list(self._words.items()):
            slot = self._slots.get(word['slot']) if word['slot'] is not None else None
            stable = slot is not None and slot.get('stable')
            settled = self._now is not None and self._now >= word['end_ms'] + self.settle_ms
            old = self._now is not None and self._now >= word['end_ms'] + self.window_ms
            if final or stable or settled or old:
                ready.append(self._decide(self._words.pop(key), slot, 'final' if final else 'stable' if stable else
                                          'settled' if settled else 'window'))
        if not ready:
            return None
        return self.spec.validate(dict(kind='verdicts', reference=self.reference_name,
                                       start_ms=min(item['start_ms'] for item in ready),
                                       end_ms=max(item['end_ms'] for item in ready), verdicts=ready, model=self._model()))

    @staticmethod
    def _decide(word, slot, why):
        if slot is not None and slot.get('score') is not None:
            match, score = bool(slot['enrolled']), slot['score']
        else:
            match, score = word['provisional'], word['score']
        verdict = dict(word_id=word['word_id'], utterance_id=word['utterance_id'], start_ms=word['start_ms'],
                       end_ms=word['end_ms'], slot=word['slot'], match=match, provisional=word['provisional'],
                       changed=match != word['provisional'], settled_by=why)
        if score is not None:
            verdict['slot_score'] = score
        if verdict['changed']:
            verdict['reason'] = 'slot_identity_enrolled' if match else 'slot_identity_other'
        return verdict


class WholeBlocks:
    """A streaming diarization worker behind a sample clock: audio in whole ``STEP_MS`` blocks, gap-free capture times.

    Models that keep a recurrent speaker cache start a new session at any discontinuity; a frame whose time does not
    follow the sample clock (more than ``max_drift_ms``) is one, and starts it over here as well.
    """
    owns_process = True

    def __init__(self, module, max_drift_ms=250):
        self.module, self.spec, self.max_drift_ms = module, module.spec, max_drift_ms
        self.name, self.version = getattr(module, 'name', 'diarization'), getattr(module, 'version', '1')
        self._clear()

    def _clear(self):
        self._rest, self._end_ms, self._sequence = None, None, 0

    async def process(self, frame):
        import numpy as np
        samples = mono_float(frame.data)
        if frame.sample_rate != SAMPLE_RATE:
            raise ValueError('slot activity needs 16 kHz mono audio')
        held = 0 if self._rest is None else len(self._rest)
        if self._end_ms is not None and abs(self._end_ms + 1000 * (held + len(samples)) / SAMPLE_RATE
                                            - frame.timestamp_ms) > self.max_drift_ms:
            self._clear()
            held = 0
        if self._end_ms is None:
            self._end_ms = frame.timestamp_ms - 1000 * len(samples) / SAMPLE_RATE
        samples = samples if not held else np.concatenate([self._rest, samples])
        block = SAMPLE_RATE * STEP_MS // 1000
        whole = len(samples) // block * block
        self._rest = samples[whole:] if whole < len(samples) else None
        if not whole:
            return None
        self._end_ms += 1000 * whole / SAMPLE_RATE
        self._sequence += 1
        sensor = frame.sensor if frame.sensor is None else replace(frame.sensor, sequence=self._sequence,
                                                                   capture_sequence=None)
        return await _maybe(self.module.process(replace(frame, data=samples[:whole], timestamp_ms=round(self._end_ms),
                                                        sensor=sensor)))

    async def finish(self):
        return await _maybe(self.module.finish()) if callable(getattr(self.module, 'finish', None)) else None

    async def reset(self):
        self._clear()
        if callable(getattr(self.module, 'reset', None)):
            await _maybe(self.module.reset())

    async def close(self):
        if callable(getattr(self.module, 'close', None)):
            await _maybe(self.module.close())


async def _maybe(value):
    return await value if inspect.isawaitable(value) else value
