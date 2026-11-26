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
  confirmed when the second recogniser heard the same word within ``near_ms`` of it (one to one, nearest first). When
  only the first hearing has it and someone spoke there, it stays when the voice evidence for the person is *clear*
  (``keep_clear_voice``; ``clear``: ``speaker`` = the person's diarized speaker alone is active at the word,
  ``score`` = the per-word score is at least ``threshold + clear_margin``, ``either`` or ``both``), is ``uncertain``
  when the person's speaker overlaps another one and the score is not clear, and is revoked (``not_heard``) when the
  voice points elsewhere or is absent; revoked (``no_speech``) when nobody spoke there. An occurrence only the second
  recogniser heard is added, final. ``vote``: more ``speech.transcript_words`` sources (any recogniser the item also
  reads) vote; a first-hearing word that one of them also heard counts as heard twice, and with
  ``vote_adds_once: false`` a second-hearing word is added only when a voter heard it too. ``decide: "majority"``
  (with voters) replaces the pairing rule: a word is said when at least two of the recognisers (the first hearing,
  the second one and the voters) wrote it within ``near_ms``; a word only one of them wrote is dropped (revoked, or not
  added) unless ``keep_single`` and no other recogniser wrote anything near it and the voice is clear.
  ``decide: "primary"`` (owner decision 31, the default when voters are read: the best recogniser is never outvoted):
  the first hearing is the primary recogniser (the recipe's live ``asr`` item, the registry's ``audio.asr`` default,
  today ElevenLabs Scribe v3 Live). A word it wrote that another recogniser also wrote is said (the voice decides who);
  a word only it wrote stands unless it is *contested* (the other recognisers, ``contest_by: "all"`` or ``"any"``,
  heard speech there and wrote other words): a contested word is vetoed (revoked, ``vetoed``) only when the primary's
  own word confidence (``confidence`` or ``exp(logprob)`` on its words, when the recogniser gives one) is below
  ``low_confidence`` (None, the default: no veto, measured not to separate invented from true words), else it is
  ``uncertain`` (``contested: "uncertain"``, the default: shown, not counted, never "not said") or stands
  (``contested: "stand"``). Words only the others wrote are added as by majority (``add: "agreed"``) or when any of
  them wrote it (``add: "any"``);
- **the voice rule** (:data:`VOICE`) for each word that stays: the person's speaker active over the word and no other
  speaker over ``overlap_share`` of it -> the person; another speaker dominant -> another voice; both over
  ``overlap_share`` (overlapped speech) -> the TSE lane's decision where the word carries one
  (:mod:`gamst.providers.overlap_tse`: the person's voice extracted from the mixture, heard and scored; person ->
  the person, not the person -> another voice, uncertain -> ``uncertain_overlap``), else the per-word voice score
  decides, and below ``threshold`` the word is ``uncertain_overlap``: its verdict says ``uncertain: true``, :class:`gamst.providers.term_events.TermEvents` gives it
  the state ``uncertain`` (not counted, not revoked), and a later lane may still settle it with a verdict of its own. With no speaker active, or an unidentified one, the per-word score
  decides.

Outputs (capability ``speech.word_verdicts``): ``kind: "verdicts"`` (``[{word_id, match, provisional, changed, basis,
reason?, heard, speaker, speaker_score, word_score}]`` for every provisional word), ``kind: "utterance"`` (an
occurrence only the second recogniser heard: a final utterance of that word, ``decided_by: "verification"``) and
``kind: "window"`` (each window's speakers, their speech and score; no audio, no embedding). Nothing here is tuned on
a clip; every setting is a parameter.
"""
from __future__ import annotations

import asyncio
import math
import time

from ..capabilities import CapabilitySpec, OutputField
from .audio_ring import rms_dbfs
from .speaker_verify import centroid, cosine, embed, l2_normalize
from .term_events import parse_terms, term_occurrences

SAMPLE_RATE = 16000
RULE = dict(near_ms=1000, revoke_not_heard=True, revoke_no_speech=True, keep_clear_voice=True, clear='either',
            clear_margin=.1, vote=False, vote_adds_once=True, decide='pair', keep_single=True,
            contested='uncertain', contest_by='all', low_confidence=None, add='agreed')
DECIDE = ('pair', 'majority', 'primary')
CONTESTED = ('uncertain', 'stand')       # primary rule: a contested primary word that is not vetoed
CONTEST_BY = ('all', 'any')              # primary rule: which other recognisers must have written other words there
ADD = ('agreed', 'any')                  # primary rule: words only the others wrote (as majority / as pair)
UNCERTAIN = ('uncertain_overlap', 'contested')   # reasons that leave a word uncertain (shown, not counted, not revoked)
CLEAR = ('speaker', 'score', 'either', 'both')
VOICE = dict(threshold=.5, min_speaker_ms=1000, piece_ms=2000, overlap_share=.3, active_share=.1, word_context_ms=1500,
             min_dbfs=-45., decision='absolute', floor=None, margin=0., word_margin=0., anchor_ms=None, anchor_join_ms=1000)
DECISIONS = ('absolute', 'relative')
WINDOWS = dict(min_ms=10000, max_ms=20000, cap_ms=30000, pause_ms=500, before_ms=15000, after_ms=5000, concurrency=2,
               step_ms=None)


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
                 rule=None, voice=None, windows=None, feed_speed=1.25, capability_id='speech.word_verdicts', bank=None):
        self.recogniser, self.diarizer, self.embedder = recogniser, diarizer, embedder
        self.reference_name, self.terms_name, self.audio = reference, terms, audio
        self.rule, self.voice = _settings(RULE, rule, 'rule'), _settings(VOICE, voice, 'voice')
        if self.rule['decide'] not in DECIDE:
            raise ValueError(f"rule decide is one of {', '.join(DECIDE)}")
        if self.rule['clear'] not in CLEAR:
            raise ValueError(f"rule clear is one of {', '.join(CLEAR)}")
        for key, known in (('contested', CONTESTED), ('contest_by', CONTEST_BY), ('add', ADD)):
            if self.rule[key] not in known:
                raise ValueError(f"rule {key} is one of {', '.join(known)}")
        low = self.rule['low_confidence']
        if low is not None and not (type(low) in (int, float) and 0 < low < 1):
            raise ValueError('rule low_confidence is None (no veto) or a probability between 0 and 1')
        self.windows = _settings(WINDOWS, windows, 'window')
        # speaker decision v2 (voice ``decision: "relative"``, gamst/providers/voice_bank.py): the person's speaker in a
        # window is the one closest to the references by ``margin`` (at least ``floor``); the window's other speakers
        # become negatives; a word with no clear speaker counts when it is ``word_margin`` closer to the references than
        # to every negative; ``anchor_ms: [before, after]`` diarizes a window around each word instead of the region's
        if self.voice['decision'] not in DECISIONS:
            raise ValueError(f"voice decision is one of {', '.join(DECISIONS)}")
        if self.voice['floor'] is None:
            self.voice['floor'] = self.voice['threshold']
        anchor = self.voice['anchor_ms']
        if anchor is not None and (len(anchor) != 2 or not 0 <= anchor[0] <= self.windows['before_ms']
                                   or not 0 <= anchor[1] <= self.windows['after_ms']):
            raise ValueError('anchor_ms is [before, after] within the windows before_ms / after_ms')
        self.bank_settings, self.bank = bank, None
        # The second recogniser is a realtime API: it gets the complete audio at most ``feed_speed`` times real time,
        # evenly (a little faster than real time, so it catches up after a stall); None: as soon as it is complete
        self.feed_speed = None if feed_speed is None else float(feed_speed)
        w = self.windows
        if w['step_ms'] is not None and not 0 < w['step_ms'] <= 120000 or w['before_ms'] < 0 or w['after_ms'] < 0 \
                or not 0 < w['min_ms'] <= w['max_ms'] <= w['cap_ms'] <= 300000 or not 0 < w['pause_ms'] <= 10000:
            raise ValueError('windows need 0 < min_ms <= max_ms <= cap_ms <= 300000, 0 < pause_ms <= 10000, '
                             'non-negative before_ms and after_ms (step_ms: None, or fixed regions 0 < step_ms <= 120000)')
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'audio', 'gamst.word_verification', '1', (
                OutputField('kind', 'string'), OutputField('reference', 'string'), OutputField('start_ms', 'integer'),
                OutputField('end_ms', 'integer'), OutputField('verdicts', 'array', required=False),
                OutputField('utterance_id', 'string', required=False), OutputField('status', 'string', required=False),
                OutputField('text', 'string', required=False), OutputField('words', 'array', required=False),
                OutputField('speaker', 'object', required=False), OutputField('speakers', 'array', required=False),
                OutputField('region', 'object', required=False),
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
        if self.voice['decision'] == 'relative':
            from .voice_bank import shared
            self.bank = shared(pipeline, self.reference_name, reference['value']['embeddings'], self.bank_settings)
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
        self._voted = []               # final utterances of other recognisers this lane reads (votes)
        self._windows = {}             # index -> {"turns", "speakers", "person"} | asyncio task
        self._decided = 0              # regions decided so far
        self._regions = []             # closed regions [{"start_ms", "end_ms", "reason"}], in capture order
        self._used = set()             # ids of second-hearing occurrences that confirmed or were added
        self._outbox, self._scores, self._anchors = [], {}, {}
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
                    heard_terms=[dict(start_ms=item['start_ms'], end_ms=item['end_ms'], term=item['term'],
                                      spoken=item['spoken'], timed=item['timed'], aligned=bool(value.get('aligned')))
                                 for value in self._heard for item in term_occurrences(value, self.terms or {})][-256:],
                    diarize_ms=self.stats['diarize_ms'][-16:],
                    voted_terms=[dict(start_ms=item['start_ms'], end_ms=item['end_ms'], term=item['term'],
                                      spoken=item['spoken'], source=item['source'])
                                 for item in self._vote_occurrences(float('-inf'), float('inf'))][-256:] if self.terms else [],
                    recogniser_errors=[str(item)[:200] for item in getattr(self.recogniser, 'errors', ())][-8:],
                    heard_turns=[dict(start_ms=value['start_ms'], end_ms=value['end_ms'], words=len(value.get('words') or ()),
                                      aligned=bool(value.get('aligned'))) for value in self._heard][-64:],
                    fed_until_ms=self._fed, decided_until_ms=None if self._origin is None else self._region(self._decided)[0],
                    regions=[dict(item, end_ms=None if item['end_ms'] == float('inf') else item['end_ms'],
                                  length_ms=None if item['end_ms'] == float('inf') else item['end_ms'] - item['start_ms'])
                             for item in self._regions[-64:]], straddling=self._straddling(),
                    diarize_jobs=self.stats.get('diarize_jobs', 0),
                    **({'voice_bank': self.bank.status()} if self.bank is not None else {}))

    # ------------------------------------------------------------------ frames

    def process(self, frame):
        value = frame.data
        if isinstance(value, dict):
            if 'kind' not in value and value.get('status') == 'final' and value.get('words'):
                self._voted.append(value)      # a recogniser's own words (speech.transcript_words): a vote
                return self._take()
            if value.get('kind') == 'utterance' and value.get('status') == 'final' and \
                    value.get('reference', self.reference_name) == self.reference_name:
                words = [word for word in value.get('words') or () if isinstance(word, dict)
                         and word.get('state') == 'provisional' and isinstance(word.get('word_id'), str)]
                if words:
                    self._utterances.append(value)
                    for word in words:
                        self._open[word['word_id']] = dict(word, utterance_id=value.get('utterance_id'))
                    self._repetitions(value)
                    self._late()
        elif frame.modality == 'audio' and self._origin is None and self.memory is not None and self.memory.start_ms:
            self._origin = int(self.memory.start_ms)
        if self._task is None and self._origin is not None and not self._done:
            self._task = asyncio.ensure_future(self._work())
        return self._take()

    def _repetitions(self, value):
        """A word holding several repetitions of a whole-word term (``word:달*``: "달달달") is decided per repetition:
        each is an open word of its own (:func:`gamst.providers.term_events.part_id`, its share of the word's time)."""
        whole = set()
        for item in term_occurrences(value, self.terms or {}):
            if item.get('part') is None:
                continue
            for word, word_id in zip([word for word in item['words'] if isinstance(word.get('word_id'), str)],
                                     item['word_ids']):
                if word['word_id'] in self._open:
                    self._open[word_id] = dict(self._open[word['word_id']], word_id=word_id, start_ms=item['start_ms'],
                                               end_ms=item['end_ms'])
                    whole.add(word['word_id'])
        for word_id in whole:
            del self._open[word_id]

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
        """(start, end) of a closed region; past the closed ones: (where the next one starts, inf)."""
        if index < len(self._regions):
            return self._regions[index]['start_ms'], self._regions[index]['end_ms']
        return (self._regions[-1]['end_ms'] if self._regions else self._origin), float('inf')

    def _window(self, index):
        low, high = self._region(index)
        end = self.memory.end_ms if self.memory is not None and self.memory.end_ms is not None else low
        return max(self._origin, low - self.windows['before_ms']), min(high + self.windows['after_ms'], max(end, low))

    def _spans(self, start):
        """Merged [start, end] of every utterance of every recogniser that ends after ``start``."""
        found = sorted((value['start_ms'], value['end_ms']) for values in (self._utterances, self._heard, self._voted)
                       for value in values if type(value.get('start_ms')) is int and type(value.get('end_ms')) is int
                       and value['end_ms'] > start)
        merged = []
        for low, high in found:
            if merged and low <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([low, high])
        return merged

    def _close_regions(self, complete):
        """Close regions where speech pauses (between two utterances of any recogniser, or ``pause_ms`` into the
        silence after the last one), once the region is ``min_ms`` long and at most ``max_ms``; past ``max_ms`` the next
        pause closes it (``max_extended``); at ``cap_ms`` the cap itself does (``cap_split``, the only case in which an
        utterance is split, and said so). ``step_ms``: fixed regions."""
        w, end = self.windows, self.memory.end_ms
        if end is None or self._origin is None:
            return False
        horizon = float('inf') if self._final else min(complete, self._heard_until())
        closed = False
        while True:
            start = self._region(len(self._regions))[0]
            if start >= end:
                return closed
            if w['step_ms'] is not None:
                stop, reason = start + w['step_ms'], 'fixed'
                if stop > horizon and not self._final:
                    return closed
                if self._final and stop >= end:
                    stop, reason = float('inf'), 'end'
            else:
                stop, reason = self._boundary(start, horizon, end)
                if stop is None:
                    return closed
            self._regions.append(dict(start_ms=start, end_ms=stop, reason=reason))
            closed = True
            if stop == float('inf'):
                return closed

    def _boundary(self, start, horizon, end):
        w = self.windows
        spans = self._spans(start)
        limit = min(horizon, end) if not self._final else end
        gaps, at = [], start
        for low, high in spans:
            if low > at:
                gaps.append((at, min(low, limit)))
            at = max(at, high)
        if at < limit:
            gaps.append((at, limit))             # silence up to what is known
        gaps = [(low, high) for low, high in gaps if high > low]

        def first(low, high):
            """The first pause point in [low, high]: the middle of a gap between two utterances (an utterance ended
            there), or, in the silence after the last one, ``pause_ms / 2`` into it once ``pause_ms`` of it is known."""
            for gap_low, gap_high in gaps:
                if gap_high >= limit and not self._final:
                    if gap_high - gap_low < w['pause_ms']:
                        continue
                    earliest, latest = gap_low + w['pause_ms'] / 2, gap_high - w['pause_ms'] / 2
                    point = min(max(earliest, low), high)
                else:
                    earliest = latest = point = (gap_low + gap_high) / 2
                    if not low <= point <= high:
                        continue
                if earliest <= point <= latest:
                    return int(point)
            return None
        if self._final and end - start <= w['max_ms']:
            return float('inf'), 'end'
        found = first(start + w['min_ms'], start + w['max_ms'])
        if found is not None:
            return found, 'pause'
        if limit < start + w['max_ms']:
            return (float('inf'), 'end') if self._final else (None, None)
        found = first(start + w['max_ms'], start + w['cap_ms'])
        if found is not None:
            return found, 'max_extended'
        if limit < start + w['cap_ms']:
            return (float('inf'), 'end') if self._final else (None, None)
        return start + w['cap_ms'], 'cap_split'

    def _straddling(self):
        """Utterances (of any recogniser) that a closed region boundary cuts: 0 unless a region says ``cap_split``."""
        cuts = [item['end_ms'] for item in self._regions if item['end_ms'] != float('inf')]
        return sum(1 for values in (self._utterances, self._heard, self._voted) for value in values
                   if type(value.get('start_ms')) is int and any(value['start_ms'] < cut < value['end_ms'] for cut in cuts))

    async def _work(self):
        try:
            while True:
                complete = self._complete()
                progressed = await self._feed(complete)
                progressed = self._close_regions(complete) or progressed
                progressed = self._diarize(complete) or progressed
                progressed = await self._decide(complete) or progressed
                self._hold()
                if self._final and self._regions and self._regions[-1]['end_ms'] == float('inf') \
                        and self._decided == len(self._regions):
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
            self._decided == len(self._regions) and bool(self._regions) and self._regions[-1]['end_ms'] == float('inf')

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
        while index < len(self._regions):
            low, high = self._window(index)
            if end is None or low >= end:
                break
            if high > complete and not self._final:
                break
            if index not in self._windows:
                running = sum(isinstance(item, asyncio.Task) and not item.done() for item in self._windows.values())
                if running >= self.windows['concurrency']:
                    break
                self._windows[index] = asyncio.ensure_future(self._identify(index, low, min(high, end)))
                self.stats['diarize_jobs'] = self.stats.get('diarize_jobs', 0) + 1
                started = True
            index += 1
        return started

    async def _identify(self, index, low, high):
        """Diarization of one window, and which of its speakers is the person (each one's own sound vs the voice)."""
        result = await self._diarized(low, high)
        speakers = result['speakers']
        region = self._regions[index] if index < len(self._regions) else {}
        self._outbox.append(self.spec.validate(dict(
            kind='window', reference=self.reference_name, start_ms=int(low), end_ms=int(high),
            speakers=sorted(speakers, key=lambda item: -item['speech_ms']),
            region=dict(start_ms=int(region.get('start_ms', low)),
                        end_ms=None if region.get('end_ms', float('inf')) == float('inf') else int(region['end_ms']),
                        reason=region.get('reason')), model=self._model())))
        return result

    async def _anchored(self, start, end):
        """The diarized window anchored to one word (``anchor_ms`` before / after its end), shared by words whose ends
        are within ``anchor_join_ms``; it does not depend on where a region begins or ends."""
        before, after = self.voice['anchor_ms']
        join = self.voice['anchor_join_ms']
        key = next((key for key in self._anchors if abs(key - end) <= join), None)
        if key is None:     # a task: words of one region ask at once (:meth:`_prefetch`), each window once
            low = max(end - before, int(self.memory.start_ms or 0))
            high = min(end + after, int(self.memory.end_ms or end + after))
            self._anchors[end] = asyncio.ensure_future(self._diarized(low, high))
            self.stats['anchor_jobs'] = self.stats.get('anchor_jobs', 0) + 1
            while len(self._anchors) > 64:
                self._anchors.pop(next(iter(self._anchors)))
            key = end
        return await asyncio.shield(self._anchors[key])

    async def _prefetch(self, low, high):
        """Diarize the anchored windows of every word of a region together (at most ``concurrency`` at once)."""
        if self.voice['anchor_ms'] is None:
            return
        near = self.rule['near_ms']
        ends = sorted({item['end_ms'] for item in (*self._first_occurrences(low, high),
                                                    *self._second_occurrences(low - near, high + near))}
                      | {word['end_ms'] for word in self._open.values() if low <= word['end_ms'] < high})
        limit = asyncio.Semaphore(self.windows['concurrency'])

        async def one(end):
            async with limit:
                await self._anchored(end, end)
        await asyncio.gather(*(one(end) for end in ends))

    async def _diarized(self, low, high):
        """Turns, speakers (each one's score and vector) and who of them is the person in [low, high)."""
        samples = self.memory.span(low, high, SAMPLE_RATE)
        began = time.monotonic()
        segments = await asyncio.to_thread(self.diarizer.diarize, samples) if len(samples) >= SAMPLE_RATE else []
        self.stats['diarize_ms'].append(round(1000 * (time.monotonic() - began)))
        self.stats['windows'] += 1
        turns = [(low + int(round(1000 * start)), low + int(round(1000 * end)), str(speaker)[:40])
                 for start, end, speaker in segments if end > start]
        speakers, vectors = [], {}
        for speaker, pieces in exclusive(turns, low, high).items():
            speech = sum(end - start for start, end in pieces)
            vector = await self._speaker_vector(pieces) if speech >= self.voice['min_speaker_ms'] else None
            score = None if vector is None else round(cosine(vector, self.target), 5)
            if vector is not None:
                vectors[speaker] = vector
            speakers.append(dict(speaker=speaker, speech_ms=int(speech), score=score,
                                 person=score is not None and score >= self.voice['threshold']))
        if self.bank is not None:       # relative: the speaker closest to the person, by a margin over the others
            close = {speaker: self.bank.pos(vector) for speaker, vector in vectors.items()}
            best = max(close, key=close.get) if close else None
            others = [value for speaker, value in close.items() if speaker != best]
            clear = best is not None and close[best] >= self.voice['floor'] and (
                not others or close[best] - max(others) >= self.voice['margin'])
            for item in speakers:
                item['person'] = clear and item['speaker'] == best
                if item['speaker'] in close:
                    item['score'] = round(close[item['speaker']], 5)
            if clear:       # the other voices of this window are not the person: negatives, without labels
                for speaker, vector in vectors.items():
                    if speaker != best:
                        self.bank.add_negative(vector, (low, high, speaker))
        for speaker in {turn[2] for turn in turns} - {item['speaker'] for item in speakers}:
            speakers.append(dict(speaker=speaker, speech_ms=0, score=None, person=False))
        return dict(turns=turns, speakers=speakers, low=low, high=high, vectors=vectors,
                    person={item['speaker'] for item in speakers if item['person']},
                    unknown={item['speaker'] for item in speakers if item['score'] is None})

    async def _speaker_score(self, pieces):
        vector = await self._speaker_vector(pieces)
        return None if vector is None else round(cosine(vector, self.target), 5)

    async def _speaker_vector(self, pieces):
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
        return l2_normalize([sum(vector[i] * count for vector, count in vectors) / weight
                             for i in range(len(vectors[0][0]))])

    async def _word_score(self, start, end):
        vector = await self._word_vector(start, end)
        return None if vector is None else round(cosine(vector, self.target), 5)

    async def _word_vector(self, start, end):
        key = (start, end)
        if key not in self._scores:
            middle, half = (start + end) // 2, max(self.voice['word_context_ms'], end - start) // 2
            samples = self.memory.span(middle - half, middle + half, SAMPLE_RATE)
            vector = None
            if len(samples) >= SAMPLE_RATE // 4 and rms_dbfs(samples) >= self.voice['min_dbfs']:
                try:
                    vector = await embed(self.embedder, samples, SAMPLE_RATE)
                except ValueError:
                    vector = None
            self._scores[key] = vector
            while len(self._scores) > 4096:
                self._scores.pop(next(iter(self._scores)))
        return self._scores[key]

    async def _word_match(self, window, start, end):
        """(matched, score) of one word's own sound: the threshold, or (relative) the references against the negatives
        known now and the window's other speakers."""
        vector = await self._word_vector(start, end)
        if vector is None:
            return False, None
        if self.bank is None:
            score = round(cosine(vector, self.target), 5)
            return score >= self.voice['threshold'], score
        extra = [value for speaker, value in (window.get('vectors') or {}).items() if speaker not in window['person']]
        matched, pos, _ = self.bank.decide(vector, floor=self.voice['floor'], margin=self.voice['word_margin'],
                                           extra=extra)
        return matched, pos

    def _tse_at(self, start, end, words=None):
        """The TSE lane's decision (``tse`` on a provisional word, :mod:`gamst.providers.overlap_tse`) over [start, end),
        from ``words`` or else from the open provisional words there; None where it decided none."""
        if words is None:
            words = [word for word in self._open.values() if type(word.get('start_ms')) is int
                     and word['start_ms'] < max(end, start + 1) and word.get('end_ms', word['start_ms']) > start]
        return next((word['tse'] for word in words if isinstance(word.get('tse'), dict)
                     and word['tse'].get('decision') in ('person', 'other', 'uncertain')), None)

    async def _voice_of(self, window, start, end, words=None):
        """(match, basis, reason or None, speaker, its score, per-word score) of [start, end) in one window.

        Overlapped speech (the person's speaker and another over ``overlap_share``): the TSE lane's decision where a
        provisional word there carries one (``words``, else the open words over the span), else the per-word score.
        With ``anchor_ms`` the window is the one diarized around this word (:meth:`_anchored`)."""
        if self.voice['anchor_ms'] is not None:
            window = await self._anchored(start, end)
        shares = _share(window['turns'], start, end)
        active = {speaker: share for speaker, share in shares.items() if share >= self.voice['active_share']}
        person = max((share for speaker, share in active.items() if speaker in window['person']), default=0.)
        other = max((share for speaker, share in active.items() if speaker not in window['person']), default=0.)
        dominant = max(active, key=active.get) if active else None
        scores = {item['speaker']: item['score'] for item in window['speakers']}
        limit = self.voice['overlap_share']
        if person >= limit and other >= limit:
            tse = self._tse_at(start, end, words)
            if tse is not None:
                decision = tse['decision']
                reason = None if decision == 'person' else 'other_speaker' if decision == 'other' else 'uncertain_overlap'
                return decision == 'person', f'overlap_tse_{decision}', reason, dominant, scores.get(dominant), \
                    tse.get('score')
            matched, score = await self._word_match(window, start, end)
            return matched, 'overlap_voice_score', None if matched else 'uncertain_overlap', dominant, \
                scores.get(dominant), score
        if active and person >= other:
            if self.bank is not None:       # the person's own speech, decided final: one more reference
                speaker = max((name for name in active if name in window['person']), key=active.get)
                if speaker in (window.get('vectors') or {}):
                    self.bank.add_person(window['vectors'][speaker], (window['low'], window['high'], speaker))
            return True, 'person_speaker', None, dominant, scores.get(dominant), None
        if active and dominant not in window['unknown']:
            return False, 'other_speaker', 'other_speaker', dominant, scores.get(dominant), None
        matched, score = await self._word_match(window, start, end)
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
                ids = [word_id for word_id in item['word_ids'] if word_id in self._open]
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
        while self._decided < len(self._regions):
            low, high = self._region(self._decided)
            end = self.memory.end_ms
            task = self._windows.get(self._decided)
            if end is None or task is None or not task.done():
                return decided
            last = high == float('inf')
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
        return decided

    async def _settle(self, window, low, high):
        await self._prefetch(low, high)
        if self.rule['decide'] in ('majority', 'primary'):
            return await self._settle_majority(window, low, high, primary=self.rule['decide'] == 'primary')
        near = self.rule['near_ms']
        first = self._first_occurrences(low, high)
        second = self._second_occurrences(low - near, high + near)
        pairs = match_occurrences(first, second, near)
        voted = self._vote_occurrences(low - near, high + near) if self.rule['vote'] else []
        votes = match_occurrences(first, voted, near) if voted else {}
        verdicts, words_done = [], set()
        for index, item in enumerate(first):
            if index in pairs or index in votes:
                heard = second[pairs[index]] if index in pairs else item
                if index in pairs:
                    self._used.add(heard['key'])
                match, basis, reason, speaker, speaker_score, score = await self._voice_of(
                    window, heard['start_ms'], heard['end_ms'])
                basis = ('heard_twice+' if index in pairs else 'heard_by_vote+') + basis
            else:
                match, basis, reason, speaker, speaker_score, score = await self._heard_once(window, item)
            for word_id in item['word_ids']:
                verdicts.append(self._verdict(word_id, match, basis, reason, speaker, speaker_score, score,
                                              heard=index in pairs))
                words_done.add(word_id)
        for word_id, word in list(self._open.items()):     # the other words: by the voice alone
            if word_id in words_done or not (low <= word['end_ms'] < high):
                continue
            match, basis, reason, speaker, speaker_score, score = await self._voice_of(window, word['start_ms'], word['end_ms'])
            verdicts.append(self._verdict(word_id, match, basis, reason, speaker, speaker_score, score, heard=None))
        if verdicts:
            self.stats['verdicts'] += len(verdicts)
            self._outbox.append(self.spec.validate(dict(
                kind='verdicts', reference=self.reference_name, start_ms=min(item['start_ms'] for item in verdicts),
                end_ms=max(item['end_ms'] for item in verdicts), verdicts=verdicts, model=self._model())))
        voted_here = self._vote_occurrences(low - near, high + near) if self.rule['vote'] else []
        for item in self._second_occurrences(low, high):     # heard only the second time: final at once
            self._used.add(item['key'])
            if self.rule['vote'] and not self.rule['vote_adds_once'] and not match_occurrences([item], voted_here, near):
                continue
            match, basis, reason, speaker, speaker_score, score = await self._voice_of(window, item['start_ms'], item['end_ms'])
            self.stats['added'] += 1
            uncertain = reason == 'uncertain_overlap'
            word = dict(text=item['spoken'], start_ms=item['start_ms'], end_ms=max(item['end_ms'], item['start_ms'] + 1),
                        match=match, decided_by='verification', basis='heard_once_by_verification+' + basis,
                        **({'reason': reason} if reason else {}), **({'speaker_score': score} if score is not None else {}),
                        # a later lane may settle an uncertain word: it has an ID and says so
                        **({'state': 'uncertain', 'word_id': f"verification:{item['key']}#0"} if uncertain else {}))
            self._outbox.append(self.spec.validate(dict(
                kind='utterance', reference=self.reference_name, utterance_id=f"verification:{item['key']}",
                status='final', text=item['spoken'], start_ms=word['start_ms'], end_ms=word['end_ms'], words=[word],
                speaker=dict(reference=self.reference_name, match=match, model=self.embedder.model_id),
                model=self._model())))

    def _words_near(self, values, start, end):
        """Did any of these utterances have a word (of anything) within ``near_ms`` of [start, end)?"""
        near = self.rule['near_ms']
        return any(type(word.get('start_ms')) is int and type(word.get('end_ms')) is int and
                   word['start_ms'] < end + near and word['end_ms'] > start - near
                   for value in values for word in (value.get('words') or ()) if isinstance(word, dict))

    async def _clear_voice(self, window, start, end):
        """(clear, voice decision) of a span: the rule's ``clear`` on the diarized speaker and the per-word score."""
        decided = await self._voice_of(window, start, end)
        score = await self._word_score(start, end)
        by_speaker = decided[1] == 'person_speaker'
        by_score = score is not None and score >= self.voice['threshold'] + self.rule['clear_margin']
        clear = dict(speaker=by_speaker, score=by_score, either=by_speaker or by_score, both=by_speaker and by_score)[
            self.rule['clear']]
        return clear, decided[:5] + (score,)

    async def _settle_majority(self, window, low, high, primary=False):
        """At least two of the recognisers wrote it -> said (the voice decides who); one -> dropped, unless
        ``keep_single``, nobody else wrote anything there and the voice is clearly the person's.

        ``primary``: the first hearing (the live recogniser, the best one by published results) is never outvoted: a
        word only it wrote stands unless the others contest it (:meth:`_primary_alone`). Words only the others wrote
        are added as by majority (``add: "agreed"``) or whenever one of them wrote it (``add: "any"``)."""
        near = self.rule['near_ms']
        first = self._first_occurrences(low, high)
        second = self._second_occurrences(low - near, high + near)
        third = [dict(item, key=f"v{item['source']}:{item['end_ms']}") for item in self._vote_occurrences(low - near, high + near)]
        third = [item for item in third if item['key'] not in self._used]
        with_second = match_occurrences(first, second, near)
        with_third = match_occurrences(first, third, near)
        verdicts, words_done = [], set()
        sources = dict(first=self._utterances, second=self._heard, third=self._voted)

        async def single(item, own):
            others = [values for name, values in sources.items() if name != own]
            alone = not any(self._words_near(values, item['start_ms'], item['end_ms']) for values in others)
            if not self._speech_near(window, item['start_ms'], item['end_ms']):
                return False, 'heard_by_one+no_speech', 'no_speech', None, None, None
            if self.rule['keep_single'] and alone:
                clear, (match, basis, why, speaker, speaker_score, score) = await self._clear_voice(
                    window, item['start_ms'], item['end_ms'])
                if clear:
                    return True, 'heard_by_one+alone+clear_voice+' + basis, None, speaker, speaker_score, score
                if why == 'uncertain_overlap':
                    return False, 'heard_by_one+alone+' + basis, why, speaker, speaker_score, score
                return False, 'heard_by_one+alone+' + basis, 'heard_by_one', speaker, speaker_score, score
            return False, 'heard_by_one' + ('+alone' if alone else '+others_heard_speech'), 'heard_by_one', None, None, None

        for index, item in enumerate(first):
            heard = [name for name, pairs in (('second', with_second), ('third', with_third)) if index in pairs]
            if heard:
                span = second[with_second[index]] if index in with_second else third[with_third[index]]
                for name, pairs, items in (('second', with_second, second), ('third', with_third, third)):
                    if index in pairs:
                        self._used.add(items[pairs[index]]['key'])
                match, basis, reason, speaker, speaker_score, score = await self._voice_of(
                    window, span['start_ms'], span['end_ms'])
                basis = f"heard_by_{1 + len(heard)}({'+'.join(['first', *heard])})+" + basis
            elif primary:
                match, basis, reason, speaker, speaker_score, score = await self._primary_alone(window, item)
            else:
                match, basis, reason, speaker, speaker_score, score = await single(item, 'first')
            confidence = self._confidence(item) if primary else None
            for word_id in item['word_ids']:
                verdicts.append(self._verdict(word_id, match, basis, reason, speaker, speaker_score, score,
                                              heard='+'.join(['first', *heard]),
                                              extra=None if confidence is None else dict(primary_confidence=confidence)))
                words_done.add(word_id)
        for word_id, word in list(self._open.items()):     # the other words: by the voice alone
            if word_id in words_done or not (low <= word['end_ms'] < high):
                continue
            match, basis, reason, speaker, speaker_score, score = await self._voice_of(window, word['start_ms'], word['end_ms'])
            verdicts.append(self._verdict(word_id, match, basis, reason, speaker, speaker_score, score, heard=None))
        if verdicts:
            self.stats['verdicts'] += len(verdicts)
            self._outbox.append(self.spec.validate(dict(
                kind='verdicts', reference=self.reference_name, start_ms=min(item['start_ms'] for item in verdicts),
                end_ms=max(item['end_ms'] for item in verdicts), verdicts=verdicts, model=self._model())))
        # Words the first hearing did not write: said when the second and a voter both wrote them, else single.
        second = [item for item in second if item['key'] not in self._used and low <= item['end_ms'] < high]
        third = [item for item in third if item['key'] not in self._used]
        pairs = match_occurrences(second, third, near)
        candidates = [(item, 'second', third[pairs[index]] if index in pairs else None) for index, item in enumerate(second)]
        used_third = {third[j]['key'] for j in pairs.values()}
        candidates += [(item, 'third', None) for item in third if item['key'] not in used_third
                       and low <= item['end_ms'] < high]
        for item, own, partner in candidates:
            self._used.add(item['key'])
            if partner is not None:
                self._used.add(partner['key'])
                match, basis, reason, speaker, speaker_score, score = await self._voice_of(
                    window, item['start_ms'], item['end_ms'])
                basis = 'heard_by_2(second+third)+' + basis
            elif primary and self.rule['add'] == 'any':      # one other recogniser is enough to add a word
                match, basis, reason, speaker, speaker_score, score = await self._voice_of(
                    window, item['start_ms'], item['end_ms'])
                basis = f'heard_by_1({own})+' + basis
            else:
                match, basis, reason, speaker, speaker_score, score = await single(item, own)
                if not match and reason != 'uncertain_overlap':
                    self.stats['dropped'] = self.stats.get('dropped', 0) + 1
                    continue       # one recogniser only, not kept: nothing is added
            self._add(item, match, basis, reason, score)

    @staticmethod
    def _confidence(item):
        """The primary recogniser's own confidence (0-1) in an occurrence: the lowest of its words' ``confidence``
        or ``exp(logprob)``; None when the recogniser gave none."""
        found = []
        for word in item.get('words') or ():
            if type(word.get('confidence')) in (int, float):
                found.append(float(word['confidence']))
            elif type(word.get('logprob')) in (int, float):
                found.append(math.exp(float(word['logprob'])))
        return round(min(found), 4) if found else None

    async def _primary_alone(self, window, item):
        """A word only the primary (first) recogniser wrote. The others never remove it by outnumbering it: it stands
        (the voice decides whose it is) unless it is *contested*, i.e. the other recognisers (``contest_by``: ``all``
        of those that are read, or ``any``) wrote other words within ``near_ms`` of it (they heard speech there and
        wrote something else). A contested word is vetoed (revoked, ``vetoed``) only when the primary's own
        confidence is under ``low_confidence`` too; else it is ``uncertain`` (``contested: "uncertain"``: shown, not
        counted, never "not said") or stands (``contested: "stand"``). A recogniser that gives no word confidence is
        never vetoed. A word the voice gives to another speaker stays that speaker's."""
        start, end = item['start_ms'], item['end_ms']
        others = dict(second=self._heard, **({'third': self._voted} if self._voted else {}))
        wrote = [name for name, values in others.items() if self._words_near(values, start, end)]
        contested = len(wrote) == len(others) if self.rule['contest_by'] == 'all' else bool(wrote)
        match, basis, reason, speaker, speaker_score, score = await self._voice_of(window, start, end)
        if not contested:
            return match, 'primary+uncontested+' + basis, reason, speaker, speaker_score, score
        tag = f"primary+contested({'+'.join(wrote)})"
        confidence, low = self._confidence(item), self.rule['low_confidence']
        if low is not None and confidence is not None and confidence < low:
            return False, f'{tag}+low_confidence', 'vetoed', speaker, speaker_score, score
        if self.rule['contested'] == 'stand' or not match and reason not in UNCERTAIN:
            return match, f'{tag}+{basis}', reason, speaker, speaker_score, score
        return False, f'{tag}+{basis}', 'contested', speaker, speaker_score, score

    def _add(self, item, match, basis, reason, score):
        self.stats['added'] += 1
        uncertain = reason == 'uncertain_overlap'
        word = dict(text=item['spoken'], start_ms=item['start_ms'], end_ms=max(item['end_ms'], item['start_ms'] + 1),
                    match=match, decided_by='verification', basis=basis,
                    **({'reason': reason} if reason else {}), **({'speaker_score': score} if score is not None else {}),
                    **({'state': 'uncertain', 'word_id': f"verification:{item['key']}#0"} if uncertain else {}))
        self._outbox.append(self.spec.validate(dict(
            kind='utterance', reference=self.reference_name, utterance_id=f"verification:{item['key']}",
            status='final', text=item['spoken'], start_ms=word['start_ms'], end_ms=word['end_ms'], words=[word],
            speaker=dict(reference=self.reference_name, match=match, model=self.embedder.model_id),
            model=self._model())))

    async def _heard_once(self, window, item):
        """A first-hearing occurrence the second recogniser did not write: kept when the voice is clearly the person's,
        uncertain in unclear overlap, else revoked (the rule's settings)."""
        speech = self._speech_near(window, item['start_ms'], item['end_ms'])
        revoke = self.rule['revoke_not_heard'] if speech else self.rule['revoke_no_speech']
        reason = 'not_heard' if speech else 'no_speech'
        if not revoke:
            match, basis, why, speaker, speaker_score, score = await self._voice_of(window, item['start_ms'], item['end_ms'])
            return match, 'heard_once+' + basis, why, speaker, speaker_score, score
        if not speech or not self.rule['keep_clear_voice']:
            return False, reason, reason, None, None, None
        match, basis, why, speaker, speaker_score, _ = await self._voice_of(window, item['start_ms'], item['end_ms'])
        score = await self._word_score(item['start_ms'], item['end_ms'])
        by_speaker = basis == 'person_speaker'
        by_score = score is not None and score >= self.voice['threshold'] + self.rule['clear_margin']
        clear = dict(speaker=by_speaker, score=by_score, either=by_speaker or by_score, both=by_speaker and by_score)[
            self.rule['clear']]
        if clear:
            return True, 'heard_once+clear_voice+' + basis, None, speaker, speaker_score, score
        if why == 'uncertain_overlap':
            return False, 'heard_once+' + basis, why, speaker, speaker_score, score
        return False, f'not_heard+{basis}', reason, speaker, speaker_score, score

    def _vote_occurrences(self, low, high):
        found = []
        for value in self._voted:
            for item in term_occurrences(value, self.terms):
                if low <= item['end_ms'] < high:
                    found.append(dict(item, source=str(value.get('utterance_id', '')).rsplit(':', 1)[0]))
        return found

    def _verdict(self, word_id, match, basis, reason, speaker, speaker_score, score, *, heard, extra=None):
        word = self._open.pop(word_id)
        provisional = bool(word.get('match'))
        verdict = dict(word_id=word_id, utterance_id=word.get('utterance_id'), start_ms=word['start_ms'],
                       end_ms=word['end_ms'], match=bool(match), provisional=provisional, changed=bool(match) != provisional,
                       settled_by='verification', basis=basis)
        if reason and not match:
            verdict['reason'] = reason
            if reason in UNCERTAIN:
                verdict['uncertain'] = True        # neither surely the person's nor another's: a later lane may settle it
        verdict.update(extra or {})
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
