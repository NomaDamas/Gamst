"""Replay a recorded ASR's words at the moment it gave them, for measuring what comes after the ASR.

An ASR run once (``speech.transcript_words`` records of a store) is exported to a JSONL file, one utterance per line,
times in seconds from the start of the recording::

    {"utterance_id": "u1", "start_s": 11.6, "end_s": 12.1, "known_s": 12.9,      # known_s: when the ASR had it
     "text": "뽀또", "words": [{"text": "뽀또", "start_s": 11.6, "end_s": 12.04}]}

Only the text of words that match the counted terms is kept; every other word is ``"_"`` (its time stays), so the
file holds no transcript of what people said.

A recording of the record values themselves (capture times in ms, ``final_after_ms``: how long after the utterance's
end the ASR made it final) is read too, with ``origin_ms`` (the capture time of the recording's first sample; default:
``<file>.meta.json``, or for ``x.words.jsonl`` ``x.meta.json``). :class:`TranscriptReplay` reads the microphone only for its clock and
gives each utterance once the recording has reached ``known_s`` (or its end), with capture times of this run: the
modules after it see the same words at the same delay as from the live ASR.

    python -m gamst.providers.transcript_replay --store run.db --report run.json --terms '뽀또,re:(포토|보또)' --out words.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .asr_words import words_capability
from .term_events import matches

HIDDEN = '_'


class TranscriptReplay:
    """Pipeline module: the microphone in (as a clock), recorded ``speech.transcript_words`` out."""

    def __init__(self, path, *, model='replay', origin_ms=None):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if rows and 'end_ms' in rows[0]:
            meta = next((item for item in (Path(f'{path}.meta.json'), Path(str(path).replace('.words.jsonl', '.meta.json')))
                         if item.is_file()), Path(f'{path}.meta.json'))
            if origin_ms is None and meta.is_file():
                origin_ms = json.loads(meta.read_text())['origin_ms']
            if origin_ms is None:
                raise ValueError(f'{path} has capture times: give origin_ms (or {meta.name})')
            rows = [seconds(row, origin_ms) for row in rows]
        self.rows = sorted(rows, key=lambda row: row.get('known_s', row['end_s']))
        self.spec = words_capability(f'1.replay.{model}')
        self.name, self.version = 'gamst.transcript_replay', self.spec.version
        self.reset()

    def reset(self):
        self._base, self._next = None, 0

    def _value(self, row):
        at = lambda seconds: self._base + round(1000 * seconds)
        words = [dict(text=word['text'], start_ms=at(word['start_s']), end_ms=max(at(word['end_s']), at(word['start_s']) + 1))
                 for word in row['words']]
        return self.spec.validate(dict(utterance_id=f"replay:{row['utterance_id']}", status='final', text=row['text'],
                                       start_ms=at(row['start_s']), end_ms=max(at(row['end_s']), at(row['start_s']) + 1),
                                       words=words, **({'language': row['language']} if row.get('language') else {})))

    def process(self, frame):
        if frame.modality != 'audio' or frame.data is None or not frame.sample_rate:
            return None
        if self._base is None:
            self._base = frame.timestamp_ms - round(1000 * len(frame.data) / frame.sample_rate)
        out = []
        while self._next < len(self.rows) and self._base + 1000 * self.rows[self._next].get(
                'known_s', self.rows[self._next]['end_s']) <= frame.timestamp_ms:
            out.append(self._value(self.rows[self._next]))
            self._next += 1
        return out or None

    def finish(self):
        if self._base is None:
            return None
        out = [self._value(row) for row in self.rows[self._next:]]
        self._next = len(self.rows)
        return out or None


def seconds(value, origin_ms):
    """A recorded ``speech.transcript_words`` value (capture ms) as a replay row (seconds from ``origin_ms``)."""
    at = lambda ms: round((ms - origin_ms) / 1000, 3)
    return dict(utterance_id=str(value['utterance_id']), start_s=at(value['start_ms']), end_s=at(value['end_ms']),
                known_s=at(value['end_ms'] + int(value.get('final_after_ms') or 0)), text=value['text'],
                words=[dict(text=word['text'], start_s=at(word['start_ms']), end_s=at(word['end_ms'])) for word in value['words']],
                **({'language': value['language']} if value.get('language') else {}))


def hide(text, terms):
    """``text`` with every word that holds none of ``terms`` replaced by ``_``."""
    return HIDDEN if not any(matches(text, term) for term in terms) else text


def export(records, known, terms, start_ms):
    """Replay rows of ``speech.transcript_words`` records (``known``: record ID -> when the store learned it)."""
    rows = []
    for record in records:
        value = record['observation']['value']
        if value.get('status') != 'final' or not value.get('words'):
            continue
        words = [dict(text=hide(word.get('text', ''), terms), start_s=round((word['start_ms'] - start_ms) / 1000, 3),
                      end_s=round((word['end_ms'] - start_ms) / 1000, 3))
                 for word in value['words'] if type(word.get('start_ms')) is int and type(word.get('end_ms')) is int]
        if not words:
            continue
        seen = known.get(record['id'])
        rows.append(dict(utterance_id=str(len(rows) + 1), start_s=words[0]['start_s'], end_s=words[-1]['end_s'],
                         **({'known_s': round((seen - start_ms) / 1000, 3)} if seen is not None else {}),
                         text=' '.join(word['text'] for word in words), words=words,
                         **({'language': value['language']} if value.get('language') else {})))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m gamst.providers.transcript_replay', description=__doc__.split('\n')[0])
    parser.add_argument('--store', required=True, help='SQLite store of a run that transcribed the recording')
    parser.add_argument('--terms', required=True, help='comma-separated terms whose words keep their text')
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument('--report', help="the run's --report: its start_ms is where the recording starts")
    start.add_argument('--start-ms', type=int, help='capture time of the recording start')
    parser.add_argument('--live', action='store_true', help='the run was --as-live: known_s is when each utterance was known')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    from ..storage import Store
    store = Store(args.store)
    records = [record for record in reversed(store.query('local-user', kind='observation.signal', limit=20000))
               if (record['observation'].get('name') or '') == 'speech.transcript_words'
               or (record['observation'].get('capability') or {}).get('id') == 'speech.transcript_words']
    if not records:
        raise SystemExit(f'{args.store} has no speech.transcript_words records')
    start = args.start_ms if args.start_ms is not None else json.loads(Path(args.report).read_text())['start_ms']
    if args.report and not args.live:
        args.live = bool(json.loads(Path(args.report).read_text()).get('paced'))
    known = store.known_times([record['id'] for record in records]) if args.live else {}
    terms = [item.strip() for item in args.terms.split(',') if item.strip()]
    rows = export(records, known, terms, start)
    Path(args.out).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    print(json.dumps(dict(out=args.out, utterances=len(rows), kept=sum(word['text'] != HIDDEN for row in rows
                                                                       for word in row['words']))))


if __name__ == '__main__':
    main()
