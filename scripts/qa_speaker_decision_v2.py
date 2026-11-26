"""Speaker decision v2, measured on the owner's labelled words of the four clips with leave-one-clip-out.

    PYTHONPATH=<worktree> <sherpa-onnx venv python> scripts/qa_speaker_decision_v2.py prepare   # diarize + embed (hosted)
    PYTHONPATH=<worktree> <sherpa-onnx venv python> scripts/qa_speaker_decision_v2.py evaluate  # rules, folds (local)

Words: every place the owner labelled (person = 감스트 / other = 다른 사람; "nothing said" places are not about who).
Reference: the same-clip voice the live run used (``testbed/<clip>/hud/m1.voice.json``: the owner range, else the
automatic main speaker of the first 15 s); words inside its window are left out. ``prepare`` diarizes a window anchored
to each word (pyannoteAI Precision-3, ``before``/``after`` seconds around the word's end; two settings) and stores per
word: the word's own CAM++ embedding (1.5 s around it) and each diarized speaker's embedding (its sound where it alone
speaks inside the window), plus the speakers' shares over the word. No audio and no transcript is stored.

Rules (``evaluate``), all generic:

- ``current``: the word's speaker (diarized, the most active over the word) is the person when its score against the
  reference is >= 0.5; overlapped or no speaker: the word's own score >= 0.5. L1: the word's own score >= 0.5.
- ``v2``: (a) relative: the person's speaker in a window is the speaker closest to the references, and only when it
  is closer than every other speaker there by ``margin``; a word with no clear speaker is compared with the references
  AND with the negatives (speakers known not to be the person: the other speakers of its window and of earlier
  windows) and counts when ``pos - max(neg) >= word_margin``; (b) references = the enrolment + the person speaker's
  embedding of every word decided final earlier (``grow``), combined by ``max`` or ``centroid``; (c) context anchored to
  the word (``before``/``after``); (d) L1: the word's own embedding against the references and the negatives known
  at that moment (decided ``l4_delay_s`` after their word), ``pos - max(neg) >= l1_margin`` and ``pos >= l1_floor``.

Leave-one-clip-out: for each clip, the parameters with the fewest errors (person->other + other->person) on the other
three clips are applied to it (ties: fewer L1 errors, then the smallest change from the plain rule). All four folds are printed.
"""
import csv
import itertools
import json
import os
import subprocess
import sys

import numpy as np

T = '/Users/jinminseong/Desktop/catch-your-mind/.local-run/try-it/testbed'
LAB = '/Users/jinminseong/Desktop/홍명보/gamst/GAMST 라벨링'
SRC = '/Users/jinminseong/Desktop/홍명보/gamst/GAMST TestBed'
CLIPS = {'bbo': '감스트 뽀또', 'marine': '감스트의 해병대식 내리갈굼', 'fifa': '감스트 피파 억까', 'indi': '애교쟁이 인디깅'}
MODEL = '/Users/jinminseong/.cache/gamst/models/sherpa.campplus_zh_en/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'
RATE = 16000
CONTEXTS = ((10., 5.), (4., 2.))       # seconds before / after the word's end that pyannoteAI sees
OUT = os.path.join(T, 'v2')


def audio(name):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f'{SRC}/{name}.mp4', '-ac', '1', '-ar', str(RATE), '-f', 's16le',
                          '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def words(short):
    name = CLIPS[short]
    labels = json.load(open(f'{LAB}/{name} - 정답.json', encoding='utf-8'))
    found = []
    for row in csv.DictReader(open(f'{LAB}/{name} - 정답표.csv', encoding='utf-8-sig')):
        if not row['번호'].isdigit():
            continue
        who = (labels.get('labels') or {}).get(row['번호'], {}).get('who')
        if who in ('감스트', '다른 사람'):
            found.append(dict(start=float(row['시작(초)']), end=float(row['끝(초)']), person=who == '감스트'))
    for mark in labels.get('marks') or ():
        found.append(dict(start=mark['t'] - .4, end=mark['t'], person=mark.get('who') == '감스트'))
    return sorted(found, key=lambda item: item['end'])


def prepare():
    import sherpa_onnx
    from gamst.providers.pyannoteai import PyannoteAI
    from gamst.providers.slot_sources import HostedBatch
    from gamst.providers.word_verification import exclusive
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=MODEL, num_threads=2))

    def embed(x):
        stream = extractor.create_stream()
        stream.accept_waveform(RATE, x)
        stream.input_finished()
        v = np.array(extractor.compute(stream))
        return (v / np.linalg.norm(v)).round(5).tolist()

    client = PyannoteAI(model='precision-3', eur_per_hour=.112, allow_remote=True)
    diarizer = HostedBatch(client)
    os.makedirs(OUT, exist_ok=True)
    for short, name in CLIPS.items():
        path = f'{OUT}/{short}.json'
        if os.path.exists(path):
            continue
        x = audio(name)
        rows = []
        for word in words(short):
            centre = (word['start'] + word['end']) / 2
            item = dict(word, own=embed(x[int((centre - .75) * RATE):int((centre + .75) * RATE)]), contexts={})
            for before, after in CONTEXTS:
                low, high = max(0., word['end'] - before), min(len(x) / RATE, word['end'] + after)
                turns = [(a + low, b + low, s) for a, b, s in diarizer.diarize(x[int(low * RATE):int(high * RATE)])]
                alone = exclusive(turns, low, high)
                speakers = {}
                for speaker, pieces in alone.items():
                    sound = np.concatenate([x[int(a * RATE):int(b * RATE)] for a, b in pieces])
                    if len(sound) >= RATE:     # at least a second of that voice alone
                        speakers[speaker] = embed(sound)
                share = {}
                for a, b, s in turns:
                    shared = min(word['end'], b) - max(word['start'], a)
                    if shared > 0:
                        share[s] = share.get(s, 0) + shared / max(.05, word['end'] - word['start'])
                item['contexts'][f'{before:g}/{after:g}'] = dict(speakers=speakers, share=share)
            rows.append(item)
        json.dump(dict(words=rows, spent_eur=round(client.spent_eur, 5)), open(path, 'w'))
        print(short, len(rows), 'words', f'{client.spent_eur:.4f} EUR so far', flush=True)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def ref_score(e, refs, mode):
    if not refs:
        return -1.
    if mode == 'centroid':
        return float(unit(np.mean(refs, 0)) @ e)
    return max(float(r @ e) for r in refs)


def decide(clip, rule, p):
    """Per word: (person truth, L1 counted, final counted) under one rule and its parameters."""
    voice = json.load(open(f'{T}/{clip}/hud/m1.voice.json'))
    origin = voice.get('enrolled_from') or {}
    inside_sample = origin.get('segments_s') or [[origin['from_s'], origin['to_s']]]   # decision 28: the segments
    base = [unit(e) for e in voice['embeddings']]
    data = json.load(open(f'{OUT}/{clip}.json'))['words']
    refs, negatives, out = [(-1e9, e) for e in base], [], []
    for word in data:
        own = unit(word['own'])
        ctx = word['contexts'][p.get('context', '10/5')]
        speakers = {s: unit(e) for s, e in ctx['speakers'].items()}
        share = ctx['share']
        active = sorted(share, key=lambda s: -share[s])
        main = active[0] if active else None
        overlapped = len(active) > 1 and share[active[1]] >= .3
        if rule == 'current':
            l1 = float(unit(np.mean(base, 0)) @ own) >= .5
            if main in speakers and not overlapped:
                final = float(unit(np.mean(base, 0)) @ speakers[main]) >= .5
            else:
                final = l1
        else:
            known = lambda pool: [e for at, e in pool if at <= word['end']]
            now_refs = known(refs) if p['grow'] else base
            now_neg = known(negatives)
            pos = ref_score(own, now_refs, p['combine'])
            neg = max((float(n @ own) for n in now_neg), default=-1.)
            l1 = pos >= p['l1_floor'] and pos - neg >= p['l1_margin']
            scores = {s: ref_score(e, now_refs, p['combine']) for s, e in speakers.items()}
            person = max(scores, key=scores.get) if scores else None
            others = [scores[s] for s in scores if s != person]
            clear = person is not None and scores[person] >= p['floor'] and \
                (not others or scores[person] - max(others) >= p['margin'])
            if main in speakers and not overlapped:
                final = clear and main == person
            else:       # no clear speaker at the word: the word against the references and the negatives
                window_neg = [e for s, e in speakers.items() if s != person or not clear]
                neg = max((float(n @ own) for n in window_neg + now_neg), default=-1.)
                final = pos >= p['l1_floor'] and pos - neg >= p['word_margin']
            at = word['end'] + p['l4_delay_s']
            if clear:
                negatives += [(at, e) for s, e in speakers.items() if s != person]
                if final and main == person:
                    refs.append((at, speakers[person]))
        inside = any(low <= word['end'] <= high for low, high in inside_sample)
        out.append(dict(end=word['end'], person=word['person'], l1=bool(l1), final=bool(final), excluded=inside))
    return out


def errors(rows, view):
    rows = [row for row in rows if not row['excluded']]
    return dict(person=sum(r['person'] for r in rows), right=sum(r['person'] and r[view] for r in rows),
                person_to_other=sum(r['person'] and not r[view] for r in rows),
                other_to_person=sum(not r['person'] and r[view] for r in rows), other=sum(not r['person'] for r in rows))


GRID = dict(context=['10/5', '4/2'], grow=[False, True], combine=['max', 'centroid'], floor=[.3, .4, .5],
            margin=[0., .05, .1], word_margin=[0., .05, .1], l1_floor=[.3, .4, .5], l1_margin=[0., .05, .1],
            l4_delay_s=[20.])


NEUTRAL = dict(context='10/5', grow=False, combine='centroid', floor=.5, margin=0., word_margin=0., l1_floor=.5,
               l1_margin=0.)


def change(p):
    """How far a setting is from the plain one (ties go to the smallest change: the data must ask for it)."""
    return sum(abs(p[key] - value) if isinstance(value, float) else p[key] != value for key, value in NEUTRAL.items())


def evaluate():
    clips = [clip for clip in CLIPS if os.path.exists(f'{OUT}/{clip}.json')]
    print('rule current (reference: same clip):')
    for clip in clips:
        rows = decide(clip, 'current', {})
        print(f"  {clip:7} L1 {errors(rows, 'l1')}  final {errors(rows, 'final')}")
    grid = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    cache = {}
    for i, p in enumerate(grid):
        for clip in clips:
            rows = decide(clip, 'v2', p)
            cache[i, clip] = errors(rows, 'final'), errors(rows, 'l1')
    print(f'v2, leave-one-clip-out over {len(grid)} settings (fewest final errors on the other clips; ties: fewest L1 errors):')
    total = dict(final=dict(person_to_other=0, other_to_person=0), l1=dict(person_to_other=0, other_to_person=0))
    for held in clips:
        train = [clip for clip in clips if clip != held]
        cost = lambda i: (sum(cache[i, c][0]['person_to_other'] + cache[i, c][0]['other_to_person'] for c in train),
                          sum(cache[i, c][1]['person_to_other'] + cache[i, c][1]['other_to_person'] for c in train),
                          change(grid[i]))
        best = min(range(len(grid)), key=cost)
        final, l1 = cache[best, held]
        for view, part in (('final', final), ('l1', l1)):
            for key in total[view]:
                total[view][key] += part[key]
        print(f'  test {held:7} params {grid[best]} train errors final/L1 {cost(best)}')
        print(f'          L1 {l1}  final {final}')
    print('  total held-out:', total)


if __name__ == '__main__':
    {'prepare': prepare, 'evaluate': evaluate}[sys.argv[1]]()
