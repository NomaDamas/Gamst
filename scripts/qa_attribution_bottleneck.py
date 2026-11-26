"""Where does attribution fail? Owner-labelled words, CAM++ embeddings, three references:
 A = another recording (current), B = same clip, leave-one-out over the person's other labelled words (upper bound of a
 same-recording few-shot), C = same clip but only 2 s of the person (a realistic owner-marked sample, nearest other word)."""
import csv, json, subprocess, wave, numpy as np, sherpa_onnx, sys
M = '/Users/jinminseong/.cache/gamst/models/sherpa.campplus_zh_en/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'
E = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=M, num_threads=2))
LAB = '/Users/jinminseong/Desktop/홍명보/gamst/GAMST 라벨링'; SRC = '/Users/jinminseong/Desktop/홍명보/gamst/GAMST TestBed'
T = '/Users/jinminseong/Desktop/catch-your-mind/.local-run/try-it/testbed'
CL = {'bbo': '감스트 뽀또', 'marine': '감스트의 해병대식 내리갈굼', 'fifa': '감스트 피파 억까', 'indi': '애교쟁이 인디깅'}
def audio(name):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f'{SRC}/{name}.mp4', '-ac', '1', '-ar', '16000', '-f', 's16le', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768
def emb(x):
    s = E.create_stream(); s.accept_waveform(16000, x); s.input_finished(); v = np.array(E.compute(s)); return v / np.linalg.norm(v)
def load(p):
    w = wave.open(p); return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
e2 = load(f'{T}/e2s.wav'); A = np.mean([emb(e2[i:i + 48000]) for i in range(0, len(e2) - 48000, 48000)], 0); A /= np.linalg.norm(A)
def auc(pos, neg):
    if not pos or not neg: return None
    return np.mean([[1. if p > n else .5 if p == n else 0. for n in neg] for p in pos])
for k, name in CL.items():
    lab = json.load(open(f'{LAB}/{name} - 정답.json', encoding='utf-8'))['labels']
    rows = [r for r in csv.DictReader(open(f'{LAB}/{name} - 정답표.csv', encoding='utf-8-sig')) if r['번호'].isdigit()]
    x = audio(name); items = []
    for r in rows:
        who = lab.get(r['번호'], {}).get('who')
        if who not in ('감스트', '다른 사람'): continue
        s0, s1 = float(r['시작(초)']), float(r['끝(초)']); c = (s0 + s1) / 2
        items.append(dict(t=s1, who=who, w06=emb(x[int((c - .3) * 16000):int((c + .3) * 16000)]),
                          w15=emb(x[int((c - .75) * 16000):int((c + .75) * 16000)])))
    P = [i for i in items if i['who'] == '감스트']; N = [i for i in items if i['who'] == '다른 사람']
    print(f'== {name}: person words {len(P)}, other-voice words {len(N)}')
    for win in ('w06', 'w15'):
        a_p = [float(i[win] @ A) for i in P]; a_n = [float(i[win] @ A) for i in N]
        b = []
        for i in items:
            others = [j[win] for j in P if j is not i and abs(j['t'] - i['t']) > 1.0]
            if not others: b.append(None); continue
            ref = np.mean(others, 0); ref /= np.linalg.norm(ref); b.append(float(i[win] @ ref))
        b_p = [s for s, i in zip(b, items) if i['who'] == '감스트' and s is not None]; b_n = [s for s, i in zip(b, items) if i['who'] == '다른 사람' and s is not None]
        f = lambda v: f'{min(v):.2f}-{max(v):.2f} (median {np.median(v):.2f})' if v else '-'
        print(f'  window {win[1:]}: A other recording  person {f(a_p)}  other {f(a_n)}  share of person >= 0.5: {np.mean([s >= .5 for s in a_p]):.0%}  AUC {auc(a_p, a_n)}')
        print(f'               B same clip (oracle) person {f(b_p)}  other {f(b_n)}  share of person >= 0.5: {np.mean([s >= .5 for s in b_p]) if b_p else 0:.0%}  AUC {auc(b_p, b_n)}')
