"""Per-repetition times of a repeated sound word ("달달달달달"): even split of the recogniser word vs Qwen3-ForcedAligner.

    <qwen venv python> scripts/qa_repetition_times.py CLIP.mp4 MODEL_DIR START-END:N [START-END:N ...]

Each span is one recogniser word (seconds) with N repetitions. Reference: the syllable nuclei of the sound itself
(10 ms RMS envelope, the N highest peaks at least 80 ms apart inside the word; a proxy, not the owner's ear). Compared:
(a) the word's time split evenly (what term_events does), (b) Qwen3-ForcedAligner on the word +- 0.4 s with the text
written as N separate syllables, (c) the same with the text as one word. Error: |centre - nucleus| per repetition.
No transcript is involved (the text is the counted word only).
"""
import subprocess
import sys

import numpy as np

RATE = 16000


def audio(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(RATE), '-f', 's16le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def nuclei(x, start, end, n, hop=.01, gap=.08):
    seg = x[int(start * RATE):int(end * RATE)]
    frames = [float(np.sqrt(np.mean(seg[i:i + int(.025 * RATE)] ** 2) + 1e-12))
              for i in range(0, max(1, len(seg) - int(.025 * RATE)), int(hop * RATE))]
    env = np.convolve(frames, np.ones(3) / 3, mode='same')
    peaks = [i for i in range(1, len(env) - 1) if env[i] >= env[i - 1] and env[i] >= env[i + 1]]
    chosen = []
    for i in sorted(peaks, key=lambda i: -env[i]):
        if all(abs(i - j) * hop >= gap for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    return sorted(start + (i * hop) + .0125 for i in chosen)


def main(argv):
    clip, model_dir, spans = argv[0], argv[1], argv[2:]
    import torch
    from qwen_asr import Qwen3ForcedAligner
    model = Qwen3ForcedAligner.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='mps')
    x = audio(clip)
    errors = dict(even=[], aligned_syllables=[], aligned_word=[])
    for span in spans:
        times, n = span.split(':')
        start, end = map(float, times.split('-'))
        n = int(n)
        truth = nuclei(x, start, end, n)
        even = [start + (end - start) * (i + .5) / n for i in range(n)]
        low = max(0., start - .4)
        piece = x[int(low * RATE):int((end + .4) * RATE)]
        found = {}
        for name, text in (('aligned_syllables', ' '.join(['달'] * n)), ('aligned_word', '달' * n)):
            result = model.align(audio=(piece, RATE), text=text, language='Korean')
            result = result[0] if isinstance(result, (list, tuple)) else result
            items = getattr(result, 'items', None) or getattr(result, 'words', None) or result
            parts = []
            for item in items:      # a word of k syllables: its time split evenly into k
                text_ = getattr(item, 'text', None) or item['text']
                a = float(getattr(item, 'start_time', None) if not isinstance(item, dict) else item['start_time'])
                b = float(getattr(item, 'end_time', None) if not isinstance(item, dict) else item['end_time'])
                k = max(1, len(str(text_).strip()))
                parts += [low + a + (b - a) * (j + .5) / k for j in range(k)]
            found[name] = parts
        print(f'span {start:.2f}-{end:.2f} x{n}: nuclei {[round(t, 2) for t in truth]}')
        if len(truth) != n:
            print('  fewer clear peaks than repetitions: span skipped')
            continue
        for name, centres in (('even', even), *found.items()):
            if len(centres) != n:
                print(f'  {name}: {len(centres)} items, not {n}: counted as missing')
                continue
            err = [abs(a - b) for a, b in zip(sorted(centres), truth)]
            errors[name] += err
            print(f'  {name}: {[round(t, 2) for t in sorted(centres)]} mean |error| {np.mean(err) * 1000:.0f} ms')
    for name, err in errors.items():
        if err:
            print(f'{name}: {len(err)} repetitions, mean |error| {np.mean(err) * 1000:.0f} ms, '
                  f'median {np.median(err) * 1000:.0f} ms, max {np.max(err) * 1000:.0f} ms')


if __name__ == '__main__':
    main(sys.argv[1:])
