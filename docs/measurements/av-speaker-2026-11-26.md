# Lips as a fallback for "who said this word", 2026-11-26 (Mac M2 Pro, CPU)

One 54 s clip (854x480, 30 fps, several people in a shop), the same as in `testbed-2026-11-26.md`. The clip was
also used to set the defaults below, so none of this is held-out accuracy. Other agents' jobs ran on the same Mac
at the same time; the times are rough.

## What was built

`person-term-count --with lips` (off by default) adds the camera of the same capture and three models, each in its
own environment:

| role | model | licence | where it runs |
|---|---|---|---|
| `vision.face` | MediaPipe Face Landmarker (existing pin) | Apache-2.0 | its points come back as a private payload (`mediapipe.face.video`, 9 points per face) |
| `vision.face_identity` | LVFace-B (existing pin) | MIT | the enrolled face, ArcFace alignment as in `TargetPerson` |
| `vision.active_speaker` | Light-ASD (CVPR 2023), `lightasd.ava` | MIT code and weights (in the repository at ed38c23) | new worker profile `active_speaker.lightasd`, CPU (MPS lacks an operator it uses) |

Candidates checked: LoCoNet has no licence file in its repository (all rights reserved: not usable); TalkNet is MIT but
its weights are on Google Drive without a stated licence (GAMST's existing TalkNet adapter stays as it is, not in the
registry); Light-ASD is MIT with both weight files in the repository, so they are pinned by commit and SHA256. Its
TalkSet weights are fine-tuned on LRS3 (non-commercial) and are an opt-in (`lightasd.talkset`, not installed by
setup, not measured). Only `lightasd.ava` was downloaded (4.2 MB).

Modules: `gamst/providers/active_speaker.py` (`vision.active_speaker` records: per face and 40 ms a speaking logit
or None, an identity cosine, a size in pixels, and how many slots each gate reason left unread; no position, point or
crop) and `gamst/providers/speaker_evidence.py` (the owner's rule over `speaker_check` words, feeding `terms`).

## The rule and the gate (defaults)

- Voice first. Band `[threshold - 0.20, threshold + 0.10)`, here `[0.15, 0.45)` around 0.35. Outside it the voice
  decides alone. The band was set on this clip: the person's missed words scored 0.18-0.27 and the two wrong ones
  0.42; it is a parameter (`low`, `high`).
- In the band: *for* = the enrolled face (LVFace cosine >= 0.35) has a mean speaking logit >= 0 (upstream's
  threshold) over the word +-100 ms and no other face does; *against* = another face (cosine < 0.15) does and the
  enrolled one does not; else *unknown*, and the voice decides. At least 3 readable 40 ms slots per face.
  A silent enrolled face is not evidence against (owner's rule).
- Gate per face and frame (otherwise "unknown", never "against"): face at least 64 px, nose tip within 0.6 eye
  distances of the eyes' middle along the eye line ("turned"), both mouth corners and inner lips in the picture,
  sharpness (variance of the Laplacian of the grey face at 112x112) at least 100. 64 px and 0.6 were set before
  looking at any speaking score; 100 after looking at the sharpness of blurred and steady faces in this clip
  (blurred 15-39, steady 255-603).
- The enrolled face came from two other recordings of the same person (10.6 s of "남아공전 반응" and 0-60 s of
  "홍명보 ..."), 320 embeddings, with the owner's consent statement. None of this clip's frames were used.

## Result: no word changed

`gamst recipe run` (turbo, speaker threshold 0.35), voice alone vs `--with lips`, truth = the owner's 10 words by end
time (0.6 s):

| word (s, ASR) | by | voice score | voice alone | fused | decided by | what the faces showed |
|---|---|---|---|---|---|---|
| 11.64-12.04 | person | 0.413 | count | count | voice | enrolled face turned (7 slots) |
| 12.52-12.64 | other | 0.421 | count (wrong) | count (wrong) | voice | one face, turned (identity not measured) |
| 14.0 | person | - | not transcribed | not transcribed | - | |
| 19.38-19.74 | person | 0.182 | no (miss) | no (miss) | voice | enrolled face: 2 slots read (logit -0.40), 9 turned |
| 19.74-20.14 | person | 0.242 | no (miss) | no (miss) | voice | enrolled face turned; camera on the shelf from 19.9 s |
| 20.14-20.34 | other | 0.197 | no | no | voice | no readable face |
| 20.34-20.74 | not in truth | 0.280 | no | no | voice | no face |
| 21.06-21.46 | person | 0.416 | count | count | voice | enrolled face 4 slots, logit -1.24 (model: silent) |
| 21.94-22.30 | person | 0.477 | count | count | voice | outside the band |
| 23.90-24.70 | nothing said | 0.418 | count (wrong) | count (wrong) | voice | enrolled face turned; others silent (-1.15, -0.04) |
| 27.06-27.44 | not in truth | 0.198 | no | no | voice | enrolled -0.35, other -0.85 |
| 27.98-28.10 | person | 0.271 | no (miss) | no (miss) | voice | enrolled face 9 slots, logit -0.05 (just below 0) |
| 37.66-38.20 | person | 0.573 | count | count | voice | outside the band (enrolled +0.61) |
| 45.92-46.64 | person | 0.551 | count | count | voice | outside the band |
| 47.28-47.74 | person | 0.506 | count | count | voice | outside the band |

| | counted | right | precision | recall | fixes | breaks |
|---|---|---|---|---|---|---|
| voice alone | 8 | 6 | 0.75 | 0.6 | | |
| voice + lips (defaults) | 8 | 6 | 0.75 | 0.6 | 0 | 0 |
| voice + lips, band [0.10, 0.50) or [0.05, 0.60) (replayed from the same records) | 8 | 6 | 0.75 | 0.6 | 0 | 0 |
| a silent enrolled face as "against" (not the owner's rule; replayed) | 7 | 5 | 0.71 | 0.5 | 0 | 1 (21.46) |

Why nothing changed: where the voice was unsure, the enrolled face was turned or not in the picture (11.6, 12.6,
19.4-20.7, 24.7), and where it could be read the model mostly said "not speaking" even when the person spoke. Over the
person's 10 words, the enrolled face was readable for at least 3 slots in 6; its mean logit was above 0 in 2 of
them (22.3: +0.12, 38.2: +0.61) and below in 4 (21.5: -1.24, 28.1: -0.07, 46.6: -0.02, 47.7: -0.54). The other
person from 49 s was not seen speaking either (-0.74, -2.09). Light-ASD with AVA weights does not read this footage
well enough to be evidence; on this clip the part neither helps nor harms, so it stays off by default.

## The same model, run again: the few changes sit on the threshold

The module was also run outside the recipe on the same clip (frames read with OpenCV, sound from a 16 kHz WAV,
records starting at another place), with the camera shifted by -160, -80, 0 and +80 ms against the sound, and the
recipe run's words fused with those records (replayed, same rule and gate):

| lips records from | counted | right | fixes | breaks | changed words (enrolled face's mean logit) |
|---|---|---|---|---|---|
| recipe run (above) | 8 | 6 | 0 | 0 | none (28.1: -0.05) |
| outside, +0 ms | 8 | 7 | 2 | 0 | 28.1 counted (+0.09); 24.7 excluded: another face +0.08 |
| outside, +80 ms | 8 | 7 | 2 | 0 | 28.1 (+0.12); 24.7: another face +0.33 |
| outside, -80 ms | 9 | 7 | 1 | 0 | 28.1 (+0.07) |
| outside, -160 ms | 9 | 7 | 1 | 0 | 28.1 (+0.01) |

Every change rests on a mean logit between 0.01 and 0.33, and the same word flips between runs that differ only in
decoding and window placement (28.1: -0.05 in the recipe, +0.01..+0.12 outside). 24.7 was excluded because another
face moved its lips while the owner says nothing was said there, so that "fix" may be right for a wrong reason. Asking
for a clearer logit removes them all: with `speaking_logit` 0.5 no word changes in any of the five; with 0.3 one
(24.7 at +80 ms). Shifting the camera did not make the person's own words read as speaking (readable person words
with a logit >= 0: 3 of 5 at -160 and -80 ms, 4 of 6 at 0 and +80 ms).

Conclusion for this clip: the lips part did not break a word in any run, and the fixes it made are not stable. It
stays off by default; if it is used, a `speaking_logit` well above 0 is the safer setting (not measured on other
clips).

## Speed

| run | wall time for the 54 s clip |
|---|---|
| voice alone (turbo) | 134.8 s |
| `--with lips` | 221.2 s |
| the lips module alone (faces, identity, Light-ASD; outside the recipe, 5 runs) | 67-101 s |

Per call (median over a run): face points 13-20 ms per frame, LVFace 68-94 ms per identity check (every 250 ms per
face), Light-ASD 0.47-0.81 s per face and 1 s record (six window lengths averaged, as upstream). Not realtime on this CPU with several
faces.

## Not done / open

- Light-ASD's temporal smoothing of face boxes (upstream: 13-frame median) and the XVID round trip are not reproduced
  (same as the TalkNet adapter).
- The TalkSet weights (better on in-the-wild video by upstream's Columbia numbers) are not measured: non-commercial,
  and the owner asked for one model.
- `gamst recipe enroll person-term-count --kind person` does not exist (no pose module there); enroll with
  `gamst recipe enroll person-action-count --kind person` and pass the file as `--person`.
