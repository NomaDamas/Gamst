# Ground truth v2 (2026-11-28): what changed from v1 and why

v1 (`../gwan/`, `../man/`, commit d71a5ea) stays as it was. v2 is a complete truth folder per clip, so a scorer can be
pointed at `v2/<clip>/` instead of `<clip>/`. Only `man/truth_identity.json` differs from v1; the other five files are
byte-for-byte copies.

## 요약 (한국어)

| # | 무엇 | 어디 | 왜 (출처) |
|---|---|---|---|
| 1 | 같은 순간 감스트 상자가 둘 이상이면 **둘 다 정답**, 채점은 "그중 하나" | 데이터 변경 없음, 채점기 규칙 (`gamst/eval/multi_person.py`) | 사장님 결정 1 |
| 2 | 벽의 **감스트 액자 사진** 상자 6개 구간을 `photo: true` → 채점에서 뺌 | 맨까송 person-4, -5, -6, -8 통째로; person-2 8.49–10.944초, person-12 20.521–20.57초 | 사장님 결정 2, 프레임은 Claude가 확인 |
| 3 | person-27 (42.04–63.53초, "다른 사람")을 세 조각: 42.04–44.24 다른 사람, 44.24–50.48 다른 사람, **50.48–63.53 감스트** | 맨까송 | 사장님이 Claude에게 맡김 (결정 3). `source: "claude (owner-delegated 2026-11-28)"` |

- 잘린 원래 구간 3개(person-2~mwzau1al, person-12, person-27)는 `superseded` 목록에 그대로 남겼습니다. 채점에는 쓰지 않습니다.
- 동작 정답(`truth_action.json`)과 예시(`examples.json`)는 바뀌지 않았습니다.

## Changes

### 1. Two 감스트 boxes at the same moment: both are truth, scored as "one of" (owner decision 1)

No data change. Where the owner labelled two different boxes 감스트 at the same moment (관제탑 331-420 s; 맨까송 around
9.6 s and 20.6-21 s), both labels stand. The scorer (`gamst.eval.multi_person`, PR #379) now scores such a frame once:

- **Recall:** the frame is found if GAMST names 감스트 on any of those boxes (or, for the owner's added stretch without a
  box, on any track).
- **Precision:** naming 감스트 on any of them is correct; the frame counts as one true positive.
- **Actions:** in a moment with several 감스트 boxes, 감스트 events of different tracks that overlap in time are one
  event: it is matched if any of them overlaps the occurrence.

After the photo exclusion (change 2), 맨까송's two-box moments at 9.6 s and 20.52-20.55 s are no longer two live boxes:
one of the two was the photo.

### 2. The framed photo of 감스트 on the wall (owner decision 2)

The owner labelled the framed photo 감스트 on purpose and said scoring may leave photos out. Claude looked at every
labelled 감스트 box of 맨까송 (crops of each segment every 0.2-0.5 s, and every stored box around each change). The boxes
on the photo are marked `photo: true` (with `photo_by` and `photo_note`) and are left out of identity scoring: neither
a hit nor an error.

| segment | time (s) | what the box shows | v2 |
|---|---|---|---|
| person-4 | 2.002-3.203 | 2.00-2.13 another person (black suit), 3.00-3.20 the photo | whole segment `photo` |
| person-5 | 2.002-5.072 | 2.00-2.97 other people, 3.24-5.07 the photo (most of it) | whole segment `photo` |
| person-6 | 2.502-5.606 | 2.50-3.30 other people, 3.47-4.91 the photo (most), 5.14-5.61 감스트's back | whole segment `photo` |
| person-8 | 9.009-11.011 | the photo throughout | whole segment `photo` |
| person-2~mwzau1al | 0-10.944 | 0-8.47 감스트 himself (back, orange shirt), 8.51-10.94 the photo | cut at 8.49: `person-2~v2a` 0-8.49 G, `person-2~v2b` 8.49-10.944 G `photo` |
| person-12 | 20.521-26.96 | 20.52-20.55 the photo right after the cut, 20.59-20.82 a man in a suit with the photo behind, 20.95- 감스트 | cut at 20.57: `person-12~v2a` 20.521-20.57 G `photo`, `person-12~v2b` 20.57-26.96 G |

- A segment is marked whole when most of its boxes are the photo: the owner's G answer was about the photo, so the few
  boxes on someone else at its edges are left out with it rather than given a label the owner did not give.
- Where most of a segment is 감스트 himself, it was cut at the stored box where the photo starts, and the live part keeps
  the owner's G.
- Not changed: `person-7~mwzawq94` (16.216-22.623, G). At 21.75-22.42 its box drifts onto the people on the left with
  the photo above them. It is a person box, not the photo, so it stays as the owner labelled it.
- No photo box was found in the other G segments of 맨까송 (person-7 pieces, -10, -16, -20, -21, -22~mwzaztxl, -26).
  관제탑 was not checked for photos; the owner's decision named 맨까송.

### 3. 맨까송 track person-27 (42.042-63.53 s, v1 label "O") (owner decision 3, delegated to Claude)

The owner delegated this segment to Claude. Claude looked at frames with its box and set the labels below, then
checked them again on every stored box around both cuts and on crops every 0.5 s before writing anything.

| piece | time (s) | label | what the box shows (checked) |
|---|---|---|---|
| `person-27~v2a` | 42.042-44.24 | O | the man in a black suit, bow tie and glasses (boxes 42.04-44.21) |
| `person-27~v2b` | 44.24-50.48 | O | 44.24-45.01 the same man; from 45.30 inserted stream footage (boxes 45.41-50.45): a man in a red Man Utd shirt with hair and no glasses. 50.48 is a scene cut |
| `person-27~v2c` | 50.48-63.53 | G | 감스트: orange Man Utd shirt, shaved head, thin glasses (50.48 bent over, 51-62 face visible, 63.0 fading to black) |

- `source` of the three pieces: `claude (owner-delegated 2026-11-28)`. `proposed` keeps Claude's first proposal (G).
- The original segment is kept in `superseded` (`superseded_by` names the pieces).
- The second check refined one description: 44.20 s and 44.27 s are the same shot (no visible cut at 44.24 s), and the
  box stays on the suited man until 45.01 s; the inserted footage is on screen from 45.30 s. Both pieces are O, so the
  split time between them changes no label; it was kept as decided.
- The owner's added stretch (50.55-63.263 s, G, no box) now overlaps `person-27~v2c`. Both are kept. The one-of rule
  scores each of those frames once.

## Counts

| 맨까송 identity | v1 | v2 |
|---|---|---|
| segments scored: G / O / S | 14 / 17 / 0 | 11 / 18 / 0 |
| G segments marked `photo` (not scored) | 0 | 6 (4 whole, 2 cut pieces) |
| superseded (kept, not scored) | 0 | 3 |
| added stretches (G, no box) | 1 | 1 |
| source owner / claude (owner-delegated) | 31 / 0 | 32 / 3 |

관제탑: unchanged (100 segments: 48 G, 43 O, 9 S). Actions: unchanged (관제탑 46 감스트 + 22 other, 맨까송 16 감스트).

## Fields added in v2 (`v2/man/truth_identity.json`)

- `version: 2`, `changes`, `superseded` (list of the cut segments, each with `superseded: true`, `superseded_by`).
- Per segment: `photo` (bool), `photo_by`, `photo_note`; on cut pieces `replaces` (the v1 segment) and `split_of`
  (the GAMST segment); `changed` on every segment that differs from v1.
- A scorer that does not know `photo` counts those segments as v1 did (G). `gamst.eval.multi_person` (PR #379) leaves
  them out, and also anything with `superseded: true`.
