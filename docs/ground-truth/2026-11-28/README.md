# Owner ground truth: two multi-person clips (labelled 2026-11-27 – 28)

The owner labelled two clips by hand: who is 감스트, and when 감스트 (or someone else) performed the action. These labels are the reference for every GAMST measurement of "a given person's given action" on these clips. The clips are not in the repository, because they are broadcast footage under copyright. File paths below use `<TestBed>` for the owner's local TestBed folder and `<try-it>` for the local run folder.

## 요약 (한국어)

| 영상 | 동작 정의 (사장님) | 감스트 동작 | 다른 사람 동작 | 인물 구간 (감스트 / 다른 사람 / 쓰지 않음) |
|---|---|---|---|---|
| 감스트-관제탑 모음집 (522초, 720p 30fps) | **관제탑:** 두 팔을 오므렸다가 → 양옆으로 폈다가 → 다시 접는 것 = 1회. 연속으로 하면 1회씩 따로 셈 | **46** | 22 | 48 / 43 / 9 구간 |
| 감스트 맨까송 – 맨가대 (67초, 360p 30fps) | **펀치:** 감스트가 펀치를 한 사실 = 1회 | **16** | 0 | 14 / 17 / 0 구간, 직접 추가 1 |

- **라벨링 방법:**
  1. Claude가 프레임을 보고 먼저 제안했습니다(점선 "제안").
  2. 사장님이 영상을 보며 하나씩 승인하거나 고쳤습니다. 키는 Enter / 1 / 2 / Backspace / X(나누기) / `[ ]`(시작·끝)입니다.
  3. 놓친 것은 사장님이 직접 추가했습니다.
  
  **이 파일의 라벨은 모두 사장님 승인(`source: owner`)입니다.** Claude 제안만 있고 승인되지 않은 것은 들어 있지 않습니다.
- **주의:** 관제탑의 정의는 "두 팔을 위로 쭉 뻗기"가 아니라 **"오므림 → 양옆으로 폄 → 접음"** 입니다. 처음 GAMST 후보를 만든 규칙(`rule_gwan*.json`, 팔을 위로)과 다릅니다. 이후 측정은 이 정의를 따릅니다.

## Files (per clip: `gwan/` = 관제탑, `man/` = 맨까송)

- `truth_action.json`
  - `occurrences`: each has `start`, `end` (seconds from the start of the video) and `who` (`감스트` or `other`). It also carries `source` (always `owner` here), `origin` (which candidate or missed item it came from) and `reason`.
  - Each occurrence is one action (one 관제탑 cycle or one punch). Repetitions are separate.
  - `counts`: per person.
  - `candidates`: the GAMST candidates the owner reviewed, from the old run `cand_loose` / `cand_first`, kept for traceability.
- `truth_identity.json`
  - `segments`: one per GAMST track segment the owner answered. Fields:
    - `label`: `G` = 감스트, `O` = other person, `S` = not used (not a person, or a duplicate box);
    - `start`, `end`;
    - `boxes`: per-frame boxes;
    - `split_of`: set when the owner cut a mixed segment;
    - `person`, `person_name`.
  - `added`: stretches the owner added where 감스트 was visible but no box existed. They have no box.
  - Identity is scored frame by frame by box overlap (`score_multi.py`).
  - Segments may overlap in time: two boxes at once. So the summed seconds per label can exceed the clip length (관제탑 G: 678.8 s on a 522 s clip).
- `examples.json`: example clips the owner trimmed for few-shot. 관제탑 has 7 (62.05–63.95 s, 64.13–64.76, 64.96–65.56, 65.83–66.53, 71.41–72.21, 72.38–73.11, 75.87–76.77). 맨까송 has none. It also stores the `gamst recipe enroll` command that uses them.

## Statistics

| | 관제탑 | 맨까송 |
|---|---|---|
| 감스트 action duration: median / min / max | 0.72 / 0.33 / 2.0 s | 0.17 / 0.10 / 1.07 s |
| Identity segments answered | 100 (48 G, 43 O, 9 S) | 31 (14 G, 17 O) + 1 added |
| Frames per second | 30 | 29.97 |

## How it was made

- **Tool:** a local labelling tool (`<try-it>/label_video/`, not in the repository) served on 127.0.0.1. It has one task per screen ("① 감스트 동작 정답", "② 인물 정답"), a zoomable timeline, frame stepping, block split (X), edge resizing (`[ ]` and drag), undo, and a stale-page guard. Every save is an owner edit.
- **Claude proposals first:** Claude looked at extracted frames: crops of each track and frame sequences over each candidate. It proposed labels with a one-line reason and marked each one as a proposal. The owner approved or changed every label used here.
- **Limits:**
  - The 관제탑 truth was reviewed against the GAMST candidates of the old run, plus the owner's own added items. An action that neither GAMST, Claude nor the owner noticed is not in it.
  - 맨까송 is 360p. In some punches only the back is visible (the reason text says so).
- **Snapshot:** taken at 2026-11-28 03:21 (`<try-it>/testbed/ground_truth_owner_20261128-032103/`).

## Use

- `score_multi.py SHORT RUN` (label tool) compares a GAMST run with these files. It gives identity precision and recall per frame, and per person: action count, count error, precision, recall and start/end error.
- **Rule for every measurement on these clips:**
  - Thresholds or rules chosen while looking at these labels are in-sample, and the report says so.
  - Few-shot examples come from `examples.json`. Occurrences inside example ranges are excluded from scoring (lesson 3).
