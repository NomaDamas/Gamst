# Wrap-up: a given person's action, found from text and photos (2026-11-27 – 28)

Everything merged into Main on 2026-11-27 and 2026-11-28: PRs #373 (one pose model, every frame, in real time), #375
(Sapiens2 1B opt-in), #374 (person identity), #378 (the owner's ground truth), #376 (`compose_recipe`), #379 (an action
given as text, clips or both), #380 (the HUD's setup step and live view) and #381 (identity recall). Every number below
is copied from the measurement documents linked in each row; nothing was measured again for this summary. Thresholds
chosen while looking at the owner's labels are **in-sample**; "held-out" says on which half they were picked.

## 요약 (한국어)

- **GAMST가 이제 하는 일:** "특정인의 특정 행동"을 글과 사진으로 지정하면, 여러 사람이 나오는 영상에서 그 사람이 그 행동을 한
  **횟수와 매번의 시작·끝 시각**을 돌려줍니다. 다른 사람의 같은 행동도 따로 셉니다(사람마다 한 줄).
  - 행동: 글(규칙), 예시 영상(few-shot), 또는 둘 다(사장님 결정 58). 세기(count)는 규칙과 예시가 **둘 다** 맞을 때, 찾기(search)는
    **둘 중 하나**만 맞아도 셉니다.
  - 사람: 이름·설명 **글과 사진이 둘 다** 있어야 합니다(결정 59). 얼굴(LVFace-B, 옵션 SCRFD 검출기) + 전신 외형(PersonViT)으로
    트랙마다 누구인지 붙이고, 나중에 소급합니다.
  - 지정 방법: HUD 설정 화면(`gamst recipe hud`), 명령줄, 또는 MCP `compose_recipe`(에이전트가 요청 하나로 구성·실행).
- **실시간:** 가장 정확한 모델 하나(RTMW-l)가 모든 프레임을 셉니다(결정 37). 샷건 영상(16회)을 라이브로 60 fps 4회·30 fps 6회
  실행(모델 캐시 뒤) 모두 16/16, 건너뛴 프레임 0, 잃은 프레임 0, 모든 프레임을 라이브 레인이 처리. 프레임당 14.0 ms(실시간 계수
  0.93), 지연 p50 190–216 ms(60 fps)·24–26 ms(30 fps).
- **사장님 정답 (2026-11-27–28, #378):** 관제탑(522초) = 두 팔을 오므렸다가 → 양옆으로 폈다가 → 다시 접기 1회, **감스트 46회 ·
  다른 사람 22회**. 맨까송(67초, 360p) = 펀치 **16회**(모두 감스트). 정답 v2: 같은 순간 감스트 상자가 둘이면 "그중 하나", 벽의
  감스트 액자 사진은 채점 제외, person-27은 50.48초부터 감스트.
- **정답 v2 결과 (감스트, 예시 7개 구간 제외: 관제탑 정답 39회):**

  | | 정밀 / 재현 (인물, 프레임) | 동작 F1 in-sample | 동작 F1 held-out (홀→짝 / 짝→홀) |
  |---|---|---|---|
  | 관제탑, 규칙+예시 찾기(search) | 99.8 % / 94.3 % (#381 설정) | **0.889** (#379 인물), **0.860** (#381 인물) | 0.778 / 0.895 (#379 인물) |
  | 관제탑, 규칙+예시 세기(count) | 〃 | 0.829 (#379), **0.861** (#381) | 0.710 / 0.895 (#379) |
  | 맨까송, 펀치 규칙 | 97.8 % / 67.9 % (#381 설정) | 0.462 (#379), **0.600** (#381) | 0.571 / 0.000 (#379) |

  - #381 인물 설정의 held-out(홀수 장면에서 고르고 짝수 장면에서 잼): 관제탑 99.6 % / 93.9 %, 맨까송 97.9 % / 72.6 %.
  - held-out 동작 F1은 #379의 기록 실행(그때의 인물 설정)에서 홀수/짝수 장면마다 설정을 고른 값입니다. #381 인물로는 고른 설정
    하나의 장면별 점수만 있습니다(찾기 0.889 / 0.829, 세기 0.923 / 0.788, 펀치 0.571 / 0.500). held-out이 아닙니다.
- **남은 오류:** ① 맨까송 펀치 규칙(장면마다 달라 held-out 0.571 / 0.000), ② 저해상도 옆얼굴(맨까송 재현 67.9 %), ③ held-out
  흔들림(반쪽마다 정답 7–21회라 1–2회가 F1을 크게 움직임), ④ calibration(예시로 규칙 숫자 맞추기)은 기본 꺼짐: 한 장면 예시로는
  count F1 0.829 → 0.710으로 나빠졌습니다.
- **실행:** 아래 "How to run" (HUD 한 줄 명령, MCP `compose_recipe` 요청 예).
- **영상 3개(로컬):** 관제탑 글로 찾기, 샷건 실시간 세기, 뽀또 음성 세기. 사장님 결정 67(2026-11-28)로 이 세 개는 GitHub 이슈에
  올려 README에 넣습니다(결정 35의 예외). 올리기는 사장님이 영상을 본 뒤입니다.

## What GAMST does now

| capability | how it is specified | what comes back | measured (owner truth v2 unless said) |
|---|---|---|---|
| A movement, counted on every frame by one model, live | text → rule (`docs/movements/*.rule.json`), example clips, or both; `--timing realtime` | each occurrence with start / end and evidence (rule steps, nearest example); L2 corrections marked `LATE` | shotgun clip (16 slams): live runs after the model cache, 4 at 60 fps and 6 at 30 fps: 16/16, 0 skipped, 0 lost, latency p50 190-216 ms (60 fps), 14.0 ms per frame; forced backlog 16/16 in 5 of 5 with L2 corrections, no double ([testbed](measurements/testbed-2026-11-26.md) "One model, every frame") |
| The same on a finished file | `--timing post`: every frame, nothing late; waits for each frame's face and body scores | the same | 관제탑 522 s: 15,670 of 15,670 frames, 0 lost ([testbed](measurements/testbed-2026-11-26.md) "Post mode lost 608 frames") |
| A given person (who) | a name or description **and** photos (decision 59), or a range of a video, or camera seconds; checked (faces agree, one person per photo) | `who` on every event, resolved retroactively; one row per person, others as `P2`, `P3`, ... | identity frames, chosen setting (#381): 관제탑 P 99.8 % / R 94.3 % (held-out even scenes 99.6 / 93.9); 맨까송 97.8 / 67.9 (held-out 97.9 / 72.6) ([identity recall](measurements/identity-recall-2026-11-28.md) §7) |
| A given person's given action, text only | rule written from the owner's words | as above | 관제탑 rule as written: 20 of 21 events right, F1 0.667; tuned in-sample 0.861, held-out 0.727 / 0.895 ([action spec](measurements/action-spec-2026-11-28.md)) |
| ... clips only (few-shot) | 7 owner clips | as above | in-sample 0.850, held-out 0.638 / 0.872; the threshold derived from the 7 clips of one scene finds 1 of 39 |
| ... text + clips, count mode | rule AND an example | as above | in-sample 0.829 (#379 identity), 0.861 (#381); held-out 0.710 / 0.895; precision 93.5 % |
| ... text + clips, search mode | rule OR an example | as above | in-sample **0.889** (#379), 0.860 (#381); held-out 0.778 / 0.895; recall 92.3 % (36 of 39), 39 of 39 in the #381 run that did not wait for scores |
| A punch (맨까송, 360p) | text → rule (`punch.rule.json`: fist by `hand_open` on the raw-score hand gate, fast wrist) | as above | in-sample 0.462 (#379), 0.600 (#381); held-out 0.571 / 0.000 |
| Composed by an agent | MCP `compose_recipe` / `gamst recipe compose`: `{inputs, who, what, mode, timing}` | a validated recipe with a reason per choice, or one sentence per missing piece; `start` runs it | scripted MCP demo: shotgun 16, all 감스트, end times within 0.02 s of the owner's ([compose-recipe](compose-recipe.md)) |
| Set up and watched in a window | `gamst recipe hud`: input (kind detected), who (name + photos, checked at once), what (text, rule, clips trimmed in the window), count / search, realtime / post | the live view: boxes per person, count per person, events, a clickable timeline with each occurrence's evidence | 관제탑 first 90 s: 9 occurrences, all 감스트, 9 of the owner's 14 found; `b.mp4` as a camera stand-in: 16, all 감스트, no frame lost ([hud-demo](hud-demo.md) §0) |
| Words of one person | unchanged (decision 48) | — | 뽀또 v8: 7 of 9 counted right ([owner-labels](measurements/owner-labels-2026-11-26.md)) |

## Experiment timeline (2026-11-27 – 28)

| PR / day | what changed | result | lesson |
|---|---|---|---|
| #373, 11-27 | One pose model (RTMW-l) on every frame; batches of 8, one CoreML session per fixed batch size, warm-up at load, shared memory | 23 → 14 ms per crop; first live run 16/16, latency p50 213 ms, backlog 0.4 s instead of 3.8 s | 39, 40 |
| #373 | L2 correction by the same model over backlog intervals; owned ranges, a safety net between any two lanes | forced backlog: 17 for 16 before the fix, 16/16 in 5 of 5 after | 44, 45 |
| #373 | CoreML compile cache instead of `$TMPDIR` copies | 218 folders / 16 GB of temp copies gone; a second load 0.19 s instead of 4.6 s | 38 |
| #375, 11-27 | Sapiens2 1B as an opt-in (licence stated, never automatic) | 16/16, the same as RTMW-l, at 5.5-6.5 s per frame (76-217 times the cost) | 97 |
| #373, 11-27 evening | A black frame (YOLOX NMS on CoreML refused zero boxes) | the run lost 108 frames at the 맨까송 fade; fixed with static input shapes: same boxes, 27.1 ms | 59 |
| #373 | Post mode reads at the pace of processing (no video memory, no L2) | 608 frames lost before; 15,670 of 15,670 after | 60 |
| #373, 11-28 | RTMW's raw keypoint score on the wire; hands gated at raw 3.0 | `hand_open < 1.0` is a fist in 7 of 10 sampled crops instead of 7 of 20 | 61, 77, 78 |
| #374, 11-27 – 28 | Person identity: LVFace-B face + PersonViT body ("무조건 SOTA"), one person one track, everyone a person, scene cuts, track splits by face | two tracks on 감스트 at once 13 → 0 frames; bindings over a scene cut 5 → 0 | 46-58, 68-76 |
| #376, 11-27 | `compose_recipe` (MCP) and `gamst recipe compose`; the semantic lane as a causal state chain (off by default) | the composed modules equal the hand-assembled ones (test) | 62-67 |
| #378, 11-28 | The owner's labels: 관제탑 46 감스트 + 22 other, 맨까송 16 punches; v2 (one of, photos, person-27) | the reference for everything below | 94 |
| #379, 11-28 | An action as text, clips or both; count = both, search = either; the rule written from the owner's words (fold → spread → fold, not "arms up") | 관제탑 search in-sample 0.889 (v2), held-out 0.778 / 0.895; calibration made count worse (0.829 → 0.710) | 80-85 |
| #380, 11-28 | The HUD's setup step and live view; no consent question (installing is the consent), the input kind detected | 관제탑 90 s: 9 of 14 found, all 감스트; shotgun: 16, all 감스트 | 95, 98-102 |
| #381, 11-28 | Identity recall: causes counted first; SCRFD detector, `reject_after=2`, `reach_back`, thresholds 0.15 / 0.10, post waits for scores | 관제탑 recall 85.6 → 94.3 % (v2), 맨까송 62.6 → 67.9 %; punch F1 0.462 → 0.600 | 86-93, 96 |

## The owner's ground truth

[docs/ground-truth/2026-11-28/](ground-truth/2026-11-28/README.md), labelled by the owner (every label `source: owner`;
v2 adds three Claude labels delegated by the owner, kept with their source).

| clip | the action (owner) | by 감스트 | by others | identity segments |
|---|---|---|---|---|
| 감스트-관제탑 모음집 (522 s, 720p, 30 fps) | **관제탑**: arms folded in front → spread out to both sides → folded again = 1; each repetition counts | **46** | **22** | 100 (48 G, 43 O, 9 not used) |
| 감스트 맨까송 (67 s, 360p) | **punch**: 감스트 punched = 1 (here mostly a fist brought down onto the chair) | **16** | 0 | v2: 11 G, 18 O, 6 photo segments left out, 1 added stretch |

- 7 example clips for 관제탑 (62.05-76.77 s); occurrences inside them are not scored, so 감스트's truth is 39 and
  everyone's 61. 맨까송 has no example clips.
- [v2 CHANGELOG](ground-truth/2026-11-28/v2/CHANGELOG.md): two 감스트 boxes at one moment are both right and scored once
  ("one of"); the framed photo of 감스트 on the wall is left out; 맨까송 person-27 is 감스트 from 50.48 s. The action
  truth is the same file in v1 and v2.

## Results on truth v2

### Identity (frames of 감스트, real post runs, [identity-recall §7](measurements/identity-recall-2026-11-28.md))

| run | 관제탑 all | 관제탑 even scenes (held out) | 맨까송 all | 맨까송 even scenes (held out) |
|---|---|---|---|---|
| before #381 (face 0.35, pose face points) | 99.6 / 85.6 | 99.2 / 87.3 | 99.7 / 62.6 | 100 / 61.8 |
| **#381 chosen** (SCRFD, 0.15 / 0.10, `reject_after=2`, appearance 0.85, `reach_back`, waits for scores) | **99.8 / 94.3** | 99.6 / 93.9 | **97.8 / 67.9** | 97.9 / 72.6 |
| the same, not waiting for scores | 99.3 / 92.0 | | 98.7 / 65.8 | |
| without the detector, not waiting | 99.1 / 89.4 | | 98.8 / 72.3 | |

The settings were picked on the odd scenes of v1 and not tuned again on v2.

### Action F1, 감스트 (matched / events / truth; example ranges excluded)

In-sample = the setting picked on all scenes. Held-out = picked on one half of the scenes (the run's own scene cuts),
scored on the other. The 관제탑 odd half holds the examples' scene, so "odd → even" is the honest few-shot number.

| method | in-sample, #379 identity | in-sample, #381 identity | held-out odd → even (#379) | held-out even → odd (#379) |
|---|---|---|---|---|
| 관제탑 text (rule) | 31/33/39, 0.861 | — | 12/15/18, 0.727 | 17/17/21, 0.895 |
| 관제탑 clips only | 34/41/39, 0.850 | — | 15/29/18, 0.638 | 17/18/21, 0.872 |
| 관제탑 text + clips, count | 29/31/39, 0.829 | 31/33/39, **0.861** | 11/13/18, 0.710 | 17/17/21, 0.895 |
| 관제탑 text + clips, count, calibrated | 22/23/39, 0.710 | — | 3/3/18, 0.286 | 13/13/21, 0.765 |
| 관제탑 text + clips, search | 36/42/39, **0.889** | 37/47/39, 0.860 | 14/18/18, 0.778 | 17/17/21, 0.895 |
| 맨까송 punch rule | 6/10/16, 0.462 | 9/14/16, **0.600** | 4/7/7, 0.571 | 0/4/9, 0.000 |

- With #381's identity the same fixed settings score per half (not held-out picks): search 0.889 / 0.829, count
  0.923 / 0.788, punch 0.571 / 0.500 (odd / even).
- #381 run that did not wait for scores: search 39/47/39, 0.907 (every one of 감스트's 39 occurrences found), count
  0.877, punch 0.621. The waiting run (the post default now) is the one reported as "chosen".
- Everyone (all people, 61 occurrences): text + clips, count, in-sample 46/51/61, 0.821; search 60/90/61, 0.795
  (v1, [action spec](measurements/action-spec-2026-11-28.md)).

## Remaining errors and what to try next

| error | evidence | next (not measured) |
|---|---|---|
| **맨까송 punch rule.** The rule picked on one half finds 0 of 9 on the other (held-out 0.571 / 0.000); punches are mostly a fist brought down on the chair, at 360p, sometimes from behind | [action spec](measurements/action-spec-2026-11-28.md) "맨까송"; 3 of 16 still given to someone else with #381 | owner example clips of the punch (few-shot, then text + clips); a negative clip of hands moving without a punch; the impact sound as in the shotgun recipe (`--with impact`) |
| **Low-resolution side faces.** 맨까송 identity recall 67.9 %; 감스트 in profile at the top edge scores 0.28-0.33; neither the pose points nor SCRFD give a face on 64 of 244 sampled 감스트 boxes | [identity recall](measurements/identity-recall-2026-11-28.md) §1-2 | a learned face super-resolution (a Lanczos upscale changed nothing, AUC 0.899 → 0.904); whole-body appearance carried within a scene was built and changed nothing (`scene_threshold`, off) |
| **Held-out variance.** A half holds 7-21 occurrences, so one or two decide the F1: 관제탑 search 0.778 vs 0.895, clips only 0.638 vs 0.872 | the tables above | more labelled clips of the same actions; report a held-out range, not one number |
| **Calibration off by default.** The 7 examples come from one scene and start at the spread: calibrating the rule from them widened the fold to 2.94 shoulder widths and merged repetitions (count 0.829 → 0.710, held-out 0.286) | lessons 81, 83 | examples from several scenes and at least one negative before calibration is turned on |
| 관제탑 331-420 s | the truth labels two different people 감스트 at once; 2 occurrences there (370.8-372.4 s) go to the other person | v2 scores them "one of"; nothing left to fix in GAMST for this stretch |

## How to run

### The HUD (setup step, then the live view in the same window)

```bash
gamst recipe hud --input "감스트-관제탑 모음집.mp4" --who 감스트 --photo a.jpg --photo b.jpg --photo c.jpg \
  --action "관제탑: 두 팔을 앞으로 오므렸다가 양옆으로 폈다가 다시 접는다" --rule docs/movements/gwanjetap.rule.json \
  --example "감스트-관제탑 모음집.mp4@62.051-63.951" --mode search --timing post --out-dir gamst-hud
```

- Every option only fills in the window; `--start` starts at once when nothing is missing. A camera: `--input 0`
  (realtime). Keys and every element of the window: [hud-demo.md](hud-demo.md) §0-1.
- The identity settings of #381 are options: `--set person_binding.face_threshold=0.15 --set
  person_binding.face_reject=0.1 --set person_binding.reject_after=2 --set person_binding.body_threshold=0.85 --set
  person_binding.reach_back=true` after `--`, and the SCRFD detector in the profile's `vision.face_identity` config
  (`detector_path`, `detector_sha256`); see [person-identity.md](person-identity.md).

### MCP `compose_recipe` (an agent composes and runs it)

```bash
gamst mcp --clips DIR
```

```json
{"name": "gwan-gamst", "inputs": {"video": "감스트-관제탑 모음집.mp4"},
 "who": {"name": "감스트", "text": "탈색 짧은 머리, 검은 뿔테", "photos": ["gamst1.jpg", "gamst2.jpg", "gamst3.jpg"],
         "consent": "owner, for counting on his own clips"},
 "what": {"movement": {"text": "관제탑: 두 팔을 앞으로 오므렸다가 양옆으로 폈다가 다시 접는다 (1회씩 따로)",
                       "rule": {"steps": ["... docs/movements/gwanjetap.rule.json ..."]},
                       "examples": ["감스트-관제탑 모음집.mp4@62.051-63.951"]}},
 "mode": "search", "timing": "post", "start": true, "wait": true}
```

Then `count_events {capability_id: temporal.exemplar_events, event: exemplar.matched, who: 감스트}` returns each
occurrence's start and end with its evidence, and `get_timeline` the same as intervals. Request keys, refusals and the
answer: [compose-recipe.md](compose-recipe.md).

## Sources

- Design: [design-2026-11-27-integrated-perception.md](design-2026-11-27-integrated-perception.md)
- Decisions 35-67: [decision-log-2026-11-26.md](decision-log-2026-11-26.md) · lessons 38-102:
  [lessons-2026-11-26.md](lessons-2026-11-26.md)
- Measurements: [testbed](measurements/testbed-2026-11-26.md) (realtime single model, L2, black frame, post, raw-score
  hand gate, Sapiens2), [person identity](measurements/person-identity-2026-11-27.md),
  [action spec](measurements/action-spec-2026-11-28.md), [identity recall](measurements/identity-recall-2026-11-28.md)
- Truth: [ground-truth/2026-11-28](ground-truth/2026-11-28/README.md) (v1, v2 with
  [CHANGELOG](ground-truth/2026-11-28/v2/CHANGELOG.md))
- How-to: [person-identity.md](person-identity.md), [compose-recipe.md](compose-recipe.md), [hud-demo.md](hud-demo.md)
- Previous wrap-up (words and movements, 11-25 – 27):
  [wrapup-2026-11-26-person-words-and-movements.md](wrapup-2026-11-26-person-words-and-movements.md)
