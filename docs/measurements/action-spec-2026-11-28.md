# A given person's given action: text, clips or both, on the owner's truth (2026-11-28)

Measured on the owner's labelled clips (`docs/ground-truth/2026-11-28/`, PR #378) with the code of branch
`feat/action-spec`. Owner decisions 58-63 ([decision log](../decision-log-2026-11-26.md)). Every number below is
measured; thresholds chosen while looking at these labels are **in-sample**, and the held-out rows say what they are.

## 요약 (한국어)

- **무엇을 쟀나:** 관제탑(522초, 30fps)과 맨까송(67초, 360p)에서 "감스트의 동작"과 "모든 사람의 동작"을 사장님 정답과 비교했습니다.
  - 관제탑: 글(규칙)만, 예시 7개만(few-shot), 글+예시(count = 둘 다, search = 둘 중 하나).
  - 맨까송: 글(펀치 규칙)만. 사장님 예시가 없어 few-shot은 해당 없음.
- **방법:** 실제 recipe 실행(post 모드, RTMW-l, PersonViT 인물 연결) 한 번에서 각 사람의 관절을 기록했습니다. 같은 카운터 코드로 다시 돌려 방법마다 쟀습니다(replay).
  - 같은 규칙의 replay는 실제 실행과 관제탑 37/37, 맨까송 159/162 사건이 같았습니다. 나머지 3개는 소수 4자리 반올림 때문입니다.
  - 고른 설정은 실제 실행으로 한 번 더 확인했습니다(아래 "실제 실행 확인").
- **채점:**
  - 사장님 채점기(score_multi.py)의 규칙 그대로입니다. 시간으로 1:1 짝을 짓고, 0.5초 허용입니다.
  - 예시 7개 구간과 겹치는 정답·사건은 뺐습니다. 그래서 관제탑 감스트 정답은 46개가 아니라 39개, 모두는 68개가 아니라 61개입니다.
- **held-out:** GAMST 자체 장면 전환(scene.cut)으로 장면을 나누고, 홀수 장면에서 고른 설정을 짝수 장면에서 쟀습니다(반대도 같음).
  - 관제탑 홀수 장면(0–89초)에 사장님 예시 장면이 있습니다. 그래서 few-shot의 정직한 held-out은 "홀수→짝수"입니다.

| 영상 · 방법 (감스트) | 그대로 / 기본값 | 안 맞춘 장면 기준 (held-out) | 맞춘 장면 그대로 (in-sample) |
|---|---|---|---|
| 관제탑 · 글(규칙) | 20/21개 맞음, F1 0.667 (정밀 95%, 재현 51%) | 홀→짝 0.727, 짝→홀 0.895 | 31/33, F1 0.861 |
| 관제탑 · 예시만(few-shot) | 예시에서 정한 문턱(0.138): 1개만 셈, F1 0.05 | 홀→짝 0.612, 짝→홀 0.872 | 34/42, F1 0.839 (문턱 0.35) |
| 관제탑 · 글+예시, count(둘 다) | — | 홀→짝 0.710, 짝→홀 0.895 | 29/31, F1 0.829 (정밀 94%) |
| 관제탑 · 글+예시, search(둘 중 하나) | — | 홀→짝 0.757, 짝→홀 0.895 | 36/43, F1 0.878 (재현 92%) |
| 맨까송 · 글(펀치 규칙) | 8/34, F1 0.32 | 홀→짝 0.571, 짝→홀 0.000 | 6/10, F1 0.462 |

- **가장 좋은 설정:**
  - 관제탑: 글+예시 search 모드. in-sample F1 0.878, held-out 0.757 / 0.895.
  - count 모드(둘 다)는 정밀도가 가장 높습니다: 94% in-sample, held-out 85% / 100%.
  - 규칙: 오므림 = 두 손목 거리가 어깨너비의 2.4배 미만. 폄 = 2.8배 초과, 손목이 어깨 높이 ±1.5, 팔꿈치 130° 초과. 예시 문턱은 0.35(count)·0.30(search)입니다.
- **calibration(예시로 규칙 숫자 맞추기):** 이번 7개 예시에서는 오히려 나빠졌습니다. count in-sample 0.829 → 0.710입니다.
  - 원인 1: 첫 예시(62.05–63.95초, 1.9초짜리 느린 1회)가 폄→오므림 사이 시간을 1.5초에서 3.27초로 늘렸습니다. 그 바람에 연속 반복이 한 사건으로 이어졌습니다(merged 2 → 6).
  - 원인 2: 예시가 오므림 기준을 2.4배에서 2.94배로, 폄 기준을 2.8배에서 2.40배로 넓혔습니다. 두 자세의 경계가 겹쳤습니다(규칙 단계가 그 경계에서 바로 넘어감).
  - 그래서 측정의 기본은 "글의 숫자"입니다. 기능은 있지만 예시가 한 장면뿐일 때는 쓰지 않는 편이 낫습니다.
- **오류 분해:**
  - 관제탑 count 최고 설정, 감스트: 사람 착오 0, 반복 쪼갬 1, 없는 동작 1. 놓침은 사람 6, 합쳐짐 2, 못 봄 2입니다.
  - 맨까송 최고 규칙, 감스트: 없는 동작 4. 놓침은 사람 7, 못 봄 3입니다.
  - **두 영상 모두 지금 가장 큰 한계는 인물 연결입니다.** 감스트 표시 프레임 중 GAMST가 감스트라고 한 비율이 관제탑 70%, 맨까송 53%입니다(정밀도 99.6% / 82.3%). 이 PR에서는 고치지 않습니다. 다음 목표입니다.
- **예시를 누구에게서 배우나:**
  - 이제 클립마다 점(클릭)이나 `who`(인물 연결)로 사람을 고를 수 있습니다.
  - 관제탑 예시 7개는 원래도 가장 큰 사람이 감스트(person-1)였습니다. 그래서 전/후 결과가 거의 같습니다(아래 표).
  - `examples.json`에는 사장님 클릭이 없습니다(click: null). 그래서 클릭 위치는 사장님 인물 정답의 감스트 상자 가운데로 했습니다.
- **맨까송 펀치:** 규칙이 장면마다 달라 held-out이 흔들립니다(0.571 / 0.000). 감스트 펀치는 대부분 의자를 주먹으로 내리치는 동작이고, 360p라 손이 잘 안 보입니다.

## Method

- **Run:** `gamst recipe run person-action-count`, post mode, RTMW-l 384x288 + YOLOX-m (CoreML), person binding with
  LVFace-B + PersonViT (`profile-identity-personvit.json`), `--person 감스트.person.json`. `python -m gamst.eval.pose_dump
  record` kept every frame the per-track counter was given (poses with the 12 body and 42 hand joints, raw scores, boxes,
  tracks; the binding's records) in a local file (`try-it/testbed/action_eval/*.poses.jsonl.gz`, not in the repo).
- **Replay:** each movement value ran through `gamst.eval.pose_dump.replay`: a fresh `TrackExemplar` (the recipe's
  counter for RTMW poses) over the recorded frames. Replaying the run's own rule gave the run's events: 관제탑 37 of 37
  identical, 맨까송 159 of 162 (3 differ by one frame at a rate threshold; the dump keeps 4 decimals).
- **Few-shot values** were built from the recorded frames of the person the binding names 감스트 inside the 7 example
  ranges, with the functions `gamst recipe enroll` uses (`template`, `derive`, `calibrate`:
  `gamst.eval.pose_dump.movement_value`).
- **Score:** `gamst.eval.multi_person` (the owner's `score_multi.py` rules): greedy one-to-one by time, overlap or a gap
  of at most 0.5 s; who of an event = the binding resolved retroactively (unknown is not 감스트). Example ranges excluded
  from both sides (lesson 3). Errors: `identity` (right motion, wrong person), `split` (a second event on one
  occurrence), `merged` (one event over two occurrences; for everyone this includes two people doing it at the same
  moment), `extra`, `missed`.
- **Halves:** the run's own `scene.cut` events (관제탑 12 scenes, 맨까송 18). Odd = 1st, 3rd, ... scene.
  관제탑 odd half: 21 감스트 + 1 other occurrence (holds the examples' scene, 0-89 s); even: 18 + 21. 맨까송 odd 9, even 7.
- **Grids** (`try-it/testbed/action_eval/evaluate.py`): 관제탑 rule 144 settings (fold gap 1.8-2.7, fold elbow < 120 or
  none, spread gap 2.4/2.8, height window ±1.5/2/3, within 1/1.5/2 s); few-shot enter 0.18-0.40; text + few-shot over
  the 3 rules picked on all / odd / even scenes (held out: the rule picked on the same half) x enter x all/any x
  calibrated or not (60); 맨까송 324 punch settings (fist `hand_open` < 0.9-1.4 or none, wrist rate 2-8 shoulder widths/s,
  away from the shoulder and/or downwards, hold 0/34/67 ms).
- **Rules as written from the owner's text (v0)** are `docs/movements/gwanjetap.rule.json` and `punch.rule.json`; the
  fist gate uses the raw-score hand gate (`HAND_MIN_RAW_SCORE` 3.0, PR #373).

## Results (F1 = 2PR/(P+R); "짝" = matched pairs)
## 관제탑 (gwan)

### text only (rule)


**감스트**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| as written (rule {"Df": 1.2, "Af": 120, "Ds": 2.4, "H": 0.8, "As": 130, "W": 1500}) | 39 | 21 | -18 | 20 | 95.2% | 51.3% | 0.667 | 0.47 | 0.28 |
| best in-sample (all scenes) | 39 | 33 | -6 | 31 | 93.9% | 79.5% | 0.861 | 0.50 | 0.31 |
| tuned on odd scenes → tested on even | 18 | 15 | -3 | 12 | 80.0% | 66.7% | 0.727 | 0.40 | 0.27 |
| tuned on even scenes → tested on odd | 21 | 17 | -4 | 17 | 100.0% | 81.0% | 0.895 | 0.53 | 0.33 |
|   (same, in-sample on odd) | 21 | 19 | -2 | 19 | 100.0% | 90.5% | 0.950 | 0.53 | 0.43 |
|   (same, in-sample on even) | 18 | 16 | -2 | 14 | 87.5% | 77.8% | 0.824 | 0.48 | 0.28 |

**모두**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| as written (rule {"Df": 1.2, "Af": 120, "Ds": 2.4, "H": 0.8, "As": 130, "W": 1500}) | 61 | 32 | -29 | 31 | 96.9% | 50.8% | 0.667 | 0.45 | 0.25 |
| best in-sample (all scenes) | 61 | 60 | -1 | 50 | 83.3% | 82.0% | 0.826 | 0.52 | 0.30 |
| tuned on odd scenes → tested on even | 39 | 42 | +3 | 30 | 71.4% | 76.9% | 0.741 | 0.60 | 0.31 |
| tuned on even scenes → tested on odd | 22 | 23 | +1 | 20 | 87.0% | 90.9% | 0.889 | 0.54 | 0.34 |
|   (same, in-sample on odd) | 22 | 25 | +3 | 21 | 84.0% | 95.5% | 0.894 | 0.55 | 0.43 |
|   (same, in-sample on even) | 39 | 37 | -2 | 29 | 78.4% | 74.4% | 0.763 | 0.48 | 0.26 |

- best in-sample: rule {"Df": 2.4, "Af": null, "Ds": 2.8, "H": 1.5, "As": 130, "W": 1500}
- picked on odd: rule {"Df": 2.1, "Af": null, "Ds": 2.4, "H": 1.5, "As": 130, "W": 1500}
- picked on even: rule {"Df": 2.4, "Af": null, "Ds": 2.8, "H": 1.5, "As": 130, "W": 1500}
- errors of the best in-sample (감스트): wrong {'identity': 0, 'split': 1, 'extra': 1}, missed {'identity': 6, 'merged': 1, 'missed': 1}; everyone: wrong {'identity': 0, 'split': 2, 'extra': 8}, missed {'identity': 0, 'merged': 8, 'missed': 3}

### few-shot only (7 owner examples; occurrences inside them excluded)


**감스트**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| as written (few-shot enter=derived) | 39 | 1 | -38 | 1 | 100.0% | 2.6% | 0.050 | 0.43 | 0.00 |
| best in-sample (all scenes) | 39 | 42 | +3 | 34 | 81.0% | 87.2% | 0.839 | 0.35 | 0.42 |
| tuned on odd scenes → tested on even | 18 | 31 | +13 | 15 | 48.4% | 83.3% | 0.612 | 0.39 | 0.50 |
| tuned on even scenes → tested on odd | 21 | 18 | -3 | 17 | 94.4% | 81.0% | 0.872 | 0.29 | 0.34 |
|   (same, in-sample on odd) | 21 | 22 | +1 | 19 | 86.4% | 90.5% | 0.884 | 0.31 | 0.33 |
|   (same, in-sample on even) | 18 | 16 | -2 | 13 | 81.2% | 72.2% | 0.765 | 0.47 | 0.48 |

**모두**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| as written (few-shot enter=derived) | 61 | 2 | -59 | 2 | 100.0% | 3.3% | 0.064 | 0.25 | 0.03 |
| best in-sample (all scenes) | 61 | 85 | +24 | 56 | 65.9% | 91.8% | 0.767 | 0.44 | 0.49 |
| tuned on odd scenes → tested on even | 39 | 71 | +32 | 36 | 50.7% | 92.3% | 0.655 | 0.46 | 0.56 |
| tuned on even scenes → tested on odd | 22 | 25 | +3 | 20 | 80.0% | 90.9% | 0.851 | 0.34 | 0.36 |
|   (same, in-sample on odd) | 22 | 33 | +11 | 22 | 66.7% | 100.0% | 0.800 | 0.35 | 0.35 |
|   (same, in-sample on even) | 39 | 45 | +6 | 33 | 73.3% | 84.6% | 0.786 | 0.43 | 0.48 |

- best in-sample: few-shot enter=0.35
- picked on odd: few-shot enter=0.4
- picked on even: few-shot enter=0.3
- errors of the best in-sample (감스트): wrong {'identity': 0, 'split': 4, 'extra': 4}, missed {'identity': 3, 'merged': 0, 'missed': 2}; everyone: wrong {'identity': 0, 'split': 16, 'extra': 13}, missed {'identity': 0, 'merged': 4, 'missed': 1}

### text + few-shot, combine=all (count mode), rule numbers as written


**감스트**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best in-sample (all scenes) | 39 | 31 | -8 | 29 | 93.5% | 74.4% | 0.829 | 0.48 | 0.27 |
| tuned on odd scenes → tested on even | 18 | 13 | -5 | 11 | 84.6% | 61.1% | 0.710 | 0.38 | 0.28 |
| tuned on even scenes → tested on odd | 21 | 17 | -4 | 17 | 100.0% | 81.0% | 0.895 | 0.53 | 0.33 |
|   (same, in-sample on odd) | 21 | 18 | -3 | 18 | 100.0% | 85.7% | 0.923 | 0.52 | 0.36 |
|   (same, in-sample on even) | 18 | 14 | -4 | 12 | 85.7% | 66.7% | 0.750 | 0.41 | 0.19 |

**모두**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best in-sample (all scenes) | 61 | 51 | -10 | 46 | 90.2% | 75.4% | 0.821 | 0.47 | 0.28 |
| tuned on odd scenes → tested on even | 39 | 34 | -5 | 29 | 85.3% | 74.4% | 0.795 | 0.55 | 0.34 |
| tuned on even scenes → tested on odd | 22 | 20 | -2 | 19 | 95.0% | 86.4% | 0.905 | 0.51 | 0.31 |
|   (same, in-sample on odd) | 22 | 21 | -1 | 19 | 90.5% | 86.4% | 0.884 | 0.51 | 0.34 |
|   (same, in-sample on even) | 39 | 31 | -8 | 27 | 87.1% | 69.2% | 0.771 | 0.44 | 0.26 |

- best in-sample: rule {"Df": 2.4, "Af": null, "Ds": 2.8, "H": 1.5, "As": 130, "W": 1500} + few-shot enter=0.35 all text numbers
- picked on odd: rule {"Df": 2.1, "Af": null, "Ds": 2.4, "H": 1.5, "As": 130, "W": 1500} + few-shot enter=0.35 all text numbers
- picked on even: rule {"Df": 2.4, "Af": null, "Ds": 2.8, "H": 1.5, "As": 130, "W": 1500} + few-shot enter=0.35 all text numbers
- errors of the best in-sample (감스트): wrong {'identity': 0, 'split': 1, 'extra': 1}, missed {'identity': 6, 'merged': 2, 'missed': 2}; everyone: wrong {'identity': 0, 'split': 2, 'extra': 3}, missed {'identity': 0, 'merged': 10, 'missed': 5}

### text + few-shot, combine=all (count mode), calibrated rule numbers


**감스트**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best in-sample (all scenes) | 39 | 23 | -16 | 22 | 95.7% | 56.4% | 0.710 | 0.48 | 0.25 |
| tuned on odd scenes → tested on even | 18 | 3 | -15 | 3 | 100.0% | 16.7% | 0.286 | 0.47 | 0.15 |
| tuned on even scenes → tested on odd | 21 | 13 | -8 | 13 | 100.0% | 61.9% | 0.765 | 0.50 | 0.27 |
|   (same, in-sample on odd) | 21 | 8 | -13 | 8 | 100.0% | 38.1% | 0.552 | 0.39 | 0.20 |
|   (same, in-sample on even) | 18 | 9 | -9 | 8 | 88.9% | 44.4% | 0.593 | 0.47 | 0.25 |

**모두**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best in-sample (all scenes) | 61 | 37 | -24 | 35 | 94.6% | 57.4% | 0.714 | 0.51 | 0.30 |
| tuned on odd scenes → tested on even | 39 | 12 | -27 | 12 | 100.0% | 30.8% | 0.471 | 0.35 | 0.17 |
| tuned on even scenes → tested on odd | 22 | 16 | -6 | 15 | 93.8% | 68.2% | 0.789 | 0.48 | 0.24 |
|   (same, in-sample on odd) | 22 | 11 | -11 | 10 | 90.9% | 45.5% | 0.606 | 0.38 | 0.18 |
|   (same, in-sample on even) | 39 | 15 | -24 | 15 | 100.0% | 38.5% | 0.556 | 0.40 | 0.16 |

- best in-sample: rule {"Df": 2.4, "Af": null, "Ds": 2.8, "H": 1.5, "As": 130, "W": 1500} + few-shot enter=0.35 all calibrated
- picked on odd: rule {"Df": 2.1, "Af": null, "Ds": 2.4, "H": 1.5, "As": 130, "W": 1500} + few-shot enter=0.26 all calibrated
- picked on even: rule {"Df": 2.4, "Af": null, "Ds": 2.8, "H": 1.5, "As": 130, "W": 1500} + few-shot enter=0.26 all calibrated
- errors of the best in-sample (감스트): wrong {'identity': 1, 'split': 0, 'extra': 0}, missed {'identity': 5, 'merged': 6, 'missed': 6}; everyone: wrong {'identity': 0, 'split': 0, 'extra': 2}, missed {'identity': 0, 'merged': 11, 'missed': 15}

### text + few-shot, combine=any (search mode), rule numbers as written


**감스트**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best in-sample (all scenes) | 39 | 43 | +4 | 36 | 83.7% | 92.3% | 0.878 | 0.48 | 0.42 |
| tuned on odd scenes → tested on even | 18 | 19 | +1 | 14 | 73.7% | 77.8% | 0.757 | 0.26 | 0.31 |
| tuned on even scenes → tested on odd | 21 | 17 | -4 | 17 | 100.0% | 81.0% | 0.895 | 0.53 | 0.33 |
|   (same, in-sample on odd) | 21 | 21 | +0 | 20 | 95.2% | 95.2% | 0.952 | 0.43 | 0.49 |
|   (same, in-sample on even) | 18 | 16 | -2 | 14 | 87.5% | 77.8% | 0.824 | 0.48 | 0.28 |

**모두**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best in-sample (all scenes) | 61 | 90 | +29 | 60 | 66.7% | 98.4% | 0.795 | 0.53 | 0.45 |
| tuned on odd scenes → tested on even | 39 | 56 | +17 | 39 | 69.6% | 100.0% | 0.821 | 0.45 | 0.36 |
| tuned on even scenes → tested on odd | 22 | 23 | +1 | 20 | 87.0% | 90.9% | 0.889 | 0.54 | 0.34 |
|   (same, in-sample on odd) | 22 | 28 | +6 | 22 | 78.6% | 100.0% | 0.880 | 0.52 | 0.47 |
|   (same, in-sample on even) | 39 | 37 | -2 | 29 | 78.4% | 74.4% | 0.763 | 0.48 | 0.26 |

- best in-sample: rule {"Df": 2.4, "Af": null, "Ds": 2.8, "H": 1.5, "As": 130, "W": 1500} + few-shot enter=0.3 any text numbers
- picked on odd: rule {"Df": 2.1, "Af": null, "Ds": 2.4, "H": 1.5, "As": 130, "W": 1500} + few-shot enter=0.26 any text numbers
- picked on even: rule {"Df": 2.4, "Af": null, "Ds": 2.8, "H": 1.5, "As": 130, "W": 1500} + few-shot enter=derived any text numbers
- errors of the best in-sample (감스트): wrong {'identity': 0, 'split': 5, 'extra': 2}, missed {'identity': 3, 'merged': 0, 'missed': 0}; everyone: wrong {'identity': 0, 'split': 17, 'extra': 13}, missed {'identity': 0, 'merged': 1, 'missed': 0}

### text + few-shot, combine=any (search mode), calibrated rule numbers


**감스트**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best in-sample (all scenes) | 39 | 37 | -2 | 32 | 86.5% | 82.0% | 0.842 | 0.37 | 0.31 |
| tuned on odd scenes → tested on even | 18 | 24 | +6 | 15 | 62.5% | 83.3% | 0.714 | 0.41 | 0.46 |
| tuned on even scenes → tested on odd | 21 | 26 | +5 | 20 | 76.9% | 95.2% | 0.851 | 0.39 | 0.43 |
|   (same, in-sample on odd) | 21 | 22 | +1 | 19 | 86.4% | 90.5% | 0.884 | 0.37 | 0.43 |
|   (same, in-sample on even) | 18 | 22 | +4 | 15 | 68.2% | 83.3% | 0.750 | 0.55 | 0.44 |

**모두**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best in-sample (all scenes) | 61 | 76 | +15 | 57 | 75.0% | 93.4% | 0.832 | 0.42 | 0.40 |
| tuned on odd scenes → tested on even | 39 | 68 | +29 | 37 | 54.4% | 94.9% | 0.692 | 0.43 | 0.49 |
| tuned on even scenes → tested on odd | 22 | 35 | +13 | 22 | 62.9% | 100.0% | 0.772 | 0.45 | 0.38 |
|   (same, in-sample on odd) | 22 | 32 | +10 | 21 | 65.6% | 95.5% | 0.778 | 0.39 | 0.38 |
|   (same, in-sample on even) | 39 | 67 | +28 | 37 | 55.2% | 94.9% | 0.698 | 0.52 | 0.55 |

- best in-sample: rule {"Df": 1.2, "Af": 120, "Ds": 2.4, "H": 0.8, "As": 130, "W": 1500} + few-shot enter=0.3 any calibrated
- picked on odd: rule {"Df": 2.1, "Af": null, "Ds": 2.4, "H": 1.5, "As": 130, "W": 1500} + few-shot enter=0.35 any calibrated
- picked on even: rule {"Df": 2.4, "Af": null, "Ds": 2.8, "H": 1.5, "As": 130, "W": 1500} + few-shot enter=0.3 any calibrated
- errors of the best in-sample (감스트): wrong {'identity': 0, 'split': 4, 'extra': 1}, missed {'identity': 7, 'merged': 0, 'missed': 0}; everyone: wrong {'identity': 0, 'split': 13, 'extra': 6}, missed {'identity': 0, 'merged': 4, 'missed': 0}

## 맨까송 (man)

### text only (rule)


**감스트**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| as written (rule {"F": 1.1, "R": 3, "kinds": ["dist"], "hold": 0}) | 16 | 34 | +18 | 8 | 23.5% | 50.0% | 0.320 | 0.32 | 0.50 |
| best in-sample (all scenes) | 16 | 10 | -6 | 6 | 60.0% | 37.5% | 0.462 | 0.37 | 0.53 |
| tuned on odd scenes → tested on even | 7 | 7 | +0 | 4 | 57.1% | 57.1% | 0.571 | 0.07 | 0.09 |
| tuned on even scenes → tested on odd | 9 | 4 | -5 | 0 | 0.0% | 0.0% | 0.000 | — | — |
|   (same, in-sample on odd) | 9 | 1 | -8 | 1 | 100.0% | 11.1% | 0.200 | 0.60 | 0.77 |
|   (same, in-sample on even) | 7 | 7 | +0 | 5 | 71.4% | 71.4% | 0.714 | 0.44 | 0.48 |

**모두**

| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | F1 | 시작오차(초) | 끝오차(초) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| as written (rule {"F": 1.1, "R": 3, "kinds": ["dist"], "hold": 0}) | 16 | 81 | +65 | 15 | 18.5% | 93.8% | 0.309 | 0.28 | 0.39 |
| best in-sample (all scenes) | 16 | 27 | +11 | 12 | 44.4% | 75.0% | 0.558 | 0.29 | 0.36 |
| tuned on odd scenes → tested on even | 7 | 21 | +14 | 4 | 19.1% | 57.1% | 0.286 | 0.07 | 0.09 |
| tuned on even scenes → tested on odd | 9 | 15 | +6 | 6 | 40.0% | 66.7% | 0.500 | 0.18 | 0.21 |
|   (same, in-sample on odd) | 9 | 13 | +4 | 5 | 38.5% | 55.6% | 0.455 | 0.31 | 0.36 |
|   (same, in-sample on even) | 7 | 13 | +6 | 5 | 38.5% | 71.4% | 0.500 | 0.44 | 0.48 |

- best in-sample: rule {"F": 1.0, "R": 3, "kinds": ["dist", "down"], "hold": 34}
- picked on odd: rule {"F": null, "R": 8, "kinds": ["down"], "hold": 67}
- picked on even: rule {"F": 1.2, "R": 4, "kinds": ["dist", "down"], "hold": 34}
- errors of the best in-sample (감스트): wrong {'identity': 0, 'split': 0, 'extra': 4}, missed {'identity': 7, 'merged': 0, 'missed': 3}; everyone: wrong {'identity': 0, 'split': 1, 'extra': 14}, missed {'identity': 0, 'merged': 0, 'missed': 4}

(few-shot: not applicable, the owner gave no example clips for 맨까송)

## Whom the examples are learnt from (few-shot only, 관제탑)

Before this PR `gamst recipe enroll` learnt each clip from the pose with the most visible channels (largest visible).
Now a clip can name a point (`FILE@A-B@X,Y`) or the person of `who` (`--from-who`). `examples.json` holds no owner click
(`click: null` for all 7), so the point was the centre of the owner's own 감스트 box (`truth_identity.json`, label G)
at each clip's start. Matched / events, example ranges excluded (39 감스트, 61 everyone):

| enter | largest visible (before) 감스트 | 모두 | owner's box (click) 감스트 | 모두 | person binding 감스트 (`--from-who`) | 모두 |
|---|---|---|---|---|---|---|
| derived | 6/6 (enter 0.163) | 11/11 | 1/1 (0.138) | 2/2 | 1/1 (0.138) | 2/2 |
| 0.22 | 20/20 | 36/37 | 20/20 | 36/37 | 20/20 | 36/37 |
| 0.26 | 27/29 | 48/55 | 27/29 | 48/55 | 27/29 | 48/55 |
| 0.30 | 29/33 | 52/69 | 30/34 | 53/70 | 30/34 | 53/70 |
| 0.35 | 34/41 | 56/83 | 34/42 | 56/85 | 34/42 | 56/85 |
| 0.40 | 35/52 | 58/103 | 35/53 | 58/104 | 35/53 | 58/104 |

In all 7 clips the largest visible person was already 감스트 (track person-1; a second person, person-16 / -18 / -19,
is in every clip), so the fix changes almost nothing here: the click and the binding chose the same track. The derived
`enter` still differs (0.163 vs 0.138) and so do one or two events per row; the cause was not traced (the frames and
the track are the same). The selector matters where someone else is larger than the person meant; that was not
the case in these examples.
