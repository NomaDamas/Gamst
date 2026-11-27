# Person-identity recall on the owner's truth (2026-11-28)

Measured on the owner's labelled clips (`docs/ground-truth/2026-11-28/`, PR #378) with the code of branch
`feat/identity-recall` (on PR #379). Every number below is measured. Settings chosen while looking at these labels are
**in-sample**; the held-out rows say on which half they were picked.

## 요약 (한국어)

- **무엇을 했나:** 감스트 표시 프레임 중 GAMST가 감스트라고 한 비율(재현율)이 낮았습니다(관제탑 70 %, 맨까송 53 %). 놓친 프레임을 원인별로 먼저 세고, 원인마다 고친 뒤, 두 영상에서 실제 실행(post 모드)으로 다시 쟀습니다.
- **원인 (고치기 전 실제 실행):**

  | 원인 | 관제탑 (감스트 프레임 17,561) | 맨까송 (1,835) |
  |---|---:|---:|
  | 같은 순간 정답의 다른 감스트 상자가 이미 감스트로 불림 (한 사람 = 한 트랙 규칙) | 2,523 | 97 |
  | 트랙이 다른 사람(P2…)에게 묶임 | 814 | 1 |
  | 얼굴 점수는 문턱 이상인데 다른 트랙/다른 사람과 경쟁에서 짐 | 662 | 140 |
  | 흔들린 얼굴 한 번에 묶임이 풀림(얼굴 반박) | 571 | 32 |
  | 그 구간에서 얼굴이 한 번도 점수 나지 않음 | 356 | 111 |
  | 얼굴이 문턱(0.35) 아래 | 348 | 284 |
  | 얼굴이 너무 작음(눈 사이 12픽셀 미만) | 0 | 16 |
  | 사람 상자 자체가 없음(사장님이 직접 추가한 구간) | 0 | 179 |
  | 컷 뒤 재확인 대기 | 0 (첫 묶임은 컷까지 거슬러 올라감) | 0 |

  - 확인 사진: `try-it/testbed/idrecall/check/` (저장소 밖).
  - **관제탑 331–420초**: 정답이 **서로 다른 두 사람**을 동시에 감스트로 표시한 구간이 있습니다(344·362·385초 사진). 놓친 5,274 프레임 중 3,483이 이 구간입니다.
  - **맨까송**: 벽의 **감스트 액자 사진**도 감스트(G)로 표시되어 있습니다. 또 정답 "다른 사람"으로 틀렸다고 센 214 프레임 중 211이 한 구간(42–63.5초)인데, 그 구간 메모는 "0:51부터 감스트"라고 적혀 있고 사진(55초)도 감스트입니다. **사장님 확인이 필요합니다.**
- **가설 검증:** "포즈 모델 얼굴 점이 병목"이라는 가설은 **틀렸습니다.** 포즈 얼굴 점으로 얼굴을 맞춘 비율이 관제탑 95 %, 맨까송 71 %였고, SCRFD 얼굴 검출기는 87 %, 59 %였습니다. 감스트와 다른 사람을 가르는 힘(AUC)도 같았습니다(0.909 vs 0.907, 0.891 vs 0.899). 확대(Lanczos)도 나아지지 않았습니다(0.899).
- **고친 것:**
  1. **SCRFD-10GF 얼굴 검출기**(InsightFace, 16.9 MB, sha256 `5838f7fe…5b91`, 비상업 연구용 — GAMST 비상업이라 허용)로 머리 주변에서 얼굴과 5점을 찾아 LVFace에 넣는 후보 `lvface.b_scrfd`(옵션). 이 Mac에서 CPU 87 ms, CoreML 24 ms, 결과 같음. 효과는 "더 많이 찾기"가 아니라 "흔들린 얼굴을 얼굴로 안 봄"이었습니다(얼굴 반박으로 풀림 43 → 16, replay).
  2. **`reject_after=2`**(옵션): 반박하는 얼굴이 두 번 연속일 때만 풀림. 첫 번째 뒤엔 200 ms마다 다시 봄.
  3. **`reach_back`**(옵션): 다시 묶인 트랙이 직전 묶임 끝까지 거슬러 올라감(그 사람이 다른 트랙에 있던 시간, 그 트랙의 반박 얼굴은 넘지 않음).
  4. **문턱**(옵션): LVFace 얼굴 0.15 / 반박 0.10, PersonViT 0.85. 장면 홀/짝 반쪽에서 각각 골랐고 두 쪽이 거의 같은 설정을 골랐습니다(짝수 쪽은 외형 0.80).
  5. **post 모드는 얼굴 점수를 기다림(`wait_for_scores`, 이제 `--timing post`의 기본):** 전에는 post에서도 점수 계산이 따로 돌아 느리면 버려졌습니다(검출기를 켜면 맨까송 396개 중 75개). 관제탑 실행 시간 10.3 → 10.6분.
  6. **장면 안 외형 전파(`scene_threshold`)**: 요청대로 만들었지만 재현율이 변하지 않아(관제탑 3–4개 구간, 맨까송 0개) 꺼 두었습니다.
- **결과 (실제 실행, post; 전 → 고른 설정):**

  | | 관제탑 정밀 / 재현 | 맨까송 정밀 / 재현 |
  |---|---|---|
  | 전체 | 99.6 % / 70.0 % → **99.8 % / 77.1 %** | 82.0 % / 53.1 % → **77.8 % / 64.7 %** |
  | 짝수 장면 (홀수에서 고른 설정 = held-out) | 99.2 % / 73.9 % → 99.6 % / 79.5 % | 85.2 % / 51.5 % → 78.5 % / 68.2 % |
  | 홀수 장면 (in-sample) | 100 % / 64.7 % → 100 % / 73.9 % | 79.7 % / 54.4 % → 77.1 % / 61.8 % |
  | 정답이 두 곳을 동시에 감스트라 한 프레임 제외 | 99.5 % / 85.4 % → 99.7 % / 93.6 % | 80.8 % / 61.2 % → 75.7 % / 71.8 % |
  | 맨까송, 라벨이 의심되는 한 구간(42–63.5초) 제외 | | 99.7 % / 53.1 % → 98.0 % / 64.7 % |

  - 맨까송 정밀도가 떨어진 것은 거의 모두 그 한 구간입니다(감스트가 더 많이 불리면 그 구간의 "틀림"도 늘어남).
  - 검출기 없이(포즈 얼굴 점) 같은 방식으로 고른 설정: 관제탑 99.1 % / 73.1 %, 맨까송 81.7 % / 67.2 %(점수를 기다리지 않은 실행).
- **동작 개수에 미친 효과 (#379 최고 설정, 감스트 F1):**

  | 동작 | 전 | 고른 설정 (점수 기다림) | 같은 설정, 기다리지 않음 | 검출기 없이 |
  |---|---|---|---|---|
  | 관제탑 규칙+예시 search | 0.878 (36/43, 사람 착오 놓침 3) | 0.851 (37/48, 2) | **0.897** (39/48, 0) | 0.884 (38/47, 1) |
  | 관제탑 규칙+예시 count | 0.829 (29/31, 6) | 0.861 (31/33, 4) | **0.877** (32/34, 3) | 0.877 (32/34, 3) |
  | 맨까송 펀치 규칙 | 0.462 (6/10, 7) | 0.600 (9/14, 3) | **0.621** (9/13, 3) | 0.429 (6/12, 7) |

  - 관제탑 search에서 "기다림"이 놓친 2개(370.8–372.4초)는 정답이 서로 다른 두 사람을 동시에 감스트라 한 구간 안입니다.
  - 맨까송 펀치는 검출기가 있어야 좋아졌습니다(사람 착오 7 → 3). 검출기 없이 고른 설정은 그대로였습니다(7).
- **남은 것:** 관제탑은 정답의 "두 사람 동시 G" 구간과 다른 사람(P-n)에게 묶이는 경우, 맨까송은 얼굴이 옆/뒤를 보는 구간(검출기도 얼굴을 못 찾음)과 상자가 없는 구간입니다.

## Method

- **Runs:** `gamst recipe run person-action-count`, post mode, RTMW-l 384x288 + YOLOX-m, person binding with LVFace-B +
  PersonViT (`profile-identity-personvit.json`; the detector run: the same profile with `detector_path` +
  `detector_sha256` in the `vision.face_identity` config), `--person 감스트.person.json` (23 faces enrolled from another
  clip), recorded by `python -m gamst.eval.pose_dump record` (now with the 7 face points and the binding's per-track
  scores). Settings by `--set person_binding.KEY=VALUE`. Scripts: `try-it/testbed/idrecall/` (not in the repo).
- **Identity score:** `gamst.eval.multi_person.score_identity` (the owner's `score_multi.py` rules): each labelled box
  is matched to the run's box (IoU >= 0.5, within one frame) and counts as 감스트 when that track is bound to 감스트 then
  (bindings resolved retroactively); the owner's added stretches (맨까송, no box) count when any track is 감스트.
- **Causes** (`diag.py`): each missed 감스트 frame gets the first that applies: no run box; another of the truth's
  감스트 boxes is named 감스트 at that moment; the track is bound to another person; the frame is in a binding withdrawn
  by a contradicting face; else, over the track's segment in that scene: 감스트's face scored and reached 0.35 (lost the
  claim), scored only below 0.35, was never scored although the pose points gave a face crop on some frames, or the
  crop was never possible (eyes < 12 px). Re-confirmation after a cut is not a cause here: a segment's first binding
  reaches back to the cut. What remains of latency is a track bound *again*: 371 관제탑 frames lay between a track's
  release and its re-binding in the same scene while 감스트 was on no other track (fixed by `reach_back`).
- **Face evidence** (`evidence.py`, `faceq.py`): every 6th frame, every track: the pose points' aligned face (as the
  run), the SCRFD face in the head region (bilinear and Lanczos warp), the PersonViT body; scored against the enrolled
  face on the boxes the owner labelled 감스트 (G) or someone else (O).
- **Replay** (`gamst.eval.binding_replay`): the real `PersonBinding` run over the recorded poses with the cached
  embeddings instead of its models (results decided 100 ms after their frame, the run's median; the run's scene cuts).
  Replaying the run's own settings: 관제탑 P 99.8 % / R 69.1 % (run 99.6 / 70.0), 맨까송 77.3 / 57.6 (run 82.0 / 53.1).
  Grids: 440 settings (face source pose / SCRFD / either / Lanczos; face threshold 0.10-0.35; reject 0.05-0.15;
  reject_after 1-3; scene_threshold off / 0.8 / 0.9; appearance 0.80-0.90; reach_back), scored per frame on all, odd
  and even scenes; objective = mean of the two clips' frame F1 on the half.
- **Halves:** the base run's own `scene.cut` events (관제탑 12 scenes, 맨까송 18), odd = 1st, 3rd, ... scene, as in #379.
- **Action F1:** #379's movements replayed through the per-track counter on each run's recorded poses and with that
  run's bindings: 관제탑 rule {Df 2.4, Af none, Ds 2.8, H 1.5, As 130, W 1500} + the 7 examples, search (any, enter 0.30)
  and count (all, enter 0.35, `gwan_best.json`); 맨까송 punch rule {F 1.0, R 3, dist+down, hold 34}. Scored as in #379
  (0.5 s, the example ranges excluded). On the base run this gives #379's numbers exactly (search 36/43/39, count
  29/31/39, punch 6/10/16).

## 1. Why 감스트 was not named (base run)

| cause | 관제탑 frames | share | 맨까송 frames | share |
|---|---:|---:|---:|---:|
| another of the truth's 감스트 boxes is named at that moment (one person, one track) | 2,523 | 14.4 % | 97 | 5.3 % |
| track bound to another person (P2 ...) | 814 | 4.6 % | 1 | 0.1 % |
| face reached 0.35 but lost the claim (another track or person within the margin) | 662 | 3.8 % | 140 | 7.6 % |
| binding withdrawn by a contradicting face | 571 | 3.3 % | 32 | 1.7 % |
| no face scored in the segment (a pose crop was possible on some frames) | 356 | 2.0 % | 111 | 6.0 % |
| face scored below 0.35 only | 348 | 2.0 % | 284 | 15.5 % |
| face too small (eyes < 12 px) on every frame | 0 | — | 16 | 0.9 % |
| no person box (the owner's added stretches) | 0 | — | 179 | 9.8 % |
| named 감스트 | 12,287 | 70.0 % | 975 | 53.1 % |

What the check frames (`try-it/testbed/idrecall/check/`) show:

- 관제탑 344 s, 362 s, 385 s: two different people (white Tottenham shirt and blue Chelsea shirt; navy Puma shirt and
  a white Carlsberg shirt) are both labelled 감스트. 2,589 관제탑 frames have two 감스트 boxes that do not overlap
  (IoU < 0.3), 2,339 of them in 331-420 s, which also holds 3,483 of the 5,274 missed frames.
- 관제탑 85 s: 감스트 mid-관제탑, the face motion-blurred; that face contradicted him and withdrew the binding.
- 관제탑 218 s: 감스트 bent over, back of the head; the track was bound to another person's face (P3).
- 관제탑 268 s: 감스트 on track 45 with face scores 0.72-0.82; short extra tracks on him took the person in turns
  (`seen_on_another_track`), and each re-binding started at that frame (not back).
- 맨까송 3.7 s, 9.6 s, 20.6 s: 감스트 in profile and his framed photo on the wall, both labelled 감스트.
- 맨까송 38.7 s: 감스트 in profile at the top edge, 360p; face scores 0.28-0.33.
- 맨까송 55 s: 감스트 (orange Man Utd shirt, shaved head, glasses) on track 27, whose segment (42.0-63.5 s) is labelled
  "other" though its note says 감스트 from 0:51; 211 of the run's 214 "wrongly named" frames are in it.

## 2. The face evidence (sampled every 6th frame, the boxes labelled 감스트 / other)

| | 관제탑 감스트 boxes | 관제탑 others | 맨까송 감스트 | 맨까송 others |
|---|---|---|---|---|
| sampled boxes | 2,943 | 2,191 | 244 | 477 |
| aligned face from the pose points | 2,810 (95 %) | 2,031 | 173 (71 %) | 448 |
| SCRFD face (score >= 0.5) at the head | 2,561 (87 %) | 1,523 | 143 (59 %) | 424 |
| either | 2,815 | 2,037 | 180 | 461 |
| median score vs the enrolled face: pose points / SCRFD | 0.405 / 0.425 | 0.031 / 0.046 | 0.336 / 0.341 | 0.006 / 0.003 |
| AUC 감스트 vs others: pose points / SCRFD / SCRFD + Lanczos | 0.909 / 0.907 / 0.904 | | 0.891 / 0.899 / 0.899 | |

The pose points were not the bottleneck: they gave a face more often than the detector, and the detector's faces
separate 감스트 no better. Where neither gives a face on 맨까송 (64 of 244 sampled 감스트 boxes) the face is turned away or
a few pixels wide. Enlarging the face with a Lanczos warp instead of a bilinear one changed nothing (AUC 0.899 /
0.904); a learned super-resolution was not tried.

## 3. Settings (replay, in-sample and held out)

Frame precision / recall (%) of 감스트.

| setting (replay) | 관제탑 all | 맨까송 all | 관제탑 odd | 맨까송 odd | 관제탑 even | 맨까송 even |
|---|---|---|---|---|---|---|
| as before (pose points, 0.35 / 0.15, the first contradiction releases) | 99.8 / 69.1 | 77.3 / 57.6 | 100.0 / 64.0 | 75.8 / 57.2 | 99.6 / 72.9 | 79.2 / 57.9 |
| SCRFD face only | 99.5 / 69.5 | 76.8 / 60.8 | 100.0 / 67.3 | 76.1 / 60.8 | 99.2 / 71.1 | 77.8 / 60.6 |
| face threshold 0.20 only | 99.8 / 68.6 | 79.0 / 66.4 | 100.0 / 63.4 | 78.8 / 67.1 | 99.6 / 72.5 | 79.3 / 65.4 |
| reject 0.10 only | 99.8 / 69.7 | 77.7 / 59.0 | 100.0 / 64.7 | 75.8 / 57.2 | 99.7 / 73.4 | 80.0 / 61.1 |
| reject_after 2 only | 99.8 / 69.1 | 77.7 / 59.0 | 100.0 / 63.3 | 75.8 / 57.2 | 99.7 / 73.5 | 80.0 / 61.1 |
| scene_threshold 0.8 only | 99.8 / 69.1 | 77.3 / 57.6 | 100.0 / 64.0 | 75.8 / 57.2 | 99.6 / 72.9 | 79.2 / 57.9 |
| appearance 0.85 only | 99.8 / 69.1 | 77.3 / 57.6 | 100.0 / 64.0 | 75.8 / 57.2 | 99.6 / 72.9 | 79.2 / 57.9 |
| **chosen: SCRFD, 0.15 / 0.10, reject_after 2, appearance 0.85, reach_back** | 99.6 / 76.6 | 78.6 / 68.5 | 100.0 / 72.6 | 78.0 / 66.7 | 99.3 / 79.6 | 79.4 / 70.5 |
| the same without reach_back | 99.6 / 74.1 | 78.5 / 67.8 | 100.0 / 70.2 | 77.8 / 65.8 | 99.3 / 77.0 | 79.3 / 70.1 |
| pose points, 0.15 / 0.10, reject_after 3, appearance 0.80, reach_back (best without the detector) | 99.1 / 74.5 | 79.8 / 69.3 | 100.0 / 69.8 | 79.1 / 68.5 | 98.6 / 78.0 | 80.6 / 70.0 |

- **Picked on the odd scenes** (both clips): SCRFD, 0.15 / 0.10, reject_after 2, appearance 0.85, reach_back = the
  chosen setting. On the even scenes (held out): 관제탑 99.3 / 79.6 (before 99.6 / 72.9), 맨까송 79.4 / 70.5 (before
  79.2 / 57.9); mean frame F1 0.815 (before 0.755).
- **Picked on the even scenes:** the same with appearance 0.80. On the odd scenes (held out): 관제탑 100.0 / 71.4
  (before 100.0 / 64.0), 맨까송 78.0 / 66.7 (before 75.8 / 57.2); mean F1 0.776 (before 0.716).
- **In-sample best (all scenes):** the chosen setting, mean F1 0.799 (before 0.738).
- Each change alone moves little; they work together: with the detector fewer blurred faces contradict, so a low face
  threshold binds without the odd faces undoing it. Below 0.15 the face threshold lost recall on 관제탑 (without
  reach_back: 0.15 → 74.1 %, 0.12 → 72.5 %, 0.10 → 72.8 %).
- Without the detector the best setting is close (in-sample mean F1 0.796 vs 0.799); see section 5 for the real runs.

## 4. Real runs (post mode): identity

Base = the settings before this branch, recorded again with this branch's dump (which only adds the face points and
the binding's scores): recall 70.0 % and 53.1 % here, 69.9 % and 52.7 % in #379's run (the binding's scorer runs beside
the lane, so two runs differ by a few frames). **Chosen** = SCRFD, face 0.15 / reject 0.10, `reject_after=2`,
appearance 0.85, `reach_back`, and post mode waiting for the scores (now the post default). Also run: the same settings
without waiting (as post mode was before), and the best setting without the detector (pose points, 0.15 / 0.10,
`reject_after=3`, appearance 0.80, `reach_back`, not waiting).

| frames of 감스트, precision / recall | base | chosen | chosen, not waiting | without the detector, not waiting |
|---|---|---|---|---|
| 관제탑 all | 99.6 / 70.0 | **99.8 / 77.1** | 99.3 / 75.2 | 99.1 / 73.1 |
| 관제탑 even scenes (picked on odd: held out) | 99.2 / 73.9 | 99.6 / 79.5 | 98.9 / 77.8 | 98.6 / 76.7 |
| 관제탑 odd scenes (in-sample) | 100 / 64.7 | 100 / 73.9 | 100 / 71.7 | 100 / 68.2 |
| 관제탑 without frames naming two separate places 감스트 | 99.5 / 85.4 | 99.7 / 93.6 | 99.2 / 92.2 | 98.9 / 88.6 |
| 맨까송 all | 82.0 / 53.1 | **77.8 / 64.7** | 81.4 / 62.0 | 81.7 / 67.2 |
| 맨까송 even scenes (held out) | 85.2 / 51.5 | 78.5 / 68.2 | 86.4 / 64.4 | 82.0 / 69.5 |
| 맨까송 odd scenes (in-sample) | 79.7 / 54.4 | 77.1 / 61.8 | 77.5 / 60.0 | 81.4 / 65.2 |
| 맨까송 without frames naming two separate places 감스트 | 80.8 / 61.2 | 75.7 / 71.8 | 79.4 / 67.9 | 79.8 / 74.4 |
| 맨까송 without the segment labelled "other" whose note says 감스트 (42.0-63.5 s) | 99.7 / 53.1 | 98.0 / 64.7 | 98.8 / 62.0 | 98.9 / 67.2 |
| 관제탑 face results / jobs / replaced / stale | 1,448 / 2,038 / 161 / 0 | 1,219 / 2,086 / 0 / 0 | 1,172 / 2,364 / 283 / 2 | 1,464 / 2,124 / 167 / 0 |
| 관제탑 decision delay p50 / max (ms) | 100 / 500 | 33 / 33 | 167 / 900 | 100 / 500 |
| 맨까송 face results / jobs / replaced / stale | 216 / 328 / 37 / 5 | 180 / 329 / 0 / 4 | 173 / 396 / 75 / 15 | 204 / 318 / 42 / 6 |
| 관제탑 run time (522 s clip) | 10.0 min | 10.6 min | 10.3 min | 10.2 min |

- 맨까송's lower precision is that one segment: without it precision is 98-99.7 % in every run; the more 감스트 is
  named, the more of its frames count as wrong.
- Not waiting, the detector's slower face jobs (decision delay p50 167 ms on the CPU) were replaced before they ran
  (관제탑 161 → 283, 맨까송 37 → 75), so the run got less than its replay (맨까송 recall 62.0 % vs 68.5 %). Waiting,
  every job came back (decision at the next frame, 33 ms) for 0.6 min more on 관제탑.
- Without the detector 맨까송's identity recall is higher (67.2 %), but see section 5: the punches were not.

Causes after (chosen run, rules of section 1): 관제탑 another 감스트 box named 2,903 (2,523 before), lost the claim
528 (662), another person 443 (814), no face scored 78 (356), below the threshold 49 (348), withdrawn by a face 24
(571); 맨까송 another 감스트 box named 201 (97), no face scored 151 (111), lost the claim 81 (140), no box 72 (179),
below the threshold 64 (284), another person 35 (1), withdrawn 34 (32), too small 8 (16).

## 5. End effect: 감스트's action count (#379's best movements)

감스트's matched / events / truth, F1, and the occurrences missed because the event was given to someone else.

| movement | base | chosen | chosen, not waiting | without the detector |
|---|---|---|---|---|
| 관제탑 rule + examples, search | 36 / 43 / 39, 0.878, 3 | 37 / 48 / 39, 0.851, 2 | 39 / 48 / 39, **0.897**, 0 | 38 / 47 / 39, 0.884, 1 |
| 관제탑 rule + examples, count | 29 / 31 / 39, 0.829, 6 | 31 / 33 / 39, 0.861, 4 | 32 / 34 / 39, **0.877**, 3 | 32 / 34 / 39, 0.877, 3 |
| 맨까송 punch rule | 6 / 10 / 16, 0.462, 7 | 9 / 14 / 16, 0.600, 3 | 9 / 13 / 16, **0.621**, 3 | 6 / 12 / 16, 0.429, 7 |
| odd / even scenes F1: search | 0.930 / 0.821 | 0.889 / 0.809 | 0.909 / 0.837 | 0.909 / 0.857 |
| odd / even F1: count | 0.895 / 0.750 | 0.923 / 0.788 | 0.923 / 0.824 | 0.923 / 0.824 |
| odd / even F1: punch | 0.154 / 0.615 | 0.571 / 0.500 | 0.571 / 0.533 | 0.154 / 0.533 |

- The two occurrences the chosen run gives to someone else on 관제탑 (370.8-372.4 s) are inside 331-420 s, where the
  truth labels two different people 감스트 at the same time. Every run with the new settings has one wrong 감스트 event
  more at 516.9 s, where 감스트 and two others do it together: his track (labelled 감스트, face scores 0.2-0.3, named only
  with the new thresholds) gave three events for his two occurrences, and the breakdown calls the third "identity"
  because another person's occurrence there is unmatched.
- On 맨까송 the punch scene (5-9 s, 감스트 in profile hitting the chair) needed the detector: 3 punches given to
  someone else with it, 7 without.
- Search mode on 관제탑 found all 39 of 감스트's occurrences in the run that did not wait.

## 6. What was built

- `gamst/providers/face_detect.py`: SCRFD decoding (the official InsightFace rules), the face that covers the head
  region's centre, ArcFace alignment from its five points, `DetectedFaceEmbedder` (LVFace's `model_id`, so references
  stay valid). The face worker runs it when its config names `detector_path` + `detector_sha256`; registry candidate
  `lvface.b_scrfd` (not automatic: the weight is inside InsightFace's `buffalo_l.zip`, sha256
  `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f`; `det_10g.onnx` 16,923,827 bytes, sha256
  `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`). Runs on this Mac on the CPU (87 ms a 640 x 640
  frame) and CoreML (24 ms), the same faces.
- `person_match.head_region` / `face_crop`: what a face model is given (aligned face, or the head region).
- `PersonBinding` options: `reject_after`, `reach_back`, `scene_threshold` (within-scene appearance propagation)
  default to the earlier behaviour; `wait_for_scores` is on for `--timing post` (the recipe passes the timing) and off
  in realtime.
- `gamst.eval.binding_replay`: the binding over a recorded dump with cached evidence; `pose_dump` keeps the face points
  and the binding's per-track scores.
- Thresholds stay provisional and per model: these are LVFace-B and PersonViT on two broadcast clips with a reference
  enrolled from a third.
