# Realtime vs post on the owner's truth: why realtime lost accuracy, and what changed (2026-11-28)

Measured on this Mac (Apple silicon, CoreML) with branch `fix/realtime-post-gap` (PR #384) against the owner's truth v2
(`docs/ground-truth/2026-11-28/v2/`), scored as in PR #383 (`gamst.eval.multi_person`, 0.5 s, the example ranges left
out). Configuration of every run: Main's chosen one (RTMW-l 384x288 + YOLOX-m, SCRFD + LVFace + PersonViT person binding,
face 0.15 / reject 0.10, reject_after 2, body 0.85, reach_back, `감스트.person.json`); 관제탑 the rule + 7 examples in
search mode (`gwan_search.json`), 맨까송 the punch rule (`man_best.json`). Every run went through `try-it/guard.sh` with
`GAMST_SPILL_FLOOR_GB=5`; free disk was checked before and after each (9.5-11 GB). Every number below is measured; one
run per row unless said otherwise.

## 요약 (한국어)

- **무엇이 문제였나 (맨까송, 프레임 손실 0인데 F1 0.38 vs 0.60):** 동작을 세는 것은 같았습니다(모두 F1 post 0.57, 실시간 0.49). 다른 것은 **누가** 했는지입니다(감스트 맞음 9 vs 4).
  1. **사람 연결(누구인지)이 L2 구간에서 잘려 나갔습니다.** 파이프라인이 사람 연결 이벤트를 "개수"처럼 L2가 맡은 시간 범위로 잘랐습니다. 그래서 L2 작업자가 예열(pre-roll) 동안 넘겨받거나 얼굴로 찾은 연결이 카운터와 저장소에 닿지 않았습니다. 이전 실행에서 L2 29번 중 **넘겨받기(handover) 0번**이었습니다.
  2. **100 ms 중복 제거가 사람을 보지 않았습니다.** 두 사람의 펀치(또는 연결 이벤트)가 다른 레인에서 100 ms 안에 오면 하나로 합쳤습니다.
  3. **L2 작업자는 얼굴 점수를 기다리지 않았습니다.** 라이브 작업이 먼저라 L2의 얼굴 점수는 1초를 넘겨 버려졌습니다.
  4. **L1/L2 경계마다 상태가 끊깁니다.** 이 Mac에서 실시간은 처리 능력을 넘습니다(라이브가 2,013 프레임 중 954만 처리, L2 29번). 경계마다 새 추적기·새 연결 상태로 시작하므로, post에서 7.6초에 얼굴로 정해져 5.1초까지 거슬러 올라가는 연결이 실시간에서는 다른 레인에 떨어져 거슬러 올라가지 못했습니다. 놓친 감스트 펀치는 모두 경계에서 2.5초 안이었습니다(경계가 1.5–3초마다 있음).
  5. **라이브 레인 자신도 얼굴 점수를 버립니다**(278개 중 오래됨 44, 교체 61). post는 0입니다.
- **고친 것:** (1) 사람 연결은 레인마다 통째로 둠(`events_by_lane`), (2) 중복 제거는 사람별, (3) L2는 자기 얼굴 점수를 기다림, (4) 넘겨받기는 끊긴 바로 그 프레임에서, (5) L2가 이어받은 트랙을 처음 연결하면 라이브 쪽 같은 사람 트랙의 끊기 전 구간까지 거슬러 올라감(`by: l2`).
- **저장(관제탑, 15,670 중 6,777 프레임 잃음):** 파일을 실시간으로 다시 볼 때는 메모리(256 MB)를 넘는 프레임을 **위치와 CRC만 기억**하고, L2가 그 프레임에 오면 파일에서 다시 읽습니다. 디스크 0 바이트입니다. 카메라 프레임은 PNG(무손실)로 압축해 저장합니다.
- **결과 (전 → 후):**

  | | 관제탑 search | 맨까송 punch |
  |---|---|---|
  | post (같은 설정, 이 코드) | 0.860 (37/47/39) | 0.600 (9/14/16) |
  | 실시간 전 (#383) | 0.610 (#383 채점) / **0.408** (저장소 기준), **6,777 프레임 잃음** | 0.381 / 0.381 |
  | 실시간 후 (기본값) | **0.747**, **0 잃음**, 디스크 0 B | 0.261 / 0.273 (2회) |
  | 실시간 후 + 라이브도 점수 기다림 + L2 넘김 10초 | **0.837** (36/47/39), 0 잃음 | **0.538** (7/10/16) |

  - **채점 주의:** #383의 실시간 채점은 기록 파일의 모든 사건(L2 예열, L2가 맡은 범위의 L1 사건, 두 레인의 같은 사건)을 셌습니다. 저장소가 실제로 남긴 사건으로 세면 #383 관제탑 실시간은 0.610이 아니라 **0.408**(10/10/39)입니다. 아래 "후" 숫자는 모두 저장소 기준입니다(post는 둘이 같음).
  - 관제탑 post 목표는 0.89가 아니라 **0.860**입니다. #383의 0.889는 이 실행과 다른 연결 설정(identity-recall 전)의 post입니다. 같은 설정의 post는 identity-recall 문서도 0.860입니다.
  - **목표(실시간 = post)는 완전히 이루지 못했습니다.** 가장 가까운 것은 "라이브도 점수를 기다리고, L2로 넘기는 기준을 1초에서 10초로" 한 실행입니다: 관제탑 **0.837 vs 0.860**(맞음 36 vs 37, 말함 47 vs 47), 맨까송 0.538 vs 0.600(7 vs 9).
  - 그 대가는 지연입니다. L1(바로 센 것) p50 / 최대: 관제탑 8.4초 / 20.1초, 맨까송 2.9초 / 9.5초. L2 보정: 관제탑 p50 282초, 최대 437초; 맨까송 32초.
  - 기본값(라이브는 기다리지 않음, 1초에 넘김)은 바꾸지 않았습니다. L1 지연을 얼마까지 받을지는 사장님 결정입니다.
- **결론:** 이 Mac에서는 실시간이 처리 능력을 넘고(맨까송 RTF 2.5, 관제탑 4.4), 결정 41의 "L1 확정 + L2 구간 보정" 구조는 경계마다 상태가 끊겨 post와 같은 수가 나올 수 없습니다. 같은 수를 보장하려면 **끊기지 않는 한 줄(post와 같은 순서로 모든 프레임을 처리하는 확정 레인) + L1은 잠정(나중에 확정/취소)** 구조가 필요합니다. 저장소에 이미 잠정→확정/취소 상태(`event.revision`)가 있습니다. 결정 41을 바꾸는 일이라 사장님 결정이 필요합니다.

## 1. Causes, with evidence (맨까송, the #383 realtime run, `man_punch_rt`)

| what | post (`man.CW` replayed, = this branch's post) | realtime #383 |
|---|---|---|
| everyone F1 (matched / said) | 0.57 (12 / 26) | 0.49 (11 / 29) |
| 감스트 matched / said | 9 / 14 | 4 / 5 |
| frames the live lane processed | 2,013 | 954 (29 L2 intervals, 384 ms pre-roll each, up to 15 pending) |
| `handover` bindings in L2 forks | — | **0** of 29 forks |
| face jobs: stale / replaced / total (live module) | 4 / 0 / 329 | 53 / 54 / 224 |

- **Binding events trimmed like counts.** `Pipeline._commit` trimmed every module's events to the lane's owned range
  by end time (the L2 rule for counts). For `person_binding` that removed a fork's handover and face bindings made in
  its pre-roll (never seen by the fork's counter, never stored), and the live lane's bindings inside a range an L2
  interval had claimed. Reproduced with stand-ins: `tests/test_realtime_pose.py::LaneStateCatchUp` (with the old
  behaviour as a control, L2's hits are nobody's).
- **The 100 ms safety net keyed by `person`**, which neither punch nor binding events carry: two people's events from
  two lanes within 100 ms were one (`SafetyNetByPerson`).
- **L2 forks did not wait for scores**: their jobs go after the live lane's and were dropped as older than 1 s.
- **Not causes:** track ids renamed across forks (the fork's binding and counter both name tracks `<track>@<from_ms>`,
  checked on the run); the pre-roll length for the punch rule (the rule holds 34 ms; the pre-roll is 384 ms); hand
  raw-score gates (L2 poses carry `raw_score` through the same `committed()` path).
- **Boundaries (after the fixes).** For each 감스트 punch post matched and realtime missed, the distance to the nearest
  L1/L2 boundary (`lanes.py` in the scratch folder):

  | run | intervals | time owned by L2 | both matched | post only (L2 / L1 moment) | missed: s to nearest boundary |
  |---|---|---|---|---|---|
  | `man_rt` (fixes, defaults) | 20 | 45 % | 3 | 1 / 5 | 1.4, 0.9, 0.4, 0.4, 2.2, 2.5 |
  | `man_rt2` (+ reach-back) | 22 | 49 % | 3 | 4 / 2 | 0.4, 0.5, 0.0, 0.8, 2.2, 0.6 |
  | `man_rtw` (+ live waits for scores) | 35 | 85 % | 5 | 1 / 3 | 0.6, 0.5, 1.3, 1.4 |
  | `man_rtlag` (L2 after 10 s, live does not wait) | 2 | 31 % | 3 | 2 / 4 | 22.0, 21.6, 21.1, 20.3, 0.6, 5.2 |
  | `man_rtwlag` (live waits + L2 after 10 s) | 2 | 32 % | 6 | 0 / 3 | 18.6, 2.2, 2.5 |

  - With a cut every 1.5-3 s every miss is near a boundary. With only 2 cuts the live lane's own misses remain (its
    face scores are dropped: `man_rtlag`); waiting for them as well closes most of the gap (`man_rtwlag`).
  - Example (`man_rt`, 5.07-8.3 s): post binds track person-2 to 감스트 at 7.64 s (face 0.715), reaching back to 5.07 s;
    realtime's interval 6.74-8.49 s puts that face in a fork, whose binding then reached back only to the fork's
    start, so the live lane's punches at 5.3-6.6 s stayed nobody's. `by: l2` reach-back fixes exactly this case (seen
    in `man_rt2` at 0.2-3.24 s), but the next boundary breaks the same way elsewhere.

## 2. Fixes (code)

- `gamst/providers/person_binding.py`: `events_by_lane = True` (the Pipeline keeps the events of such a module whole in
  every lane); forks `wait_for_scores` for their own jobs only (`_Scorer.waiting` counts the job in flight); the
  handover happens at the cut's own frame (`snapshot()['at_ms']`) and never to an untracked pose; a fork's first own
  binding of a track that continues a live track nobody held at the cut reaches back over that live track (`by: l2`, a
  closed binding with `until_ms`, floored by the person's earlier bindings and by a face of that track that said
  someone else; `intervals_of` keeps it beside the track's other bindings).
- `gamst/pipeline.py`: lane-state events skip the owned-range trim and the safety net; the safety net keys by
  `person`, else `person_id`, else `who`.
- `gamst/providers/video_memory.py`, `gamst/inputs.py`: frames beyond the memory keep their file position
  (`VideoFileInput.position_of`, `rereader()` -> `FileFrames`) or are PNG-compressed (camera), see 3.

## 3. Backlog storage

| | 관제탑 realtime before (#383) | 관제탑 realtime after (`gwan_rt`) |
|---|---|---|
| frames lost | **6,777 / 15,670** (spill bound 0.8 GB) | **0 / 15,670** |
| spilled to disk | 0.8 GB (full) | **0 B** |
| frames kept by position (re-read) | — | 34,906 (frames of the lane and of L2 intervals; 96.5 GB of pixels not stored) |
| most waiting at once | — | 28,664 frames (79 GB of pixels; 86 % of the clip); 6,419 with live waiting + 10 s lag |
| re-read mismatches (CRC) | — | 0 of 41,356 reads |
| re-read cost | — | 1.19 ms per read on average (49.2 s in total, seeks included, beside the event loop) |

맨까송 realtime: 951 frames kept by position, 0 mismatches in 989 reads (0.93 ms each), 0 B spilled.

- **Seeking is exact on these files:** 200 random OpenCV seeks per clip, 0 frames differing from the sequential decode;
  seek 8 ms (360p) / 18 ms (720p), in-order re-read 0.47 ms per 720p frame. A CRC-32 costs 0.08 ms per 720p frame. A
  mismatch would decode again from the file's start; a frame that still differs is a declared gap, never another frame.
- **RAM per waiting frame kept by position:** about 0.9 KB (tracemalloc, 10,000 frames): an hour of 30 fps is ~93 MB.
- **Live cameras** (no position): lossless compression on a worker thread, measured on every 15th frame of the clips
  (one thread):

  | 720p30 backlog | ratio | KB per frame | encode / decode ms | minutes per GB |
  |---|---|---|---|---|
  | raw (before) | 1 | 2,700 | — | 0.22 |
  | PNG level 1 (default), 관제탑 frames | 5.94 | 455 | 22.1 / 9.4 | 1.3 |
  | zlib level 1, 관제탑 frames | 4.75 | 569 | 13.1 / 2.9 | 1.0 |
  | PNG level 1, the same frames + Gaussian noise σ=2 (a camera stand-in, not a camera) | 1.88 | 1,434 | 38.9 / 15.2 | 0.4 |
  | position of a seekable file | ∞ | 0 on disk | 1.19 ms per re-read | unbounded by disk |

  360p (맨까송) PNG-1: ratio 2.84, 8.0 / 3.4 ms. Decoded H.264 is smoother than a sensor, so a real camera's ratio is
  likely nearer the noisy row; no camera was measured. PNG at 22 ms per 720p frame is 66 % of one core at 30 fps, on
  its own threads (two by default), not on the event loop.

## 4. Realtime vs post, after

| run | F1 감스트 (matched / said / truth) | everyone F1 | lost | L2 intervals | L1 latency p50 / max | L2 latency p50 / max | wall (RTF) |
|---|---|---|---|---|---|---|---|
| 맨까송 post (control, this code) | **0.600** (9 / 14 / 16) | 0.571 | 0 | — | — | — | 120 s (1.78) |
| 맨까송 realtime #383 (before) | 0.381 (4 / 5 / 16) | 0.439 | 0 | 29 | 0.94 / 1.32 s | 29.9 / 39.2 s | 135 s (2.00) |
| 맨까송 realtime, fixes | 0.261 (3 / 7 / 16) | 0.513 | 0 | 20 | 0.99 / 1.30 s | 62.0 / 66.0 s | 173 s (2.57) |
| 맨까송 realtime, fixes + reach-back | 0.273 (3 / 6 / 16) | 0.611 | 0 | 22 | 0.61 / 1.15 s | 26.6 / 76.2 s | 170 s (2.53) |
| ... + live waits for scores | 0.370 (5 / 11 / 16) | 0.465 | 0 | 35 | 1.17 / 1.66 s | 92.5 / 128.6 s | 211 s (3.14) |
| ... + L2 after 10 s (live does not wait) | 0.273 (3 / 6 / 16) | 0.600 | 0 | 2 | 2.24 / 10.20 s | 37.0 / 37.3 s | 110 s (1.64) |
| ... + live waits + L2 after 10 s | **0.538** (7 / 10 / 16) | 0.556 | 0 | 2 | 2.92 / 9.48 s | 32.2 / 32.6 s | 117 s (1.74) |
| 관제탑 post (control, this code) | **0.860** (37 / 47 / 39) | 0.800 | 0 | — | — | — | 655 s (1.25) |
| 관제탑 realtime #383 (before) | 0.408 (10 / 10 / 39); #383's scorer: 0.610 (18 / 20) | 0.735 | **6,777** | 76 | 0.99 / 1.51 s | 21.5 / 25.6 s | 563 s (1.08) |
| 관제탑 realtime, fixes + reach-back | 0.747 (28 / 36 / 39) | 0.683 | **0** | 185 | — (6 L1 events) | 1,311 / 1,733 s | 2,291 s (4.38) |
| 관제탑 realtime, + live waits + L2 after 10 s | **0.837** (36 / 47 / 39) | 0.773 | **0** | 22 | 8.42 / 20.1 s | 282 / 437 s | 944 s (1.81) |

- F1 is of what the Store kept (`event_states`, who from the stored bindings). #383 scored every event of the pose dump,
  which in realtime also holds events the Store dropped (an L2 pre-roll's, the live lane's inside a range L2 owns, one
  event seen by two lanes): on #383's 관제탑 run that gave 0.610 for 132 events, the Store kept 60 (0.408). On 맨까송 the
  two agree (0.381). Post: the same in both.
- Latency = when the Store had the count (`committed_ms`, the `--realtime` clock shift removed) minus the event's own
  end, over every counted event of the lane (L1: final at once; L2: `decided_by: catch_up`).
- Run-to-run spread is large in realtime: the same configuration gave 0.261 and 0.273 on 맨까송, and the boundaries fall
  differently every run. Post gave the same events as Main's post runs.
- 관제탑 after the fixes: forks wait for their scores, so L2 is slower; 185 intervals covered 481.6 of 522 s, up to 161
  were pending, and corrections came up to 29 minutes late. Accuracy rose (0.61 -> 0.75, nothing lost), latency became
  unusable for a live view.

## 5. What would make realtime equal post

Realtime here exceeds the machine (RTMW-l + YOLOX-m + SCRFD/LVFace/PersonViT for up to 4 people: post itself runs at
RTF 1.25-1.78). In decision 41's design the live lane's counts are final and L2 corrects intervals, each a fresh chain
of forks; every boundary restarts the tracker, the identity state and the scorer's timing, and post has none. The
fixes above make each boundary lose less, but the number of boundaries grows exactly when the machine is short, and the
count then depends on where they fall. A design that gives post's count by construction:

- one **settling lane** that never jumps: every frame in capture order, the person binding waiting for its scores (what
  post does), late when the machine is short;
- an **L1 preview** (a fork that jumps to the newest frame while the settling lane is behind) whose counts are
  `state: provisional`; the settling lane then marks each `final` or `revoked` and adds what L1 missed (the Store's
  `event.revision`, `gamst/counting.py`, already used for words; the HUD already draws "-1 취소" and "+1 추가").

This changes decision 41 ("live counts are final; nothing is revoked") and is not implemented here. Until then the
measured trade-off is the "live waits + L2 after 10 s" rows: `--set person_binding.wait_for_scores=true` and
`catch_up.lag_ms` 10000 give 관제탑 0.837 vs post 0.860 and 맨까송 0.538 vs 0.600, at an L1 latency of p50 8.4 / 2.9 s,
max 20.1 / 9.5 s, and L2 corrections p50 282 / 32 s.

## Commands

- Runs: `scratchpad/rtgap/rt_run.sh NAME gwan|man MOVEMENT realtime|post` (as #383's `gamst_rt.sh` with this branch's
  code; refuses to start below the floor + 300 MB), via `try-it/guard.sh`. Extra settings by `EXTRA="--set ..."`.
- Scores: `scratchpad/rtgap/score.py NAME CLIP` (#383's `gamst_rt_score.py` plus what the Store kept: the
  `event_states` events with who from the stored bindings; on #383's run both give 0.381). Outputs:
  `try-it/testbed/rtgap/` (not in the repository).
