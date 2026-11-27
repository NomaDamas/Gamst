# Sports: generic parts, recipes per sport, stacks per vendor

Status (2026-11-28): implemented on branch `feat/sports-recipes` (stacked on PR #373). Every part is tested on
synthetic data only (`tests/test_sports.py`); **no accuracy on real sports footage has been measured yet**. The
dataset runs are the next stage, on GCP (section 9).

## 요약 (한국어)

- **구조:** 스포츠 전용 엔진은 만들지 않았다. 범용 부품을 GAMST에 넣고, 종목은 레시피(JSON)로 정의한다. 에이전트는 종목마다
  미리 만든 레시피를 쓰거나, 같은 부품으로 직접 조립한다(`skills/gamst-sports/SKILL.md`).
- **레시피 × 스택 × 타이밍:**
  - 레시피는 부품을 어떻게 잇는지다.
  - 스택은 각 부품 자리에 누구의 모델을 넣는지다. `ultralytics`는 Ultralytics 자사 조합, `roboflow`는 Roboflow 조합, `sota`는
    GAMST가 고른 조합이다.
  - 타이밍(`realtime`·`post`)은 같은 스택 안에서 모델 크기와 처리 방식을 바꾼다.
  - `gamst recipe setup 레시피 --stack 이름`은 그 스택에 필요한 모델만 설치한다.
- **레시피 6개:** `racket-rally`(테니스·배드민턴·탁구·패델·스쿼시), `team-field`(축구·농구·풋살·핸드볼·하키·배구), `body-reps`(피트니스·줄넘기·로잉·복싱),
  `swing`(골프·야구·테니스 스트로크), `body-rotation`(피겨·다이빙·체조, 실험적), `billiards`.
- **철학:** LLM은 프레임을 보지 않는다. 공이 꺾인 순간(타구·바운스), 랠리, 소유 전환, 위치·이동 거리, 기준 동작과의 차이를 시각과 근거가
  붙은 기록으로 받는다. 1 fps로는 볼 수 없는 수 ms짜리 순간을 GAMST가 모든 프레임에서 잡는다.
- **측정:** 합성 데이터 테스트만 했다. 실제 영상 정확도는 측정하지 않았다(GCP 단계).

## 1. Where it sits in GAMST

```
inputs        camera · video · take · microphone                                (unchanged)
references    person / movement / words                                        (unchanged, #373)
              + court: the four corners the camera sees (--court), a sport preset (--sport), a demonstrated movement
recipe        sports recipe (parts linked by consumes)  ×  stack (models)  ×  timing (realtime | post)
pipeline      perception lanes (one model per role in a run, every frame)
                detect · ball-model · object-track · yolo26-pose / rtmw-pose · team-split · audio-onsets
              computation (no model: physics, geometry, counting)
                ball-track · contacts · rally · possession · occupancy · form · body-spin · object-motion
              realtime: L1 + L2 correction where a part can fork (#373)   post: every frame, no correction
context       sports.contacts · sports.rally · sports.possession · sports.occupancy · sports.form · sports.spin ·
              sports.motion (+ the existing temporal.exemplar_events, temporal.repetition_count, temporal.coincidence)
MCP → LLM     count_events · get_timeline · get_observations · create_watch (unchanged)
```

Nothing new is needed on the MCP side: every sports module commits events in the usual shape (`events` with `event`,
`timestamp_ms` or `start_ms`/`end_ms`), so `count_events('sports.contacts', event='contact.hit')` counts hits and
`create_watch` can fire on `rally.ended`.

## 2. Parts

### Perception (models, each in its own environment)

| part | role (registry) | candidates | output |
|---|---|---|---|
| `detect` | `vision.detect` | `rfdetr.nano`, `rfdetr.large` (Apache-2.0), `yolo26.n/s/m/l/x` (AGPL-3.0, with Ultralytics' tracker) | `objects` |
| `ball-model` | `vision.ball` (new) | `tracknetv3.shuttlecock` (MIT; `ensemble` causal or weight) | `frames` of ball candidates |
| `object-track` | `vision.object_track` (new, in process) | `gamst.ocsort`; `trackers.2_7_1` (ByteTrack, BoT-SORT, OC-SORT, SORT, C-BIoU); `tracker: none` for a detector that tracks itself | `objects` with track IDs |
| `yolo26-pose` | `vision.pose` | `yolo26.pose_n/s/m/l/x` (17 COCO points) | joints (private), like RTMW |
| `team-split` / `team-split-siglip` | – / `vision.image_embedding` (new) | colour histogram (no model) / `siglip.base_patch16_224` (Apache-2.0) | `objects` with `team` |
| `audio-onsets` | – (existing) | – | `audio.onset` events |

### Computation (no model)

| part | what it computes | record |
|---|---|---|
| `ball-track` | one ball's flight: gating by a constant-velocity prediction, `ball.found` / `ball.lost` | `sports.ball` |
| `contacts` | breaks of the ball's path (turn ≥ `min_turn_deg` or speed ratio): **hit** near a player box (who, team), **bounce** (down→up, court position, in/out, half), **deflection**; impact sound beside each | `sports.contacts` |
| `rally` | hits until no contact for `max_gap_ms`, the ball lost, or two bounces on one half | `sports.rally` |
| `possession` | holder within `reach` for `hold_ms`; pass (same team), turnover (other team), loose; `estimated: true` | `sports.possession` |
| `occupancy` | positions (court metres with `--court`), distance, speed, heat-map grid per track and team, every `every_ms` | `sports.occupancy` |
| `form` | per counted movement: joint-angle ranges reached vs the demonstrated examples, DTW score | `sports.form` |
| `body-spin` | half turns from the sign of the shoulder/hip line; jumps (ankles off the ground line), airtime, turns in the air | `sports.spin` |
| `object-motion` | shots (first mover after stillness), object-to-object contacts | `sports.motion` |
| `ball-sampling` check | `gamst recipe check RECIPE --sport S --fps F`: samples between the fastest two contacts, travel per frame | – |
| court map | homography from four marked corners (`gamst/sports/court.py`, normalised DLT) | used by the parts |

## 3. Recipes

| recipe | sports (`--sport`) | parts | records |
|---|---|---|---|
| `racket-rally` | tennis (default), badminton, table-tennis, padel, squash | detect, object-track, ball-model, ball-track, audio-onsets (optional), contacts, rally, occupancy | contacts, rally, occupancy |
| `team-field` | football (default), futsal, basketball, handball, ice-hockey, volleyball | detect, object-track, team-split, ball-track, possession, contacts, occupancy | possession, contacts, occupancy |
| `body-reps` | fitness, rope skipping, rowing, boxing | pose, pose-channels, exemplar, repetition, form | exemplar events, repetitions, form |
| `swing` | golf, baseball, tennis strokes | pose, pose-channels, exemplar, audio-onsets, coincidence, form | exemplar events, impact coincidences, form |
| `body-rotation` | figure skating, diving, gymnastics, dance (experimental) | pose, body-spin | spin |
| `billiards` | pool, carom, snooker | detect, object-track, object-motion | motion |

A sport without a recipe is composed from the same parts: an agent writes a recipe file with these `use` names
(`gamst recipe run my-sport.json`). MCP `compose_recipe` (integrated design, #373) is not built yet.

## 4. Stacks: whose models fill the parts

Numbers are the authors' (benchmarks and hardware as they state them), **not measured by GAMST**.

| stack | detect | track | ball | pose | team | source of the claim |
|---|---|---|---|---|---|---|
| `ultralytics` | YOLO26 (realtime m, fallback s; post x) | Ultralytics BoT-SORT inside the detector | the detector's `sports ball` | YOLO26-pose (realtime m, fallback s; post x) | colour | Ultralytics sports page: YOLO + its trackers + pose; YOLO26-n/s/m COCO 40.9 / 48.6 / 53.1 mAP at 1.7 / 2.5 / 4.7 ms (T4 TensorRT); "sub-5 ms" with no model or hardware named |
| `roboflow` | RF-DETR (realtime large, fallback nano; post large) | trackers ByteTrack | the detector's `sports ball` | – (Roboflow publishes no pose model) | SigLIP + PCA + k-means | Roboflow Sports (MIT) logic; RF-DETR N/S/M/L COCO 48.4 / 53.0 / 54.7 / 56.5 (T4); trackers' SportsMOT HOTA ByteTrack 73.0 |
| `sota` (default) | RF-DETR large (fallback nano) | trackers BoT-SORT | TrackNetV3 (causal realtime, weighted ensemble post) | RTMW-l (fallback MediaPipe full) | colour | RF-DETR as above; BoT-SORT SportsMOT HOTA 73.8; TrackNetV3 shuttlecock F1 98.56; RTMW-l COCO-WholeBody 70.1 AP |

`gamst recipe stacks RECIPE` prints, per stack and timing, which part and candidate each model module uses. Ultralytics
publishes no sports-trained weights, so its stack runs COCO weights; the page's own figures cannot be reproduced as
stated, and the comparison is made on the same dataset clips with the same GAMST computation parts behind each stack.

## 5. Realtime and post

| | realtime | post |
|---|---|---|
| input | a camera, or a file given at its own pace (`--timing realtime`) | a finished file |
| model choice | at start, `detect` and `ball-model` items with `choose` are measured on synthetic frames; the candidate is kept if it gives the input's frame rate, else `choose.fallback`, and the run says so (`model_choice`); pose as #373 (`pose_choice`) | the stack's post candidate, every frame |
| ball | TrackNetV3 `causal`: the newest window's last heatmap, at once | `weight`: the authors' temporal ensemble over `seq_len` windows, answered `seq_len - 1` frames later (`frame_ms`) |
| tracking | online trackers | the same online trackers (CAMELTrack and offline re-linking are not wired) |
| team split | fit on the first 5 s | fit on 30 s (`fit_ms` per stack) |
| results | committed at once; the pose lane's L2 corrections arrive later (#373) | committed once |

`gamst recipe setup RECIPE --stack S` installs both timing variants and the realtime fallbacks (`--timing` for one).

## 6. Commands

```bash
gamst recipe list                                        # the sports recipes are among them
gamst recipe stacks racket-rally                         # which models each stack uses, realtime and post
gamst recipe check racket-rally --sport badminton --fps 30   # physics: exits 1, badminton needs 47 fps
gamst recipe setup racket-rally --stack ultralytics --dry-run
gamst recipe run racket-rally --stack sota --sport tennis --video match.mp4 --court court.json --timing post
gamst recipe run team-field --stack roboflow --camera 0 --court pitch.json
gamst recipe run body-reps --stack ultralytics --video squats.mp4 --exemplar squat.json
```

`court.json`: `{"court": "tennis.singles", "points": [[x, y], [x, y], [x, y], [x, y]]}`, the corners the camera sees in
the order far left, far right, near right, near left, normalised to 0..1 (or pixels with `"image_size": [w, h]`).
Courts: `gamst/sports/court.py` (`COURTS`).

## 7. What is not counted, and limits

- A sound alone never makes a contact; a ball no model sees makes none.
- Hit vs bounce is geometry (near a player box vs a down-to-up turn); a lob or a high bounce in a broadcast view can be
  confused. In/out is a point-in-polygon test of one camera's projection, not line calling.
- Passes and possession are estimates (`estimated: true`), never official statistics. Team letters are arbitrary.
- A ball in the air is off the court plane: its speed is in image units per second, never m/s.
- Broadcast cuts and replays are not detected (fixed camera assumed).
- TrackNetV3's released checkpoint is trained on badminton; it is used untrained for tennis and other balls.
- `body-rotation` is checked on synthetic joints only.

## 8. Licences

GAMST stays Apache-2.0. Ultralytics and YOLO26 weights are AGPL-3.0 or Ultralytics Enterprise: they run only in the
`ultralytics` environment that `--stack ultralytics` sets up (`gamst-sdk[sports-ultralytics]`). RF-DETR (nano, large),
trackers, SigLIP and RTMW are Apache-2.0; TrackNetV3 code and checkpoints MIT (its README); supervision MIT. The
TrackNetV3 network is rewritten in `gamst/providers/tracknet_ball.py` from its MIT `model.py` with attribution.
Not used: sn-gamestate and T-DEED (GPL-3.0), repositories with no licence (several tennis court detectors, TTNet,
Deep-EIoU), GolfDB (CC BY-NC).

## 9. Measurement plan (GCP stage)

| step | data (part of an existing dataset, local evaluation only, never redistributed) | metrics | compared |
|---|---|---|---|
| M1 racket | TrackNet tennis (hit/bounce per frame), OpenTTGames one test video (CC BY-NC-SA, bounce/net events), Shuttlecock Trajectory one match | ball F1 at 4 px, hit and bounce precision/recall, time error (ms), rally length, ms per frame, real-time factor | 3 stacks × realtime/post |
| M2 body | RepCount-A part (rep start/end), the owner's shotgun clip | count error, time error per occurrence | sota (RTMW) vs ultralytics (YOLO26-pose) |
| M3 team | SportsMOT sample (CC BY-NC 4.0), then SoccerNet (NDA) | HOTA, possession and estimated-pass precision/recall on 1-2 hand-counted minutes | 3 stacks |

Realtime numbers are measured on the owner's Mac only (realtime depends on the device); post numbers may run on GCP.
Downloads (weights, dataset parts) are listed with file name, source and size and approved before they happen.

## 10. Not built yet

WASB ball model (5 sports), TrackNetV3 InpaintNet (post), CAMELTrack and appearance ReID for BoT-SORT, MCP
`compose_recipe`, basketball shot zones, swing tempo, court keypoint models (the court is marked by hand), and
Ultralytics' own `solutions` classes (Heatmap, AIGym, SpeedEstimator) — GAMST's `occupancy`, `repetition` / `exemplar`
and `ball-track` compute the same quantities, so the Ultralytics stack swaps only the models.
