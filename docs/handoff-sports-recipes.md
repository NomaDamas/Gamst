# HANDOFF feat/sports-recipes

Worktree `.local-run/wt-sports`. Branched from Main at `e137433`, then rebased onto PR #373 (`feat/realtime-pose`,
`278b8d4`) on the owner's OK (2026-11-28): the PR is stacked on #373 and is retargeted to Main when #373 merges.

## 0. Current state (2026-11-28) — read first

- Owner decisions: build per the final plan; GCP for measurement; everything before GCP is done in this branch.
- Done: the parts, stacks, six recipes, adapters and docs below; `tests/test_sports.py` (39 tests, synthetic data,
  one real Pipeline + Store test). See `docs/sports.md`.
- Not done (needs downloads or GCP): any run with real weights or footage; M1-M3 measurements; pinning the TrackNetV3
  checkpoint digest (Google Drive zip, no published digest).
- Not built: WASB, InpaintNet, CAMELTrack, ReID for BoT-SORT, MCP `compose_recipe`, basketball shot zones, swing tempo.

| file | what |
|---|---|
| `gamst/sports/{court,presets,objects}.py` | homography from four marked corners, sport presets + `ball_sampling`, object contracts |
| `gamst/providers/sports_*.py` | ball, contacts, rally, possession, occupancy, team, form, spin, motion, track |
| `gamst/providers/ultralytics_sports.py` | YOLO26 detect + Ultralytics tracker (`ultralytics.objects`), YOLO26-pose (`ultralytics.pose`) |
| `gamst/providers/tracknet_ball.py` | TrackNetV3 network (from its MIT model.py), causal / weighted ensemble streaming |
| `gamst/providers/siglip_embedding.py` | SigLIP vision embedding of crops (`image.embedding`) |
| `gamst/crew/registry.py` | YOLO26 n..x and pose n..x (pinned from release digests), roles `vision.ball`, `vision.image_embedding`, `vision.object_track` |
| `gamst/recipes/perception_parts.py` | sports parts (`detect`, `ball-model`, `yolo26-pose`, ... `object-motion`) |
| `gamst/recipes/__init__.py` | `variant` (sport → stack → timing), `--stack`, `--sport`, `--court`, `stacks` action, `choose_models`, setup of both timings + fallbacks |
| `gamst/recipes/{racket-rally,team-field,body-reps,swing,body-rotation,billiards}.json` | the recipes |
| `skills/gamst-sports/SKILL.md`, `docs/sports.md` | agent recipe, the map |

Next step (GCP): owner installs `gcloud` and logs in; VM price stated first; list downloads (file, source, size) for the
owner's OK; run M1 (docs/sports.md section 9) per stack and timing; record numbers in
`docs/measurements/sports-*.md` and the recipes' `measured`.

## 1. State (2026-11-27)

- Reading done: the earlier sports discussion, README, AGENTS.md, positioning, runtime-and-recipes, recipe JSONs,
  `gamst/recipes/parts.py` + `perception_parts.py`, the MCP tool list.
- Research done (below). Nothing downloaded, nothing measured, no code yet.
- Waiting for the owner: first sport, test clip source, disk space, Ultralytics case scope.
- Disk: 3.9 GiB free on the data volume (100% used) at 2026-11-27. Ask before any download.

## 2. Already in GAMST that sports recipes reuse

| Need | Part / provider |
|---|---|
| Detection | `vision.detect`: RF-DETR nano/large (Apache-2.0, `detection` extra); YOLOE (ultralytics, AGPL, `yoloe` extra) |
| Tracking | OC-SORT (pure NumPy, `docs/ocsort.md`), IoU tracker |
| Line / zone / reversal events | `image-events` rules on `ImageTrajectories` and `rfdetr.trajectories` |
| Tennis ball crossing a line | `rfdetr.tennis_image` (`docs/tennis-image-task.md`), fixed camera, abstains outside its envelope |
| Pose | MediaPipe, RTMW (rtmlib venv), Sapiens2 (opt-in) |
| Movement vs a reference | `exemplar` (streaming DTW, `movement_match.py`) |
| Impact sound | `audio-onsets`, `coincidence` |
| Rotation | `rotation`, `flip`, `angular-motion`, `object-spin` |
| Scoreboard | RapidOCR (`ocr` extra) |

Missing: possession hand-off, pitch/court homography, team split, occupancy heat map, heat-map ball tracker,
`compose_recipe` (approved design, not in Main).

## 3. Research (gh api 2026-11-27; benchmark numbers are the authors', not ours)

| Component | Repo | Stars | Code licence | Weights licence | Benchmark (authors) |
|---|---|---|---|---|---|
| Ultralytics YOLO | ultralytics/ultralytics | 63,796 | AGPL-3.0 | AGPL / enterprise | page: "sub-5 ms" with no model or hardware |
| RF-DETR | roboflow/rf-detr | 10,180 | Apache-2.0 | Apache-2.0 (nano..large) | not re-checked |
| Roboflow Sports | roboflow/sports | 5,518 | MIT | soccer `.pt` are YOLOv8 (ultralytics) | none published |
| trackers | roboflow/trackers | 4,024 | Apache-2.0 | ReID weights nv | SportsMOT HOTA: ByteTrack 73.0, BoT-SORT+ReID 75.5, McByte 76.5 |
| TrackNetV3 | qaz812345/TrackNetV3 | 339 | MIT (file) | MIT (README states checkpoints) | Shuttlecock test F1 98.56 |
| WASB-SBDT | nttcom/WASB-SBDT | 212 | MIT | nv | F1@4px: tennis 95.6, badminton 93.1, soccer 88.2, volleyball 88.0, basketball 82.6 (1.5M params) |
| rtmlib (RTMPose/RTMO) | Tau-J/rtmlib | 705 | Apache-2.0 | OpenMMLab | RTMPose-m 75.8 COCO AP |
| E2E-Spot | jhong93/spot | 89 | BSD-3 | no licence | FS-Perf mAP@1 85.1 |
| CAMELTrack | TrackingLaboratory/CAMELTrack | 139 | Apache-2.0 | nv | SportsMOT HOTA 80.3 |

Not usable as shipped: roboflow/sports YOLOv8 weights (AGPL), sn-gamestate (GPL-3), T-DEED (GPL-3), Deep-EIoU,
TTNet and most tennis repos (no licence), GolfDB (CC BY-NC). PnLCalib weights are CC BY 4.0 but the code is GPL-2+.
No maintained, benchmarked pose-vs-reference OSS was found; GAMST's `exemplar` DTW covers it.

## 4. Owner direction (2026-11-27)

- Accepted: a sports category of generic parts; per sport, an agent composes the parts or a recipe pipeline is shipped.
  All sports proposed, framed as "what 1 fps misses" and efficiency.
- Use parts of existing datasets (no own filming for now). Local disk is short: free space or use GCP (not decided).
- Licence is not a selection criterion (general open source, not commercial): choose by measured accuracy.

## 5. Detector facts (authors' numbers, T4 TensorRT; not measured on this Mac)

| Model | COCO AP | T4 latency | Licence |
|---|---|---|---|
| YOLO26 n / s / m | 40.9 / 48.6 / 53.1 | 1.7 / 2.5 / 4.7 ms (CPU ONNX n 39 ms) | AGPL-3.0 |
| RF-DETR N / S / M / L | 48.4 / 53.0 / 54.7 / 56.5 | 2.3 / 3.5 / 4.4 / 6.8 ms | Apache-2.0 |
| D-FINE X (Obj365) | 59.3 | 12.9 ms | Apache-2.0 |
| SAM 3 | concept segmentation, not a COCO detector | ~30 ms per image on H200, 848M params | SAM License |

RF-DETR's README re-benchmarks YOLO with its own protocol (YOLO26-N 40.3); cross-README numbers are not equal footing.

## 6. Datasets with event labels (local evaluation only, never redistributed)

| Sport | Dataset | Licence | Events |
|---|---|---|---|
| Table tennis | OpenTTGames | CC BY-NC-SA 4.0, per-video download (~35 GB total) | bounce / net per frame |
| Tennis | TrackNet V1 | none stated | hit / bounce per frame |
| Badminton | ShuttleSet (+ TrackNetV2 videos) | nv | stroke times |
| Fitness | RepCount-A | none stated | rep start/end |
| Gymnastics | FineGym | annotations CC BY-NC 4.0, videos YouTube | element start/end |
| Diving | FineDiving | signed release agreement | step transitions |
| Soccer | SoccerNet | NDA | ball actions, tracking |
| Team MOT | SportsMOT | CC BY-NC 4.0 | player boxes + IDs |

## 7. Final plan v1 sent to the owner (2026-11-27, waiting for approval)

- Build on #373's integrated structure (timing realtime/post, one model per run chosen by a start benchmark, video
  memory, L2 correction): proposed to stack this branch on `feat/realtime-pose` (PR base = that branch).
- Recipe = sport graph of generic parts; **stack** = which candidate fills each role (`ultralytics`, `roboflow`, `sota`);
  `--timing realtime|post` picks the stack's variant; `recipe setup --stack X` installs only those models.
- Parts: perception (detect, ball, track, pose, court keypoints, team split, audio onsets) and computation (court-map,
  hits, bounce, rally, possession, occupancy, speed, form score, ball sampling check).
- Recipes: racket-rally, team-field, body-reps (+form), swing, body-rotation, billiards; sport presets on top.
- Measure racket first (TrackNet tennis, OpenTTGames, Shuttlecock): ball F1@4px, hit P/R and time error, rally length,
  ms/frame and real-time factor per stack. Realtime numbers on this Mac; post numbers may run on GCP.
- Open questions: stacking on #373, compute (disk/GCP), who builds `compose_recipe`, the name "stack".
