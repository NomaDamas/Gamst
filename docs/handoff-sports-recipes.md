# HANDOFF feat/sports-recipes

Worktree `.local-run/wt-sports`, branched from Main `feat/humansense-20261112-154922` at `e137433`. No PR yet.

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

## 4. Proposal sent to the owner (not yet approved)

See the session briefing; recorded here once the owner decides.
