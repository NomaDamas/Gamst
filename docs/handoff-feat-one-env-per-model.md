# Handoff: feat/one-env-per-model

Worktree `.local-run/wt-crew`, branch `feat/one-env-per-model` from `0eee40a`, with `f7daab0` (feat/sota-modules and the
MediaPipe graph) merged in 23b14c2. Committed, not pushed, no PR.
Design: [crew.md](crew.md), [private-payloads.md](private-payloads.md). No environment was created, nothing was installed or downloaded, no model was run.

## Done
1. **Isolation.** `gamst recipe setup` and `gamst setup` share one rule, from the registry (`runs_in_host`): models go into
   `<prefix>/venvs/<environment>`; input/view roles and `in_process` candidates stay in the running interpreter.
   `--isolation current` is explicit. `_validate_plan` recomputes the environments and compares placement fields with the
   registry. Added beyond the request: role `view.hud` (`kind="view"`), because the HUD draws with Pillow in the Pipeline
   process and Pillow used to arrive only with YOLOE's packages.
2. **Coin flip.** `yoloe.26s` and `efficienttam.ti_512` share `yoloe-efficienttam`. `TakeFlips(python=...)` launches
   `refine` / `refine-server` with the profile's interpreter; different or missing interpreters ask for setup. Counting code
   and its parameters are untouched. `--efficienttam-checkpoint` carries the refiner's file, now in another directory than the
   detector's. `gamst demo rotation` uses the profile's environment when `--worker-python` is not given.
3. **Pose worker.** `mediapipe-pose` is `PoseWorkerPlugin`, the owner of a private `mediapipe.body.video` worker (VIDEO mode, no
   lane, every frame). It commits `scene.poses` without joints and passes the joints on the derived input `pose_joints`;
   `target_person` reads that and passes the enrolled person's on `target_joints`, which `pose_channels` reads. Each pose names
   its own image's size, so a crop is corrected with the crop's size. Public capability versions are unchanged.
4. **Weight root.** `~/.cache/gamst/models/<model>/<file>` (`GAMST_MODELS`); flat `--models` directories stay readable.
5. **Worker GAMST.** Checkout (editable, path + commit in the plan) / release pin / git pin. Version not bumped.
6. **OpenCV.** Test: no registry environment holds two cv2 distributions.

## Read before merging
- **Private payloads (owner/coordinator decision, done).** The earlier `keypoints` output is gone. One mechanism now carries
  joints, embeddings and waveforms; it replaced `PRIVATE_AUDIO`, the `embedding` field and the Sapiens2 `keypoints` field. What can
  still enter a record is listed in private-payloads.md: **owner decision** whether the legacy face/pose plugins and the MediaPipe
  graph should move to it (the dashboard overlay reads their landmarks from records).
- **Exemplars.** Channel values change on every non-square image (angles, wrist offsets, wrist gap). VIDEO mode also tracks
  and smooths, so landmarks differ slightly from the old IMAGE mode. Exemplar files written before this change are refused with
  "enroll it again" (files now carry `channels`). Match thresholds (.12 / .18) were not re-measured.
- **feat/sota-modules is merged here.** Recipes that linked `pose:scene.poses` or `target_person:scene.target_poses` into
  `pose-channels` must link the joints inputs (`"joints": "pose_joints", "emits": ["pose_joints"]`); the old link raises an error.
  The MediaPipe graph's `scene.poses` still works. `sapiens2_pose.to_poses` reports `image_width` / `image_height`.
- **Crops.** In the default recipe pose runs on full camera frames before `target_person`, so tracking is never reset. A pose module fed by
  changing crops gets each crop's own aspect, but VIDEO mode starts a new segment whenever the size changes (existing rule).
- **Whisper.** Bundle candidates (`faster_whisper.small`, `nemotron3`) always get their own weight folder.
- **`gamst install`** (`scripts/local_models.py`) keeps its own `.local-models/`; it was not part of the list.
- A worker that fails on one frame is closed (existing ProcessProvider rule) and then fails on every frame.

## Tests
`PYTHONPATH=. ~/.cache/gamst/demo/venv/bin/python -m unittest discover -s tests -q`: Ran 1362, errors 15, skipped 47. The 15
are the known ones (test_stereo_context 7, test_stereo_metric 6, test_sam_audio_native 2). Right after the merge: 1354.

Unit-tested with fakes: `test_one_env_per_model.py` (placement, tampering, apply, worker GAMST, weight root, refine interpreter, the
real VIDEO adapter on a fake detector through the recipe's own links, with and without `target_person`, crops, aspect,
enrollment) and `test_private_payload.py` (guard names, refusals, contracts, and the real worker subprocess and wire with a
stand-in detector: honest, wrong shape, too many, out of range, NaN). `test_model_workers.py` runs the same wire for
embeddings, waveform and 4 x 308 joints with stand-in models.

Actually run: `gamst recipe setup <recipe> --dry-run` for the three recipes (plans only), and read-only lookups in the
owner's flat `~/.cache/gamst/demo/models` (both weights found; all three SHA256 match the registry).

Not run: pip resolution of any environment, editable install, the refine process in the shared environment, the MediaPipe
worker in its environment, VIDEO mode on real frames, a coin take regression.

Slip during the work: a ported test used the new default root and wrote a 25-byte file to `~/.cache/gamst/models/custom/`.
The file and directory were removed and the test now uses a temporary root. The owner's weight files are unchanged.
