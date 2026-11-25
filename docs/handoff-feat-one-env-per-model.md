# Handoff: feat/one-env-per-model

Worktree `.local-run/wt-crew`, branch `feat/one-env-per-model` from `0eee40a`. Committed, not pushed, no PR.
Design: [crew.md](crew.md). No environment was created, nothing was installed or downloaded, no model was run.

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
3. **Pose worker.** `mediapipe-pose` is now `PoseWorkerPlugin` around the `mediapipe.body.video` worker, added without a lane so
   VIDEO tracking sees every frame. It emits the same `scene.poses` signals, so `pose:scene.poses` links are unchanged.
   `landmarks=True` is opt-in (body only), changes the spec version (`.k33`), and adds 33 bounded `[x, y, visibility,
   presence]` rows per pose. `PoseChannels` accepts the plugin list and a `{poses, summary}` value, and scales x by
   the reported image width/height.
4. **Weight root.** `~/.cache/gamst/models/<model>/<file>` (`GAMST_MODELS`); flat `--models` directories stay readable.
5. **Worker GAMST.** Checkout (editable, path + commit in the plan) / release pin / git pin. Version not bumped.
6. **OpenCV.** Test: no registry environment holds two cv2 distributions.

## Read before merging
- **Field name.** The worker field is `keypoints`, not `landmarks`: `CapabilitySpec.validate` refuses a field called
  `landmarks` (`RAW_KEYS`) so that joint positions never become context. I did not change that guard. `PoseWorkerPlugin`
  consumes the keypoints and commits only `scene.poses`, from which context already strips them, and `role_module`
  refuses a profile entry with `landmarks: true`. **Owner decision:** is a differently named field acceptable inside
  the private worker exchange, or should worker-private outputs get their own transport?
- **Exemplars.** Channel values change on every non-square image (angles, wrist offsets, wrist gap). VIDEO mode also tracks
  and smooths, so landmarks differ slightly from the old IMAGE mode. Exemplar files written before this change are refused with
  "enroll it again" (files now carry `channels`). Match thresholds (.12 / .18) were not re-measured.
- **feat/sota-modules** touches the same places. Expected conflicts: `recipes/__init__.py` (`_reference_value`, `enroll`),
  `recipes/parts.py` (`_mediapipe_pose`, `tracker` is already passed), `pose_exemplar.py` (`_landmarks` is the same code),
  `crew/modules.py`, `person-action-count.json` (keep theirs, plus the `compute` text). `sapiens2_pose.to_poses` should add
  `image_width` / `image_height` to each pose, otherwise its channels stay uncorrected (records then have no `image_aspect`).
- **`gamst install`** (`scripts/local_models.py`) keeps its own `.local-models/`; it was not part of the list.
- A worker that fails on one frame is closed (existing ProcessProvider rule) and then fails on every frame.

## Tests
`PYTHONPATH=. ~/.cache/gamst/demo/venv/bin/python -m unittest discover -s tests -q`: Ran 1278, errors 15, skipped 38. The 15
are the known ones (test_stereo_context 7, test_stereo_metric 6, test_sam_audio_native 2). Base `0eee40a`: 1254 tests.

Unit-tested with fakes (`tests/test_one_env_per_model.py`, 24 tests): placement and tampering, apply with fake
pip/venv/downloads, worker GAMST, weight root and flat lookups, refine interpreter (Popen patched), the pose worker with
the real VIDEO adapter on a fake detector through a real Pipeline, aspect correction, enrollment.

Actually run: `gamst recipe setup <recipe> --dry-run` for the three recipes (plans only), and read-only lookups in the
owner's flat `~/.cache/gamst/demo/models` (both weights found; all three SHA256 match the registry).

Not run: pip resolution of any environment, editable install, the refine process in the shared environment, the MediaPipe
worker in its environment, VIDEO mode on real frames, a coin take regression.

Slip during the work: a ported test used the new default root and wrote a 25-byte file to `~/.cache/gamst/models/custom/`.
The file and directory were removed and the test now uses a temporary root. The owner's weight files are unchanged.
