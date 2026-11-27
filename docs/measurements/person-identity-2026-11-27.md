# Person identity: model choice, enrollment, binding (2026-11-27)

Branch `feat/person-identity` (Draft PR against PR #373). Mac M2 Pro, 32 GB. Every run one at a time through
`try-it/guard.sh`. Runs: `try-it/testbed/identity/run.sh NAME VIDEO PERSON` (store `NAME.sqlite`, report `NAME.json`,
peak RSS `NAME.rss`); statistics from the stores by `scratchpad/binding_stats.py`; embedder speed by
`scratchpad/bench_embed.py`. Pose model in every run: RTMW-l 384x288 + YOLOX-m (CoreML), the recipe default.

**Nothing here is an accuracy.** There are no owner-labelled multi-person clips yet. What is measured: consistency of the
enrollment, how the binding behaves on two testbed clips, timing. Where a "truth" is used it says what it rests on.

## 1. The whole-body appearance model (replaces SOLIDER)

Criteria (decision 44): official release; permissive licence for code and weights; published or verifiable digest;
<= 500 MB; runs on CPU / MPS / CoreML. Stars, licences and digests read from the GitHub API and the Hugging Face API on
2026-11-27; benchmark numbers without re-ranking from the papers or model zoos named.

| candidate (stars) | code / weights licence | weights | Market-1501 R1 / mAP | MSMT17 R1 / mAP | fails |
|---|---|---|---|---|---|
| **OSNet-AIN x1.0, MSMT17** (KaiyangZhou/deep-person-reid, 4,949) | MIT / MIT (HF model card) | authors' HF repo `kaiyangzhou/osnet`, 17,293,009 B, sha256 published | 70.1 / 43.3 cross-domain (trained on all of MSMT17, tested on Market; torchreid MODEL_ZOO) | n/a (trained on all of it) | none |
| OSNet x1.0 / OSNet-IBN x1.0, MSMT17 (same repo) | MIT / MIT | same HF repo, ~17.3 MB, sha256 published | 66.6 / 37.5, 66.5 / 37.2 cross-domain | n/a | none (less accurate) |
| OSNet x1.0, same-domain checkpoints | MIT / MIT | Google Drive only | 94.2 / 82.6 | 74.9 / 43.8 | digest |
| fast-reid SBS R50-ibn / R101-ibn (JDAI-CV, 4,013) | Apache-2.0 | GitHub release v0.1.1, 308 / 538 MB, `digest: null` | 95.7 / 89.3, 96.3 / 90.3 | 83.9 / 60.6, 84.8 / 62.8 | digest (R101 also size) |
| CLIP-ReID ViT-B/16 (532) | MIT | Google Drive only | 95.5 / 89.6 | 88.7 / 73.4 | digest |
| TransReID ViT-B (1,072) | MIT | Google Drive only | 95.1 / 89.0 | 85.3 / 67.8 | digest |
| SOLIDER-REID Swin-B (99) | MIT; LUPerson pretraining non-commercial | Google Drive only | 96.9 / 93.9 | 90.7 / 77.1 | digest, data terms |
| PersonViT ViT-B (55) | Apache-2.0 / "cc" without a variant; LUPerson pretraining | authors' HF, 349 MB, sha256 published | 97.6 / 95.0 | 92.0 / 80.8 | licence |
| PASS ViT-B (72) | Apache-2.0; LUPerson | Google Drive only | 96.8 / 93.0 | 88.2 / 71.8 | digest |
| KPR, BPBReID | Hippocratic 3.0 | Google Drive only | 96.6 / 93.2, 95.7 / 89.4 | - | licence, digest |

Sources: torchreid `docs/MODEL_ZOO.md`; fast-reid `MODEL_ZOO.md`; CLIP-ReID arXiv 2211.13977 Table 2; TransReID,
SOLIDER-REID and PASS READMEs; PersonViT arXiv 2408.05398 Table 2; KPR arXiv 2407.18112 Table 1.

**Chosen: OSNet-AIN x1.0 MSMT17** (`osnet.ain_x1_0_msmt17`), the only candidate that met every criterion. Its checkpoint
has 4,101 identity classes, i.e. all MSMT17 identities (the "combineall" model zoo row), so its published number is
cross-domain and far below SOLIDER's same-domain one: the criteria cost accuracy. MSMT17's own terms are research use
under a signed agreement: a data-terms question, separate from the MIT weight licence.

Conversion (`gamst/providers/osnet_export.py`): the official `torchreid/models/osnet_ain.py` at f8cd150f (sha256
`a43b4958…`), state dict loaded strictly, eval feature (512-d) exported to ONNX opset 17 with a dynamic batch.
ONNX file 0384d520f47253e1a2be517273442b416670a47c33762a1084a5c55f6983b863; largest difference to PyTorch on 4 random
crops 2.7e-05 (run `osnet-export`, torch 2.9.1 of the sapiens2 environment, onnx 1.19.1 from a separate folder).

## 2. Downloads (owner approved LVFace-B; OSNet-AIN met the criteria)

| file | bytes | sha256 | check |
|---|---|---|---|
| `models/lvface.b/LVFace-B_Glint360K.onnx` | 455,533,594 | 9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382 | equals the registry pin |
| `models/osnet.ain_x1_0_msmt17/osnet_ain_x1_0_msmt17_256x128_…_jitter.pth` | 17,293,009 | 8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f | equals the HF LFS oid at revision a5c5cc03 |
| `models/osnet.ain_x1_0_msmt17/src/osnet_ain.py` | 22 kB | a43b495866a4b1645cd134a30c3011592e8c6826ba4781955a135588cf46cd25 | pinned GitHub revision f8cd150f |
| `models/osnet.ain_x1_0_msmt17/osnet_ain_x1_0_msmt17.onnx` (made here) | about 9 MB | 0384d520… | section 1 |

No new environment: both run in the existing `venvs/rtmlib` (onnxruntime 1.32, opencv, numpy).

## 3. Embedder speed (run `id-bench-*`, one crop per call, 50 calls after 5 warm-up)

| model | CPU p50 / p95 ms (two runs) | CoreML ALL | CoreML CPU+GPU | CoreML CPU+Neural Engine | first CoreML load |
|---|---|---|---|---|---|
| LVFace-B (112x112) | 60.0 / 61.3, 59.6 / 61.5 | 47.3 / 50.8 | 46.6 / 54.7 | 35.2 / 45.1 | 116-176 s, 1.9 GB compiled cache |
| OSNet-AIN (256x128) | 10.4 / 10.5, 9.0 / 9.7 | 5.4 / 7.0 | 3.9 / 8.5 | 9.3 / 9.6 | 0.7-2.4 s |

With its dynamic batch dimension LVFace aborted on CoreML (`mps.matmul` verification failure, 51 partitions); fixing the
batch to 1 (`add_free_dimension_override_by_name`) made it run. LVFace-B is the cost: 35-60 ms per face on every unit. The compiled
caches were removed after the runs (regenerable).
The runs below used the CPU (the default).

## 4. Enrollment

| run | input | samples (exactly one person) | face crops kept / dropped | face agreement | body crops | body agreement | wall |
|---|---|---|---|---|---|---|---|
| `id-enroll-a` | shop clip `a.mp4` 0-10 s | 35 (93 frames skipped: 0 or 2+ people) | 27 / 3 | 0.424 -> 0.526 after dropping 3 | 35 | 0.764 | 36.2 s |
| `id-enroll-photos` | 5 photos saved from `a.mp4` (5.0-9.5 s) | 5 | 4 / 1 | 0.602 | 5 | 0.754 | 26.2 s |

- The first run was refused by the check (agreement 0.42 < 0.5). The pairwise matrix (`id-enroll-a-raw`) showed three
  crops (3.2-3.8 s, a turned head) at about 0.0 against every other crop; the rest agreed 0.2-0.9. That led to dropping
  disagreeing crops by majority (the voice reference's rule).
- Photo vs range reference (same clip, so not independent): face centroids 0.906, body centroids 0.960; each photo face
  against the range centroid 0.64-0.86.

## 5. Binding on the shop clip (several people, the reference's own video)

`a_post2`: `a.mp4` (54.4 s, 30 fps, 854x480), reference `gamst_a_0-10.json`, post mode, slam rule as the movement.

| measure | value |
|---|---|
| people per frame (pose model) | 1: 555, 2: 743, 3: 225, 4: 105 frames |
| tracks started by the tracker | 324 names; 275 pose entries without a track; about 49 real tracks |
| frames with the person bound to a track | 1,202 of 1,628 (73.8 %) |
| bindings | 47 by face, 0 by body; released 11 (face contradicted), 35 (face seen on another track) |
| tracks bound to the person | 15; visible span median 5.2 s, max 24.7 s |
| retroactive bindings | 8, reaching back median 1.2 s, max 11.8 s |
| face score, bound track | median 0.567 (p10 0.489, p90 0.759) |
| face score, best other track | median 0.046 (p90 0.132) |
| body score, bound track / best other track | median 0.874 / 0.639 (other p90 0.768) |
| events of the rule (no slams in this clip: false matches of other movements) | 9; 2 of them became the person's afterwards (retroactive), 7 unknown |
| wall, peak RSS | 110.9 s for 54.4 s of video; 4,105 MB |
| busy per module | pose 51.5 s; person_binding 60.7 s (p95 174 ms on a frame that embeds faces); exemplar 0.3 s |

Before two fixes (`a_post`, same input): 302 bindings and 301 releases, the person changed track 100 times. Causes:
poses without a track id became one-frame tracks that took the person, and a track took the person from a track that
was still showing the face. Fixes: untracked poses are never bound; a person moves only when the new track's face beats
the holder's face in the same frame. After: 47 / 46, 43 changes.

What this does and does not show: the face scores of the track the person was bound to and of the other tracks barely
overlap (p10 0.49 vs p90 0.13), and the bound track was the person in the frames looked at by eye, but no frame is
labelled, so how often the binding is right or wrong is **not measured**. The appearance scores overlap much more (other
tracks up to 0.77); the default appearance threshold was raised from 0.6 to 0.8 because 0.6 is below the median of the
other tracks: a value read from one unlabelled clip, to be set on the owner's clips (decision 6).

## 6. Another video, another day: the slam clip

`b_post`: `b.mp4` (11.9 s, 60 fps, 1280x720), the same reference (made on the shop clip: other clothes, camera and
light), so appearance is not used across videos; the face carries it. Truth for the count: `truth_hit.json` (16 slams,
the owner's). Who: the only person in all 714 frames, assumed to be 감스트 from the clip's title (not owner-labelled).

| measure | value |
|---|---|
| tracks | 1 (714 frames, one person each) |
| frames bound | 666 of 714 (93.3 %); 3 bindings by face, 2 releases (face contradicted: score < 0.15 while the head was down) |
| face score, bound | median 0.377 (p10 0.302, p90 0.431): lower than in the reference's own video (0.567) |
| count of the per-track counter | 16 counted, 16 right, 0 extra, mean time error 8.6 ms (`score.py`) |
| who of the 16 events | 감스트 for all 16, when counted |
| wall, peak RSS | 40.9 s; 4,126 MB |
| busy | pose 11.5 s, person_binding 4.6 s (6.4 ms per frame on average; p95 14.8 ms) |

## 6b. Repeat with the final code

`a_post3` and `b_post2` (after the last commit: untracked poses continue a counter by box overlap): the binding results
are identical to `a_post2` and `b_post` (same bindings, releases, frames bound, resolved who of every event); `b_post2`
again 16 counted, 16 right, 8.6 ms. Wall 105.7 s and 20.4 s (the first `b_post` included compiling CoreML models).

## 8. Identity beside the pose lane, tracker swaps (follow-up, same day)

Changes measured here:
- Faces are looked at when due: every 200 ms while a track is unconfirmed, every 1 s once it is bound.
- Crops go to a scorer beside the pose lane. It is one per binding, shared with its L2 forks, takes live jobs first and
  newest first, and a newer crop of a track replaces one that has not started.
- A track that appears where the person's track just left continues the person (`continuation`).
- A binding ends when its track was last seen, and never overlaps the same person's previous track.
- Bindings close at the end of input.

Every run here: shop clip `a.mp4`, the reference `gamst_a_0-10.json`, with `GAMST_SPILL_FLOOR_GB=5`. Without that
variable the video memory's 10 GB disk floor, new on the base branch, made the spill bound 0 on this disk and the lanes
lost frames (`a_post4`, first try).

**First attempt: a lane of its own** (`a_post4` post, `a_live` realtime, before the scorer).
- The binding consumed the pose lane's joints in a lane of its own.
- It found a pipeline bug: a spilling lane stored the derived joints input in the video memory, which turned
  `PrivatePoses` into an ndarray, so the binding never paired a frame. This is fixed in `Pipeline._enqueue`: only images
  are kept. A test covers it.
- With the fix it kept up in post mode (busy 26.8 s for 54 s).
- In realtime it lost identity for every frame the pose lane caught up by L2. An L2 chain feeds only the forks of its
  own lane, so those 460 frames never reached the binding.
- Hence the scorer design: the binding stays in the pose lane, where L2 forks it with handover, and no model runs in
  that lane.

Post mode (`a_post3`: before these changes; `a_post5`: after):

| | before (`a_post3`) | after (`a_post5`) |
|---|---|---|
| frames with the person bound | 1,202 of 1,628 (73.8 %) | 1,211 (74.4 %) |
| the person moved to another track id | 43 times, all by face | 14: 6 continuation, 8 face |
| face embeddings | one per track every 200 ms | 241 (jobs 317, 38 replaced) |
| time inside the pose lane | 60.7 s (the models ran there) | 0.39 s |
| decision delay (crop to decision) | 0 | median 100 ms, max 500 ms |

Realtime (`--as-live`). `a_live_nobind` is the same run without binding.

| run | pose frames live / by L2, lost | pose ms per frame (busy / frames) | binding time in the lane | live binding record latency p50 / p95 | frames bound | moves (continuation / face / body) |
|---|---|---|---|---|---|---|
| `a_live_nobind` | 1,261 / 367, 0 lost | 36.0 | – | – | – | – |
| `a_live2` (per-module scorer) | 1,176 / 452, 0 lost | 37.5 | 0.19 s | 523 / 1,542 ms | 77.5 % | 9 / 1 / 2 |
| `a_live3` (per-module scorer) | 1,105 / 523, 0 lost | 39.6 | 0.21 s | 542 / 1,450 ms | 31.9 % | 2 / 1 / 1 |
| `a_live4` (shared scorer) | 1,197 / 431, 0 lost | 37.0 | 0.19 s | 507 / 1,530 ms | 82.0 % | 6 / 6 / 1 |
| `a_live5` (shared scorer) | 1,168 / 460, 0 lost | 36.6 | 0.18 s | 485 / 1,523 ms | 74.6 % | 9 / 12 / 1 |

- **Pose is never blocked or lost.**
  - In every run: 0 gaps, 0 lost frames, 0 frames skipped without being processed.
  - The pose model (RTMW, 2-4 people on this clip) is slower than the input by itself (36 ms per frame against 33 ms);
    the no-binding run shows the same. Its L2 caught up 10-14 intervals per run.
  - Pose per frame was 36.0 ms without binding and 36.6-39.6 ms with it. A slowdown of about 1-10 %, from the models
    sharing the Mac, is possible but not separated from run-to-run variance.
- **The binding keeps up.**
  - The scorer had 0-1 job waiting at the end of each run.
  - Decision delay: median 67-133 ms, max 300-633 ms.
  - Binding records of the live lane were known 0.5 s after capture (median), 1.5 s at p95; this follows the pose lane's
    own latency.
  - Records of L2 chains came 19-22 s after capture (median), with the pose L2 corrections.
  - A per-module scorer let the L2 forks' jobs starve the live lane: `a_live3` bound only 31.9 % of frames. With one
    shared scorer, live jobs first, the two repeats bound 82.0 % and 74.6 %.
- **Tracker swaps.**
  - The tracker named about 50-70 tracks for about 2-3 people in 54 s.
  - Before: the person changed track id 43 times, every time waiting for a face, with releases at up to 4.6 s.
  - After: 13-22 moves per realtime run, 6-9 of them by continuation (box overlap within 1 s, no contradicting face).
  - The time the person had no track at a move: median 0.03-0.27 s, max 4.5-8.2 s.
  - Whether a continuation was right is not measured: nothing is labelled. A wrong one lasts until the next face check,
    at most about 1 s while the face is visible.

## 9. Enrolling over MCP while the run goes on (no `--person`)

Script: `scratchpad/mcp_midrun.py`. It runs the shop clip `--as-live` with no person given (the binding runs and binds
nobody) and calls the MCP tools in the run's process through `MCPServer.call_tool`, the path `gamst-mcp-live` reaches.

1. At 2.5 s: `enroll_person {name: 감스트, consent, camera_seconds: 7}`.
2. When it answers: `set_instruction {kind: exemplar, who: 감스트}`, keeping the current movement.
3. At 53.5 s: `count_events` with and without `who`, and `get_timeline {who}`.

| run | enroll | crops kept | agreement face / body | basis | who set at | person bound back to | frames bound |
|---|---|---|---|---|---|---|---|
| `mcp_midrun2` | 7.1 s | 23 faces, 30 bodies | 0.52 / 0.75 | face, body (this run's session) | 9.5 s | the track's first frame (0.0 s) | – |
| `mcp_midrun3` (final code) | 7.2 s | 23 faces, 30 bodies | 0.53 / 0.77 | face, body | 9.5 s | the first frame (0.0 s; retroactive up to 9.6 s) | 63.3 % |

- The reference was saved as `감스트.person.json`: mode 600, embeddings only.
- `list_instructions` showed the person as usable and bound.
- `count_events {who}` counted 1 of the 6 matches of the slam rule (`mcp_midrun3`). The rule has no true event on this
  clip, so this checks the plumbing only.
- `get_timeline {who}` gave 11 intervals of the person on screen.

## 10. Unmeasured, after the follow-up

- Whether a continuation or any other binding is right (no labelled multi-person clip).
- How much the identity models slow the pose model, apart from run-to-run variance.
- The appearance thresholds (PersonViT 0.9, OSNet-AIN 0.8) are **provisional**: read from one unlabelled clip.
- Realtime on the slam clip with binding. Post mode `b_post3` still counts 16 of 16, all who = 감스트, with no move.

## 11. The state-of-the-art appearance model: PersonViT vs OSNet-AIN (owner: "무조건 SOTA")

**Choice.** The owner accepts non-commercial terms and Google Drive releases (GAMST is not used commercially). The ranking
uses published same-domain results without re-ranking, for models whose authors released an MSMT17 checkpoint
(research by subagent and by hand, 2026-11-27; sources in the PR and design doc).

| model | MSMT17 mAP / R1 | Market-1501 mAP / R1 | weights |
|---|---|---|---|
| **PersonViT ViT-B/16** (hustvl/PersonViT, 55 stars) | **80.8 / 92.0** (official `train_log.txt`: `RE_RANKING: False`) | 95.0 / 97.6 | authors' HF `lakeAGI/PersonViTReID`, 349,329,029 B |
| SOLIDER-REID Swin-B | 77.1 / 90.7 | 93.9 / 96.9 | Google Drive |
| CLIP-ReID ViT-B/16 (+SIE, needs camera IDs) | 75.8 / 89.7 | about 90.5 / 95.4 | Google Drive |
| TransReID-SSL ViT-B (384) | 75.0 / 89.6 | 93.2 / 96.7 | Google Drive |
| PASS ViT-B (384) | 74.3 / 89.7 | 93.3 / 96.9 | Google Drive |

Higher published numbers without released weights were left out: CLIP-SCGI 88.2, PromptSG 87.2 (empty repo),
PersonMAE 79.8, HPL 79.0, DRFormer 78.7.

**Download and conversion.**
- Checkpoint `transformer_120.pth`: sha256 `e1562c57…b4`. This digest was computed here, and it equals the HF LFS digest.
- Loaded with `torch.load(weights_only=True)`.
- Built from the official `vit_pytorch.py` at 7dc5b37d (sha256 `21435d31…a4`), with the configuration of its MSMT17
  log: 256x128, stride 16, no SIE / JPM, the feature before the BN neck, mean and std 0.5.
- Exported to ONNX by `gamst/providers/personvit_export.py`: 343,087,913 B, sha256 `46810e4b…2a`, largest difference to
  PyTorch 2.2e-05.
- Speed per crop, the same benchmark as section 3:

  | | CPU | CoreML ALL | Neural Engine |
  |---|---|---|---|
  | PersonViT p50 / p95 | 59.2 / 83.7 ms | 35.6 / 39.1 ms (first compile 151 s, 1.5 GB cache, removed) | 37.2 ms |
  | OSNet-AIN p50 | 9.6 ms | 4.1 ms | – |

**Separation on the shop clip.**
- Script: `scratchpad/identity/separation.py`, after the enrollment range.
- Who is who: LVFace score ≥ 0.40 is the person, ≤ 0.10 is someone else. **This is a proxy, not labels.** A turned head
  of the person can land on the "someone else" side.
- 404 pose samples: 169 the person, 222 others.

| model | AUC | person p10 / median | others median / p90 / p99 |
|---|---|---|---|
| OSNet-AIN | 0.952 | 0.663 / 0.781 | 0.615 / 0.676 / 0.759 |
| PersonViT | **0.971** | 0.891 / 0.940 | 0.713 / 0.812 / 0.938 |

The two models' scores are on different scales, so the appearance thresholds are now per model and **provisional**
(`APPEARANCE` in `person_binding.py`):

| model | binds by appearance at | clearly someone else at |
|---|---|---|
| PersonViT | 0.9 | 0.7 |
| OSNet-AIN | 0.8 | 0.6 |

**In the pipeline** (same reference range, `gamst_a_0-10.json` for OSNet and `gamst_a_personvit.json` for PersonViT):

| run | model | frames bound | track-id moves (continuation / face / body) | body AUC, bound vs other track |
|---|---|---|---|---|
| `osn_a_post` | OSNet, post | 74.4 % | 14 (6 / 8 / 0) | 0.903 |
| `pvit_a_post` | PersonViT, threshold 0.8 | 77.3 % | 16 (7 / 8 / 1) | 0.925 |
| `pvit_a_post3` | PersonViT, its thresholds | 81.8 % | 17 (7 / 9 / 1) | 0.926 |
| `osn_a_live` | OSNet, realtime | 76.6 % | 26 (11 / 14 / 1) | 0.911 |
| `pvit_a_live` | PersonViT, threshold 0.8, realtime | 78.1 % | 17 (8 / 7 / 2) | 0.979 |
| `osn_a_live2` | OSNet, realtime, loaded Mac | 75.6 % | 21 (12 / 8 / 1) | 0.939 |
| `pvit_a_live3` | PersonViT, its thresholds, realtime, loaded Mac | 86.1 % | 18 (9 / 9 / 0) | 0.937 |
| `osn_b_post` / `pvit_b_post2` | slam clip, post | 98.7 % / 98.7 % | 0 / 0 | – (one person) |

- In every run: pose 0 lost and 0 gaps, the binding 0 lost.
- The slam clip still counts 16 of 16 with both models, all events who = 감스트.
- "Loaded Mac": the last two realtime runs ran while other programs on the Mac (outside GAMST) kept the load average at
  13-20. The pose model's live lane then covered only about 720 of 1,628 frames (about 1,170 before); L2 caught up the
  rest.
- The pipeline AUCs come from a few dozen fresh scores per run and vary from run to run.
- The offline measurement above (404 samples) is the better comparison of the two models, and it still rests on a face
  proxy.

**Conclusion.**
- On this clip PersonViT separates the person from the others better than OSNet: AUC 0.971 vs 0.952 offline.
- With its own threshold it bound the person in more frames: 81.8 % vs 74.4 % in post mode, 86.1 % vs 75.6 % in
  realtime under the same load.
- It costs 59 ms per crop on the CPU instead of 10 ms. That cost is off the pose lane (the scorer), and bodies are
  looked at only when due.
- Not measured: accuracy against labels.

## 7. Unmeasured (first round)

- How often a binding is right or wrong, on any clip (no labelled multi-person clip).
- The thresholds (face 0.35 / 0.15, appearance 0.8, margins, agreement 0.5): parameters, not set on labelled people.
- Identity when two people cross and the tracker swaps them; retroactivity never crosses a track break.
- Realtime: LVFace-B costs 50-60 ms per face; with several people it does not fit in a 33 ms frame inside the pose lane
  (`a_post2`: the binding was busier than the pose model). Not run in realtime mode here.
- The voice part of a person reference in a run (it is stored and checked; speaker attribution uses it in the speech
  recipes, not wired to tracks here).
- MCP `enroll_person` with real models (the unit tests use test doubles).
