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

## 12. One person, one track; everyone a person; scene cuts and track switches (2026-11-28)

The owner saw "감스트" as several people at the same moment in the label tool on the two multi-person clips
(`try-it/testbed/multi/gwan` = 감스트-관제탑 모음집, 522 s, a compilation joined by one-second dissolves;
`try-it/testbed/multi/man` = 맨까송, 63 s, hard cuts). Runs: post mode, the label tool's runner unchanged
(`label_video/run_multi.py`, its options: `--person 감스트.person.json --mode search`, profile `profile-identity.json`:
MediaPipe pose, LVFace-B face, OSNet-AIN appearance) and a copy of it in the scratchpad that also writes the per-frame
face readings, the duplicate boxes and `NAME.persons.json`; outputs in `try-it/testbed/multi_fix/` only. Statistics:
`scratchpad/overlap/*.py`. Labels: the owner's labels in `truth_identity.json` (4 tracks 감스트, 12 others, read while the
owner was still labelling) and Claude's proposals (`claude_proposals.json`, not truth). **No claim of correctness rests
on them; they are used only where named.**

### 12.1 Where the doubling was (before)

GAMST's own bindings (`person.bound` / `person.unbound`, resolved per frame with `intervals_of` / `resolve`):

| run | frames with two tracks bound to 감스트 | all at a handover instant | of these: new binding retroactive / same-frame face |
|---|---|---|---|
| gwan `cand_first` (label tool) | 12 (0.40 s) | 12 | 11 / 1 |
| gwan `cand_loose` (label tool) | 8 (0.27 s) | 8 | 5 / 3 |
| man `cand_first` (label tool) | 2 (0.07 s) | 2 | 2 / 0 |
| gwan `base` (this branch before the fix) | 13 (0.43 s) | 13 | 10 / 3 |
| man `base` | 2 (0.07 s) | 2 | 2 / 0 |

No two bindings of 감스트 overlapped over any span; every double was one frame where the old binding ended at `ts`
and the new one started at `ts` (both ends inclusive). By the boxes at that frame: box IoU >= 0.5 (duplicate boxes)
1 + 4 + 0 + 2 + 0; a scene change within 1 s 4 + 3 + 2 + 4 + 2; two different places 7 + 1 + 0 + 7 + 0.

Where the owner saw several 감스트 at once:

| source | gwan `cand_first` | gwan `cand_loose` | man `cand_first` |
|---|---|---|---|
| label tool rows (one guess per whole track, its majority) overlapping in time | 14 pairs, 16.8 s with two in view | 14 pairs, 17.0 s | 2 pairs, 7.2 s |
| Claude's proposals (whole tracks labelled 감스트) | 33 pairs, 27.9 s with two in view (duplicate 7 pairs / 2.5 s, scene change 4 / 2.2 s, two places 16 / 24.4 s) | – | 4 pairs, 7.6 s (all at scene changes) |
| the owner's labels so far | 2 pairs over time, 0.4 s in view (the dissolve at 89.5-89.9 s: the old and the new scene's 감스트 both on screen) | – | – |

The two-places cases in the proposals are tracks the tracker carried from one person to another (e.g. person-24 is
감스트 seated in scene 2 and continues on the man in the Chelsea shirt after the dissolve at 184.7 s): a label per whole
track cannot be right on both parts. GAMST's binding of person-24 ended at 186.2 s, 1.5 s after the cut (see 12.4).

**Duplicate boxes.** Track pairs with box IoU >= 0.5 in one frame (gwan `cand_first` / man `cand_first`): 43 / 14 pairs,
461 / 124 frames. Cause by time and size: a new track id while the old box lingered (within 0.5 s of either track's
start or end) 35 / 12 pairs; one box inside the other (area ratio < 0.6) 2 / 1; both tracks alive before and after with
similar boxes 6 / 1. By the labels of the two tracks, many pairs joined 감스트 and someone else (gwan 115 of 461 frames,
man 69 of 124), so box overlap alone is not "the same body".

### 12.2 What changed (commits on `feat/person-identity`)

1. **One person, one track at any timestamp** (`TrackBindings`): a handover ends the old binding at its track's last
   frame before the new one starts; a retroactive start never reaches to or before the end of the person's earlier
   bindings; binding a person moves them off the track that holds them. When several tracks claim a person in a frame,
   the best keeps them: face higher by the margin, then appearance higher by the margin, then continuity; the others
   go to unknown or to their own best other person.
2. **Duplicate boxes**: box IoU >= 0.5 *and* joints within 0.05 of the box height (median): the track seen first is
   kept, the other is `duplicate_of` it, not scored, bound or counted (binding and per-track counter, same rule).
   Safety net keyed by person: one action of a person on two tracks within 100 ms counts once.
3. **Everyone is a person** (`person_id`: P1, P2, ...). A face that is nobody known twice on a track starts a person;
   faces bound in the run join the person; enrolled 감스트 is a person with a name. Naming later: `person.named`,
   `enroll_person {person_id}`, or a names map (label tool), all retroactive.
4. **Scene cuts** (`scene_cut.py`): at a cut every binding ends, every track starts a new segment, tracks not seen
   since are finished, appearance from before the cut binds nobody after it, the person needs a fresh confirmation
   (a face after the cut); readers also end any binding that reaches over a cut (a dissolve is decided up to 1.5 s
   late).
5. **Track switches** (the tracker swaps people): a face that contradicts the bound person releases it at once and
   withdraws the binding within the track back to the last confirmation; the track is split there (`track.split`).
   A sustained appearance jump (two looks in a row below `body_reject` against the track's recent look, agreeing with
   each other) splits it too; one odd look does not.
6. **Continuation** (a new track where the person just was) now needs the new track's face, or its appearance of this
   scene; a box alone no longer carries a person.

### 12.3 Before / after (post mode; `base`/`base2` = this branch before, `fix` = item 1-2 only, `final` = all)

| clip | run | two tracks on 감스트 (frames) | 감스트 on a track whose face says not 감스트 | 감스트 bindings over a scene cut | frames with 감스트 bound | owner labels: G->G / G->not G / O->G (s) |
|---|---|---|---|---|---|---|
| 관제탑 | base2 | 13 (0.43 s) | 10.13 s | 5 | 82.7 % | 169.5 / 15.9 / 0 |
| 관제탑 | fix | 0 | – | 5 | 82.5 % | 169.3 / 16.1 / 0 |
| 관제탑 | final | 0 | 2.47 s | 0 | 84.8 % | 172.2 / 13.1 / 0 |
| 맨까송 | base2 | 2 (0.07 s) | 6.47 s | 7 | 69.5 % | (no owner labels) |
| 맨까송 | fix | 0 | – | 6 | 69.6 % | |
| 맨까송 | final | 0 | 1.63 s | 0 | 52.7 % | |

- "Face says not 감스트": a frame of a track bound to 감스트 (final bindings, retroactive) whose latest face reading
  within the last second scores 감스트 below 0.15. The remaining 2.47 s on 관제탑 are at dissolves (track 60 at 372.6 s,
  track 103 at 477.5 s: the face crop is a blend of two scenes) and 0.3 s on track 22 at 88.0 s.
- Owner labels: 25 owner segments on 관제탑 (4 감스트, 18 others, 3 skipped), read while the owner was still labelling;
  per frame inside each labelled segment. None of the runs put 감스트 on a segment the owner labelled as someone else.
- 맨까송: fewer frames bound to 감스트 (69.5 % -> 52.7 %): bindings now end at each of the 17 cuts and need a face after
  it; appearance no longer carries 감스트 over a cut. Without labels this is not shown to be better or worse.
- Scene cuts detected by the run: 관제탑 11 (89.1, 184.7, 277.8, 372.0, 413.0, 477.5, 480.7, 510.9, 514.2, 519.2,
  520.9 s), 맨까송 17. Seen by eye on 관제탑: 8 scene changes (dissolves) plus zooms within a scene (480.7, 519.2); on
  맨까송 the hard cuts, one fade to black at 63.2 s and one false cut from a person moving close to the camera (the
  offline run of the same detector found it at 1.5 s; the pipeline run found 2.1 s).
- Track switches detected (`track.split`), all by a contradicting face, none by appearance: 관제탑 38 (e.g. 35.6, 50.2,
  71.6, 80.6, 88.0, 126.8-138.2 (track 25, 5 times), 182.0, 189.0, 192.2, 220.6-237.8, 316.8, 318.4, 372.6, 412.8, 465.0,
  477.0, 517.2 s), 맨까송 8 (21.2, 23.2, 52.3, 53.5, 56.5, 58.9, 61.3, 61.3 s). The appearance test (OSNet-AIN, 0.6) did
  not fire on these clips.
- Duplicate boxes suppressed (pose-frame pairs, `duplicate_of`): 관제탑 438, 맨까송 68; a person moved to the kept
  track (`by: duplicate`) 15 / 4 times.
- Action candidates (label-tool merge, loose rule) per person, final: 관제탑 P1 (감스트) 33, P4 9, P7 3, P3 2, P10 2,
  unknown 11; 맨까송 P2 18, P1 (감스트) 15, P3 8, P4 2, P5 1, unknown 35. Before (base2): 관제탑 감스트 27 / unknown 33,
  맨까송 감스트 19 / unknown 63.

### 12.4 The owner's case at 3:05 (관제탑, 180-210 s)

Check frames: `try-it/testbed/multi_fix/check/gwan_305_*.jpg` (base2 left, final right; red = 감스트, green = another
person id, grey = nobody), every 2 s.

| track | base2 | final |
|---|---|---|
| person-24 (scene 2: 감스트 seated on the right; after the dissolve at 184.7 s the tracker keeps it on the man in the Chelsea shirt) | 감스트 until 186.2 s, so 1.5 s on the other man after the cut | 감스트 until 184.4 s; after the cut P4 (the Chelsea man, the same id as his scene-2 track person-28), later P5 / P3 for short stretches |
| person-32 (scene 3: 감스트 standing, red shirt) | 감스트 from 185.7 s, a gap 198.4-198.7 s | 감스트 (P1) from 185.4 s, no gap |
| person-35 (the Chelsea man, jumping at 198 s) | 감스트 198.0-198.7 s (a continuation from person-32's box) | P4 |

What the frames show: at 184 s the Chelsea man jumps in scene 2 (final: P4); at 186-192 s 감스트 stands in the red
shirt (person-32, 감스트 in both); the man on the right at 192 s (person-24) is unknown in both at that moment; at 198 s
base2 had 감스트 on the jumping Chelsea man (person-35), final has P4. In this window, 감스트 on the wrong man: base2
2.2 s, final 0. The owner's view showed person-24 as 감스트 until 196 s because the label tool labels a whole track by its
majority; with the persons file it is split at the cut.

Other check frames (`gwan_contra_*.jpg`): at 317.5 s base2 had 감스트 on the man in the blue shirt seen from behind
(track 49), final has nobody; at 414.5 s base2 had 감스트 on a face in a poster on the wall (track 87), final has a
person of its own there (P8); at 373.2 and 478.0 s (dissolves) final has 감스트 on 감스트's tracks while the faces
are blended.

### 12.5 Persons found (final)

| clip | persons (>= 1 s) | seconds per person | segments per person | two tracks on one person |
|---|---|---|---|---|
| 관제탑 | 10 (8) | P1 감스트 467, P4 241, P7 95, P2 79.5, P3 50.2, P8 45.7, P10 4.2, P9 1.5, P6 0.6, P5 0.2 | 51, 36, 4, 29, 25, 31, 2, 1, 2, 1 | 0 frames |
| 맨까송 | 7 (5) | P1 감스트 37.0, P3 34.1, P2 31.2, P6 3.5, P4 1.8, P5 0.6, P7 0.2 | 17, 15, 12, 1, 1, 3, 1 | 0 frames |

People seen by eye (Claude, not labels): 관제탑 about 6-8 (감스트; the man in the Honze hoodie, scene 1; the man in
the Chelsea shirt, scenes 2-4; the man in the PUMA shirt, scene 5; two or three performers on the TV stage) plus faces
on posters; 맨까송 about 5 performers. P4 follows the Chelsea man over three scenes by face. P3 appears from 33 s to
477 s in several scenes and is probably more than one person (a weak face merged); P8 is mostly the wall poster.
Tracker tracks that reach over a scene cut: 20 / 19; person segments over a cut: 0 / 0.

The per-run persons file the label tool reads: `try-it/testbed/multi_fix/{gwan,man}/final.persons.json`,
`{start_ms, cuts, persons: [{person_id, name, segments: [{track_id, start, end, start_ms, end_ms}]}]}` (seconds from
the first frame; `end: null` = to the end of the clip). Field name: `person_id`.

### 12.6 관제탑 events per repetition (rule runs, counter only)

Runs `rule_{strict,loose}_{before,after}` in `try-it/testbed/multi_fix/gwan/`: the same post-mode pipeline without the
person binding (`--without person_binding`), the strict rule (`ex_gwan_rule.json` = `rule_gwan.json`) and the loose
one (`ex_gwan_search.json` = `rule_gwan_search.json`), before (2885bb5) and after (017532e) the repetition fix. Persons
from the `final` run's bindings (the tracker is deterministic: the same track ids).

| rule | events | per person | longest event | events > 2.5 s | label-tool candidates joining several events (0.3 s merge) | candidates > 2.5 s |
|---|---|---|---|---|---|---|
| strict, before | 3 | P1 감스트 2, P7 1 | 0.63 s | 0 | 0 | 0 |
| strict, after | 3 | P1 감스트 2, P7 1 | 0.63 s | 0 | 0 | 0 |
| loose, before | 66 | P1 감스트 35, P4 9, P7 6, P3 2, P10 2, unknown 12 | 1.97 s | 0 | 3 | 1 (person-88, 470.8-473.6 s, 5 events) |
| loose, after | 66 | the same | 1.53 s | 0 | 4 | 0 |

- The state machine: the restart after the last step was already immediate; the event started at the first step's
  onset; `within_ms` was counted from the first step's completion. A series where the fold was not reached between
  two extends was one event from the first extend; now the event starts at the repetition's own extend (an extend let
  go for 100 ms and held again). On these runs that shortened the longest event (1.97 -> 1.53 s); the count is the same.
- The long spans the owner saw are mostly the label tool's: it joins matches of one track that are less than 0.3 s
  apart (`MERGE_S` in `run_multi.py`), e.g. 5 events of person-88 at 470.8-473.6 s became one candidate. GAMST's own
  events never exceeded 2.5 s here. With quick repetitions (fold, then the next extend within 0.3 s) a 0.3 s merge
  joins separate repetitions.
