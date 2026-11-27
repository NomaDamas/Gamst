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

## 7. Unmeasured

- How often a binding is right or wrong, on any clip (no labelled multi-person clip).
- The thresholds (face 0.35 / 0.15, appearance 0.8, margins, agreement 0.5): parameters, not set on labelled people.
- Identity when two people cross and the tracker swaps them; retroactivity never crosses a track break.
- Realtime: LVFace-B costs 50-60 ms per face; with several people it does not fit in a 33 ms frame inside the pose lane
  (`a_post2`: the binding was busier than the pose model). Not run in realtime mode here.
- The voice part of a person reference in a run (it is stored and checked; speaker attribution uses it in the speech
  recipes, not wired to tracks here).
- MCP `enroll_person` with real models (the unit tests use test doubles).
