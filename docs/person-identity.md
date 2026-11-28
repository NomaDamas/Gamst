# Person identity: who did it

GAMST never identifies a person from a name or a description. "Who" needs a **few-shot person reference**: photos, a
range of a recording, or a few seconds in front of the camera, with a consent record. Everything that names a person
(`who`) is refused before it starts when there is no reference or the reference fails the check below (owner decision 44).

Design: [design-2026-11-27-integrated-perception.md](design-2026-11-27-integrated-perception.md), sections 2-3.
Measurements: [measurements/person-identity-2026-11-27.md](measurements/person-identity-2026-11-27.md).

## What identifies a person

| evidence | model | trusted where |
|---|---|---|
| face | LVFace-B (`lvface.b`, MIT, 455,533,594 bytes, sha256 pinned) on an ArcFace-aligned crop from the pose model's own face points | everywhere |
| face, found by a face detector (option) | `lvface.b_scrfd`: the same LVFace-B, on a crop aligned with the five points of SCRFD-10GF (InsightFace `det_10g.onnx`, 16,923,827 bytes, sha256 `5838f7fe…5b91`; non-commercial research weights, accepted) run on the region around the track's head (`gamst/providers/face_detect.py`). Same `model_id`, so references enrolled either way are the same. Not the default: on the owner's clips it found fewer faces than the pose points and separated 감스트 from others equally ([measurements](measurements/identity-recall-2026-11-28.md)) | everywhere |
| whole-body appearance (body and clothes) | **PersonViT ViT-B/16, MSMT17** (`personvit.vitb_msmt17`, the default since the owner's decision "무조건 SOTA", 2026-11-27): official MSMT17 mAP 80.8 / Rank-1 92.0 without re-ranking, the best published same-domain result with released weights; converted once to ONNX (`gamst/providers/personvit_export.py`). Option: OSNet-AIN x1.0 MSMT17 (`--set person_binding.body_candidate=osnet.ain_x1_0_msmt17`) | only in the video or live run it was seen in: clothes change |
| voice | the voice enrollment (CAM++), kept in the person reference | binds speakers (words), not tracks |

A person reference is one JSON file (embeddings only, never images, mode 600): `face`, `body`, optional `voice`, the
`samples` it came from (each showed exactly one person), the `session` (which recording the body samples came from), the
`consent` record and its `quality`.

**Terms.** PersonViT's weights are pretrained on LUPerson (non-commercial) and fine-tuned on MSMT17 (research terms);
its Hugging Face card says only "cc". OSNet-AIN is MIT but trained on MSMT17 too. Both are used because GAMST is not used
commercially and the datasets serve tests and demos only (owner, 2026-11-27): non-commercial / research.

## The check (enforced)

`gamst/person_reference.py`, `check()`:

- a consent record;
- exactly one person in every sample: a photo with two people is refused; video frames with nobody or several people are
  skipped;
- at least 3 usable face crops whose embeddings agree (mean pairwise cosine >= 0.5), or a voice whose pieces agree;
- body crops, when given, agree too (>= 0.5);
- a reference with body crops only is usable only in the recording it came from.

A few crops that disagree with the rest (a turned head that the pose points align badly) are dropped by majority, as
for a voice reference (`cleaned()`); two people never form a majority of one, so a mixed reference is refused.
The thresholds are parameters, not values learnt from labelled people (unmeasured).

The refusal is one sentence, the same on every surface (`refusal()`; the HUD shows its short ASCII form, `hud_line()`):

```
who='감스트' refused before start: only 1 usable face crop(s), at least 3 needed (a visible, frontal face). Few-shot identity is required: give a few-shot person reference: CLI 'gamst recipe enroll RECIPE --kind person ...' ...; MCP 'enroll_person' {...}
```

## Enroll

```bash
# photos (each must show exactly one person)
gamst recipe enroll person-action-count --kind person --name 감스트 --consent "who consented, to what, when" \
    --image a.jpg --image b.jpg --image c.jpg --out gamst.json
# a range of a recording (frames with exactly one person are used)
gamst recipe enroll person-action-count --kind person --name 감스트 --consent "..." \
    --video clip.mp4 --start 3 --seconds 10 --out gamst.json
# seconds in front of a camera
gamst recipe enroll person-action-count --kind person --name 감스트 --consent "..." --camera 0 --seconds 5 --out gamst.json
# keep a voice in the same reference
... --voice-file voice.json
```

It prints how many face and body crops were kept, how many were dropped, the agreement and what identifies the person
(`basis`). Nothing is saved when the check fails.

## Count who did what

`person-action-count` now counts **every tracked person** (`track_exemplar.py`) and binds tracks to enrolled persons
(`person_binding.py`). Each event carries `track_id` and `who` (the person, or `unknown`).

```bash
gamst recipe run person-action-count --video clip.mp4 --exemplar move.json --person gamst.json            # every track, with who
gamst recipe run person-action-count --video clip.mp4 --exemplar move.json --person gamst.json --who 감스트 # count = 감스트's
gamst recipe run person-action-count --video clip.mp4 --exemplar move.json                               # nobody yet: enroll over MCP
gamst recipe run person-action-count --video clip.mp4 --exemplar move.json --person gamst.json --with target_person   # one person only (before)
```

- **Binding.** A track is bound to the person when its face scores at least 0.35 and beats every other enrolled person by
  0.05; without a face, by appearance of this run only (PersonViT >= 0.9, OSNet-AIN >= 0.8: **provisional**, read
  from one unlabelled clip with faces as a proxy; all thresholds stay provisional until the owner's labelled
  multi-person clips). A face
  that contradicts (< 0.15) releases the track. A pose the tracker gave no track is never bound. A person moves to
  another track only when that track's face beats the holder's face in the same frame. **Measured on the owner's
  labelled clips** ([identity-recall](measurements/identity-recall-2026-11-28.md)): these defaults name 감스트 on 70 %
  (관제탑) and 53 % (맨까송) of his frames; picked on either half of the scenes, `lvface.b_scrfd` with face 0.15,
  reject 0.10, `reject_after=2`, appearance 0.85 and `reach_back` named him on 77 % and 65 % in real post-mode runs
  (precision 99.8 % and 77.8 %; on 맨까송 the lost precision is one segment whose label is in question). They are LVFace-B / PersonViT numbers on two broadcast clips, set per run with
  `--set person_binding.KEY=VALUE`; the defaults are unchanged.
- **Where it runs.** The binding module sits in the pose lane (it pairs each camera frame with its joints, and an L2
  chain forks it with the pose model), but no face or body model runs there: crops of the tracks that are due go to a
  scorer beside the lane, and the scores are decided on at the first frame after they return (100 ms median on the
  shop clip). A track's face is looked at every 200 ms while it is unconfirmed and every 1 s once it is bound: every
  track is confirmed, no face on every frame.
- **Tracker swaps.** When the track that holds the person leaves the frame and a never-bound track appears where it
  was within 1 s (box overlap >= 0.3), the person continues on it (`by: continuation`) when its face or its appearance
  of this scene says so and nothing contradicts.
- **Retroactive.** The first binding of a track reaches back to the track's first frame, but never before the person was
  released from another track (one person is on one track at a time, also in the past). Readers resolve it at read
  time (`count_events`, `get_timeline`), so a count of a person can grow after the fact. A binding ends when its track
  was last seen; at the end of input every open binding is closed.
- **Nobody enrolled.** Without `--person` the binding still runs and binds nobody; a person enrolled while the run goes
  on (MCP `enroll_person`) is bound from then on, and back to the first frame of tracks that are still followed.
- **One person, one track.** A person (named or not) is on at most one track at any timestamp, also in the past and at
  the instant of a handover: the old binding ends at its track's last frame before the new one starts. When several
  tracks claim a person in a frame, the best keeps them: the face higher by the margin, then the appearance higher by
  the margin, then continuity (the holder, else the track seen first). The others go to unknown or to their own best
  other person.
- **Duplicate boxes.** Two poses of one frame whose boxes overlap (IoU >= 0.5) *and* whose joints agree (median
  distance <= 0.05 of the box height) are one body seen twice: the track seen first is kept, the other is reported
  `duplicate_of` and is not scored, bound or counted (the per-track counter applies the same rule). Box overlap alone
  is not enough: on the owner's clips most overlapping pairs were a new track id while the old box lingered, often on
  two different people. One action of a person counted on two tracks within 100 ms counts once (the L2 safety net,
  keyed by person: per-track counter `by_who`, `count_events` with `who`).
- **Everyone is a person (P1, P2, ...).** A face that is nobody known (below the reject score for every person) twice
  on one track starts a person of its own. An enrolled person is a person with a name. Faces bound in this run join
  the person's face, so the person keeps their id across scene cuts and after leaving the frame; appearance binds
  only within one recording. Records carry `person_id` (tracks, `persons`, binding events, counted events,
  `by_person`), and `who` when the person has a name.
- **Naming later.** `enroll_person {name, consent, person_id}` enrolls a person the run already follows (its faces and
  appearance of this run, the same check) and names it; a `person.named` event names the person back to its first
  frame. Readers that get names from elsewhere (the owner's label tool) pass them as `names` to `intervals_of` /
  `persons_of`.
- **Scene cuts** (`gamst/providers/scene_cut.py`, every camera frame as a 32 x 18 thumbnail): a hard cut (half the
  cells change from one frame to the next) or a dissolve (40 % of the cells differ 0.5 s before and after a moment;
  the cut is the middle of the strongest stretch, decided about 0.5-1.5 s later). At a cut every binding ends, every
  track starts a new segment (a track the tracker carried over the cut onto someone else is two segments), tracks not
  seen since the cut are finished, and appearance from before the cut binds nobody after it
  (`appearance_across_cuts=false`: a compilation's scenes are other recordings).
- **Track switches.** When the tracker swaps two people (crossing, occlusion), a face on the track that contradicts
  the bound person (below 0.15) releases it at once and withdraws the binding within the track back to the last frame
  that confirmed the person; the track is split there (`track.split`, `reason: face`) and the new part is bound on its
  own. A sustained jump in the track's appearance (two looks in a row below the model's `body_reject` against its
  recent look, agreeing with each other) splits it too (`reason: appearance`); one odd look (an occlusion) does not.
- **Continuation needs evidence.** A new track where the person just was gets the person only on its own face or its
  appearance of this scene; a box alone does not carry a person.
- **Persons file.** `python -m gamst.providers.person_binding RUN.json OUT.json [--names NAMES.json]` writes
  `{start_ms, cuts, persons: [{person_id, name, segments: [{track_id, start, end}]}]}` for a reader with one row per
  person.
- **Two contradicting faces (option `reject_after`).** With `reject_after=2` a bound person is released only by a
  second contradicting face in a row; after the first the track's face is looked at again at the unconfirmed pace
  (200 ms) and a face that does not contradict clears it. The release still reaches back to the last confirmation, so
  the frames between are not kept for the person. Default 1 (the first contradicting face releases, as before).
- **Back over a gap (option `reach_back`).** A track bound again (it held someone before in this segment) reaches back
  to the end of its last binding, as a first binding reaches back to the track's start; never over the person's
  bindings on other tracks or past a face of the track that said someone else.
- **Post mode waits for the scores (`wait_for_scores`, on with `--timing post`).** Each frame waits until the face and
  body scores it asked for are back, so none is replaced or dropped as stale; they are decided on at the next frame.
  Before, a finished file ran the scorer beside the lane as in realtime and lost jobs whenever the models were slower
  than the pose lane (with the face detector: 75 of 396 face jobs replaced on 맨까송). Realtime keeps the scorer
  beside the lane. Measured cost: 10.3 → 10.6 min for the 522 s 관제탑 clip.
- **Within the scene (option `scene_threshold`).** A person the face bound in a scene reaches the scene's other track
  segments that nothing bound, by the appearance of this scene (the looks of the segments bound to them): backward to
  segments that ended before the face was seen (a `person.bound` and a `person.unbound` over that segment's own span,
  `reason: scene_ended`) and forward to later ones (`by: scene`). Never over a cut, never onto a segment whose faces all
  said "not this person" (best below `face_reject`), never over a time the person was on another track. Off by
  default: on the owner's clips it changed nothing measurable ([measurements](measurements/identity-recall-2026-11-28.md)).
- **Limits.** A track switch between two people crossing can move an identity until a face corrects it; retroactivity
  never crosses a track break except by continuation. A person whose face is never seen stays unknown unless the
  appearance of this recording binds them. A face seen badly can start a second person for someone already known (no
  merging yet). The per-track counter does not end a movement at a scene cut. L2 forks share the persons but each
  lane's bindings are kept by that lane; overlap between a live binding and a fork's over a skipped interval is not
  checked.

## MCP

| tool | what |
|---|---|
| `enroll_person` | `{name, consent, images | video {file, start_s, end_s} | camera_seconds | person_id}`: files from the owner's `--clips` folders, or the next seconds of the running camera; the same check; saves the reference in the first `--clips` folder and adds the person to the running binding |
| `set_instruction` | `who`: count the movement for that person; refused with the refusal sentence without a checked reference; `who` alone keeps the current movement |
| `count_events` | `who`: only that person's events (resolved from the bindings, retroactive); `person_id`: one person of the run, named or not |
| `get_timeline` | events of a track show `who` and `person_id`; with `who` (or `person_id`) only that person's events are left and `who.seen` says when the person was on screen |
| `list_instructions` | `persons`: enrolled names, whether usable, what identifies them |

The binding runs also without `--person`, so an LLM can enroll a person and set `who` on a run that started with
nobody (measured on the shop clip: [measurements](measurements/person-identity-2026-11-27.md) section 9).
