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
| whole-body appearance (body and clothes) | OSNet-AIN x1.0, MSMT17 (`osnet.ain_x1_0_msmt17`, MIT code and weights, sha256 published by the authors), converted once to ONNX | only in the video or live run it was seen in: clothes change |
| voice | the voice enrollment (CAM++), kept in the person reference | binds speakers (words), not tracks |

A person reference is one JSON file (embeddings only, never images, mode 600): `face`, `body`, optional `voice`, the
`samples` it came from (each showed exactly one person), the `session` (which recording the body samples came from), the
`consent` record and its `quality`.

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
  0.05; without a face, by appearance (>= 0.8, **provisional**: read from one unlabelled clip) of this run only. A face
  that contradicts (< 0.15) releases the track. A pose the tracker gave no track is never bound. A person moves to
  another track only when that track's face beats the holder's face in the same frame.
- **Where it runs.** The binding module sits in the pose lane (it pairs each camera frame with its joints, and an L2
  chain forks it with the pose model), but no face or body model runs there: crops of the tracks that are due go to a
  scorer beside the lane, and the scores are decided on at the first frame after they return (100 ms median on the
  shop clip). A track's face is looked at every 200 ms while it is unconfirmed and every 1 s once it is bound: every
  track is confirmed, no face on every frame.
- **Tracker swaps.** When the track that holds the person leaves the frame and a never-bound track appears where it
  was within 1 s (box overlap >= 0.3), the person continues on it (`by: continuation`) unless its face or appearance of
  that frame contradicts; the face confirms it like any track.
- **Retroactive.** The first binding of a track reaches back to the track's first frame, but never before the person was
  released from another track (one person is on one track at a time, also in the past). Readers resolve it at read
  time (`count_events`, `get_timeline`), so a count of a person can grow after the fact. A binding ends when its track
  was last seen; at the end of input every open binding is closed.
- **Nobody enrolled.** Without `--person` the binding still runs and binds nobody; a person enrolled while the run goes
  on (MCP `enroll_person`) is bound from then on, and back to the first frame of tracks that are still followed.
- **Limits.** A track switch between two people crossing can move an identity until a face corrects it; retroactivity
  never crosses a track break except by continuation.

## MCP

| tool | what |
|---|---|
| `enroll_person` | `{name, consent, images | video {file, start_s, end_s} | camera_seconds}`: files from the owner's `--clips` folders, or the next seconds of the running camera; the same check; saves the reference in the first `--clips` folder and adds the person to the running binding |
| `set_instruction` | `who`: count the movement for that person; refused with the refusal sentence without a checked reference; `who` alone keeps the current movement |
| `count_events` | `who`: only that person's events (resolved from the bindings, retroactive) |
| `get_timeline` | events of a track show `who`; with `who` only that person's events are left and `who.seen` says when the person was on screen |
| `list_instructions` | `persons`: enrolled names, whether usable, what identifies them |

The binding runs also without `--person`, so an LLM can enroll a person and set `who` on a run that started with
nobody (measured on the shop clip: [measurements](measurements/person-identity-2026-11-27.md) section 9).
