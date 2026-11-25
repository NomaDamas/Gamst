# MediaPipe Framework graph as a Pipeline module

> **Status (2026-11-25): the C++ side is source only. It has not been compiled or run**, and no MediaPipe graph
> has been loaded through it. The build waits for the owner's disk decision (external disk, cleaned space, or a GPU
> server). What is tested is the Python side, against a Python stand-in for the runner
> (`tests/fake_mediapipe_runner.py`). No accuracy or speed is claimed for the graph.

`gamst/mediapipe_graph/` runs one user-composed MediaPipe `CalculatorGraph` (C++) as **one module** of
`gamst.pipeline.Pipeline`, in its own process. MediaPipe schedules the calculators inside the graph; GAMST decides
what feeds the graph, which branches are on, and what becomes context.

```
input (camera, file, take) -> Pipeline lane (queue=N) -> gamst_graph_runner (C++ process, MediaPipe graph)
                                                          -> packets (JSON) -> GAMST records -> Watches, MCP
```

In GAMST's layers the runner is perception. Records are the only thing that leaves it: images, masks and unknown
packet types never do.

## What overlaps and what is added

| MediaPipe Framework | GAMST | How they connect |
|---|---|---|
| packet timestamps (µs) | capture clock (`timestamp_ms`) | packet timestamp = `timestamp_ms * 1000`. Ties inside one millisecond get +1 µs and map back to the exact millisecond. The graph never uses a clock of its own. |
| stream sync by timestamp | capture-time records, `SensorStreams`, record links | `GamstFrameCalculator` bundles every output of one timestamp. A record link now carries the **record's** capture time, so a module downstream is not given a later frame's time. |
| `FlowLimiterCalculator` | bounded Pipeline lanes (`queue=N`) | The lane is the only GAMST queue and the only place frames are normally dropped (`dropped` in the status). GAMST keeps at most `max_in_flight` frames (the FlowLimiter's side packet) in the runner. If the graph still drops a frame, its record says `processed: false`. |
| graph composition (`.pbtxt`) | Pipeline composition | The graph is one module; its records feed other modules through `consumes`. |
| graph start / close | module lifecycle | `start`, `reset` (fresh run), `close`; switched by `Pipeline.enable/disable`. |

Added by MediaPipe (in graph and build sources; not yet run here): parallel C++ calculator scheduling, TFLite
inference calculators, one-euro landmark smoothing, landmark tracking from the previous frame, and GPU image
operations (Linux only, `holistic_gpu.pbtxt`).

## The bundled graph

`graphs/holistic_cpu.pbtxt` composes MediaPipe's own holistic subgraphs: `PoseLandmarkCpu` →
`HandLandmarksLeftAndRightCpu` and `FaceLandmarksFromPoseCpu`.

- Branches `pose`, `hands`, `face`: a `GateCalculator` before each, fed by `enable_<branch>` (one bool per frame). A
  branch that is off gets no input. Hands and face follow the pose, so `pose` off stops all three.
- Smoothing: pose inside its subgraph (`smooth_landmarks`); hands and face with `LandmarksSmoothingCalculator`. The
  hand and face filter values are the pose ones, **not tuned**.
- Side packets: `model_complexity` (0/1/2), `smooth_landmarks`, `refine_face_landmarks` (478 points),
  `use_prev_landmarks`, `max_in_flight` (1..16).
- One person (the holistic graph tracks one).

Your own graph: declare `input_stream: "input_video"`, optional `enable_<branch>` bool inputs, and end with a
`GamstFrameCalculator` (`FRAME:` the throttled image, `STREAM:<i>:` each output, `BUNDLE:gamst_frame`), wired to the
FlowLimiter's `FINISHED`. Add its calculators to `runner/BUILD`. Map its streams to records with
`streams={'stream': 'record.name'}`.

## Use

```python
from gamst.mediapipe_graph import MediaPipeGraphModule

graph = MediaPipeGraphModule.from_manifest('/path/out/f8ef212d5c96-cpu/manifest.json', branches={'face': False})
pipeline.add_module('pose', graph, consumes='camera', queue=1)        # its own bounded lane
pipeline.add_module('channels', channels, consumes='pose:scene.poses', spec=channels.spec)  # PoseChannels, unchanged
await pipeline.configure('pose', {'branches': {'hands': False}})       # next frame, same run
await pipeline.configure('pose', {'side_packets': {'model_complexity': 2}})  # fresh run
await pipeline.configure('pose', {'restart': True})                    # after a failure, never implicit
```

Without `queue=` each frame is awaited, which is what an offline file needs (nothing is dropped in GAMST). With
`max_in_flight` above 1, a call returns the records of whatever finished; the last frames come back on the next frame
or at `finish`. Records that arrive later than another module's newer record carry the `out_of_order` flag.

Over MCP, `configure_pipeline_module` takes the same `options`. It can only change branches and side packets the owner
declared, with the same type; text side packets (which may be file paths) are fixed and shown as `<text>`.

| Record | Value |
|---|---|
| `scene.poses`, `scene.pose_count` | the shape of `MediaPipePosePlugin` plus `world_landmarks` (metres); `track_id` is null |
| `scene.hands`, `scene.hand_count` | `[{handedness, landmarks (21), bbox}]` |
| `scene.face_landmarks`, `scene.face_count` | `[{frame_face_index, landmarks (468 or 478), bbox}]` |
| `vision.mediapipe_graph.frame` (capability `vision.mediapipe_graph`) | `processed`, `run`, `dropped`, `latency_ms`, `branches`, `counts` |

Landmark arrays are in the Store for other modules; the context packet carries only compact values. A branch that is
off writes no record; a branch that is on and finds nothing writes an empty one. If the runner was not checked against
the pinned revision, the capability version says `unverified-build`.

Recipe part: `mediapipe-graph` (in place of `mediapipe-pose`), with `"queue": 1` on the module item for live input.

## Install (one path: the role registry)

Role `vision.mediapipe_graph`, candidates `graph.holistic_cpu` and `graph.holistic_gpu`. There is no binary to
download, so `gamst setup` refuses it (`automatic=False`) instead of pretending: build it, then give the profile role
`config.manifest`. `from_manifest` checks the MediaPipe commit and the SHA-256 of the runner, every graph and every
model file. Model licences: see the MediaPipe model cards (not reviewed here).

## Build (not done yet)

Pin (`pin.json`): MediaPipe **v0.10.35**, commit `f8ef212d5c962c0e853db7e59d217056b187084b`, Bazel 7.4.1. It is the
release GAMST already uses from Python (`mediapipe==0.10.35`). v1.2.3 exists (2026-11-11) and still has the holistic
module; moving to it is a separate decision.

```bash
export GAMST_MEDIAPIPE_BUILD_DIR=/Volumes/External/gamst-mediapipe   # no default on purpose
gamst/mediapipe_graph/build.sh --check   # tools, OpenCV, free space; downloads nothing
gamst/mediapipe_graph/build.sh           # CPU, macOS or Linux
gamst/mediapipe_graph/build.sh --gpu     # Linux with EGL
```

It needs bazelisk, OpenCV 4 (`brew install opencv`) and python3 with numpy. About 8-10 GB is an estimate, not measured.
The script fetches exactly that commit, puts `runner/` and `graphs/` in it as `//mediapipe/gamst`, builds, and stages
the runner, graphs and model files with a `manifest.json`. `WORKSPACE.snippet` is for building from another workspace.

## Protocol v1

stdin: one text line per command; `FRAME` is followed by its bytes.

| Command | Effect |
|---|---|
| `SIDE <name> <bool\|int\|float\|double\|string> <value>` | side packet for the next run; answered by `ack` |
| `BRANCH <name> <0\|1>` | `enable_<name>` from the next frame; `ack` |
| `START` / `RESET` | start a run / finish the current one and start a fresh one; `ready` |
| `FRAME <capture_us> <width> <height>` + `width*height*3` BGR bytes | one frame; capture times increase within a run |
| `CLOSE` (or end of input) | finish and exit |

stdout: one JSON object per line. `hello` (protocol, MediaPipe revision, branches, side packets), `ready` (run),
`ack`, `packet` (`run`, `ts_us`, `stream`, `kind`, `value`), `done` (`run`, `ts_us`, `processed`: every frame up to
`ts_us` is finished; its packets came first), `error` (`stage`, `code`, `fatal`), `closed`. Landmarks are
`[x, y, z, visibility|null, presence|null]`; a non-finite number is `null` and GAMST then drops that landmark list.
Native error text goes to stderr only; GAMST keeps the status code.
