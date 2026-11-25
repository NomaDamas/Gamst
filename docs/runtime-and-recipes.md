# Direction: an observation runtime with executable recipes

Status: owner direction, recorded 2026-11-24 after the coin-flip demo. It complements
[positioning.md](positioning.md) by saying what GAMST must own so that it is worth
installing in the age of agentic coding.

## The question

An agent such as Claude can write a bespoke pipeline in minutes, for example: capture a webcam, run optical flow,
detect an audio onset, cut the last 5 s, call a VLM and print JSON. So why install GAMST?

## Answer

If GAMST only helps assemble multimodal pipelines (camera wrappers, VLM wrappers, prompt templates, examples), it is
not worth installing. That is code an agent generates per app.

GAMST is worth installing only as a **high-frequency observation, event and evidence runtime** that agents use. The
SDK and MCP are interfaces to that runtime, not the product. Its value is not that nobody else can build it, but that
nobody has to rebuild it, differently and with new bugs, in every project.

The VLM does not get reflexes. GAMST adds the sensing and memory layer in front of it:

```text
high-frequency observers -> state and events on one time axis -> slow model only when needed -> grounded answer
```

GAMST answers four questions for the model:

- what is changing now;
- when an important event started;
- which evidence the model should see;
- what happened around that event.

When VLMs get faster and cheaper, a pure cost-saving layer is absorbed. A shared time axis, event schema, evidence
and replay are not.

## What the runtime must own, and where GAMST is today

| Runtime responsibility | Today in this repository |
|---|---|
| One time axis across sensors (capture timestamps, sequences, epochs) | Yes: `SensorMetadata`, capture clocks; the iPhone stream carries sensor timestamps per frame |
| Frame drops, lag and bounded queues | Yes: SensorStreams lanes with queue/drop/latency metrics (`get_stream_status`) |
| Different rates per sensor and model | Yes: per-lane providers; 240 fps capture with counting after the take |
| Slow-model calls gated by events, not polling | Yes: Watches, CallController |
| One record schema for every observation | Yes: registered observations; e.g. `vision.rotation.refined` |
| Evidence around an event, kept with the record | Partial: a take keeps `raw.gmst` and the followed trace (`refined.json`, `refined.mp4`); no general "clip around event" yet |
| Replay of a recording to the same result | Partial: `--iphone-replay`, and `refine` reproduces counts from `raw.gmst`; no generic `gamst replay RUN` |
| Benchmark against ground truth | Missing as a tool: human counts and timings are recorded by hand in `docs/halo-coin.md` |
| Swap the model without touching app code | Yes for consumers (Gemini/OpenAI-style providers, MCP clients) |
| Local-first retention and deletion | Yes: bounded retention, `delete_data` |
| Hardware requirements checked before a run | Yes: `gamst recipe check`; frame rate against the sampling limit |

## Executable recipes, not blog recipes

A recipe is not prose that says how to wire things. It is a file the runtime runs, replays and scores.

What exists (2026-11-25), in `gamst/recipes/<name>.json`:

```bash
gamst recipe setup coin-flip            # install only the parts the pipeline uses
gamst recipe check coin-flip --fps 240  # can this camera count the recipe's speed?
gamst recipe run coin-flip --iphone 9123
gamst recipe run coin-flip --take TAKE_DIR
gamst recipe check rotation --symmetry 3 --max-rpm 1500 --fps 240
gamst recipe run rotation --take TAKE_DIR --roi .3,.2,.4,.6 --symmetry 3 --max-rpm 1500
```

Still missing:

```bash
gamst replay runs/<run>                 # same input, same records
gamst benchmark runs/<run> --truth truth.json
```

A recipe declares:

- hardware: the camera frame rate it was measured with, the camera and the compute;
- a pipeline of existing GAMST parts, one per step: capture, find, follow, count;
- the records it emits;
- what it was measured against.

The runtime stays one runtime. A recipe only chooses and connects its parts:

- `setup` installs the union of the parts' extras, pinned weights and pinned source checkouts, and nothing else;
- `run` starts the existing runtime with those parts.

The sampling check is physics, not tuning. An object whose look repeats `symmetry` times per turn is countable only
below `fps / (2 × symmetry)` rev/s:

- a flipping coin's outline repeats every half turn (symmetry 2): countable below `fps / 4`;
- a 3-blade fan turning in the image plane (symmetry 3): countable below `fps / 6`, so 40 rev/s at 240 fps.

`check` and `run` refuse a known frame rate below that limit. The runtime marks counts above it `undersampled`.

`skills/<name>/SKILL.md` stays as the agent-facing pointer: it tells an agent which recipe to run and how to read the
results over MCP:

- `skills/gamst-coin-flip` points at the `coin-flip` recipe, measured against a person's counts;
- `skills/gamst-rotation` points at the `rotation` recipe, measured on synthetic fans and two real stock clips, not
  yet against a human count.

The first recipe is **What 1 fps misses**: a high-frequency physical event that a 1 fps model cannot see. The coin
flip is its first instance:

- Gemini Live (1 fps input) answered "Zero".
- GAMST counted 14.4 flips from 114 frames.
- A person counted 15.

## How we decide whether the SDK deserves to exist

Compare two set-ups for the same task: a new developer who asks Claude to build the pipeline directly, and the same
developer installing GAMST and running the recipe. GAMST must win on these, or we stop calling it an SDK and let
agents generate the code:

1. Runs end to end within 30 minutes of install.
2. Reproduces the same records from the same recording (replay).
3. Keeps application code unchanged when the model or provider changes.
4. Makes event evidence and replay easier to inspect than the bespoke version.
5. Serves a second, different project without changes to the runtime.

## Order of work

1. Keep one runtime: capture, timestamps, ring buffer of recent raw media, event schema, gated model calls, replay.
2. Make one strong recipe executable (`What 1 fps misses`, coin flip first) with `run`, `replay` and `benchmark`.
3. Run the comparison above against a Claude-built pipeline and record the result here.
4. Only then add the second recipe.
