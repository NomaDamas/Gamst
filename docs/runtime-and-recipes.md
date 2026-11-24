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

## Executable recipes, not blog recipes

A recipe is not prose that says how to wire things. It is a file the runtime runs, replays and scores:

```bash
gamst run recipes/coin-flip.yaml        # capture -> observers -> events -> records + evidence
gamst replay runs/<run>                 # same input, same records
gamst benchmark runs/<run> --truth truth.json
```

A recipe declares four things:

- inputs: sensors, rates and crop;
- the observers and refiners to run;
- the events and records to emit;
- what to compare against ground truth.

`skills/<name>/SKILL.md` stays as the agent-facing pointer: it tells an agent which recipe to run and how to read the
results over MCP. Today `skills/gamst-coin-flip` is that pointer, but the recipe behind it is still a set of demo
flags, not a file.

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
