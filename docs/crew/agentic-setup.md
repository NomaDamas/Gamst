# Agentic GAMST setup

GAMST (`gamst.crew.setup`) can turn high-level perception goals into an explicit, inspectable
install plan. `plan` is read-only. `apply` installs only that plan, verifies every
downloaded file, and writes the profile consumed by GAMST. Resource totals are
planning reservations, not a benchmark or host-acceptance claim.

## Roles and candidates

| Role | Candidates (default is hardware-dependent) | GAMST provider / integration |
|---|---|---|
| `vision.detect` | `rfdetr.nano`, `rfdetr.large`, optional `yoloe.26s` / `yoloe.26x`, `custom` fine-tuned Nano checkpoint | `rfdetr.nano`, `rfdetr.large`, `yoloe.local` |
| `vision.track` | `opencv.flow` | `opencv.flow` |
| `vision.rotation` | `gamst.rotation`; no learned model, GAMST's own per-frame tracker over detections, an ROI or motion regions | `vision.rotation` |
| `vision.pose` | `mediapipe.pose_lite`, `mediapipe.pose_heavy` | `mediapipe.body.video` |
| `vision.hand` | `mediapipe.hand` | `mediapipe.hand.video` |
| `vision.face` | `mediapipe.face` | `mediapipe.face.video` |
| `vision.depth` | `depth_anything_v2.small`, `video_depth.large` | `depth_anything_v2.small`, `video_depth_anything.large` |
| `vision.semantic_scene` | `smolvlm.500m`, `qwen3.5.9b`, `qwen3.5.endpoint` | SmolVLM direct GAMST adapter; `streamingvlm.local` for Qwen/endpoints |
| `vision.physical_concepts` | `pg.instructblip` | `pg.physical.local` |
| `audio.vad` | `silero.vad` | `silero.activity` |
| `audio.asr` | `elevenlabs.scribe_v3_live.hosted` (default, streaming, hosted), `sherpa.zipformer_ko_streaming`, `voxtral.mini_4b_realtime_2602.mlx`, `qwen3.asr_1_7b`, `faster_whisper.small` and others (`gamst recipe roles`) | `StreamingWords` in the GAMST process; `streaming.asr.words`; `qwen3.asr`; Faster-Whisper direct plugin |
| `audio.events` | `yamnet`, optional GPL-licensed `ced.base` | `yamnet.events`, `ced.onnx` |
| `audio.diarization` | `nemotron3` (speaker activity slots, not identity) | `nemotron3.diarization` |
| `audio.prosody` | `dsp` pitch/energy/rate features | `audio.prosody` |
| `audio.vocal_expression` | `emotion2vec.seed` | `emotion2vec.seed.onnx` |
| `physio.rppg` | `factorizephys`; an estimate, not a medical measurement | `factorizephys.local` |

The registry in `gamst/crew/registry.py` records each candidate's packages,
weight URL/digest status, devices, RAM/VRAM/disk reservations and their source,
license, and verification status. `VERIFY.md` measurements are from the RTX PRO
6000 Blackwell 96 GiB host, not promises for this computer. Values without a
measurement source are estimates. Several models need license review; CED is
never a default, Video Depth Anything Large is non-commercial research only,
and model estimates are not physical or medical measurements.

## Hardware tiers and defaults

| Tier | Detection | Semantic scene | ASR |
|---|---|---|---|
| `cpu-only` | RF-DETR Nano | SmolVLM2 500M (CPU) | Scribe v3 Live (hosted) |
| `apple-mps` | RF-DETR Nano (MPS) | SmolVLM2 500M (CPU) | Scribe v3 Live (hosted) |
| `cuda-lt24gb` | RF-DETR Nano | SmolVLM2 500M (CPU) | Scribe v3 Live (hosted) |
| `cuda-ge24gb` | RF-DETR Large | Qwen3.5-9B | Scribe v3 Live (hosted) |
| `cuda-ge80gb` | RF-DETR Large | Qwen3.5-9B | Scribe v3 Live (hosted) |

The ASR default is the streaming ElevenLabs Scribe v3 Live on every tier (owner decision: the product path is a
streaming recogniser; Whisper is a non-default choice). Setup installs only its WebSocket client. The audio leaves the
machine and is billed ($0.22 per hour, 2026-11-26), so nothing is sent without `--allow-remote-asr` (or
`allow_remote: true`) and `ELEVENLABS_API_KEY`; without the opt-in a run stops before it starts and names the local
choices (`--asr sherpa.zipformer_ko_streaming`, `--asr voxtral.mini_4b_realtime_2602.mlx`). Measured on one 54 s Korean
shop clip: [streaming-asr-2026-11-26.md](../measurements/streaming-asr-2026-11-26.md).

Other role defaults are MediaPipe Pose Lite/Hand/Face, OpenCV tracking,
model-free rotation, Depth Anything V2 Small, Silero VAD, YAMNet, Nemotron-3,
DSP prosody, emotion2vec, and FactorizePhys. An override never silently changes
an incompatible model into another model: the plan keeps the selection and adds
a warning. Qwen3.5-9B requires CUDA and at least 24 GiB VRAM. Tossed-coin
rotation requires at least 160 fps capture; about 240 fps is recommended for a
36–40 turns/s coin. Model execution speed does not increase camera capture fps.

Goals expand as follows, with recursive role dependencies deduplicated:

- `rotation`: detection + rotation (the rotation capability tracks every frame itself)
- `speech`: VAD + ASR
- `audio`: VAD + ASR + audio events + prosody
- `scene`: detection + semantic scene
- `depth`: detection + depth
- `body`: body pose only
- `pose`: body + hand + face
- `physical`: detection + physical concepts
- `physio`: face + rPPG
- `all`: every role

## CLI

`gamst setup` runs this in-process (same code as MCP and `gamst recipe setup`):

```sh
gamst setup plan --goals rotation
gamst setup plan --goals speech \
  --override audio.asr=moonshine.tiny --json > plan.json
gamst setup apply plan.json \
  --prefix "$HOME/.local/share/gamst"
```

Use a specific profile path for tests or isolated deployments:

```sh
gamst setup apply pose-plan.json \
  --prefix /tmp/gamst-prefix --profile /tmp/gamst-profile.json
```

For a user fine-tuned RF-DETR Nano-compatible checkpoint, use an existing
absolute path. Planning hashes the actual file and `apply` copies it only after
verification:

```sh
gamst setup plan --goals rotation \
  --override vision.detect=custom:/absolute/models/fine-tuned.pth --json > plan.json
```

Compatible candidates share a venv (for example MediaPipe pose/hand/face), while
conflicting Torch, Transformers and vLLM dependency sets stay isolated.
Installation is idempotent:
matching environment markers and matching existing weight digests are reused.
An existing weight with the wrong digest is never overwritten. Downloads use a
partial file and atomic rename. Applying a plan merges its selected roles into
an existing version 1 profile and leaves other roles untouched.

`apply` refuses unpinned snapshots, missing source URLs, manual-preparation
bundles, wrong Python versions, and plans whose packages or weights differ from
the registry. In particular, a Hugging Face commit alone is not a per-file
SHA256. Qwen snapshots, SmolVLM, Moonshine, emotion2vec export/head preparation,
PG canonical assets, and CED's 527-label JSON require their documented explicit
preparation; the installer does not pretend those bundles are complete. This is
why `plan` can be useful even when `apply` correctly refuses a candidate.

## Recipes

`gamst recipe setup <recipe>` plans exactly the roles its parts name
(`make_plan(roles=..., dependencies=False)`), with the same environments and the same weight root as
`gamst setup` (see [crew.md](../crew.md)). It then runs the same `apply_plan` and writes the same profile; `gamst recipe run` reads
the role entries from that profile.

## MCP

The tools are on the one GAMST MCP server (`gamst mcp`):

- `plan_setup`: `{ "goals": ["rotation", "speech"], "overrides": {"audio.asr": "sherpa.zipformer_ko_streaming"} }`,
  or `{ "roles": {"vision.pose": "mediapipe.pose_full"}, "dependencies": false, "isolation": "current" }`
- `apply_setup`: `{ "plan": PLAN_OBJECT, "prefix": "/optional/prefix", "profile": "/optional/profile.json",
  "models_dir": "/optional/weight/root" }`, or supply `plan_path` instead of `plan`.

Both tools have the same validation and refusal behavior as the CLI.

## GAMST profile schema

Default path: `~/.config/gamst/profile.json`.

```json
{
  "version": 1,
  "hardware": {
    "os": "darwin",
    "arch": "arm64",
    "python": "3.11.9",
    "accelerator": "mps",
    "tier": "apple-mps",
    "ram_bytes": 17179869184,
    "vram_bytes": 0,
    "disk_free_bytes": 8000000000
  },
  "roles": {
    "vision.pose": {
      "model": "mediapipe.pose_lite",
      "provider_profile": "mediapipe.body.video",
      "python": "/prefix/venvs/mediapipe/bin/python",
      "config": {
        "path": "/home/me/.cache/gamst/models/mediapipe.pose_lite/pose_landmarker_lite.task",
        "sha256": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
      }
    }
  }
}
```

`provider_profile` and `python` may be `null` for built-in reducers or direct
GAMST adapters. Role configuration is provider-specific and always references
verified local artifacts when the provider requires them.
