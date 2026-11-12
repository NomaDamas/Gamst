# 🤖 Agent setup: install → capture → inspect → stop

## What is installed?

HumanSense is a Python package (`humansense-sdk`, imported as `humansense`). Its SDK owns processing and SQLite memory. The MCP stdio server, loopback WebSocket service and browser dashboard wrap that runtime. They are not separate neural models. Running separate processes does not automatically share live capture.

Model adapter code is included; weights and inference servers are not. The SDK can ingest text without a camera or model. Media adapters run only after installing their dependencies and supplying valid local weights. Image/reasoning/Companion APIs require an actual configured compatible server. SAM3, biometric identity and automatic speaker-to-face association are not implemented.

## 1. Install on the machine with the camera/microphone

```bash
git clone https://github.com/NomaDamas/catch-your-mind.git
cd catch-your-mind
# Additional model and latency work is on this branch, after it is pushed:
git switch feat/humansense-models-latency
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[vision,audio,yolo,asr,vad]'
```

Omit extras you do not need. Use `.[mediapipe]` for face measurements or `.[onnx]` for contract-specific exported models. Review backend and model licenses separately. Download licensed weights yourself to a known local directory; package installation does not supply them. Do not substitute arbitrary filenames or claim an installed model based on this example.

## 2. Register an MCP server in the agent client

Replace paths with this machine's actual venv/checkout paths:

```json
{
  "mcpServers": {
    "humansense": {
      "command": "/absolute/path/catch-your-mind/.venv/bin/python",
      "args": ["-m", "humansense.mcp"],
      "cwd": "/absolute/path/catch-your-mind"
    }
  }
}
```

The agent client starts the MCP process; do not also start a disconnected capture process expecting it to share memory. Camera/mic remain off until `start_capture`. The server runs on the machine where `command` executes: a client on a Mac does not gain access to a Linux camera merely by using Linux-looking paths. Use an actual remote MCP execution/transport arrangement if the agent client is on another machine.

## 3. Start collection with participant consent

Agent MCP tool call `start_capture` arguments, with actual existing model paths:

```json
{
  "consent": true,
  "video": 0,
  "audio": true,
  "subject_id": "local-user",
  "fps": 10,
  "yolo_model_path": "/absolute/path/models/person-detector.pt",
  "vad_model_path": "/absolute/path/models/silero_vad.jit",
  "asr_model_path": "/absolute/path/models/whisper-directory"
}
```

Optional `face_model_path` adds MediaPipe landmarks. Use `onnx_model_path` instead of YOLO for an export that meets the documented ONNX contract. Omit model arguments to use basic HOG person detection and audio energy only. Energy is not speech, and the basic configuration has no transcription. A text-only agent can skip capture and call `push_text`.

`subject_id` must be provided/verified by the application. It is not biometric recognition. Multiple people in a frame are scene detections; the microphone is not automatically assigned to one face. Inspect `get_capture_status` and errors before claiming collection is active.

## 4. Query and respond

| Agent tool | Example arguments | Use |
| --- | --- | --- |
| `get_capture_status` | `{}` | Devices, errors, frames, drops and capture timings |
| `get_state` | `{"subject_id":"local-user"}` | Latest measured state and unavailable modalities |
| `get_events` | `{"subject_id":"local-user","limit":20}` | Recent speech/interaction/behavior changes |
| `get_context` | `{"subject_id":"local-user","window":"5m"}` | Window summary and baseline trends |
| `get_metrics` | `{}` | Actual measured SDK/plugin p50/p95/p99 |
| `analyze` | `{"subject_id":"local-user","window":"5m"}` | Explicit reasoning pass if a backend is configured |
| `chat` | `{"subject_id":"local-user","text":"Please explain this briefly"}` | Generated context-aware reply if Companion is configured |

Use SDK async iterators or WebSocket subscriptions for pushed live events. MCP queries read current/recent data and do not automatically invoke analysis on every frame. Capture loops continue until stop/error/pause. Processing rate depends on the model; a requested fps is a target, not a measured guarantee. A slow image-model call also delays that observer's video processing.

## 5. Optional VLM / reasoning / Companion

Supply environment variables in the agent client's MCP server environment (or the launching shell). All model names must refer to models actually installed on your server.

```bash
export HUMANSENSE_VISION_URL=http://127.0.0.1:11434/v1/chat/completions
export HUMANSENSE_VISION_MODEL=YOUR_INSTALLED_IMAGE_MODEL
export HUMANSENSE_REASONING_URL=http://127.0.0.1:11434/v1/chat/completions
export HUMANSENSE_REASONING_MODEL=YOUR_INSTALLED_TEXT_MODEL
export HUMANSENSE_COMPANION_URL=http://127.0.0.1:11434/v1/chat/completions
export HUMANSENSE_COMPANION_MODEL=YOUR_INSTALLED_TEXT_MODEL
```

Vision automatically processes selected video frames with per-stream rate limiting. Reasoning and chat are explicit tools. Local endpoints must accept the documented image/JSON/chat formats. For remote services, HTTPS and the corresponding `*_ALLOW_REMOTE=true` are required; obtain participant consent for image/text transfer and configure keys via `*_KEY_ENV`. Never paste keys into tool arguments or this documentation.

Outputs distinguish measured facts, unverified model interpretations and calibrated predictions. Missing data stays unavailable; model scores are not probabilities about inner feelings. Verify user needs by asking rather than stating an inferred emotion as fact.

## 6. Stop, pause and delete

Use `pause_capture` / `resume_capture` for temporary suspension and `stop_capture` to release devices. `delete_data` removes retained and derived observations and stops an active MCP capture session. External applications, backups and authorized provider retention are outside the SDK deletion boundary.

## Validation status

Adapters and runtime paths are software-tested, including real MCP stdio communication. Actual camera/model inference, SAM3, acoustic speech-end latency and eight-hour soak remain incomplete. See [model requirements](additional-models.md), [measured latency and scope](latency.md) and [completion audit](completion-audit.md). Read an actual host measurement report before claiming any specific model works or meets latency targets.
