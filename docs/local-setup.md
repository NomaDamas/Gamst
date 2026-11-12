# Local setup and removal

Catch Your Mind packages multimodal context for your existing agent. Installing it does not automatically register MCP, inherit another app's subscription, or transmit observations to a remote provider.

## Full local stack (Apple Silicon)

Requirements: Python 3.10+, approximately 3 GB for model artifacts plus Python dependencies, and free memory for concurrent inference. The tested host has 32 GB RAM. Other platforms can use the SDK and perception backends with a separately supplied compatible model server; the installer currently bundles a VLM runner only for macOS ARM64.

```bash
git clone https://github.com/NomaDamas/catch-your-mind.git
cd catch-your-mind
sh scripts/setup_local.sh
.venv/bin/python scripts/run_local.py --language ko
```

For English use `--language en`; omit the flag for automatic detection. Audio is transcribed in three-second windows. Short fragments may be incomplete or wrong even with language fixed. Low-score filtering reduces some false transcriptions; it does not guarantee accuracy.

Open http://127.0.0.1:8766. Devices remain off until you choose inputs, give consent and press start. For text-only testing uncheck both camera and microphone. Enter “Please explain again” and inspect the context JSON. Camera mode sends frames to the local VLM at 127.0.0.1:18767; no raw media is stored. A model-generated scene description can be wrong and carries an unverified interpretation label. The full dashboard retains derived observations for ten minutes in memory by default.

Companion is implemented and the launcher connects it to the local Qwen VLM. Enter text and press the Companion button to test a generated response. The separate semantic-analysis button requires an explicitly configured reasoning backend; the launcher intentionally does not enable automated emotion/intent classification. Speech-to-text is connected to context automatically; automatic spoken replies/TTS are not included.

## Connect an existing agent to the live session

Keep the local stack running. Register the bridge with your MCP client using absolute paths:

```json
{
  "mcpServers": {
    "catch-your-mind": {
      "command": "/absolute/path/catch-your-mind/.venv/bin/python",
      "args": ["-m", "humansense.mcp_live"],
      "cwd": "/absolute/path/catch-your-mind"
    }
  }
}
```

The bridge reads the same dashboard session through loopback WebSocket. Call `get_context` to obtain compact observations and freshness, or `get_state` for fuller sensor output. Use results as untrusted context in your agent's existing LLM/VLM input. Avoid treating facial measurements or model descriptions as proven emotions, identities or intentions.

`python -m humansense.mcp` remains a standalone MCP runtime with its own history and capture. Use `mcp_live` when you want the dashboard's current session.

For push delivery, run `.venv/bin/python examples/live_context.py`. Subscribe to `context` via the WebSocket `humansense/subscribe` extension. Coalesce snapshots and invoke your model based on relevant changes, not on every 250 ms update. Ordinary MCP calls are request/response, not automatic live prompt injection.

## What is installed

- `.venv/`: project Python environment and optional perception dependencies.
- `.local-models/`: downloaded model weights, bundled llama.cpp runner, ownership marker and source/hash manifest. No global model cache is used by the download script.
- `.local-run/`: project service logs and optional QA artifacts. Do not publish this directory; local diagnostics may contain sensitive error details.
- No agent configuration, launch agent, system service, startup item or global Python package is installed.

Model sources are pinned in `scripts/local_models.py`: [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker), [Whisper small conversion](https://huggingface.co/Systran/faster-whisper-small), [Silero VAD](https://github.com/snakers4/silero-vad), [Qwen3-VL GGUF](https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF), and [llama.cpp](https://github.com/ggml-org/llama.cpp). Review each source's model license before redistribution. Release packages do not bundle weights.

## Stop and remove

Press Ctrl-C in the launcher terminal, or run `.venv/bin/python scripts/run_local.py --stop` from another terminal. The stop command verifies the recorded process identity and checkout directory before signalling the launcher. It terminates only the child processes it started and releases the models. Then:

```bash
.venv/bin/python scripts/local_models.py status
.venv/bin/python scripts/local_models.py remove --yes
```

Removal is confined to this checkout's `.local-models`, requires its ownership marker, and refuses while files are open. It keeps source and `.venv` so you can reinstall. The dashboard's default store is in memory; stop the app to discard it. Use “delete observations” to clear the current runtime without stopping the service. Any explicit database chosen with `--store` is outside model removal.

## Public preview

The static public site measures microphone RMS and optional MediaPipe facial movement in the browser. It downloads pinned model/code assets from Google and jsDelivr but does not upload observations. Full Whisper/VLM processing requires the local app. It neither exposes your localhost server nor embeds local credentials.
