# Realtime context contract

The runtime is an observation layer. Fast sensor measurements continue while bounded background workers run speech transcription and optional local VLM scene description.

- **Audio:** browser sends half-second mono 16 kHz windows. RMS and VAD run on receipt. ASR accumulates three seconds and runs in a queue capped at four incoming windows. On overflow it drops old input and resets partial audio to avoid joining disconnected speech.
- **Video:** browser samples at up to 4 Hz with at most one HTTP frame request in flight. Face and person measurements run independently from a one-frame VLM queue. VLM starts at most once every three seconds; older pending frames are replaced.
- **Delivery:** authenticated local dashboard NDJSON and loopback WebSocket context snapshots are delivered at approximately 4 Hz. This is not a 250 ms end-to-end inference guarantee.
- **Aggregation:** the full dashboard enables `sensor_inference_interval_ms=250` and ten-minute retention. Every signal and speech transition is retained, while full sensor-derived inference is sampled. Text turns still infer immediately. The SDK default remains per-signal inference for compatibility; enable the interval for high-rate capture.
- **Freshness:** each compact observation includes capture timestamp, age, `fresh` (at most five seconds old), evidence ID and model provenance. Consumers must account for stale or unavailable modalities.
- **Memory:** raw frames are held only for processing. Compact face context includes eight strongest blendshapes per face and omits landmark arrays. Raw images/audio and arbitrary structured signals are excluded from the agent context packet.
- **Stop/delete:** queued jobs are cancelled and generations invalidated; late model results cannot repopulate stopped/deleted observations. Native inference may finish internally before releasing its resources.
- **Transparency:** worker queue depth, drops and last nonempty-result processing time are exposed in dashboard state. VLM descriptions are model-generated and unverified. Context itself does not execute tools or determine inner emotion.

## WebSocket

The full launcher exposes the dashboard's exact runtime at `ws://127.0.0.1:8765`. Native local clients are accepted; browser origins are not implicitly permitted.

```json
{"jsonrpc":"2.0","id":1,"method":"humansense/subscribe","params":{"types":["context"]}}
```

Notifications use `notifications/humansense/context`. Other subscriptions are `event`, `signal`, and `inference`. Runtime records are bounded; consumers should retain only their newest needed context. A slow client is not entitled to every historical video frame.

For standard MCP clients use `humansense.mcp_live` as a stdio bridge, then call `get_context`. Registering an MCP server does not automatically add context to every LLM request; the host agent must call the tools or integrate a streaming client.
