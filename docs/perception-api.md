# Sensory context through Python, MCP and WebSocket

Applications register capability contracts and provider factories explicitly. Discovery describes registered metadata and active lanes; it never imports a model, downloads weights, starts a worker or enables a sensor. A registered contract is not proof of installed dependencies or validated model support.

The same methods are available on `HumanSense` and as MCP tools:

- `list_capabilities()` returns the exact registered contracts, including modality, streaming mode, windows, lookahead, units, model version and license. Registration alone does not activate anything.
- `get_sensor_status()` reports typed SensorStreams managers with stable per-instance IDs, registered factories, active sensor lanes, queue size, drops, reset/error counts and last processing latency. `get_stream_status` remains the separate legacy MediaRouter view.
- `get_observations(subject_id, window, capability_id, source_id, sensor_id, limit, max_bytes)` queries retained registered evidence, newest first. Filters run in SQL before the limit. Legacy unregistered payloads are excluded. Whole records are retained or omitted; `truncated` identifies count/byte omissions. This is history, not a freshness claim.
- `get_transcripts(subject_id, window, limit, max_bytes)` exposes the same revised/final/interrupted utterances used by context, with evidence and source epochs. It does not concatenate obsolete partial hypotheses.
- `get_object_memory(subject_id, text, entity_id, limit, max_bytes)` returns last-seen evidence and visibility. It does not invent a current location or infer real-world identity. `possibly_more` identifies the underlying 100-candidate search cap.

Query entry budgets are explicit (1024–262144 bytes), with at most 100 requested entries. Defaults are 20 entries/65536 bytes for observations and objects, 16 entries/32768 bytes for transcripts. JSON container metadata adds a small amount beyond the entry budget. Object results include their bounded evidence histories. These APIs are read-only and do not implicitly remember arbitrary records or configure model execution. Application registration/configuration remains Python-owned in this change.

```python
contracts = await human.list_capabilities()
lanes = await human.get_sensor_status()
recent = await human.get_observations(capability_id='audio.yamnet.events', limit=10)
transcripts = await human.get_transcripts(subject_id='alice')
remembered = await human.get_object_memory(text='cup')
```

MCP callers use `tools/call` with the same tool names/arguments. The stdio server and live WebSocket MCP bridge use the same tool registry and runtime. No new protocol version or custom MCP streaming guarantee is introduced. Existing access/consent and loopback transport boundaries continue to apply.

## User-defined context delivery cadence

WebSocket clients may configure their own context subscription independently of capture or inference cadence:

```json
{"jsonrpc":"2.0","id":1,"method":"humansense/subscribe","params":{"types":["context"],"subject_id":"alice","window":"5m","interval_ms":100}}
```

The acknowledgement echoes accepted `context` options: subject, window and interval. Interval is an integer 50–60000 ms; window is a positive duration at most one day. Defaults are local-user, five minutes and 250 ms. These options apply to context snapshots only, not event/signal subscriptions. Unknown options and invalid values reject the update while preserving the prior subscription. Sending `types: []` cancels publication. Reconnecting creates a new subscription with defaults; specify options again. Retained runtime evidence survives transport reconnect, but no raw sensor stream is automatically resumed.

Each context publisher produces a snapshot, waits for its send, then sleeps for the configured interval. This provides bounded work and one snapshot in flight per publisher, not a guaranteed wall-clock FPS. Processing time, network backpressure and scheduler delay can lower delivered frequency. The timestamp and freshness fields describe snapshot generation; a slow consumer must compare capture/snapshot times against its own receipt time. There is no accumulating context queue or replay backlog. Event/signal subscribers retain the runtime's separately bounded queue/drop behavior. All sends on one connection share transport backpressure; a stalled client does not stop independent runtime ingestion.

This interval is user-selected, not automatic LLM comprehension calibration. Connection profiling and requested/recommended/applied cadence reporting remain #12. Full model configuration over remote APIs, observability UI coverage and all remaining P09 gates stay open in #11.

## Verification

`python -m unittest discover -s tests -p test_perception_api.py -v` checks nonactivating discovery, Python/MCP equivalence, SQL filtering and truncation, revised transcripts, object evidence, cadence validation/reconfiguration, subject selection and slow-consumer bounds. A real loopback WebSocket test exercises tool calls, context notifications and reconnect with retained transcripts. It runs when the optional `websockets` dependency is installed; CI explicitly installs that extra and runs this test. Other tests use synthetic derived observations and a controlled slow transport. No new model weights are downloaded; no ASR or VLM quality claim follows from transport tests.
