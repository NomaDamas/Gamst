# Sensory context through Python, MCP and WebSocket

Applications register capability contracts and provider factories explicitly. Discovery describes registered metadata and active lanes; it never imports a model, downloads weights, starts a worker or enables a sensor. A registered contract is not proof of installed dependencies or validated model support.

The same methods are available on `HumanSense` and as MCP tools:

- `list_capabilities(offset=0, limit=100, max_bytes=65536, revision=None)` returns a bounded page of exact registered contracts, including modality, streaming mode, windows, lookahead, units, model version and license. Registration alone does not activate anything.
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

## Capability discovery pagination

Contracts are sorted by capability ID. Follow `next_offset` with the returned `revision` to read subsequent pages; a changed registry rejects continuation and requires restarting at zero. The revision hashes the complete registered contract metadata, not installed model state. `total_count`, `returned_count`, `truncated` and `entry_byte_budget` expose omissions explicitly. Limits and entry byte budgets use the same bounds as evidence queries. Whole contracts are never shortened; if the next contract cannot fit, the call fails with its required byte count so callers can increase the budget. Container metadata is additional to the entry budget. Empty registries return an empty terminal page.

Discovery does not retain a server-side pagination session or import providers. Sensor status remains a separately bounded-by-runtime-capacity snapshot; byte-budgeted status pagination is still open in #11.

## Passive per-connection cadence profile

The custom WebSocket method `humansense/context_profile` takes an empty parameter object (or no params) and a JSON-RPC request ID. It reports only this connection's current subscription; it is not an MCP tool and does not expose another client's profile. A newly connected/unsubscribed client has no configured interval or recommendation. Reconnecting or successfully changing subscriptions starts a new profile. Invalid subscription updates preserve the existing profile and schedule.

The result separates `requested_interval_ms`, `applied_interval_ms` and `recommended_interval_ms`. The first two remain the user's accepted idle delay **after a send completes**. They are not a promised snapshot-to-snapshot period. `work` gives rolling p50/p95 time from `get_context` through serialization, send-lock waiting and WebSocket send completion. It excludes sensor inference and subsequent network/LLM processing. `observed_completion_intervals` describes actual gaps between locally completed sends, including idle delay and scheduler delay. Send completion does not establish remote receipt or comprehension.

After at least eight completed deliveries, a simple heuristic recommends an idle delay at least as long as rolling p95 work time, within 50–60000 ms. This aims to avoid spending more than approximately half the publisher's time generating/sending snapshots under stable conditions; it is not a measured capacity guarantee. `recommendation_exceeds_supported_range` flags work that would require a larger delay. The implementation retains at most 64 work samples and 64 completion gaps, plus counters; it does not retain media, context bodies or subject IDs. `running` and `failures` describe the publisher. Cancelled/incomplete sends are not counted as completed samples.

No endpoint probe or sensor activation is performed. `consumer_comprehension` and `consumer_input_capacity` remain `unknown`; `automatic_application` is false. To apply a recommendation, the client explicitly submits a new subscription. This implements passive transport profiling only. Hardware/model/backend/precision-keyed profile caching, explicit capability handshake and opt-in bounded synthetic endpoint probes remain open in #12. This is separate from sensor calibration and statistical confidence calibration.

```json
{"jsonrpc":"2.0","id":10,"method":"humansense/context_profile","params":{}}
```

Validation includes bounded deterministic profile calculations, invalid reconfiguration, unsubscribe reset, and real loopback WebSocket profile queries before subscription and after reconnect. Generated contexts and controlled transports are software evidence, not consumer LLM comprehension validation.
