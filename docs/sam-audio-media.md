# Private separation media ownership

This is the software transport/ownership slice of [#221](https://github.com/NomaDamas/Gamst/issues/221), under [#201](https://github.com/NomaDamas/Gamst/issues/201). It does not implement or install SAM Audio. The reserved `sam_audio.text_temporal.local` worker profile fails construction until the separately reviewed native adapter exists. Its discovery schema/configuration hash describes a transport contract, not a supported trained checkpoint. Fixture-engine tests are not model inference or quality evidence.

## Explicit local ownership

Ordinary HumanSense operation retains no separation waveforms. Within the runtime event loop, an application with ingestion consent can explicitly grant one local owner:

```python
owner = human.enable_separation_media(
    subject_id="alice", source_id="microphone", sensor_id="ambient",
    ttl_ms=30000, capacity=8, max_bytes=16 * 1024 * 1024,
)
```

Pass this object as `separation_owner=owner` when explicitly opening the reserved ProcessProvider profile. Every worker needs a fresh grant; factories must not reuse an owner across workers. The owner checks exact subject/source/sensor, provider epoch and HumanSense generation. Its first actual request binds the epoch. A generation change revokes all grants, including grants for another subject, consistent with existing runtime-wide processing reset behavior. A new runtime grant is necessary after revocation.

`owner.retrieve(reference, "target", stream_epoch=epoch)` and the corresponding `"residual"` call are the explicit authorized local Python API. They return immutable float32 bytes only to the exact active owner and epoch. `owner.lookup(reference, stream_epoch=epoch)` returns copied availability metadata. A reference is an identifier; it does not grant authorization. This is a capability boundary inside the application, not isolation against Python code already controlling the process.

No network waveform retrieval endpoint or MCP media tool is installed. Existing API/MCP discovery, observations, history, signals and Context subscriptions carry bounded metadata only. A future network retrieval deployment needs its own explicit authorization/access design; forwarding the owner object or exposing a reference-based unauthenticated download is not supported. Application code must not automatically fetch references into LLM Context.

## Closed private wire contract

Only the reserved SAM profile with an explicit owner can negotiate `sam-audio-pair-v1` during worker startup. Legacy profiles keep the same startup response and reject all nonempty response payloads. Only the reserved profile's `process` operation can return the private binary pair; controls cannot publish media.

Inputs are copied mono little-endian float32 arrays at exactly48000Hz, with48000..192000 samples (1..4seconds), exactly one named sensor channel and a capture-end timestamp covering the complete window. Nonfinite/out-of-normalized-range inputs are rejected. Source identity, capture sequence, epoch, calibration/coordinate metadata, input SHA256, sample count/interval, runtime generation and config identity are bound to the actual request. These are SDK bounds, not measured model limits. No implicit resampling, downmixing or modification of ambient input occurs.

The future native adapter returns `SeparationPair(observation, target, residual, padded_samples)`. Both waveforms are already trimmed, immutable little-endian float32 bytes containing exactly the original number of samples. The child supplies no Context reference. Its original observation must conform to the registered schema and have an empty `media` object. Codec padding is bounded to at most48000 additional samples; native code must separately prove correct codec-hop reconciliation. Target/residual samples must be finite with absolute value at most16; out-of-bound values are rejected, never clipped or normalized. Metadata reports counts outside the normalized[-1,1] range.

Responses declare exactly target then residual, contiguous offsets, exact lengths and SHA256. Maximum waveform response is1536000bytes; headers are limited to16KiB. The parent checks request ID, profile negotiation, binding and declared sizes before reading waveform bytes, then validates layouts/hashes/values before admission. Wrong source/subject/epoch/generation, duplicate/extra components, trailing/truncated data, unsupported encodings and malformed native results invalidate the owned worker. Existing private pipe stdout/log isolation and timeout/cancel/reap behavior remain in force.

Only the parent creates opaque references and overwrites authoritative source-window evidence/padding counts from the actual request. Output is an estimate, not proof of successful source identification or exact additive reconstruction. Target and residual do not imply `residual = input - target`. No native streaming or seamless cross-window audio claim follows from this protocol.

## Memory and lifecycle

Defaults are30seconds TTL,8completed pairs and16MiB shared memory budget. Explicit ceilings are60seconds,32pairs and64MiB. Settings are fixed for the runtime's media store; another grant cannot silently change them. Grants are bounded to32active owners. A configured budget can reject larger windows when its in-flight reservation does not fit.

The same byte budget accounts for retained pairs and conservative parent in-flight reservations of32bytes per input sample (up to6144000bytes for a4second request). This covers the copied pipe input, response bytes, decoded pair and temporary validation copies. Retention evicts complete oldest pairs atomically; it never preserves only one component. SensorStreams' queued/capture snapshots have their existing separate packet/buffer budgets. Child interpreter, native tensor/GPU allocations and model weights are outside this parent media budget and remain the native-runtime acceptance responsibility. No model memory/performance measurement is claimed.

TTL enforcement uses monotonic deadlines and a scheduled expiry callback, including when nothing reads the store. Wall-clock expiry is descriptive metadata only. Metadata says `available_at_publication` and `explicit-owner-lookup-required`; history can outlive waveform retention. A later lookup is authoritative and fails when media has expired, been evicted or been revoked. Owner lookup must not be replaced by trusting a historical availability flag.

Runtime pause, processing reset, data deletion, consent withdrawal and close immediately revoke grants/drop retained bytes before waiting on workers. Stream-clear intent synchronously revokes its provider's owner; provider reset drops previous references and allows rebinding to the new lane epoch. Worker failure, timeout, cancellation or close clears its media. In-flight reservations remain until owned request cleanup completes, preventing a cancellation/clear from prematurely making that memory budget available. Failed observation publication rolls back its newly admitted pair. Already returned immutable bytes are owned by the explicit retrieving application and cannot be recalled; applications must apply their own downstream retention policy.

## Software validation

```sh
python3 -m unittest discover -s tests -p test_separation_media.py -v
python3 -m unittest discover -s tests -p test_process_provider.py -v
```

The focused tests use a real owned subprocess running an explicitly labeled fixture engine. They exercise typed roundtrip, source binding, component validation, malformed/excessive responses, unchanged ambient PCM, owner/epoch access, atomic retention, actual TTL callback, cancellation/reaping, resets/clear, API/MCP metadata-only behavior and progress of an independent sensor while separation stalls. They require no SAM installation, checkpoint, model download or dependency installation. Actual trained inference, gated asset identity, physical/runtime evidence, latency/memory measurements and held-out separation quality remain open under #201/#187/#29.
