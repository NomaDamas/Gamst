# Nonterminal Silero/Qwen utterance endpoints

`SileroEndpointASR` composes the concrete `SileroActivity` and `QwenStreamingASR`
providers. An application explicitly loads both in a compatible runtime and gives
the composition exclusive ownership. It then registers `pipeline.spec` and a
factory returning a fresh pipeline with SensorStreams. Discovery does not load
models. No checkpoint is bundled or downloaded by this composition.

```python
from humansense.providers.endpoint_asr import SileroEndpointASR

# asr and vad are already explicitly loaded, unused/reset concrete providers.
pipeline = SileroEndpointASR(asr, vad, minimum_speech_ms=160, silence_ms=640)
streams.register(pipeline.spec, lambda: pipeline)  # one stream only for this instance
await streams.open('microphone-source', 'microphone-sensor', [pipeline.spec.id])
# Submit captured InputFrames normally; VAD finals do not close this stream.
```

The pipeline fingerprint includes both model versions, Qwen hop/chunk/segment and
language configuration, Silero thresholds, endpoint durations and policy version.
Code/model licenses remain separate: Qwen is Apache-2.0, Silero MIT. See their
individual local runtime guides for trusted file checks and dependency pins. The
combined Torch/vLLM/device runtime has not been executed; compatibility and real
quality/latency evidence are required in #68/#76. Direct instances run off the
SensorStreams event loop, where a hung native call cannot be forcibly timed out.
The explicit `qwen3.asr.endpoint` ProcessProvider profile adds an owned child with
startup/inference timeouts; see the activation example below.

## Capture-time policy

Normalized float32/float64 and signed int16 mono arrays are accepted under the
Qwen fixed-hop contract. PCM16 is converted on a bounded copy using /32768 before
either branch, so VAD and ASR receive the same amplitude scale. Caller-owned
arrays are not mutated.

All accepted PCM goes to Qwen, including leading/trailing silence. There is no
speech-only gating, preroll buffer, extra audio queue or audio cleanup. Ambient
consumers can use their own unchanged lane. This avoids dropping boundary audio
but does not prevent ASR hallucinations on silence or reduce silence inference cost.

Silero applies its configured start/stop thresholds to non-overlapping 32 ms
windows. Consecutive active windows must reach `minimum_speech_ms` to arm an
endpoint. Before arming, any inactive window resets that active run. After arming,
a consecutive inactive run of `silence_ms` triggers a final. Brief pauses are kept
within the same utterance. Durations must be multiples of 32 ms: speech 32..2000,
silence 32..5000. Defaults are 160/640 ms. The existing maximum Qwen segment length
still applies even if the endpoint policy never arms.

The selected final includes the entire input hop in which the qualifying decision
becomes available. Qwen hops must be 10..32 ms, with its existing divisibility
constraints; default is 20 ms. Thus a decision ending at 128 ms may finalize at
140 ms. This is a hop boundary, not a sample-accurate speech boundary. Trailing
silence is included. Additional capture quantization is less than one input hop;
inference scheduling and context-delivery latency are separate and unmeasured.

A final resets VAD policy/model state and its incomplete analysis window. Those
pending samples already reached Qwen; no ASR samples are discarded. Subsequent VAD
windows realign to the next hop, and the next Qwen hop begins a distinct utterance
ID with contiguous capture time. The existing source/sequence continuity is kept.
A gap, invalid source, invalid hop or backend exception resets both providers;
SensorStreams interrupts the affected partial through its existing error path.

## Manual boundary and ordering

Direct provider callers may use `pipeline.process(frame, manual_endpoint=True)`
(or `asr.process(frame, endpoint='manual_endpoint')`) to end after that exact hop.
This is not a queued remote control. Do not call it concurrently with a managed
lane: SensorStreams owns that provider. A managed VAD endpoint needs no manual
call. The terminal `streams.finish()` remains available for end of input.

Each accepted hop returns at most one partial or final revision. An endpoint
finalizes before creating the would-be partial, so no unpublished revision is
skipped. At a simultaneous maximum-segment boundary, `segment_limit` wins; otherwise
manual intent wins over VAD. Final records carry the declared reason. A later
empty `finish()` emits nothing. These are software control decisions, not evidence
that the person completed a sentence or intended to give up their conversational turn.

## Evidence and remaining work

`python3 -m unittest discover -s tests -p 'test_qwen_endpoint.py' -v` uses the real
composition, SileroActivity and Qwen adapter with labeled scoring/engine doubles.
It checks 20/32 ms alignment, exact ASR sample counts, brief pauses/noise/silence,
manual/cap precedence, contiguous revisions, source/gap resets, backend errors,
policy identity and two successive finals through SensorStreams and MCP while the
stream stays open. It proves control flow, not VAD/ASR accuracy or native execution.

#76 retains sustained concurrency/overload evidence and actual speech-boundary validation.
#68 retains real Korean/English recognition, false splits/merges, endpoint and
capture-to-context latency, runtime compatibility and resource/process-tree evidence.
No required family or overall goal is complete from these fixture tests.

## Explicit isolated profile

```python
from humansense import ProcessProvider

config = {
    'asr': {'path': '/absolute/qwen-bundle', 'files': qwen_file_sha256_manifest},
    'vad': {'path': '/absolute/silero_vad.jit', 'sha256': silero_checkpoint_sha256},
    'minimum_speech_ms': 160,
    'silence_ms': 640,
}
spec = ProcessProvider.capability('qwen3.asr.endpoint', config)  # metadata only
provider = await ProcessProvider.open(
    '/absolute/selected-runtime/bin/python', 'qwen3.asr.endpoint', config,
    startup_timeout=120, inference_timeout=30)
# Use this provider in one SensorStreams lane; close it when finished.
```

All four top-level config keys are required; unknown keys are rejected. This fixed
profile uses Qwen 20 ms hops, 2000 ms decode chunks, 30000 ms cap and automatic
language, with Silero .6/.35 activity thresholds. Its schema/fingerprint is identical
to an in-process composition with those exact settings. The parent must have the
same SDK version as the selected child interpreter. Discovery neither reads files
nor imports optional ML dependencies. See [Qwen local runtime](qwen-local-runtime.md)
for the fixed Qwen artifact layout and source-audited version pins. No additional
Torch/vLLM compatibility claim is made for the composite.

The small Silero artifact is hash-checked before Qwen engine allocation, and again
before its own trusted JIT load. Qwen performs its existing manifest/offline/version
validation. A failed second load or mismatching constructed contract closes already
created providers. Child startup failure or timeout also invokes ProcessProvider's
owned-process teardown. Python close drops engine references; it does not prove
native GPU allocations are synchronously released. Actual native descendant-process
and GPU/resource behavior remains a device acceptance gate.

The child alone receives the Qwen profile's offline/telemetry flags and disabled
vLLM V1 multiprocessing setting; parent environment is unchanged. Environment flags
are not a network sandbox. JIT loading requires an explicitly trusted artifact;
SHA-256 integrity alone does not establish trust or model authenticity. No implicit
download occurs in this loader.

Profile tests verify metadata-only discovery in a clean Python process, strict
config validation, pre-load hash rejection and partial ownership cleanup with
explicit loader doubles. A real Python child also executes the concrete endpoint
composition over framed PCM16 IPC using labeled engine/scoring doubles; this verifies
pipe integration and finalization, not real Torch/vLLM/model execution.

## Provider-level endpoint without another audio hop

`QwenStreamingASR.endpoint()`, `SileroEndpointASR.endpoint()` and
`await ProcessProvider.endpoint()` flush only the current utterance with
`finalization_reason='manual_endpoint'`. They do not close the provider or insert
silent/synthetic PCM. Empty or repeated endpoints return None. The next sequential
capture hop starts a new utterance; capture ownership/order checks remain active.
The composite resets its VAD analysis window and endpoint policy after the flush,
without discarding ASR audio. A simultaneous completed hard-cap final remains final;
a later endpoint has nothing to finalize.

The worker operation is serialized with other pipe requests and uses the existing
inference timeout and bounded context-response validation. An unsupported provider
or old worker rejects the operation and the parent closes its owned child; it does
not silently report success. Error, timeout and cancellation also invalidate that
worker, requiring explicit reopen. Parent/worker SDK versions must match.

These provider primitives are also exposed through the [managed endpoint control](managed-endpoints.md).
Do not call a managed lane's provider directly while its queue is processing.
A direct application must serialize calls and publish non-null results at the last
capture timestamp using a distinct monotonically increasing **output** sequence.
Input sequences still describe actual captured hops; do not inject a fake input to
make sequence numbers advance. The managed API implements separate output/capture sequences and serialized
control; direct applications must still provide this ordering themselves. The normal terminal
`finish_sensor_stream` MCP tool still closes the selected stream.

Tests exercise endpoint/next-input continuity, actual child-pipe dispatch with the
concrete composite plus engine doubles, unsupported operation, invalid output,
timeout/cancellation teardown, and manual final publication through transcript
memory/MCP with separate output sequences. These remain control-flow evidence.
