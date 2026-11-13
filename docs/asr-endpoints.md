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
quality/latency evidence are required in #68/#76. The current ProcessProvider
profiles do not load this composite; this is an in-process provider run off the
SensorStreams event loop. A hung native call cannot be forcibly timed out here.

## Capture-time policy

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

#76 retains isolated composite profile loading, serialized remote manual endpoint
control, concurrency/overload/clear evidence and actual speech-boundary validation.
#68 retains real Korean/English recognition, false splits/merges, endpoint and
capture-to-context latency, runtime compatibility and resource/process-tree evidence.
No required family or overall goal is complete from these fixture tests.
