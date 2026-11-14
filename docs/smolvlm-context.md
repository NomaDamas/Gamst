# Small VLM capture windows as context

`smolvlm2.500m.local` connects the [local SmolVLM engine](smolvlm-native.md)
to SensorStreams, the observation store, Python/API and MCP. It produces a
semantic hypothesis from1..4chronological images. It does not invoke the
consumer LLM, decide when a larger VLM is needed, or change consumer cadence.

## Responsibilities and bounds

Capture supplies original BGR pixels, integer capture milliseconds and
`SensorMetadata` with source/sensor IDs and an explicit image coordinate frame.
The provider owns copies of at most4frames (at most1080p each), validates source,
epoch, image shape, channel metadata and calibration identity, and hashes every
owned original image. Those hashes are evidence identifiers, not stored media;
the caller retains originals separately if reinspection is needed.

The inference window is count-based (`window_frames`, default4, range1..4),
with strictly increasing timestamps, `max_gap_ms` (default2000, range1..10000)
and an absolute10second span limit. `sequence_step` (default1, range1..10000)
allows explicitly declared regular subsampling of original capture indices.
The provider handles ordinary sequence gaps itself so declared strides also
work when original indices are in `SensorMetadata.sequence`. Missing sampled captures (including queue loss) break the declared capture
index step and clear the window. Processing faults force a manager reset.
Unexpected sequence gaps, excessive time gaps, source/epoch/calibration/shape
changes start a new segment. Duplicate or backward time/index and malformed
inputs reject and clear pending provider state. No interpolation or invented
frames/timestamps are used. Caller clocks remain unverified.

The emitted window mode honestly describes multi-image inference, not native
stateful video streaming. There is no future-frame lookahead beyond collecting
the configured window. Inference latency and queue delay are additional.
Full1080p BGR needs6,220,800bytes per frame: explicitly select an8MiB packet
budget instead of the default4MiB. Existing SensorStreams queue/aggregate byte
limits remain enforced. The provider's own retained window is separate (up to
24,883,200pixel bytes before native preprocessing/model allocations).

## Integration

Use an already provisioned dedicated runtime and explicit local asset hashes
from the native-engine documentation. No model installation, Hub request or
camera activation occurs through discovery or configuration.

```python
from humansense import HumanSense, ProcessProvider, SensorStreams

# config contains the native checkpoint/tokenizer paths/hashes and optional
# device/precision/max_new_tokens, plus window_frames/max_gap_ms/sequence_step.
async with HumanSense(consent=True) as human:
    worker = await ProcessProvider.open(
        existing_python, 'smolvlm2.500m.local', config,
        startup_timeout=120, inference_timeout=30,
    )
    streams = SensorStreams(human, max_packet_bytes=8*1024*1024)
    try:
        streams.register(worker.spec, lambda: worker)
        await streams.open('camera', 'video', [worker.spec.id])
        # frame_source supplies InputFrame(..., modality='video', timestamp_ms=...,
        # sensor=SensorMetadata('camera','video',sequence,coordinate_frame='pixels')).
        # Leave stream_epoch=None so SensorStreams assigns its own lifecycle epoch.
        async for captured in frame_source:
            await streams.submit(captured)
        await streams.finish('camera', 'video')
    finally:
        await streams.clear()
        await worker.close()
```

Consumer polling/subscriptions run independently in another task at the user's
chosen cadence, using existing `get_observations`/Python/API endpoints. `submit`
does not wait for VLM inference. Use a distinct owned worker per active source;
concurrent camera streams must not share one process/provider instance. If
queues drop frames, the stream resets state before admitting later observations.
After a worker timeout/error, inspect stream diagnostics and explicitly reopen;
there is no implicit model restart or fallback that hides the failure.

`finish` drains queued frames and describes any final partial window, marking
`final_partial=true` and `complete=false` where appropriate. It retains the
last original capture time; it does not create another camera capture. Normal
clear/reset discards pending images without describing them. Owned-worker
startup/inference deadlines and cancellation use the existing process supervisor.
Clear joins cleanup and prevents late publication; native kernels cannot be
preempted by the synchronous engine itself.

## Observations and freshness

Output separates `semantic`, `window` and `model` objects. It contains original
capture times/indices/hashes, source/epoch/calibration/shape, a segment ID and
canonical window evidence hash; model/revision/checkpoint/tokenizer/prompt/
preprocessing/precision provenance; measured input/output token counts,
truncation and native-call elapsed milliseconds. Cost remains unknown. The
reference revision identifies the intended architecture, not proof that a
caller-supplied checkpoint contains official trained weights.

A new segment emits `collecting` with null text once. During subsequent normal
window collection the provider emits no replacement: the completed observation
keeps its original timestamp, never a refreshed polling time. Source changes or
gaps emit the new collecting state and clear the prior description. Empty
model text is `empty_generation`, not evidence that the scene is empty.

Freshness is1000ms. A slow or old recorded window can therefore be historical
when delivered. Neither timestamps nor freshness are rewritten to imply a
current scene. Consumers must check capture time/freshness and stream errors.
A malformed input/inference failure emits no replacement; previous observations
remain historical evidence with their original time and diagnostics report the
failure. They must not be interpreted as successful new measurements.

All text is untrusted model output. Confidence is null, quality validation is
false, complete-scene coverage is false, and cross-window KV cache reuse is
false. This does not overwrite numeric specialist facts or claim identity,
emotion, force, temperature or a complete inventory of the scene.

## Validation

Run `python -m unittest discover -s tests -p test_smolvlm_context.py -v`.
Deterministic windows test copies/hashes, ordering, epochs/calibration/shape,
subsampling/gaps, malformed input, final partials and empty output. Real owned
child fixtures verify wire→provider→store→MCP, original timestamps, independent
polling during delayed inference, timeout/cancel cleanup and suppression after
clear. Backend doubles are explicitly fixtures, not semantic-quality evidence.

The pinned CI `test_smolvlm_native.py` also executes a generated tiny untrained
SmolVLM through the production loader/worker/context/MCP. Only its architecture
factory is replaced with the tiny fixture; tokenizer/weight hashes, actual HF
preprocessing/generation, process wire and publication are real. This is not
execution of the trained500M checkpoint. Actual weights, held-out semantics,
multimodal/device performance and matched quality/cost remain mandatory
[#150](https://github.com/NomaDamas/Gamst/issues/150),
[#40](https://github.com/NomaDamas/Gamst/issues/40) and
[#57](https://github.com/NomaDamas/Gamst/issues/57).
Policy/sentinel/escalation work remains [#133](https://github.com/NomaDamas/Gamst/issues/133).
