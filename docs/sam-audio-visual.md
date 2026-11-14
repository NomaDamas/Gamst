# SAM visual windows: native Context and recorded replay

This implements the bounded visual-window path for [#202](https://github.com/NomaDamas/Gamst/issues/202): packet validation, causal frame selection, the native SAM Audio worker, scoped media ownership, Context/MCP metadata and recorded replay. The existing SAM Audio profile accepts an optional frozen `visual_policy`; no separate visual model is registered. Actual trained separation, device resource measurements and timing acceptance remain in #203/#187.

`humansense.sam_visual_packet.visual_frame` accepts an original mono float32
48kHz audio InputFrame, plain RGB uint8 NCHW frames and uint8 N1HW masks. Its
keyword arguments supply original capture intervals in microseconds, strictly
increasing capture sequences, visual SensorMetadata, track/segment IDs, mask
producer/revision and clock declarations. It returns a copy of the audio frame
whose data is an immutable composite packet. The visual sensor's
`capture_sequence` identifies the first original frame. Channels must be
`('R', 'G', 'B')`. No camera is opened, format converted or model downloaded.

Mask polarity is explicitly `zero-kept`: original RGB pixels under zeros are
retained by the official `frame * mask.eq(0)` operation; ones are erased. All-one
masks are rejected. An all-zero mask explicitly retains the entire frame; this
does not prove that the entire image belongs to the target. Mask and clock
accuracy remain caller declarations, not verified physical facts.

Packets have exact closed fields, per-frame RGB/mask SHA256, a PCM digest and
explicit original source epochs. At most128 original frames, dimensions16..640
by16..480, a64KiB JSON header and a total16MiB wire packet are admitted. The total
includes audio and masks; the maximum individual dimensions cannot all be used
at once. Duplicate fields, mismatched lengths, unsupported masks and corrupted
payloads fail closed. `unpack_visual_packet` returns detached metadata and
immutable PCM/RGB/mask bytes, not model results.

`visual_request_binding` must run independently in parent and worker. It checks
the outer InputFrame against the original audio subject, capture timestamp,
capture sequence, source and geometry. Runtime sequence/epoch assignment is
allowed without overwriting original capture evidence. Audio end is derived
from `timestamp_ms * 1000`; sample count is checked against the PCM byte count.
There is no independent packet audio timestamp to silently disagree. It compares
`visual_policy(packet_header)` with the frozen provider configuration using
canonical JSON equality, including clock declarations, source identities,
geometry, track/segment, mask producer and mask polarity. It derives the full
request hash, PCM hash and visual inventory hash locally; worker output cannot
supply these ownership proofs. Hash integrity does not authenticate a sensor.

`humansense.sam_audio_visual.align_visual_frames` maps each actual audio feature
time to the latest eligible original exposure. At48kHz each sample is exactly
125 sixth-microseconds. This avoids rounding fractional sample timestamps.
Eligibility requires the mapped exposure end **plus** declared clock error to
be at or before the feature. Worst-case age uses exposure end **minus** error.
A bounded pre-roll permits a frame to cover the first feature; missing, future
or stale coverage is rejected. Strictly increasing exposure ends and
nonregressing starts are required within a packet. A repeated selected index
means a past capture was held; it is not a newly captured observation. Across
overlapping windows the same original capture may be reused honestly. The
stateful consumer rejects reuse of one capture ID with different pixel/mask or
interval evidence. Its bounded 512-capture history commits only after successful
inference and output validation, and resets on relevant source/clock/track changes.
An unseen capture ID older than the retained sequence frontier is rejected.

The native consumer gathers exactly M=ceil(sample_count/hop) masked frames
before calling the pinned official processor. Its tensor load_video maps M
frames to M feature frames without changing indices. This is an explicit SDK
causal timestamp alignment policy, not equivalence to the upstream video-file
nearest-frame policy. Selection is causal only conditional on the declared
clock error bounds. SAM whole-window separation itself remains noncausal and
is not native incremental streaming.

Before gather or preprocessing, `visual_allocation_bounds` admits retained
three retained original RGB copies, two expanded uint8 copies, resized uint8, source float32 and three
resized float32 intermediates under128MiB. It also bounds expanded RGB to64MiB.
This accounts conservatively for known preprocessing tensors, not model
activations, encoder workspace, allocator overhead, GPU copies or measured peak
memory. The native consumer uses the actual encoder image
size and positive bounded encoder batch configuration (1..64); changing batch size does
not avoid the official transform of the whole sequence.

## Verification

Run packet tests with `python -m unittest discover -s tests -p test_sam_visual_packet.py`.
Alignment tests additionally compare100 generated cases to independent Fraction
arithmetic. To execute the actual pinned processor oracle, set
`GAMST_SAM_PROCESSOR_SOURCE` to your existing local processor.py at revision
`bb4c6999d2677c7402360e426afc01ddfad6dce0`, then run
`python -m unittest discover -s tests -p test_sam_audio_visual.py`.
The oracle verifies the exact file digest and executes the original function
bodies, checking all512 supported M-to-M counts and RGB/mask polarity. It does
not download source or weights. Without that path only this optional oracle is
skipped. Synthetic data demonstrates the contracts, not separation quality.

## Runtime ownership

Build the composite frame with `visual_frame`, then freeze
`visual_policy(unpack_visual_packet(frame.data)[0])` into the SAM Audio provider
configuration alongside the authenticated native artifact manifest. An empty
text prompt is accepted only with visual conditioning. The native implementation
uses the pinned processor's actual `mask_videos`, gathers the selected frames,
and supplies `masked_videos` to the official processor and separation model.

Before opening `ProcessProvider`, explicitly grant both sources:

```python
owner = human.enable_separation_media(
    subject_id=subject_id, source_id="microphone", sensor_id="mic0",
    visual_source={"source_id": "camera", "sensor_id": "cam0"},
    max_bytes=64 * 1024 * 1024,
)
```

The camera identity must match the frozen visual policy; an audio-only grant
cannot authorize visual packets. The owner reserves output media plus twice the
composite request size before dispatch. This is an ownership budget, not a peak
process-memory measurement. Cancellation and stream clearing reap the owned
worker and invalidate reservations and late results. Context and MCP expose
bounded alignment metadata, original capture intervals, hashes and the frozen
clock declaration (clock/alignment IDs, offset, error and maximum age).
`ages_sixth_us` is explicitly nominal age from the mapped exposure end; add
`6 * clock_error_us` for conservative worst-case age. They do not contain RGB tensors or inline separated waveforms.

## Recorded replay

`python -m humansense.sam_visual_replay --manifest recording.json --sha256 HASH
--consent --python /path/to/prepared/worker/python` replays immutable `.gsamv`
packets through the same owned native worker. Prepare the worker dependencies and
authenticated model assets separately; replay installs and downloads nothing.
The exact manifest has `schema_version: 1`, the provider `config`, and `packets`
entries containing only `file` (a unique local basename) and `sha256`. It admits
1..1024 packets and at most 128 MiB total input. All packets are hash/policy
checked before worker startup and rechecked when consumed. Original source epoch
changes reset the worker. Capture timestamps remain historical, not replay time.

Output is metadata-only JSONL. Transient media references are discarded before
each row is emitted, and `media_retained` and `trained_quality_verified` are false.
A failure or interruption may leave a valid output prefix; that prefix does not
prove the entire recording completed. Inference and cross-window history errors
can still occur after preflight. SAM whole-window inference is still noncausal;
replaying packets does not make it a stateful incremental neural model.

`test_sam_visual*.py` covers the actual worker/Context/MCP and replay transport
using an explicit fixture separation backend, plus cancellation and clearing
while inference is running. The optional original-processor oracle checks actual
masking/gather behavior. Neither proves trained separation quality.
