# Optional SAM3.1 bounded video masklets

**Status: implemented adapter; real-weight acceptance pending.** Sampled-image SAM3 and EfficientTAM tiny512 remain the defaults. The explicit `sam3.1.video` profile targets Meta's native SAM3.1 Object Multiplex predictor and is not enabled by selecting a different SAM3 image checkpoint.

The source target is [SAM3.1 revision f664253](https://github.com/facebookresearch/sam3/tree/f664253c90122f5d968a17c3f75a62893b4e15e7), released 2026-03-27. Code and checkpoints use the [custom SAM License](https://github.com/facebookresearch/sam3/blob/f664253c90122f5d968a17c3f75a62893b4e15e7/LICENSE), not Apache-2.0. Weights are the separately gated [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) asset; access, authentication and model terms must be handled outside gamst. No token or download path exists in this provider.

## Native bounded chunks

The official video API accepts a list of images and constructs a stateful clip session. This adapter bounds that session to 1..16 frames and 1..4 objects, applies one configured text prompt at frame zero, propagates the complete chunk synchronously, then closes/evicts the native session. It buffers frames before a full chunk; `finish()` processes a final partial chunk. This is bounded video inference with synchronous backpressure, **not causal per-frame tracking** and not an unbounded live mask cache. Native confirmation can use later frames in the same chunk, so `causal=false` and `lookahead_ms` are explicit.

```python
from gamst import ProcessProvider

config = {
    'source_dir': '/absolute/sam3',
    'path': '/absolute/sam3.1.pt',
    'sha256': '<64 lowercase hexadecimal characters>',
    'prompt': 'person',
    'device': 'cuda',
    'max_frames': 8,
    'max_gap_ms': 1000,
    'max_objects': 4,
    'max_track_gap': 2,
}
spec = ProcessProvider.capability('sam3.1.video', config)
worker = await ProcessProvider.open('/absolute/sam31-runtime/bin/python',
                                    'sam3.1.video', config)
try:
    observation = await worker.process(frame)
    # At managed end-of-input, finish emits a buffered partial chunk, if any.
finally:
    await worker.close()
```

The official pinned installation requires Python 3.12+, CUDA and Torch 2.10.0. The `sam31-video` extra follows the official pyproject/install instructions, including ftfy 6.1.1 and the declared NumPy/timm/iopath ranges; upstream does not exact-pin every transitive dependency. Use an isolated runtime with the pinned source checkout. The loader verifies an existing checkpoint up to 8 GiB by streaming SHA256, sets offline flags, disables FlashAttention-3/compilation/warm-up, and supplies the local checkpoint explicitly. CPU/MPS substitution is rejected rather than silently attempted.

Source/epoch/geometry changes and timestamp/sequence gaps discard an incomplete buffer and start a new chunk. Each result contains compact timestamped masklets: native object ID scoped by chunk, exact capture sequence/timestamp, normalized bbox, original-size foreground area and a lossy 16×16 coverage grid. Full-resolution masks never enter context. Tracks absent beyond `max_track_gap` are marked expired; all remaining native IDs are session-evicted at chunk close. IDs are chunk-local, not verified identity or cross-chunk re-identification. Empty masks create no observation and cannot refresh presence. Output/object/frame limits keep the observation within the bounded JSON contract.

```sh
PYTHONPATH=tests:. .venv/bin/python -m unittest test_sam31_video -v
```

Tests use a fake native backend to exercise buffering, partial finish, timestamped occupancy, track expiry and chunk-local IDs, discontinuity drops, malformed masks, synchronous backpressure metadata, output bounds and terminal close. No gated weights or CUDA runtime are needed. Real checkpoint/session execution, native output parity, prompt/occlusion quality, actual GPU memory plateau, chunk latency and capture-to-context freshness remain mandatory acceptance work. The published H100 Object Multiplex speedup is not a local latency measurement.
