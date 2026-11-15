# Optional Video Depth Anything Small streaming depth

**Status: implemented adapter; real-weight acceptance pending.** `depth_anything_v2.small` remains the default frame-local provider. The explicit `video_depth_anything.small` profile uses the native temporal cache of Video Depth Anything Small rather than repeating DAV2 inference.

The target is Chen et al., [*Video Depth Anything: Consistent Depth Estimation for Super-Long Videos*](https://arxiv.org/abs/2501.12375), CVPR 2025 Highlight, at [upstream revision 4f5ae231](https://github.com/DepthAnything/Video-Depth-Anything/tree/4f5ae23172ba60fd7bc11ef671cca678842c7072). Only the Small relative-depth checkpoint is supported. Small code/model terms declare Apache-2.0; Base/Large are CC-BY-NC-4.0 and are not interchangeable configuration options. Upstream describes this training-free streaming mode as experimental and reports lower ScanNet accuracy than offline inference; no offline-parity or realtime claim is made here.

## Explicit local worker

Provision the pinned source checkout and Small `video_depth_anything_vits.pth` independently, accept and verify the actual artifact terms, and record its SHA256. Discovery and import never download or load weights. The worker requires the existing absolute source/checkpoint paths and a fresh explicitly selected interpreter. CPU float32 is the default; CUDA must be requested and available. There is no device or frame-local fallback.

```python
from gamst import ProcessProvider

config = {
    'source_dir': '/absolute/Video-Depth-Anything',
    'path': '/absolute/video_depth_anything_vits.pth',
    'sha256': '<64 lowercase hexadecimal characters>',
    'device': 'cpu',             # or 'cuda'
    'grid_size': 3,
    'max_gap_ms': 1000,
    'max_frames': 1000000,
}
spec = ProcessProvider.capability('video_depth_anything.small', config)
worker = await ProcessProvider.open('/absolute/vda-runtime/bin/python',
                                    'video_depth_anything.small', config)
try:
    observation = await worker.process(frame)
finally:
    await worker.close()
```

The `video-depth` extra follows the pinned upstream requirements for Torch 2.1.1, TorchVision 0.16.1, NumPy 1.24.0 and einops 0.4.1; upstream leaves OpenCV unpinned. Use a dedicated environment, not the DAV2 Transformers environment. Checkpoint bytes are limited to 256 MiB, SHA-verified before native imports, decoded with `weights_only=True`, checked tensor-by-tensor and loaded strictly into the Small architecture. Hub/offline flags are set before importing upstream code.

## State and output boundaries

Input is ordered uint8 BGR, at least 32 pixels per axis, at most 1920×1080 pixels and with aspect ratio at most 1.78. The adapter supplies an owned RGB copy. Source/epoch/coordinate/calibration/geometry changes, capture-sequence or timestamp gaps, and `max_frames` reset the native transform and 32-entry temporal-cache bookkeeping. Stale timestamps are rejected. Inference is synchronous with no internal queue; `ProcessProvider` and `SensorStreams` supply deadlines and bounded backpressure.

Output is relative inverse depth: larger is relatively nearer, not metres. As with DAV2, the adapter clips native values at per-frame p05/p95 and publishes grid-cell medians. Consequently `temporal_tracking=true` describes the native model state, while `temporal_comparability=false` explicitly prevents interpreting normalized cell changes as approach speed or metric distance. Flat maps abstain with null cells. Summary records capture timestamp/sequence, reset count, processed frames, cache bounds, source revision and artifact-verification status. No dense depth map enters context.

```sh
PYTHONPATH=tests:. .venv/bin/python -m unittest test_video_depth -v
```

Tests inject a fake streaming predictor and cover RGB ownership, grid order, timestamps, discontinuity/session resets, cache metadata, flat-map abstention, malformed output, artifact hashes, close/reset and dependency-light discovery. They do not load the model. Real-weight loading, pinned-source numerical parity, temporal ordinal quality, cache-reset parity, long-run memory and sustained CPU/CUDA latency remain mandatory acceptance work.
