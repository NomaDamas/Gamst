# Local detection, continuity and object memory

`RFDETRObjects` runs RF-DETR Nano 1.12.1 on an explicitly supplied local checkpoint. Model construction verifies the required SHA256 and creates an inference-only `model.inference(compile=False)` snapshot. This uses upstream eager inference optimizations without compilation caches or CUDA graph capture. Discovery does not import RF-DETR, download weights or open a camera. The version attached to observations includes package version, eager runtime and checkpoint digest.

Install `humansense-sdk[detection]` in a dedicated runtime. RF-DETR 1.12.1 requires Transformers 5, whereas the existing `huggingface` extra requires Transformers 4; these extras must not be resolved into the same environment. The standalone SDK/core and remote context consumer need neither ML environment. Cross-process deployment remains part of #13; this adapter does not pretend an in-process dependency conflict is solved.

```python
from humansense import SensorStreams
from humansense.providers.detection import RFDETRObjects

provider = RFDETRObjects.from_local(
    '/models/rf-detr-nano.pth', sha256=CHECKPOINT_SHA256, device='cpu')
streams = SensorStreams(human)
spec = provider.capability(provider.version)
streams.register(spec, lambda: provider)  # one provider instance per stream
await streams.open('camera', 'device0', [spec.id])
```

Submit uint8 BGR images with a sensor coordinate frame explicitly naming the camera's normalized image coordinates, e.g. `camera.normalized`. Pixel input is limited to 4096 per side and 16 MiB. The adapter converts BGR to RGB and prevents the native result from retaining a source image. Normalized boxes describe the current image, never metric position. Timestamps must strictly increase within a stream; reset before switching source, sensor or coordinate frame. SensorStreams handles dropped-sequence state resets. Resolution changes break associations.

The neural model runs **frame inference**, without temporal model state. A separate class-constrained IoU tracker retains at most `max_objects` associations (default 32, maximum 64), expires missing tracks after `max_age_ms` (default 1000), and issues new non-recycled IDs after reset. Ambiguous competing matches retire old associations and produce `uncertain_new` IDs. This is conservative geometric continuity, not re-identification, camera-motion compensation or identity verification. Fast movement and complete crossings without intermediate ambiguity can still create incorrect geometric associations; labeled natural-video tracking quality remains a required validation in #17.

Output contains `objects` and `summary`. Each object has a track ID, label, model score, association status and normalized-image location. `summary.truncated` means the output cap dropped detections. Pass the reported snapshot completeness into memory rather than treating a capped output as evidence that other objects disappeared:

```python
record = human.store.latest_signals('local-user', 0, now_ms)[0]
value = record['observation']['value']
await human.remember_objects(
    record['id'], complete_snapshot=value['summary']['complete_snapshot'])
remembered = await human.find_objects(text='cup')
```

An empty complete observation marks prior entities unseen. Memory preserves last-seen evidence without inventing their current location. Reappearance after track expiry or an ambiguous crossing creates a distinct entity; semantic instance retrieval remains separate work.

## Validation

Contract/continuity tests use a deterministic backend and are not model-inference evidence:

```sh
python -m unittest discover -s tests -p test_detection_provider.py -v
```

For actual local checkpoint execution, the replay command runs a real image, the same image again, a synthetic 5-pixel translation and a blank image through SensorStreams, context storage and memory. It records checkpoint/input hashes, dependencies, device, durations and bounded observations. The fixture is deliberately identified as a transformed image replay, not natural-video validation:

```sh
PYTHONPATH=. python examples/validate_rfdetr.py \
  --checkpoint /models/rf-detr-nano.pth --sha256 CHECKPOINT_SHA256 \
  --image /fixtures/dog.jpg --device cpu --output evidence.json
```

RF-DETR Nano and the package are Apache-2.0 according to the [upstream repository](https://github.com/roboflow/rf-detr). Other RF-DETR variants have different licenses; this adapter does not enable them. The exact installed 1.12.1 `predict` source supplies RGB, `include_source_image=False`, pixel `xyxy` and per-detection `class_name` contracts. Model scores are uncalibrated detector outputs. This PR advances #17/#5; natural-video crossings/occlusions, GPU latency and end-to-end VLM cost/quality gates are not established by unit tests or this replay.

### Actual execution evidence

The 2026-11-14 local replay used macOS 15.6.1 arm64, RF-DETR 1.12.1, Torch 2.8.0, Torchvision 0.23.0, Transformers 5.20.2, Supervision 0.31.7 and NumPy 2.2.6. The [upstream Nano checkpoint](https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth) has SHA256 `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d`. The external [Darknet example photograph](https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg), downloaded only for local validation and not redistributed here, has SHA256 `5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866`.

MPS eager replay produced bicycle, dog and car candidates and emitted an empty snapshot for the blank frame. Objects remained retrievable as unseen memories. An initial replay with fixed capture timestamps loaded in 20.51 s and recorded submit-to-stored-observation times of 857.23, 80.60, 84.42 and 40.58 ms. A subsequent replay timestamped at each actual submission loaded in 32.97 s and recorded 8747.19, 90.78, 184.34 and 228.42 ms. That long first inference expires the first three track IDs; the next pair of images shares newly allocated IDs. This demonstrates why cold-start latency and capture timestamps cannot be excluded from continuity evidence. The example explicitly counts both fresh and stale context entries; an entry's presence alone is not freshness.

These four-frame smoke tests are not a p95 benchmark, representative accuracy evaluation or guaranteed streaming rate. The prior unspecialized MPS path took 25.29 s on its first frame; separate process runs with different warm-up/load state do not establish a controlled speedup. Eager CPU inference also ran successfully (930.67, 779.32, 603.90 and 401.26 ms in one run). Native dependencies emitted duplicate AVFoundation class warnings from OpenCV/PyAV on this host; no package or library was removed to hide them.
