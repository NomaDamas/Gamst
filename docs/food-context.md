# Food segmentation context

`food.segformer_b0` runs an explicitly selected local SegFormer-B0 semantic
segmentation checkpoint. It emits food labels, native-grid area fractions,
normalized boxes and bounded 16×16 mask coverage to the observation store/MCP.
This is frame inference: repeated frames can be submitted to `SensorStreams`,
with independent consumer cadence. It is not a recurrent streaming model, an
instance tracker, a nutrition estimator or an automatic VLM/LLM caller.

## Runtime and assets

Use an existing dedicated Python with the `food` optional dependencies:
Transformers 4.57.6, PyTorch 2.8.0, huggingface-hub 0.36.0, safetensors 0.7.0,
NumPy and Pillow. Discovery imports none of the ML runtime. Opening the provider
loads only the explicit absolute local safetensors path; no Hub lookup,
`from_pretrained`, remote code, pickle or automatic installation is used.
Do not combine this environment with the Transformers 5.23 diarization runtime.

The checkpoint must contain the exact **complete SegFormer-B0 segmentation
state**, including the task-trained classifier head and normalization buffers.
The loader verifies SHA256, a 64 MiB limit, exact keys/shapes/dtypes and finite
values. A backbone-only checkpoint is rejected. Configuration supplies the
exact head label order, explicit background/optional unknown class, training
provenance, target cuisine and weight license. These are user declarations,
not evidence that a checkpoint was trained or licensed correctly.

```python
import sys
from humansense import ProcessProvider

config = dict(
    checkpoint_path='/absolute/path/food.safetensors',
    checkpoint_sha256='REPLACE_WITH_ACTUAL_64_CHARACTER_SHA256',
    labels=['background', 'rice', 'tofu'],  # exact task-trained head order
    background_id=0, unknown_id=None,
    training_dataset='your-dataset-v1', target_cuisine='your-target-cuisine',
    model_source='your-checkpoint-provenance', checkpoint_license='your-weight-license',
    input_size=512, threshold=0.5, max_labels=16, device='cpu',
)
# Inside your async application, select the already provisioned interpreter.
worker = await ProcessProvider.open(sys.executable, 'food.segformer_b0', config)
# Register worker.spec with SensorStreams and submit video InputFrame objects.
# Always close the owning streams/worker in finally; no automatic model download.
```

Input is a plain uint8 BGR H×W×3 array, each axis 32..2160 and at most
1920×1080 pixels. A full 1080p BGR frame uses 6,220,800 bytes, above the
default SensorStreams 4 MiB packet budget. Select
`SensorStreams(human, max_packet_bytes=8*1024*1024)` for that input size,
or use a smaller capture resolution. This increases the stream admission
budget, not the provider's pixel limit. Preprocessing owns the input, converts BGR→RGB, uses Pillow
bilinear square resize to the explicitly configured 256 or 512, rescales by
255 and applies ImageNet RGB mean/std. This must match the selected training
pipeline. Output is the **native 64×64 or 128×128 logit grid**, not official
original-resolution bilinear-upsampled prediction parity. A class may combine
multiple disconnected foods; boxes are semantic extents, not instances.

## Output interpretation

At most 16 food labels survive a top-softmax threshold and area ranking.
Each has a 16×16 row-major coverage grid. Area means fraction of the image
on the native grid, not volume or mass. Accepted food, background, declared
unknown class, below-threshold and omitted food fractions partition that grid.
`truncated` reports omitted food classes. An empty result replaces prior food
labels; it does not prove absence of food. Softmax scores are uncalibrated.
An explicit unknown class is not proof of out-of-distribution detection.

The result includes checkpoint/label-map/captured-pixel hashes and declared
provenance. The observation envelope retains capture time and source metadata;
polling does not refresh capture time. Raw images and full-resolution masks
are not inserted into context. The caller must retain images separately if
later visual verification is required. Freshness is 1000 ms; slow inference
can produce a stale observation rather than a fabricated new timestamp.

`mass_grams` and `calories_kcal` remain null. Temperature and nutrient
availability remain false. Model quality, training verification, score
calibration and unknown detection remain explicitly unvalidated.

## Evidence and acceptance

Local tests cover deterministic logits, exact partitions, preprocessing,
malformed outputs, context size and abstention. The dedicated CI runtime also
executes a **generated untrained** complete B0 graph, checks exact reference
logits, rejects a missing learned tensor, and passes inference through an owned
worker, SensorStreams and MCP. Those checks establish software execution only.
No trained model weights are installed by these tests.

Target-cuisine trained weights, held-out known/unknown food masks, calibration,
license verification and measured device latency/cost are tracked in
[#148](https://github.com/NomaDamas/Gamst/issues/148), under food capability
[#36](https://github.com/NomaDamas/Gamst/issues/36). This adapter does not claim
food SOTA, production accuracy, real-time throughput or the overall VLM savings
gate. Unsupported preprocessing/head architectures need a separate adapter.

Primary references: [SegFormer](https://github.com/NVlabs/SegFormer),
[Transformers SegFormer](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/segformer),
[FoodSAM](https://github.com/jamesjg/FoodSAM),
[FoodSeg103](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1).
FoodSAM is a heavier research alternative; dataset names are not runnable models.
