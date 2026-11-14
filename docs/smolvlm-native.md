# Local small semantic model engine

`SmolVLMNative` is the explicit local inference engine for
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, reference revision
`4cde9d208e19519c36917aa170fffc7e381aa0ad`. It accepts one to four chronological
BGR images and capture timestamps, and returns bounded **untrusted semantic
text**, token counts and truncation status. It does not call the user's LLM.

This engine is a component of required small-VLM context support (#40/#133).
It is not yet a SensorStreams provider or MCP integration. Capture-bound
window accumulation, lifecycle/reset and observation publication remain a
separate required integration step. Do not list engine-only support as full
streaming integration. Actual trained-model/device acceptance is #150.

## Explicit runtime and assets

Select an already provisioned dedicated environment with the `smolvlm` extra:
Torch2.8.0, TorchVision0.23.0, Transformers4.57.6, huggingface-hub0.36.0,
safetensors0.7.0, num2words0.5.14, NumPy and Pillow. No automatic installation
or model download occurs. CPU supports float32; CUDA supports float32 or
explicit bfloat16 on a compatible device. There is no automatic device choice.

Supply absolute paths and SHA256 for the complete `model.safetensors` and
`tokenizer.json`. The files must be regular, not symlinks, and at most2500MiB
and32MiB respectively. The checkpoint limit permits the official float32
architecture; it is not a runtime peak-memory promise. Hashes and file identity
are verified; weight tensors are copied into owned storage and file identity
is checked again after loading. Keep assets immutable while opening.

The production architecture is fixed in code. The loader does not execute
local config files, downloaded code or pickle, and does not use Hub APIs or
`from_pretrained`. It requires exact state keys/shapes and finite floating
weights, plus tokenizer vocabulary49280 and the expected special-token IDs.
A matching hash is caller-selected provenance, not proof of official training.
Weights and source model use Apache2.0 according to the official model card;
verify any substituted/fine-tuned checkpoint's terms separately.

```python
from humansense.providers.smolvlm_native import SmolVLMNative

model = SmolVLMNative.from_local(
    checkpoint_path='/absolute/model.safetensors',
    checkpoint_sha256='REPLACE_WITH_ACTUAL_SHA256',
    tokenizer_path='/absolute/tokenizer.json',
    tokenizer_sha256='REPLACE_WITH_ACTUAL_SHA256',
    device='cpu', precision='float32', max_new_tokens=64,
)
try:
    result = model.predict([bgr_frame_0, bgr_frame_1], [capture_ms_0, capture_ms_1])
finally:
    model.close()
```

Calls are synchronous and serialized; this native class cannot preempt a GPU
kernel or blocked native generation. Use an owned worker for hard cancellation
of the application task; that provider integration is not included here.
Calling close waits for the current call. No implicit background work exists.

## Bounds and semantics

Each input is a plain uint8 BGR H×W×3 array with32..2160px axes and at most
1920×1080pixels. The1..4 capture timestamps are strictly increasing integer
milliseconds within10seconds. The caller supplies timestamps; this engine
does not verify camera timing. Input images are copied, BGR→RGB converted, then
the pinned image processor resizes to longest-edge512 followed by square512
with Lanczos, rescale1/255 and mean/std0.5. Image splitting is disabled.
This intentionally differs from the official high-resolution split-image
pipeline and video processor: quality/parity must be evaluated, not assumed.

A fixed prompt interleaves64image tokens per image with relative capture times.
The prepared input is bounded to1024text/image tokens and four512×512images;
generation uses greedy decoding, one beam and at most128new tokens. Returned
text is at most8192UTF-8bytes. Special tokens are removed; the prompt prefix is
never returned. Token counts are measurements, not a cost estimate.

KV cache is used only within one generate call. No KV state is retained between
windows. Repeated calls and chronological multi-image prompting are **not
native stateful streaming**. This is the image/multi-image path of a
video-capable model, not official video-preprocessing parity. Model-generated
text remains untrusted data even if it resembles an instruction. There are no
calibrated certainty scores and no claim of measured physical quantities.

## Validation and remaining gates

`python -m unittest discover -s tests -p test_smolvlm_native.py -v` checks
configuration, bounded local assets and optional-runtime isolation. In the
pinned CI environment it additionally constructs the full500M architecture on
meta (no weights), runs an actually executed **tiny untrained SmolVLM** with a
generated tokenizer, tests the shared strict safetensors loader, RGB processing,
image-token counts, greedy reference parity, repeated-call cache isolation,
input bounds and malformed weights. This is algorithm/processor evidence,
not real500M-checkpoint loading, trained semantic quality or device throughput.
No trained model weights are installed by those tests.

Actual500Mcheckpoint/tokenizer, trained semantics, sustained device behavior and
quality/cost comparison remain required in [#150](https://github.com/NomaDamas/Gamst/issues/150),
[#40](https://github.com/NomaDamas/Gamst/issues/40) and
[#57](https://github.com/NomaDamas/Gamst/issues/57). Small parameter count alone
is not proof of cheaper or faster inference at matched quality.

Sources: [official model](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
[pinned configuration](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/blob/4cde9d208e19519c36917aa170fffc7e381aa0ad/config.json),
[Transformers API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/smolvlm).
