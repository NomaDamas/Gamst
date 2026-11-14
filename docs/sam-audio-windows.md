# SAM-Audio complete-window contracts

This is the reusable input/output validation slice of [#201](https://github.com/NomaDamas/Gamst/issues/201), tracked in [#223](https://github.com/NomaDamas/Gamst/issues/223). It does not load a model or register an inference provider. Actual native integration, capture provenance, text-tokenizer truncation, media ownership and trained inference acceptance remain separate requirements.

```python
from humansense.sam_audio_windows import prepare_window, validate_outputs

window = prepare_window(
    pcm, sample_rate=48000, prompt="the ringing telephone",
    hop_length=verified_config_hop,
    anchors=[("+", 0.0, 0.2)],
)
# Native adapter passes window.array().reshape(1, -1) to the processor.
# After native output conversion to CPU NumPy, preserving its one-item lists:
target, residual = validate_outputs(
    target_list, residual_list,
    sample_count=window.sample_count, hop_length=window.hop_length,
)
```

Input must be a plain one-dimensional float32 array, mono 48kHz, with 48,000–192,000 finite samples in [-1,1]. The one-to-four-second limit is SDK policy, not a model capability claim. No resampling, mixing or missing-sample padding occurs. `PreparedWindow.pcm` owns immutable little-endian bytes; `array()` returns a writable private copy. Its PCM digest binds those original bytes. `prompt_sha256` binds the prompt, manual anchors, sample rate, explicit hop and source revision; it is not a checkpoint or capture-evidence identity.

Prompts are nonempty and at most4,096 UTF8 bytes. Up to32 manual `+`/`-` anchors use window-relative seconds. Both endpoints map using `ceil(seconds*48000/hop)`, matching the [pinned processor](https://github.com/facebookresearch/sam-audio/blob/bb4c6999d2677c7402360e426afc01ddfad6dce0/sam_audio/processor.py). Empty quantized intervals and overlaps are rejected. Adjacent intervals are allowed; caller ordering is retained. No auto-span or visual mode is implied. Hop must come from the verified effective codec configuration, within the SDK bound1..48,000;1920 is only a test/example value.

An endpoint whose floating-point quantization exceeds the actual codec feature count is rejected rather than silently clipped. For example,48,100 samples with hop100 and an anchor ending at48,100/48,000 seconds produce this edge case in the upstream formula.

The [pinned model](https://github.com/facebookresearch/sam-audio/blob/bb4c6999d2677c7402360e426afc01ddfad6dce0/sam_audio/model/model.py) returns target and residual as one-waveform lists for batch1, with codec-aligned lengths. The adapter must convert each tensor to NumPy without changing batch structure. Validation requires the exact `ceil(original_samples/hop)*hop` length and finite float32 values within the defensive SDK output range[-16,16], including padding. It trims only the final codec tail and returns detached little-endian arrays. It never clips/normalizes estimates or asserts that they add to the ambient signal.

Tests exercise analytic timing boundaries, malformed/oversized input, byte limits, mutation isolation, exact padding and output rejection. They do not execute the upstream processor or trained model. Official runtime parity, model quality, latency and memory remain unverified. Complete windows introduce collection delay and do not establish native causal streaming or seamless cross-window audio.
