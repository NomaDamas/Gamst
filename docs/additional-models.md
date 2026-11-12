# Additional model adapters

Local adapters load explicit files only. Installation, weights and model-license review are separate. No SAM3 compatibility or latest-model benchmark is inferred from having a generic adapter.

## YOLO person detector

```bash
pip install -e '.[yolo]'
python -m humansense dashboard --yolo-model /path/to/trusted-yolo-weights.pt
```

`UltralyticsYOLOPlugin` uses an installed Ultralytics runtime with a local model, filters the configured person class, and emits count, normalized boxes and stream-local tracks. It supports CPU execution and explicitly selected devices. No weight-name auto-download is used. Use trusted weights: PyTorch model deserialization is not a sandbox. Review Ultralytics and model-weight licenses (including applicable AGPL/commercial conditions) for your distribution; upstream licenses were not fetched in this restricted environment.

MCP: call `start_capture` with `yolo_model_path`. Select one person detector per configuration. Native Ultralytics inference remains unverified here because dependencies/weights are unavailable.

## Contract-specific ONNX detection

```bash
pip install -e '.[onnx]'
python -m humansense dashboard --onnx-model /path/to/exported-person.onnx
```

`ONNXPersonPlugin` requires a model with `humansense.contract=person_xyxy_score_class_v1` in ONNX metadata. Preprocessing is stretched BGR→RGB, float32 NCHW `[1,3,H,W]`, divided by 255. Output is postprocessed/post-NMS `[N,6]` or `[1,N,6]`: x1,y1,x2,y2 in resized input pixels, score, class ID. Raw YOLO/SAM tensors do not satisfy this contract. Set model metadata only after validating its real preprocessing and output semantics. The adapter does not make arbitrary exports compatible by renaming them.

`ONNXFeaturePlugin` accepts a caller-provided preprocessor and declared output label mapping for vector model scores. It requires `humansense.contract=feature_model_scores_v1` and finite scores in [0,1]. No facial or acoustic pattern is invented or treated as a calibrated inner-state probability.

## Image-model / VLM endpoint

```bash
export HUMANSENSE_VISION_URL=http://127.0.0.1:11434/v1/chat/completions
export HUMANSENSE_VISION_MODEL=YOUR_INSTALLED_IMAGE_MODEL
python -m humansense dashboard
```

The endpoint must accept JPEG image messages and structured JSON completions. `VisionDescriptionPlugin` rate-limits per stream (default 1000ms), reduces image dimensions and validates a narrow observations/expressions schema. It emits `scene.model_description` marked **unverified model interpretation**, with insufficient quality and null probability. The image is transiently encoded for inference and never persisted by the adapter.

For an authorized remote image provider, explicitly set `HUMANSENSE_VISION_ALLOW_REMOTE=true` and optionally `HUMANSENSE_VISION_KEY_ENV=YOUR_KEY_VARIABLE`. This **transmits images**, unlike text-only reasoning, and requires participant consent for that transfer. Credentials remain in environment variables. Rate limiting limits request frequency, not worst-case inference latency: a slow VLM still delays that video processing call. Use a separate low-frequency video observer if a high-frequency detector must continue independently.

## Audio finalization

The existing faster-whisper adapter now supports residual `flush`; HumanSense automatically finalizes after a VAD stop decision, or applications can call `await human.flush_audio()` at an externally verified utterance boundary. See [latency scope](latency.md#speech-finalization-path).

## Verification boundary

Unit tests inject native runtimes or HTTP responses to exercise preprocessing contracts, output parsing, subject isolation, cancellation and provenance. They do not prove the corresponding actual model ran. Use the model replay and hardware measurement commands in [latency](latency.md) on a host with installed backends and licensed assets. SAM3, diarization and biometric association are still not implemented or validated.
