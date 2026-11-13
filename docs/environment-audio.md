# Ambient sound context with YAMNet

The adapter classifies an explicitly selected ambient audio channel into bounded event candidates. It preserves ambient input independently of speech cleanup and ASR. It does not transcribe speech, infer emotion, identify speakers, or guarantee alarm detection.

`YAMNetEvents` uses 975 ms trailing windows of normalized 16 kHz PCM, with a configurable hop (20–975 ms, default 480 ms). This is windowed streaming, not recurrent neural inference. The first output needs 15,600 samples; there is no zero padding. Each submitted block is at most one second and yields at most one latest window. Larger hops in capture delivery therefore skip intermediate windows. Twenty-millisecond capture blocks preserve the normal cadence. Collection latency is additional to inference and queue latency.

```python
from humansense import ProcessProvider

config = dict(path='/models/yamnet.tflite', hop_ms=480, top_k=8,
              score_threshold=0.05)
spec = ProcessProvider.capability('yamnet.events', config)

async def create_environment_classifier():
    return await ProcessProvider.open(
        '/opt/audio-venv/bin/python', 'yamnet.events', config)

streams.register(spec, create_environment_classifier)
await streams.open('microphone', 'ambient', [spec.id])
```

Use the standard SensorStreams InputFrame/SensorMetadata submission contract, with capture-end timestamps and monotonically increasing sequences. NumPy multichannel arrays require an explicit `channel` index; channel names, when supplied, must match the array. Byte input is mono little-endian PCM16. Mono numeric values must be finite and normalized to [-1, 1]. Resampling is an explicit upstream responsibility. A sample-clock discontinuity greater than 1 ms clears the accumulated window; tolerance does not accumulate. Runtime sequence gaps reset the lane and rotate its epoch. Switching source, sensor, stream epoch or channel layout requires reset. Closing releases the classifier and prevents further inference.

The capability is `audio.yamnet.events`, producing `audio.environment_events`. Output includes top-k labels, indices, scores, absolute window boundaries, sample count, chosen channel, threshold and counts before truncation. The full 521-class model result is validated before selection. Scores are nonexclusive, hierarchical and uncalibrated: Animal and Dog can both score highly. A score is not a probability that a real event occurred. LLM consumers must preserve candidate wording and timestamp/freshness metadata. Silence may be a candidate; an empty thresholded result is valid. No raw audio is placed into context.

## Artifact and dependencies

Install `humansense-sdk[environment-audio]` in the selected runtime, which pins MediaPipe 0.10.35. Core discovery does not import NumPy or MediaPipe. Local loading requires the official MediaPipe YAMNet v1 artifact and verifies SHA256:

`4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf`

There is no implicit download. [Official artifact](https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite), [MediaPipe audio classifier documentation](https://developers.google.com/edge/mediapipe/solutions/audio/audio_classifier), [YAMNet model card](https://www.kaggle.com/models/google/yamnet/tensorFlow2/yamnet/1?tfhub-redirect=true) and [TensorFlow Lite specialization](https://github.com/tensorflow/tfhub.dev/blob/master/assets/docs/google/models/yamnet/classification/tflite/lite/1.md) document the model and artifact. YAMNet and MediaPipe use Apache-2.0 licensing. Float32 input does not imply every internal weight/operator is float32. Model files remain application-managed local assets.

## Actual inference evidence

On 2026-11-14, macOS 15.6.1 arm64 CPU, the pinned model ran in a separate Python worker against four labeled ESC-50 recordings, each five seconds. PCM16 44.1 kHz recordings were explicitly resampled using SciPy 1.15.3 to 16 kHz and replayed in 20 ms blocks at absolute sample-clock deadlines. This is paced prerecorded input, not live microphone validation. Parent NumPy was 2.2.6; the worker used MediaPipe 0.10.35. Thirty-six actual inference windows were stored as context; the final observation was fresh.

Worker startup: 1389.81 ms. Submit-to-stored-observation p50/p95/max: 11.33/72.29/77.52 ms. Capture-end-to-observation p50/p95/max: 13/82/83 ms. These measurements exclude the required 975 ms collection span. Parent peak RSS: 104.59 MiB; child peak RSS: 128.66 MiB; child CPU time: 1.41 s. Resource values cover this replay and model loading, not a device-independent requirement. An accelerated replay had p50/p95 10.33/22.28 ms; the paced run demonstrates why a single fast timing must not be advertised as a universal realtime guarantee.

Peak candidates across windows included Dog 0.9414 / Bark 0.8008 for dog; Rain 0.9180 for rain; Knock 0.7383 for wooden-door knocks. For clock_alarm, Telephone bell ringing 0.8516 exceeded Alarm clock 0.7383: this is a concrete confusion, not a perfect classification claim. Four selected recordings do not establish benchmark accuracy, domain robustness or a safety alarm capability.

Fixtures and SHA256:

- `1-100032-A-0.wav`: `f40a849a2375c8c63312a73dd2dd6c74007301fcc21b4be2ece29a642831e3d8`
- `1-17367-A-10.wav`: `d732acd2e0c7c40405968010f49f250740c86998fc416eff2ed5330fc8c89dea`
- `1-13613-A-37.wav`: `2e28b25e14611723cdcc0becfa36468a8d0329a0c141e4f1d894bd6bb5773f34`
- `1-101336-A-30.wav`: `e738266e395e378102f74e0920b05ff5937543788420d5b2a6b8d3df8bdb3910`

[ESC-50's official repository](https://github.com/karoldvl/ESC-50) supplies recordings and metadata. ESC-50 is CC BY-NC 3.0; its ESC-10 subset is CC BY. The dog and rain fixtures are ESC-10 entries; alarm and knock are not. These are local research validation inputs, not redistributed SDK assets or commercial demo media.

```sh
PYTHONPATH=. python examples/validate_yamnet.py \
  --model /models/yamnet.tflite \
  --input /fixtures/1-100032-A-0.wav /fixtures/1-17367-A-10.wav \
          /fixtures/1-13613-A-37.wav /fixtures/1-101336-A-30.wav \
  --labels-csv /fixtures/esc50.csv \
  --worker-python /opt/audio-venv/bin/python --pace --output evidence.json
```

The replay parent additionally needs NumPy and SciPy; these are not new core dependencies. The report includes input hashes, model provenance, all event windows, resampling clipping counts, latency and process resource measurements. Unit tests use a clearly synthetic 521-class backend to verify buffering, gaps, channel selection, validation, lifecycle and preservation alongside an independent destructive speech-cleanup test lane. Those tests do not substitute for the actual model evidence above. Wider environmental sound coverage, live-device evaluation and broader audio package acceptance remain tracked in #24 and #6. This adapter does not establish the VLM cost/quality gate.
