# Native speech-activity context

`SileroActivity` connects a local Silero JIT model to SensorStreams and the isolated `silero.activity` worker. It supplies speech-activity candidates to an LLM. It does not transcribe speech, identify a speaker, detect emotion, or decide whether the assistant should interrupt playback. Sensing remains independent of playback; AEC and turn-taking policy have separate acceptance gates.

Input is explicitly mono normalized 16 kHz PCM, at most one second per submission. Numeric sample vectors and little-endian PCM16 bytes are accepted. Channel selection, echo cancellation and resampling are upstream responsibilities. Source metadata is mandatory; capture timestamps denote block end, not callback arrival. Owner identity includes subject, source, sensor, stream epoch and channel names. Switching owners requires reset.

The provider retains the model's recurrent state and consumes every contiguous 512-sample (32 ms) frame. It keeps fewer than 512 leftover samples and returns up to 32 decisions per submission, preserving starts and stops within a larger input block. Each decision has its own start/end timestamps, score, active state and transition. End times account for pending samples rather than incorrectly assigning the entire capture-block timestamp to every decision. The output state also reports pending samples and thresholds. Scores are uncalibrated candidates; default start/stop hysteresis is 0.6/0.35. This implementation has no minimum-duration speech segmentation or hangover policy.

This is native stateful streaming with a 32 ms analysis step and no future lookahead. Collection, capture delivery, queueing and inference still add latency. No partial frame is padded or flushed into a fabricated end-of-stream decision. A discontinuity greater than 1 ms resets both pending samples and neural state; rounding tolerance does not accumulate. Sequence gaps reset through SensorStreams. Invalid model output clears recurrent state; worker failure invalidates the process instance. Explicitly reopen after worker failure. Close discards retained model references. Long-running accuracy depends on the checkpoint and domain; this change does not establish indefinite-call robustness.

```python
from humansense import ProcessProvider

config = dict(path='/models/silero_vad.jit', sha256=CHECKPOINT_SHA256)
spec = ProcessProvider.capability('silero.activity', config)

async def create_vad():
    return await ProcessProvider.open('/opt/audio-venv/bin/python', 'silero.activity', config)

streams.register(spec, create_vad)
await streams.open('microphone', 'mono', [spec.id])
```

Install `humansense-sdk[vad]` in the selected environment. Core discovery imports no Torch and starts no process. Checkpoints are trusted application inputs, bounded to 32 MiB and verified against the supplied SHA256 before loading. Hash verification detects mismatch; it is not a sandbox for untrusted TorchScript. No weights download implicitly. The existing legacy SileroVADPlugin API remains available; this provider implements the new capability/stream contract.

[Silero's official source](https://github.com/snakers4/silero-vad) and [pinned utilities](https://github.com/snakers4/silero-vad/blob/066409f3d9db45a30d21dfcbf9147826ffdd7052/src/silero_vad/utils_vad.py) describe fixed-size inference and state reset. The model is MIT licensed; SDK adapter code is Apache-2.0. No checkpoint or audio recording is bundled.

## Reproduction

The existing model source is pinned in `scripts/local_models.py` to Silero commit `066409f3d9db45a30d21dfcbf9147826ffdd7052`. Tested JIT SHA256:

`e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`

The actual replay uses the [Whisper test JFK recording](https://github.com/openai/whisper/blob/main/tests/jfk.flac), original SHA256 `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`. The source is 44.1 kHz stereo, eleven seconds. Preparation explicitly selects channel zero and resamples using SciPy; it does not downmix channels or silently resample inside the provider:

```python
import soundfile as sf
from scipy.signal import resample_poly
samples, rate = sf.read('/fixtures/jfk.flac', dtype='float32')
assert rate == 44100 and samples.shape[1] == 2
sf.write('/fixtures/jfk16.wav', resample_poly(samples[:, 0], 160, 441),
         16000, subtype='PCM_16')
```

```sh
PYTHONPATH=. python examples/validate_vad.py \
  --model /models/silero_vad.jit \
  --sha256 e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720 \
  --input /fixtures/jfk16.wav --worker-python /opt/audio-venv/bin/python \
  --output vad-evidence.json
```

The replay parent needs NumPy and SoundFile; preparation additionally needs SciPy. It adds one second of generated silence before and after speech, then sends 20 ms chunks at absolute sample-clock deadlines. It records all decisions, checks the expected count and contiguous decision boundaries, and reports freshness, submit latency, decision-end latency and separate parent/child peak RSS. This is prerecorded CPU model evidence, not live-microphone or Korean/English ASR validation. Synthetic model unit tests separately cover hysteresis transitions, leftover timing, gaps, owner changes, malformed PCM, bounds and lifecycle. Family #23 stays open for native ASR partial/final/revision and bilingual validation; neither VAD scores nor windowed Whisper text prove that requirement.

## Recorded execution on 2026-11-14

macOS 15.6.1 arm64 CPU, Python 3.10.11, Torch 2.8.0 in the worker; parent NumPy 2.2.6 / SoundFile 0.12.1, preparation SciPy 1.15.3. Prepared input SHA256: `648c55933f3a16466ed279263281bd5a4f2668ac9f527984bd416df121d920f8`.

The 13-second paced stream produced all 406 complete 32 ms decisions with contiguous boundaries and twelve start/stop transitions. No complete leading/trailing generated-silence frame was active; the final context was fresh. There is no manually aligned speech-boundary ground truth, so this is execution/stream continuity evidence, not a measured recall or onset-accuracy claim. The remaining eight milliseconds were left pending rather than padded.

Startup took 2242.87 ms. Submit-to-observation p50/p95/max was 8.71/29.17/111.62 ms; decision-end-to-observation was 23/65/120 ms, additional to the decision's 32 ms sample span. Parent peak RSS was 58.92 MiB; child peak RSS 195.06 MiB and child CPU time 2.12 seconds. Torch was not imported in the parent. These are one replay's measurements including cold behavior, not a guaranteed device-independent latency. Separate CPU process memory is material even though the model artifact is small.
