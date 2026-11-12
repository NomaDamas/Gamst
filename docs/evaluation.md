# Evaluation and release gates

The supplied tests establish deterministic software behavior, not psychological validity or trained-model accuracy.

## Reproducible software checks

Run `python -m unittest discover -s tests -v`. Test fixtures cover missing modalities, explicit clarification, subject isolation, stale timestamps, numeric baselines, structured conflicts, consent/pause, retention/deletion, model adapter contracts and MCP stdio. Local camera, microphone, MediaPipe, ASR and WebSocket dependency/hardware checks must be run separately on a supported machine.

## Model and product evaluation still required

1. With participant consent, annotate observable events and self-reported interaction needs. Do not label hidden emotions or deceptive intent as sensor truth.
2. Split sessions by participant; report false alarms, misses, abstentions, time alignment and per-participant differences.
3. For predictive probabilities, define an outcome such as an explicit clarification request within the next turn. Fit calibration on a separate validation split; publish Brier score/reliability curves and held-out results. Until then return null probabilities.
4. Compare text-only, audio-only, video-only and fused input using the same sessions. Evaluate incremental value of video rather than assuming it.
5. Compare agent suggestions with no suggestion using participant satisfaction, successful task completion and interruption frequency.
6. Report hardware/model/version, processing FPS, event-generation p50/p95, ASR delay after end of speech, state-query p95, CPU/RAM/GPU, dropped items, queue delay and timestamp error. Run an eight-hour consented soak before claiming sustained operation.

Targets from the product proposal (15–30 FPS, p95 200 ms lightweight event generation, p95 1 s ASR updates, p95 50 ms state queries) remain unverified engineering targets. No performance result is fabricated.

## Deferred capabilities

Stable biometric re-identification; diarized speaker-to-face association; trained and validated confusion/interest/engagement models; external fact verification; deployment-validated needs predictions; validated personal adaptation; SAM3/VLM/REA/LiveKit/Pipecat adapters; authenticated remote transport; eight-hour hardware soak. These require additional code, assets and evaluation. Stream-local IoU tracking, evidence-citing model-assisted contradiction candidates and held-out calibration infrastructure are now implemented; actual trained-model execution and real labeled validation remain unverified. The alpha's local SDK/MCP provides the contract and working measurable subset.

## Host hardware smoke test

In a normal host terminal, install the package and corresponding extras, then run:

```bash
pip install -e '.[vision,audio]'
python scripts/verify_live.py --consent --camera 0 --audio --seconds 30
```

Optionally add `--face-model /path/face.task --vad-model /path/vad.jit --asr-model /path/whisper-directory` after installing `.[mediapipe,vad,asr]`. Speak during the test. The JSON report shows actual signal counts, device errors and modalities without saving images, audio or transcript text. A successful basic test proves only that input flows through installed adapters, not the proposal's full semantic or performance targets.
