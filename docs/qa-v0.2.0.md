# v0.2.0 alpha QA

Tested on an Apple M2 Pro Mac with 32 GB memory, Python 3.10, on 2026-11-13 (Asia/Seoul). Measurements below are observations from this host, not universal latency/accuracy guarantees.

## Verified

- 124 automated tests passed, including compact-context raw-media exclusion, background queue bounds, stop cancellation, automatic transcript-to-text ingestion, stale observations, sampled inference preserving speech transitions, and video drop behavior.
- Real local MediaPipe face model loaded and detected one face with 52 blendshapes in Google's official portrait fixture; a measured call took 29 ms.
- Silero JIT loaded and returned voice activity signals on the public JFK speech fixture.
- Whisper small int8 CPU transcription processed 12 seconds of padded English fixture audio in 5.54 seconds and 9 seconds of padded Korean system-generated speech in 4.00 seconds. These are offline totals; the live path accumulates three-second windows before inference.
- Pinning the input language and using greedy decoding removed an unrelated phrase observed in an earlier short-window Korean test. Some spelling/word-boundary errors remain; this is not a calibrated accuracy benchmark.
- Qwen3-VL 2B Q8 via local llama.cpp produced a scene description from the official portrait fixture (606 ms warm call) and a Korean Companion reply (79 ms warm call). Warm timings exclude model load and are not guaranteed for longer prompts.
- Live HTTP NDJSON, WebSocket context and MCP stdio bridge were exercised against the same dashboard runtime. A sample WebSocket run delivered context approximately every 250–252 ms. After aggregation optimization, the fixture run's video-ingest p95 was 42.0 ms; the latest ASR window took 1.61 seconds and the latest scene description took 722 ms.
- Stop and delete cleared generated test observations; no late background result repopulated the store.
- With explicit user consent, a final 15-second native camera/microphone smoke test processed **148 camera frames and 148 microphone windows, with zero dropped audio windows and no plugin/device errors**. Face measurements, voice activity and five local VLM scene descriptions were emitted. No live speech transcription was emitted in that session, so spoken-language behavior is validated only by fixtures. Raw camera/audio was not saved, and report output contains counters rather than live descriptions/transcripts.
- VLM backpressure intentionally replaced 39 pending video frames in the native smoke test. It does not attempt to describe every camera frame. Latest-frame replacement and freshness are part of the contract.
- Public frontend JavaScript syntax and external model asset availability were checked. HTTP deployment checks and package smoke tests are recorded in the GitHub release.

## Limits and follow-up

- Native browser control was unavailable in this session. Desktop/mobile visual QA and browser permission/button interactions were **not** automated. The reference design was inspected through its HTML/CSS; no pixel-perfect comparison is claimed.
- The public preview uses browser microphone RMS and MediaPipe face measurements. Its browser inference path has not been end-to-end exercised here. Whisper, Silero and local VLM QA refer to the Python stack.
- Three-second non-overlapping ASR windows can split words and produce transcription mistakes. Language pinning, low-score filtering and bounded queues reduce some issues but are not diarization or guaranteed speech recognition.
- No emotion accuracy, hidden-intention recognition, identity matching or lie detection claim is made. Blendshapes are facial measurements; VLM descriptions are unverified model output.
- macOS emitted duplicate AVFoundation class warnings from OpenCV/PyAV. The native and fixture tests completed without a crash, but other package/OS combinations require testing.
- Hardware QA is a short smoke test, not a long-duration soak or production qualification. Retention, cancellation and frame replacement are tested, but high-load behavior is hardware-dependent.

## Reproduce

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check site/app.js
# Download official fixtures to the ignored QA directory before model tests.
mkdir -p .local-run/qa
curl -fL https://storage.googleapis.com/mediapipe-assets/portrait.jpg -o .local-run/qa/portrait.jpg
curl -fL https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac -o .local-run/qa/jfk.flac
# Keep the local VLM server running, then:
.venv/bin/python scripts/qa_models.py
# Hardware tests require the operator's fresh consent; never start them implicitly.
```

The model installer writes source URLs, pinned revisions and SHA-256 hashes in `.local-models/manifest.json`. Runtime/model/QA artifacts are ignored by Git and are not included in release packages.
