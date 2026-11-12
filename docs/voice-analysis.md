# Voice analysis and real-time context

The installed stack currently uses Whisper small for transcripts, Silero VAD for
speech activity, and AudioLevelPlugin for RMS/dBFS/energy. The optional SUPERB Wav2Vec2 emotion model can now be installed into this
checkout. Pitch, tremor and clinical voice-state models are not bundled.

The important runtime contract is continuous context delivery. Audio collection
and context streaming continue independently of a classifier. A classifier's
first output requires its observation window plus model inference time; context
snapshots do not have to wait for it. Switching dashboard tabs does not stop any
of these paths.

## Public model options

- [SUPERB Wav2Vec2 emotion recognition](https://huggingface.co/superb/wav2vec2-base-superb-er):
  the model card identifies English, Apache-2.0, 16 kHz input and four IEMOCAP
  emotion classes. Its Transformers audio-classification interface fits the
  existing adapter. Korean accuracy has not been measured in this project.
- [emotion2vec+ base](https://huggingface.co/emotion2vec/emotion2vec_plus_base):
  the model card lists nine categories and provides FunASR/ModelScope inference.
  Its weights have a separately identified model license rather than a standard
  Apache-2.0 declaration. It needs its own adapter; it is not claimed to work
  through the Transformers adapter or installed by default.

SUPERB is the supported optional local install. emotion2vec+ remains an adapter
candidate. Neither has a project-specific accuracy certification.

## Install, use and remove

```bash
sh scripts/setup_voice.sh
# After stopping any active capture:
.venv/bin/python scripts/run_local.py --stop
.venv/bin/python scripts/run_local.py --language ko
```

The launcher detects `.local-models/voice-emotion/plugins.json` and loads the
classifier automatically unless `--without-voice-emotion` is specified. An explicit
`--plugins-config` takes precedence. Existing installed models are preserved.
Only the project virtual environment and owned model directory are changed.

In the dashboard camera panel, choose the speech recognition language: English (`en`), Chinese (`zh`), Korean (`ko`), or automatic detection. Stop capture before changing language; the next capture uses the new setting without a server restart. This controls Whisper transcription, not the voice emotion model or interface language.

Select Microphone, consent and Start. After the first 3 seconds,
Observability shows the latest four voice scores. Choose **시계열 보기**, or open
Categories/Time series → `classification.voice_emotion`, for bars, relative-score
donut and per-category histories. MCP context includes the same named observation.
Labels are `neu` (neutral), `hap` (happy), `ang` (angry), `sad` (sad).

```bash
.venv/bin/python scripts/run_local.py --stop
.venv/bin/python scripts/local_models.py remove-voice --yes
```

This removes only the voice model folder and its manifest entries. The optional
Python libraries stay in `.venv`; removing all models still uses `remove --yes`.

## Measured local verification

On the Apple M2 Pro, real public model-card WAV fixtures produced all four scores
and evidence IDs through MCP. Median model-call latency was about 130 ms for a
3-second window with two Torch CPU threads. The example happy clip's final window
had `hap=0.929`; the neutral clip had `neu=0.534`. These are fixture outputs, not
an accuracy evaluation or claims about Korean speech. Reproduce with
`PYTHONPATH=. .venv/bin/python scripts/qa_voice.py` after placing the model-card
fixtures in `.local-run/qa/` as described in that script.


## Rolling-window classification

The Hugging Face adapter accepts `window_seconds` and `hop_seconds`. Default
`window_seconds=0` preserves classification of individual input blocks. For voice
emotion, a starting configuration can analyze the latest 3 seconds every 1 second:

```json
[{"provider":"huggingface","capacity":4,"options":{
  "path":"./my-local-voice-emotion-model",
  "modality":"audio",
  "name":"classification.voice_emotion",
  "revision":"YOUR_PINNED_REVISION",
  "exclusive":true,
  "window_seconds":3,
  "hop_seconds":1
}}]
```

The model directory must already exist. Start with
`python scripts/run_local.py --plugins-config plugins.json` when the current
capture session is stopped. This does not download a model. The rolling buffer
is bounded; pause/reset/drop handling clears it. Changing the stream owner or
sample rate also starts a fresh window. Processing queues report overload rather
than claiming a guaranteed model output rate.

Classifier scores enter `classification.voice_emotion` in the same streamed
context, MCP output and category charts as other observations. Individual labels
can be selected in the chart's detail tabs. Scores retain the model's labels and
provenance; they are not converted into clinical measurements.
