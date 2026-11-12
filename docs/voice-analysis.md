# Voice analysis and real-time context

The installed stack currently uses Whisper small for transcripts, Silero VAD for
speech activity, and AudioLevelPlugin for RMS/dBFS/energy. It does not currently
load a voice emotion, pitch, tremor or clinical voice-state model.

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

These are model options, not newly installed or accuracy-validated backends.

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
