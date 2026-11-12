# Output contract v0.1

All outputs are JSON-serializable dictionaries. Timestamps are Unix milliseconds. Evidence links use record IDs; modalities do not imply same-person ownership. The application provides subject attribution; scene detections remain frame-local.

| Layer | Type | Update |
| --- | --- | --- |
| Signal | observation.signal | Ingested measurement or supplied utterance/claim |
| Event | observation.event | Detected discrete change/request |
| State | observation.state | Requested current snapshot |
| Context | observation.context | Requested window summary |
| Inference | human.inference | Input-triggered hypotheses and response recommendation |

Required envelope fields: `schema_version`, `id`, `type`, `subject_id`, `timestamp_ms`, `window`, `models`, `quality`, `observation`, `evidence`. Inference includes `predictions.conversation`, `predictions.behavior`, `predictions.consistency`, `modalities`, and `recommended_action`.

`model_score` describes an upstream model's output, not a person's emotional truth. `probability` is null and `calibrated` false for all current built-in hypotheses. The validator rejects non-null uncalibrated probabilities. A future calibrated backend must define the outcome and calibration metadata; this package does not ship such a model. Quality reports evidence limitations rather than an invented confidence percentage.

`text.utterance` stores explicitly supplied text; raw microphone/camera samples are not persisted. Numeric signal trends describe measurements and do not diagnose the person. A matching utterance is a repeated utterance, not necessarily a question or confusion. Response delay is measured from ingestion of an assistant turn, not a verified end-of-playback timestamp.

Input methods may emit several stored records and return the final inference or signal. Plugin processing returns a list of produced signal records. Consumers should use `events()` for low-volume changes and `observe()` for inferences. Subscribers use bounded queues and drop oldest queued records on backpressure; inspect state diagnostics for drops. The history store is the audit path within retention.

Out-of-order inputs are flagged and current-state queries use timestamp ordering. Late observations may be stored within retention; they do not rewrite the previously delivered event stream. The system is not a cross-device clock synchronizer: applications must supply aligned Unix timestamps and verified subject attribution.

Retention is enforced on ingestion and queries. Deletion removes retained records and derived views; deleted data may still exist in an application's copies, filesystem snapshots or external backups. The SDK does not promise secure erasure of SQLite pages. No cloud transfer or model download is performed by the built-in adapters.
