# Register, predict, verify, calibrate

The SDK supports custom interaction targets without pretending a prompt trains a model. Supply a score from an actual detector/scorer, record the subsequent observed outcome, and fit calibration on separate labeled datasets. No psychological model is bundled. Labels such as a real assistance request must be defined and obtained from participants or explicit application events.

```python
human.define_prediction(
    name="needs_help", target="user_requests_assistance",
    score_name="interaction.help_model_score",
    features=["speech", "text", "history"],
    scorer_model_id="my-help-scorer", scorer_model_version="1",
)
# Your model produced this score; do not fabricate it or call it a probability.
await human.push_signal("interaction.help_model_score", model_output, "text",
                        model_id="my-help-scorer", model_version="1")
inference = (await human.get_history(type="human.inference"))[-1]
# After the defined outcome actually becomes observable:
await human.record_feedback(inference["id"], observed_help_request,
                            prediction="needs_help")
```

`features` documents the scorer inputs, not an automatically generated feature extractor or a training prompt. The runtime checks model ID/version, finite score range and quality. Absent calibration, `probability=null`. Feedback stores the original prediction snapshot and links the inference, so it cannot silently change previous predictions. Feedback does not automatically retrain the model.

## Fit held-out calibration

Create train/validation JSONL files with `{"id":"unique-event-id","score":0.8,"outcome":true}`. Scores are prior model outputs and outcomes are later observed events. Separate participants/sessions/time ranges, use globally unique IDs, and select thresholds before examining held-out results. The tool detects ID overlap, not semantic duplicate sessions or invalid target labels.

```bash
python -m humansense.calibrate \
  --training train.jsonl --validation validation.jsonl \
  --training-dataset train-participants-v1 --validation-dataset heldout-participants-v1 \
  --target user_requests_assistance \
  --scorer-model-id my-help-scorer --scorer-model-version 1 \
  --output help-calibration.json
```

The equal-width histogram fits probabilities as actual training-bin event frequencies. It reports held-out Brier score, ECE, sample counts and coverage. Default minimums are 20 training and 20 held-out samples per supported bin. Unsupported bins, local calibration error or failed global acceptance criteria cause abstention. An artifact is still written when validation fails, but the command exits 1 and the runtime will not emit a probability. This is an empirical baseline, not evidence that its target transfers to new populations.

```python
from pathlib import Path
from humansense import CalibrationArtifact
artifact = CalibrationArtifact.from_json(Path("help-calibration.json").read_text())
human.define_prediction("needs_help", target=artifact.target,
    score_name="interaction.help_model_score", calibration=artifact)
```

Artifacts are trusted data, not signed or proof that the dataset exists. Verify dataset provenance and validation conditions yourself. Synthetic test fixtures prove arithmetic and safeguards only. `delete_data` removes runtime feedback and observations and drops active imported calibration artifacts because the runtime cannot untrain unknown subject contributions. External artifact files/backups are managed by the application.
