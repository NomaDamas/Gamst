# Multimodal decision providers

`perception.multimodal_decision` has two explicit implementations of the same
`decision.multimodal` output contract:

- `rules` is the default, deterministic, zero-dependency
  `MultimodalDecisionProvider`.
- `intern-decision-2b` and `intern-decision-4b` are optional learned local
  providers. They are never selected implicitly and never fall back to rules.

Both providers select fresh registered observations, keep only the newest item
per capability, apply `max_evidence`, and calculate the input watermark and
evidence ages in host code. Both return `decision`, `selected_capabilities`,
`evidence_ids`, `reason`, `confidence`, `abstained`, `modalities`,
`input_watermark`, `input_window_ms`, `evidence_ages_ms`, and
`requested_capabilities`.

## Pinned checkpoints and runtime

| Factory kind | Hugging Face model | Pinned revision |
|---|---|---|
| `intern-decision-2b` | `internlm/Intern-Decision-2B` | `85c5657708a09884ff9fd8d7f57789b410a97b33` |
| `intern-decision-4b` | `internlm/Intern-Decision-4B` | `38c9dafbbef52dd04bae943c0c6ac918419adfd5` |

The adapter verifies the checkpoint's `inference.py` and `config.json` against
hashes from those revisions before executing the official local inference
wrapper. Weights are loaded with the wrapper's `local_files_only=True` path.
Importing GAMST, discovering providers, importing this adapter, and constructing
the rules provider do not import Torch/Transformers or load/download weights.
An explicit existing absolute local model directory is required for a learned
provider. A missing/incompatible runtime or checkpoint raises at construction;
there is no fake or deterministic fallback.

The official checkpoint requirements select **Python 3.12+**, Torch **2.9.1**,
TorchVision **0.24.1**, Transformers **5.14.1**, and Pillow **>=10,<13**.
Install the `decision` extra only in a dedicated environment compatible with the
other providers selected for that process:

```sh
python3.12 -m venv /opt/gamst-decision-venv
/opt/gamst-decision-venv/bin/python -m pip install -e '.[decision]'
# Explicit asset provisioning is separate from GAMST construction/startup.
/opt/gamst-decision-venv/bin/hf download internlm/Intern-Decision-2B \
  --revision 85c5657708a09884ff9fd8d7f57789b410a97b33 \
  --local-dir /models/Intern-Decision-2B
```

The upstream release tests native HF inference on Linux/CUDA/BF16 with SDPA.
GAMST also exposes explicit `device='cpu'` and dtype choices, but does not claim
upstream-equivalent CPU latency or quality. Model loading and first inference
have their own costs.

## Usage

```python
from gamst.providers.intern_decision import decision_provider

rules = decision_provider()  # deterministic default; no optional dependencies

learned = decision_provider(
    'intern-decision-2b',
    model_path='/models/Intern-Decision-2B',
    device='cuda',
    dtype='bfloat16',
    deadline_ms=250,
    min_confidence=0.6,
    max_evidence=16,
)
value = learned.decide(
    retained_registered_observation_records,
    now_ms=event_time_ms,
    requested_capabilities=['vision.objects', 'audio.events'],
    input_window_ms=5000,
)
learned.close()
```

Here `retained_registered_observation_records` and `event_time_ms` are supplied
by the host's observation store/event loop. `learned.spec` has the same output
fields as the rules provider, and `learned.next_sensor()` supplies sensor
metadata when publishing the returned value. The existing `Gamst` convenience
method continues to construct the rules provider by default.

This is an internal record-decision adapter, not a frame-processing
`ProcessProvider` worker. Consequently its factory kinds are not entries in
`provider_profiles.py`, whose workers implement sensor `process` operations.
Use the dedicated interpreter/environment for this provider and do not combine
it with incompatible Transformers profiles in one environment.

## Model boundary and abstention

Only bounded observation values, capability/modality names, evidence ages,
window size, and requested capabilities enter `DecisionEngine.predict`. The
request has no `images`, media bytes, record IDs, or free-form model-generated
provenance. Evidence IDs, selected capabilities, timestamps, and the watermark
are copied only from the host selection. Text inside observations is untrusted
evidence, not instructions.

The single official `choice` question has this fixed ordered candidate set:

1. `fuse_multimodal_context`
2. `escalate_vlm`
3. `request_fresh_sensor`
4. `request_more_audio`
5. `request_more_video`
6. `insufficient_evidence`

The provider validates the answer type, candidate distribution, selected choice,
and finite confidence. It returns `decision='insufficient_evidence'` and
`abstained=true` for deadline expiry, malformed/schema-inconsistent output,
unknown choices, unavailable fresh evidence, stale-only input, model inference
failure, or confidence below `min_confidence` (default **0.6**). A synchronous
Python model call cannot be forcibly cancelled at the default **250 ms**
deadline: its late result is discarded, and the provider prevents another model
call from overlapping it.

The adapter constructs the engine with `temperature=1.0`; therefore the output
field remains **uncalibrated candidate `confidence`**, not a calibrated GAMST
probability. The checkpoint cards' separately fitted temperature scaling and
the paper's 96-case calibration pilot are not calibration on GAMST traces.
`min_confidence` is a fail-closed coverage gate, not evidence of calibration.
A deployment seeking calibrated confidence must fit and validate it separately
on representative GAMST data.

## License and verification

The Intern-Decision repository and both selected model cards declare
**Apache-2.0**. The checkpoints derive from Qwen3.5 and ship `LICENSE-QWEN`;
that notice is also Apache-2.0 at the pinned revisions. Retain both licenses and
applicable upstream notices when redistributing. Runtime libraries and any
substituted checkpoint require their own review.

The software tests inject fake `DecisionEngine.predict` callables and require no
weights. They cover schema validation, host-owned evidence IDs, fixed-choice
enforcement, raw-media exclusion, timeout/low-confidence/stale abstention, the
rules default, and unavailable-runtime construction errors. They do not
establish trained-model accuracy, calibration, latency, or device acceptance.
