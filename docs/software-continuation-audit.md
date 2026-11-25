# Software continuation audit — 2026-11-15

The user authorized completion of the remaining approved software, reviews, CI
and merges, with actual experiments, model installation and model/sensor
validation deferred. This audit covers that software phase. It does not close
the [24-family product acceptance map](perception-acceptance.md).

## Integrated software

- The former persistent recording A/B/C runtime and full-lifetime evaluator are
  preserved as historical contracts in the [benchmark and evaluation archive](archive/benchmarks-and-evaluation.md);
  they are not current SDK facilities.
- Selected provider additions: MediaPipe VIDEO gestures and camera/model-bound
  head orientation; bounded tennis image events, exact reference-bound rules and
  supplied projective ground-plane transforms; STFT/MFCC and eligibility-gated
  HNR/jitter/shimmer; explicit WebRTC noise suppression and automatic gain control.
  Their provider documents state input contracts, abstention and lifecycle limits.
- The [candidate matrix](candidate-execution-matrix.md) accounts separately for
  every named candidate, dependency/license boundary and unsupported alternative.
  API/MCP publication and optional-runtime CI coverage accompany selected paths.

## Validation and independent review

The 20-recording A/B/C regression executes 60 actual fixture children, preserves
all original task/source/runtime evidence, and finalizes synthetic original
financial documents. Its local continuation run passed in 238.475 seconds. The
fixture freezes a 500 ms acquisition lateness allowance before activation for
loaded CI; this is not a deployment latency claim. Synthetic cost gates can pass
while actual acceptance and overall quality/latency gates remain false.

Focused runtime/failure-prefix, financial-boundary, provider and numerical
regressions have been run. Independent GPT-6.1 Sol High reviews covered runtime,
finance, gesture, prosody, sports/reference geometry and duplex changes. The
projective review found raw-coefficient underflow; normalized computation now
passes positive/negative subnormal scale-equivalence tests while preserving the
original configuration identity. Final head-orientation and integration reviews,
stable-head full-suite results and CI/merge evidence belong in the integration PR.
No pending check is marked passed by this document.

Reproduction from the integration checkout (already provisioned interpreter):

```sh
python3 -m unittest discover -s tests -v
node tests/dashboard_ui_test.cjs
python3 -m gamst demo
git diff --check
```

The CI core matrix uses Python 3.10–3.13. The separate motion-provider job runs
optional NumPy/OpenCV and native-interface fixture checks. Optional dependency
skips are explicit; mocked native boundaries and generated graphs prove software
contracts, not trained model accuracy. No model weights were installed for this
continuation and no live sensor capture or actual-model experiment was run.

## Deferred acceptance and external prerequisites

Real model/device compatibility, calibrated sensor truth, held-out quality,
acoustic performance, deployment memory/latency, source/isolation authenticity
and matched real A/B/C cost improvements remain unverified. Existing acceptance
issues, including #57/#135/#205/#206, remain governed by their real evidence gates.

RF-Pose/RF-Action reproducible runtime/assets and radio hardware (#189), authentic
per-request financial originals (#257/#277), and the upstream identity of
“multimodal Jev/cheap Jev” (#136) remain explicit prerequisites. An RSSI parser,
account aggregate or guessed alternative does not satisfy those requirements.
The [remaining sensory gates](remaining-sensory-gates.md) retain their context.
Halo demos are outside this continuation.
