# Evidence-linked image event candidates

```mermaid
flowchart LR
  A[Observed path history] --> B[Explicit bounded image rules]
  B --> C[Candidate plus rule and observation references]
  C --> D[Context or MCP consumer]
```

The optional `event_rules` configuration adds ROI entry/exit, finite-line crossing
and axis-reversal candidates to `ImageTrajectories` and the owned
`rfdetr.trajectories` pixel worker. The provider owns geometry evaluation over its
existing bounded observed history. The application owns camera assumptions,
rule selection and sport-specific interpretation. This layer makes no score,
contact, possession or rule-violation claim and invokes no LLM/VLM.

```python
rules = [
    {'id': 'zone', 'kind': 'roi', 'bounds': [0.4, 0.3, 0.8, 0.7]},
    {'id': 'line', 'kind': 'line', 'start': [0.4, 0.2], 'end': [0.4, 0.8]},
    {'id': 'turn', 'kind': 'reversal', 'axis': 'x', 'min_displacement': 0.01},
]
# Local structured snapshots:
provider = ImageTrajectories(event_rules=rules)
# Or add to the existing explicit local pixel-worker configuration:
config['event_rules'] = rules
worker = await ProcessProvider.open(existing_python, 'rfdetr.trajectories', config)
```

Use the imports/lifecycle in [image trajectories](image-trajectories.md) or
[RF-DETR streaming trajectories](rfdetr-trajectories.md). Rules are disabled by
default; existing default output contracts/version hashes remain unchanged.
When enabled, one to four exact-schema rules with unique identifiers are copied
at construction and bind the capability version. Configuration is <=8KiB. Rule
coordinates are normalized image coordinates (x right, y down), not meters.
Rules are tied to that image geometry: the caller must configure a new provider
if a resized/reframed camera makes the chosen region semantically inappropriate.

ROI bounds are `[x_min,y_min,x_max,y_max]`, with positive width/height greater
than twice the fixed1e-6 boundary deadband. An entry candidate needs an outside
endpoint followed by a strictly inside endpoint; exit is the reverse. A point
in the boundary deadband abstains. Outside-to-outside travel through a region
is not reported as observed entry/exit. This is endpoint classification, not a
continuous occupancy detector.

Lines have distinct finite start/end points more than2e-6 apart. Candidates
require strict intersection of the finite configured segment and the segment
between the last two observed centroids. Touching endpoints, collinearity,
near-boundary cases and crossing only the infinite line extension abstain.
Signed side is the2D cross product `(end-start) × (point-start)`, so output
`line_negative_to_positive` / `line_positive_to_negative` refers to the declared
line direction in image coordinates. It is not a metric direction.

Reversal needs three consecutive retained observations; configuring it with
max_samples2 rejects. Axis must be x/y. Both signed displacements must exceed
min_displacement (1e-6..1) in magnitude and change sign. This is a displacement
threshold, not a speed or acceleration threshold. No smoothing or sport-specific
meaning is implied. Slow/noisy motion below the threshold abstains.

Enabled output adds `events` and `event_summary`. Each event has:

```python
{
    'kind': 'roi_entry', 'rule_index': 0, 'object_index': 0,
    'observation_indices': [6, 7], 'interpretation': 'candidate',
}
```

Resolve rule_index against event_summary.rules and object_index against the
**same output's** objects array. Then resolve observation_indices against that
object's observations array. These samples carry original capture timestamps,
centroids and evidence hashes; the object carries track/segment ID. Indices are
local to this one immutable observation, not stable across updates. Keep the
whole context record when persisting/forwarding an event. The event's time range
is the min/max of its referenced observed timestamps, not an exact event time.
Reversal references three points; ROI/line reference two. No intermediate or
future position is synthesized. Rules also have a canonical SHA256 in the
summary. Hashes provide provenance binding, not physical truth.

There are at most32 candidates (8 objects ×4 rules); all are emitted, with no
silent event truncation. Current detector metadata is stored once and joined by
track_id. Max Unicode labels, histories, rules and candidates are tested against
the64KiB value budget. `truncated=false` means no generated candidate was dropped;
it does not mean no real event was missed. `coverage=observed_endpoints_only` and
`between_observations=unobserved` apply even with a complete detector snapshot.
`input_snapshot_complete=false` marks truncated detector input. An empty events
array is not proof of no activity.

Gaps, missing tracks, ambiguous/new associations, source/epoch/geometry changes,
invalid input and unknown camera motion cannot create cross-segment events.
RF-DETR partial snapshots clear both stages and emit no candidates. Static-camera
mode is still caller-declared; undetected motion/ID errors can invalidate the
candidate. Rules do not change user delivery cadence or wake a consumer model.
Each delivered snapshot contains candidates for its latest pair/triple only;
a slower consumer must explicitly retain event records if it needs every event.

Tests cover analytic geometry, deadbands/finite segments, reversal, provenance
boundaries, output limits and actual pixel-worker/Context/MCP propagation with
predictor fixtures. #174 is the generic software slice. #134 still requires
camera compensation, improved sports tracking, task packs and independently
annotated real clips. Real cost/quality/latency acceptance remains #57/#135;
no models are installed by this feature and synthetic fixtures do not satisfy
trained-model or sports-accuracy gates.
