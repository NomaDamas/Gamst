# Deterministic physical-state precomputation

**Status: implemented adapter; real-weight acceptance pending.** This provider
has no learned model or weights, so there is no weight download or weight-parity
test to perform. The status above follows the repository's candidate acceptance
convention; calibrated-device replay, held-out trajectory/contact validation,
and physical-parameter accuracy acceptance also remain pending.

`gamst.providers.physical_state.PhysicalState` is an optional, dependency-free
CPU provider. It converts complete, caller-associated **calibrated metric 3-D**
snapshots into bounded `physics.state` observations. It does not capture a
camera, microphone or tactile device, download a model, infer depth, associate
objects, or generate future images. The existing default providers are unchanged.

The implementation reuses `MetricKinematics` for source/calibration ownership,
strict capture ordering, discontinuity handling and output-frame translation.
`ImageTrajectories` and `CompensatedTrajectories` remain the observed 2-D image
paths; this provider deliberately abstains rather than treating their normalized
coordinates as metres. `GroundPlane` remains the explicit image-to-plane
projection path. PhysicalState consumes already calibrated 3-D plane geometry;
it does not duplicate homography estimation or declare bounding-box contact.

## Input boundary and units

Submit a structured video `InputFrame`, with `SensorMetadata` matching the
configured calibration and coordinate frame. No pixels or raw audio/tactile
samples are accepted. Audio and tactile entries are upstream observations on the
same declared clock.

```python
from gamst import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.physical_state import PhysicalState

provider = PhysicalState(
    calibration_id='depth-cal-1', input_frame='world', output_frame='world',
    clock_id='capture-clock', gravity_m_s2=(0, 0, -9.80665),
    horizon_ms=600, prediction_steps=3,
    jerk_spectral_density_m2_s5=10.0,
)

frame = InputFrame({
    'units': 'm',
    'tracker_id': 'my-tracker', 'tracker_version': 'tracker-v1',
    'clock_id': 'capture-clock',
    'objects': [{
        'track_id': 'ball', 'position_m': [0, 0, 1],
        'position_variance_m2': [0.0001, 0.0001, 0.0004],
        'input_dimensionality': 3, 'calibrated': True,
        'shape': {'kind': 'sphere', 'radius_m': 0.1},
        'linear_drag_coefficient_s_inv': 0.0,
        'mass_kg': None, 'mass_calibrated': False,
        'is_container': False,
    }],
    'planes': [{
        'plane_id': 'floor', 'point_m': [0, 0, 0],
        'normal_unit': [0, 0, 1],
        'friction_coefficient': 0.4, 'restitution': None,
    }],
    'audio_onsets': [],
    'tactile_events': [],
}, 'video', 0, sensor=SensorMetadata(
    'camera', 'depth', 0, 'world', calibration_id='depth-cal-1',
    stream_epoch='stream-1'))
observation = provider.process(frame)
```

Every top-level and per-object field shown is required. A snapshot contains at
most eight unique tracks, four planes, eight audio onsets and eight tactile
events. Positions and shape dimensions are metres, velocity is m/s, acceleration
is m/s², variance is m², linear drag is 1/s, mass is kg, force is N, impulse is
N·s, and timestamps are integer milliseconds. Plane normals must be unit length
and point toward the supported object. A plane `restitution=null` selects the
explicit provider default; its friction may be `null` to force grounded-model
abstention. Axis-aligned boxes use `{'kind':'aabb',
'half_extents_m':[hx,hy,hz]}`. Shapes, planes and gravity are expressed in the
configured output frame. Translation through `MetricKinematics` is supported;
non-identity rotation is rejected because this adapter's diagonal measurement
covariance and AABB semantics must not be silently rotated into a different
contract.

A complete snapshot retires tracks that disappeared. Source, subject, tracker,
calibration, stream epoch, sequence-kind, skipped-sequence and time-gap changes
start new filter segments. Invalid input, reset and close clear state. A first
sample initializes position from its measurement but retains deliberately broad
velocity/acceleration priors; `status=initializing` and the covariance expose
that warmup rather than presenting the zero prior as a measurement.

### Explicit 2-D/uncalibrated abstention

An image-only track must use `input_dimensionality=2`, `calibrated=false`, null
`position_m`, `position_variance_m2`, shape, drag and mass, and false
`mass_calibrated`/`is_container`. It returns `status=abstained`, reason
`calibrated_metric_3d_position_and_variance_required`, null state/covariance and
no hypotheses. It cannot generate contacts, relations or aligned physical
events. Supplying numeric metric fields while claiming no calibration is
rejected; the provider never silently upgrades 2-D image motion to physics.

## 1. Constant-acceleration Kalman state

Each calibrated track has a nine-element state ordered as:

```text
[x_m, y_m, z_m, vx_m_s, vy_m_s, vz_m_s, ax_m_s2, ay_m_s2, az_m_s2]
```

The transition per axis for capture interval `dt` is
`p' = p + dt*v + dt²*a/2`, `v' = v + dt*a`, `a' = a`. Process noise is the
standard continuous white-jerk covariance; the configured spectral density has
units m²/s⁵. Position measurements use their three explicit variances and
Joseph-form scalar updates. The output includes the full symmetric 9×9
covariance, state order, `last_observed_ms`, sample count and segment ID. The
filter assumes independent position-noise axes; callers must not encode an
already correlated full covariance as three unrelated variances.

Orientation quaternion, angular velocity and angular acceleration are explicit
null abstentions. Translational tracks do not establish orientation or angular
dynamics. Finite-difference jerk, identity, physical calibration and
measurement accuracy are not inferred.

## 2. Multiple short-horizon trajectory hypotheses

Every calibrated object returns two entries under `hypotheses`:

- `free_fall_gravity_linear_drag` uses the filtered position/velocity and the
  analytic solution of `a = gravity - drag*velocity`. With zero drag it reduces
  to the constant-gravity projectile equation.
- `grounded_coulomb_friction` is estimated only when a supporting-plane contact
  is present and its friction coefficient is supplied. It projects velocity
  onto the plane and integrates tangent gravity plus Coulomb friction with a
  bounded substep. Otherwise it returns `abstained_no_supporting_plane` or
  `abstained_friction_unknown` and an empty prediction list.

Each hypothesis contains one to four future `predictions` within the configured
10–2000 ms horizon. A prediction has original-clock timestamp, horizon, metric
position/velocity and a 3×3 position covariance. `envelope_radius_m` is three
times the square root of the trace of that covariance. The covariance is a
constant-acceleration/white-jerk linearized envelope for hypothesis comparison,
not a calibrated collision probability or guaranteed error bound;
`coverage_calibrated=false` is explicit. Parameter uncertainty, unmodelled
aerodynamics, restitution errors and contact-model error are not magically
covered by this radius. Validate held-out future detections with position error
and empirical envelope calibration before relying on a deployment threshold.

## 3. Contact, tactile evidence, impulse and force abstention

Sphere/sphere, AABB/AABB, mixed-shape and track/plane candidates combine contact
geometry with relative normal velocity. Output includes gap, penetration,
normal, approaching status and the frame-aligned timestamp; no subframe impact
time is invented. Mixed curved/box geometry is conservative and all contact is
a candidate, never a verified physical event. A velocity warmup yields
`geometry_only_velocity_warmup` rather than a derivative fabricated from one
point.

Tactile entries have this exact schema:

```python
{
    'event_id': 'touch-17', 'timestamp_ms': 500, 'track_id': 'ball',
    'force_spike': True, 'slip': False,
    'force_n': 2.5, 'force_calibrated': True,
}
```

`force_spike` and `slip` are upstream observations, not inferences made from
image motion. Related tactile IDs and slip evidence are attached to contacts.
A calibrated tactile force remains identified as a **tactile measurement**, not
the contact's normal force. An uncalibrated numeric force is rejected; use
`force_n=null`, `force_calibrated=false`.

Without calibrated mass and an approaching relative normal velocity, contact
stays `contact_candidate`, `normal_impulse_n_s=null`, and `normal_force_n=null`.
With calibrated masses, the adapter may emit a restitution-model normal impulse
for an approaching collision. It still emits no numeric normal force: contact
duration is unknown. Mass, restitution, friction, geometry and tactile
calibration are caller declarations and remain unverified by the SDK.

## 4. Support and containment relations

Plane/object contacts whose geometry opposes gravity produce `support`
candidates with supporter/supported IDs, gap and null normal force. Objects
marked `is_container=true` produce `containment` candidates when the complete
sphere/AABB geometry is inside another sphere/AABB; `clearance_m` reports the
minimum geometric margin. These relations describe the supplied approximate
shapes only. They do not establish a stable support polygon, static friction,
a closed container, attachment, grasp intent or verified physical truth.
Contact, relation and aligned-event lists have deterministic caps of 12, 16
and 4 respectively; summary counts and truncation flags preserve the
distinction between absence and an output cap.

## 5. Cross-modal physical-event alignment

Audio onset entries contain `event_id`, `timestamp_ms`, `impact_like`, optional
0–1 `confidence`, and optional `track_id`. The adapter retains a short bounded
history of visible deceleration events, defined by positive
`-dot(acceleration, velocity)/speed` above the configured threshold after filter
warmup. An `impact_candidate` is emitted only when:

1. an impact-like audio onset,
2. a tactile `force_spike` for the same track, and
3. visible deceleration for that track

have a total timestamp span no greater than `event_window_ms`. If audio has a
track assertion it must agree with the tactile track. Output preserves all
three timestamps and evidence IDs, the span, audio confidence, visible
deceleration and separately calibrated tactile force/slip. No fused probability
is invented. The common clock is a required caller declaration;
`clock_alignment_verified=false` and `event_verified=false` remain explicit.
RF/radar motion, grasp/fall classifiers and general audio-to-object localization
are outside this adapter.

## Validation and limits

Run only the dependency-free synthetic module:

```sh
PYTHONPATH=tests:. .venv/bin/python -m unittest test_physical_state -v
```

The tests compare zero-drag projectile predictions with analytic futures,
check growing covariance envelopes, detect a synthetic sphere collision, gate
impulse on calibrated mass, verify support/containment and three-modality event
alignment, and require 2-D abstention. They do not establish real sensor
calibration, physical accuracy, force accuracy, envelope coverage or realtime
performance.

Learned world models are **optional future work and are not implemented** here.
[Cosmos](https://github.com/NVIDIA/Cosmos),
[V-JEPA 2](https://arxiv.org/html/2506.09985), and
[Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) may be evaluated
later for action-conditioned rollouts, latent prediction or refined simulation.
Their generated futures/contact forces must remain probabilistic hypotheses,
never measurements or substitutes for held-out physical validation. This
provider always reports `learned_world_model_used=false`.
