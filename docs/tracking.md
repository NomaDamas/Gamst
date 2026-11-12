# Stream-local person tracking

`PersonTracker` associates normalized detector boxes between frames using IoU. The OpenCV plugin emits `scene.person_tracks` alongside counts and boxes. Custom SAM/YOLO detectors may call the same tracker after converting their boxes to normalized xywh.

```python
from humansense.tracking import PersonTracker
tracker = PersonTracker(iou_threshold=.2, max_age_ms=1500)
tracks = tracker.update(boxes, timestamp_ms, stream_id="camera-0")
```

Track IDs persist across matching geometry and remain separate across streams. Missing tracks expire after the configured age. Stale timestamps are rejected without changing state. Crossing/overlapping candidates expose `association_quality="uncertain"` and sticky `association_history_uncertain`, because geometry cannot establish identity through a crossing. Reset clears geometry while preserving the instance's counter to prevent accidental ID reuse in retained history.

Every track sets `identity_verified=false` and `speaker_association="unavailable"`. A track is not a real name, biometric identity or verified audio speaker. No multi-camera re-identification, diarization, face-to-speaker association or SAM3 model is bundled. The detector must be separately installed and tested; tracker fixtures validate geometry, not perception accuracy.


## Camera guides and runtime controls

The local demo overlays normalized MediaPipe face landmarks , body pose joints and visible OpenCV
person track boxes on the camera image. Face guides, body joints and person regions have
independent switches; they only change the overlay. Settings model switches
stop the corresponding inference and are shared with MCP `get_features` and
`configure_feature`. Stop capture before changing models. Old observations remain
in history; disabling a model does not erase evidence. Start, Stop and Clear are
separate buttons inside one rounded control group.

Friendly chart titles (for example, “감지된 사람 수” for `scene.person_count`)
do not change the machine-readable observation keys. Face movement charts are
split by **current-frame face index**. These indexes are not stable identity IDs
and must not be interpreted as a person's longitudinal emotional history.

## Whole body, face, multiple people, and emotion

Current responsibility boundaries:

- OpenCV HOG supplies whole-person rectangles and geometric stream-local track IDs.
  It can miss seated, cropped, occluded or overlapping people.
- MediaPipe Pose supplies up to four bodies with 33 landmarks, geometric wrist-above-shoulder indicators and temporary body tracks. Uncertain associations omit the track ID; low-visibility measurements are null.
- MediaPipe Face supplies up to four faces, landmarks and 52 blendshape values per face.
  A face can be present when the body is outside the frame.
- SUPERB supplies four voice emotion scores for the audio stream's rolling window.
  The microphone is not separated into speakers. Scores are not assigned to a face
  or body, even when there is only one visible person.
- Qwen VLM describes sampled scenes. Its prose does not establish identity, stable
  person tracking or an emotion ground truth.

The next integration contract should keep `stream_id`, `track_id`, `region_type`
(`body`, `face`, `object`), normalized geometry, source timestamp, association
quality and model provenance on every entity observation. Face-to-body association
must be nullable. Occlusion and ambiguous matches must create gaps or uncertainty,
not silently transfer one person's history to another. Speaker diarization and
speaker-to-face association are separate optional providers. Until both are
verified, voice emotion stays stream-level.

The installed pose provider emits skeletal keypoints for body motion; a segmentation
provider can emit object masks and tracked regions. [Meta SAM 3](https://github.com/facebookresearch/sam3)
is a candidate for segmentation and tracking, not a bundled pose or emotion model.
No SAM weights, SAM execution adapter or mask overlay is installed here. Body pose is installed separately and does not use SAM.
The existing plugin API is the integration boundary, not a tested SAM integration.

For future per-person emotion tracking, a classifier must emit entity-linked
scores plus timestamps and model metadata. Trend storage should retain missing
and uncertain associations rather than smoothing through them. The current demo
plots stream-level voice scores and per-frame facial movements separately.
