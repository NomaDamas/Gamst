# Stream-local person tracking

`PersonTracker` associates normalized detector boxes between frames using IoU. The OpenCV plugin emits `scene.person_tracks` alongside counts and boxes. Custom SAM/YOLO detectors may call the same tracker after converting their boxes to normalized xywh.

```python
from humansense.tracking import PersonTracker
tracker = PersonTracker(iou_threshold=.2, max_age_ms=1500)
tracks = tracker.update(boxes, timestamp_ms, stream_id="camera-0")
```

Track IDs persist across matching geometry and remain separate across streams. Missing tracks expire after the configured age. Stale timestamps are rejected without changing state. Crossing/overlapping candidates expose `association_quality="uncertain"` and sticky `association_history_uncertain`, because geometry cannot establish identity through a crossing. Reset clears geometry while preserving the instance's counter to prevent accidental ID reuse in retained history.

Every track sets `identity_verified=false` and `speaker_association="unavailable"`. A track is not a real name, biometric identity or verified audio speaker. No multi-camera re-identification, diarization, face-to-speaker association or SAM3 model is bundled. The detector must be separately installed and tested; tracker fixtures validate geometry, not perception accuracy.
