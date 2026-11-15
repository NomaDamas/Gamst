"""Bounded task-label gesture scores with the existing hand geometry formatter."""
from dataclasses import replace
import math
from types import SimpleNamespace

from .hand_context import MediaPipeHandContext


GESTURES = ('None', 'Closed_Fist', 'Open_Palm', 'Pointing_Up', 'Thumb_Down',
            'Thumb_Up', 'Victory', 'ILoveYou')


class MediaPipeGestureContext:
    """Internal VIDEO formatter; the owning VIDEO adapter serializes calls."""
    def __init__(self, detector, image_api, *, sha256, max_hands=4):
        self.spec = self.capability(sha256, max_hands=max_hands)
        self.detector, self.result = detector, None
        self.closed = False
        self.geometry = MediaPipeHandContext(
            SimpleNamespace(detect=self._detect, close=lambda: None), image_api,
            sha256=sha256, max_hands=max_hands)

    @staticmethod
    def capability(sha256, *, max_hands=4):
        base = MediaPipeHandContext.capability(sha256, max_hands=max_hands)
        return replace(base, id='vision.mediapipe.gesture_context',
            output_name='vision.hand_gestures', provider='mediapipe.gesture.image',
            version=base.version+'.labels8.threshold05.v1')

    def _detect(self, image):
        self.result = self.detector.detect(image)
        return self.result

    @staticmethod
    def _categories(raw):
        if not isinstance(raw, (list, tuple)) or len(raw) > len(GESTURES):
            raise ValueError('at most eight gesture categories per hand required')
        values = []; seen = set()
        for category in raw:
            label = getattr(category, 'category_name', None)
            score = getattr(category, 'score', None)
            if (type(label) is not str or label not in GESTURES or label in seen or
                    type(score) not in (int, float) or not math.isfinite(score) or
                    not 0 <= score <= 1):
                raise ValueError('unsupported or malformed gesture category')
            seen.add(label)
            if score >= .5:
                values.append(dict(label=label, score=float(score)))
        return sorted(values, key=lambda item: (-item['score'], item['label']))

    def process(self, frame):
        if self.closed:
            raise RuntimeError('gesture formatter is closed')
        try:
            value = self.geometry.process(frame)
            raw = getattr(self.result, 'gestures', None)
            if not isinstance(raw, (list, tuple)) or len(raw) != len(value['hands']):
                raise ValueError('gesture/hand association count differs')
            for hand, categories in zip(value['hands'], raw):
                scores = self._categories(categories)
                hand['gesture_candidates'] = scores
                # The reference task's None is an explicit unknown class, not
                # proof that the hand performed no gesture or intended nothing.
                best = scores[0] if scores else None
                hand['gesture_status'] = 'candidate' if best and best['label'] != 'None' else 'unknown'
            value['summary'].update(gesture_classification='task_label_candidates',
                gesture_score_threshold=.5, confidence_calibrated=False,
                intent='unavailable', command_authority=False,
                smoothing='none_by_adapter')
            return self.spec.validate(value)
        finally:
            self.result = None  # No native result/landmark history is retained.

    def close(self):
        if not self.closed:
            self.closed = True
            self.result = None
            self.geometry.close()
