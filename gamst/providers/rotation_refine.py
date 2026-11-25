"""Take-level rotation refinement: named detections seed a promptable video segmenter.

Live tracking at 240 fps has to be cheap, so it loses a small, glinting object whenever its
look changes. After a take, there is time for the slower, stronger path GAMST's design
reserves for sparse use:

1. **What** - an open-vocabulary detector (YOLOE, text prompt such as ``"metal coin"``) runs on
   every few frames; confident hits are grouped in time into episodes (one per throw).
2. **Which** - each episode's best detection seeds a promptable video segmenter (EfficientTAM,
   SAM 2 family) that follows that one object forwards and backwards through the episode.
   Identity comes from frame-to-frame continuity, not from appearance matching, so the object
   may change its look completely (face, edge, glint) without becoming another object.
3. **How it moved** - every frame's mask gives an ellipse; an empty mask between visible frames
   is the object edge-on. The same :class:`ObjectSpinCounter` turns the axis-ratio series into
   turns, and :func:`classify_trajectory` checks the path was a throw.

No model is trained or fine-tuned for the object; both models are used as published.
"""
from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata
from ..plugins import InputFrame
from .object_spin import ObjectSpinCounter
from .rotation import classify_trajectory


def episodes(detections, *, gap_ms=500, min_score=.2):
    """Group (timestamp_ms, frame_index, score, box) detections into throws.

    Each episode keeps its best hit as ``seed`` and the best hit of every frame as ``prompts``,
    the candidate seeds :func:`refine` follows one by one.
    """
    hits = sorted((item for item in detections if item[2] >= min_score), key=lambda item: item[0])
    groups, current = [], []
    for item in hits:
        if current and item[0] - current[-1][0] > gap_ms:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    output = []
    for group in groups:
        best = {}
        for item in group:
            if item[1] not in best or item[2] > best[item[1]][2]:
                best[item[1]] = item
        output.append(dict(seed=max(group, key=lambda item: item[2]), prompts=[best[index] for index in sorted(best)],
                           first_ms=group[0][0], last_ms=group[-1][0], hits=len(group)))
    return output


def ellipses(masks):
    """{frame_index: mask or None} -> {frame_index: (cx, cy, major, minor, angle_deg) or None}."""
    import cv2
    import numpy as np
    output = {}
    for index, mask in masks.items():
        if mask is None or int(mask.sum()) < 8:
            output[index] = None
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            output[index] = None
            continue
        (cx, cy), (a, b), angle = cv2.fitEllipse(contour)
        output[index] = (float(cx), float(cy), float(max(a, b)), float(min(a, b)),
                         float(angle if a >= b else angle + 90) % 180)
    return output


def flight_span(stamps, shapes, *, tolerance=.1):
    """(first, last) frame of the free flight around the apex, or None if no throw is visible.

    In the hand an object is pushed around; in the air it follows one parabola. Starting at the
    highest point, the window grows while a single parabola still fits within ``tolerance``
    object diameters, so release and catch end it without any hand detection.
    """
    import numpy as np
    seen = sorted(index for index, shape in shapes.items() if shape is not None)
    if len(seen) < 8:
        return None
    size = float(np.median([shapes[index][2] for index in seen]))
    t = {index: stamps[index] / 1000 for index in seen}

    def fits(window):
        ts = np.array([t[i] for i in window]) - t[window[0]]
        ys = np.array([shapes[i][1] for i in window])
        fit = np.polyfit(ts, ys, 2)
        rms = float(np.sqrt(np.mean((ys - np.polyval(fit, ts)) ** 2)))
        return rms <= tolerance * size, 2 * float(fit[0]) / size

    apex = seen.index(min(seen, key=lambda index: shapes[index][1]))
    lo, hi = max(0, apex - 2), min(len(seen) - 1, apex + 2)
    grown = True
    while grown:
        grown = False
        for step in ((-1, 0), (0, 1)):
            low, high = lo + step[0], hi + step[1]
            if low < 0 or high >= len(seen):
                continue
            ok, _ = fits(seen[low:high + 1])
            if ok:
                lo, hi, grown = low, high, True
    ok, accel = fits(seen[lo:hi + 1])
    if not ok or not 150 <= accel <= 3000 or t[seen[hi]] - t[seen[lo]] < .08:
        return None
    return seen[lo], seen[hi]


def count_spins(stamps, shapes, *, expected_max_rev_s=40.0, track_id='target', max_edge_frames=6,
                min_coverage=.8, span=None):
    """Spin events from per-frame ellipses; ``stamps[i]`` is frame ``i``'s capture time (ms).

    A short run of frames with no mask between frames with masks is taken as edge-on (minor axis
    ~0): a segmenter loses a coin exactly when it is a one-pixel line. Longer runs are unobserved.
    ``span`` (first, last) overrides :func:`flight_span`, e.g. to count a flight up to a frame.
    """
    import numpy as np
    seen = sorted(index for index, shape in shapes.items() if shape is not None)
    if len(seen) < 6:
        return []
    span = flight_span(stamps, shapes) if span is None else span
    if span is not None:
        seen = [index for index in seen if span[0] <= index <= span[1]]
        if len(seen) < 6:
            return []
    major = float(np.median([shapes[index][2] for index in seen]))
    edge_on = set()
    for before, after in zip(seen, seen[1:]):
        if 1 < after - before <= max_edge_frames + 1:
            edge_on.update(range(before + 1, after))
    counter = ObjectSpinCounter(expected_max_rev_s=expected_max_rev_s)
    events, path = [], []
    first, last = seen[0], seen[-1]
    for sequence, index in enumerate(range(first, last + 1)):
        shape = shapes.get(index)
        if shape is not None:
            axes, observed = [shape[2], shape[3]], True
            path.append((stamps[index], shape[0], shape[1], shape[2]))
        else:
            axes, observed = ([major, .02 * major], True) if index in edge_on else (None, False)
        sensor = SensorMetadata(source_id='refine', sensor_id=track_id, sequence=sequence,
                                capture_sequence=sequence, coordinate_frame='image', stream_epoch=track_id)
        events += counter.process(InputFrame({'objects': [{'track_id': track_id, 'axes': axes, 'observed': observed}]},
                                             'video', int(stamps[index]), subject_id='refine', sensor=sensor))['events']
    # Close the episode: one frame after the counter's gap limit with no observation.
    sensor = SensorMetadata(source_id='refine', sensor_id=track_id, sequence=last - first + 1,
                            capture_sequence=last - first + 1, coordinate_frame='image', stream_epoch=track_id)
    events += counter.process(InputFrame({'objects': []}, 'video', int(stamps[last]) + counter.max_gap_ms + 1,
                                         subject_id='refine', sensor=sensor))['events']
    ended = [event for event in events if event['type'] == 'object.spin.ended']
    trajectory = classify_trajectory(path)
    # Frames the object was lost for longer than an edge-on pass hide turns: the count is then
    # only a lower bound and is reported as partial rather than as a measurement.
    coverage = round((len(seen) + len(edge_on)) / (last - first + 1), 3)
    return [dict(event, trajectory=trajectory, coverage=coverage, flight_frames=[first, last],
                 status='partial' if coverage < min_coverage and event['status'] == 'counting' else event['status'])
            for event in ended]


class EfficientTAMVideo:
    """Bidirectional single-object mask tracking with the upstream EfficientTAM video predictor.

    ``source_dir`` is a checkout of yformer/EfficientTAM at the revision GAMST pins; every pinned
    source file and the checkpoint are verified by SHA256 before anything is imported or loaded.
    """

    def __init__(self, source_dir, checkpoint_path, checkpoint_sha256, *, device='cpu',
                 config='configs/efficienttam/efficienttam_ti_512x512.yaml'):
        import sys
        from .efficienttam_sources import SOURCES
        root = Path(source_dir).expanduser().resolve(strict=True)
        for name, digest in SOURCES.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'EfficientTAM source {name} differs from the pinned revision')
        if hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest() != checkpoint_sha256:
            raise ValueError('EfficientTAM checkpoint hash mismatch')
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
        self.predictor = build_efficienttam_video_predictor(config, str(checkpoint_path), device=device)

    def track(self, images, prompts):
        """``images``: upright BGR frames of one episode; ``prompts``: [(index, xyxy box)] of the one
        object. Returns {index: bool mask or None}, propagated both ways from the first prompt."""
        import cv2
        import numpy as np
        masks = {}
        with tempfile.TemporaryDirectory() as directory:
            for index, image in enumerate(images):
                cv2.imwrite(f'{directory}/{index:05d}.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            state = self.predictor.init_state(video_path=directory)
            for index, box in prompts:
                self.predictor.add_new_points_or_box(state, frame_idx=index, obj_id=1,
                                                     box=np.array(box, np.float32))
            first = min(index for index, _ in prompts)
            for reverse in (False, True):
                for index, _, logits in self.predictor.propagate_in_video(
                        state, start_frame_idx=first, reverse=reverse):
                    mask = (logits[0, 0] > 0).cpu().numpy()
                    masks[index] = mask if mask.any() else None
            self.predictor.reset_state(state)
        return masks


def tiles(height, width, *, size=640, overlap=96):
    """Top-left corners of overlapping ``size`` tiles covering the image at native resolution.

    Detectors resize their input to a fixed side; a small object in a tall frame shrinks below
    what they can see. Tiling keeps it at native size without changing the model.
    """
    def starts(extent):
        if extent <= size:
            return [0]
        values = list(range(0, extent - size + 1, size - overlap))
        return values if values[-1] + size >= extent else values + [extent - size]
    return [(x, y) for y in starts(height) for x in starts(width)]


def _rank(event):
    """Prefer a full count over a partial one, a throw over other motion, then what was seen."""
    return (event['status'] == 'counting', event['trajectory'].get('kind') == 'free_fall',
            event['coverage'], event['duration_ms'])


def moving_frames(stamps, grays, *, pad_ms=300, spread=4.0):
    """Frame indices around motion, for spending detector time only where something happens.

    ``grays``: [(frame_index, small grayscale image)] in time order. A sample moves when its
    difference from the previous sample exceeds the take's own noise floor (median +
    ``spread`` x MAD); every frame within ``pad_ms`` of a moving sample is returned.
    """
    import numpy as np
    if len(grays) < 3:
        return set(range(len(stamps)))
    energy = [(index, float(np.mean(np.abs(image.astype(np.int16) - previous.astype(np.int16)))))
              for (_, previous), (index, image) in zip(grays, grays[1:])]
    values = np.array([value for _, value in energy])
    floor = float(np.median(values))
    threshold = floor + spread * max(float(np.median(np.abs(values - floor))), .5)
    moving = [stamps[index] for index, value in energy if value > threshold]
    return {index for index, stamp in enumerate(stamps)
            if any(abs(stamp - moment) <= pad_ms for moment in moving)}


def refine(frames, detections, segmenter, *, label, pad_ms=350, expected_max_rev_s=40.0, max_seeds=3,
           stop_when_complete=False, max_throws=None, progress=None):
    """Refined rotation events for one take.

    ``frames``: list of (timestamp_ms, decode) where ``decode()`` returns the upright BGR frame;
    ``detections``: [(timestamp_ms, frame_index, score, (x0, y0, x1, y1))] for ``label``.

    An open-vocabulary hit can be the wrong thing (a button, a hand). Each of an episode's best
    ``max_seeds`` hits is followed on its own and only the best-observed throw is kept: a wrong
    seed yields no throw, or one that is less complete, instead of corrupting the right one.
    With ``stop_when_complete`` the next seeds are skipped once a seed has given a complete
    free-fall count (seen in at least 90% of the flight), which saves most of the time.
    ``max_throws`` prioritizes episodes by their best detection score and stops after that many
    complete throws. ``progress('tracking', done, total)`` reports each completed seed track.
    """
    if max_throws is not None and (type(max_throws) is not int or max_throws < 1):
        raise ValueError('max_throws must be a positive integer or None')
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable or None')

    def complete(event):
        return (event['status'] == 'counting' and event['trajectory'].get('kind') == 'free_fall'
                and event['coverage'] >= .9)

    results = []
    stamps = [stamp for stamp, _ in frames]
    slug = re.sub(r'[^A-Za-z0-9._-]+', '-', label).strip('-') or 'object'
    ordered = list(enumerate(episodes(detections)))
    total = sum(min(max_seeds, len(episode['prompts'])) for _, episode in ordered)
    done = complete_count = 0
    if max_throws is not None:
        ordered.sort(key=lambda item: -item[1]['seed'][2])
    for number, episode in ordered:
        start = next(i for i, stamp in enumerate(stamps) if stamp >= episode['first_ms'] - pad_ms)
        end = max(i for i, stamp in enumerate(stamps) if stamp <= episode['last_ms'] + pad_ms)
        images = [frames[index][1]() for index in range(start, end + 1)]
        candidates = []
        for _, index, score, box in sorted(episode['prompts'], key=lambda item: -item[2])[:max_seeds]:
            shapes = {start + i: shape for i, shape in ellipses(segmenter.track(images, [(index - start, box)])).items()}
            done += 1
            if progress is not None:
                progress('tracking', done, total)
            # The followed object's ellipse per frame ([frame, cx, cy, major, minor, angle]) is the
            # evidence behind the count: a render or an audit can replay it without the models.
            trace = [[i, *(round(v, 1) for v in shapes[i])] for i in sorted(shapes) if shapes[i] is not None]
            candidates += [dict(event, label=label, seed_score=round(float(score), 3), seed_frame=index,
                                seed_box=[round(float(v), 1) for v in box], frames=[start, end], shapes=trace)
                           for event in count_spins(stamps, shapes, expected_max_rev_s=expected_max_rev_s,
                                                    track_id=f'{slug}-{number + 1}')]
            best = max(candidates, key=_rank) if candidates else None
            if stop_when_complete and best is not None and complete(best):
                break
        if candidates:
            best = max(candidates, key=_rank)
            results.append(best)
            if max_throws is not None and complete(best):
                complete_count += 1
                if complete_count >= max_throws:
                    break
    if max_throws is not None:
        results.sort(key=lambda event: event['timestamp_ms'])
    return results


SPEC = CapabilitySpec(
    'vision.rotation.refined', 'vision.rotation.refined', 'video', 'gamst.rotation_refine', '1',
    (OutputField('objects', 'array'), OutputField('events', 'array')), 'offline',
    interpretation='measurement', sensor_requirements=('uint8-bgr-camera', 'capture-timestamps'),
    license='Apache-2.0')


async def publish(human, results, *, take_id, clock):
    """Store each refined throw as a registered observation and remember its named object.

    ``clock(capture_ms)`` maps a frame's capture time to the runtime's timestamp; every time an
    agent reads (record, ``started_ms``/``ended_ms``, event) is on that clock, the capture time is
    kept as ``capture_timestamp_ms``. ``flips`` are half turns. Each throw is one ``objects``
    snapshot, so :class:`ObjectMemory` keeps the object with its evidence.
    """
    human.register_capability(SPEC)  # idempotent for the same contract
    epoch = re.sub(r'[^A-Za-z0-9._-]+', '-', take_id)[:128]
    records = []
    for sequence, event in enumerate(results):
        ended = clock(event['timestamp_ms'])
        attributes = {key: event[key] for key in ('turns', 'rev_s', 'status', 'coverage', 'duration_ms')}
        attributes.update(flips=None if event['turns'] is None else round(2 * event['turns'], 1),
                          trajectory=event['trajectory'].get('kind'),
                          started_ms=ended - event['duration_ms'], ended_ms=ended)
        value = dict(objects=[dict(track_id=event['track_id'], label=event['label'], attributes=attributes)],
                     events=[dict(event, type='rotation.ended', axis='flip', source='refined', timestamp_ms=ended,
                                  capture_timestamp_ms=event['timestamp_ms'])])
        sensor = SensorMetadata(source_id='gamst.take', sensor_id='refine', sequence=sequence,
                                capture_sequence=sequence, coordinate_frame='image', stream_epoch=epoch)
        record = await human.push_observation(SPEC.id, value, sensor=sensor, timestamp_ms=ended)
        await human.remember_objects(record['id'], complete_snapshot=False)
        records.append(record)
    return records
