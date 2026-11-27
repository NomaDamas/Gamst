"""Who is speaking on screen: lips that move with the sound (active-speaker detection), per visible face.

Pipeline module :class:`ActiveSpeaker` reads the camera and the microphone of the same capture and, for every face
it can see, says from moment to moment how much its lips move with the sound. Three models, each in its own
environment (one worker process each), none of them new to GAMST except the last:

- faces: the ``vision.face`` role (MediaPipe Face Landmarker, VIDEO mode). Its points (eyes, nose, lips and the
  face's extent) come back as a private payload (:mod:`gamst.private_payload`) and are used here only;
- identity: the ``vision.face_identity`` role (LVFace), the same ArcFace alignment and embedding as
  :class:`gamst.providers.person_match.TargetPerson`, against the enrolled ``person`` reference (consent record);
- lips: the ``vision.active_speaker`` role, Light-ASD (Liao et al., CVPR 2023, MIT code and weights,
  github.com/Junhua-Liao/Light-ASD at ``SOURCE_REVISION``). Its input follows the upstream demo
  (``Columbia_test.py``): 25 fps grey 112x112 face crops (:func:`gamst.providers.talknet.prepare_talknet_face`,
  the same geometry) and 13 MFCC per 10 ms (:func:`gamst.providers.talknet.mfcc_frames`); a window is scored in
  pieces of 1..6 s and the six results are averaged, as upstream does. A score is the model's raw "speaking"
  logit (upstream calls a face speaking above 0); it is not a probability.

Faces are followed across frames by OC-SORT (:mod:`gamst.providers.ocsort`). A camera at another rate is put on
the model's 25 fps clock by capture time: each 40 ms slot takes the first frame captured in it.

Each record (``vision.active_speaker``) covers ``hop_ms`` of capture time, written once the ``lookahead_ms`` after it
was seen too (the model looks both ways), and holds numbers only: per face a stream-local track name, its identity
score against the enrolled person (cosine, uncalibrated; None without a reference or a usable face), its size in
pixels, and one value per 40 ms slot: the speaking logit, or None where the face was not seen well enough to say
(the gate, :meth:`ActiveSpeaker._gate`: smaller than ``min_face_px``, turned further than ``max_yaw``, the mouth
outside the picture, or sharper than ``min_sharpness`` it is not), and per face how many slots each reason left unread.
No position, point, crop or embedding is recorded.
"""
from __future__ import annotations

import base64
import hashlib
import inspect
import math

from ..capabilities import CapabilitySpec, OutputField

SLOT_MS = 40                                   # the model's 25 fps
SOURCE_REVISION = 'ed38c232de5efe0261dbd68627c0ade7cdfe14eb'
SOURCES = {                                    # upstream model files, in the order they import each other
    'Encoder.py': (6158, 'c4a85b8aa5eaab428f79836c5d9511a01ac45bfeb6dccc75e06c1e9eff25520c'),
    'Classifier.py': (1047, '21fdebc3c08ee3a386111719035f06461b29548b9b900eff3e17e4083f4377f8'),
    'Model.py': (1344, '6094bfaab7d5a701e1580ee6ec89c7cfea2baacf62b3ceafff5558f31dd3fc8d'),
}
# Weights in the upstream repository at SOURCE_REVISION. TalkSet is built from VoxCeleb2 and LRS3; LRS3 is
# for non-commercial research, so those weights are an opt-in (``allow_noncommercial``).
WEIGHTS = {
    'd44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317': 'pretrain_AVA_CVPR.model',
    'efc375833887eefa9d209dc92810e18519b04c3c73ea35a549f2a7f40b7d94d5': 'finetuning_TalkSet.model',
}
NONCOMMERCIAL = frozenset({'efc375833887eefa9d209dc92810e18519b04c3c73ea35a549f2a7f40b7d94d5'})
DURATIONS = (1, 2, 3, 4, 5, 6)   # upstream writes {1,1,1,2,2,2,3,3,4,5,6}: a Python set, so these six
MAX_FRAMES = 150                 # 6 s at 25 fps per model call


def check_weights(sha256, allow_noncommercial=False):
    if sha256 not in WEIGHTS:
        raise ValueError('Light-ASD weights must be one of the pinned upstream files')
    if sha256 in NONCOMMERCIAL and allow_noncommercial is not True:
        raise PermissionError('the TalkSet weights were trained on LRS3 (non-commercial): set allow_noncommercial')


# ---------------------------------------------------------------- the model (worker side)

class LightASD:
    """Light-ASD from a pinned, hash-checked source checkout and weight file; CPU or MPS float32."""

    def __init__(self, model, head, torch, device):
        self.model, self.head, self.torch, self.device = model, head, torch, device

    @classmethod
    def from_local(cls, source_dir, checkpoint_path, sha256, *, device='cpu', allow_noncommercial=False):
        from pathlib import Path
        from .talknet import _classes
        from .vocal_expression import _read
        check_weights(sha256, allow_noncommercial)
        blobs = {name: _read(Path(source_dir) / 'model' / name, size, digest) for name, (size, digest) in SOURCES.items()}
        checkpoint = _read(checkpoint_path, 16 * 1024 * 1024, sha256)
        import io
        import torch
        if device == 'mps' and not torch.backends.mps.is_available() or device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f'{device} is not available here')
        model_class = _classes(blobs, ('Model', 'ASD_Model'))
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            model = model_class().float().eval()
            head = torch.nn.Linear(128, 2).float().eval()
        weights = torch.load(io.BytesIO(checkpoint), map_location='cpu', weights_only=True)
        expected = {'model.' + key for key in model.state_dict()} | {'lossAV.FC.weight', 'lossAV.FC.bias',
                                                                       'lossV.FC.weight', 'lossV.FC.bias'}
        if not isinstance(weights, dict) or set(weights) != expected:
            raise ValueError('Light-ASD checkpoint must hold exactly the model and its heads')
        model.load_state_dict({key[6:]: value for key, value in weights.items() if key.startswith('model.')}, strict=True)
        head.load_state_dict({key[10:]: value for key, value in weights.items() if key.startswith('lossAV.FC.')},
                             strict=True)
        return cls(model.to(device), head.to(device), torch, device)

    def _logits(self, features, faces):
        torch, model = self.torch, self.model
        audio = torch.from_numpy(features).unsqueeze(0).to(self.device)
        video = torch.from_numpy(faces.astype('float32')).unsqueeze(0).to(self.device)
        out = model.forward_audio_visual_backend(model.forward_audio_frontend(audio), model.forward_visual_frontend(video))
        return self.head(out)[:, 1].float().cpu().numpy()     # upstream lossAV(labels=None): the "speaking" logit

    def scores(self, faces, pcm, durations=DURATIONS):
        """One logit per face frame: ``faces`` (T, 112, 112) uint8, ``pcm`` int16 of T * 640 + 240 samples."""
        import numpy as np
        from .talknet import mfcc_frames
        frames = len(faces)
        if not 1 <= frames <= MAX_FRAMES or faces.shape[1:] != (112, 112) or len(pcm) != frames * 640 + 240:
            raise ValueError('need 1..150 face frames and their T * 640 + 240 samples')
        features = mfcc_frames(pcm)[:4 * frames]
        results = []
        with self.torch.inference_mode():
            for seconds in durations:
                size, values = 25 * seconds, []
                for first in range(0, frames, size):
                    last = min(frames, first + size)
                    values.append(self._logits(np.ascontiguousarray(features[4 * first:4 * last]), faces[first:last]))
                results.append(np.concatenate(values)[:frames])
        scores = np.mean(results, axis=0).astype(np.float32)
        if scores.shape != (frames,) or not np.isfinite(scores).all():
            raise ValueError('invalid Light-ASD scores')
        return scores


def pack_window(faces, pcm):
    """The worker frame: face crops then PCM, as bytes (``unpack_window``)."""
    return faces.astype('uint8').tobytes() + pcm.astype('<i2').tobytes()


def unpack_window(data, frames):
    import numpy as np
    size = frames * 112 * 112
    if type(data) is not bytes or type(frames) is not int or not 1 <= frames <= MAX_FRAMES or \
            len(data) != size + 2 * (frames * 640 + 240):
        raise ValueError('an active-speaker window is T face crops and T * 640 + 240 samples')
    faces = np.frombuffer(data, np.uint8, count=size).reshape(frames, 112, 112).copy()
    return faces, np.frombuffer(data, '<i2', offset=size).astype(np.int16)


# ---------------------------------------------------------------- face geometry (no model)

def face_geometry(rows, width, height):
    """Pixel facts of one face from its private rows (``face_context.POINTS``); nothing here is recorded.

    Returns dict(box, points5, size, yaw, mouth_inside): ``box`` the extent of the face clipped to the picture,
    ``yaw`` how far the nose tip sits sideways (along the line of the eyes) from the middle of the eyes, in eye
    distances: 0 for a frontal face, growing as the face turns aside.
    """
    pixels = [(float(x) * width, float(y) * height) for x, y in rows]
    (x0, y0), (x1, y1) = pixels[7], pixels[8]
    inside = lambda point: 0 <= point[0] < width and 0 <= point[1] < height   # noqa: E731
    eyes = math.dist(pixels[0], pixels[1])
    middle = ((pixels[0][0] + pixels[1][0]) / 2, (pixels[0][1] + pixels[1][1]) / 2)
    axis = ((pixels[1][0] - pixels[0][0]) / eyes, (pixels[1][1] - pixels[0][1]) / eyes) if eyes >= 1 else None
    yaw = None if axis is None else abs((pixels[2][0] - middle[0]) * axis[0] + (pixels[2][1] - middle[1]) * axis[1]) / eyes
    box = (max(0., x0), max(0., y0), min(float(width), x1), min(float(height), y1))
    return dict(box=box, points5=pixels[:5], size=max(x1 - x0, y1 - y0), yaw=yaw,
                mouth_inside=all(inside(pixels[i]) for i in (3, 4, 5, 6)))


# ---------------------------------------------------------------- the Pipeline module

class _Track:
    def __init__(self):
        self.crops = {}          # slot -> 112x112 grey crop
        self.usable = {}         # slot -> the face was seen well enough to read its lips
        self.sizes = {}          # slot -> face size in pixels
        self.identity = []       # (timestamp_ms, cosine) against the enrolled face
        self.last_identity_ms = None


class ActiveSpeaker:
    """Camera + microphone in; per face and 40 ms, how much its lips move with the sound (``vision.active_speaker``)."""
    owns_process = True     # its three models run in their workers

    def __init__(self, faces, lips, identity=None, *, reference='person', hop_ms=1000, lookahead_ms=1000,
                 context_ms=2000, min_face_px=64, max_yaw=.6, min_sharpness=100., identity_every_ms=250, max_age_ms=600,
                 iou_threshold=.3, ring_seconds=20.):
        if hop_ms % SLOT_MS or lookahead_ms % SLOT_MS or context_ms % SLOT_MS or not SLOT_MS <= hop_ms <= 4000 or \
                (hop_ms + lookahead_ms + context_ms) // SLOT_MS > MAX_FRAMES:
            raise ValueError('hop, lookahead and context are whole 40 ms slots, together at most 6 s')
        from .audio_ring import AudioRing
        from .ocsort import OCSORTPersonTracker
        self.faces, self.lips, self.identity_model, self.reference_name = faces, lips, identity, reference
        self.hop_ms, self.lookahead_ms, self.context_ms = int(hop_ms), int(lookahead_ms), int(context_ms)
        self.min_face_px, self.max_yaw, self.identity_every_ms = float(min_face_px), float(max_yaw), int(identity_every_ms)
        self.min_sharpness = float(min_sharpness)
        self.ring = AudioRing(16000, ring_seconds)
        self.tracker = OCSORTPersonTracker(iou_threshold=iou_threshold, max_age_ms=max_age_ms)
        self.gallery = None
        model = getattr(lips, 'model_id', 'lightasd')
        self.spec = CapabilitySpec(
            'vision.active_speaker', 'vision.active_speaker', 'video', 'gamst.active_speaker',
            '1.' + hashlib.sha256(repr((model, getattr(faces, 'model_id', None), getattr(identity, 'model_id', None),
                                        hop_ms, lookahead_ms, context_ms, min_face_px, max_yaw, min_sharpness)).encode()).hexdigest()[:16],
            (OutputField('start_ms', 'integer'), OutputField('end_ms', 'integer'), OutputField('slot_ms', 'integer'),
             OutputField('faces', 'array'), OutputField('model', 'object')),
            'window', window_ms=self.hop_ms + self.lookahead_ms + self.context_ms, lookahead_ms=self.lookahead_ms,
            stateful=True, reset_on_gap=True, freshness_ms=5000, interpretation='estimate', devices=('cpu', 'mps'),
            license='adapter:Apache-2.0; Light-ASD code and weights MIT; LVFace MIT; MediaPipe Apache-2.0')
        self.reset()

    # -- references
    def bind(self, pipeline):
        reference = pipeline.references.get(self.reference_name)
        if reference is None:
            return      # no enrolled face: every face is reported with identity None
        if reference['kind'] != 'person':
            raise ValueError(f'active_speaker needs a person reference named {self.reference_name!r}')
        self.set_reference(reference['value'])

    def set_reference(self, value):
        from .person_match import PersonGallery
        face = (value or {}).get('face') or []
        if not face:
            raise ValueError('the person reference has no face embeddings')
        if self.identity_model is None or value.get('face_model') != self.identity_model.model_id:
            raise ValueError(f"the person reference was enrolled with {value.get('face_model')!r}; this module runs "
                             f"{getattr(self.identity_model, 'model_id', None)!r}")
        self.gallery = PersonGallery(face=face)

    def reset(self):
        self.ring.reset()
        self.tracker.reset()
        self.tracks = {}
        self.cursor = None          # start of the next record (ms, a slot edge)
        self.video_ms = None
        self.timings = dict(faces=[], lips=[], identity=[])

    async def close(self):
        for worker in (self.faces, self.lips, self.identity_model):
            close = getattr(worker, 'close', None)
            value = close() if callable(close) else None
            if inspect.isawaitable(value):
                await value

    # -- frames
    async def process(self, frame):
        if frame.modality == 'audio':
            if frame.data is not None:
                if frame.sample_rate != 16000:
                    raise ValueError('active_speaker needs 16 kHz mono audio')
                if self.ring.append(frame.data, int(frame.timestamp_ms), 16000):
                    self._restart()
            return await self._ready()
        if frame.modality != 'video' or not hasattr(frame.data, 'shape'):
            return None
        await self._see(frame.data, int(frame.timestamp_ms))
        return await self._ready()

    def _restart(self):
        """A gap in the sound: nothing is scored across it."""
        self.tracks, self.cursor = {}, None
        self.tracker.reset()

    async def _see(self, image, ts):
        import time
        from .person_match import align_face
        from .talknet import prepare_talknet_face
        height, width = image.shape[:2]
        if self.video_ms is not None and ts <= self.video_ms:
            return
        self.video_ms = ts
        if self.cursor is None:
            self.cursor = (ts // SLOT_MS + 1) * SLOT_MS
        started = time.perf_counter()
        rows = await self.faces.points(image, ts)
        self.timings['faces'].append(time.perf_counter() - started)
        found = [face_geometry(face, width, height) for face in rows]
        boxes = [dict(x=item['box'][0] / width, y=item['box'][1] / height, width=(item['box'][2] - item['box'][0]) / width,
                      height=(item['box'][3] - item['box'][1]) / height) for item in found]
        keep = [index for index, box in enumerate(boxes) if box['width'] > 0 and box['height'] > 0]
        tracked = self.tracker.update([boxes[index] for index in keep], ts, 'faces')
        slot = ts // SLOT_MS
        for item in tracked:
            if not item['visible']:
                continue
            index = next((index for index in keep if boxes[index] == item['bbox']), None)
            if index is None:
                continue
            face, name = found[index], 'face-' + item['track_id'].rsplit('-', 1)[-1]
            track = self.tracks.setdefault(name, _Track())
            x0, y0, x1, y1 = face['box']
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            why = self._gate(image, face)
            usable = why is None
            if slot not in track.crops:
                track.crops[slot] = prepare_talknet_face(image, (x0, y0, x1, y1))
                track.usable[slot], track.sizes[slot] = why, face['size']
            if self.gallery is not None and usable and (
                    track.last_identity_ms is None or ts - track.last_identity_ms >= self.identity_every_ms):
                if getattr(self.identity_model, 'face_input', 'aligned') == 'region':     # it finds the face itself
                    from .person_match import points_region
                    aligned = points_region(image, face['points5'])
                else:
                    aligned = align_face(image, face['points5'])
                if aligned is not None:
                    started = time.perf_counter()
                    try:
                        score = self.gallery.face_score(await self.identity_model.embed(aligned))
                    except ValueError:
                        score = None
                    self.timings['identity'].append(time.perf_counter() - started)
                    track.last_identity_ms = ts
                    if score is not None:
                        track.identity.append((ts, score))

    def _gate(self, image, face):
        """Why this face cannot be read (``small``, ``turned``, ``mouth_outside``, ``blurred``), or None.

        Sharpness is the variance of the Laplacian of the face, grey, resized to 112 x 112 (motion blur lowers it).
        """
        import cv2
        if face['size'] < self.min_face_px:
            return 'small'
        if face['yaw'] is None or face['yaw'] > self.max_yaw:
            return 'turned'
        if not face['mouth_inside']:
            return 'mouth_outside'
        x0, y0, x1, y1 = (int(round(value)) for value in face['box'])
        grey = cv2.cvtColor(cv2.resize(image[y0:y1, x0:x1], (112, 112)), cv2.COLOR_BGR2GRAY)
        return 'blurred' if cv2.Laplacian(grey, cv2.CV_64F).var() < self.min_sharpness else None

    async def _ready(self, final=False):
        """Records whose interval and lookahead have been seen and heard."""
        outputs = []
        while self.cursor is not None and self.video_ms is not None and self.ring.end_ms is not None:
            end = self.cursor + self.hop_ms
            seen = int(min(self.video_ms, self.ring.end_ms - 15))
            if not final and seen < end + self.lookahead_ms:
                break
            if final and seen < self.cursor + SLOT_MS:
                break
            outputs.append(await self._record(self.cursor, min(end, (seen // SLOT_MS + 1) * SLOT_MS) if final else end,
                                              seen))
            self.cursor = end
        return outputs or None

    async def _record(self, start, end, seen):
        import time
        import numpy as np
        first, last = start // SLOT_MS, end // SLOT_MS                   # slots [first, last)
        window_first = (start - self.context_ms) // SLOT_MS
        window_last = min(last + self.lookahead_ms // SLOT_MS, (seen - 15) // SLOT_MS)
        faces = []
        for name, track in sorted(self.tracks.items()):
            run = self._run(track, first, last, window_first, window_last)
            if run is None:
                continue
            begin, stop = run                                          # contiguous slots [begin, stop)
            pcm = self.ring.span(begin * SLOT_MS, stop * SLOT_MS + 15)
            if len(pcm) < (stop - begin) * 640 + 240 - 16 or stop - begin < 3:
                continue
            pcm = np.clip(np.round(np.asarray(pcm, np.float64) * 32768), -32768, 32767).astype(np.int16)
            pcm = np.pad(pcm, (0, max(0, (stop - begin) * 640 + 240 - len(pcm))))[:(stop - begin) * 640 + 240]
            crops = np.stack([track.crops[slot] for slot in range(begin, stop)])
            started = time.perf_counter()
            scores = await self.lips.scores(crops, pcm)
            self.timings['lips'].append(time.perf_counter() - started)
            speaking = [round(float(scores[slot - begin]), 3) if begin <= slot < stop and slot in track.usable
                        and track.usable[slot] is None else None for slot in range(first, last)]
            unread = {}
            for slot in range(first, last):
                if track.usable.get(slot):
                    unread[track.usable[slot]] = unread.get(track.usable[slot], 0) + 1
            if not any(value is not None for value in speaking) and not any(slot in track.crops for slot in range(first, last)):
                continue
            sizes = [track.sizes[slot] for slot in range(first, last) if slot in track.sizes]
            known = [score for ts, score in track.identity if start - 2000 <= ts < end + self.lookahead_ms]
            faces.append(dict(track=name, identity=round(sum(known) / len(known), 4) if known else None,
                              identity_samples=len(known), size_px=int(round(float(np.median(sizes)))) if sizes else None,
                              speaking=speaking, unread=unread))
        for track in self.tracks.values():                             # what no later window needs
            for slot in [slot for slot in track.crops if slot < window_first]:
                del track.crops[slot], track.usable[slot], track.sizes[slot]
            track.identity = [item for item in track.identity if item[0] >= start - 4000]
        self.tracks = {name: track for name, track in self.tracks.items() if track.crops}
        return self.spec.validate(dict(start_ms=int(start), end_ms=int(end), slot_ms=SLOT_MS, faces=faces,
                                       model=dict(lips=getattr(self.lips, 'model_id', None),
                                                  faces=getattr(self.faces, 'model_id', None),
                                                  identity=getattr(self.identity_model, 'model_id', None),
                                                  enrolled=self.gallery is not None)))

    @staticmethod
    def _run(track, first, last, window_first, window_last):
        """The longest run of consecutive slots of this track that overlaps [first, last), inside the window."""
        best = None
        slot = window_first
        while slot < window_last:
            if slot not in track.crops:
                slot += 1
                continue
            begin = slot
            while slot < window_last and slot in track.crops:
                slot += 1
            if begin < last and slot > first and (best is None or slot - begin > best[1] - best[0]):
                best = (begin, slot)
        if best is None:
            return None
        begin, stop = best
        if stop - begin > MAX_FRAMES:
            begin = stop - MAX_FRAMES
        return begin, stop

    async def finish(self):
        return await self._ready(final=True)

    def status(self):
        def median(values):
            return round(1000 * sorted(values)[len(values) // 2], 2) if values else None
        return {f'{name}_ms_median': median(values) for name, values in self.timings.items()}


# ---------------------------------------------------------------- worker clients (this process)

class RemoteFacePoints:
    """The ``vision.face`` worker (MediaPipe Face Landmarker, VIDEO mode) as private face points."""

    def __init__(self, worker, sha256):
        self.worker, self.model_id, self._sequence = worker, f'mediapipe-face:{sha256[:16]}', 0

    @classmethod
    async def open(cls, python, config, *, startup_timeout=300, inference_timeout=60):
        from ..process_provider import ProcessProvider
        config = dict(path=config['path'], sha256=config['sha256'], max_entities=int(config.get('max_entities', 4)),
                      allow_skips=True)
        worker = await ProcessProvider.open(python, 'mediapipe.face.video', config, startup_timeout=startup_timeout,
                                            inference_timeout=inference_timeout, private=True)
        return cls(worker, config['sha256'])

    async def points(self, image, timestamp_ms):
        import numpy as np
        from ..capabilities import SensorMetadata
        from ..plugins import InputFrame
        self._sequence += 1
        frame = InputFrame(np.ascontiguousarray(image), 'video', int(timestamp_ms), 'local-user', None,
                           SensorMetadata(source_id='gamst.model', sensor_id='faces', sequence=self._sequence,
                                          coordinate_frame='image'))
        _, private = await self.worker.query(frame)
        return private.take('joints')

    async def close(self):
        await self.worker.close()
