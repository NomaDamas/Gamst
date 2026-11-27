"""Team split without labels: cluster the players' appearance, then give every track its team.

Reads the video input and players (``players``: an ``objects`` module with track IDs). For a player box it takes the
torso (the middle of the box, where the shirt is) and describes it:

- ``embed: "color"`` (default, no model): a hue/saturation histogram and the mean value of the torso pixels;
- ``embed: "siglip"``: an image embedding from a SigLIP worker (``image.embedding``, Roboflow Sports' choice:
  SigLIP + dimension reduction + k-means). Roboflow reduces with UMAP; here the reduction is PCA to three
  dimensions (NumPy only), so the two are not identical.

During the first ``fit_ms`` of play it collects up to ``samples_per_track`` crops from each track, then k-means
(``teams`` clusters, k-means++ seeded, deterministic) fixes the teams; ``realtime`` keeps a short fit window, ``post``
a long one (docs/sports.md, realtime and post). After the fit each track votes over its first ``votes`` crops and
keeps that team. The record re-commits the players as ``objects`` with ``attributes.team`` ("A", "B" ...) so a
possession module reads teams from it. A goalkeeper or referee in other colours lands in the nearer team unless
``teams`` is 3 or more; which letter is which side is arbitrary (the LLM is told so).
"""
from __future__ import annotations

import threading
from collections import Counter, OrderedDict

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..sports.objects import boxes, object_value

LETTERS = 'ABCDEF'


def hsv(rgb):
    """uint8 RGB (..., 3) -> hue in [0, 1), saturation and value in [0, 1] (NumPy, no OpenCV)."""
    import numpy as np
    rgb = np.asarray(rgb, dtype=np.float32) / 255.
    high, low = rgb.max(-1), rgb.min(-1)
    delta = high - low
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    safe = np.where(delta == 0, 1, delta)
    hue = np.where(high == r, ((g - b) / safe) % 6, np.where(high == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6
    hue = np.where(delta == 0, 0, hue)
    saturation = np.where(high == 0, 0, delta / np.where(high == 0, 1, high))
    return hue, saturation, high


def color_feature(crop_bgr, hue_bins=12):
    """A torso crop -> hue histogram of saturated pixels, share of unsaturated pixels, mean value."""
    import numpy as np
    hue, saturation, value = hsv(np.asarray(crop_bgr)[..., ::-1])
    saturated = saturation > .25
    hist = np.histogram(hue[saturated], bins=hue_bins, range=(0, 1))[0].astype(np.float32)
    total = max(1, hue.size)
    return np.concatenate([hist / total, [1 - saturated.mean(), float(value.mean())]]).astype(np.float32)


def kmeans(points, k, *, iterations=50):
    """Deterministic k-means++ (first seed: the point farthest from the mean) -> (centroids, labels)."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    if len(points) < k:
        raise ValueError(f'{len(points)} samples cannot make {k} clusters')
    centroids = [points[np.argmax(((points - points.mean(0)) ** 2).sum(1))]]
    while len(centroids) < k:
        distance = np.min([((points - c) ** 2).sum(1) for c in centroids], axis=0)
        centroids.append(points[int(np.argmax(distance))])
    centroids = np.asarray(centroids)
    labels = np.zeros(len(points), dtype=int)
    for _ in range(iterations):
        labels = np.argmin(((points[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1)
        moved = np.asarray([points[labels == i].mean(0) if (labels == i).any() else centroids[i] for i in range(k)])
        if np.allclose(moved, centroids):
            break
        centroids = moved
    # Order clusters by their first point so "A" is stable for the same data.
    order = sorted(range(k), key=lambda i: int(np.argmax(labels == i)) if (labels == i).any() else len(labels))
    remap = {old: new for new, old in enumerate(order)}
    return centroids[order], np.asarray([remap[label] for label in labels])


def pca(points, dimensions=3):
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    mean = points.mean(0)
    _, _, vt = np.linalg.svd(points - mean, full_matrices=False)
    return mean, vt[:dimensions]


class TeamSplit:
    def __init__(self, *, players, video, player_labels=('person',), teams=2, fit_ms=5000, samples_per_track=3,
                 min_samples=12, votes=3, embed='color', embedder=None, downsample=2, capability_id='sports.teams'):
        if type(teams) is not int or not 2 <= teams <= len(LETTERS):
            raise ValueError(f'teams must be 2..{len(LETTERS)}')
        if embed not in ('color', 'siglip'):
            raise ValueError('embed is color or siglip')
        if embed == 'siglip' and embedder is None:
            raise ValueError('embed siglip needs an image.embedding worker')
        self.players, self.video = identifier(players, 'module name'), video
        self.player_labels, self.teams, self.fit_ms = tuple(player_labels), teams, int(fit_ms)
        self.samples_per_track, self.min_samples, self.votes = int(samples_per_track), int(min_samples), int(votes)
        self.embed, self.embedder, self.downsample = embed, embedder, max(1, int(downsample))
        self._lock = threading.RLock()
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'video', 'gamst.sports_team', f'1.{embed}-{teams}-{fit_ms}',
            (OutputField('objects', 'array'), OutputField('summary', 'object')),
            'frame', stateful=True, freshness_ms=1000, interpretation='estimate', devices=('cpu',), license='Apache-2.0')
        self.reset()

    def reset(self):
        self._frames = OrderedDict()     # capture ms -> downsampled BGR frame (the few the detector has not answered)
        self._samples = {}               # track -> [feature]
        self._start_ms = None
        self._model = None               # (mean, components, centroids)
        self._assigned = {}              # track -> team letter
        self._votes = {}                 # track -> Counter

    async def _feature(self, crop):
        if self.embed == 'color':
            return color_feature(crop)
        import numpy as np
        return np.asarray(await self.embedder.embed(np.ascontiguousarray(crop)), dtype=np.float32)

    def _crop(self, image, box):
        height, width = image.shape[:2]
        x0, x1 = int((box.x + .2 * box.width) * width), int((box.x + .8 * box.width) * width)
        y0, y1 = int((box.y + .15 * box.height) * height), int((box.y + .55 * box.height) * height)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        return image[max(0, y0):y1, max(0, x0):x1]

    def _fit(self):
        import numpy as np
        features = [feature for items in self._samples.values() for feature in items]
        tracks = [track for track, items in self._samples.items() for _ in items]
        mean, components = pca(features, min(3, len(features[0])))
        projected = (np.asarray(features) - mean) @ components.T
        centroids, labels = kmeans(projected, self.teams)
        self._model = (mean, components, centroids)
        for track, label in zip(tracks, labels):
            self._votes.setdefault(track, Counter())[LETTERS[label]] += 1
        for track, votes in self._votes.items():
            self._assigned[track] = votes.most_common(1)[0][0]

    def _classify(self, feature):
        import numpy as np
        mean, components, centroids = self._model
        point = (np.asarray(feature) - mean) @ components.T
        return LETTERS[int(np.argmin(((centroids - point) ** 2).sum(1)))]

    async def process(self, frame):
        import numpy as np
        source = None if frame.sensor is None else frame.sensor.sensor_id
        if isinstance(frame.data, np.ndarray):
            if frame.modality == 'video':
                self._frames[frame.timestamp_ms] = frame.data[::self.downsample, ::self.downsample]
                while len(self._frames) > 90:
                    self._frames.popitem(last=False)
            return None
        if source != self.players or not isinstance(frame.data, dict):
            return None
        image = self._frames.get(frame.timestamp_ms)
        for older in [ms for ms in self._frames if ms <= frame.timestamp_ms]:
            del self._frames[older]
        players = [box for box in boxes(frame.data, self.player_labels) if box.track_id is not None]
        if self._start_ms is None and players:
            self._start_ms = frame.timestamp_ms
        if image is not None:
            for box in players:
                fitting = self._model is None
                wanted = (len(self._samples.get(box.track_id, ())) < self.samples_per_track if fitting
                          else box.track_id not in self._assigned or
                          sum(self._votes.get(box.track_id, Counter()).values()) < self.votes)
                if not wanted:
                    continue
                crop = self._crop(image, box)
                if crop is None:
                    continue
                feature = await self._feature(crop)
                if fitting:
                    self._samples.setdefault(box.track_id, []).append(feature)
                else:
                    votes = self._votes.setdefault(box.track_id, Counter())
                    votes[self._classify(feature)] += 1
                    self._assigned[box.track_id] = votes.most_common(1)[0][0]
        with self._lock:
            count = sum(len(items) for items in self._samples.values())
            if self._model is None and self._start_ms is not None and count >= max(self.min_samples, self.teams) and \
                    frame.timestamp_ms - self._start_ms >= self.fit_ms:
                self._fit()
        labelled = [type(box)(box.track_id, box.label, box.score, box.x, box.y, box.width, box.height,
                              self._assigned.get(box.track_id)) for box in boxes(frame.data, None)]
        value = object_value(labelled, tracking='team_split')
        value['summary'].update(fitted=self._model is not None, embed=self.embed,
                                teams=dict(Counter(team for team in self._assigned.values())))
        return self.spec.validate(value)
