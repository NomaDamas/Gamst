"""``audio.diarization`` candidates whose slots come from speaker turns, not a local frame grid.

* :class:`HostedLiveDiarization`: pyannoteAI **Live-1** streaming diarization (hosted, WebSocket, up to 8
  speakers). **Live**: 100 ms blocks go out as they are captured; the slots come back as speaker start/end
  events, and a chunk is committed up to the server's ``last_predicted_timestamp`` (no earlier event follows).
* :class:`OfflineDiarization`: a whole-recording pipeline (pyannoteAI Precision-3 / Community-1 hosted,
  DiariZen, pyannote community-1 local). **Not live**: it keeps the session's sound and emits one record when the
  session finishes. It is a reference to compare the live candidates against, never the live default.

Both emit exactly the record every ``audio.diarization`` candidate emits (:func:`.diarization.slot_record`):
speaker labels become slots 0-7 in the order they first speak. A turn has no frame scores, so a slot's
``mean_probability`` is the share of the chunk it is active and ``max_probability`` is 1 or 0.

Hosted: the audio leaves the device and each stream or job is billed; nothing runs without ``allow_remote: true``.
In process, a cost line is printed before each paid call (:class:`.pyannoteai.PyannoteAI`); a worker's output is
closed once it starts, so the recipe says the price when it opens a hosted candidate (``recipes.parts._role``).
The key is read from the environment only.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import uuid

from ..capabilities import CapabilitySpec, OutputField
from .diarization import SLOTS, SlotStream, slot_record

CHUNK = 1600                # Live-1 takes exactly 100 ms of 16 kHz float32 per binary message
HOSTED = 'hosted-unversioned'


def _spec(name, version, *, live, license):
    return CapabilitySpec(f'audio.{name}.diarization', 'audio.speaker_activity', 'audio', f'{name}.diarization',
                          '1.' + hashlib.sha256(json.dumps(version).encode()).hexdigest(),
                          (OutputField('diarization', 'object'),), 'native',
                          window_ms=100 if live else 0, freshness_ms=5000 if live else 3_600_000,
                          stateful=True, reset_on_gap=True, interpretation='classification', devices=('cpu',),
                          sensor_requirements=('16000hz-mono-pcm', 'whole-10ms-capture-blocks'), license=license)


class _Turns:
    """Speaker labels -> slots, and turns (capture ms) cut into chunks as they become final."""

    def __init__(self):
        self.slots, self.open, self.closed, self.dropped = {}, {}, [], 0

    def slot(self, speaker):
        if speaker not in self.slots:
            if len(self.slots) >= SLOTS:
                self.dropped += 1
                return None
            self.slots[speaker] = len(self.slots)
        return self.slots[speaker]

    def start(self, speaker, at):
        slot = self.slot(speaker)
        if slot is not None and slot not in self.open:
            self.open[slot] = at

    def end(self, speaker, at):
        slot = self.slot(speaker)
        if slot is not None and slot in self.open:
            begin = self.open.pop(slot)
            if at > begin:
                self.closed.append((slot, begin, at))

    def turn(self, speaker, begin, end):
        slot = self.slot(speaker)
        if slot is not None and end > begin:
            self.closed.append((slot, begin, end))

    def chunk(self, start, end):
        """Activity over [start, end) in 10 ms steps; turns that end before ``end`` are dropped afterwards."""
        start, end = start // 10 * 10, end // 10 * 10
        if end <= start:
            return None
        spans = {slot: [] for slot in range(SLOTS)}
        for slot, begin, stop in self.closed:
            begin, stop = max(begin, start) // 10 * 10, min(stop, end) // 10 * 10
            if stop > begin:
                spans[slot].append([begin, stop])
        for slot, begin in self.open.items():
            begin = max(begin, start) // 10 * 10
            if end > begin:
                spans[slot].append([begin, end])
        self.closed = [item for item in self.closed if item[2] > end]
        activity = []
        for slot in range(SLOTS):
            merged = []
            for begin, stop in sorted(spans[slot]):
                if merged and begin <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], stop)
                else:
                    merged.append([begin, stop])
            share = sum(stop - begin for begin, stop in merged) / (end - start)
            activity.append(dict(slot=slot, intervals_ms=merged, mean_probability=round(share, 4),
                                 max_probability=1. if merged else 0.))
        return dict(start_ms=start, end_ms=end, frame_ms=10, activity=activity)


def _connect(url):
    from websockets.sync.client import connect
    return connect(url, open_timeout=30, max_size=2 ** 20)


class HostedLiveDiarization(SlotStream):
    """pyannoteAI Live-1 over one WebSocket per session. ``connect(url)`` -> object with send/recv/close (tests)."""

    def __init__(self, client, *, connect=None, heartbeat_ms=200, finish_wait_s=15.):
        if not 200 <= int(heartbeat_ms) <= 60000:
            raise ValueError('heartbeat_ms is 200..60000 (the API range)')
        self.client, self.connect = client, connect or _connect
        self.heartbeat_ms, self.finish_wait_s = int(heartbeat_ms), float(finish_wait_s)
        self.mode = 'hosted.live-1'
        self.spec = _spec('pyannoteai.live', [client.endpoint, 'live-1', heartbeat_ms], live=True,
                          license='hosted service per pyannoteAI terms; adapter Apache-2.0')
        import numpy as np
        self.np, self._socket, self._lock = np, None, threading.RLock()
        self.closed, self.streamed_s = False, 0.
        self.reset()

    # ------------------------------------------------------------------ session

    def _clear(self):
        self._hangup(wait=False)
        self._pending = self.np.empty(0, self.np.float32)
        self._owner = self._last = self._origin = None
        self._turns, self._predicted, self._emitted, self._received = _Turns(), 0., None, 0
        self._session = uuid.uuid4().hex

    def reset(self):
        with self._lock:
            self._clear()
            self._finished = False

    def close(self):
        with self._lock:
            self._clear()
            self.closed = True

    def _open(self):
        url = self.client.live_stream(heartbeat_ms=self.heartbeat_ms)
        self._socket, self._events, self._error = self.connect(url), queue.Queue(), None
        self._reader = threading.Thread(target=self._read, args=(self._socket, self._events), daemon=True)
        self._reader.start()

    def _read(self, socket, events):
        try:
            while True:
                message = socket.recv()
                if isinstance(message, str):
                    events.put(json.loads(message))
        except Exception as error:      # closed by the server (normal at the end) or the network
            events.put({'type': '_closed', 'error': None if 'OK' in type(error).__name__ else repr(error)[:200]})

    def _hangup(self, *, wait):
        socket, self._socket = getattr(self, '_socket', None), None
        if socket is None:
            return
        try:
            if wait:
                socket.send(json.dumps({'type': 'end_of_stream'}))
                self._reader.join(self.finish_wait_s)
        finally:
            socket.close()
            self.client.streamed(self.streamed_s)
            self.streamed_s = 0.

    def _send(self, values):
        self._pending = self.np.concatenate((self._pending, values))
        while len(self._pending) >= CHUNK:
            self._socket.send(self._pending[:CHUNK].astype('<f4').tobytes())
            self._pending = self._pending[CHUNK:]
            self.streamed_s += CHUNK / 16000

    def _drain(self):
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            kind, data = event.get('type'), event.get('data') or {}
            if kind == 'error':
                raise RuntimeError(f"pyannoteAI live: {str(event.get('message'))[:200]}")
            if kind == '_closed':
                if event.get('error'):
                    raise RuntimeError(f"pyannoteAI live connection ended: {event['error']}")
                self._predicted = float('inf')
                continue
            if kind == 'heartbeat':
                self._predicted = max(self._predicted, float(data.get('last_predicted_timestamp') or 0))
                continue
            at = self._origin + int(round(1000 * float(data.get('timestamp', 0))))
            if kind == 'diarization_speaker_start':
                self._turns.start(str(data.get('speaker'))[:40], at)
            elif kind == 'diarization_speaker_end':
                self._turns.end(str(data.get('speaker'))[:40], at)

    def _chunks(self, *, final):
        captured = self._origin + self._received // 16
        until = captured if final else min(captured, self._origin + int(1000 * min(self._predicted, 1e9)))
        chunk = self._turns.chunk(self._emitted, until)
        if chunk is None:
            return []
        self._emitted = chunk['end_ms']
        return [chunk]

    def _result(self, chunks, *, final):
        return slot_record(self.spec, session_id=self._session, chunks=chunks, capture_end_ms=self._last[0],
                           final=final, mode=self.mode, threshold=None, lookahead_ms=None,
                           source_revision=HOSTED, checkpoint_sha256=None, artifact_hashes_verified=False)

    # ------------------------------------------------------------------ stream

    def process(self, frame):
        with self._lock:
            if self.closed or self._finished:
                raise RuntimeError('diarization session is closed/finished; reset before reuse')
            try:
                values = self._accept(frame)
                if self._socket is None:
                    self._open()
                    self._emitted = self._origin
                self._send(values)
                self._received += len(values)
                self._drain()
                chunks = self._chunks(final=False)
                return self._result(chunks, final=False) if chunks else None
            except Exception:
                self._clear()
                raise

    def finish(self):
        with self._lock:
            if self.closed:
                raise RuntimeError('diarization provider is closed')
            if self._finished:
                return None
            try:
                if self._socket is None:
                    return None
                if len(self._pending):       # the API takes whole 100 ms messages: the last one is padded with silence
                    self._send(self.np.zeros(CHUNK - len(self._pending), self.np.float32))
                self._hangup(wait=True)
                self._drain()
                chunks = self._chunks(final=True)
                return self._result(chunks, final=True) if chunks else None
            finally:
                self._clear()
                self._finished = True


class OfflineDiarization(SlotStream):
    """A whole-session pipeline: ``backend.diarize(samples) -> [(start_s, end_s, speaker)]`` once, at ``finish``.

    Not live: ``process`` only keeps the sound (at most ``max_seconds``). The record is final at once.
    """

    def __init__(self, backend, *, name, version, license, mode='offline', max_seconds=3600):
        if not callable(getattr(backend, 'diarize', None)):
            raise ValueError('explicit offline diarization backend required')
        self.backend, self.mode, self.max_samples = backend, mode, int(max_seconds) * 16000
        self.spec = _spec(name, version, live=False, license=license)
        self.source_revision, self.checkpoint = getattr(backend, 'revision', HOSTED), getattr(backend, 'sha256', None)
        self.verified = bool(getattr(backend, 'verified', False))
        import numpy as np
        self.np, self._lock, self.closed = np, threading.RLock(), False
        self.reset()

    def _clear(self):
        self._parts, self._owner, self._last, self._origin, self._received = [], None, None, None, 0
        self._session = uuid.uuid4().hex

    def reset(self):
        with self._lock:
            self._clear()
            self._finished = False

    def close(self):
        with self._lock:
            self._clear()
            close = getattr(self.backend, 'close', None)
            if callable(close):
                close()
            self.closed = True

    def process(self, frame):
        with self._lock:
            if self.closed or self._finished:
                raise RuntimeError('diarization session is closed/finished; reset before reuse')
            values = self._accept(frame)      # a gap starts a new session: the sound before it is dropped
            if self._received + len(values) > self.max_samples:
                raise ValueError(f'an offline reference keeps at most {self.max_samples // 16000} s of sound')
            self._parts.append(values)
            self._received += len(values)
            return None

    def finish(self):
        with self._lock:
            if self.closed:
                raise RuntimeError('diarization provider is closed')
            if self._finished:
                return None
            try:
                if not self._parts:
                    return None
                samples = self.np.concatenate(self._parts)
                turns = _Turns()
                for start, end, speaker in self.backend.diarize(samples):
                    turns.turn(str(speaker)[:40], self._origin + int(round(1000 * start)),
                               self._origin + int(round(1000 * end)))
                chunk = turns.chunk(self._origin, self._origin + self._received // 16)
                return slot_record(self.spec, session_id=self._session, chunks=[chunk] if chunk else [],
                                   capture_end_ms=self._last[0], final=True, mode=self.mode, threshold=None,
                                   lookahead_ms=None, source_revision=self.source_revision,
                                   checkpoint_sha256=self.checkpoint, artifact_hashes_verified=self.verified)
            finally:
                self._clear()
                self._finished = True


class HostedBatch:
    """pyannoteAI ``/v1/diarize`` (Precision-3, Community-1 ...) as an offline backend."""

    def __init__(self, client, *, exclusive=False):
        self.client, self.exclusive = client, bool(exclusive)

    def diarize(self, samples):
        output = self.client.diarize(samples, exclusive=self.exclusive)
        segments = output.get('exclusiveDiarization' if self.exclusive else 'diarization') or ()
        return [(float(item['start']), float(item['end']), item['speaker']) for item in segments
                if isinstance(item.get('start'), (int, float)) and isinstance(item.get('end'), (int, float))]


def _require(packages, environment):
    from importlib.metadata import PackageNotFoundError, version
    for package, expected in packages:
        try:
            installed = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f'install {package}=={expected} in the {environment} environment') from error
        if installed.split('+')[0] != expected:
            raise RuntimeError(f'the {environment} environment requires {package}=={expected}')


def _verify(root, files):
    from pathlib import Path
    root = Path(root).expanduser()
    for name, (size, sha) in files.items():
        digest, count = hashlib.sha256(), 0
        with open(root / name, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 22), b''):
                digest.update(block)
                count += len(block)
        if count != size or digest.hexdigest() != sha:
            raise ValueError(f'{root / name} is not the pinned file (size/SHA256 mismatch)')
    return root


def _turns_of(annotation):
    return [(float(turn.start), float(turn.end), speaker) for turn, _, speaker in annotation.itertracks(yield_label=True)]


# DiariZen-Large-s80-v2 (BUT-FIT/diarizen-wavlm-large-s80-md-v2) and the WeSpeaker ResNet34 embedding its pipeline
# names (pyannote/wespeaker-voxceleb-resnet34-LM), laid out as one directory: the hub files + embedding/.
DIARIZEN_REVISION = '027b3e221b9d81dce2be794084f1dbd3ba2da403'
DIARIZEN_SOURCE = {'url': 'https://github.com/BUTSpeechFIT/DiariZen', 'revision': '844f5555b0a98acd0931511fc641a8c5b8ba92c7'}
DIARIZEN_PACKAGES = (('torch', '2.1.1'), ('torchaudio', '2.1.1'), ('numpy', '1.26.4'), ('transformers', '4.46.3'))
DIARIZEN_FILES = {
    'config.toml': (657, '6eb3dab9c3892efc4a271ef5f88b5453e6e96789961084a15e88ca352de98503'),
    'pytorch_model.bin': (278071526, '689a10f0cbde6345b4bf9b1670cf3196488634b34e5b07bea0fe40d508e1196b'),
    'plda/plda.npz': (133852, '9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255'),
    'plda/xvec_transform.npz': (134376, '325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f'),
    'embedding/pytorch_model.bin': (26645418, '366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56'),
}


class DiariZenBackend:
    """DiariZen (WavLM-Conformer segmentation + VBx clustering) on one whole recording. Weights CC BY-NC 4.0."""
    revision, sha256 = DIARIZEN_REVISION, DIARIZEN_FILES['pytorch_model.bin'][1]

    def __init__(self, pipeline):
        self.pipeline, self.verified = pipeline, True

    @classmethod
    def from_local(cls, path, *, allow_noncommercial=False):
        if allow_noncommercial is not True:
            raise PermissionError('DiariZen weights are CC BY-NC 4.0 (non-commercial, research use): '
                                  'set allow_noncommercial to opt in')
        _require(DIARIZEN_PACKAGES, 'diarizen')
        root = _verify(path, DIARIZEN_FILES)
        from diarizen.pipelines.inference import DiariZenPipeline
        return cls(DiariZenPipeline(diarizen_hub=root.absolute(),
                                    embedding_model=str((root / 'embedding' / 'pytorch_model.bin').absolute())))

    def diarize(self, samples):
        import io
        from .pyannoteai import wav_bytes
        return _turns_of(self.pipeline(io.BytesIO(wav_bytes(samples))))


# pyannote/speaker-diarization-community-1 (gated: the owner accepts its conditions on Hugging Face and downloads
# the snapshot); pyannote.audio 4 loads the local directory.
COMMUNITY1_REVISION = '3533c8cf8e369892e6b79ff1bf80f7b0286a54ee'
COMMUNITY1_PACKAGES = (('pyannote.audio', '4.1.1'), ('torch', '2.8.0'))
COMMUNITY1_FILES = {
    'config.yaml': (444, None),
    'segmentation/pytorch_model.bin': (5906507, '7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e'),
    'embedding/pytorch_model.bin': (26646242, '6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929'),
    'plda/plda.npz': (133852, '9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255'),
    'plda/xvec_transform.npz': (134376, '325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f'),
}


class CommunityOneBackend:
    """pyannote community-1 run locally (CC BY 4.0, gated download). Not run here: no accepted gate on this machine."""
    revision, sha256 = COMMUNITY1_REVISION, COMMUNITY1_FILES['segmentation/pytorch_model.bin'][1]

    def __init__(self, pipeline, torch):
        self.pipeline, self.torch, self.verified = pipeline, torch, True

    @classmethod
    def from_local(cls, path):
        _require(COMMUNITY1_PACKAGES, 'pyannote-community1')
        root = _verify(path, {name: item for name, item in COMMUNITY1_FILES.items() if item[1]})
        import torch
        from pyannote.audio import Pipeline
        return cls(Pipeline.from_pretrained(str(root)), torch)

    def diarize(self, samples):
        output = self.pipeline({'waveform': self.torch.from_numpy(samples)[None], 'sample_rate': 16000})
        return _turns_of(getattr(output, 'speaker_diarization', output))


def local_offline(config):
    config = dict(config)
    backend = config.pop('backend')
    if backend == 'diarizen':
        return OfflineDiarization(DiariZenBackend.from_local(config.pop('path'), **config), name='diarizen',
                                  version=[DIARIZEN_REVISION], mode='offline.diarizen-large-s80-v2',
                                  license='weights:CC-BY-NC-4.0;code:MIT;embedding:CC-BY-4.0')
    if backend == 'community-1':
        return OfflineDiarization(CommunityOneBackend.from_local(config.pop('path'), **config), name='pyannote.community1',
                                  version=[COMMUNITY1_REVISION], mode='offline.community-1',
                                  license='weights:CC-BY-4.0;pyannote.audio:MIT')
    raise ValueError('offline backend is diarizen or community-1')


def local_capability(config):
    backend = config.get('backend')
    if backend == 'diarizen':
        if config.get('allow_noncommercial') is not True:
            raise PermissionError('DiariZen weights are CC BY-NC 4.0: set allow_noncommercial to opt in')
        allowed, name, version, license = {'backend', 'path', 'allow_noncommercial'}, 'diarizen', [DIARIZEN_REVISION], \
            'weights:CC-BY-NC-4.0;code:MIT;embedding:CC-BY-4.0'
    elif backend == 'community-1':
        allowed, name, version, license = {'backend', 'path'}, 'pyannote.community1', [COMMUNITY1_REVISION], \
            'weights:CC-BY-4.0;pyannote.audio:MIT'
    else:
        raise ValueError('offline backend is diarizen or community-1')
    if set(config) - allowed or type(config.get('path')) is not str:
        raise ValueError(f'offline {backend} takes {", ".join(sorted(allowed))} (a local directory path)')
    return _spec(name, version, live=False, license=license)


def hosted_live(config):
    from .pyannoteai import PyannoteAI
    config = dict(config)
    options = {key: config.pop(key) for key in ('heartbeat_ms', 'finish_wait_s') if key in config}
    return HostedLiveDiarization(PyannoteAI(**config), **options)


def hosted_offline(config):
    from .pyannoteai import PyannoteAI
    config = dict(config)
    exclusive = config.pop('exclusive', False)
    client = PyannoteAI(**config)
    return OfflineDiarization(HostedBatch(client, exclusive=exclusive), name='pyannoteai.offline',
                              version=[client.endpoint, client.model, exclusive], mode=f'hosted.{client.model}',
                              license='hosted service per pyannoteAI terms; adapter Apache-2.0')


def hosted_capability(profile, config):
    """Discovery without a key, a network call or numpy: the same spec the worker will report."""
    known = {'endpoint', 'api_key_env', 'model', 'eur_per_hour', 'eur_per_voiceprint', 'allow_remote', 'poll_s',
             'timeout_s'} | ({'heartbeat_ms', 'finish_wait_s'} if profile == 'pyannoteai.live.diarization' else {'exclusive'})
    if set(config) - known or 'eur_per_hour' not in config:
        raise ValueError(f'{profile} takes {", ".join(sorted(known))} (eur_per_hour required for the cost line)')
    endpoint = config.get('endpoint', 'https://api.pyannote.ai').rstrip('/')
    if profile == 'pyannoteai.live.diarization':
        return _spec('pyannoteai.live', [endpoint, 'live-1', config.get('heartbeat_ms', 200)], live=True,
                     license='hosted service per pyannoteAI terms; adapter Apache-2.0')
    return _spec('pyannoteai.offline', [endpoint, config.get('model', 'precision-2'), config.get('exclusive', False)],
                 live=False, license='hosted service per pyannoteAI terms; adapter Apache-2.0')
