"""Bounded same-clock face/PCM windows for the TalkNet worker boundary.

This module aligns media; it does not infer speaking, identity or clock sync.
"""
from dataclasses import asdict
import json
import struct
import threading
import uuid

from .capabilities import SensorMetadata, bounded_json, identifier
from .plugins import InputFrame

_MAGIC = b'GTAV001\n'
_MAX_HEADER = 8192
_MAX_PACKET = 2_100_000
_KEYS = {'version', 'clock_id', 'session_id', 'track_id', 'subject_id', 'start_ms',
         'video_end_ms', 'audio_end_ms', 'frames', 'audio_sensor', 'video_sensor',
         'audio_sequence', 'video_sequence', 'synchronization_verified', 'identity_verified'}


def _sensor(sensor):
    value = asdict(sensor)
    value.pop('sequence')
    value.pop('capture_sequence')
    value['channels'] = list(value['channels'])
    value['sequence_kind'] = 'capture' if sensor.capture_sequence is not None else 'sensor'
    return value


def _metadata(value):
    value = bounded_json(value, max_bytes=_MAX_HEADER)
    if type(value) is not dict or set(value) != _KEYS or type(value['version']) is not int or value['version'] != 1:
        raise ValueError('invalid audiovisual packet metadata')
    for name in ('clock_id', 'session_id', 'track_id', 'subject_id'):
        identifier(value[name], name)
    n, start, end, audio_end = (value[k] for k in ('frames', 'start_ms', 'video_end_ms', 'audio_end_ms'))
    if (type(n) is not int or n not in (25, 50, 75, 100, 125, 150) or
            any(type(x) is not int or not 0 <= x <= 2**53-1 for x in (start, end, audio_end)) or
            end != start+n*40 or audio_end != end+15):
        raise ValueError('invalid audiovisual capture interval')
    if value['synchronization_verified'] is not False or value['identity_verified'] is not False:
        raise ValueError('alignment declarations do not verify sync or identity')
    for lane in ('audio', 'video'):
        source = value[lane+'_sensor']
        if type(source) is not dict or source.get('sequence_kind') not in ('capture', 'sensor'):
            raise ValueError('invalid audiovisual source provenance')
        source = dict(source)
        source.pop('sequence_kind')
        if type(source.get('channels')) is not list:
            raise ValueError('invalid audiovisual channels')
        source['channels'] = tuple(source['channels'])
        try:
            sensor = SensorMetadata(sequence=0, **source)
        except (TypeError, ValueError) as exc:
            raise ValueError('invalid audiovisual source provenance') from exc
        if lane == 'audio' and len(sensor.channels) != 1:
            raise ValueError('mono audio provenance required')
        pair = value[lane+'_sequence']
        if (type(pair) is not list or len(pair) != 2 or
                any(type(v) is not int or not 0 <= v <= 2**63-1 for v in pair) or pair[1] < pair[0]):
            raise ValueError('invalid audiovisual sequence range')
        if lane == 'video' and pair[1]-pair[0] != n-1:
            raise ValueError('video sequence range must cover every face frame')
    return value


def _encode(metadata, pcm, faces):
    header = json.dumps(_metadata(metadata), separators=(',', ':'), ensure_ascii=True).encode()
    packet = _MAGIC+struct.pack('!I', len(header))+header+pcm+faces
    if len(packet) > _MAX_PACKET:
        raise ValueError('audiovisual packet exceeds bound')
    return packet


def decode_talknet_window(packet):
    """Validate a local worker packet; return metadata and owned PCM/face arrays."""
    if type(packet) is not bytes or not 12 <= len(packet) <= _MAX_PACKET or packet[:8] != _MAGIC:
        raise ValueError('invalid audiovisual packet')
    size = struct.unpack('!I', packet[8:12])[0]
    if not 1 <= size <= _MAX_HEADER or len(packet) < 12+size:
        raise ValueError('invalid audiovisual header length')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate audiovisual metadata key')
            result[key] = value
        return result
    try:
        metadata = _metadata(json.loads(packet[12:12+size], object_pairs_hook=pairs))
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError('invalid audiovisual JSON') from exc
    n = metadata['frames']
    samples = n*640+240
    offset = 12+size
    if len(packet) != offset+samples*2+n*112*112:
        raise ValueError('audiovisual payload length mismatch')
    import numpy as np
    pcm = np.frombuffer(packet, '<i2', count=samples, offset=offset).copy()
    faces = np.frombuffer(packet, np.uint8, offset=offset+samples*2).reshape(n, 112, 112).copy()
    return metadata, pcm, faces


class TalkNetWindows:
    """One face track and two declared same-clock lanes; no resampling or padding.

    Face timestamps are interval ends at25fps; audio is capture-end mono16kPCM16.
    push methods return zero or more immutable InputFrame byte packets.
    """
    def __init__(self, *, clock_id, source_id='audiovisual', sensor_id='face-audio',
                 subject_id='local-user', window_frames=25, hop_frames=5):
        for item in (clock_id, source_id, sensor_id, subject_id):
            identifier(item)
        if type(window_frames) is not int or window_frames not in (25, 50, 75, 100, 125, 150):
            raise ValueError('window must be1..6whole seconds at25fps')
        if type(hop_frames) is not int or not 1 <= hop_frames <= window_frames:
            raise ValueError('hop must be1..window_frames')
        import numpy as np
        self.np = np
        self.clock_id, self.source_id, self.sensor_id, self.subject_id = clock_id, source_id, sensor_id, subject_id
        self.window_frames, self.hop_frames = window_frames, hop_frames
        self._lock = threading.RLock()
        self.closed = False
        self._emitted = self._resets = self._prefix_drops = 0
        self._reason = 'initial'
        self._clear()

    def _clear(self):
        self._audio, self._video = {}, {}
        self._owners, self._last = {}, {}
        self._track = self._next = None
        self._session = uuid.uuid4().hex
        self._sequence = 0

    def _reset(self, reason):
        self._clear()
        self._resets += 1
        self._reason = reason

    def reset(self):
        with self._lock:
            self._reset('explicit')

    def close(self):
        with self._lock:
            self._clear()
            self.closed = True

    def status(self):
        with self._lock:
            return dict(closed=self.closed, session_id=self._session, emitted=self._emitted,
                        resets=self._resets, last_reset_reason=self._reason,
                        unsupported_prefix_face_frames=self._prefix_drops,
                        buffered_audio_ms=len(self._audio)*10, buffered_face_frames=len(self._video),
                        synchronization_verified=False)

    def _frame(self, frame, lane, clock_id, duration):
        if self.closed:
            raise RuntimeError('audiovisual assembler is closed')
        if (clock_id != self.clock_id or not isinstance(frame, InputFrame) or
                frame.modality != lane or frame.subject_id != self.subject_id or
                not isinstance(frame.sensor, SensorMetadata) or
                type(frame.timestamp_ms) is not int or not duration <= frame.timestamp_ms <= 2**53-1):
            raise ValueError('same-clock capture-end frame/subject metadata required')
        sensor = frame.sensor
        owner = _sensor(sensor)
        seq = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        previous = self._last.get(lane)
        if lane in self._owners and owner != self._owners[lane]:
            self._reset('source_changed')
        elif previous:
            if frame.timestamp_ms <= previous[0] or seq <= previous[1]:
                raise ValueError('audiovisual capture order must increase')
            if frame.timestamp_ms-duration != previous[0] or seq != previous[1]+1:
                self._reset('capture_gap')
        self._owners[lane] = owner
        self._last[lane] = (frame.timestamp_ms, seq)
        return seq

    def push_audio(self, frame, *, clock_id):
        with self._lock:
            try:
                if not isinstance(frame, InputFrame) or frame.sample_rate != 16000 or type(frame.sample_rate) is not int:
                    raise ValueError('16kHz PCM16 frame required')
                data = frame.data
                if type(data) is bytes:
                    if not 320 <= len(data) <= 32000 or len(data) % 320:
                        raise ValueError('audio must be10..1000ms in whole10ms blocks')
                    data = self.np.frombuffer(data, '<i2').copy()
                elif (type(data) is not self.np.ndarray or data.dtype != self.np.dtype('int16') or
                      data.ndim != 1 or not 160 <= len(data) <= 16000 or len(data) % 160):
                    raise ValueError('mono int16 audio must be10..1000ms in whole10ms blocks')
                else:
                    data = data.copy()
                if not isinstance(frame.sensor, SensorMetadata) or len(frame.sensor.channels) != 1:
                    raise ValueError('one explicit mono channel required')
                duration = len(data)//16
                seq = self._frame(frame, 'audio', clock_id, duration)
                if len(self._audio)+duration//10 > self.window_frames*4+100:
                    self._reset('buffer_overflow')
                    seq = self._frame(frame, 'audio', clock_id, duration)
                start = frame.timestamp_ms-duration
                for index in range(duration//10):
                    self._audio[start+index*10] = (data[index*160:(index+1)*160].copy(), seq)
                return self._drain()
            except Exception:
                self._reset('invalid_audio')
                raise

    def push_face(self, frame, *, track_id, clock_id):
        with self._lock:
            try:
                identifier(track_id, 'track_id')
                if self._track is not None and track_id != self._track:
                    self._reset('track_changed')
                seq = self._frame(frame, 'video', clock_id, 40)
                if frame.data is None:
                    self._reset('face_absent')
                    return []
                data = frame.data
                if type(data) is not self.np.ndarray or data.dtype != self.np.uint8 or data.shape != (112, 112):
                    raise ValueError('explicit112x112grayscale uint8 face crop required')
                start = frame.timestamp_ms-40
                if self._next is not None and start < self._next:
                    self._prefix_drops += 1
                    return []
                if len(self._video)+1 > self.window_frames+25:
                    self._reset('buffer_overflow')
                    seq = self._frame(frame, 'video', clock_id, 40)
                self._track = track_id
                start = frame.timestamp_ms-40
                self._video[start] = (data.copy(), seq)
                if self._next is None:
                    self._next = start
                return self._drain()
            except Exception:
                self._reset('invalid_video')
                raise

    def _drain(self):
        if not self._audio or self._next is None:
            return []
        if (min(self._audio)-self._next) % 10:
            raise ValueError('audio/video clock grids do not align; calibrate upstream')
        # Earlier audio cannot arrive in an ordered lane. Discard unsupported
        # prefix face frames instead of manufacturing silence or shifting audio.
        if self._next < min(self._audio):
            steps = (min(self._audio)-self._next+39)//40
            self._next += steps*40
            self._prefix_drops += sum(k < self._next for k in self._video)
            self._video = {k: v for k, v in self._video.items() if k >= self._next}
        # Neither lane needs media before the next possible window. Keeping an
        # unsupported audio prefix could trigger overflow just before a match.
        self._audio = {k: v for k, v in self._audio.items() if k >= self._next}
        output = []
        n = self.window_frames
        while True:
            start = self._next
            video_keys = range(start, start+n*40, 40)
            audio_keys = range(start, start+n*40+20, 10)
            if not all(k in self._video for k in video_keys) or not all(k in self._audio for k in audio_keys):
                break
            video = [self._video[k] for k in video_keys]
            audio = [self._audio[k] for k in audio_keys]
            pcm = self.np.concatenate([v[0] for v in audio])[:n*640+240].astype('<i2', copy=False)
            faces = self.np.stack([v[0] for v in video])
            meta = dict(version=1, clock_id=self.clock_id, session_id=self._session,
                track_id=self._track, subject_id=self.subject_id, start_ms=start,
                video_end_ms=start+n*40, audio_end_ms=start+n*40+15, frames=n,
                audio_sensor=self._owners['audio'], video_sensor=self._owners['video'],
                audio_sequence=[audio[0][1], audio[-1][1]], video_sequence=[video[0][1], video[-1][1]],
                synchronization_verified=False, identity_verified=False)
            packet = _encode(meta, pcm.tobytes(), faces.tobytes())
            sensor = SensorMetadata(self.source_id, self.sensor_id, self._sequence,
                                    channels=('mono',), stream_epoch=self._session)
            output.append(InputFrame(packet, 'video', meta['audio_end_ms'], self.subject_id, 16000, sensor))
            self._sequence += 1
            self._emitted += 1
            self._next += self.hop_frames*40
            self._audio = {k: v for k, v in self._audio.items() if k >= self._next}
            self._video = {k: v for k, v in self._video.items() if k >= self._next}
        return output
