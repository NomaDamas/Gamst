"""Optional Moonshine v2 Tiny Streaming English ASR.

Implemented adapter; real-weight acceptance pending. The default ASR providers
remain unchanged. Upstream source is pinned to
https://github.com/moonshine-ai/moonshine-v2/tree/49bc3af5bb0d767d5156fb579fa5f9853b559bf3
and its README declares MIT for the code and English models. No model download,
microphone capture, or Moonshine import occurs until ``from_local`` is called.
"""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import re

from ..capabilities import SensorMetadata, identifier
from ..transcripts import transcript_capability

MOONSHINE_SOURCE_REVISION = '49bc3af5bb0d767d5156fb579fa5f9853b559bf3'
MOONSHINE_RUNTIME_VERSION = '0.0.45'
MOONSHINE_MODEL_ID = 'tiny-streaming-en'
MOONSHINE_LICENSE = 'MIT'
MODEL_FILES = frozenset(('adapter.ort', 'cross_kv.ort', 'decoder_kv.ort',
    'encoder.ort', 'frontend.ort', 'streaming_config.json', 'tokenizer.bin'))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class _MoonshineRuntime:
    """Recreate the native session to discard audio without finalizing it."""
    def __init__(self, factory):
        self.factory = factory
        self.transcriber = None
        self.closed = False
        self.reset()

    def reset(self):
        self.close()
        self.closed = False
        self.transcriber = self.factory()

    def start(self):
        self.transcriber.start()

    def add_audio(self, values):
        self.transcriber.add_audio(values, 16000)

    def update_transcription(self):
        return self.transcriber.update_transcription()

    def stop(self):
        return self.transcriber.stop()

    def close(self):
        if not self.closed:
            self.closed = True
            if self.transcriber is not None:
                self.transcriber.close()
            self.transcriber = None


class MoonshineStreamingASR:
    """Translate Moonshine line revisions into GAMST full-replacement updates.

    The injected runtime must implement start/reset/add_audio/
    update_transcription/stop/close. Tests use a lifecycle double, never a
    recognition substitute. Native line timestamps remain relative to the
    current session and are joined to the first capture timestamp here.
    """
    def __init__(self, runtime, *, model_version, hop_ms=20, update_ms=100,
                 max_segment_ms=30000):
        identifier(model_version)
        if type(hop_ms) is not int or not 10 <= hop_ms <= 100:
            raise ValueError('hop_ms must be 10..100')
        if type(update_ms) is not int or not hop_ms <= update_ms <= 1000 or update_ms % hop_ms:
            raise ValueError('update_ms must be hop_ms..1000 and divisible by hop_ms')
        if type(max_segment_ms) is not int or not update_ms <= max_segment_ms <= 30000 or max_segment_ms % hop_ms:
            raise ValueError('segment limit must be update_ms..30000 and divisible by hop_ms')
        if any(not callable(getattr(runtime, name, None)) for name in
               ('start', 'reset', 'add_audio', 'update_transcription', 'stop', 'close')):
            raise ValueError('an explicitly loaded Moonshine streaming runtime is required')
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install numpy in the selected Moonshine runtime') from exc
        self.np, self.runtime = np, runtime
        self.hop_ms, self.update_ms, self.max_segment_ms = hop_ms, update_ms, max_segment_ms
        self.spec = self.capability(model_version, update_ms=update_ms)
        self.closed = False
        self.owner = self.last_sequence = self.last_end = self.expected_end = None
        self.pending = deque()
        self.session_generation = 0
        self._clear_session()
        self.runtime.start()

    @staticmethod
    def artifact_version(files):
        if type(files) is not dict or set(files) != MODEL_FILES:
            raise ValueError('declare SHA256 for exactly the seven Tiny Streaming model files')
        for digest in files.values():
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('Moonshine file hashes must be lowercase SHA256')
        combined = hashlib.sha256(json.dumps(files, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        return (MOONSHINE_MODEL_ID+'.'+MOONSHINE_RUNTIME_VERSION+'.'+
                MOONSHINE_SOURCE_REVISION[:12]+'.sha256.'+combined)

    @classmethod
    def verify_local(cls, path, files):
        version = cls.artifact_version(files)
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError('supply an existing local Tiny Streaming English model directory')
        total = 0
        for name, expected in files.items():
            artifact = root/name
            if not artifact.is_file():
                raise ValueError('missing Moonshine model component: '+name)
            total += artifact.stat().st_size
            if artifact.stat().st_size == 0 or total > 512*1024*1024:
                raise ValueError('Moonshine model bundle must be nonempty and at most 512 MiB')
            if _sha256(artifact) != expected:
                raise ValueError('Moonshine model component SHA256 mismatch: '+name)
        return root, version

    @classmethod
    def from_local(cls, path, files, **kwargs):
        """Load only an explicitly hashed local bundle; never call model download APIs."""
        root, version = cls.verify_local(path, files)
        import importlib.metadata
        try:
            installed = importlib.metadata.version('moonshine-voice')
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('install moonshine-voice=='+MOONSHINE_RUNTIME_VERSION+
                               ' in the explicitly selected Moonshine runtime') from exc
        if installed != MOONSHINE_RUNTIME_VERSION:
            raise RuntimeError('this source-audited adapter requires moonshine-voice=='+
                               MOONSHINE_RUNTIME_VERSION)
        from moonshine_voice import ModelArch, Transcriber
        max_segment_ms = kwargs.get('max_segment_ms', 30000)
        def factory():
            # Disable Stream.add_audio's implicit duplicate polling. This adapter
            # polls on capture timestamps and therefore has deterministic updates.
            return Transcriber(model_path=str(root), model_arch=ModelArch.TINY_STREAMING,
                               update_interval=max_segment_ms/1000+1)
        runtime = _MoonshineRuntime(factory)
        try:
            return cls(runtime, model_version=version, **kwargs)
        except BaseException:
            runtime.close()
            raise

    @staticmethod
    def capability(model_version, *, update_ms=100):
        if type(update_ms) is not int or not 10 <= update_ms <= 1000:
            raise ValueError('update_ms must be 10..1000')
        return transcript_capability('audio.moonshine_v2.streaming', 'moonshine_voice.native',
            model_version, streaming_mode='native', window_ms=update_ms, license=MOONSHINE_LICENSE)

    def _clear_session(self):
        self.session_generation += 1
        self.session_start = None
        self.elapsed_ms = 0
        self.lines = {}

    def reset(self):
        if self.closed:
            return
        self.owner = self.last_sequence = self.last_end = self.expected_end = None
        self.pending.clear()
        self._clear_session()
        self.runtime.reset()
        self.runtime.start()

    def close(self):
        if not self.closed:
            self.closed = True
            self.pending.clear()
            self.runtime.close()

    def _ingest(self, transcript, finalization_reason):
        lines = getattr(transcript, 'lines', None)
        if not isinstance(lines, (list, tuple)) or len(lines) > 256:
            raise ValueError('Moonshine must return a bounded transcript line list')
        for line in lines:
            line_id = getattr(line, 'line_id', None)
            text = getattr(line, 'text', None)
            complete = getattr(line, 'is_complete', None)
            start = getattr(line, 'start_time', None)
            duration = getattr(line, 'duration', None)
            if type(line_id) is not int or not 0 <= line_id <= 2**64-1:
                raise ValueError('invalid Moonshine line identifier')
            if type(text) is not str or len(text) > 4096 or type(complete) is not bool:
                raise ValueError('Moonshine output exceeds transcript bounds or has invalid types')
            if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in (start, duration)):
                raise ValueError('Moonshine line timestamps must be finite and nonnegative')
            start_ms = self.session_start+round(start*1000)
            end_ms = self.session_start+round((start+duration)*1000)
            if start_ms < self.session_start or end_ms < start_ms or end_ms > self.session_start+self.elapsed_ms+1:
                raise ValueError('Moonshine line timestamps exceed captured audio')
            state = self.lines.get(line_id)
            if state is None:
                if len(self.lines) >= 256:
                    raise ValueError('too many Moonshine lines in one session')
                state = self.lines[line_id] = dict(revision=-1, text='', final=False)
            if state['final']:
                if text != state['text'] or not complete:
                    raise ValueError('Moonshine mutated a completed transcript line')
                continue
            if not complete and not text and state['revision'] < 0:
                state['text'] = text
                continue
            if not complete and text == state['text']:
                continue
            state['revision'] += 1
            state['text'] = text
            state['final'] = complete
            result = dict(utterance_id='moonshine.'+str(self.session_generation)+'.'+str(line_id), revision=state['revision'],
                text=text, status='final' if complete else 'partial', start_ms=start_ms,
                end_ms=end_ms, language='en')
            if complete:
                result['finalization_reason'] = finalization_reason
            self.pending.append(result)
            if len(self.pending) > 64:
                raise ValueError('Moonshine transcript update queue exceeded 64 entries')

    def _finish_segment(self, reason):
        if self.closed:
            raise RuntimeError('ASR provider is closed')
        if self.session_start is None:
            return self.pending.popleft() if self.pending else None
        try:
            transcript = self.runtime.stop()
            self._ingest(transcript, reason)
            self.runtime.reset()
            self.runtime.start()
            self._clear_session()
            return self.pending.popleft() if self.pending else None
        except Exception:
            self.reset()
            raise

    def endpoint(self):
        return self._finish_segment('manual_endpoint')

    def finish(self):
        return self._finish_segment('input_end')

    def process(self, frame, *, endpoint=None):
        if self.closed:
            raise RuntimeError('ASR provider is closed')
        try:
            if endpoint not in (None, 'manual_endpoint', 'vad_endpoint'):
                raise ValueError('endpoint must be manual_endpoint, vad_endpoint or None')
            np, sensor, data = self.np, frame.sensor, frame.data
            if frame.modality != 'audio' or type(frame.sample_rate) is not int or frame.sample_rate != 16000:
                raise ValueError('Moonshine streaming requires 16 kHz mono PCM')
            if not isinstance(sensor, SensorMetadata) or len(sensor.channels) != 1:
                raise ValueError('declare exactly one mono sensor channel')
            if type(frame.timestamp_ms) is not int or frame.timestamp_ms < self.hop_ms:
                raise ValueError('capture end must be integer milliseconds covering the hop')
            if not isinstance(data, np.ndarray) or data.shape != (16*self.hop_ms,) or data.dtype not in (np.dtype('float32'), np.dtype('float64'), np.dtype('int16')):
                raise ValueError('supply exactly one mono hop as normalized float or int16 array')
            values = data.astype(np.float32)
            if data.dtype == np.int16:
                values /= 32768.
            if not np.isfinite(values).all() or np.abs(values).max() > 1:
                raise ValueError('PCM must be finite and normalized')
            owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch, sensor.channels)
            if self.owner is not None and (owner != self.owner or sensor.sequence != self.last_sequence+1 or
                frame.timestamp_ms <= self.last_end or frame.timestamp_ms != self.expected_end+self.hop_ms):
                raise ValueError('ASR continuity lost; reset and interrupt the prior partial before retrying')
            self.owner, self.last_sequence, self.last_end = owner, sensor.sequence, frame.timestamp_ms
            self.expected_end = frame.timestamp_ms if self.expected_end is None else self.expected_end+self.hop_ms
            if self.session_start is None:
                self.session_start = frame.timestamp_ms-self.hop_ms
            self.elapsed_ms += self.hop_ms
            self.runtime.add_audio(values)
            if self.elapsed_ms % self.update_ms == 0 and self.elapsed_ms != self.max_segment_ms:
                self._ingest(self.runtime.update_transcription(), 'vad_endpoint')
            if self.elapsed_ms == self.max_segment_ms:
                return self._finish_segment('segment_limit')
            if endpoint is not None:
                return self._finish_segment(endpoint)
            return self.pending.popleft() if self.pending else None
        except Exception:
            self.reset()
            raise
