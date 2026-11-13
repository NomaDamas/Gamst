"""Bounded adapter for explicitly loaded Qwen3ASRModel vLLM streaming engines.

No model loader or installation runs here. Actual inference validation is separate.
"""
import uuid
from ..capabilities import SensorMetadata, identifier
from ..transcripts import transcript_capability


_LANGUAGES = {'English': 'en', 'Korean': 'ko', 'Chinese': 'zh', 'Japanese': 'ja',
              'German': 'de', 'French': 'fr', 'Spanish': 'es'}


class QwenStreamingASR:
    def __init__(self, engine, *, model_version, hop_ms=20, chunk_ms=2000,
                 max_segment_ms=30000, language=None):
        identifier(model_version)
        if type(hop_ms) is not int or not 10 <= hop_ms <= 100:
            raise ValueError('hop_ms must be 10..100')
        if type(chunk_ms) is not int or not 100 <= chunk_ms <= 2000 or chunk_ms % hop_ms:
            raise ValueError('chunk_ms must be 100..2000 and divisible by hop_ms')
        if type(max_segment_ms) is not int or not chunk_ms <= max_segment_ms <= 30000 or max_segment_ms % hop_ms:
            raise ValueError('segment limit must be chunk_ms..30000 and divisible by hop_ms')
        if language is not None and language not in _LANGUAGES:
            raise ValueError('select a supported explicit language name or None')
        if getattr(engine, 'backend', None) != 'vllm' or any(not callable(getattr(engine, name, None))
            for name in ('init_streaming_state', 'streaming_transcribe', 'finish_streaming_transcribe')):
            raise ValueError('an explicitly loaded Qwen3ASRModel vLLM engine is required')
        max_tokens = getattr(getattr(engine, 'sampling_params', None), 'max_tokens', None)
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise ValueError('engine sampling_params.max_tokens must be 1..512')
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install numpy in the selected ASR runtime') from exc
        self.np, self.engine, self.language = np, engine, language
        self.hop_ms, self.chunk_ms, self.max_segment_ms = hop_ms, chunk_ms, max_segment_ms
        self.spec = self.capability(model_version, chunk_ms=chunk_ms)
        self.closed = False
        self.reset()

    @staticmethod
    def capability(model_version, *, chunk_ms=2000):
        if type(chunk_ms) is not int or not 100 <= chunk_ms <= 2000:
            raise ValueError('chunk_ms must be 100..2000')
        return transcript_capability('audio.qwen3_asr.streaming', 'qwen3_asr.vllm', model_version,
            streaming_mode='native', window_ms=chunk_ms, license='Apache-2.0')

    def reset(self):
        self.state = None
        self.owner = self.last_sequence = self.last_end = self.expected_end = None
        self._clear_segment()

    def _clear_segment(self):
        self.state = None
        self.start_ms = None
        self.elapsed_ms = 0
        self.revision = -1
        self.last_text = self.last_language = None
        self.utterance_id = None

    def close(self):
        self.reset()
        self.closed = True
        self.engine = None  # Engine disposal remains with its explicit runtime owner.

    def _update(self, final):
        text, language = self.state.text, self.state.language
        if type(text) is not str or len(text) > 4096 or type(language) is not str or len(language) > 64:
            raise ValueError('ASR output exceeds transcript bounds or has invalid types')
        language = _LANGUAGES.get(language, 'und')
        if not final and (not text or (text, language) == (self.last_text, self.last_language)):
            return None
        self.revision += 1
        self.last_text, self.last_language = text, language
        decoded_ms = self.elapsed_ms if final else self.elapsed_ms//self.chunk_ms*self.chunk_ms
        return dict(utterance_id=self.utterance_id, revision=self.revision, text=text,
            status='final' if final else 'partial', start_ms=self.start_ms,
            end_ms=self.start_ms+decoded_ms, language=language)

    def process(self, frame):
        if self.closed: raise RuntimeError('ASR provider is closed')
        try:
            np, sensor, data = self.np, frame.sensor, frame.data
            if frame.modality != 'audio' or type(frame.sample_rate) is not int or frame.sample_rate != 16000:
                raise ValueError('Qwen streaming requires 16 kHz mono PCM')
            if not isinstance(sensor, SensorMetadata) or len(sensor.channels) != 1:
                raise ValueError('declare exactly one mono sensor channel')
            if type(frame.timestamp_ms) is not int or frame.timestamp_ms < self.hop_ms:
                raise ValueError('capture end must be integer milliseconds covering the hop')
            if not isinstance(data, np.ndarray) or data.shape != (16*self.hop_ms,) or data.dtype not in (np.dtype('float32'), np.dtype('float64'), np.dtype('int16')):
                raise ValueError('supply exactly one mono hop as normalized float or int16 array')
            values = data.astype(np.float32)
            if data.dtype == np.int16: values /= 32768.
            if not np.isfinite(values).all() or np.abs(values).max() > 1:
                raise ValueError('PCM must be finite and normalized')
            owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch, sensor.channels)
            if self.owner is not None and (owner != self.owner or sensor.sequence != self.last_sequence+1 or
                frame.timestamp_ms <= self.last_end or frame.timestamp_ms != self.expected_end+self.hop_ms):
                raise ValueError('ASR continuity lost; reset and interrupt the prior partial before retrying')
            self.owner, self.last_sequence, self.last_end = owner, sensor.sequence, frame.timestamp_ms
            self.expected_end = frame.timestamp_ms if self.expected_end is None else self.expected_end+self.hop_ms
            if self.state is None:
                self.state = self.engine.init_streaming_state(language=self.language,
                    unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=self.chunk_ms/1000)
                self.start_ms = frame.timestamp_ms-self.hop_ms
                self.utterance_id = uuid.uuid4().hex
            self.elapsed_ms += self.hop_ms
            self.engine.streaming_transcribe(values, self.state)
            if self.elapsed_ms == self.max_segment_ms:
                return self.finish()
            return self._update(False)
        except Exception:
            self.reset()
            raise

    def finish(self):
        """Flush once without new audio; caller must publish with a new output sequence."""
        if self.closed: raise RuntimeError('ASR provider is closed')
        if self.state is None: return None
        try:
            self.engine.finish_streaming_transcribe(self.state)
            result = self._update(True)
            self._clear_segment()
            return result
        except Exception:
            self.reset()
            raise
