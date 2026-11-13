"""In-process Silero/Qwen composition; models must be explicitly loaded first."""
from dataclasses import replace
import hashlib
import json

from .qwen_asr import QwenStreamingASR
from .vad import SileroActivity


class SileroEndpointASR:
    """Preserve every ASR hop, finalize after sustained VAD inactivity.

    Takes exclusive ownership of both providers. It does not gate silence or
    start capture. This synchronous provider belongs in a SensorStreams lane.
    """
    def __init__(self, asr, vad, *, minimum_speech_ms=160, silence_ms=640):
        if not isinstance(asr, QwenStreamingASR) or not isinstance(vad, SileroActivity):
            raise ValueError('explicit QwenStreamingASR and SileroActivity required')
        if asr.closed or vad.closed or asr.owner is not None or vad.last_ts is not None:
            raise ValueError('providers must be open and unused or explicitly reset')
        if asr.hop_ms > 32:
            raise ValueError('endpoint composition requires ASR hops at most 32 ms')
        for value, maximum in ((minimum_speech_ms, 2000), (silence_ms, 5000)):
            if type(value) is not int or not 32 <= value <= maximum or value % 32:
                raise ValueError('endpoint durations must be bounded positive multiples of 32 ms')
        configuration = dict(asr_version=asr.spec.version, vad_version=vad.spec.version,
            hop_ms=asr.hop_ms, chunk_ms=asr.chunk_ms, max_segment_ms=asr.max_segment_ms,
            language=asr.language, start_threshold=vad.start_threshold, stop_threshold=vad.stop_threshold,
            minimum_speech_ms=minimum_speech_ms, silence_ms=silence_ms, policy_version=1)
        digest = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.spec = replace(asr.spec, id='audio.qwen3_asr.silero_endpoint',
            provider='qwen3_asr.silero', version='pipeline.sha256.'+digest,
            window_ms=max(32, asr.chunk_ms), license='Apache-2.0 AND MIT')
        self.asr, self.vad = asr, vad
        self.minimum_speech_ms, self.silence_ms = minimum_speech_ms, silence_ms
        self.closed = False
        self._clear_policy()

    def _clear_policy(self):
        self.active_ms = self.quiet_ms = 0
        self.armed = False

    def reset(self):
        self._clear_policy()
        try:
            self.asr.reset()
        finally:
            self.vad.reset()

    def process(self, frame, *, manual_endpoint=False):
        if self.closed:
            raise RuntimeError('endpoint provider is closed')
        try:
            if type(manual_endpoint) is not bool:
                raise ValueError('manual_endpoint must be boolean')
            # Silero's array path expects normalized floats; Qwen also accepts
            # PCM16. Normalize a bounded copy once so both see the same scale.
            np, data = self.asr.np, frame.data
            if isinstance(data, np.ndarray) and data.dtype == np.int16:
                if data.shape != (16*self.asr.hop_ms,):
                    raise ValueError('supply exactly one mono PCM16 hop')
                frame = replace(frame, data=data.astype(np.float32)/32768.)
            activity = self.vad.process(frame)
            endpoint = 'manual_endpoint' if manual_endpoint else None
            for decision in activity['decisions'] if activity else ():
                if decision['active']:
                    self.active_ms = min(self.minimum_speech_ms, self.active_ms+32)
                    self.armed = self.armed or self.active_ms >= self.minimum_speech_ms
                    self.quiet_ms = 0
                else:
                    if not self.armed:
                        self.active_ms = 0
                    self.quiet_ms = min(self.silence_ms, self.quiet_ms+32)
                    if self.armed and self.quiet_ms >= self.silence_ms and endpoint is None:
                        endpoint = 'vad_endpoint'
            # Finalize instead of emitting a partial then a final: one revision
            # and one output sequence per accepted capture hop, no lost revision.
            result = self.asr.process(frame, endpoint=endpoint)
            if result is not None and result['status'] == 'final':
                self.vad.reset()
                self._clear_policy()
            return result
        except Exception:
            self.reset()
            raise

    def finish(self):
        if self.closed:
            raise RuntimeError('endpoint provider is closed')
        try:
            return self.asr.finish()
        finally:
            self.vad.reset()
            self._clear_policy()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.asr.close()
            finally:
                self.vad.close()
