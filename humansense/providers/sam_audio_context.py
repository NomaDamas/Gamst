"""Capture-bound complete-window SAM estimates; media stays in the owned pipe."""
from copy import deepcopy
import hashlib
import re
import threading

from ..capabilities import bounded_json
from ..sam_audio_artifact import _absolute
from ..sam_audio_windows import prepare_window
from ..separation_media import SeparationPair, capability as transport_capability, frame_binding, pack_pair
from ..provider_wire import encode_frame


def configuration(value):
    value = bounded_json(value)
    required = {'manifest_path', 'manifest_sha256', 'prompt'}
    if (type(value) is not dict or not required <= set(value) or
            set(value)-required-{'anchors', 'device', 'ode_steps', 'random_seed'}):
        raise ValueError('explicit local SAM manifest, digest and prompt required')
    _absolute(value['manifest_path'])
    if type(value['manifest_sha256']) is not str or re.fullmatch('[0-9a-f]{64}', value['manifest_sha256']) is None:
        raise ValueError('frozen manifest SHA256 required')
    prompt = value['prompt']
    if type(prompt) is not str or not prompt.strip() or len(prompt) > 4096 or len(prompt.encode('utf-8')) > 4096:
        raise ValueError('bounded nonempty UTF8 prompt required')
    value.setdefault('anchors', [])
    # Duration/hop dependent anchor validation happens on each actual window.
    anchors = value['anchors']
    if type(anchors) is not list or len(anchors) > 32:
        raise ValueError('at most32 manual anchors required')
    for anchor in anchors:
        if (type(anchor) is not list or len(anchor) != 3 or type(anchor[0]) is not str or
                anchor[0] not in ('+', '-') or any(type(v) not in (int, float) for v in anchor[1:]) or
                not 0 <= anchor[1] < anchor[2] <= 4):
            raise ValueError('bounded window-relative manual anchor required')
    value.setdefault('device', 'cpu'); value.setdefault('ode_steps', 32); value.setdefault('random_seed', 0)
    if type(value['device']) is not str or value['device'] not in ('cpu', 'cuda'):
        raise ValueError('explicit cpu or cuda device required')
    if type(value['ode_steps']) is not int or not 1 <= value['ode_steps'] <= 64:
        raise ValueError('midpoint ODE steps must be1..64')
    if type(value['random_seed']) is not int or not 0 <= value['random_seed'] < 2**63:
        raise ValueError('bounded explicit random seed required')
    return value


class SAMAudioContext:
    def __init__(self, backend, config):
        self.config = configuration(config)
        self.spec = self.capability(self.config)
        self.backend = backend
        self.lock = threading.RLock()
        self.closed = False
        self._previous = None
        self._reason = 'initial'

    @staticmethod
    def capability(config):
        return transport_capability(dict(configuration(config), native_adapter_version=1,
            window_policy=dict(sample_rate=48000, min_samples=48000, max_samples=192000,
                max_feature_frames=512, partial_window='reject', output_amplitude_limit=16,
                additional_lookahead_ms=0)))

    @classmethod
    def from_local(cls, config):
        from .sam_audio_native import SAMAudioNative
        config = configuration(config)
        backend = SAMAudioNative.from_local(config)
        try:
            return cls(backend, config)
        except BaseException:
            backend.close()
            raise

    def process(self, frame):
        with self.lock:
            if self.closed:
                raise RuntimeError('SAM context closed')
            header, payload = encode_frame(frame)
            binding = frame_binding(header, payload, generation=0, config_id=self.spec.version)
            # Preparation consumes the exact immutable wire snapshot validated
            # above, never a second read of the application's mutable array.
            import numpy as np
            prepared = prepare_window(np.frombuffer(payload, dtype='<f4').copy(),
                sample_rate=48000, prompt=self.config['prompt'], anchors=self.config['anchors'],
                hop_length=self.backend.hop_length)
            sensor = binding['sensor']
            identity = (binding['subject_id'], sensor['source_id'], sensor['sensor_id'],
                        sensor['stream_epoch'], sensor['coordinate_frame'], sensor['calibration_id'], tuple(sensor['channels']), sensor['capture_sequence'] is not None)
            sequence = sensor['capture_sequence'] if sensor['capture_sequence'] is not None else sensor['sequence']
            reason = self._reason
            if self._previous is not None:
                previous_identity, previous_sequence, previous_end = self._previous
                if identity != previous_identity or sequence <= previous_sequence or binding['capture_end_ms'] <= previous_end:
                    raise ValueError('SAM capture lane changed or regressed without reset')
                if sequence != previous_sequence+1 or binding['capture_start_ms'] > previous_end+1:
                    reason = 'gap'
            target, residual, tokens, inference_ms = self.backend.predict(prepared)
            model = deepcopy(self.backend.provenance)
            fixture = getattr(self.backend, 'fixture_engine', False)
            if fixture:
                model['fixture_engine'] = True
            observation = dict(separation=dict(status='fixture-estimate' if fixture else 'estimated',
                quality_validated=False, ambient_preserved=True, additive_reconstruction_verified=False,
                native_streaming=False, cross_window_continuity_verified=False, discontinuity=reason is not None),
                window=dict(capture_start_ms=binding['capture_start_ms'], capture_end_ms=binding['capture_end_ms'],
                    sample_rate=48000, sample_count=prepared.sample_count, padded_sample_count=prepared.padded_samples,
                    trimmed_padding_samples=prepared.padded_samples-prepared.sample_count, inference_ms=inference_ms,
                    collection_ms=prepared.sample_count/48, additional_lookahead_ms=0, whole_window=True,
                    partial_window_policy='reject', discontinuity=reason is not None, clock_verified=False),
                prompt=dict(sha256=prepared.prompt_sha256, raw_text_sha256=hashlib.sha256(prepared.prompt.encode()).hexdigest(),
                    utf8_bytes=len(prepared.prompt.encode()), anchor_count=len(prepared.anchors),
                    feature_spans=[list(span) for span in prepared.feature_spans], **tokens),
                model=model, media={})
            if reason is not None:
                observation['window']['discontinuity_reason'] = reason
            observation = self.spec.validate(observation)
            pair = SeparationPair(observation, target, residual, prepared.padded_samples)
            pack_pair(pair, binding, self.spec)
            self._previous = (identity, sequence, binding['capture_end_ms'])
            self._reason = None
            return pair

    def reset(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('SAM context closed')
            self._previous = None; self._reason = 'reset'

    def finish(self):
        return None  # No partial windows or buffered tail.

    def close(self):
        with self.lock:
            self.closed = True
            self._previous = None
            self.backend.close()
