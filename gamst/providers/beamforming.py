"""Causal two-microphone integer-delay sum; raw output stays outside context."""
import math
from dataclasses import dataclass, replace
from ..capabilities import CapabilitySpec, OutputField, identifier
from .direction import GCCPHATDirection


@dataclass(frozen=True)
class BeamformedBlock:
    frame: object  # Mono InputFrame for an explicitly selected downstream processor.
    context: dict


class DelaySumBeamformer:
    def __init__(self, *, steering_delay_seconds, output_sensor_id, hop_ms=20, **array_profile):
        if type(hop_ms) is not int or not 5 <= hop_ms <= 100:
            raise ValueError('hop_ms must be 5..100')
        if 'window_ms' in array_profile:
            raise ValueError('configure hop_ms, not window_ms')
        self.validator = GCCPHATDirection(**array_profile)
        if type(steering_delay_seconds) not in (int, float) or not math.isfinite(steering_delay_seconds) or abs(steering_delay_seconds) > self.validator.spacing/self.validator.speed:
            raise ValueError('steering delay must lie within the declared array geometry')
        self.delay = float(steering_delay_seconds)
        self.output_sensor_id = identifier(output_sensor_id)
        self.hop_ms = hop_ms
        self.spec = self.capability(hop_ms=hop_ms)
        self.continuity_resets = 0
        self.reset()

    @staticmethod
    def capability(*, hop_ms=20):
        if type(hop_ms) is not int or not 5 <= hop_ms <= 100:
            raise ValueError('hop_ms must be 5..100')
        return CapabilitySpec('audio.delay_sum', 'audio.selection', 'audio', 'gamst.delay_sum', '1',
            (OutputField('selection', 'object'),), 'native', window_ms=hop_ms,
            stateful=True, reset_on_gap=True, interpretation='estimate', devices=('cpu',),
            sensor_requirements=('synchronized-two-channel-pcm', 'array-calibration'), license='Apache-2.0')

    def reset(self):
        self.history = None
        self.owner = self.last_sequence = self.last_end = self.expected_end = None

    def close(self):
        self.reset()
        self.validator.close()

    def transform(self, frame):
        """Return None on nonzero-delay warmup, otherwise PCM plus bounded context.

        Call once per hop. Feed the returned frame to VAD/ASR; publish only context.
        """
        try:
            values = self.validator._input(frame, self.hop_ms)
            if frame.sensor.sensor_id == self.output_sensor_id:
                raise ValueError('output sensor ID must differ from input sensor ID')
        except Exception:
            self.reset()
            raise
        np, sensor, rate = self.validator.np, frame.sensor, frame.sample_rate
        owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch, rate)
        end, sequence = frame.timestamp_ms, sensor.sequence
        if self.owner == owner and (sequence <= self.last_sequence or end <= self.last_end):
            self.reset()
            raise ValueError('sequence and capture end must increase')
        if self.owner is not None and (self.owner != owner or sequence != self.last_sequence+1 or
                                     abs(end-(self.expected_end+self.hop_ms)) > 1.):
            self.continuity_resets += 1
            self.reset()
        self.expected_end = end if self.expected_end is None else self.expected_end+self.hop_ms
        self.owner, self.last_sequence, self.last_end = owner, sequence, end
        lag = int(round(self.delay*rate))  # t0-t1: positive means mic1 arrives earlier.
        delays = (max(0, -lag), max(0, lag))
        retained = abs(lag)
        previous = self.history
        combined = values if previous is None else np.concatenate((previous, values))
        self.history = combined[-retained:].copy() if retained else None
        if retained and previous is None:
            return None  # Never expose zero padding or pretend it was captured audio.
        offset, length = len(combined)-len(values), len(values)
        mono = (combined[offset-delays[0]:offset-delays[0]+length, 0] +
                combined[offset-delays[1]:offset-delays[1]+length, 1])*.5
        output = replace(frame, data=mono.astype(np.float32),
            sensor=replace(sensor, sensor_id=self.output_sensor_id, channels=('beam',)))
        selection = dict(method='integer_delay_and_sum', input_sensor_id=sensor.sensor_id,
            output_sensor_id=self.output_sensor_id, input_channels=list(sensor.channels),
            calibration_id=sensor.calibration_id, coordinate_frame=sensor.coordinate_frame,
            requested_delay_seconds=self.delay, applied_delay_seconds=lag/rate,
            quantization_error_seconds=lag/rate-self.delay, channel_delays_samples=list(delays),
            alignment_latency_ms=retained*1000/rate, hop_ms=self.hop_ms, lookahead_ms=0,
            start_ms=end-self.hop_ms, end_ms=end, sample_rate=rate,
            rms=float(np.sqrt(np.mean(mono*mono))), peak=float(np.max(np.abs(mono))),
            input_clipping_present=bool(np.any(np.abs(combined) >= 1)),
            continuity_resets=self.continuity_resets, far_field_assumption=True,
            speaker_identity_known=False, noise_reduction_verified=False,
            physical_calibration_verified=False)
        return BeamformedBlock(output, {'selection': selection})

    def process(self, frame):
        """SensorStreams provider mode: publish selection metrics, discard derived PCM."""
        result = self.transform(frame)
        return None if result is None else result.context
