"""Probability cloud: a user-written rule evaluated continuously as an evidence-accumulating risk.

The user declares zones (normalized image rectangles) and evidence terms; each capture supplies the light
models' outputs (tracked foot points, typed signal scores such as an audio-event score). The risk is a
bounded log-odds state

    L_t = exp(-dt/tau) * L_{t-1} + gain * (sum_i w_i * e_i - bias) * dt,    risk_t = sigmoid(L_t)

where presence evidence is a Gaussian kernel of the distance from each point to the zone (1 inside) and
signal evidence is the caller's score in [0, 1]. Evidence raises the cloud, absence lets it decay, and the
alarm uses hysteresis (ON at risk >= on, OFF at risk <= off) so it does not flicker. `risk` is an
uncalibrated 0..1 score, never reported as a probability (GAMST rejects uncalibrated probabilities).
Alarm transitions are timestamped events with their top contributing terms, so "how many times / when /
why" is answerable later. The provider infers no identity and no intent; it only evaluates the declared rule.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame


def _rect(value):
    if type(value) is not list or len(value) != 4 or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in value) \
            or not (value[0] < value[2] and value[1] < value[3]):
        raise ValueError('zone rect must be [x0, y0, x1, y1] normalized with x0 < x1 and y0 < y1')
    return [float(v) for v in value]


class RiskField:
    def __init__(self, rule: dict, *, max_gap_ms: int = 2000):
        rule = bounded_json(rule, max_bytes=8192)
        allowed = {'id', 'zones', 'evidence', 'bias', 'tau_s', 'gain', 'on', 'off'}
        if type(rule) is not dict or not {'id', 'zones', 'evidence'} <= set(rule) <= allowed:
            raise ValueError('rule requires id, zones, evidence (optional bias, tau_s, gain, on, off)')
        identifier(rule['id'], 'rule id')
        zones = rule['zones']
        if type(zones) is not list or not 1 <= len(zones) <= 8:
            raise ValueError('rule needs 1..8 zones')
        self.zones = {}
        for z in zones:
            if type(z) is not dict or set(z) != {'id', 'rect'}:
                raise ValueError('zone requires id and rect')
            self.zones[identifier(z['id'], 'zone id')] = _rect(z['rect'])
        terms = rule['evidence']
        if type(terms) is not list or not 1 <= len(terms) <= 16:
            raise ValueError('rule needs 1..16 evidence terms')
        self.terms = []
        for t in terms:
            if type(t) is not dict or 'kind' not in t or 'id' not in t:
                raise ValueError('evidence term requires id and kind')
            identifier(t['id'], 'evidence id')
            w = t.get('weight', 1.0)
            if type(w) not in (int, float) or not -10 <= w <= 10:
                raise ValueError('weight must be within +-10')
            if t['kind'] == 'presence':
                if set(t) - {'id', 'kind', 'zone', 'weight', 'sigma'} or t.get('zone') not in self.zones:
                    raise ValueError('presence term requires a declared zone (optional weight, sigma)')
                sigma = t.get('sigma', 0.05)
                if type(sigma) not in (int, float) or not 0 < sigma <= 1:
                    raise ValueError('sigma must be in (0, 1]')
                self.terms.append({'id': t['id'], 'kind': 'presence', 'zone': t['zone'], 'weight': float(w), 'sigma': float(sigma)})
            elif t['kind'] == 'signal':
                if set(t) - {'id', 'kind', 'signal', 'weight'} or not isinstance(t.get('signal'), str):
                    raise ValueError('signal term requires a signal name (optional weight)')
                identifier(t['signal'], 'signal name')
                self.terms.append({'id': t['id'], 'kind': 'signal', 'signal': t['signal'], 'weight': float(w)})
            else:
                raise ValueError('evidence kind must be presence or signal')
        self.bias, self.tau_s = float(rule.get('bias', 0.35)), float(rule.get('tau_s', 1.5))
        self.gain, self.on, self.off = float(rule.get('gain', 10.0)), float(rule.get('on', 0.8)), float(rule.get('off', 0.4))
        if not 0 <= self.bias <= 10 or not 0.05 <= self.tau_s <= 600 or not 0 < self.gain <= 100 or not 0 < self.off < self.on < 1:
            raise ValueError('require bias 0..10, tau_s 0.05..600, gain (0, 100], 0 < off < on < 1')
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 60000:
            raise ValueError('max_gap_ms must be 1..60000')
        self.rule_id, self.max_gap_ms = rule['id'], max_gap_ms
        self.rule_sha256 = hashlib.sha256(json.dumps(rule, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.spec = CapabilitySpec(
            'rule.risk_field.' + rule['id'], 'rule.risk', 'video', 'gamst.risk_field', '1.' + self.rule_sha256[:16],
            (OutputField('risk', 'number'), OutputField('alarm', 'boolean'),
             OutputField('log_odds', 'number'), OutputField('contributions', 'object'), OutputField('events', 'array'),
             OutputField('rule_sha256', 'string')),
            'native', stateful=True, reset_on_gap=True, freshness_ms=max_gap_ms, interpretation='estimate',
            devices=('cpu',), sensor_requirements=('normalized-image-points', 'typed-signal-scores'), license='Apache-2.0')
        self._lock = threading.RLock()
        self._closed = False
        self._clear()

    def _clear(self):
        self._owner = self._last = None
        self._log_odds, self._alarm = -3.0, False

    def reset(self):
        with self._lock:
            self._clear()

    def close(self):
        with self._lock:
            self._clear()
            self._closed = True

    def process(self, frame):
        with self._lock:
            if self._closed:
                raise RuntimeError('risk field provider is closed')
            try:
                return self._process(frame)
            except Exception:
                self._clear()
                raise

    def _distance(self, zone, point):
        x0, y0, x1, y1 = self.zones[zone]
        x, y = point
        return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))

    def _process(self, frame):
        if not isinstance(frame, InputFrame) or frame.modality != 'video' or not isinstance(frame.sensor, SensorMetadata):
            raise ValueError('structured video InputFrame with sensor metadata required')
        identifier(frame.subject_id)
        ts = frame.timestamp_ms
        if type(ts) is not int or not 0 <= ts <= 2**53 - 1:
            raise ValueError('integer capture timestamp required')
        data = bounded_json(frame.data, max_bytes=16384)
        if type(data) is not dict or not set(data) <= {'points', 'signals'}:
            raise ValueError('data may contain only points and signals')
        points = data.get('points', [])
        if type(points) is not list or len(points) > 64 or any(
                type(p) is not list or len(p) != 2 or any(type(v) not in (int, float) or not -0.5 <= v <= 1.5 for v in p) for p in points):
            raise ValueError('points must be at most 64 normalized [x, y] pairs')
        signals = data.get('signals', {})
        if type(signals) is not dict or len(signals) > 32 or any(
                type(v) not in (int, float) or not 0 <= v <= 1 for v in signals.values()):
            raise ValueError('signals must map names to scores in [0, 1]')
        sensor = frame.sensor
        owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch)
        sequence = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        if owner == self._owner and self._last and (ts <= self._last[0] or sequence <= self._last[1]):
            raise ValueError('capture timestamps and sequence must strictly increase')
        if owner != self._owner or not self._last or ts - self._last[0] > self.max_gap_ms:
            self._clear()  # unknown interval: restart from the prior instead of extrapolating
            dt = 0.0
        else:
            dt = (ts - self._last[0]) / 1000.0
        self._owner, self._last = owner, (ts, sequence)
        contributions = {}
        for t in self.terms:
            if t['kind'] == 'presence':
                e = sum(math.exp(-self._distance(t['zone'], p) ** 2 / (2 * t['sigma'] ** 2)) for p in points)
            else:
                e = float(signals.get(t['signal'], 0.0))
            contributions[t['id']] = round(t['weight'] * e, 4)
        if dt > 0:
            self._log_odds = math.exp(-dt / self.tau_s) * self._log_odds + self.gain * (sum(contributions.values()) - self.bias) * dt
            self._log_odds = max(-6.0, min(6.0, self._log_odds))
        p = 1.0 / (1.0 + math.exp(-self._log_odds))
        events = []
        top = sorted(((v, k) for k, v in contributions.items() if v > 0), reverse=True)[:3]
        if not self._alarm and p >= self.on:
            self._alarm = True
            events.append({'event': 'alarm_on', 'timestamp_ms': ts, 'risk': round(p, 4),
                           'because': [k for _, k in top], 'rule_id': self.rule_id})
        elif self._alarm and p <= self.off:
            self._alarm = False
            events.append({'event': 'alarm_off', 'timestamp_ms': ts, 'risk': round(p, 4), 'rule_id': self.rule_id})
        return self.spec.validate({'risk': round(p, 4), 'alarm': self._alarm, 'log_odds': round(self._log_odds, 4),
                                   'contributions': contributions, 'events': events, 'rule_sha256': self.rule_sha256})
