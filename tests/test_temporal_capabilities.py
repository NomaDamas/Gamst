import math
import random
import time
import unittest

from gamst import Gamst, SensorMetadata
from gamst.mcp import MCPServer
from gamst.plugins import InputFrame
from gamst.providers.flight import G, FlightPhysics
from gamst.providers.repetition import RepetitionCounter
from gamst.providers.rhythm import RhythmEstimator
from gamst.providers.risk_field import RiskField


def sensor(seq, epoch='e1', source='cam', sensor_id='s'):
    return SensorMetadata(source, sensor_id, seq, stream_epoch=epoch)


def frame(data, seq, ts, modality='video', epoch='e1'):
    return InputFrame(data, modality, ts, sensor=sensor(seq, epoch))


def jump_series(fps, flight_s=0.6, takeoff_s=1.51, total_s=4.0, noise=0.0, seed=1):
    rnd = random.Random(seed)
    for i in range(int(total_s * fps)):
        t = i / fps
        tau = t - takeoff_s
        z = 0.5 * G * tau * (flight_s - tau) if 0 <= tau <= flight_s else 0.0
        z += rnd.gauss(0, noise)
        yield i, int(t * 1000), {'foot_height_m': z, 'body_height_m': 1.0 + z}


class RepetitionTests(unittest.TestCase):
    def test_counts_each_period_once_with_drifting_posture(self):
        p = RepetitionCounter(channels=3)
        events = []
        for i in range(30 * 30):
            ts = int(i * 1000 / 30)
            value = [math.sin(2 * math.pi * ts / 2000), 0.01 * i, 0.0]  # 2 s period on a drifting baseline
            events += p.process(frame({'channels': value}, i, ts))['events']
        final = p.finish()
        self.assertEqual(final['count'], 15)
        self.assertAlmostEqual(final['period_s'], 2.0, places=1)
        self.assertEqual([e['index'] for e in events + final['events']], list(range(1, 16)))

    def test_one_fps_cannot_resolve_fast_repetitions(self):
        p = RepetitionCounter(channels=1, max_gap_ms=1500)
        for i in range(30):
            p.process(frame({'channels': [math.sin(2 * math.pi * i / 0.8)]}, i, i * 1000))  # 0.8 s period seen at 1 Hz
        self.assertNotEqual(p.finish()['count'], int(30 / 0.8))

    def test_gap_resets_instead_of_bridging(self):
        p = RepetitionCounter(channels=1)
        for i in range(30 * 10):
            p.process(frame({'channels': [math.sin(2 * math.pi * i / 60)]}, i, int(i * 1000 / 30)))
        out = p.process(frame({'channels': [0.0]}, 1000, 60000))
        self.assertEqual(out['count'], 0)
        self.assertEqual(out['status'], 'warmup')

    def test_rejects_wrong_channel_count(self):
        with self.assertRaises(ValueError):
            RepetitionCounter(channels=2).process(frame({'channels': [1.0]}, 0, 0))


class FlightTests(unittest.TestCase):
    def test_ballistic_fit_recovers_flight_between_frames(self):
        for fps in (30, 60):
            p = FlightPhysics(max_gap_ms=2000)
            events = [e for i, ts, d in jump_series(fps) for e in p.process(frame(d, i, ts))['events']]
            self.assertEqual(len(events), 1)
            jump = events[0]
            self.assertEqual(jump['timing_method'], 'ballistic_fit')
            self.assertAlmostEqual(jump['flight_s'], 0.6, delta=0.005)
            self.assertAlmostEqual(jump['height_flight_m'], G * 0.36 / 8, delta=0.005)
            self.assertAlmostEqual(jump['fitted_gravity_m_s2'], G, delta=0.1)
            self.assertAlmostEqual(jump['rise_m'], G * 0.36 / 8, delta=0.02)

    def test_noise_and_offset(self):
        p = FlightPhysics(max_gap_ms=2000, calibration_offset_m=0.1)
        events = [e for i, ts, d in jump_series(30, noise=0.005) for e in p.process(frame(d, i, ts))['events']]
        self.assertAlmostEqual(events[0]['height_flight_m'], G * 0.36 / 8 + 0.1, delta=0.01)

    def test_one_fps_sees_no_jump(self):
        p = FlightPhysics(max_gap_ms=2000, baseline_s=1.0)
        events = [e for i, ts, d in jump_series(1) for e in p.process(frame(d, i, ts))['events']]
        self.assertEqual(events, [])

    def test_imu_free_fall(self):
        p = FlightPhysics(mode='imu', max_gap_ms=100)
        events = []
        for i in range(300):
            t = i / 100
            a = 0.05 if 1.0 <= t < 1.5 else 1.0
            events += p.process(InputFrame({'accel_g': a}, 'imu', int(t * 1000), sensor=sensor(i)))['events']
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]['flight_s'], 0.5, delta=0.011)


class RhythmTests(unittest.TestCase):
    def test_rate_and_threshold_events(self):
        p = RhythmEstimator(band='heart', above_per_minute=100)
        events, last = [], None
        for i in range(30 * 40):
            ts = int(i * 1000 / 30)
            bpm = 72 if ts < 20000 else 110
            last = p.process(frame({'value': math.sin(2 * math.pi * bpm / 60 * ts / 1000)}, i, ts))
            events += last['events']
        self.assertEqual(last['status'], 'measuring')
        self.assertAlmostEqual(last['per_minute'], 110, delta=2)
        self.assertEqual([e['event'] for e in events], ['rate_above'])

    def test_pos_pulse_from_skin_colour(self):
        p = RhythmEstimator(band='heart', input_kind='rgb')
        for i in range(30 * 20):
            ts = int(i * 1000 / 30)
            w = math.sin(2 * math.pi * 1.25 * ts / 1000)
            last = p.process(frame({'rgb': [120 + 0.2 * w, 100 + 0.5 * w, 90]}, i, ts))
        self.assertAlmostEqual(last['per_minute'], 75, delta=2)

    def test_undersampled_stream_reports_no_number(self):
        p = RhythmEstimator(band='heart', max_gap_ms=1500)
        for i in range(40):
            last = p.process(frame({'value': math.sin(i)}, i, i * 1000))
        self.assertEqual(last['status'], 'undersampled')
        self.assertNotIn('per_minute', last)


RULE = {'id': 'stove_child', 'zones': [{'id': 'stove', 'rect': [0.6, 0.5, 0.9, 0.9]}],
        'evidence': [{'id': 'near_stove', 'kind': 'presence', 'zone': 'stove', 'weight': 0.3, 'sigma': 0.05},
                     {'id': 'boiling', 'kind': 'signal', 'signal': 'audio.boiling', 'weight': 0.3}]}


def kitchen(p, boil, near, seconds=6.0, start=0, seq0=0):
    out = []
    for k in range(int(seconds * 30)):
        x = 0.7 if near else 0.2
        r = p.process(frame({'points': [[x, 0.7]], 'signals': {'audio.boiling': boil}}, seq0 + k, start + int(k * 1000 / 30)))
        out.append(r)
    return out


class RiskFieldTests(unittest.TestCase):
    def test_requires_the_combination(self):
        for boil, near, alarm in ((1.0, False, False), (0.0, True, False), (1.0, True, True)):
            out = kitchen(RiskField(RULE), boil, near)
            self.assertEqual(out[-1]['alarm'], alarm, (boil, near))

    def test_decays_with_hysteresis_and_explains(self):
        p = RiskField(RULE)
        on = kitchen(p, 1.0, True)
        off = kitchen(p, 0.0, False, start=6000, seq0=1000)
        events = [e for r in on + off for e in r['events']]
        self.assertEqual([e['event'] for e in events], ['alarm_on', 'alarm_off'])
        self.assertEqual(set(events[0]['because']), {'near_stove', 'boiling'})
        self.assertGreater(events[1]['timestamp_ms'], 6000)  # alarm survives briefly after leaving (decay)

    def test_rule_validation(self):
        with self.assertRaises(ValueError):
            RiskField({'id': 'x', 'zones': [], 'evidence': []})
        with self.assertRaises(ValueError):
            RiskField(dict(RULE, evidence=[{'id': 'a', 'kind': 'presence', 'zone': 'missing'}]))


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_events_are_answerable_through_mcp(self):
        now = int(time.time() * 1000) - 20000
        p = RepetitionCounter(channels=1)
        async with Gamst(consent=True) as human:
            human.register_capability(p.spec)
            for i in range(30 * 12):
                ts = now + int(i * 1000 / 30)
                f = frame({'channels': [math.sin(2 * math.pi * i / 45)]}, i, ts)
                value = p.process(f)
                if value['events']:
                    await human.push_observation(p.spec.id, value, sensor=f.sensor, timestamp_ms=ts)
            page = await MCPServer(human).call_tool('get_observation_page',
                                                    {'capability_id': p.spec.id, 'events_only': True})
            text = str(page)
            self.assertIn("'event': 'repetition'", text)
            self.assertIn("'index': 1", text)

    async def test_risk_alarms_are_storable_and_queryable(self):
        # GAMST rejects uncalibrated "probability" values; the risk field must publish cleanly.
        now = int(time.time() * 1000) - 20000
        p = RiskField(RULE)
        async with Gamst(consent=True) as human:
            human.register_capability(p.spec)
            for k in range(30 * 6):
                ts = now + int(k * 1000 / 30)
                f = frame({'points': [[0.7, 0.7]], 'signals': {'audio.boiling': 1.0}}, k, ts)
                value = p.process(f)
                if value['events']:
                    await human.push_observation(p.spec.id, value, sensor=f.sensor, timestamp_ms=ts)
            page = await MCPServer(human).call_tool('get_observation_page',
                                                    {'capability_id': p.spec.id, 'events_only': True})
            self.assertIn("'event': 'alarm_on'", str(page))
            self.assertIn("'because'", str(page))


if __name__ == '__main__':
    unittest.main()
