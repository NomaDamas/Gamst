import dataclasses
import json
import unittest

from gamst import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.physical_state import PhysicalState


def make_provider(**kwargs):
    options = dict(calibration_id='cal1', input_frame='world', output_frame='world',
                   clock_id='capture-clock', gravity_m_s2=(0, 0, -9.81),
                   jerk_spectral_density_m2_s5=1e-8, horizon_ms=400,
                   prediction_steps=4, event_window_ms=120)
    options.update(kwargs)
    return PhysicalState(**options)


def track(track_id='ball', position=(0, 0, 1), *, variance=1e-10, drag=0,
          radius=.1, mass=None, mass_calibrated=False, shape=None, container=False):
    return dict(track_id=track_id, position_m=list(position),
        position_variance_m2=[variance]*3, input_dimensionality=3, calibrated=True,
        shape=shape or dict(kind='sphere', radius_m=radius),
        linear_drag_coefficient_s_inv=drag, mass_kg=mass,
        mass_calibrated=mass_calibrated, is_container=container)


def uncalibrated(track_id='image-only'):
    return dict(track_id=track_id, position_m=None, position_variance_m2=None,
        input_dimensionality=2, calibrated=False, shape=None,
        linear_drag_coefficient_s_inv=None, mass_kg=None,
        mass_calibrated=False, is_container=False)


def frame(seq, timestamp_ms, objects, *, planes=None, audio=None, tactile=None):
    data = dict(units='m', tracker_id='tracker', tracker_version='v1',
        clock_id='capture-clock', objects=objects, planes=planes or [],
        audio_onsets=audio or [], tactile_events=tactile or [])
    sensor = SensorMetadata('camera', 'depth', seq, 'world',
                            calibration_id='cal1', stream_epoch='epoch1')
    return InputFrame(data, 'video', timestamp_ms, sensor=sensor)


def plane(**kwargs):
    value = dict(plane_id='ground', point_m=[0, 0, 0], normal_unit=[0, 0, 1],
                 friction_coefficient=.4, restitution=None)
    value.update(kwargs)
    return value


def audio(event_id='audio-impact', timestamp_ms=500, track_id='ball'):
    return dict(event_id=event_id, timestamp_ms=timestamp_ms, impact_like=True,
                confidence=.8, track_id=track_id)


def tactile(event_id='tactile-impact', timestamp_ms=500, track_id='ball', **kwargs):
    value = dict(event_id=event_id, timestamp_ms=timestamp_ms, track_id=track_id,
                 force_spike=True, slip=False, force_n=None, force_calibrated=False)
    value.update(kwargs)
    return value


class PhysicalStateTests(unittest.TestCase):
    def test_projectile_prediction_matches_analytic_future(self):
        provider = make_provider()

        def position(t):
            return (.5*t, 0, 1+2*t-.5*9.81*t*t)

        for sequence in range(9):
            t = sequence*.05
            result = provider.process(frame(sequence, sequence*50, [track(position=position(t))]))
        free = result['objects'][0]['hypotheses'][0]
        self.assertEqual(free['hypothesis'], 'free_fall_gravity_linear_drag')
        for sample in free['predictions']:
            future_t = sample['timestamp_ms']/1000
            expected = position(future_t)
            for actual, wanted in zip(sample['position_m'], expected):
                self.assertAlmostEqual(actual, wanted, delta=2e-4)
        self.assertEqual(result['geometry']['state_order'][:3], ['x_m', 'y_m', 'z_m'])
        self.assertFalse(result['geometry']['physical_parameters_verified'])

    def test_prediction_covariance_envelope_grows_with_horizon(self):
        provider = make_provider(jerk_spectral_density_m2_s5=2.)
        for sequence in range(3):
            result = provider.process(frame(sequence, sequence*100,
                [track(position=(sequence*.1, 0, 1), variance=1e-4)]))
        samples = result['objects'][0]['hypotheses'][0]['predictions']
        radii = [sample['envelope_radius_m'] for sample in samples]
        traces = [sum(sample['position_covariance_m2'][i][i] for i in range(3))
                  for sample in samples]
        self.assertTrue(all(a < b for a, b in zip(radii, radii[1:])), radii)
        self.assertTrue(all(a < b for a, b in zip(traces, traces[1:])), traces)
        self.assertTrue(all(not sample['coverage_calibrated'] for sample in samples))

    def test_synthetic_collision_contact_and_mass_gated_impulse(self):
        provider = make_provider(contact_tolerance_m=1e-6)
        result = None
        for sequence in range(8):
            t = sequence/10
            objects = [track('a', (-1.2+t, 0, 1), radius=.5),
                       track('b', (1.2-t, 0, 1), radius=.5)]
            result = provider.process(frame(sequence, sequence*100, objects))
        self.assertEqual(len(result['contacts']), 1, result['contacts'])
        contact = result['contacts'][0]
        self.assertEqual(contact['kind'], 'contact_candidate')
        self.assertAlmostEqual(contact['gap_m'], 0, delta=1e-8)
        self.assertAlmostEqual(contact['relative_normal_velocity_m_s'], -2, delta=1e-5)
        self.assertIsNone(contact['normal_impulse_n_s'])
        self.assertIsNone(contact['normal_force_n'])

        calibrated = make_provider(contact_tolerance_m=1e-6)
        for sequence in range(8):
            t = sequence/10
            objects = [track('a', (-1.2+t, 0, 1), radius=.5, mass=1, mass_calibrated=True),
                       track('b', (1.2-t, 0, 1), radius=.5, mass=1, mass_calibrated=True)]
            result = calibrated.process(frame(sequence, sequence*100, objects))
        contact = result['contacts'][0]
        self.assertEqual(contact['kind'], 'contact_estimate')
        self.assertAlmostEqual(contact['normal_impulse_n_s'], 1, delta=1e-4)
        self.assertIsNone(contact['normal_force_n'])

    def test_uncalibrated_2d_track_abstains_without_state_or_predictions(self):
        provider = make_provider()
        result = provider.process(frame(0, 0, [uncalibrated()]))
        item = result['objects'][0]
        self.assertEqual(item['status'], 'abstained')
        self.assertEqual(item['reason'], 'calibrated_metric_3d_position_and_variance_required')
        self.assertIsNone(item['state'])
        self.assertIsNone(item['covariance'])
        self.assertEqual(item['hypotheses'], [])
        self.assertEqual(result['contacts'], [])
        self.assertEqual(result['summary']['abstained_track_count'], 1)

    def test_support_containment_and_grounded_friction_hypothesis(self):
        provider = make_provider()
        objects = [track('resting', (0, 0, .5), radius=.5),
                   track('box', (5, 0, 1), shape=dict(kind='aabb', half_extents_m=[2, 2, 2]),
                         container=True),
                   track('inside', (5, 0, 1), radius=.25)]
        for sequence in range(3):
            result = provider.process(frame(sequence, sequence*100, objects, planes=[plane()]))
        resting = next(item for item in result['objects'] if item['track_id'] == 'resting')
        grounded = resting['hypotheses'][1]
        self.assertEqual(grounded['status'], 'estimated')
        self.assertEqual(grounded['support_plane_id'], 'ground')
        self.assertTrue(any(r['relation'] == 'support' and r['supported_track_id'] == 'resting'
                            for r in result['relations']))
        containment = next(r for r in result['relations'] if r['relation'] == 'containment')
        self.assertEqual(containment['container_track_id'], 'box')
        self.assertEqual(containment['contained_track_id'], 'inside')
        self.assertGreater(containment['clearance_m'], 0)

    def test_cross_modal_impact_alignment_and_tactile_force_calibration(self):
        provider = make_provider(deceleration_threshold_m_s2=1.)
        positions = [0, .1, .2, .3, .38, .44]
        for sequence, x in enumerate(positions):
            events_audio = [audio()] if sequence == 5 else []
            events_tactile = [tactile(force_n=2.5, force_calibrated=True)] if sequence == 5 else []
            result = provider.process(frame(sequence, sequence*100, [track(position=(x, 0, 1))],
                audio=events_audio, tactile=events_tactile))
        self.assertEqual(len(result['events']), 1, result['objects'][0]['state'])
        event = result['events'][0]
        self.assertEqual(event['kind'], 'impact_candidate')
        self.assertLessEqual(event['alignment_span_ms'], 120)
        self.assertGreater(event['visible_deceleration_m_s2'], 1)
        self.assertEqual(event['tactile_force_n'], 2.5)
        self.assertEqual(event['tactile_force_status'], 'calibrated_measurement')
        self.assertIsNone(event['fused_probability'])
        self.assertFalse(event['event_verified'])

    def test_maximum_snapshot_stays_bounded_and_reports_truncation(self):
        provider = make_provider()
        objects = [track(str(index), (index*.001, 0, .5+index*.01),
                         radius=.5+index*.01, container=True) for index in range(8)]
        for sequence in range(3):
            result = provider.process(frame(sequence, sequence*100, objects, planes=[plane()]))
        self.assertLess(len(json.dumps(result)), 65536)
        self.assertEqual(result['summary']['calibrated_track_count'], 8)
        self.assertTrue(result['summary']['contacts_truncated'])
        self.assertTrue(result['summary']['relations_truncated'])
        self.assertEqual(len(result['contacts']), 12)
        self.assertEqual(len(result['relations']), 16)

    def test_invalid_uncalibrated_force_and_source_changes_fail_closed(self):
        provider = make_provider()
        provider.process(frame(0, 0, [track()]))
        bad = frame(1, 100, [track()], tactile=[tactile(force_n=3, force_calibrated=False)])
        with self.assertRaises(ValueError):
            provider.process(bad)
        changed = frame(2, 200, [track()])
        changed = dataclasses.replace(changed, sensor=dataclasses.replace(
            changed.sensor, calibration_id='other'))
        with self.assertRaises(ValueError):
            provider.process(changed)
        recovered = provider.process(frame(3, 300, [track()]))
        self.assertEqual(recovered['objects'][0]['sample_count'], 1)
        provider.close()
        provider.close()
        with self.assertRaises(RuntimeError):
            provider.process(frame(4, 400, [track()]))


if __name__ == '__main__':
    unittest.main()
