import copy
from fractions import Fraction
import random
import unittest

from benchmarks.perception_comparison import plan_digest
from benchmarks.stream_cost_ledger import evaluate
from benchmarks.whole_stream_comparison import compare_whole_stream
from test_stream_cost_ledger import fixture as ledger_fixture
from test_three_arm_comparison import fixture as plan_fixture


def fixture(costs=((10, 7, 8),)*20):
    plan, _ = plan_fixture()
    plan['comparison'].update(min_pairs=20, bootstrap_repetitions=500, bootstrap_seed=7)
    protocol, ledger = ledger_fixture()
    protocol['comparison_plan_sha256'] = ledger['comparison_plan_sha256'] = plan_digest(plan)
    originals = copy.deepcopy((protocol['streams'], ledger['streams'], ledger['entries']))
    protocol['streams'], ledger['streams'], ledger['entries'] = [], [], []
    groups = []
    for i, amounts in enumerate(costs):
        recording = format(i+1, '064x')
        group = dict(recording_sha256=recording, streams={})
        for arm, amount in zip('ABC', amounts):
            old = 'stream-'+arm; sid = old+'-'+str(i)
            p = copy.deepcopy(next(s for s in originals[0] if s['id'] == old))
            p.update(id=sid, recording_sha256=recording, execution_namespace='exec-'+sid, cache_namespace='cache-'+sid)
            for cid, req in p['requirements'].items():
                req.update(state='required' if cid == 'resident_idle' else 'not_applicable',
                           lanes=[dict(resource_id=sid, lane_id=cid)] if cid == 'resident_idle' else [])
            protocol['streams'].append(p)
            observed = copy.deepcopy(next(s for s in originals[1] if s['id'] == old)); observed['id'] = sid
            ledger['streams'].append(observed)
            entry = copy.deepcopy(next(e for e in originals[2] if e['stream_id'] == old and e['component_id'] == 'resident_idle'))
            entry.update(stream_id=sid, resource_id=sid, quantity=amount*2,
                         source_quantity=dict(numerator=amount*2, denominator=1), billed_amount=amount)
            entry['identity'].update(invoice_id=sid, document_sha256=plan_digest(sid))
            ledger['entries'].append(entry); group['streams'][arm] = sid
        groups.append(group)
    policy = dict(version='matched_recording_elapsed_hour_v1', threshold=dict(numerator=4, denominator=5),
                  interval='nearest_rank_95', bootstrap_seed=7, bootstrap_repetitions=500, min_recordings=20)
    return plan, protocol, ledger, groups, policy


def run(data):
    plan, protocol, ledger, groups, policy = data
    ledger['protocol_sha256'] = plan_digest(protocol)
    report = evaluate(protocol, ledger, expected_protocol_sha256=plan_digest(protocol), expected_ledger_sha256=plan_digest(ledger))
    return compare_whole_stream(plan, protocol, ledger, report, groups, policy)


class WholeStreamComparison(unittest.TestCase):
    def test_exact_threshold_whole_overhead_and_unchanged_inputs(self):
        data = fixture(); run(data); before = copy.deepcopy(data)
        result = run(data)
        self.assertEqual(before, data)
        self.assertTrue(result['primary_meets_whole_stream_cost_gate'])
        self.assertEqual(result['comparisons']['C_vs_A']['cost_ratio'], dict(numerator=4, denominator=5))
        self.assertEqual(result['canonical_arms']['A']['monetary_atomic'], 200)
        self.assertEqual(result['nearest_rank_indices'], dict(lower=12, upper=487))
        self.assertFalse(result['whole_pipeline_cost_verified'])
        self.assertEqual(result['actual_acceptance'], 'incomplete')
        # All costs here are explicit idle overhead. No cheap task subtotal can hide them.
        costly = run(fixture(((10, 7, 11),)*20))
        self.assertFalse(costly['primary_meets_whole_stream_cost_gate'])

    def test_undefined_bootstrap_draw_never_narrows_interval(self):
        result = run(fixture(((0, 0, 0),)*19+((10, 1, 1),)))
        comparison = result['comparisons']['C_vs_A']
        self.assertEqual(comparison['cost_ratio'], dict(numerator=1, denominator=10))
        self.assertGreater(comparison['undefined_baseline_draws'], 0)
        self.assertIsNone(comparison['interval_95'])
        self.assertFalse(result['primary_meets_whole_stream_cost_gate'])
        zero = run(fixture(((0, 0, 0), (0, 0, 0))))
        self.assertIsNone(zero['comparisons']['C_vs_A']['cost_ratio'])

    def test_few_recordings_are_not_task_replicates(self):
        result = run(fixture(((10, 1, 1),)))
        self.assertFalse(result['enough_recordings'])
        self.assertFalse(result['primary_meets_whole_stream_cost_gate'])

    def test_optional_capture_counters_must_be_known(self):
        data = fixture()
        for stream in data[1]['streams']:
            stream['sources'][0].update(semantics='captured_only', required_counters=['captured'])
        data[2]['streams'][0]['counters'][0]['end']['attempted'] = None
        result = run(data)
        self.assertFalse(result['comparable_source_opportunities'])
        self.assertFalse(result['primary_meets_whole_stream_cost_gate'])

    def test_source_identity_and_semantics_and_availability_are_matched(self):
        for change in ('identity', 'semantics', 'availability'):
            with self.subTest(change=change):
                data = fixture()
                if change == 'identity':
                    data[1]['streams'][0]['sources'][0]['id'] = 'other-camera'
                    data[2]['streams'][0]['counters'][0]['source_id'] = 'other-camera'
                elif change == 'semantics':
                    data[1]['streams'][0]['sources'][0]['semantics'] = 'captured_only'
                else:
                    intervals = data[2]['streams'][0]['availability']
                    intervals[0]['end_ns'] -= 1; intervals[1]['start_ns'] -= 1
                self.assertFalse(run(data)['primary_meets_whole_stream_cost_gate'])

    def test_sampling_and_vlm_differences_are_permitted(self):
        data = fixture(); data[2]['streams'][0]['counters'][0]['end'].update(sampled=1, vlm_calls=100)
        self.assertTrue(run(data)['primary_meets_whole_stream_cost_gate'])

    def test_reordered_groups_have_identical_bootstrap(self):
        data = fixture(((10, 2, 3), (20, 8, 9), (30, 6, 4)))
        result = run(data); data[3].reverse()
        self.assertEqual(result['comparisons'], run(data)['comparisons'])

    def test_inventory_policy_and_report_forgery_rejected(self):
        for mode in ('missing', 'duplicate', 'policy', 'report'):
            with self.subTest(mode=mode):
                data = fixture()
                if mode == 'missing': data[3].pop()
                if mode == 'duplicate': data[3][1] = copy.deepcopy(data[3][0])
                if mode == 'policy': data[4]['threshold']['numerator'] = 5
                if mode != 'report':
                    with self.assertRaises(ValueError): run(data)
                else:
                    plan, protocol, ledger, groups, policy = data
                    ledger['protocol_sha256'] = plan_digest(protocol)
                    report = evaluate(protocol, ledger, expected_protocol_sha256=plan_digest(protocol), expected_ledger_sha256=plan_digest(ledger))
                    report['streams']['stream-C-0']['known_monetary_atomic'] = 0
                    with self.assertRaisesRegex(ValueError, 'recomputed'):
                        compare_whole_stream(plan, protocol, ledger, report, groups, policy)

    def test_independent_exact_bootstrap_oracle_and_draw_preflight(self):
        costs = tuple((10+i, 3+i%3, 2+i%7) for i in range(20))
        result = run(fixture(costs))
        rng = random.Random(7)
        ratios = []
        for _ in range(500):
            indices = [rng.randrange(20) for _ in range(20)]
            ratios.append(Fraction(sum(costs[i][2] for i in indices), sum(costs[i][0] for i in indices)))
        ratios.sort()
        interval = result['comparisons']['C_vs_A']['interval_95']
        self.assertEqual(interval['lower'], dict(numerator=ratios[12].numerator, denominator=ratios[12].denominator))
        self.assertEqual(interval['upper'], dict(numerator=ratios[487].numerator, denominator=ratios[487].denominator))
        data = fixture(((10, 7, 8),)*201)
        data[0]['comparison']['bootstrap_repetitions'] = data[4]['bootstrap_repetitions'] = 5000
        data[1]['comparison_plan_sha256'] = data[2]['comparison_plan_sha256'] = plan_digest(data[0])
        with self.assertRaisesRegex(ValueError, 'million'):
            run(data)

    def test_required_proxy_and_incomplete_cost_cannot_pass(self):
        data = fixture()
        stream = data[1]['streams'][0]
        stream['requirements']['capture_sensor'].update(state='required', lanes=[dict(resource_id='proxy-device', lane_id='proxy')])
        result = run(data)
        self.assertFalse(result['primary_meets_whole_stream_cost_gate'])
        self.assertTrue(any('proxy' in r['reason'] for r in result['comparability_reasons']))


if __name__ == '__main__':
    unittest.main()
