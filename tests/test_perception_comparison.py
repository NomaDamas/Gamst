import copy
import unittest
from benchmarks.perception_comparison import compare, plan_digest


def fixture(evidence='synthetic', mode='priced'):
    config = {key: 'declared' for key in ('model', 'revision', 'precision', 'hardware',
              'policy', 'warm_state', 'sampling', 'window_lookahead')}
    plan = dict(schema_version=1, evidence=evidence, held_out=True, evaluation_id='fixture',
        scoring_protocol='exact fixture scores; not model evaluation',
        latency_protocol='capture through answer ready including queues/providers',
        trace_provenance='generated unit-test traces',
        systems={'baseline': dict(config), 'candidate': dict(config)}, min_pairs=20,
        bootstrap_repetitions=500, bootstrap_seed=42,
        tasks=[dict(id=str(i), input_sha256=f'{i:064x}', critical_facts=['fact']) for i in range(20)],
        cost=dict(mode=mode, currency='USD', per_million={k: 1 for k in
            ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens')},
            unit='joules', definition='whole pipeline measured energy'))
    rows = []
    for task in plan['tasks']:
        for system in ('baseline', 'candidate'):
            candidate = system == 'candidate'
            rows.append(dict(task_id=task['id'], system=system, plan_sha256=plan_digest(plan),
                input_sha256=task['input_sha256'], latency_ms=50 if candidate else 100,
                quality_score=1, correct_critical_facts=['fact'], peak_memory_bytes=100,
                dropped_frames=0, usage={k: 1 if candidate else 4 for k in
                    ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens', 'calls')},
                resource_cost=1 if candidate else 4))
    return plan, rows


class Comparison(unittest.TestCase):
    def run_case(self, plan, rows):
        return compare(plan, rows, expected_plan_sha256=plan_digest(plan))

    def test_synthetic_cannot_pass_and_priced_cache_charges_are_counted(self):
        plan, rows = fixture()
        result = self.run_case(plan, rows)
        self.assertFalse(result['meets_numeric_gates'])
        self.assertFalse(result['gates']['measured_held_out'])
        self.assertAlmostEqual(result['point']['cost_ratio'], .25)
        self.assertAlmostEqual(float(result['summaries']['baseline']['cost']), 320/1e6)
        self.assertEqual(result, self.run_case(plan, list(reversed(rows))))

    def test_numeric_measured_gates_and_resource_cost(self):
        for mode in ('priced', 'resource'):
            plan, rows = fixture('measured', mode)
            result = self.run_case(plan, rows)
            self.assertTrue(result['meets_numeric_gates'])
            self.assertEqual(result['paired_bootstrap_95']['quality_delta'], [0, 0])
        plan, rows = fixture('measured', 'unknown')
        result = self.run_case(plan, rows)
        self.assertFalse(result['gates']['cost'])
        self.assertIsNone(result['summaries']['candidate']['cost'])

    def test_missing_mismatched_and_duplicate_pairs_fail_closed(self):
        plan, rows = fixture()
        for bad in (rows[:-1], rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):
                self.run_case(plan, bad)
        for field, value in (('input_sha256', 'f'*64), ('plan_sha256', 'f'*64),
                             ('quality_score', float('nan')), ('latency_ms', -1),
                             ('correct_critical_facts', ['unknown'])):
            bad = copy.deepcopy(rows)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                self.run_case(plan, bad)
        changed = copy.deepcopy(plan)
        changed['systems']['candidate']['policy'] = 'changed after run'
        with self.assertRaises(ValueError):
            compare(changed, rows, expected_plan_sha256=plan_digest(plan))

    def test_quality_and_rare_critical_fact_regressions_fail(self):
        plan, rows = fixture('measured')
        rows[1]['correct_critical_facts'] = []
        result = self.run_case(plan, rows)
        self.assertFalse(result['gates']['critical_facts'])
        self.assertEqual(result['missing_critical_facts'], [{'task_id': '0', 'fact': 'fact'}])
        rows[1]['correct_critical_facts'] = ['fact']
        rows[1]['quality_score'] = .9
        self.assertFalse(self.run_case(plan, rows)['gates']['quality'])

    def test_unknown_price_zero_baseline_and_inconclusive_quality(self):
        plan, rows = fixture('measured')
        for row in rows:
            if row['system'] == 'baseline':
                row['usage'] = {key: 0 for key in row['usage']}
        result = self.run_case(plan, rows)
        self.assertFalse(result['gates']['cost'])
        plan, rows = fixture('measured')
        # Equal aggregate quality with paired gains and losses is inconclusive.
        for i in range(0, len(rows), 2):
            rows[i]['quality_score'] = .5
            rows[i+1]['quality_score'] = 1 if i % 4 else 0
        result = self.run_case(plan, rows)
        self.assertEqual(result['point']['quality_delta'], 0)
        self.assertLess(result['paired_bootstrap_95']['quality_delta'][0], 0)
        self.assertFalse(result['gates']['quality'])

    def test_strict_cost_boundary_does_not_round_into_a_pass(self):
        from decimal import Decimal, localcontext
        for tiny_price in (1e-15, 1e-300):
            plan, rows = fixture('measured')
            plan['cost']['per_million'] = dict(input_tokens=1, output_tokens=tiny_price,
                cache_read_tokens=0, cache_write_tokens=0)
            for row in rows:
                candidate = row['system'] == 'candidate'
                row['plan_sha256'] = plan_digest(plan)
                row['usage'] = dict(input_tokens=80 if candidate else 100,
                    output_tokens=1 if candidate else 0, cache_read_tokens=0,
                    cache_write_tokens=0, calls=1)
            with localcontext() as context:
                context.prec = 6
                result = self.run_case(plan, rows)
            self.assertEqual(result['point']['cost_ratio'], .8)  # display rounding only
            self.assertGreater(Decimal(result['exact_point']['cost_ratio']), Decimal('.8'))
            self.assertFalse(result['gates']['cost'])
            self.assertFalse(result['meets_numeric_gates'])
            for row in rows:
                row['usage']['output_tokens'] = 0
            self.assertTrue(self.run_case(plan, rows)['gates']['cost'])

    def test_all_prices_required_and_small_sample_not_accepted(self):
        plan, rows = fixture()
        del plan['cost']['per_million']['cache_write_tokens']
        with self.assertRaises(ValueError):
            self.run_case(plan, rows)
        plan, rows = fixture('measured')
        plan['tasks'] = plan['tasks'][:2]
        rows = rows[:4]
        for row in rows:
            row['plan_sha256'] = plan_digest(plan)
        self.assertFalse(self.run_case(plan, rows)['gates']['sample_count'])
