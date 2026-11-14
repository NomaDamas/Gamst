import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from benchmarks.perception_comparison import compare, plan_digest
from benchmarks.three_arm_comparison import compare_three_arm, project_pair_plan, _read
from test_perception_comparison import fixture as paired_fixture


def fixture(evidence='measured', mode='priced'):
    old, old_rows = paired_fixture(evidence, mode)
    common = {k: copy.deepcopy(v) for k, v in old.items() if k not in ('schema_version', 'systems')}
    plan = dict(schema_version=1, primary_comparison='C_vs_A', comparison=common,
                systems={a: copy.deepcopy(old['systems']['baseline' if a == 'A' else 'candidate']) for a in 'ABC'},
                baseline_sampling=dict(period_ms=1000, phase_ms=0, time_origin_policy='recording_start', audio_policy='same frozen ASR', task_policy='same causal tasks'))
    plan['systems']['A']['sampling'] = 'periodic_1fps'
    rows = []
    for row in old_rows:
        for arm in ('A',) if row['system'] == 'baseline' else ('B', 'C'):
            new = {k: copy.deepcopy(v) for k, v in row.items() if k != 'system' and (k != 'resource_cost' or mode == 'resource')}
            new.update(arm=arm, plan_sha256=plan_digest(plan), configuration_sha256=plan_digest(plan['systems'][arm]))
            rows.append(new)
    return plan, rows


def run(plan, rows):
    return compare_three_arm(plan, rows, expected_plan_sha256=plan_digest(plan), expected_rows_sha256=plan_digest(rows))


class ThreeArm(unittest.TestCase):
    def test_projection_parity_canonical_a_and_caller_ownership(self):
        plan, rows = fixture(); before = copy.deepcopy((plan, rows))
        result = run(plan, rows)
        self.assertEqual((plan, rows), before)
        a_hashes = []
        for arm in ('B', 'C'):
            pair = project_pair_plan(plan, arm, expected_plan_sha256=plan_digest(plan))
            derived = []
            for row in rows:
                if row['arm'] in ('A', arm):
                    new = {k: copy.deepcopy(v) for k, v in row.items() if k not in ('arm', 'configuration_sha256')}
                    new.update(system='baseline' if row['arm'] == 'A' else 'candidate', plan_sha256=plan_digest(pair))
                    derived.append(new)
            self.assertEqual(result['comparisons'][arm+'_vs_A'], compare(pair, derived, expected_plan_sha256=plan_digest(pair)))
            a_hashes.append([r['source_row_sha256'] for r in result['projections'][arm+'_vs_A']['mapping'] if r['arm'] == 'A'])
        self.assertEqual(*a_hashes)
        self.assertEqual(set(result['summaries']), {'A', 'B', 'C'})
        pair['systems']['baseline']['model'] = 'mutated'
        self.assertEqual((plan, rows), before)
        self.assertTrue(result['primary_meets_numeric_gates'])
        self.assertEqual(result['actual_acceptance'], 'incomplete')
        for key in ('source_verified', 'receipts_verified', 'sampling_execution_verified', 'recording_cluster_uncertainty_verified'):
            self.assertFalse(result[key])

    def test_b_is_diagnostic_and_c_quality_gate_remains(self):
        plan, rows = fixture()
        original = run(plan, rows)['comparisons']['C_vs_A']
        for row in rows:
            if row['arm'] == 'B':
                row.update(quality_score=0, correct_critical_facts=[], latency_ms=1000)
        result = run(plan, rows)
        self.assertEqual(result['comparisons']['C_vs_A'], original)
        self.assertFalse(result['comparisons']['B_vs_A']['meets_numeric_gates'])
        self.assertEqual(result['diagnostic_only'], ['B_vs_A'])
        next(r for r in rows if r['arm'] == 'C')['correct_critical_facts'] = []
        self.assertFalse(run(plan, rows)['primary_meets_numeric_gates'])

    def test_missing_duplicate_extra_and_binding_fail_closed(self):
        plan, rows = fixture()
        for bad in (rows[:-1], rows+[rows[0]], rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):run(plan, bad)
        for key, value in (('arm', 'D'), ('task_id', 'unknown'), ('input_sha256', 'f'*64),
                           ('plan_sha256', 'f'*64), ('configuration_sha256', 'f'*64), ('latency_ms', -1),
                           ('quality_score', float('nan'))):
            bad = copy.deepcopy(rows); bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):run(plan, bad)
        with self.assertRaises(ValueError):
            compare_three_arm(plan, rows, expected_plan_sha256='f'*64, expected_rows_sha256=plan_digest(rows))
        with self.assertRaises(ValueError):
            compare_three_arm(plan, rows, expected_plan_sha256=plan_digest(plan), expected_rows_sha256='f'*64)

    def test_frozen_sampling_and_exact_schema(self):
        plan, rows = fixture()
        for key, value in (('period_ms', 999), ('period_ms', True), ('phase_ms', 1000), ('phase_ms', -1), ('time_origin_policy', 'implicit'), ('audio_policy', '')):
            bad = copy.deepcopy(plan); bad['baseline_sampling'][key] = value
            with self.assertRaises(ValueError):project_pair_plan(bad, 'C', expected_plan_sha256=plan_digest(bad))
        for mutate in (lambda p:p.update(primary_comparison='B_vs_A'), lambda p:p['systems']['A'].update(sampling='1fps allegedly'),
                       lambda p:p.update(runner={}), lambda p:p['systems'].update(D={})):
            bad = copy.deepcopy(plan); mutate(bad)
            with self.assertRaises(ValueError):project_pair_plan(bad, 'C', expected_plan_sha256=plan_digest(bad))
        epoch = copy.deepcopy(plan); epoch['baseline_sampling']['time_origin_policy'] = 'capture_epoch'
        derived = project_pair_plan(epoch, 'C', expected_plan_sha256=plan_digest(epoch))
        self.assertEqual(derived['three_arm_origin']['baseline_sampling']['time_origin_policy'], 'capture_epoch')
        self.assertNotEqual(plan_digest(derived), plan_digest(project_pair_plan(plan, 'C', expected_plan_sha256=plan_digest(plan))))
        missing = copy.deepcopy(plan); del missing['baseline_sampling']['time_origin_policy']
        with self.assertRaises(ValueError):project_pair_plan(missing, 'C', expected_plan_sha256=plan_digest(missing))
        with self.assertRaises(ValueError):project_pair_plan(plan, 'A', expected_plan_sha256=plan_digest(plan))
        bad = copy.deepcopy(rows); bad[0]['receipt_verified'] = True
        with self.assertRaises(ValueError):run(plan, bad)

    def test_unknown_synthetic_resource_and_sample_gate(self):
        for evidence, mode, expected in (('synthetic', 'priced', False), ('measured', 'unknown', False), ('measured', 'resource', True)):
            plan, rows = fixture(evidence, mode)
            self.assertEqual(run(plan, rows)['primary_meets_numeric_gates'], expected)
        plan, rows = fixture(); plan['comparison']['min_pairs'] = 21
        for row in rows:row['plan_sha256'] = plan_digest(plan)
        self.assertFalse(run(plan, rows)['primary_meets_numeric_gates'])

    def test_itemized_cost_keeps_one_summary_per_arm(self):
        plan, rows = fixture()
        plan['comparison']['cost'] = dict(mode='itemized', currency='USD', scope='full_pipeline',
            accounting_protocol='fixture complete quantity declaration, not receipts',
            rate_card={'gpu': dict(provider='local', revision='fixture', unit='ms', category='local_compute', per_unit=.001)})
        for row in rows:
            row['plan_sha256'] = plan_digest(plan)
            row['cost_quantities'] = {'gpu': 100 if row['arm'] == 'A' else 20}
        result = run(plan, rows)
        self.assertEqual(result['summaries']['A']['cost'], '2.000')
        self.assertEqual(result['summaries']['C']['cost'], '0.400')
        self.assertFalse(result['receipts_verified'])
        rows[0]['cost_quantities'] = {}
        with self.assertRaises(ValueError):run(plan, rows)

    def test_json_duplicate_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path = Path(directory)/'bad.json'
            for content in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
                path.write_text(content)
                with self.assertRaises(ValueError):_read(path)

    def test_cli_reads_frozen_files_without_modifying_them(self):
        plan, rows = fixture('synthetic')
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root = Path(directory)
            (root/'plan.json').write_text(json.dumps(plan)); (root/'rows.json').write_text(json.dumps(rows))
            completed = subprocess.run([sys.executable, '-m', 'benchmarks.three_arm_comparison', '--plan', str(root/'plan.json'),
                '--rows', str(root/'rows.json'), '--plan-sha256', plan_digest(plan), '--rows-sha256', plan_digest(rows)],
                text=True, capture_output=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result['actual_acceptance'], 'incomplete')
            self.assertEqual(json.loads((root/'rows.json').read_text()), rows)
