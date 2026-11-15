import copy
import unittest

from benchmarks.recorded_source import digest
from benchmarks.recording_runtime_contract import (LIMITS, VIEW, validate_runtime_policy,
    validate_latency_policy, validate_adapter, validate_endpoint, validate_fixture_responses,
    strict_json)
from gamst.call_policy import CallPolicy
from dataclasses import asdict
from test_three_arm_comparison import fixture as plan_fixture


def fixture():
    plan, _ = plan_fixture('synthetic')
    tasks = []
    for i, task in enumerate(plan['comparison']['tasks']):
        question = {'question': 'Original question '+str(i)}
        task['input_sha256'] = digest(question)
        tasks.append(dict(task_id=task['id'], input=question, input_sha256=digest(question),
            opens_ns=0, snapshot_ns=(i+1)*1000000, deadline_ns=90000000, window='1h'))
    manifest = dict(schema_version=1, recording_id='original', origin_capture_ns=1000000000,
        duration_ns=100000000, sources=[dict(source_id='camera', sensor_id='video',
            stream_epoch='old', format='bgr8_hwc', modality='video', calibration_id=None)],
        records=[dict(source_id='camera', sensor_id='video', sequence=1, capture_ns=1000000000,
            offset_ns=0, format='bgr8_hwc', stream_epoch='old', payload_path='frame.bin',
            payload_bytes=32*32*3, payload_sha256='a'*64, shape=[32,32,3])],
        availability=[dict(source_id='camera', intervals=[dict(start_ns=0, end_ns=100000000,
                                                              state='available')])])
    latency = dict(schema_version=1, kind='recording_latency_policy',
        comparison_plan_sha256=digest(plan), clock='same_host_monotonic_ns',
        receive_boundary='validated_result_acceptance',
        failure_policy='max_deadline_and_later_observed_failure', late_answer='retain_as_failed',
        snapshot_policy='single_original_snapshot')
    adapter = dict(adapter_id='fixture', implementation='fixture_completion_v1',
        evidence_class='fixture', provider='test.fixture', model_id='test/model', model_revision='1',
        implementation_sha256='a'*64, dependency_inventory_sha256='b'*64,
        config=dict(responses_sha256='c'*64, delay_ms=0, settlement='unknown'))
    arms = []; profile_arms = []
    for arm in 'ABC':
        delivery = dict(source_id='camera', profile='acquire_only' if arm == 'A' else 'fixture_bgr',
            config_sha256=None if arm == 'A' else 'd'*64, binding_id=None if arm == 'A' else 'vision')
        periodic = None if arm == 'B' else dict(scheduler='recorded_visual_v1', lane_id='periodic',
            period_ns=1000000000, phase_ns=0, time_origin_policy='recording_start', max_age_ms=1000,
            deadline_tolerance_ns=50000000, prompt_policy_sha256='e'*64)
        lanes = [dict(lane_id='answer', role='text_answer', adapter_id='fixture', controller_id='main',
                      maximum_charge=10, ttl_ms=1000, binding_id='text')]
        if periodic is not None:
            lanes.append(dict(lane_id='periodic', role='periodic_small', adapter_id='fixture',
                controller_id='main', maximum_charge=10, ttl_ms=1000, binding_id='visual'))
        arms.append(dict(arm=arm, recording_policies=[dict(recording_id='original', deliveries=[delivery],
            periodic=periodic, fill_chain=None, text_lane_ids=['answer'])],
            controllers=[dict(controller_id='main', policy=asdict(CallPolicy(mode='escalation')),
                              budget_sha256='f'*64, max_requests=256)], lanes=lanes,
            task_queue=dict(capacity=64, assignment='earliest_deadline_then_frozen_task_index',
                            overflow='refuse', late='fail_without_refresh')))
        profile_arms.append(dict(arm=arm, deliveries=[copy.deepcopy(delivery)]))
    profiles = dict(schema_version=2, kind='recording_runtime_profiles', recording_id='original',
                    source_manifest_sha256=digest(manifest), arms=profile_arms)
    policy = dict(schema_version=1, kind='recording_runtime_policy', comparison_plan_sha256=digest(plan),
        latency_policy_sha256=digest(latency), accounting_policy_sha256=None, numeric_policy_sha256=None,
        evaluation_mode='structural_only', source_context_view=VIEW,
        recordings=[dict(recording_id='original', source_manifest_sha256=digest(manifest),
            tasks_sha256=digest(tasks), profiles_sha256=digest(profiles), primary_visual_source_id='camera',
            subject_id='local-user', source_bindings=[dict(source_id='camera', coordinate_frame=None,
                                                         channels=['B','G','R'])])],
        adapters=[adapter], arms=arms, limits=copy.deepcopy(LIMITS))
    bundles = {'original': dict(manifest=manifest, tasks=tasks, profiles=profiles)}
    return policy, plan, latency, bundles


class RuntimeContract(unittest.TestCase):
    def validate(self, case):
        policy, plan, latency, bundles = case
        return validate_runtime_policy(policy, comparison_plan=plan, latency_policy=latency,
                                       source_bundles=bundles)

    def test_complete_original_binding_and_detached_typed_result(self):
        case = fixture(); validated = self.validate(case)
        self.assertEqual(validated.sha256, digest(case[0]))
        detached = validated.document(); detached['limits']['max_answer_bytes'] = 1
        self.assertEqual(validated.document(), case[0])
        self.assertEqual(validate_latency_policy(case[2], digest(case[1])), case[2])

    def test_exact_schema_baseline_and_controller_admission_refusals(self):
        mutations = [
            lambda u: u.update(extra=True),
            lambda u: u.update(source_context_view='arbitrary_filter'),
            lambda u: u['limits'].update(max_answer_bytes=8193),
            lambda u: u['arms'][0]['controllers'][0].update(max_requests=257),
            lambda u: u['arms'][0]['recording_policies'][0]['periodic'].update(phase_ns=1),
            lambda u: u['arms'][1]['recording_policies'][0].update(periodic=copy.deepcopy(u['arms'][0]['recording_policies'][0]['periodic'])),
            lambda u: u['arms'][2]['lanes'][0].update(maximum_charge=9),
            lambda u: u['recordings'][0].update(tasks_sha256='0'*64),
        ]
        for mutation in mutations:
            case = fixture(); mutation(case[0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.validate(case)

    def test_explicit_local_adapter_no_environment_or_remote_url(self):
        good = ('http://127.0.0.1:8080/v1/chat/completions', 'https://[::1]/v1/chat/completions')
        for endpoint in good: self.assertEqual(validate_endpoint(endpoint), endpoint)
        for endpoint in ('http://localhost/v1/chat/completions', 'http://example.com/v1/chat/completions',
                         'http://127.0.0.1/api/chat', 'http://user@127.0.0.1/v1/chat/completions',
                         'http://127.0.0.1/v1/chat/completions?model=x'):
            with self.assertRaises(ValueError): validate_endpoint(endpoint)
        adapter = fixture()[0]['adapters'][0]
        adapter.update(implementation='local_chat_completions_v1', evidence_class='real', config=dict(
            endpoint=good[0], api_key_env=None, timeout_ms=100, response_bytes=8192,
            temperature_millionths=0, retries=0, redirects=False, trust_env=False,
            server_revision_sha256='e'*64, pricing_identity_sha256=None))
        self.assertEqual(validate_adapter(adapter), adapter)
        adapter['config']['retries'] = True
        with self.assertRaises(ValueError): validate_adapter(adapter)

    def test_fixture_rows_have_no_wildcard_fallback_or_duplicate_operation(self):
        row = dict(invocation_ordinal=0, lane_id='answer', call_ordinal=0, text='fixture answer',
                   usage=None, synthetic_receipt_sha256=None)
        value = dict(schema_version=1, kind='fixture_completions', rows=[row])
        self.assertEqual(validate_fixture_responses(value), value)
        value['rows'].append(copy.deepcopy(row))
        with self.assertRaises(ValueError): validate_fixture_responses(value)

    def test_raw_duplicate_nonfinite_and_integer_lexemes_reject(self):
        for payload in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}',
                        b'{"x":9223372036854775808}', b'"\xff"'):
            with self.assertRaises(ValueError): strict_json(payload)


if __name__ == '__main__': unittest.main()
