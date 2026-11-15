"""Strict upstream #270 declarations. Validation never constructs a backend."""
from dataclasses import asdict, dataclass
import ipaddress
import json
import re
from urllib.parse import urlsplit

from gamst.capabilities import identifier
from gamst.call_policy import CallPolicy
from .recorded_source import _keys, _integer, digest, encoded, validate_manifest
from .replay_clock import validate_tasks
from .three_arm_comparison import validate_three_arm_plan


N = 2**63-1
ARMS = ('A', 'B', 'C')
VIEW = 'exclude_sdk_evaluation_snapshots_v1'
LIMITS = dict(max_owned_media_bytes=268435456, max_snapshot_bytes=32768,
    max_snapshot_artifact_bytes=4194304, max_answer_bytes=8192,
    max_control_frame_bytes=65536, max_control_queue_frames=64,
    max_control_queue_bytes=1048576, max_tick_rows_per_run=16384,
    max_task_rows_per_run=64, max_runtime_metadata_bytes_per_run=16777216)


def sha(value, *, nullable=False):
    if value is None and nullable: return None
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('lowercase SHA256 required')
    return value


def text(value, maximum, *, printable=False):
    if type(value) is not str or not 1 <= len(value.encode('utf-8')) <= maximum:
        raise ValueError('bounded nonempty UTF-8 text required')
    if printable and any(not ' ' <= c <= '~' for c in value):
        raise ValueError('printable ASCII required')
    return value


def enum(value, choices):
    if type(value) is not str or value not in choices:
        raise ValueError('unknown recording runtime enum')
    return value


def boolean(value):
    if type(value) is not bool: raise ValueError('exact boolean required')
    return value


def rows(value, maximum, minimum=0):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise ValueError('bounded runtime list required')
    return value


def strict_json(payload, *, maximum=65536):
    """Bound raw bytes, duplicate keys and finite/tree work before conversion."""
    if type(payload) is not bytes or not 1 <= len(payload) <= maximum:
        raise ValueError('bounded runtime JSON bytes required')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate runtime JSON key')
            result[key] = value
        return result
    def invalid(_): raise ValueError('nonfinite runtime JSON number')
    def integer(token):
        if len(token.lstrip('-')) > 19: raise ValueError('runtime integer lexeme exceeds N63')
        number = int(token)
        if not -N <= number <= N: raise ValueError('runtime integer exceeds N63')
        return number
    try:
        value = json.loads(payload.decode('utf-8'), object_pairs_hook=pairs,
                           parse_constant=invalid, parse_int=integer)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('strict UTF-8 runtime JSON required') from error
    # Upstream metadata is <=64KiB, not the SDK sensory depth12 contract.
    remaining = [65536]
    def walk(item, depth=0):
        remaining[0] -= 1
        if depth > 32 or remaining[0] < 0: raise ValueError('runtime JSON tree exceeds bound')
        if type(item) is dict:
            if len(item) > 1024: raise ValueError('runtime object exceeds bound')
            for key, child in item.items():
                text(key, 256); walk(child, depth+1)
        elif type(item) is list:
            if len(item) > 4096: raise ValueError('runtime array exceeds bound')
            for child in item: walk(child, depth+1)
        elif type(item) is str:
            if len(item.encode('utf-8')) > maximum: raise ValueError('runtime string exceeds bound')
        elif type(item) is float:
            import math
            if not math.isfinite(item): raise ValueError('finite runtime number required')
        elif item is not None and type(item) not in (bool, int):
            raise ValueError('strict runtime JSON required')
    walk(value)
    return value


def detached(value, maximum=65536):
    return strict_json(encoded(value), maximum=maximum)


def validate_latency_policy(value, comparison_sha256):
    expected = dict(schema_version=1, kind='recording_latency_policy',
        comparison_plan_sha256=sha(comparison_sha256), clock='same_host_monotonic_ns',
        receive_boundary='validated_result_acceptance',
        failure_policy='max_deadline_and_later_observed_failure',
        late_answer='retain_as_failed', snapshot_policy='single_original_snapshot')
    if type(value) is not dict or value != expected or type(value.get('schema_version')) is not int:
        raise ValueError('exact L1 latency policy required')
    return detached(value)


def validate_endpoint(value):
    text(value, 2048)
    parsed = urlsplit(value)
    if (parsed.scheme not in ('http', 'https') or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path != '/v1/chat/completions'):
        raise ValueError('explicit local chat-completions endpoint required')
    try:
        host = ipaddress.ip_address(parsed.hostname)
        port = parsed.port
    except (ValueError, TypeError) as error:
        raise ValueError('literal loopback endpoint required') from error
    if not host.is_loopback or (port is not None and not 1 <= port <= 65535):
        raise ValueError('literal loopback endpoint required')
    return value


def validate_adapter(value):
    _keys(value, ('adapter_id', 'implementation', 'evidence_class', 'provider', 'model_id',
        'model_revision', 'implementation_sha256', 'dependency_inventory_sha256', 'config'))
    for key in ('adapter_id', 'provider', 'model_revision'): identifier(value[key])
    text(value['model_id'], 256, printable=True)
    for key in ('implementation_sha256', 'dependency_inventory_sha256'): sha(value[key])
    implementation = enum(value['implementation'], ('fixture_completion_v1',
        'local_chat_completions_v1', 'smolvlm_process_v1'))
    config = value['config']
    if implementation == 'fixture_completion_v1':
        if value['evidence_class'] != 'fixture': raise ValueError('fixture provenance required')
        _keys(config, ('responses_sha256', 'delay_ms', 'settlement'))
        sha(config['responses_sha256']); _integer(config['delay_ms'], 0, 500)
        enum(config['settlement'], ('unknown', 'synthetic_explicit'))
    elif implementation == 'local_chat_completions_v1':
        if value['evidence_class'] != 'real': raise ValueError('real client provenance required')
        _keys(config, ('endpoint', 'api_key_env', 'timeout_ms', 'response_bytes',
            'temperature_millionths', 'retries', 'redirects', 'trust_env',
            'server_revision_sha256', 'pricing_identity_sha256'))
        validate_endpoint(config['endpoint'])
        if config['api_key_env'] is not None: identifier(config['api_key_env'])
        _integer(config['timeout_ms'], 10, 300000); _integer(config['response_bytes'], 1, 1048576)
        _integer(config['temperature_millionths'], 0, 0); _integer(config['retries'], 0, 0)
        if config['redirects'] is not False or config['trust_env'] is not False:
            raise ValueError('redirects/environment proxy disabled explicitly')
        sha(config['server_revision_sha256']); sha(config['pricing_identity_sha256'], nullable=True)
    else:
        if value['evidence_class'] != 'real': raise ValueError('real process provenance required')
        _keys(config, ('python_asset_sha256', 'worker_config_sha256', 'startup_timeout_ms', 'inference_timeout_ms'))
        sha(config['python_asset_sha256']); sha(config['worker_config_sha256'])
        _integer(config['startup_timeout_ms'], 1, 120000); _integer(config['inference_timeout_ms'], 10, 300000)
    return detached(value)


def validate_delivery(value):
    _keys(value, ('source_id', 'profile', 'config_sha256', 'binding_id'))
    identifier(value['source_id'])
    profile = enum(value['profile'], ('acquire_only', 'pcm_level', 'fixture_bgr', 'opencv_lk', 'opencv_fill'))
    if profile == 'acquire_only':
        if value['config_sha256'] is not None or value['binding_id'] is not None:
            raise ValueError('acquire-only cannot hide a provider binding/configuration')
    else:
        sha(value['config_sha256']); identifier(value['binding_id'])
    return value


def validate_fixture_responses(value):
    value = detached(value, 1048576)
    _keys(value, ('schema_version', 'kind', 'rows'))
    _integer(value['schema_version'], 1, 1)
    if value['kind'] != 'fixture_completions': raise ValueError('fixed fixture response inventory required')
    seen = set()
    for row in rows(value['rows'], 16384):
        _keys(row, ('invocation_ordinal', 'lane_id', 'call_ordinal', 'text', 'usage', 'synthetic_receipt_sha256'))
        _integer(row['invocation_ordinal'], 0, 95); identifier(row['lane_id'])
        _integer(row['call_ordinal'], 0, 16383); text(row['text'], 8192)
        key = (row['invocation_ordinal'], row['lane_id'], row['call_ordinal'])
        if key in seen: raise ValueError('duplicate fixture scenario operation')
        seen.add(key); sha(row['synthetic_receipt_sha256'], nullable=True)
        if row['usage'] is not None:
            _keys(row['usage'], ('input_tokens', 'output_tokens', 'cached_input_tokens'))
            for number in row['usage'].values(): _integer(number, 0, 10**12)
            if row['usage']['cached_input_tokens'] > row['usage']['input_tokens']:
                raise ValueError('cached tokens exceed original input tokens')
    return value


def validate_assets(value):
    value = detached(value, 1048576)
    _keys(value, ('schema_version', 'kind', 'assets', 'dependencies', 'resident_state'))
    _integer(value['schema_version'], 1, 1)
    if value['kind'] != 'recording_runtime_assets': raise ValueError('fixed runtime asset inventory required')
    seen = set()
    for asset in rows(value['assets'], 256):
        _keys(asset, ('asset_id', 'kind', 'path', 'sha256', 'bytes'))
        identifier(asset['asset_id'])
        if asset['asset_id'] in seen: raise ValueError('duplicate runtime asset')
        seen.add(asset['asset_id'])
        enum(asset['kind'], ('python', 'checkpoint', 'worker_config', 'provider_config',
                             'fixture_responses', 'system_template', 'rule'))
        path = text(asset['path'], 4096)
        if '\x00' in path: raise ValueError('NUL asset path rejected')
        sha(asset['sha256']); _integer(asset['bytes'], 0, N)
    seen = set()
    for dependency in rows(value['dependencies'], 256):
        _keys(dependency, ('distribution', 'version', 'installation_sha256'))
        for key in ('distribution', 'version'): identifier(dependency[key])
        if dependency['distribution'] in seen: raise ValueError('duplicate runtime dependency')
        seen.add(dependency['distribution']); sha(dependency['installation_sha256'])
        if dependency['version'] in ('installed', 'latest'): raise ValueError('actual effective version required')
    seen = set()
    for resident in rows(value['resident_state'], 256):
        _keys(resident, ('resource_id', 'state', 'evidence_sha256'))
        identifier(resident['resource_id'])
        if resident['resource_id'] in seen: raise ValueError('duplicate resource resident state')
        seen.add(resident['resource_id']); enum(resident['state'], ('absent', 'resident', 'unknown'))
        sha(resident['evidence_sha256'], nullable=True)
    return value


def validate_profiles_v2(value, manifest, policies):
    manifest = validate_manifest(manifest)
    value = detached(value)
    _keys(value, ('schema_version', 'kind', 'recording_id', 'source_manifest_sha256', 'arms'))
    _integer(value['schema_version'], 2, 2)
    if (value['kind'] != 'recording_runtime_profiles' or value['recording_id'] != manifest['recording_id']
            or value['source_manifest_sha256'] != digest(manifest)):
        raise ValueError('profile-v2 original recording binding mismatch')
    entries = rows(value['arms'], 3, 3)
    if [item.get('arm') for item in entries] != list(ARMS): raise ValueError('exact ordered ABC profiles required')
    sources = {source['source_id']: source for source in manifest['sources']}
    for item in entries:
        _keys(item, ('arm', 'deliveries'))
        deliveries = rows(item['deliveries'], 16, len(sources))
        if len(deliveries) != len(sources): raise ValueError('one delivery per original source required')
        seen = set()
        for delivery in deliveries:
            validate_delivery(delivery); sid = delivery['source_id']
            if sid not in sources or sid in seen: raise ValueError('unknown/duplicate profile source')
            seen.add(sid); source = sources[sid]
            if (source['format'] == 'pcm_s16le_mono') != (delivery['profile'] in ('pcm_level', 'acquire_only')):
                # acquire_only is legal for either modality.
                if delivery['profile'] != 'acquire_only': raise ValueError('profile modality mismatch')
            if item['arm'] == 'A' and source['modality'] == 'video' and delivery['profile'] != 'acquire_only':
                raise ValueError('A has no specialist vision')
        if deliveries != policies[item['arm']]['deliveries']:
            raise ValueError('executed profile differs from frozen arm policy')
    for source in manifest['sources']:
        if source['modality'] == 'audio':
            definitions = [next(d for d in arm['deliveries'] if d['source_id'] == source['source_id']) for arm in entries]
            # Binding identities may differ by arm; behavior/configuration cannot.
            if len({(d['profile'], d['config_sha256']) for d in definitions}) != 1:
                raise ValueError('ABC audio policy must be identical')
    return value


def _periodic(value, lane_ids, baseline=None):
    _keys(value, ('scheduler', 'lane_id', 'period_ns', 'phase_ns', 'time_origin_policy',
        'max_age_ms', 'deadline_tolerance_ns', 'prompt_policy_sha256'))
    enum(value['scheduler'], ('recorded_visual_v1', 'smolvlm_sentinel_v1'))
    if value['lane_id'] not in lane_ids: raise ValueError('unknown periodic lane')
    period = _integer(value['period_ns'], 1, N)
    _integer(value['phase_ns'], 0, period-1); _integer(value['deadline_tolerance_ns'], 0, period-1)
    enum(value['time_origin_policy'], ('recording_start', 'capture_epoch'))
    _integer(value['max_age_ms'], 1, 60000); sha(value['prompt_policy_sha256'])
    if baseline is not None and (period != 10**9 or value['phase_ns'] != baseline['phase_ms']*10**6
            or value['time_origin_policy'] != baseline['time_origin_policy']):
        raise ValueError('A must preserve original frozen 1Hz capture schedule')


def _fill(value, lanes):
    _keys(value, ('implementation', 'source_id', 'selection_period_ns', 'phase_ns', 'max_lateness_ns',
        'fill_config_sha256', 'rule_sha256', 'small_lane_id', 'strong_lane_id', 'strong_opt_in',
        'charge_ceilings', 'chain_ceiling', 'pricing_revision', 'unit', 'stage_timeout_ms', 'retention_ms'))
    if value['implementation'] != 'fill_conflict_v1': raise ValueError('only actual supported fill rule allowed')
    identifier(value['source_id']); period = _integer(value['selection_period_ns'], 1, N)
    _integer(value['phase_ns'], 0, period-1); _integer(value['max_lateness_ns'], 0, period-1)
    sha(value['fill_config_sha256'])
    from gamst.fill_conflict_rule import RULE_SHA
    if value['rule_sha256'] != RULE_SHA: raise ValueError('unsupported fill predicate/rule')
    if value['small_lane_id'] not in lanes or lanes[value['small_lane_id']]['role'] != 'fill_small':
        raise ValueError('explicit fill-small lane required')
    boolean(value['strong_opt_in'])
    strong = value['strong_lane_id']
    if value['strong_opt_in'] != (strong is not None): raise ValueError('explicit strong lane opt-in required')
    if strong is not None:
        if strong not in lanes or lanes[strong]['role'] != 'fill_strong': raise ValueError('explicit strong lane required')
        if lanes[strong]['controller_id'] != lanes[value['small_lane_id']]['controller_id']:
            raise ValueError('fill small/strong must share actual escalation controller')
    _keys(value['charge_ceilings'], ('initial', 'reinspection', 'strong'))
    for amount in value['charge_ceilings'].values():
        if amount is not None: _integer(amount, 0, 1_000_000_000_000)
    _integer(value['chain_ceiling'], 0, 3_000_000_000_000)
    if sum(x for x in value['charge_ceilings'].values() if x is not None)>value['chain_ceiling']:
        raise ValueError('fill stage authorizations exceed original chain ceiling')
    identifier(value['pricing_revision']); identifier(value['unit'])
    _integer(value['stage_timeout_ms'], 10, 10000); _integer(value['retention_ms'], 1, 60000)


@dataclass(frozen=True, slots=True)
class ValidatedRuntimePolicy:
    canonical: bytes

    @property
    def sha256(self): return digest(self.document())

    def document(self): return strict_json(self.canonical)


def validate_runtime_policy(value, *, comparison_plan, latency_policy, source_bundles):
    """Validate U1 plus all original manifests/tasks/profile-v2 declarations.

    Configuration/asset bytes and C/Q/I bindings receive a separate mandatory
    preflight before activation; this typed value alone is not spawn authority.
    U2 additionally requires the independently validated E2/K2 numeric contract.
    """
    value = detached(value)
    _keys(value, ('schema_version', 'kind', 'comparison_plan_sha256', 'latency_policy_sha256',
        'accounting_policy_sha256', 'numeric_policy_sha256', 'evaluation_mode', 'source_context_view',
        'recordings', 'adapters', 'arms', 'limits'))
    version = _integer(value['schema_version'], 1, 2)
    if value['kind'] != 'recording_runtime_policy' or value['source_context_view'] != VIEW:
        raise ValueError('explicit SDK source Context view required')
    plan = validate_three_arm_plan(comparison_plan, expected_plan_sha256=value['comparison_plan_sha256'])
    validate_latency_policy(latency_policy, value['comparison_plan_sha256'])
    if value['latency_policy_sha256'] != digest(latency_policy): raise ValueError('L1 binding mismatch')
    sha(value['accounting_policy_sha256'], nullable=version == 1)
    if version == 1:
        if value['numeric_policy_sha256'] is not None or value['evaluation_mode'] != 'structural_only':
            raise ValueError('U1 cannot activate numeric reconciliation')
    else:
        sha(value['numeric_policy_sha256'])
        if value['evaluation_mode'] != 'matched_source_exposure_full_lifetime_v2': raise ValueError('U2 numeric mode required')
    _keys(value['limits'], LIMITS)
    for key, maximum in LIMITS.items():
        _integer(value['limits'][key], 1, maximum)
        if key != 'max_owned_media_bytes' and value['limits'][key] != maximum:
            raise ValueError('fixed runtime limits cannot change')
    adapters = {}
    for adapter in rows(value['adapters'], 64, 1):
        checked = validate_adapter(adapter)
        if checked['adapter_id'] in adapters: raise ValueError('duplicate adapter identity')
        adapters[checked['adapter_id']] = checked
    recordings = rows(value['recordings'], 32, 1)
    if type(source_bundles) is not dict or set(source_bundles) != {r.get('recording_id') for r in recordings}:
        raise ValueError('all original source bundles required')
    original_tasks = {}; policies = {}; manifests = {}; declarations = {}
    for recording in recordings:
        _keys(recording, ('recording_id', 'source_manifest_sha256', 'tasks_sha256', 'profiles_sha256',
            'primary_visual_source_id', 'subject_id', 'source_bindings'))
        rid = identifier(recording['recording_id']); identifier(recording['subject_id'])
        if rid in declarations: raise ValueError('duplicate original recording')
        _keys(source_bundles[rid], ('manifest', 'tasks', 'profiles'))
        bundle = source_bundles[rid]; manifest = validate_manifest(bundle['manifest'])
        tasks = validate_tasks(bundle['tasks'], manifest['duration_ns'])
        if manifest['recording_id'] != rid: raise ValueError('original recording identity mismatch')
        for key, document in (('source_manifest_sha256', manifest), ('tasks_sha256', tasks), ('profiles_sha256', bundle['profiles'])):
            if recording[key] != digest(document): raise ValueError('original recording declaration digest mismatch')
        sources = {s['source_id']: s for s in manifest['sources']}
        primary = recording['primary_visual_source_id']
        if primary not in sources or sources[primary]['format'] != 'bgr8_hwc': raise ValueError('selected original BGR source required')
        if any(min(r['shape'][:2]) < 32 for r in manifest['records'] if r['source_id'] == primary):
            raise ValueError('runtime visual adapters require at least32x32')
        bindings = rows(recording['source_bindings'], 16, len(sources)); seen = set()
        for binding in bindings:
            _keys(binding, ('source_id', 'coordinate_frame', 'channels'))
            sid = binding['source_id']
            if sid not in sources or sid in seen: raise ValueError('exact original source bindings required')
            seen.add(sid)
            if binding['coordinate_frame'] is not None: identifier(binding['coordinate_frame'])
            if binding['channels'] != (['B', 'G', 'R'] if sources[sid]['modality'] == 'video' else None):
                raise ValueError('explicit original BGR/PCM channel metadata required')
        if seen != set(sources): raise ValueError('missing original source binding')
        for task in tasks:
            _keys(task['input'], ('question',)); text(task['input']['question'], 4096)
            if task['task_id'] in original_tasks: raise ValueError('original task belongs to exactly one recording')
            original_tasks[task['task_id']] = task['input_sha256']
        declarations[rid] = recording; manifests[rid] = manifest; policies[rid] = {}
    plan_tasks = {t['id']: t['input_sha256'] for t in plan['comparison']['tasks']}
    if original_tasks != plan_tasks or len(original_tasks) > 1000: raise ValueError('unchanged complete original plan tasks required')
    arms = rows(value['arms'], 3, 3)
    if [arm.get('arm') for arm in arms] != list(ARMS): raise ValueError('exact ordered ABC arms required')
    answer_identity = []; audio_identity = {}
    for arm in arms:
        _keys(arm, ('arm', 'recording_policies', 'controllers', 'lanes', 'task_queue'))
        controllers = {}
        for controller in rows(arm['controllers'], 8, 1):
            _keys(controller, ('controller_id', 'policy', 'budget_sha256', 'max_requests'))
            cid = identifier(controller['controller_id'])
            if cid in controllers: raise ValueError('duplicate controller')
            _keys(controller['policy'], asdict(CallPolicy()))
            CallPolicy(**controller['policy']); sha(controller['budget_sha256'])
            _integer(controller['max_requests'], 1, 256); controllers[cid] = controller
        lanes = {}; bindings = set()
        for lane in rows(arm['lanes'], 8, 1):
            _keys(lane, ('lane_id', 'role', 'adapter_id', 'controller_id', 'maximum_charge', 'ttl_ms', 'binding_id'))
            lid = identifier(lane['lane_id']); identifier(lane['binding_id'])
            if lid in lanes or lane['binding_id'] in bindings: raise ValueError('unique immutable semantic lane binding required')
            if lane['adapter_id'] not in adapters or lane['controller_id'] not in controllers: raise ValueError('unknown lane adapter/controller')
            enum(lane['role'], ('periodic_small', 'fill_small', 'fill_strong', 'text_answer'))
            if lane['maximum_charge'] is not None: _integer(lane['maximum_charge'], 0, N)
            _integer(lane['ttl_ms'], 1, 3600000)
            lanes[lid] = lane; bindings.add(lane['binding_id'])
        queue = arm['task_queue']; _keys(queue, ('capacity', 'assignment', 'overflow', 'late'))
        _integer(queue['capacity'], 0, 64)
        if (queue['assignment'], queue['overflow'], queue['late']) != ('earliest_deadline_then_frozen_task_index', 'refuse', 'fail_without_refresh'):
            raise ValueError('fixed bounded task scheduling required')
        seen = set(); used_lanes = set()
        for policy in rows(arm['recording_policies'], 32, len(declarations)):
            _keys(policy, ('recording_id', 'deliveries', 'periodic', 'fill_chain', 'text_lane_ids'))
            rid = policy['recording_id']
            if rid not in declarations or rid in seen: raise ValueError('one policy per original recording')
            seen.add(rid); policies[rid][arm['arm']] = policy
            for delivery in rows(policy['deliveries'], 16, 1): validate_delivery(delivery)
            text_ids = rows(policy['text_lane_ids'], 8, 1)
            if len(set(text_ids)) != len(text_ids): raise ValueError('duplicate answer lane')
            for lid in text_ids:
                if lid not in lanes or lanes[lid]['role'] != 'text_answer': raise ValueError('text-answer lane required')
                used_lanes.add(lid)
                lane = lanes[lid]
                if adapters[lane['adapter_id']]['implementation']=='smolvlm_process_v1':raise ValueError('Smol is a fixed visual perception worker, not a text answer consumer')
                controller = controllers[lane['controller_id']]
                answer_identity.append((adapters[lane['adapter_id']], controller['policy'], controller['budget_sha256'],
                                        lane['maximum_charge'], lane['ttl_ms']))
            periodic = policy['periodic']; fill = policy['fill_chain']
            if arm['arm'] == 'B':
                if periodic is not None or fill is not None: raise ValueError('B has specialists only')
            else:
                if periodic is None: raise ValueError('A/C require explicit periodic perception')
                _periodic(periodic, lanes, plan['baseline_sampling'] if arm['arm'] == 'A' else None)
                is_smol=adapters[lanes[periodic['lane_id']]['adapter_id']]['implementation']=='smolvlm_process_v1'
                if is_smol!=(periodic['scheduler']=='smolvlm_sentinel_v1'):
                    raise ValueError('Smol and Visual periodic drivers are distinct explicit profiles')
                if lanes[periodic['lane_id']]['role'] != 'periodic_small': raise ValueError('periodic-small lane required')
                used_lanes.add(periodic['lane_id'])
            if arm['arm'] == 'A' and fill is not None: raise ValueError('A cannot trigger conditional fill inference')
            if fill is not None:
                _fill(fill, lanes)
                if controllers[lanes[fill['small_lane_id']]['controller_id']]['policy']['mode']!='escalation':
                    raise ValueError('fill requires actual escalation controller policy')
                if any(adapters[lanes[lid]['adapter_id']]['implementation']=='smolvlm_process_v1' for lid in
                        (fill['small_lane_id'],fill['strong_lane_id']) if lid is not None):
                    raise ValueError('fill JSON question stages require VisualContextCalls compatible client')
                if fill['source_id'] != declarations[rid]['primary_visual_source_id']: raise ValueError('fill must use selected original visual source')
                used_lanes.add(fill['small_lane_id'])
                if fill['strong_lane_id'] is not None: used_lanes.add(fill['strong_lane_id'])
        if seen != set(declarations) or used_lanes != set(lanes): raise ValueError('unused/missing runtime lane or recording')
    if any(identity != answer_identity[0] for identity in answer_identity):
        raise ValueError('ABC text consumers must have identical adapter/model/policy/charge semantics')
    for rid, bundle in source_bundles.items(): validate_profiles_v2(bundle['profiles'], manifests[rid], policies[rid])
    return ValidatedRuntimePolicy(encoded(value))


def validate_visual_prompt(value,expected_sha256):
    _keys(value,('schema_version','kind','question'))
    _integer(value['schema_version'],1,1)
    if value['kind']!='recording_visual_prompt' or digest(value)!=expected_sha256:
        raise ValueError('original finite visual prompt policy required')
    text(value['question'],4096)
    if not value['question'].strip():raise ValueError('nonempty original visual question required')
    return value['question']
