"""Opt-in frozen Context evidence without feeding evaluation wrappers back in.

These capabilities describe a detached bundle, never a new sensor measurement.
SourceContextView is SDK-issued and applies before Context assembly. Default
Context queries keep their existing behavior.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from .capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from .call_controller import _digest


CAPABILITY_ID = 'evaluation.context_snapshot.v1'
VIEW_NAME = 'exclude_sdk_evaluation_snapshots_v1'
_ISSUER = object()
_HASH_FIELDS = ('input_sha256', 'task_sha256', 'source_manifest_sha256',
                'mapping_sha256', 'context_sha256')
_TIME_FIELDS = ('snapshot_timestamp_ms', 'scheduled_offset_ns',
                'observed_begin_ns', 'observed_end_ns')
_FIELDS = ('kind', 'task_id') + _HASH_FIELDS + _TIME_FIELDS + ('watermarks', 'context')
_WATERMARK_FIELDS = ('source_index', 'last_acquired_ordinal', 'last_submitted_sequence',
                     'processed_sequence', 'published_sequence', 'processed_known', 'published_known')


def evaluation_snapshot_implementation_sha256():
    """The exact installed publisher source identity, also frozen by the runner."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('lowercase SHA256 required')
    return value


def _integer(value, maximum=2**63-1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError('bounded nonnegative integer required')
    return value


def _spec(implementation):
    fields = tuple(OutputField(key, 'string') for key in ('kind', 'task_id') + _HASH_FIELDS)
    fields += tuple(OutputField(key, 'integer') for key in _TIME_FIELDS)
    fields += (OutputField('watermarks', 'array'), OutputField('context', 'object'))
    return CapabilitySpec(CAPABILITY_ID, 'evaluation.context_snapshot', 'events',
        'gamst.recording_snapshot', '1.' + implementation, fields, 'offline',
        freshness_ms=1000, interpretation='estimate', devices=('cpu',),
        sensor_requirements=('evaluation_only_frozen_context',), license='Apache-2.0')


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SourceContextView:
    """Immutable view owned by exactly one SDK publisher/subject/generation."""
    _publisher: object

    def __init__(self, publisher, *, _issuer=None):
        if _issuer is not _ISSUER:
            raise TypeError('SourceContextView must be issued by HumanSense')
        object.__setattr__(self, '_publisher', publisher)

    @property
    def name(self):
        return VIEW_NAME

    def _exclusion(self, human, subject_id):
        publisher = self._publisher
        publisher._check()
        if (publisher._human is not human or publisher.subject_id != subject_id
                or publisher.context_view is not self
                or human._evaluation_snapshot_publishers.get(subject_id) is not publisher):
            raise ValueError('Context view belongs to another host or subject')
        spec = human.perception_capabilities.get(CAPABILITY_ID)
        if _digest(spec.describe()) != publisher.capability_sha256:
            raise ValueError('registered evaluation snapshot contract changed')
        return (spec.id, spec.provider, spec.version)


class EvaluationSnapshotPublisher:
    """One bounded recording task-evidence publisher, never an inference client."""
    __slots__ = ('_human', '_subject_id', '_generation', '_session_id', '_spec',
                 '_capability_sha256', '_view', '_sequence', '_task_ids', '_bytes')

    def __init__(self, human, subject_id, implementation_sha256, *, _issuer=None):
        if _issuer is not _ISSUER:
            raise TypeError('EvaluationSnapshotPublisher must be issued by HumanSense')
        self._human = human
        self._subject_id = subject_id
        self._generation = human._generation
        self._session_id = human._session_id
        self._spec = _spec(implementation_sha256)
        self._capability_sha256 = _digest(self._spec.describe())
        self._view = SourceContextView(self, _issuer=_ISSUER)
        self._sequence = self._bytes = 0
        self._task_ids = set()

    @property
    def subject_id(self):
        return self._subject_id

    @property
    def context_view(self):
        return self._view

    @property
    def capability_sha256(self):
        return self._capability_sha256

    def _check(self):
        if self._human is None:
            raise RuntimeError('evaluation snapshot publisher was retired')
        self._human._check()
        if (self._session_id != self._human._session_id
                or self._generation != self._human._generation
                or self._human._evaluation_snapshot_publishers.get(self.subject_id) is not self):
            raise RuntimeError('evaluation snapshot publisher was invalidated')

    def _retire(self):
        # Privacy invalidation is synchronous, including before async cleanup.
        # Externally retained old views remain permanently unusable and do not
        # retain subject IDs, original task IDs or the owning HumanSense graph.
        self._task_ids.clear()
        self._subject_id = self._session_id = self._generation = self._human = None
        self._sequence = self._bytes = 0

    async def publish(self, envelope):
        self._check()
        value = bounded_json(envelope, max_bytes=65536)
        if type(value) is not dict or set(value) != set(_FIELDS):
            raise ValueError('exact frozen Context envelope fields required')
        if value['kind'] != 'frozen_context_bundle_not_new_sensor_measurement':
            raise ValueError('frozen Context bundle kind required')
        identifier(value['task_id'], 'original task ID')
        for key in _HASH_FIELDS:
            _sha(value[key])
        for key in _TIME_FIELDS:
            _integer(value[key], 2**53-1 if key == 'snapshot_timestamp_ms' else 2**63-1)
        if value['observed_end_ns'] < value['observed_begin_ns']:
            raise ValueError('ordered actual snapshot interval required')
        context = value['context']
        if (type(context) is not dict or context.get('type') != 'observation.context'
                or context.get('subject_id') != self.subject_id
                or context.get('timestamp_ms') != value['snapshot_timestamp_ms']
                or _digest(context) != value['context_sha256']):
            raise ValueError('snapshot must preserve its original Context identity/time/digest')
        policy = context.get('context_policy', {})
        if (policy.get('view') != VIEW_NAME
                or policy.get('excluded_capability_sha256') != self.capability_sha256):
            raise ValueError('source Context must use the SDK-issued exclusion view')
        import json
        context_bytes = len(json.dumps(context, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=True, allow_nan=False).encode())
        if context_bytes > 32768:
            raise ValueError('frozen source Context exceeds32768 bytes')
        watermarks = value['watermarks']
        if type(watermarks) is not list or len(watermarks) > 16:
            raise ValueError('bounded source watermarks required')
        seen = set()
        for watermark in watermarks:
            if type(watermark) is not dict or set(watermark) != set(_WATERMARK_FIELDS):
                raise ValueError('exact source watermark required')
            source = _integer(watermark['source_index'], 15)
            if source in seen:
                raise ValueError('duplicate source watermark')
            seen.add(source)
            for key in ('last_acquired_ordinal', 'last_submitted_sequence',
                        'processed_sequence', 'published_sequence'):
                if watermark[key] is not None:
                    _integer(watermark[key])
            for known, sequence in (('processed_known', 'processed_sequence'),
                                    ('published_known', 'published_sequence')):
                if type(watermark[known]) is not bool or (not watermark[known] and watermark[sequence] is not None):
                    raise ValueError('unknown delivery watermark must remain null')
        if (self._sequence >= 64 or value['task_id'] in self._task_ids
                or self._bytes + context_bytes > 4*1024**2):
            raise ValueError('recording snapshot task/byte capacity exhausted')
        # Reserve synchronously before await: concurrent snapshots cannot reuse a
        # task ID/sequence or exceed the recording bound. Failed publication never
        # refunds an original task into a hidden retry.
        self._sequence += 1
        self._bytes += context_bytes
        self._task_ids.add(value['task_id'])
        sensor = SensorMetadata('evaluation.tasks.' + self._session_id, value['task_id'],
            self._sequence, stream_epoch=self._session_id + ':snapshots:' + str(self._generation),
            capture_sequence=self._sequence)
        record = await self._human.push_observation(CAPABILITY_ID, value, sensor=sensor,
            subject_id=self.subject_id, timestamp_ms=value['snapshot_timestamp_ms'])
        self._check()
        # Match the complete record contract consumed by TextContextCalls, not
        # just its observation.value. Do not trim/flatten to fit.
        bounded_json(record, max_bytes=131072)
        return record


def create_snapshot_publisher(human, subject_id, implementation_sha256):
    human._check()
    identifier(subject_id, 'snapshot subject')
    expected = _sha(implementation_sha256)
    if expected != evaluation_snapshot_implementation_sha256():
        raise ValueError('publisher differs from frozen installed implementation')
    prior = human._evaluation_snapshot_publishers.get(subject_id)
    if prior is not None and prior._generation == human._generation:
        prior._check()
        return prior
    if prior is None and len(human._evaluation_snapshot_publishers) >= 16:
        raise ValueError('at most16 evaluation snapshot subjects per host')
    publisher = EvaluationSnapshotPublisher(human, subject_id, expected, _issuer=_ISSUER)
    human.register_capability(publisher._spec)
    human._evaluation_snapshot_publishers[subject_id] = publisher
    return publisher


def context_exclusion(human, subject_id, view):
    if view is None:
        return None
    if type(view) is not SourceContextView:
        raise TypeError('SDK-issued SourceContextView required')
    return view._exclusion(human, subject_id)
