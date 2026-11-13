"""Bounded full-replacement ASR revisions; no model or speech recognition here."""
from collections import OrderedDict
from copy import deepcopy
import json

from .capabilities import CapabilitySpec, OutputField, bounded_json, identifier
from .schema import timestamp

TRANSCRIPT_OUTPUT = 'speech.transcript_update'


def transcript_capability(capability_id, provider, version, *, streaming_mode,
                          window_ms=0, lookahead_ms=0, freshness_ms=5000, license='unspecified'):
    return CapabilitySpec(capability_id, TRANSCRIPT_OUTPUT, 'audio', provider, version,
        (OutputField('utterance_id', 'string'), OutputField('revision', 'integer'),
         OutputField('text', 'string'), OutputField('status', 'string'),
         OutputField('start_ms', 'integer'), OutputField('end_ms', 'integer'),
         OutputField('language', 'string')), streaming_mode, window_ms=window_ms,
        lookahead_ms=lookahead_ms, freshness_ms=freshness_ms, stateful=True,
        reset_on_gap=True, interpretation='semantic_description', license=license)


class TranscriptMemory:
    def __init__(self, *, max_entries=256, max_streams=128, max_bytes=1024*1024,
                 retention_ms=86400000):
        for value, bound in ((max_entries, 4096), (max_streams, 512),
                             (max_bytes, 16*1024*1024), (retention_ms, 86400000)):
            if type(value) is not int or not 1 <= value <= bound:
                raise ValueError('invalid transcript bounds')
        self.max_entries, self.max_streams = max_entries, max_streams
        self.max_bytes, self.retention_ms = max_bytes, retention_ms
        self.entries, self.streams = OrderedDict(), {}

    def prune(self, now):
        self.entries = OrderedDict((k, v) for k, v in self.entries.items()
            if now-v['updated_ms'] <= self.retention_ms)
        self.streams = {k: v for k, v in self.streams.items() if now-v[0] <= self.retention_ms}

    def prepare(self, record, *, now_ms=None):
        """Validate without mutation, before the runtime persists/emits the record."""
        now = timestamp(now_ms)
        record = bounded_json(record, max_bytes=131072)
        obs = record.get('observation', {})
        sensor, capability = obs.get('sensor'), obs.get('capability')
        if record.get('type') != 'observation.signal' or obs.get('name') != TRANSCRIPT_OUTPUT or not sensor or not capability:
            raise ValueError('transcripts require a registered audio capability and sensor evidence')
        if obs.get('modality') != 'audio' or record.get('quality', {}).get('status') != 'sufficient':
            raise ValueError('transcripts require sufficient audio evidence')
        value = obs.get('value')
        required = {'utterance_id', 'revision', 'text', 'status', 'start_ms', 'end_ms', 'language'}
        if type(value) is not dict or set(value) != required:
            raise ValueError('invalid transcript update fields')
        identifier(value['utterance_id'], 'utterance_id')
        if type(value['revision']) is not int or not 0 <= value['revision'] <= 2**31-1:
            raise ValueError('revision must be a nonnegative integer')
        if type(value['text']) is not str or len(value['text']) > 4096:
            raise ValueError('transcript text must contain at most 4096 characters')
        if value['status'] not in ('partial', 'final'):
            raise ValueError('status must be partial or final')
        identifier(value['language'], 'language')
        if len(value['language']) > 32:
            raise ValueError('language identifier too long')
        ts = record['timestamp_ms']
        if type(ts) is not int or ts > now or now-ts > self.retention_ms:
            raise ValueError('transcript evidence must be captured within retention, not in the future')
        start, end = value['start_ms'], value['end_ms']
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= ts or end-start > 300000:
            raise ValueError('utterance boundaries must be capture milliseconds, at most five minutes')
        epoch = sensor.get('stream_epoch') or obs.get('processing_epoch')
        identifier(epoch, 'stream epoch')
        namespace = (record['subject_id'], sensor['source_id'], sensor['sensor_id'], capability['id'], epoch)
        key = namespace+(value['utterance_id'],)
        previous = self.entries.get(key)
        if previous and now-previous['updated_ms'] > self.retention_ms:
            previous = None
        watermark = self.streams.get(namespace)
        if watermark and now-watermark[0] > self.retention_ms:
            watermark = None
        sequence = sensor['sequence']
        if watermark and (ts < watermark[0] or sequence <= watermark[1]):
            raise ValueError('transcript stream capture order must strictly advance')
        if not watermark and sum(now-v[0] <= self.retention_ms for v in self.streams.values()) >= self.max_streams:
            raise ValueError('transcript stream limit reached; clear or expire retained streams')
        if previous:
            old = previous['update']
            if value == old and previous['status'] != 'interrupted':
                entry = deepcopy(previous)  # Exact payload retry: no duplicated text.
            else:
                if previous['status'] in ('final', 'interrupted'):
                    raise ValueError('closed utterance cannot be revised; use a new utterance ID')
                if value['revision'] != old['revision']+1:
                    raise ValueError('revision must increase by one')
                if start != old['start_ms'] or end < old['end_ms']:
                    raise ValueError('utterance start is fixed and capture end cannot move backwards')
                entry = None
        else:
            if value['revision'] != 0:
                raise ValueError('first retained update must have revision zero')
            entry = None
        if entry is None:
            entry = dict(subject_id=record['subject_id'], source_id=sensor['source_id'],
                sensor_id=sensor['sensor_id'], capability_id=capability['id'], epoch=epoch,
                update=deepcopy(value), status=value['status'], updated_ms=ts,
                evidence_id=record['id'], models=record['models'], sequence=sequence,
                freshness_ms=capability['freshness_ms'])
        if len(json.dumps(entry, ensure_ascii=True).encode()) > self.max_bytes:
            raise ValueError('single transcript exceeds memory byte budget')
        return key, entry, (ts, sequence), now

    def commit(self, prepared):
        key, entry, watermark, now = prepared
        self.prune(now)
        self.streams[key[:-1]] = watermark
        self.entries[key] = entry
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries or len(json.dumps(list(self.entries.values()), ensure_ascii=True).encode()) > self.max_bytes:
            self.entries.popitem(last=False)

    def interrupt(self, source_id=None, sensor_id=None, capability_id=None):
        for entry in self.entries.values():
            if (entry['status'] == 'partial' and
                (source_id is None or entry['source_id'] == source_id) and
                (sensor_id is None or entry['sensor_id'] == sensor_id) and
                (capability_id is None or entry['capability_id'] == capability_id)):
                entry['status'] = 'interrupted'

    def snapshot(self, subject_id='local-user', *, now_ms=None, start_ms=0, limit=16, max_bytes=32768):
        now = timestamp(now_ms)
        if type(limit) is not int or not 1 <= limit <= 100 or type(max_bytes) is not int or not 1024 <= max_bytes <= 1048576:
            raise ValueError('invalid transcript query bounds')
        self.prune(now)
        selected = [v for v in reversed(self.entries.values()) if v['subject_id'] == subject_id and v['updated_ms'] >= start_ms]
        selected.sort(key=lambda v: (v['update']['start_ms'], v['updated_ms']), reverse=True)
        output, size = [], 0
        for entry in selected:
            item = deepcopy(entry)
            item['fresh'] = entry['status'] != 'interrupted' and now-entry['updated_ms'] <= entry['freshness_ms']
            cost = len(json.dumps(item, ensure_ascii=True).encode())
            if len(output) >= limit or size+cost > max_bytes:
                break
            output.append(item)
            size += cost
        output.reverse()
        return dict(utterances=output, retained_count=len(selected), truncated=len(output) < len(selected),
                    interpretation='model_transcription_unverified')

    def clear(self, subject_id=None):
        self.entries = OrderedDict((k, v) for k, v in self.entries.items() if subject_id is not None and k[0] != subject_id)
        self.streams = {k: v for k, v in self.streams.items() if subject_id is not None and k[0] != subject_id}
