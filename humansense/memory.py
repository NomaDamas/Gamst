"""Bounded observation memory. Similarity is never automatic identity proof."""
from collections import OrderedDict
from copy import deepcopy
import json
import uuid

from .capabilities import bounded_json, identifier
from .schema import timestamp


class ObjectMemory:
    def __init__(self, *, max_entities=256, history_size=8, retention_ms=86400000,
                 visibility_ms=1000, max_bytes=4*1024*1024):
        for value, upper in ((max_entities, 4096), (history_size, 64),
                             (retention_ms, 2592000000), (visibility_ms, 60000),
                             (max_bytes, 64*1024*1024)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError('invalid memory bounds')
        self.max_entities, self.history_size = max_entities, history_size
        self.retention_ms, self.visibility_ms, self.max_bytes = retention_ms, visibility_ms, max_bytes
        self._entities = OrderedDict()
        self._latest = {}

    def _prune(self, now):
        for key in list(self._entities):
            if now-self._entities[key]['last_seen_ms'] > self.retention_ms:
                del self._entities[key]
            else:
                entry = self._entities[key]
                entry['history'] = [h for h in entry['history'] if now-h['timestamp_ms'] <= self.retention_ms]
                if entry['history']:
                    entry['first_seen_ms'] = entry['history'][0]['timestamp_ms']
        self._latest = {key: value for key, value in self._latest.items()
                        if now-value[0] <= self.retention_ms}
        while len(self._latest) > self.max_entities*2:
            del self._latest[min(self._latest, key=self._latest.get)]

    def observe(self, record, *, objects_field='objects', complete_snapshot=True, now_ms=None,
                live=True):
        """Accept an already stored registered observation with explicit object schema."""
        now = timestamp(now_ms)
        record = bounded_json(record, max_bytes=131072)
        identifier(objects_field, 'objects_field')
        if type(complete_snapshot) is not bool or type(live) is not bool:
            raise ValueError('invalid snapshot semantics')
        observation = record.get('observation', {})
        sensor, capability = observation.get('sensor'), observation.get('capability')
        if record.get('type') != 'observation.signal' or not sensor or not capability:
            raise ValueError('object memory requires registered sensor evidence')
        if record.get('quality', {}).get('status') != 'sufficient':
            raise ValueError('insufficient evidence cannot update object memory')
        epoch = sensor.get('stream_epoch') or observation.get('processing_epoch')
        identifier(epoch, 'observation epoch')
        ts = timestamp(record['timestamp_ms'])
        if ts > now or now-ts > self.retention_ms:
            raise ValueError('evidence timestamp outside memory retention')
        namespace = (record['subject_id'], sensor['source_id'], sensor['sensor_id'], capability['id'], epoch)
        order = (ts, sensor['sequence'], record['id'])
        previous = self._latest.get(namespace)
        if previous and (ts < previous[0] or sensor['sequence'] < previous[1] or
                         sensor['sequence'] == previous[1] and record['id'] != previous[2]):
            raise ValueError('out-of-order snapshot')
        objects = observation.get('value', {}).get(objects_field)
        if not isinstance(objects, list) or len(objects) > 256:
            raise ValueError('objects field must be a list of at most 256 objects')
        validated, tracks = [], set()
        for item in objects:
            item = bounded_json(item, max_bytes=4096, forbid_raw=True)
            if not isinstance(item, dict) or set(item)-{'track_id', 'label', 'attributes', 'location', 'relations'}:
                raise ValueError('unknown object fields')
            track = identifier(item.get('track_id'), 'track_id')
            label = item.get('label')
            if not isinstance(label, str) or not 1 <= len(label) <= 256 or track in tracks:
                raise ValueError('objects need a label and unique track IDs')
            if 'location' in item and not sensor.get('coordinate_frame'):
                raise ValueError('location requires a sensor coordinate frame')
            for name in ('attributes', 'location'):
                if name in item and not isinstance(item[name], dict):
                    raise ValueError(f'{name} must be an object')
            if 'relations' in item and not isinstance(item['relations'], list):
                raise ValueError('relations must be an array')
            tracks.add(track)
            validated.append(item)
        # Validate everything before mutating visibility or history.
        self._prune(now)
        if complete_snapshot:
            for key, entity in self._entities.items():
                if key[:-1] == namespace:
                    entity['in_latest_snapshot'] = False
        for item in validated:
            key = namespace+(item['track_id'],)
            entry = self._entities.get(key)
            if entry is None:
                entry = dict(entity_id=str(uuid.uuid4()), subject_id=record['subject_id'],
                    source_id=sensor['source_id'], sensor_id=sensor['sensor_id'],
                    capability_id=capability['id'], epoch=epoch, track_id=item['track_id'],
                    first_seen_ms=ts, history=[])
            evidence = dict(record_id=record['id'], timestamp_ms=ts, object=deepcopy(item),
                            coordinate_frame=sensor.get('coordinate_frame'), models=deepcopy(record.get('models', [])))
            # Reprocessing the same stored observation must not inflate history.
            if not entry['history'] or entry['history'][-1]['record_id'] != record['id']:
                entry['history'].append(evidence)
                entry['history'] = entry['history'][-self.history_size:]
            entry.update(last_seen_ms=ts, in_latest_snapshot=live, label=item['label'],
                         last_observation=evidence)
            self._entities[key] = entry
            self._entities.move_to_end(key)
        self._latest[namespace] = order
        while self._entities and (len(self._entities) > self.max_entities or
                len(json.dumps(list(self._entities.values()), ensure_ascii=True).encode()) > self.max_bytes):
            self._entities.popitem(last=False)
        self._prune(now)
        return {'observed': len(validated), 'retained_entities': len(self._entities)}

    def find(self, *, subject_id='local-user', text=None, entity_id=None, now_ms=None, limit=20):
        now = timestamp(now_ms)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('limit must be 1..100')
        if text is not None and (not isinstance(text, str) or len(text) > 256):
            raise ValueError('text query must be at most 256 characters')
        self._prune(now)
        results = []
        for entry in reversed(self._entities.values()):
            if entry['subject_id'] != subject_id or entity_id is not None and entry['entity_id'] != entity_id:
                continue
            if text and text.casefold() not in json.dumps(entry['last_observation']['object'], ensure_ascii=False).casefold():
                continue
            result = deepcopy(entry)
            fresh = now-entry['last_seen_ms'] <= self.visibility_ms
            result['currently_visible'] = entry['in_latest_snapshot'] and fresh
            result['visibility'] = 'visible' if result['currently_visible'] else 'previously_seen'
            result['current_location'] = None
            result['identity_basis'] = 'provider_track_within_source_epoch'
            results.append(result)
            if len(results) == limit:
                break
        return results

    def clear(self, subject_id=None):
        before = len(self._entities)
        self._entities = OrderedDict((k, v) for k, v in self._entities.items()
                                     if subject_id is not None and k[0] != subject_id)
        self._latest = {k: v for k, v in self._latest.items() if subject_id is not None and k[0] != subject_id}
        return before-len(self._entities)

    def invalidate_visibility(self, source_id=None, sensor_id=None, capability_id=None):
        for entry in self._entities.values():
            if ((source_id is None or entry['source_id'] == source_id)
                    and (sensor_id is None or entry['sensor_id'] == sensor_id)
                    and (capability_id is None or entry['capability_id'] == capability_id)):
                entry['in_latest_snapshot'] = False
