"""Scope-bound stateless replay cursors for one open local store."""
import base64
import hashlib
import hmac
import json

from .capabilities import identifier


def _encode(store, value):
    payload=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    return base64.urlsafe_b64encode(hmac.digest(store._cursor_key,payload,'sha256')+payload).decode()


def _decode(store, cursor):
    try:
        if type(cursor) is not str or not 1 <= len(cursor) <= 2048: raise ValueError()
        raw=base64.b64decode(cursor.encode('ascii'),altchars=b'-_',validate=True)
        signature,payload=raw[:32],raw[32:]
        if not hmac.compare_digest(signature,hmac.digest(store._cursor_key,payload,'sha256')): raise ValueError()
        value=json.loads(payload)
        if (type(value) is not dict or set(value) != {'after','scope','retention_revision'}
                or type(value['after']) is not int or not 0 <= value['after'] <= 2**63-1
                or type(value['retention_revision']) is not int or value['retention_revision'] < 0):
            raise ValueError()
        return value
    except (ValueError,TypeError,UnicodeError) as exc:
        raise ValueError('invalid or expired observation cursor; restart without cursor') from exc


def observation_page(store, *, subject_id, cursor=None, limit=20, max_bytes=65536,
                     capability_id=None, source_id=None, sensor_id=None, events_only=False):
    if type(subject_id) is not str or not 1 <= len(subject_id) <= 128:
        raise ValueError('subject_id must be a string of 1..128 characters')
    if type(events_only) is not bool: raise ValueError('events_only must be boolean')
    if type(limit) is not int or not 1 <= limit <= 100 or type(max_bytes) is not int or not 1024 <= max_bytes <= 262144:
        raise ValueError('limit must be 1..100 and max_bytes 1024..262144')
    for value in (capability_id,source_id,sensor_id):
        if value is not None: identifier(value)
    scope=hashlib.sha256(json.dumps([subject_id,capability_id,source_id,sensor_id,events_only]).encode()).hexdigest()
    prior=None if cursor is None else _decode(store,cursor)
    if prior is not None and prior['scope'] != scope:
        raise ValueError('cursor belongs to different observation query filters')
    after=0 if prior is None else prior['after']
    rows,upper=store.observation_page_rows(subject=subject_id,after=after,limit=limit+1,
        capability_id=capability_id,source_id=source_id,sensor_id=sensor_id,events_only=events_only)
    items=[];size=0;blocked=None
    for position,record in rows:
        cost=len(json.dumps(record,ensure_ascii=True).encode())
        if len(items) >= limit or size+cost > max_bytes:
            if not items and cost > max_bytes: blocked=cost
            break
        items.append(record);size+=cost;after=position
    more=len(items)<len(rows)
    if not more: after=upper
    return dict(items=items,returned_count=len(items),has_more=more,
        next_cursor=_encode(store,dict(after=after,scope=scope,retention_revision=store.retention_revision)),
        blocked_record_bytes=blocked,entry_byte_budget=max_bytes,order='ingestion_ascending',
        scope='retained_registered_observations',events_only=events_only,
        retention_changed_since_cursor=None if prior is None else prior['retention_revision'] != store.retention_revision,
        upstream_completeness='unverified')
