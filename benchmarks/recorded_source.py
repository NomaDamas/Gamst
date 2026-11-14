"""Bounded original capture declarations for explicitly owned SDK replay.

Capture acquisition and downstream provider delivery are separate observations.
These declarations do not certify physical sensing, model quality or billing.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

from humansense.capabilities import identifier

MAX_MANIFEST_BYTES = 64 * 1024**2
MAX_RECORDS = 262144
MAX_PAYLOAD_BYTES = 64 * 1024**3
MAX_DURATION_NS = 3600 * 10**9


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('exact recording declaration fields required')


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('bounded exact recording integer required')
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate recording JSON key')
        result[key] = value
    return result


def decode_manifest(payload):
    if type(payload) is not bytes or len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError('recording manifest exceeds byte bound')
    def invalid(_):
        raise ValueError('nonfinite recording JSON number')
    return validate_manifest(json.loads(payload, object_pairs_hook=_pairs,
                                        parse_constant=invalid))


def validate_manifest(value):
    """Return detached original declarations; never open a source or provider."""
    _keys(value, ('schema_version', 'recording_id', 'origin_capture_ns',
                  'duration_ns', 'sources', 'records', 'availability'))
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('recorded source schema1 required')
    identifier(value['recording_id'])
    origin = _integer(value['origin_capture_ns'], 0, 2**63-1)
    duration = _integer(value['duration_ns'], 1, MAX_DURATION_NS)
    if origin + duration > 2**63-1:
        raise ValueError('capture interval exceeds integer bound')
    sources = value['sources']
    if type(sources) is not list or not 1 <= len(sources) <= 16:
        raise ValueError('one to sixteen explicitly selected capture sources required')
    inventory = {}
    for source in sources:
        _keys(source, ('source_id', 'sensor_id', 'stream_epoch', 'format', 'modality', 'calibration_id'))
        for name in ('source_id', 'sensor_id', 'stream_epoch'):
            identifier(source[name])
        if source['source_id'] in inventory or source['format'] not in ('bgr8_hwc', 'pcm_s16le_mono'):
            raise ValueError('unique typed capture sources required')
        expected = 'video' if source['format'] == 'bgr8_hwc' else 'audio'
        if source['modality'] != expected:
            raise ValueError('source modality must match frozen format')
        if source['calibration_id'] is not None:
            identifier(source['calibration_id'])
        inventory[source['source_id']] = source
    records = value['records']
    if type(records) is not list or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError('bounded nonempty capture inventory required')
    previous = {}; last_order = None; total = 0; payloads = {}
    for record in records:
        if type(record) is not dict:
            raise ValueError('capture record required')
        form = record.get('format')
        fields = ('shape',) if form == 'bgr8_hwc' else ('sample_rate', 'sample_count')
        _keys(record, ('source_id', 'sensor_id', 'sequence', 'capture_ns', 'offset_ns',
                       'format', 'stream_epoch', 'payload_path', 'payload_bytes', 'payload_sha256') + fields)
        source = inventory.get(record['source_id'])
        if source is None or any(source[k] != record[k] for k in ('sensor_id', 'format', 'stream_epoch')):
            raise ValueError('capture differs from declared source')
        seq = _integer(record['sequence'], 0, 2**63-1)
        offset = _integer(record['offset_ns'], 0, duration-1)
        order = (offset, record['source_id'], seq)
        if seq <= previous.get(record['source_id'], -1) or (last_order is not None and order <= last_order):
            raise ValueError('ordered captures and strictly advancing source sequence required')
        _integer(record['capture_ns'], 0, 2**63-1)
        if record['capture_ns'] != origin + offset:
            raise ValueError('capture time must equal original origin plus playback offset')
        previous[record['source_id']] = seq; last_order = order
        if form == 'bgr8_hwc':
            shape = record['shape']
            if type(shape) is not list or len(shape) != 3 or shape[2] != 3:
                raise ValueError('BGR HWC3 shape required')
            h = _integer(shape[0], 1, 1080); w = _integer(shape[1], 1, 1920)
            _integer(shape[2], 3, 3)
            size = h*w*3
        else:
            rate = _integer(record['sample_rate'], 8000, 48000)
            count = _integer(record['sample_count'], 1, rate//10)
            size = count*2
            if offset*rate + count*10**9 > duration*rate:
                raise ValueError('audio extends past declared recording end')
        if _integer(record['payload_bytes'], 1, 1920*1080*3) != size:
            raise ValueError('payload byte count differs from exact typed layout')
        path = record['payload_path']; sha = record['payload_sha256']
        if type(path) is not str or not 1 <= len(path) <= 4096 or '\0' in path or Path(path).is_absolute() or any(x in ('', '.', '..') for x in path.split('/')):
            raise ValueError('relative nonescaping capture payload path required')
        if type(sha) is not str or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('original capture SHA256 required')
        identity = (size, sha)
        if path in payloads and payloads[path] != identity:
            raise ValueError('one payload path cannot declare different frozen bytes')
        if path not in payloads: total += size
        payloads[path] = identity
        if total > MAX_PAYLOAD_BYTES:
            raise ValueError('joint recording payload bound exceeded')
    partitions = value['availability']
    if type(partitions) is not list or len(partitions) != len(inventory):
        raise ValueError('availability partition required for every source')
    seen = set(); interval_count = 0
    for partition in partitions:
        _keys(partition, ('source_id', 'intervals'))
        sid = partition['source_id']
        if type(sid) is not str or sid not in inventory or sid in seen:
            raise ValueError('unique known availability source required')
        seen.add(sid); intervals = partition['intervals']
        if type(intervals) is not list or not intervals:
            raise ValueError('bounded declared availability partition required')
        interval_count += len(intervals)
        if interval_count > 4096: raise ValueError('availability interval bound exceeded')
        end = 0
        for interval in intervals:
            _keys(interval, ('start_ns', 'end_ns', 'state'))
            start = _integer(interval['start_ns'], 0, duration)
            finish = _integer(interval['end_ns'], 1, duration)
            if start != end or finish <= start or interval['state'] not in ('available', 'unavailable', 'unknown'):
                raise ValueError('gap-free declared availability partition required')
            end = finish
        if end != duration: raise ValueError('availability must span declared recording')
        index = 0
        for record in records:
            if record['source_id'] != sid: continue
            while record['offset_ns'] >= intervals[index]['end_ns']: index += 1
            if intervals[index]['state'] == 'unavailable':
                raise ValueError('capture inside declared unavailable interval')
            if record['format'] == 'pcm_s16le_mono':
                # PCM describes a half-open interval, not just its first sample.
                # Compare rational endpoints without rounding away a boundary.
                rate = record['sample_rate']
                audio_end = record['offset_ns'] * rate + record['sample_count'] * 10**9
                scan = index
                while scan < len(intervals) and intervals[scan]['start_ns'] * rate < audio_end:
                    if intervals[scan]['state'] == 'unavailable':
                        raise ValueError('audio overlaps declared unavailable interval')
                    scan += 1
    data = encoded(value)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError('joint recording manifest bound exceeded')
    return json.loads(data)


def read_capture(record, asset_directory):
    """Hash actual owned bytes, not a pathname checked before an await.

    The caller supplies a validated detached record. This function runs off the
    event loop; a source scheduler must not block another source on disk I/O.
    """
    size = _integer(record['payload_bytes'], 1, 1920*1080*3)
    path = record['payload_path']
    if type(path) is not str or Path(path).is_absolute() or any(x in ('', '.', '..') for x in path.split('/')):
        raise ValueError('relative nonescaping capture path required')
    root = Path(asset_directory)
    if not root.is_absolute(): raise ValueError('explicit absolute asset directory required')
    # Open every component relative to an owned directory; no symlink traversal.
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in root.parts[1:] + tuple(path.split('/')[:-1]):
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = child
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
        fd = os.open(path.split('/')[-1], flags, dir_fd=directory)
    finally:
        os.close(directory)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise ValueError('exact regular capture payload required')
        blocks = []; remaining = size
        while remaining:
            block = os.read(fd, min(65536, remaining))
            if not block:
                raise ValueError('capture truncated during owned read')
            blocks.append(block); remaining -= len(block)
        if os.read(fd, 1) or os.fstat(fd).st_size != size:
            raise ValueError('capture grew during owned read')
        payload = b''.join(blocks)
        if hashlib.sha256(payload).hexdigest() != record['payload_sha256']:
            raise ValueError('actual capture bytes differ from frozen hash')
        return payload
    finally:
        os.close(fd)
