"""Frozen image rules bound to an exact compensated anchor capture."""
import re

from ..capabilities import bounded_json, identifier
from .image_events import ImageEventRules


def configuration(value):
    value = bounded_json(value, max_bytes=8192)
    fields = {'id', 'source_id', 'sensor_id', 'coordinate_frame', 'calibration_id',
        'image_size', 'timestamp_ms', 'capture_sequence', 'evidence_sha256', 'rules'}
    if type(value) is not dict or set(value) != fields:
        raise ValueError('exact frozen anchor reference required')
    for key in ('id', 'source_id', 'sensor_id', 'coordinate_frame'):
        identifier(value[key])
    if value['calibration_id'] is not None:
        identifier(value['calibration_id'])
    size = value['image_size']
    if (type(size) is not list or len(size) != 2 or
            any(type(x) is not int or not 16 <= x <= 4096 for x in size)):
        raise ValueError('anchor image dimensions required')
    for key, maximum in (('timestamp_ms', 2**53-1), ('capture_sequence', 2**63-1)):
        if type(value[key]) is not int or not 0 <= value[key] <= maximum:
            raise ValueError('bounded original anchor capture identity required')
    if type(value['evidence_sha256']) is not str or re.fullmatch('[0-9a-f]{64}', value['evidence_sha256']) is None:
        raise ValueError('exact anchor evidence hash required')
    ImageEventRules(value['rules'])
    return value


def evaluate(result, reference):
    """Evaluate only same-anchor original histories, preserving original indices."""
    geometry = result['geometry']
    anchor = geometry['anchor']
    matched = (all(reference[k] == geometry[k] for k in
        ('source_id', 'sensor_id', 'coordinate_frame', 'calibration_id', 'image_size')) and
        all(reference[k] == anchor[k] for k in ('timestamp_ms', 'capture_sequence', 'evidence_sha256')))
    ready = matched and geometry['status'] == 'estimated'
    rules = ImageEventRules(reference['rules'])
    paths = []
    if ready:
        for obj in result['objects']:
            if obj['anchor_id'] != anchor['id']:
                raise ValueError('event history crosses compensated anchor')
            paths.append(dict(observations=[dict(sample, centroid=sample['anchor_projection'])
                for sample in obj['observations']]))
    events = rules.evaluate(dict(objects=paths, geometry=geometry))
    events['event_summary'].update(reference=reference, anchor=anchor,
        stream_epoch=geometry['stream_epoch'],
        status='bound_estimated_anchor' if ready else 'collecting_anchor' if matched else 'reference_unbound',
        input_snapshot_complete=geometry['complete_snapshot'],
        coordinate_truth_verified=False, between_observations='unobserved')
    return events
