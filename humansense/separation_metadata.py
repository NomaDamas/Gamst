"""Closed v1 SAM separation metadata; never accepts arbitrary provider objects."""
import math
import re

from .capabilities import SensorMetadata, identifier


def _integer(low, high):
    return lambda v: type(v) is int and low <= v <= high


def _number(low, high):
    return lambda v: type(v) in (int, float) and math.isfinite(v) and low <= v <= high


def _enum(*values):
    return lambda v: type(v) is type(values[0]) and v in values


def _hash(v):
    return type(v) is str and re.fullmatch('[0-9a-f]{64}', v) is not None


def _object(value, fields, *, required=()):
    if type(value) is not dict or set(value)-set(fields) or not set(required) <= set(value):
        raise ValueError('unknown or missing SAM separation metadata fields')
    for key, item in value.items():
        if not fields[key](item):
            raise ValueError('invalid typed SAM separation metadata field: '+key)
    return True


def _sensor(value):
    keys = {'source_id', 'sensor_id', 'sequence', 'coordinate_frame', 'channels',
            'calibration_id', 'stream_epoch', 'capture_sequence'}
    if type(value) is not dict or set(value) != keys or type(value['channels']) is not list or len(value['channels']) != 1:
        return False
    data = dict(value); data['channels'] = tuple(data['channels'])
    SensorMetadata(**data)
    return value['stream_epoch'] is not None


def _binding(value):
    def identity(v):
        identifier(v)
        return True
    fields = dict(subject_id=identity, sensor=_sensor,
        capture_end_ms=_integer(0, 2**53-1), capture_start_ms=_number(0, 2**53-1),
        sample_rate=_enum(48000), sample_count=_integer(48000, 192000),
        generation=_integer(0, 2**63-1), owner_revision=_integer(0, 2**63-1),
        capability_id=_enum('audio.sam_audio.separation'),
        config_id=lambda v: type(v) is str and re.fullmatch(r'transport1\.[0-9a-f]{64}', v) is not None,
        input_sha256=_hash)
    required=set(fields)
    visual=dict(audio_input_sha256=_hash,visual_inventory_sha256=_hash,
                original_audio_epoch=identity,original_visual_epoch=identity)
    if type(value) is dict and set(value)&set(visual):
        required.update(visual)
    fields.update(visual)
    return _object(value, fields, required=required)


def _spans(value):
    if type(value) is not list or len(value) > 32:
        return False
    for span in value:
        if (type(span) is not list or len(span) != 3 or span[0] not in ('+', '-') or
                not _integer(0, 192000)(span[1]) or not _integer(1, 192000)(span[2]) or span[1] >= span[2]):
            return False
    return True


def _media(value):
    if value == {}:
        return True  # Child returns no references; only the parent supplies them.
    def component(v):
        fields = dict(sha256=_hash, bytes=_integer(48000*4, 192000*4),
                      samples_outside_normalized_range=_integer(0, 192000))
        return _object(v, fields, required=fields)
    fields = dict(reference=lambda v: type(v) is str and re.fullmatch('[0-9a-f]{32}', v) is not None,
        available_at_publication=_enum(True), availability=_enum('explicit-owner-lookup-required'),
        ttl_ms=_integer(1, 60000), expires_at_ms=_integer(0, 2**53-1),
        retention_capacity=_integer(1, 32), retention_max_bytes=_integer(1536000, 64*1024*1024),
        sample_rate=_enum(48000), sample_count=_integer(48000, 192000), encoding=_enum('<f4'),
        components=lambda v: _object(v, dict(target=component, residual=component), required=('target','residual')))
    return _object(value, fields, required=fields)


def _visual(value):
    def integer_list(v,low,high,limit):
        return type(v) is list and 1<=len(v)<=limit and all(_integer(low,high)(x) for x in v)
    def intervals(v):
        return type(v) is list and 1<=len(v)<=128 and all(
            type(row) is list and len(row)==2 and all(_integer(0,2**53-1)(t) for t in row)
            and row[0]<=row[1] for row in v)
    fields=dict(mapping_sha256=_hash,original_intervals_us=intervals,
        selected_indices=lambda v:integer_list(v,0,127,512),
        ages_sixth_us=lambda v:integer_list(v,0,1500000,512),
        original_frames=_integer(1,128),feature_frames=_integer(1,512),
        image_size=_integer(16,1024),encoder_batch_size=_integer(1,64),
        mask_polarity=_enum('zero-kept'),alignment=_enum('latest-certain-past'),
        clock_verified=_enum(False),track_identity_verified=_enum(False))
    _object(value,fields,required=fields)
    if (len(value['original_intervals_us'])!=value['original_frames'] or
            len(value['selected_indices'])!=value['feature_frames'] or
            len(value['ages_sixth_us'])!=value['feature_frames'] or
            any(i>=value['original_frames'] for i in value['selected_indices'])):
        raise ValueError('visual provenance dimensions differ')
    return True


def validate_metadata(value):
    """Every permitted nested field has fixed scalar or structural semantics.

    Additions require explicit schema/version review. In particular there is no
    arbitrary `details`, free numeric array, waveform or encoded-media field.
    """
    _object(value['separation'], dict(status=_enum('estimated', 'fixture-estimate'),
        quality_validated=_enum(False), ambient_preserved=_enum(True),
        additive_reconstruction_verified=_enum(False), native_streaming=_enum(False),
        cross_window_continuity_verified=_enum(False), discontinuity=_enum(True, False)),
        required=('status', 'quality_validated'))
    _object(value['window'], dict(capture_start_ms=_number(0, 2**53-1),
        capture_end_ms=_integer(0, 2**53-1), sample_rate=_enum(48000),
        sample_count=_integer(48000, 192000), padded_sample_count=_integer(48000, 240000),
        trimmed_padding_samples=_integer(0, 48000), source_evidence=_binding,
        inference_ms=_number(0, 600000), collection_ms=_number(1000, 4000),
        additional_lookahead_ms=_enum(0), whole_window=_enum(True),
        partial_window_policy=_enum('reject'), discontinuity=_enum(True, False),
        discontinuity_reason=_enum('initial', 'gap', 'reset', 'reconnect'),
        clock_verified=_enum(False)))
    _object(value['prompt'], dict(sha256=_hash, raw_text_sha256=_hash,
        anchors_sha256=_hash, utf8_bytes=_integer(0 if 'visual' in value['prompt'] else 1, 4096), anchor_count=_integer(0, 32),
        feature_spans=_spans, token_count_before=_integer(0, 8192),
        token_count_after=_integer(0, 512), max_tokens=_integer(1, 512), truncated=_enum(True, False), visual=_visual))
    _object(value['model'], dict(fixture_engine=_enum(True),
        source_revision=_enum('bb4c6999d2677c7402360e426afc01ddfad6dce0'),
        runtime_manifest_sha256=_hash, sam_manifest_sha256=_hash, t5_manifest_sha256=_hash,
        checkpoint_sha256=_hash, original_config_sha256=_hash, effective_config_sha256=_hash,
        device=_enum('cpu', 'cuda'), precision=_enum('float32'), predict_spans=_enum(False),
        reranking_candidates=_enum(1), ode_steps=_integer(1, 64), ode_method=_enum('midpoint'),
        random_seed=_integer(0, 2**63-1)))
    if value['separation']['status'] == 'fixture-estimate' and value['model'].get('fixture_engine') is not True:
        raise ValueError('fixture separation must identify its fixture engine')
    _media(value['media'])
    return value
