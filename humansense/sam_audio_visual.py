"""Exact causal alignment for SAM visual windows; no capture or model claims.

Clock offsets/uncertainty are caller declarations. The result retains original
indices: holding a past frame never creates a newly observed frame. Native
integration must additionally validate packet ownership, pixels and masks.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class VisualAlignment:
    indices: tuple
    feature_times_sixth_us: tuple
    ages_sixth_us: tuple
    max_age_us: int
    clock_error_us: int
    camera_to_audio_offset_us: int


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} outside integer bounds')
    return value


def align_visual_frames(intervals_us, *, audio_end_us, sample_count,
                        hop_length, camera_to_audio_offset_us,
                        clock_error_us, max_age_us):
    """Map each 48kHz feature to the latest certainly-past captured interval.

    Times use integer sixth-microseconds: one sample is exactly125 such units.
    An interval's end plus declared clock error must not exceed feature time;
    its end minus that error must not be older than max_age_us. No interpolation
    or rounding conceals missing coverage. A bounded pre-roll is allowed.
    Input intervals are camera-clock [start,end] pairs, strictly ordered by end.
    """
    end = _integer(audio_end_us, 0, 2**53-1, 'audio end')*6
    count = _integer(sample_count, 48000, 192000, 'sample count')
    hop = _integer(hop_length, 1, 48000, 'hop length')
    features = (count+hop-1)//hop
    if features > 512:
        raise ValueError('at most512 feature frames supported')
    offset = _integer(camera_to_audio_offset_us, -(2**53-1), 2**53-1, 'clock offset')
    age = _integer(max_age_us, 1, 250000, 'maximum age')
    error = _integer(clock_error_us, 0, age, 'clock error')
    start = end-count*125
    if start < 0:
        raise ValueError('audio window precedes clock origin')
    if type(intervals_us) not in (tuple, list) or not 1 <= len(intervals_us) <= 128:
        raise ValueError('one to128 original frame intervals required')
    intervals = tuple(intervals_us)
    if not 1 <= len(intervals) <= 128:
        raise ValueError('frame list changed beyond bounds')
    ends, previous_start, previous_end = [], None, None
    for interval in intervals:
        if type(interval) not in (tuple, list) or len(interval) != 2:
            raise ValueError('capture interval must be a pair')
        a, b = interval
        a = _integer(a, 0, 2**53-1, 'capture start')
        b = _integer(b, a, 2**53-1, 'capture end')
        if b-a > age:
            raise ValueError('exposure exceeds admitted age')
        if previous_start is not None and (a < previous_start or b <= previous_end):
            raise ValueError('capture intervals regress or repeat')
        mapped_start, mapped_end = (a+offset)*6, (b+offset)*6
        if mapped_start-error*6 < start-age*6 or mapped_end+error*6 > end:
            raise ValueError('capture outside admitted window and pre-roll')
        ends.append(mapped_end)
        previous_start, previous_end = a, b
    indices, times, ages = [], [], []
    cursor = -1
    for k in range(features):
        moment = start+k*hop*125
        while cursor+1 < len(ends) and ends[cursor+1]+error*6 <= moment:
            cursor += 1
        if cursor < 0 or moment-(ends[cursor]-error*6) > age*6:
            raise ValueError('missing causal visual coverage')
        indices.append(cursor)
        times.append(moment)
        ages.append(moment-ends[cursor])
    return VisualAlignment(tuple(indices), tuple(times), tuple(ages), age, error, offset)


def visual_allocation_bounds(*, frame_count, height, width, feature_count,
                             image_size):
    """Reject expanded preprocessing before gather/resize allocations.

    The caller must also include PCM, masks and headers in the16MiB wire bound.
    This conservatively admits retained originals, two uint8 gather copies, uint8 resize and
    source plus three resized float32 intermediates. It does not bound model
    activations, allocator overhead or measured peak memory.
    """
    n = _integer(frame_count, 1, 128, 'original frame count')
    h = _integer(height, 16, 480, 'height')
    w = _integer(width, 16, 640, 'width')
    m = _integer(feature_count, 1, 512, 'feature count')
    size = _integer(image_size, 16, 1024, 'vision image size')
    original = n*4*h*w  # RGB plus one binary-mask channel.
    expanded = m*3*h*w
    preprocess = original+2*expanded+m*3*size*size+4*m*3*(h*w+3*size*size)
    if original > 16*1024*1024 or expanded > 64*1024*1024 or preprocess > 128*1024*1024:
        raise ValueError('visual expansion exceeds admission budget')
    return dict(original_bytes=original, expanded_rgb_bytes=expanded,
                preprocessing_tensor_bytes=preprocess)
