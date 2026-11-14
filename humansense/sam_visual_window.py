"""Admission of a complete visual window before native RGB gather/resize."""
from dataclasses import dataclass
import hashlib
import json

from .sam_audio_visual import align_visual_frames, visual_allocation_bounds
from .sam_visual_packet import unpack_visual_packet, visual_policy, validate_visual_policy


@dataclass(frozen=True)
class PreparedVisual:
    rgb: bytes
    masks: bytes
    shape: tuple
    indices: tuple
    intervals: tuple
    ages_sixth_us: tuple
    mapping_sha256: str
    image_size: int
    batch_size: int
    clock: tuple

    def provenance(self):
        return dict(mapping_sha256=self.mapping_sha256,
            original_intervals_us=[list(v) for v in self.intervals],
            selected_indices=list(self.indices), ages_sixth_us=list(self.ages_sixth_us),
            original_frames=self.shape[0], feature_frames=len(self.indices),
            image_size=self.image_size, encoder_batch_size=self.batch_size,
            mask_polarity='zero-kept', alignment='latest-certain-past',
            clock=dict(self.clock), age_reference='nominal-mapped-exposure-end',
            clock_verified=False, track_identity_verified=False)

    def arrays(self):
        import numpy as np
        n,c,h,w=self.shape
        return (np.frombuffer(self.rgb,dtype=np.uint8).reshape(n,c,h,w).copy(),
                np.frombuffer(self.masks,dtype=np.uint8).reshape(n,1,h,w).copy())


def prepare_visual(packet, *, policy, sample_count, hop_length, image_size, batch_size):
    """Validate original bytes and exact policy before admitting expanded work.

    image_size/batch_size must come from the instantiated verified native
    encoder, never from an untrusted request's claimed model dimensions.
    """
    header,pcm,rgb,masks=unpack_visual_packet(packet)
    canon=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
    if canon(visual_policy(header))!=canon(validate_visual_policy(policy)):
        raise ValueError('visual window differs from configured policy')
    if type(sample_count) is not int or len(pcm)!=sample_count*4:
        raise ValueError('visual window and actual audio sample count differ')
    if type(batch_size) is not int or not 1<=batch_size<=64:
        raise ValueError('positive native vision batch size1..64 required')
    clock=header['clock']
    intervals=[r['interval_us'] for r in header['frames']]
    alignment=align_visual_frames(intervals,
        audio_end_us=header['audio']['metadata']['timestamp_ms']*1000,
        sample_count=sample_count,hop_length=hop_length,
        camera_to_audio_offset_us=clock['camera_to_audio_offset_us'],
        clock_error_us=clock['clock_error_us'],max_age_us=clock['max_age_us'])
    visual_allocation_bounds(frame_count=len(intervals),height=header['height'],width=header['width'],
        feature_count=len(alignment.indices),image_size=image_size)
    mapping=dict(intervals_us=intervals,indices=alignment.indices,
                 feature_times_sixth_us=alignment.feature_times_sixth_us,
                 ages_sixth_us=alignment.ages_sixth_us,clock=clock)
    digest=hashlib.sha256(canon(mapping).encode()).hexdigest()
    return PreparedVisual(rgb,masks,(len(intervals),3,header['height'],header['width']),
        alignment.indices,tuple(tuple(v) for v in intervals),alignment.ages_sixth_us,
        digest,image_size,batch_size,tuple(sorted(clock.items())))
