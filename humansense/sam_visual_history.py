"""Bounded original-capture consistency for a serialized visual consumer.

Use after packet/policy/binding validation. Staging is side-effect free; commit
only after successful inference/publication admission. Reset invalidates staged
work. This records evidence consistency, not true identity or track accuracy.
"""
from dataclasses import dataclass
import json
import threading

from .sam_visual_packet import _validate, visual_policy


@dataclass(frozen=True)
class _Pending:
    owner: object
    revision: int
    identity: str
    records: tuple
    last_sequence: int
    reason: str | None


class VisualCaptureHistory:
    CAPACITY = 512

    def __init__(self):
        self._lock=threading.RLock()
        self._owner=object()
        self._revision=0
        self._identity=None
        self._records={}
        self._last=-1
        self._reason='initial'

    def reset(self):
        with self._lock:
            self._revision+=1
            self._identity=None
            self._records={}
            self._last=-1
            self._reason='reset'

    def stage(self, header):
        # The packet decoder has already checked actual payload hashes; this
        # structural recheck does not replace that requirement.
        _validate(header)
        identity=json.dumps(dict(policy=visual_policy(header),
            subject=header['audio']['metadata']['subject_id'],
            audio_epoch=header['audio']['metadata']['sensor']['stream_epoch'],
            visual_epoch=header['visual_sensor']['stream_epoch']),
            sort_keys=True,separators=(',',':'),allow_nan=False)
        incoming=[]
        for record in header['frames']:
            incoming.append((record['capture_sequence'],
                (tuple(record['interval_us']),record['rgb_sha256'],record['mask_sha256'])))
        with self._lock:
            changed=self._identity is not None and identity!=self._identity
            entries={} if changed else dict(self._records)
            last=-1 if changed else self._last
            reason='reconnect' if changed else self._reason
            for sequence,fingerprint in incoming:
                if sequence in entries:
                    if entries[sequence]!=fingerprint:
                        raise ValueError('original visual capture ID changed evidence')
                else:
                    if sequence<=last:
                        raise ValueError('unretained or late original capture cannot be verified')
                    if last >= 0:
                        previous_interval = entries[last][0]
                        interval = fingerprint[0]
                        if interval[0] < previous_interval[0] or interval[1] <= previous_interval[1]:
                            raise ValueError('new visual capture time regressed without epoch reset')
                    entries[sequence]=fingerprint
                    last=sequence
            # Old captures outside this bounded retained inventory cannot be
            # admitted as new merely because their hash has been forgotten.
            while len(entries)>self.CAPACITY:
                del entries[next(iter(entries))]
            return _Pending(self._owner,self._revision,identity,tuple(entries.items()),last,reason)

    def commit(self, pending):
        with self._lock:
            if (type(pending) is not _Pending or pending.owner is not self._owner
                    or pending.revision!=self._revision):
                raise ValueError('visual evidence transaction invalidated')
            self._identity=pending.identity
            self._records=dict(pending.records)
            self._last=pending.last_sequence
            self._revision+=1
            self._reason=None
