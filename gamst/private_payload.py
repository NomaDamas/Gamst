"""Worker-private payloads: joint positions, identity embeddings and waveforms never become records.

Some models answer one module with data that must not reach the Store, a context or MCP. Such a worker
is opened with ``ProcessProvider.open(..., private=True)`` by the module that owns it and answers
``query`` only. Its response has two separate parts:

- a *public* value, validated by the profile's ``CapabilitySpec`` like any other output (counts, boxes);
- a *private* array, carried as bytes beside the JSON, with the shape declared for that profile in
  ``CONTRACTS``. It is never a ``CapabilitySpec`` field and never passes ``CapabilitySpec.validate``.

The owner gets a :class:`Private` object. It is not JSON, cannot be pickled, shows no content, and
``push_signal`` / ``push_observation`` refuse it. A module hands joints to other modules only as
:class:`PrivatePoses` on a derived input (``Pipeline`` never commits derived data). A private worker is
refused wherever its output would be committed: ``SensorStreams``, ``Pipeline.add_module``,
``Gamst.register``, ``WorkerModule``, ``role_module`` (so also ``ProfileRuntime``).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .capabilities import is_private_key
from .provider_wire import PAYLOAD_LIMIT

PROTOCOL = 'gamst.private.v1'


@dataclass(frozen=True)
class Contract:
    """Shape and value bounds of one profile's private array (always little-endian float32)."""
    kind: str                 # 'joints' | 'embedding' | 'waveform'
    points: int = 0           # joints per person
    columns: tuple = ()       # joints: what each point carries
    max_items: int = 0        # joints: people; embedding: vector length
    limit: float = 1e6        # largest absolute value
    private_only: bool = True # the profile has no use without its private payload

    def check_shape(self, shape, frame):
        """``frame``: the request's wire frame header. Returns the byte count."""
        if type(shape) is not list or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError('private payload shape must be a list of sizes')
        if self.kind == 'joints':
            valid = len(shape) == 3 and shape[0] <= self.max_items and shape[1:] == [self.points, len(self.columns)]
        elif self.kind == 'embedding':
            valid = len(shape) == 1 and (shape[0] == 0 or 16 <= shape[0] <= self.max_items)
        else:  # a waveform answers one block: nothing, or exactly as many samples
            samples = (frame['layout'].get('shape') or [0])[0] if frame else 0
            valid = len(shape) == 1 and shape[0] in (0, samples)
        size = 4 * math.prod(shape)
        if not valid or size > PAYLOAD_LIMIT:
            raise ValueError(f'private {self.kind} payload does not have its declared shape')
        return size

    def check_public(self, public, shape):
        """The two parts must describe the same answer."""
        if self.kind == 'joints':
            consistent = public is not None and len(public['poses']) == shape[0]
        elif self.kind == 'embedding':
            consistent = public is not None and public['usable'] == (shape[0] > 0)
        else:
            consistent = public is not None and public['samples_returned'] == shape[0]
        if not consistent:
            raise ValueError(f'private {self.kind} payload differs from its public result')


CONTRACTS = {
    'mediapipe.body.video': Contract('joints', 33, ('x', 'y', 'visibility', 'presence'), 4, 100., private_only=False),
    'sapiens2.pose': Contract('joints', 308, ('x', 'y', 'visibility'), 8, 100.),
    'speaker.embedding': Contract('embedding', max_items=4096),
    'face.embedding': Contract('embedding', max_items=4096),
    'body.embedding': Contract('embedding', max_items=4096),
    'target_speech.local': Contract('waveform', limit=1e3),
}
PRIVATE_ONLY = frozenset(profile for profile, contract in CONTRACTS.items() if contract.private_only)


def _no_copy(*_):
    raise TypeError('private payloads are not serialized or copied')


class Private:
    """One private array, held by the module that asked for it."""
    __slots__ = ('kind', '_array')

    def __init__(self, kind, array):
        self.kind, self._array = kind, array

    def take(self, kind):
        """The array, for the owning module; the holder is empty afterwards."""
        array, self._array = self._array, None
        if array is None or kind != self.kind:
            raise PermissionError(f'no private {kind} payload here')
        return array

    def __repr__(self):
        return f'<private {self.kind}>'

    __reduce__ = __reduce_ex__ = __copy__ = __deepcopy__ = _no_copy


class PrivatePoses:
    """Poses with joint positions on a derived input: read by modules, refused by every record."""
    __slots__ = ('poses',)

    def __init__(self, poses):
        self.poses = list(poses)

    def __repr__(self):
        return f'<private poses: {len(self.poses)}>'

    __reduce__ = __reduce_ex__ = __copy__ = __deepcopy__ = _no_copy


def poses_of(data):
    """The pose list of a derived joints frame or a legacy ``scene.poses`` record value, else None."""
    if isinstance(data, PrivatePoses):
        return data.poses
    return data if isinstance(data, list) else None


def public_pose(pose):
    """A pose without joint positions (what may be committed)."""
    return {key: value for key, value in pose.items() if key.lower() != 'landmarks' and not is_private_key(key)}


def landmarks(rows, columns):
    """One person's private rows as the landmark dicts pose modules read."""
    return [dict(dict(zip(columns, map(float, row)))) for row in rows]


def refuse(value):
    """Raise when ``value`` carries a private payload or a field named like one (bounded walk)."""
    pending, remaining = [value], 200_000
    while pending:
        item = pending.pop()
        remaining -= 1
        if remaining < 0:
            return  # larger than any record the Store accepts; the size bounds there apply
        if isinstance(item, (Private, PrivatePoses)):
            raise PermissionError('private worker payloads are never context')
        if isinstance(item, dict):
            for key, child in item.items():
                if isinstance(key, str) and is_private_key(key):
                    raise PermissionError(f'{key!r} is a private field and never context')
                pending.append(child)
        elif isinstance(item, (list, tuple)):
            pending.extend(child for child in item if isinstance(child, (dict, list, tuple, Private, PrivatePoses)))


def pack(profile, public, value, frame):
    """Worker side: (descriptor, bytes) of one answer's private part."""
    import numpy as np
    contract = CONTRACTS[profile]
    array = np.asarray(value, dtype='<f4')
    contract.check_shape(list(array.shape), frame)
    contract.check_public(public, array.shape)
    if not np.isfinite(array).all() or (array.size and np.abs(array).max() > contract.limit):
        raise ValueError('private payload must be finite and bounded')
    return dict(protocol=PROTOCOL, kind=contract.kind, shape=list(array.shape)), array.tobytes(order='C')


def expected_size(profile, descriptor, frame):
    """Owner side, before any byte is read: the declared size, or an error."""
    contract = CONTRACTS[profile]
    if type(descriptor) is not dict or set(descriptor) != {'protocol', 'kind', 'shape'} \
            or descriptor['protocol'] != PROTOCOL or descriptor['kind'] != contract.kind:
        raise ValueError('invalid private payload descriptor')
    return contract.check_shape(descriptor['shape'], frame)


def unpack(profile, public, descriptor, payload, frame):
    import numpy as np
    contract = CONTRACTS[profile]
    if len(payload) != expected_size(profile, descriptor, frame):
        raise ValueError('private payload length differs from its shape')
    contract.check_public(public, descriptor['shape'])
    array = np.frombuffer(payload, dtype='<f4').reshape(descriptor['shape']).copy()
    if not np.isfinite(array).all() or (array.size and np.abs(array).max() > contract.limit):
        raise ValueError('private payload must be finite and bounded')
    return Private(contract.kind, array)
