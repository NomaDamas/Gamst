"""Explicit immutable recording ownership, independent of artifact wire formats."""
from dataclasses import dataclass


class RecordingScopeOwner:
    """Supported SDK owner contract; concrete sessions validate their own tokens.

    This base deliberately grants no admission and supplies no ambient scope.
    Artifact/session implementations retain all limits and privacy obligations.
    """

    def validate_scope(self, scope):
        raise NotImplementedError

    def validate_backend(self, scope, backend):
        raise NotImplementedError

    def validate_worker_scope(self, scope):
        raise NotImplementedError

    def validate_worker_configuration(self, scope, provider, revision, config):
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class RecordingScope:
    """Opaque owner-issued handle. Construction alone never grants admission."""

    owner: RecordingScopeOwner
    scope_index: int
    scope_sha256: str
    _token: object
    binding_index: object = None

    @property
    def evaluation_owner(self):
        self.owner.validate_scope(self)
        return self.owner

    def reserve_call(self, *args, **kwargs):
        self.owner.validate_scope(self)
        if 'evaluation_scope' in kwargs:
            raise ValueError('fixed scope cannot be overridden')
        return self.owner.reserve_call(*args, evaluation_scope=self, **kwargs)

    def reserve_work(self, *args, **kwargs):
        self.owner.validate_scope(self)
        if 'evaluation_scope' in kwargs:
            raise ValueError('fixed scope cannot be overridden')
        return self.owner.reserve_work(*args, evaluation_scope=self, **kwargs)

    def invalidate(self, reason, subject_id=None):
        # HumanSense registers the root owner, so this delegate is not another
        # privacy registry entry and cannot narrow a global lifecycle operation.
        self.owner.invalidate(reason, subject_id=subject_id)

    def binding_owner(self):
        self.owner.validate_scope(self)
        return self.owner.binding_owner(self)


def evaluation_owner(observer):
    """Resolve only supported handles, preserving legacy observer identity."""
    if type(observer) is RecordingScope:
        if not isinstance(observer.owner, RecordingScopeOwner):
            raise TypeError('supported recording owner required')
        observer.owner.validate_scope(observer)
        return observer.owner
    return observer


def validate_evaluation_backend(observer, backend):
    """Check actual callable/provider identity only for recording observers."""
    if type(observer) is RecordingScope:
        owner = evaluation_owner(observer)
        owner.validate_backend(observer, backend)


def validate_evaluation_worker_scope(observer, scope=None):
    """Resolve explicit process lifecycle ownership before worker startup."""
    resolved = resolve_evaluation_observer(observer, scope)
    if type(resolved) is RecordingScope:
        if resolved.scope_index != 0:
            raise ValueError('worker lifecycle requires immutable stream scope')
        resolved.owner.validate_worker_scope(resolved)
    return resolved


def validate_evaluation_worker_configuration(observer, provider, revision, config):
    if type(observer) is RecordingScope:
        observer.owner.validate_worker_configuration(observer, provider, revision, config)


def resolve_evaluation_observer(observer, evaluation_scope=None):
    """Capture a fixed scope before await; foreign or omitted scopes fail closed."""
    owner = evaluation_owner(observer)
    if evaluation_scope is not None:
        if type(evaluation_scope) is not RecordingScope:
            raise TypeError('immutable recording scope required')
        if not isinstance(owner, RecordingScopeOwner):
            raise ValueError('recording scope requires a recording owner')
        if evaluation_scope.owner is not owner:
            raise ValueError('recording scope belongs to another owner')
        owner.validate_scope(evaluation_scope)
        if type(observer) is RecordingScope and observer is not evaluation_scope:
            raise ValueError('fixed recording scope cannot be replaced')
        return evaluation_scope
    if isinstance(owner, RecordingScopeOwner):
        if type(observer) is not RecordingScope:
            raise ValueError('explicit recording scope required')
        owner.validate_scope(observer)
    return observer
