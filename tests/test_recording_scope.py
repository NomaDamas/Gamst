import asyncio
from dataclasses import FrozenInstanceError
import unittest

from humansense.recording_scope import (RecordingScope, RecordingScopeOwner,
    evaluation_owner, resolve_evaluation_observer)


class Owner(RecordingScopeOwner):
    def __init__(self):
        self.handles = []
        self.calls = []

    def issue(self):
        scope = RecordingScope(self, len(self.handles), 'a'*64, object())
        self.handles.append(scope)
        return scope

    def validate_scope(self, scope):
        if not any(scope is value for value in self.handles):
            raise ValueError('unissued scope')

    def reserve_call(self, *, evaluation_scope):
        self.validate_scope(evaluation_scope)
        self.calls.append(evaluation_scope.scope_index)


class ScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_capture_is_immutable_and_one_owner(self):
        owner = Owner()
        handles = [owner.issue() for _ in range(65)]
        self.assertTrue(all(evaluation_owner(h) is owner for h in handles))
        release = asyncio.Event()

        async def reserve(handle):
            captured = resolve_evaluation_observer(owner, handle)
            await release.wait()
            captured.reserve_call()

        jobs = [asyncio.create_task(reserve(h)) for h in handles]
        await asyncio.sleep(0)
        with self.assertRaises(FrozenInstanceError):
            handles[0].scope_index = 64
        release.set()
        await asyncio.gather(*jobs)
        self.assertEqual(owner.calls, list(range(65)))

    def test_foreign_forged_and_missing_scopes_never_reserve(self):
        owner, foreign = Owner(), Owner()
        handle = owner.issue()
        forged = RecordingScope(owner, handle.scope_index, handle.scope_sha256,
            handle._token)
        for candidate in (forged, foreign.issue()):
            with self.assertRaises(ValueError):
                resolve_evaluation_observer(owner, candidate)
        with self.assertRaises(ValueError):
            resolve_evaluation_observer(owner)
        with self.assertRaises(ValueError):
            resolve_evaluation_observer(handle, owner.issue())
        with self.assertRaises(ValueError):
            handle.reserve_call(evaluation_scope=handle)
        self.assertEqual(owner.calls, [])

    def test_legacy_identity_and_fake_owner_attribute_remain_legacy(self):
        class Legacy:
            evaluation_owner = object()
        legacy = Legacy()
        self.assertIs(evaluation_owner(legacy), legacy)
        self.assertIs(resolve_evaluation_observer(legacy), legacy)
        self.assertIsNone(resolve_evaluation_observer(None))
        with self.assertRaises(ValueError):
            resolve_evaluation_observer(legacy, Owner().issue())


if __name__ == '__main__':
    unittest.main()
