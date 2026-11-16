# Playground capture lifecycle contract

This contract covers GAMST Playground local capture under #311/#332. The
browser and local dashboard may stream state snapshots while `beginCapture()`
is still awaiting permission, media acquisition, video playback or model
startup. A snapshot from before the current start operation must not be allowed
to stop a newly acquired camera or microphone stream.

## Required state ordering

The dashboard exposes a monotonic capture operation revision. It advances for
each accepted local start and stop operation and is returned with every state
snapshot. The client records the revision acknowledged by its current start.

- A snapshot with a lower revision than the current client operation is stale
  and is ignored for lifecycle transitions.
- A paused snapshot at the current or newer revision is authoritative and must
  release owned media, mark capture stopped and update the UI.
- A running snapshot at the current or newer revision may confirm capture and
  update the local state.
- A failed start remains visible as a bounded capture error; it must not be
  disguised as a successful camera session.

The revision is lifecycle metadata only. It does not alter capture timestamps,
evidence IDs, packet contents, privacy consent or provider behavior.

## Acceptance evidence

Issue #332 requires a focused regression that delivers a stale paused snapshot
after a successful current start and verifies that media remains active, plus a
post-stop paused snapshot that verifies owned tracks are released. Existing
dashboard and browser tests must continue to pass. This issue does not claim
camera hardware coverage on unsupported hosts or model quality.
