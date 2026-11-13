# Object observation memory

Object memory retains evidence for previously observed objects after the latest-signal store evicts their original records. It does not infer an unseen object or claim that a last-known location is current. This initial path uses provider track IDs; visual embedding association and world-coordinate mapping remain separate provider integration work.

Publish a registered capability whose output contains an `objects` array, then call `await human.remember_objects(record['id'])`. Each object requires string `track_id` and `label`; optional fields are `attributes` (object), `location` (object) and `relations` (array). Locations require the sensor's `coordinate_frame`. `objects_field` selects another declared array field when adapting a provider.

`complete_snapshot=True` means absent tracks are no longer visible in this source snapshot. Set it to false for partial/cropped observations that cannot establish absence elsewhere. Insufficient-quality observations cannot update memory. Out-of-order capture time/sequence snapshots are rejected before state mutation. No implicit object extraction from arbitrary model prose occurs.

`await human.find_objects(text="cup")` returns matching labels/attributes/relations and retained history. It includes entity ID, last-seen time, source, coordinate frame, model provenance and an evidence snapshot with the original record ID. `current_location` is always unknown in this path. `currently_visible` requires a sufficiently recent observation in the latest applicable snapshot; it becomes false when sensing pauses, resets or closes. This freshness flag is not proof that the object remained physically present between frames.

Identity is scoped to subject, source, physical sensor, capability, stream epoch and provider track ID. `SensorStreams` assigns a new epoch on every open, preventing reused track IDs after reconnect from silently becoming the same remembered entity. Direct observations without an explicit stream epoch use the runtime session/generation at ingestion. Replaying old evidence after reset cannot reactivate visibility. Similar-looking tracks remain separate; text search does not prove identity.

Defaults retain 256 entities, eight evidence snapshots per entity, 24 hours of history and up to 4 MiB of serialized entity records. Visibility freshness defaults to one second. `ObjectMemory` accepts explicit bounds for applications that replace `human.object_memory`. Returned data is copied. Memory is session-local, not persisted across SDK restarts. `delete_data(subject_id)` removes only that subject's memory; global deletion and close clear all memory. Short latest-record retention does not erase the memory's copied evidence snapshot.

This implementation addresses the memory contract and lifecycle portion of #9/#39. Actual detector/embedding association, optional world mapping and public MCP/transport exposure retain their own execution and integration gates.
