# Aligned timeline: `get_timeline`

Modules run at different speeds. Records keep their capture time, so what each module said about moment T can be put
in one row, with an explicit answer to "was it done with T?". `get_timeline` (SDK and MCP) computes that from retained
records. It runs no model, starts nothing, stores nothing.

## Which read to use

| Read | Order | What it is for |
|---|---|---|
| `get_context` | newest value per capability | "now", cheap; the values are from different moments and nothing says what is still being processed |
| `inspect_interval` | every record in an interval | full evidence of one interval, unaligned, unbounded by bytes |
| `get_observation_page` | ingestion order, cursor | replay every record once |
| `build_temporal_context` (packet) | events up to a cutoff | signed packet of at most 64 events with relations and contradictions between them |
| `count_events` | one capability | how many, with evidence |
| **`get_timeline`** | capture time, all capabilities | compare modules at the same moment; read only settled time; learn what changed since the last read |

## Call

```python
answer = await human.get_timeline(window='10s', step_ms=200)                    # or start_ms=..., end_ms=...
answer = await human.get_timeline(window='10s', settled_only=True)              # only rows every module is done with
answer = await human.get_timeline(start_ms=a, end_ms=b, as_of_ms=known)         # what was known then
delta  = await human.get_timeline(window='10s', cursor=answer['next_cursor'])   # only what changed
```

Other arguments: `subject_id`, `capabilities` (capability IDs, signal names or Pipeline module names; default all),
`max_bytes` (4096..262144, default 65536, the whole answer as ASCII JSON). `step_ms` 20..60000, at most 601 rows, 32
columns. Rows are `[t, t + step)` on an absolute grid (multiples of `step_ms`), so the same moment is the same row in
every read. The end is never later than now.

Answer: `columns` (one per capability, source, sensor and clock), `rows`, `evidence` (values, named by `ref`), and

- `complete_until_ms`, overall and per column; `rows_complete` / `rows_total`;
- row: `t_ms`, `complete`, `revision` (a hash of the content: same content, same revision), `known_ms` (when the newest
  evidence in it was learned), `cells`;
- cell: `status`, `ref` (the value), `records` (more than one in this step: the last is shown), `intervals` (refs),
  `age_ms` (the value is from an earlier step), `across_gap`, `reason`;
- evidence: `at_ms`, or `start_ms` and `end_ms`; `value`; `known_ms`; `evidence_id`; `quality`, `level` (only if given);
  `supersedes`, `revision`, `depends_on`, `depends_on_changed`; `flags` (`truncated`, `raw_fields_removed`, `value_omitted`).

| Status | Meaning |
|---|---|
| `exact` | a record captured in this row, or an interval record that overlaps it |
| `gap` | captured, declared lost, never processed |
| `off` | the module was switched off or paused (`reason`) |
| `pending` | the module has not finished this time; `ref` is its last value if still fresh |
| `sampled` | the module looks at one frame per interval by declaration; this row is between two |
| `held` | nothing new; the last value is within its `freshness_ms` |
| `none` | processed, nothing to report |
| `unknown` | no completeness information, or records of that time were removed |

Interval records (utterances, words, matches, semantic windows) are shown where they happened, not where they arrived:
`value.start_ms/end_ms`, `value.window`, and items of `events`, `words`, `segments` with their own times. An event
without times is a point at its record's capture time. Values are summarized and never interpolated (there is no
interpolation option). Joints, embeddings, media, and unregistered values whose form GAMST does not know never appear.

## Complete

Completeness comes from the Pipeline, through one function (`gamst.timeline.progress`). It reads per module
`complete_until_ms`, `gap_intervals`, `gaps`, `sampled`, `enabled`, `paused` and, if present, `complete_from_ms`
(where the account starts) and `holding_from_ms` (oldest capture time of evidence a module says it still holds in `pending_since_ms`, e.g. an open utterance).
What is missing is `unknown`. Without `complete_from_ms`, rows before the point where a module was first seen complete
are `unknown`.

`complete_until_ms` of a column is the Pipeline's value minus the capability's declared `lookahead_ms` and (window mode)
`window_ms`, and stops before a `partial` utterance; `processed_until_ms` shows the Pipeline's own value when it differs.
`asr_words` holds from the start of an utterance while its engine reports partial text; `late_intervals: true` marks a column whose intervals are reported when they are over (speech, matches) with no declared
bound: a complete row there can still gain an interval. The cursor reports that.

A row is `complete` only if every column is done with it. A column that is off does not hold a row back. Columns with
unknown completeness, an uncalibrated clock or removed records do.

## Two times and revisions

- **Capture time** is when it happened. **Knowledge time** (`known_ms`) is when the Store learned it. `as_of_ms` is a
  knowledge time: records learned later, later revisions, later calibrations and later completeness are left out. Reading
  with `as_of_ms` equal to an earlier answer's `known_ms` returns the same rows and evidence, as long as nothing was removed.
- **Records are append-only.** A correction is a new record with the fields of `TemporalEvent`: `supersedes` (the ID of
  the first record) and `revision`. `push_observation(..., supersedes=record_id)`; `retract_observation(record_id, reason)`
  adds a revision that says it was not there. Utterances keep `utterance_id` / `revision`.
- `quality={'status': ..., 'level': ...}`: the level is stored and shown as given. Nothing produces one today and nothing
  reacts to it.
- `get_context`, the temporal packet, `count_events` and the timeline use the current revision. `get_observations`,
  `get_observation_page` and `inspect_interval` return every revision.
- `count_events` takes `as_of_ms` (window and gaps as of then) and never counts an event of a superseded record.
- `depends_on` lists the record IDs a result says it came from; `depends_on_changed` those revised or retracted since.
  GAMST does not recompute a module.

## Delta reads

With `cursor`, rows are compared as of the two knowledge times. `change` is `added`, `revised` or `retracted` (evidence is
gone and nothing replaced it). A revised row has the changed cells in `cells` and the old ones in `was`; old evidence carries
`current: false`. Rows already delivered stay covered until the 601 row limit, also after the window has moved on.
`affected` lists capture intervals before the cursor's end that records learned since then added, revised or retracted, at
any age (at most 64, with `omitted`). `has_more: true`: call again with `next_cursor`; the pages are one consistent answer.

Nothing is skipped silently: `not_covered` names rows no longer compared; `restart: true` with `reason` (`retention`:
records were removed before they were read; `completeness_history`: older than the kept five minutes) means read again
without a cursor; a cursor of another query, another process or before `delete_data` is an error.

## Clocks

Preferred: a server echoes the capture time it was sent, and the record is in local capture time; no clock sync.
A source with its own clock sets `SensorMetadata.clock_id` (and `clock_epoch`, new after a restart) and the application
calls `human.add_clock_sample(clock_id, source_ms, local_ms, clock_epoch=...)`; at 8 samples `TemporalAligner`'s
linear fit places its records, with the fit in the column. Before that, its cells are `unknown` with
`reason: uncalibrated_clock`, and `uncalibrated_clocks` names it. `provider:*` clocks (local worker operations) are not source clocks.

## Cost

Synthetic store, 5 capabilities, 60 s, M2 Pro, Python 3.11, load average about 5 from other work. Medians of 9, SDK call
without MCP JSON.

| Read | 1k records | 5k | 10k |
|---|---|---|---|
| 60 s at 100 ms, first 64 KB page | 29 ms | 95 ms | 204 ms |
| 60 s at 1 s | 25 ms | 89 ms | 196 ms |
| 60 s at 100 ms, one capability | 14 ms | 41 ms | 89 ms |
| last 10 s at 200 ms (the default) | 11 ms | 20 ms | 35 ms |
| delta of that | 12 ms | 27 ms | 44 ms |
| delta covering 601 rows | 81 ms | 232 ms | 366 ms |

The event loop does not wait for a read. It runs the queries in steps of at most 200 records (at most 4.6 ms each,
measured); a thread decodes the records, builds the rows and gives the interpreter back every 0.3 ms. The read is one
state: records learned after it began are not in it; if records are removed meanwhile, `removed_until_ms` says so; if data
is deleted or the runtime closed, the read raises. One read runs at a time. With a 30 fps Pipeline on a 10k store the longest loop
stall during 15 reads was 8 ms and the longest frame interval 37.5 ms (33.3 without reads); before, a read held the loop
for its whole time, up to 529 ms, and reads made one after another added up (1790 ms). Memory and file stores work the same.
The building thread does slow other Python code while it runs.

A 64 KB page held 63 rows of 5 capabilities at 1k records and 27 at 10k (a value and up to 8 intervals per cell). An idle
delta was 2.1 KB.

## Limits

- Completeness at a past knowledge time is sampled (every 200 ms while records arrive, and at every read), in memory only,
  about five minutes. Between samples a row may look less complete than it was, never more. After a restart it is unavailable.
- Records written before knowledge time was stored count as known from when the file was opened.
- Retention is by capture time in the record's own clock and by count; what it removed makes rows `unknown`.
- Signals only: `observation.event` records and the semantic state chain (`get_semantic_state`) are not columns.
- `get_trends` averages every stored revision. A Watch does not un-fire when a record is corrected.
- An interval record committed more than five minutes after the window is not found. At most 20000 records are scanned per
  window (`scan_truncated`).
- A module that failed or was off has a gap; the Pipeline lists its last 64, gap records in the Store cover older ones,
  merged per second, so a row between two merged gaps can read `gap` although it was processed and empty.
- No model, network source or real remote clock was run; the tests use fake modules.
