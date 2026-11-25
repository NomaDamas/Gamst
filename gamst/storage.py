"""Small local store containing derived records only, never camera/audio buffers."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import json
import sqlite3
import secrets
import time
from pathlib import Path


class Store:
    _KNOWN = "COALESCE(known_ms,?)<=?"

    def __init__(self, uri=":memory:", retention_hours=24, *, max_records=5000, max_payload_bytes=32*1024*1024):
        if retention_hours <= 0:
            raise ValueError("retention_hours must be positive")
        if any(type(v) is not int or v <= 0 for v in (max_records, max_payload_bytes)):
            raise ValueError("storage limits must be positive integers")
        self.max_records, self.max_payload_bytes = max_records, max_payload_bytes
        self.evicted_records = self.oversized_records = 0
        self.retention_revision = 0
        # What removal (eviction, pruning, deletion) took away: newest capture time, newest knowledge time.
        self.lost_until_ms = self.lost_known_ms = None
        self._known_floor = 0
        self._opened_ms = time.time_ns() // 1_000_000   # records without a known_ms were known by then
        self._cursor_key = secrets.token_bytes(32)
        self.retention_ms = int(retention_hours * 3_600_000)
        path = str(uri)
        if path.startswith("sqlite:///"):
            path = path[len("sqlite:///"):]
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA secure_delete=ON")
        self.connection.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, subject TEXT NOT NULL, ts INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS subject_time ON records(subject,ts)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS record_time ON records(ts)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS subject_kind_time ON records(subject,kind,ts)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS signal_name_time ON records(subject,kind,json_extract(payload,'$.observation.name'),ts)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS numeric_signal_time ON records(subject,ts) WHERE kind='observation.signal' AND json_type(payload, '$.observation.value') IN ('integer','real')")
        # Independent AUTOINCREMENT positions never reuse a deleted record's offset.
        # Backfill old databases in their original insertion order, without rewriting payloads.
        with self.connection:
            self.connection.execute("CREATE TABLE IF NOT EXISTS record_offsets (position INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT UNIQUE NOT NULL, known_ms INTEGER)")
            # known_ms: when this store learned the record; NULL for records written before it was kept.
            if not any(row[1] == 'known_ms' for row in self.connection.execute("PRAGMA table_info(record_offsets)")):
                self.connection.execute("ALTER TABLE record_offsets ADD COLUMN known_ms INTEGER")
            self.connection.execute("CREATE INDEX IF NOT EXISTS known_time ON record_offsets(known_ms)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS record_revisions ON records(json_extract(payload,'$.supersedes')) WHERE json_type(payload,'$.supersedes')='text'")
            self.connection.execute("CREATE INDEX IF NOT EXISTS clocked_signals ON records(subject,ts) WHERE json_type(payload,'$.observation.sensor.clock_id')='text'")
            self.connection.execute("DELETE FROM record_offsets WHERE record_id NOT IN (SELECT id FROM records)")
            self.connection.execute("INSERT INTO record_offsets(record_id) SELECT id FROM records WHERE id NOT IN (SELECT record_id FROM record_offsets) ORDER BY rowid")
        self._reload_sizes()
        if self.removed_records():
            oldest = self.connection.execute("SELECT MIN(ts) FROM records").fetchone()[0]
            self.lost_until_ms = self.known_now() if oldest is None else oldest - 1
        with self._transaction():
            self._enforce_limits()

    @classmethod
    def open_read_only(cls, uri):
        """An existing store file for queries only: no pruning, no size enforcement, no schema change."""
        path = str(uri)[len("sqlite:///"):] if str(uri).startswith("sqlite:///") else str(uri)
        if not Path(path).is_file():
            raise FileNotFoundError(f"no store at {path}")
        store = cls.__new__(cls)
        store.connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        return store

    def removed_records(self):
        """How many records were ever written to this file and are no longer in it (evicted, pruned, deleted)."""
        written = self.connection.execute("SELECT seq FROM sqlite_sequence WHERE name='record_offsets'").fetchone()
        kept = self.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        return (written[0] if written else 0) - kept

    def _reload_sizes(self):
        self._sizes = OrderedDict((row[0], row[1]) for row in self.connection.execute(
            "SELECT id,length(CAST(payload AS BLOB)) FROM records ORDER BY rowid"))
        self.payload_bytes = sum(self._sizes.values())

    @contextmanager
    def _transaction(self):
        evicted, revision = self.evicted_records, self.retention_revision
        try:
            with self.connection:
                yield
        except BaseException:
            # SQLite rolls back both tables; rebuild only on failure, not every frame.
            self._reload_sizes()
            self.evicted_records = evicted
            self.retention_revision = revision+1
            raise

    def known_now(self):
        """This store's knowledge clock (ms): wall time, never backwards."""
        return max(time.time_ns() // 1_000_000, self._known_floor)

    def read_mark(self):
        """The knowledge time of a read: everything stored so far is known by it, anything later is after it."""
        known = self.known_now()
        self._known_floor = known + 1
        return known

    def _lost(self, where, args):
        capture, known = self.connection.execute(
            "SELECT MAX(ts),MAX(known_ms) FROM records JOIN record_offsets ON record_id=id WHERE " + where, args).fetchone()
        if capture is not None:
            self.lost_until_ms = max(capture, self.lost_until_ms or 0)
        if known is not None:
            self.lost_known_ms = max(known, self.lost_known_ms or 0)

    def _enforce_limits(self):
        while len(self._sizes) > self.max_records or self.payload_bytes > self.max_payload_bytes:
            key, size = self._sizes.popitem(last=False)
            self._lost("id=?", (key,))
            self.connection.execute("DELETE FROM records WHERE id=?", (key,))
            self.connection.execute("DELETE FROM record_offsets WHERE record_id=?", (key,))
            self.payload_bytes -= size
            self.evicted_records += 1
            self.retention_revision += 1

    def status(self):
        return dict(retention_ms=self.retention_ms, records=len(self._sizes),
                    max_records=self.max_records, payload_bytes=self.payload_bytes,
                    max_payload_bytes=self.max_payload_bytes, evicted_records=self.evicted_records,
                    oversized_records=self.oversized_records)

    def add(self, record):
        payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
        size = len(payload.encode('utf-8'))
        if size > self.max_payload_bytes:
            self.oversized_records += 1
            self.retention_revision += 1
            return False
        with self._transaction():
            self.connection.execute("INSERT INTO records VALUES(?,?,?,?,?)", (
                record["id"], record["subject_id"], record["timestamp_ms"], record["type"],
                payload))
            self._known_floor = self.known_now()
            self.connection.execute("INSERT INTO record_offsets(record_id,known_ms) VALUES(?,?)",
                                    (record["id"], self._known_floor))
            self._sizes[record["id"]] = size
            self.payload_bytes += size
            self._enforce_limits()
        return True

    def query(self, subject=None, start=None, end=None, kind=None, limit=1000, name=None,
              registered_only=False, capability_id=None, source_id=None, sensor_id=None,
              any_of=None, spans_only=False, clock_id=None, known_after=None, known_until=None, with_known=False):
        """Newest ``limit`` records, oldest first. ``any_of``: capability IDs, signal names or (unregistered)
        model IDs. ``spans_only``: values that name an interval. ``known_after`` / ``known_until`` bound when
        the store learned the record (records without that time are left out); ``with_known`` returns
        ``(record, known_ms)`` pairs."""
        clauses, args = [], []
        for column, value, operator in (("subject", subject, "="), ("ts", start, ">="), ("ts", end, "<="), ("kind", kind, "=")):
            if value is not None:
                clauses.append(f"{column} {operator} ?")
                args.append(value)
        if name is not None:
            clauses.append("json_extract(payload, '$.observation.name') = ?")
            args.append(name)
        if registered_only:
            clauses.append("json_type(payload, '$.observation.capability') = 'object'")
        for field, value in (("capability.id", capability_id), ("sensor.source_id", source_id),
                             ("sensor.sensor_id", sensor_id)):
            if value is not None:
                clauses.append("json_extract(payload, '$.observation."+field+"') = ?")
                args.append(value)
        if any_of is not None:
            slots = ",".join("?" for _ in any_of)
            clauses.append(f"(json_extract(payload,'$.observation.capability.id') IN ({slots})"
                           f" OR json_extract(payload,'$.observation.name') IN ({slots})"
                           f" OR (json_type(payload,'$.observation.capability') IS NULL"
                           f" AND json_extract(payload,'$.models[0].id') IN ({slots})))")
            args += [*any_of]*3
        if spans_only:
            clauses.append("instr(json_extract(payload,'$.observation.value'),'end_ms')>0")
        if clock_id is not None:
            clauses += ["json_type(payload,'$.observation.sensor.clock_id')='text'",
                        "json_extract(payload,'$.observation.sensor.clock_id')=?"]
            args.append(clock_id)
        if known_after is not None:
            clauses.append("known_ms>?")
            args.append(known_after)
        if known_until is not None:
            clauses.append(self._KNOWN)
            args += [self._opened_ms, known_until]
        joined = with_known or known_after is not None or known_until is not None
        sql = "SELECT payload" + (",known_ms" if joined else "") + " FROM records"
        if joined:
            sql += " JOIN record_offsets ON record_id=id"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC, records.rowid DESC LIMIT ?"
        args.append(max(0, min(int(limit), 100_000)))
        rows = self.connection.execute(sql, args).fetchall()
        if with_known:
            return [(json.loads(row[0]), row[1]) for row in reversed(rows)]
        return [json.loads(row[0]) for row in reversed(rows)]

    def revisions(self, subject, known_until=None, *, of=None):
        """Stored revisions and retractions, ``(identity, id, revision, retracted)`` in the order learned.

        ``of``: only those of one identity."""
        sql = ("SELECT json_extract(payload,'$.supersedes'),id,json_extract(payload,'$.revision'),"
               "kind='observation.retraction' FROM records JOIN record_offsets ON record_id=id"
               " WHERE json_type(payload,'$.supersedes')='text' AND subject=?")
        args = [subject]
        if of is not None:
            sql, args = sql + " AND json_extract(payload,'$.supersedes')=?", [*args, of]
        if known_until is not None:
            sql, args = sql + " AND " + self._KNOWN, [*args, self._opened_ms, known_until]
        return self.connection.execute(sql + " ORDER BY position", args).fetchall()

    def known_times(self, ids):
        return dict(self.connection.execute("SELECT record_id,known_ms FROM record_offsets WHERE record_id IN"
                                            " (SELECT value FROM json_each(?))", (json.dumps(list(ids)),)))

    def source_clocks(self, subject, known_until=None):
        """Sources whose records are in a clock of their own (a local worker's operation clock is not one)."""
        sql = ("SELECT json_extract(payload,'$.observation.sensor.clock_id'),"
               "json_extract(payload,'$.observation.sensor.clock_epoch'),json_extract(payload,'$.observation.capability.id'),"
               "json_extract(payload,'$.observation.name'),json_extract(payload,'$.observation.sensor.source_id'),"
               "json_extract(payload,'$.observation.sensor.sensor_id'),COUNT(*),MAX(ts)"
               " FROM records JOIN record_offsets ON record_id=id"
               " WHERE subject=? AND kind='observation.signal' AND json_type(payload,'$.observation.sensor.clock_id')='text'"
               " AND json_extract(payload,'$.observation.sensor.clock_id') NOT LIKE 'provider:%'")
        args = [subject]
        if known_until is not None:
            sql, args = sql + " AND " + self._KNOWN, [*args, self._opened_ms, known_until]
        names = ('clock_id', 'clock_epoch', 'capability_id', 'name', 'source_id', 'sensor_id', 'records', 'newest_ms')
        return [dict(zip(names, row)) for row in
                self.connection.execute(sql + " GROUP BY 1,2,3,4,5,6 ORDER BY 8 DESC LIMIT 64", args)]

    def observation_page_rows(self, *, subject, after, limit, capability_id=None,
                              source_id=None, sensor_id=None, events_only=False):
        """Internal ingestion-order selection; caller validates filters/page limits."""
        upper = self.connection.execute("SELECT seq FROM sqlite_sequence WHERE name='record_offsets'").fetchone()
        upper = upper[0] if upper else 0
        clauses = ["r.subject=?", "r.kind='observation.signal'", "o.position>?", "o.position<=?",
                   "json_type(r.payload,'$.observation.capability')='object'"]
        args = [subject, after, upper]
        for field,value in (("capability.id",capability_id),("sensor.source_id",source_id),("sensor.sensor_id",sensor_id)):
            if value is not None:
                clauses.append("json_extract(r.payload,'$.observation."+field+"')=?")
                args.append(value)
        if events_only:
            clauses += ["json_type(r.payload,'$.observation.value.events')='array'",
                        "json_array_length(r.payload,'$.observation.value.events')>0"]
        sql = "SELECT o.position,r.payload FROM record_offsets o JOIN records r ON r.id=o.record_id WHERE "
        sql += " AND ".join(clauses) + " ORDER BY o.position LIMIT ?"
        rows = self.connection.execute(sql,args+[limit]).fetchall()
        return [(position,json.loads(payload)) for position,payload in rows],upper

    def get_many(self, ids):
        if not ids:
            return []
        slots = ",".join("?" for _ in ids)
        rows = self.connection.execute(f"SELECT payload FROM records WHERE id IN ({slots})", list(ids)).fetchall()
        return [json.loads(row[0]) for row in rows]

    @staticmethod
    def _context_exclusion(exclusion):
        # Only Gamst's SDK-issued view supplies this private query seam.
        # COALESCE preserves legacy records without registered capability fields.
        if exclusion is None:
            return '', ()
        return (" AND COALESCE((json_extract(payload,'$.observation.capability.id')=?"
                " AND json_extract(payload,'$.models[0].id')=?"
                " AND json_extract(payload,'$.models[0].version')=?),0)=0", tuple(exclusion))

    def superseded(self, subject, known_until=None):
        """IDs of records that a later revision or retraction replaced."""
        winner, ids = {}, set()
        for identity, record_id, revision, _ in self.revisions(subject, known_until):
            ids.update((identity, record_id))
            if identity not in winner or revision >= winner[identity][0]:
                winner[identity] = revision, record_id
        return ids - {record_id for _, record_id in winner.values()}

    def latest_signals(self, subject, start, end, *, _exclude_capability=None, known_until=None):
        """Latest current legacy name or registered capability/source/sensor identity."""
        clause, args = self._context_exclusion(_exclude_capability)
        known, bounds = ("", ()) if known_until is None else (self._KNOWN, (self._opened_ms, known_until))
        if self.connection.execute("SELECT EXISTS(SELECT 1 FROM records WHERE json_type(payload,'$.supersedes')='text')").fetchone()[0]:
            clause += (" AND NOT EXISTS (SELECT 1 FROM records AS newer JOIN record_offsets ON record_id=newer.id"
                       " WHERE json_type(newer.payload,'$.supersedes')='text' AND json_extract(newer.payload,'$.supersedes')"
                       "=COALESCE(json_extract(records.payload,'$.supersedes'),records.id)"
                       " AND json_extract(newer.payload,'$.revision')>COALESCE(json_extract(records.payload,'$.revision'),0)"
                       + (" AND " + known if known else "") + ")")
            args = (*args, *bounds)
        if known:
            clause += f" AND id IN (SELECT record_id FROM record_offsets WHERE {known})"
            args = (*args, *bounds)
        rows = self.connection.execute("""
            SELECT payload FROM (
              SELECT payload, ROW_NUMBER() OVER (
                PARTITION BY json_extract(payload, '$.observation.name'),
                  json_extract(payload, '$.observation.capability.id'),
                  json_extract(payload, '$.observation.sensor.source_id'),
                  json_extract(payload, '$.observation.sensor.sensor_id')
                ORDER BY ts DESC, rowid DESC) AS rank
              FROM records WHERE subject=? AND kind='observation.signal' AND ts>=? AND ts<=?
            """ + clause + ") WHERE rank=1", (subject, start, end, *args)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def event_summary(self, subject, start, end, *, per_kind=8, _exclude_capability=None):
        clause, args = self._context_exclusion(_exclude_capability)
        counts = dict(self.connection.execute("""SELECT json_extract(payload,'$.event'), COUNT(*)
            FROM records WHERE subject=? AND kind='observation.event' AND ts>=? AND ts<=?
            """ + clause + " GROUP BY json_extract(payload,'$.event')", (subject, start, end, *args)).fetchall())
        rows = self.connection.execute("""SELECT payload FROM (
            SELECT payload, ts, rowid AS ordering, ROW_NUMBER() OVER (
              PARTITION BY json_extract(payload,'$.event') ORDER BY ts DESC,rowid DESC) AS rank
            FROM records WHERE subject=? AND kind='observation.event' AND ts>=? AND ts<=?
            """ + clause + ") WHERE rank<=? ORDER BY ts,ordering", (subject, start, end, *args, per_kind)).fetchall()
        return counts, [json.loads(row[0]) for row in rows]

    def repeated_text(self, subject, start, end, normalized, *, limit=16):
        args = (subject, start, end, normalized)
        condition = """subject=? AND kind='observation.signal' AND ts>=? AND ts<=?
            AND json_extract(payload,'$.observation.name')='text.utterance'
            AND json_extract(payload,'$.observation.role')='user'
            AND json_extract(payload,'$.observation.normalized')=?"""
        count = self.connection.execute("SELECT COUNT(*) FROM records WHERE "+condition, args).fetchone()[0]
        rows = self.connection.execute("SELECT id FROM records WHERE "+condition+" ORDER BY ts DESC,rowid DESC LIMIT ?", (*args, limit)).fetchall()
        return count, [row[0] for row in reversed(rows)]

    def count(self, subject, start, end, *, _exclude_capability=None):
        clause, args = self._context_exclusion(_exclude_capability)
        return self.connection.execute("SELECT COUNT(*) FROM records WHERE subject=? AND ts>=? AND ts<=?" + clause, (subject, start, end, *args)).fetchone()[0]

    def numeric_summary(self, subject, start, end, *, _exclude_capability=None):
        """Aggregate retained values in SQLite without decoding all stored JSON."""
        clause, args = self._context_exclusion(_exclude_capability)
        rows = self.connection.execute("""
          SELECT json_extract(payload, '$.observation.name') AS metric,
            json_extract(payload, '$.observation.modality') AS modality,
            json_extract(payload, '$.models[0].id') AS model_id,
            json_extract(payload, '$.models[0].version') AS model_version,
            SUM(CASE WHEN ts>=? THEN 1 ELSE 0 END),
            AVG(CASE WHEN ts>=? THEN json_extract(payload, '$.observation.value') END),
            SUM(CASE WHEN ts<? THEN 1 ELSE 0 END),
            AVG(CASE WHEN ts<? THEN json_extract(payload, '$.observation.value') END),
            MIN(CASE WHEN ts<? THEN ts END), MAX(CASE WHEN ts<? THEN ts END)
          FROM records WHERE subject=? AND kind='observation.signal' AND ts<=?
            AND json_type(payload, '$.observation.value') IN ('integer','real')
            AND json_extract(payload, '$.quality.status')='sufficient'
            AND json_extract(payload, '$.observation.name')!='statement.claim'
            AND json_extract(payload, '$.observation.name') NOT LIKE '%model_score'
        """ + clause + """
          GROUP BY metric, modality, model_id, model_version
          HAVING SUM(CASE WHEN ts>=? THEN 1 ELSE 0 END)>0
        """, (start, start, start, start, start, start, subject, end, *args, start)).fetchall()
        return rows

    def prune(self, now_ms):
        with self._transaction():
            expired = self.connection.execute("SELECT id FROM records WHERE ts < ?", (now_ms-self.retention_ms,)).fetchall()
            if expired:
                self._lost("ts < ?", (now_ms-self.retention_ms,))
                self.retention_revision += 1
            for (key,) in expired:
                self.connection.execute("DELETE FROM record_offsets WHERE record_id=?", (key,))
                self.payload_bytes -= self._sizes.pop(key, 0)
            self.connection.execute("DELETE FROM records WHERE ts < ?", (now_ms - self.retention_ms,))

    def delete(self, subject=None):
        # Even a no-op deletion invalidates prior replay cursors, across subjects.
        self._cursor_key = secrets.token_bytes(32)
        self.retention_revision += 1
        with self._transaction():
            self._lost("1" if subject is None else "subject=?", () if subject is None else (subject,))
            keys = self.connection.execute("SELECT id FROM records" + (" WHERE subject=?" if subject is not None else ""),
                                           (subject,) if subject is not None else ()).fetchall()
            for (key,) in keys:
                self.connection.execute("DELETE FROM record_offsets WHERE record_id=?", (key,))
                self.payload_bytes -= self._sizes.pop(key, 0)
            if subject is None:
                result = self.connection.execute("DELETE FROM records")
            else:
                result = self.connection.execute("DELETE FROM records WHERE subject=?", (subject,))
        self.connection.execute("VACUUM")
        return result.rowcount

    def close(self):
        self.connection.close()
