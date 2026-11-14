"""Small local store containing derived records only, never camera/audio buffers."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import json
import sqlite3
import secrets
from pathlib import Path


class Store:
    def __init__(self, uri=":memory:", retention_hours=24, *, max_records=5000, max_payload_bytes=32*1024*1024):
        if retention_hours <= 0:
            raise ValueError("retention_hours must be positive")
        if any(type(v) is not int or v <= 0 for v in (max_records, max_payload_bytes)):
            raise ValueError("storage limits must be positive integers")
        self.max_records, self.max_payload_bytes = max_records, max_payload_bytes
        self.evicted_records = self.oversized_records = 0
        self.retention_revision = 0
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
            self.connection.execute("CREATE TABLE IF NOT EXISTS record_offsets (position INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT UNIQUE NOT NULL)")
            self.connection.execute("DELETE FROM record_offsets WHERE record_id NOT IN (SELECT id FROM records)")
            self.connection.execute("INSERT INTO record_offsets(record_id) SELECT id FROM records WHERE id NOT IN (SELECT record_id FROM record_offsets) ORDER BY rowid")
        self._reload_sizes()
        with self._transaction():
            self._enforce_limits()

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

    def _enforce_limits(self):
        while len(self._sizes) > self.max_records or self.payload_bytes > self.max_payload_bytes:
            key, size = self._sizes.popitem(last=False)
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
            self.connection.execute("INSERT INTO record_offsets(record_id) VALUES(?)", (record["id"],))
            self._sizes[record["id"]] = size
            self.payload_bytes += size
            self._enforce_limits()
        return True

    def query(self, subject=None, start=None, end=None, kind=None, limit=1000, name=None,
              registered_only=False, capability_id=None, source_id=None, sensor_id=None):
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
        sql = "SELECT payload FROM records"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC, rowid DESC LIMIT ?"
        args.append(max(0, min(int(limit), 100_000)))
        rows = self.connection.execute(sql, args).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

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

    def latest_signals(self, subject, start, end):
        """Latest legacy name or registered capability/source/sensor identity."""
        rows = self.connection.execute("""
            SELECT payload FROM (
              SELECT payload, ROW_NUMBER() OVER (
                PARTITION BY json_extract(payload, '$.observation.name'),
                  json_extract(payload, '$.observation.capability.id'),
                  json_extract(payload, '$.observation.sensor.source_id'),
                  json_extract(payload, '$.observation.sensor.sensor_id')
                ORDER BY ts DESC, rowid DESC) AS rank
              FROM records WHERE subject=? AND kind='observation.signal' AND ts>=? AND ts<=?
            ) WHERE rank=1
            """, (subject, start, end)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def event_summary(self, subject, start, end, *, per_kind=8):
        counts = dict(self.connection.execute("""SELECT json_extract(payload,'$.event'), COUNT(*)
            FROM records WHERE subject=? AND kind='observation.event' AND ts>=? AND ts<=?
            GROUP BY json_extract(payload,'$.event')""", (subject, start, end)).fetchall())
        rows = self.connection.execute("""SELECT payload FROM (
            SELECT payload, ts, rowid AS ordering, ROW_NUMBER() OVER (
              PARTITION BY json_extract(payload,'$.event') ORDER BY ts DESC,rowid DESC) AS rank
            FROM records WHERE subject=? AND kind='observation.event' AND ts>=? AND ts<=?
            ) WHERE rank<=? ORDER BY ts,ordering""", (subject, start, end, per_kind)).fetchall()
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

    def count(self, subject, start, end):
        return self.connection.execute("SELECT COUNT(*) FROM records WHERE subject=? AND ts>=? AND ts<=?", (subject, start, end)).fetchone()[0]

    def numeric_summary(self, subject, start, end):
        """Aggregate retained values in SQLite without decoding all stored JSON."""
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
          GROUP BY metric, modality, model_id, model_version
          HAVING SUM(CASE WHEN ts>=? THEN 1 ELSE 0 END)>0
        """, (start, start, start, start, start, start, subject, end, start)).fetchall()
        return rows

    def prune(self, now_ms):
        with self._transaction():
            expired = self.connection.execute("SELECT id FROM records WHERE ts < ?", (now_ms-self.retention_ms,)).fetchall()
            if expired:
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
