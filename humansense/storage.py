"""Small local store containing derived records only, never camera/audio buffers."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class Store:
    def __init__(self, uri=":memory:", retention_hours=24):
        if retention_hours <= 0:
            raise ValueError("retention_hours must be positive")
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
        self.connection.commit()

    def add(self, record):
        self.connection.execute("INSERT INTO records VALUES(?,?,?,?,?)", (
            record["id"], record["subject_id"], record["timestamp_ms"], record["type"],
            json.dumps(record, ensure_ascii=False, allow_nan=False)))
        self.connection.commit()

    def query(self, subject=None, start=None, end=None, kind=None, limit=1000, name=None):
        clauses, args = [], []
        for column, value, operator in (("subject", subject, "="), ("ts", start, ">="), ("ts", end, "<="), ("kind", kind, "=")):
            if value is not None:
                clauses.append(f"{column} {operator} ?")
                args.append(value)
        if name is not None:
            clauses.append("json_extract(payload, '$.observation.name') = ?")
            args.append(name)
        sql = "SELECT payload FROM records"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC, rowid DESC LIMIT ?"
        args.append(max(0, min(int(limit), 100_000)))
        rows = self.connection.execute(sql, args).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

    def get_many(self, ids):
        if not ids:
            return []
        slots = ",".join("?" for _ in ids)
        rows = self.connection.execute(f"SELECT payload FROM records WHERE id IN ({slots})", list(ids)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def latest_signals(self, subject, start, end):
        """One current value per name; high-rate video cannot evict text/audio."""
        rows = self.connection.execute("""
            SELECT payload FROM (
              SELECT payload, ROW_NUMBER() OVER (
                PARTITION BY json_extract(payload, '$.observation.name')
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
        self.connection.execute("DELETE FROM records WHERE ts < ?", (now_ms - self.retention_ms,))
        self.connection.commit()

    def delete(self, subject=None):
        if subject is None:
            result = self.connection.execute("DELETE FROM records")
        else:
            result = self.connection.execute("DELETE FROM records WHERE subject=?", (subject,))
        self.connection.commit()
        self.connection.execute("VACUUM")
        return result.rowcount

    def close(self):
        self.connection.close()
