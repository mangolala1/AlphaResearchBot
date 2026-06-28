"""SQLite persistence layer for experiment records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.types import ExperimentRecord

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS experiments (
    alpha_id       TEXT PRIMARY KEY,
    parent_id      TEXT,
    batch_id       TEXT,
    timestamp      TEXT NOT NULL,
    hypothesis     TEXT,
    formula        TEXT,
    features       TEXT,
    mutation       TEXT,
    config         TEXT,
    metrics        TEXT,
    robustness     TEXT,
    verdict        TEXT,
    failure_reason TEXT,
    reflection     TEXT
);
"""


class ExperimentStore:
    """Manages reading and writing experiment records to SQLite."""

    def __init__(self, db_path: str = "db/experiments.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self.init_db()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            try:
                conn.execute("ALTER TABLE experiments ADD COLUMN batch_id TEXT")
            except Exception:
                pass  # column already exists

    def save_experiment(self, record: ExperimentRecord) -> None:
        """Insert or replace an experiment record."""
        row = (
            record["alpha_id"],
            record.get("parent_id"),
            record.get("batch_id"),
            record["timestamp"],
            record.get("hypothesis", ""),
            record.get("formula", ""),
            json.dumps(record.get("features", [])),
            record.get("mutation", ""),
            json.dumps(record.get("config", {})),
            json.dumps(record.get("metrics", {})),
            json.dumps(record.get("robustness", {})),
            record.get("verdict", ""),
            record.get("failure_reason"),
            record.get("reflection", ""),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (alpha_id, parent_id, batch_id, timestamp, hypothesis, formula, features,
                 mutation, config, metrics, robustness, verdict, failure_reason, reflection)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

    def load_all(self) -> list[ExperimentRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM experiments ORDER BY timestamp").fetchall()
        return [self._row_to_record(row) for row in rows]

    def load_by_id(self, alpha_id: str) -> ExperimentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE alpha_id = ?", (alpha_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            alpha_id=row["alpha_id"],
            parent_id=row["parent_id"],
            batch_id=row["batch_id"],
            timestamp=row["timestamp"],
            hypothesis=row["hypothesis"],
            formula=row["formula"],
            features=json.loads(row["features"]),
            mutation=row["mutation"],
            config=json.loads(row["config"]),
            metrics=json.loads(row["metrics"]),
            robustness=json.loads(row["robustness"]),
            verdict=row["verdict"],
            failure_reason=row["failure_reason"],
            reflection=row["reflection"],
        )
