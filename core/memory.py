"""SQLite persistence layer for experiment records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.types import ExperimentRecord

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS experiments (
    alpha_id            TEXT PRIMARY KEY,
    parent_id           TEXT,
    batch_id            TEXT,
    timestamp           TEXT NOT NULL,
    hypothesis          TEXT,
    formula             TEXT,
    features            TEXT,
    mutation            TEXT,
    config              TEXT,
    metrics             TEXT,
    robustness          TEXT,
    verdict             TEXT,
    failure_reason      TEXT,
    reflection          TEXT,
    score               REAL,
    signal_strength     REAL,
    preferred_direction INTEGER,
    sub_scores          TEXT
);
"""

_CREATE_BANDIT_TABLE = """
CREATE TABLE IF NOT EXISTS bandit_state (
    arm_id       TEXT PRIMARY KEY,
    alpha        REAL NOT NULL,
    beta         REAL NOT NULL,
    pulls        INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT
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
            conn.execute(_CREATE_BANDIT_TABLE)
            # Safe migrations for databases created before these columns existed
            for ddl in (
                "ALTER TABLE experiments ADD COLUMN batch_id TEXT",
                "ALTER TABLE experiments ADD COLUMN score REAL",
                "ALTER TABLE experiments ADD COLUMN signal_strength REAL",
                "ALTER TABLE experiments ADD COLUMN preferred_direction INTEGER",
                "ALTER TABLE experiments ADD COLUMN sub_scores TEXT",
            ):
                try:
                    conn.execute(ddl)
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
            record.get("score"),
            record.get("signal_strength"),
            record.get("preferred_direction"),
            json.dumps(record.get("sub_scores")) if record.get("sub_scores") else None,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (alpha_id, parent_id, batch_id, timestamp, hypothesis, formula, features,
                 mutation, config, metrics, robustness, verdict, failure_reason, reflection,
                 score, signal_strength, preferred_direction, sub_scores)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    # ── Bandit state (Thompson scheduler posteriors) ─────────────────────────

    def load_bandit_state(self) -> dict[str, dict]:
        """Return {arm_id: {"alpha": float, "beta": float, "pulls": int}}."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM bandit_state").fetchall()
        return {
            row["arm_id"]: {
                "alpha": row["alpha"],
                "beta": row["beta"],
                "pulls": row["pulls"],
            }
            for row in rows
        }

    def upsert_bandit_arm(
        self, arm_id: str, alpha: float, beta: float, pulls: int
    ) -> None:
        from datetime import datetime, timezone

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO bandit_state
                (arm_id, alpha, beta, pulls, last_updated)
                VALUES (?, ?, ?, ?, ?)
                """,
                (arm_id, alpha, beta, pulls, datetime.now(timezone.utc).isoformat()),
            )

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
            score=row["score"],
            signal_strength=row["signal_strength"],
            preferred_direction=row["preferred_direction"],
            sub_scores=json.loads(row["sub_scores"]) if row["sub_scores"] else None,
        )
