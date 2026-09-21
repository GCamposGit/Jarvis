"""Persistent and in-memory audit ledger for the deterministic agent harness (Milestone 4)."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from jarvis.core.harness.models import ExecutionRiskLevel, HarnessAuditEntry

logger = logging.getLogger("jarvis.core.harness.audit")


class HarnessAuditLedger:
    """Thread-safe SQLite ledger recording preflight checks, tool calls, and sandboxed executions."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None or str(db_path) == ":memory:":
            self.db_path = ":memory:"
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        else:
            self.db_path = str(Path(db_path).resolve())
            self._mem_conn = None
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS harness_audit_log (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    risk_level INTEGER NOT NULL,
                    allowed INTEGER NOT NULL,
                    reasons TEXT NOT NULL DEFAULT '[]',
                    execution_time_ms REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'completed',
                    details TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.commit()

    def record_event(self, entry: HarnessAuditEntry) -> None:
        """Record an auditable harness event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO harness_audit_log (id, timestamp, action_type, target, risk_level, allowed, reasons, execution_time_ms, status, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.timestamp,
                    entry.action_type,
                    entry.target,
                    int(entry.risk_level),
                    1 if entry.allowed else 0,
                    json.dumps(entry.reasons),
                    entry.execution_time_ms,
                    entry.status,
                    json.dumps(entry.details),
                ),
            )
            conn.commit()

    def list_events(self, limit: int = 50, only_blocked: bool = False) -> List[HarnessAuditEntry]:
        """Retrieve recent audit events."""
        sql = "SELECT * FROM harness_audit_log"
        params: List[Any] = []
        if only_blocked:
            sql += " WHERE allowed = 0"
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        results: List[HarnessAuditEntry] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            for row in cursor.fetchall():
                results.append(
                    HarnessAuditEntry(
                        id=row["id"],
                        timestamp=row["timestamp"],
                        action_type=row["action_type"],
                        target=row["target"],
                        risk_level=ExecutionRiskLevel(row["risk_level"]),
                        allowed=bool(row["allowed"]),
                        reasons=json.loads(row["reasons"]),
                        execution_time_ms=row["execution_time_ms"],
                        status=row["status"],
                        details=json.loads(row["details"]),
                    )
                )
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Calculate overall harness safety statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM harness_audit_log")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM harness_audit_log WHERE allowed = 0")
            blocked = cursor.fetchone()[0]

            cursor.execute("SELECT risk_level, COUNT(*) FROM harness_audit_log GROUP BY risk_level")
            by_risk = {str(ExecutionRiskLevel(row[0]).name): row[1] for row in cursor.fetchall()}

        return {
            "total_events": total,
            "blocked_events": blocked,
            "allowed_events": total - blocked,
            "block_rate_pct": round((blocked / total * 100), 2) if total > 0 else 0.0,
            "by_risk_level": by_risk,
        }
