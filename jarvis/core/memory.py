"""Persistent Episodic Memory and Decision Engine for Jarvis.

Provides durable, rollbackable SQLite-backed memory for user preferences,
entity facts, architectural decisions, and failure retrospectives.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("jarvis.core.memory")


class MemoryFact(BaseModel):
    """A single factual knowledge item persisted in the Second Brain."""

    model_config = ConfigDict(frozen=True)

    key: str
    value: str
    category: str = "general"  # "user_preference", "project_context", "business_rule", "tech_stack"
    tags: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DecisionRecord(BaseModel):
    """An explicit architectural or business decision made by the user or agent."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    problem_statement: str = ""
    decision: str
    alternatives: List[str] = Field(default_factory=list)
    rationale: str = ""
    recorded_by: str = "operator"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EpisodicNote(BaseModel):
    """Summary of a past interaction or work session."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    interaction_summary: str
    tags: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FailedAttempt(BaseModel):
    """Record of a failed approach or anti-pattern to prevent repetition."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    action: str
    error_pattern: str
    lesson_learned: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EpisodicMemoryEngine:
    """Thread-safe SQLite engine managing persistent knowledge, decisions, and failure retrospectives."""

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
        """Create necessary tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    tags TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_records (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    problem_statement TEXT,
                    decision TEXT NOT NULL,
                    alternatives TEXT NOT NULL DEFAULT '[]',
                    rationale TEXT,
                    recorded_by TEXT NOT NULL DEFAULT 'operator',
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_notes (
                    id TEXT PRIMARY KEY,
                    interaction_summary TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS failed_attempts (
                    id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    error_pattern TEXT NOT NULL,
                    lesson_learned TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    # --- Facts API ---
    def store_fact(
        self,
        key: str,
        value: str,
        category: str = "general",
        tags: Optional[List[str]] = None,
        confidence: float = 1.0,
    ) -> MemoryFact:
        """Store or update a factual knowledge item."""
        clean_key = key.strip().lower()
        clean_tags = tags or []
        now = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memory_facts (key, value, category, tags, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    tags = excluded.tags,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (clean_key, value.strip(), category, json.dumps(clean_tags), confidence, now, now),
            )
            conn.commit()

        return MemoryFact(
            key=clean_key,
            value=value.strip(),
            category=category,
            tags=clean_tags,
            confidence=confidence,
            created_at=now,
            updated_at=now,
        )

    def get_fact(self, key: str) -> Optional[MemoryFact]:
        """Retrieve a specific fact by key."""
        clean_key = key.strip().lower()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memory_facts WHERE key = ?", (clean_key,))
            row = cursor.fetchone()
            if row:
                return MemoryFact(
                    key=row["key"],
                    value=row["value"],
                    category=row["category"],
                    tags=json.loads(row["tags"]),
                    confidence=float(row["confidence"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
        return None

    def recall_facts(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[MemoryFact]:
        """Recall facts matching optional search query and category with tokenized ranking."""
        clean_query = query.strip() if query else None

        def _row_to_fact(row: sqlite3.Row) -> MemoryFact:
            return MemoryFact(
                key=row["key"],
                value=row["value"],
                category=row["category"],
                tags=json.loads(row["tags"]),
                confidence=float(row["confidence"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

        if not clean_query:
            sql = "SELECT * FROM memory_facts WHERE 1=1"
            params: List[Any] = []
            if category:
                sql += " AND category = ?"
                params.append(category)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)

            facts = []
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, tuple(params))
                for row in cursor.fetchall():
                    facts.append(_row_to_fact(row))
            return facts

        # Attempt 1: Exact substring match
        exact_sql = "SELECT * FROM memory_facts WHERE 1=1"
        exact_params: List[Any] = []
        if category:
            exact_sql += " AND category = ?"
            exact_params.append(category)
        exact_sql += " AND (key LIKE ? OR value LIKE ? OR tags LIKE ?)"
        wildcard = f"%{clean_query.lower()}%"
        exact_params.extend([wildcard, wildcard, wildcard])
        exact_sql += " ORDER BY updated_at DESC LIMIT ?"
        exact_params.append(limit)

        facts_by_key: Dict[str, MemoryFact] = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(exact_sql, tuple(exact_params))
            for row in cursor.fetchall():
                f = _row_to_fact(row)
                facts_by_key[f.key] = f

        if len(facts_by_key) >= limit:
            return list(facts_by_key.values())[:limit]

        # Attempt 2: Tokenized keyword search
        import re
        stopwords = {
            "qual", "quem", "como", "onde", "quando", "quanto", "quantos", "quanta", "quantas",
            "porque", "por", "que", "para", "com", "sem", "sob", "sobre", "entre", "ate", "até",
            "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas", "aquele", "aquela",
            "isso", "isto", "aquilo", "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas",
            "dele", "dela", "deles", "delas", "nosso", "nossa", "nossos", "nossas", "voce", "você",
            "ele", "ela", "eles", "elas", "nos", "nós", "de", "da", "do", "das", "dos",
            "em", "no", "na", "nos", "nas", "pelo", "pela", "pelos", "pelas", "um", "uma", "uns", "umas",
            "o", "a", "os", "as", "e", "é", "ou", "se", "fato", "guarde", "lembre", "lembre-se", "armazene",
            "salve", "registre", "jarvis", "saber", "favor", "por favor"
        }
        words = re.findall(r"[a-zA-Z0-9áéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+", clean_query.lower())
        tokens = [w for w in words if len(w) >= 3 and w not in stopwords]

        if tokens:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                all_sql = "SELECT * FROM memory_facts WHERE 1=1"
                all_params: List[Any] = []
                if category:
                    all_sql += " AND category = ?"
                    all_params.append(category)
                cursor.execute(all_sql, tuple(all_params))
                scored_facts = []
                for row in cursor.fetchall():
                    f = _row_to_fact(row)
                    if f.key in facts_by_key:
                        continue
                    key_text = f.key.lower()
                    val_text = f.value.lower()
                    tags_text = " ".join(t.lower() for t in f.tags)
                    score = 0
                    for t in tokens:
                        if t in key_text:
                            score += 4
                        if t in val_text:
                            score += 2
                        if t in tags_text:
                            score += 3
                    if score > 0:
                        scored_facts.append((score, f))

                scored_facts.sort(key=lambda x: (x[0], x[1].updated_at), reverse=True)
                for _, f in scored_facts:
                    facts_by_key[f.key] = f
                    if len(facts_by_key) >= limit:
                        break

        return list(facts_by_key.values())[:limit]

    def delete_fact(self, key: str) -> bool:
        """Delete a fact from persistent memory."""
        clean_key = key.strip().lower()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memory_facts WHERE key = ?", (clean_key,))
            conn.commit()
            return cursor.rowcount > 0

    # --- Decisions API ---
    def log_decision(
        self,
        title: str,
        decision: str,
        alternatives: Optional[List[str]] = None,
        rationale: str = "",
        problem_statement: str = "",
        recorded_by: str = "operator",
    ) -> DecisionRecord:
        """Log an explicit decision to maintain architectural consistency."""
        record_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()
        alt_list = alternatives or []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO decision_records (id, title, problem_statement, decision, alternatives, rationale, recorded_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (record_id, title.strip(), problem_statement.strip(), decision.strip(), json.dumps(alt_list), rationale.strip(), recorded_by, now),
            )
            conn.commit()

        return DecisionRecord(
            id=record_id,
            title=title.strip(),
            problem_statement=problem_statement.strip(),
            decision=decision.strip(),
            alternatives=alt_list,
            rationale=rationale.strip(),
            recorded_by=recorded_by,
            created_at=now,
        )

    def query_decisions(self, query: Optional[str] = None, limit: int = 10) -> List[DecisionRecord]:
        """Query logged decisions."""
        sql = "SELECT * FROM decision_records WHERE 1=1"
        params: List[Any] = []

        if query:
            sql += " AND (title LIKE ? OR decision LIKE ? OR rationale LIKE ?)"
            wildcard = f"%{query.strip()}%"
            params.extend([wildcard, wildcard, wildcard])

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        records = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            for row in cursor.fetchall():
                records.append(
                    DecisionRecord(
                        id=row["id"],
                        title=row["title"],
                        problem_statement=row["problem_statement"] or "",
                        decision=row["decision"],
                        alternatives=json.loads(row["alternatives"]),
                        rationale=row["rationale"] or "",
                        recorded_by=row["recorded_by"],
                        created_at=row["created_at"],
                    )
                )
        return records

    # --- Failure Retrospectives API ---
    def record_failed_attempt(
        self,
        action: str,
        error_pattern: str,
        lesson_learned: str,
    ) -> FailedAttempt:
        """Record a failed approach so future agents avoid repeating the same mistake."""
        attempt_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO failed_attempts (id, action, error_pattern, lesson_learned, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (attempt_id, action.strip(), error_pattern.strip(), lesson_learned.strip(), now),
            )
            conn.commit()

        return FailedAttempt(
            id=attempt_id,
            action=action.strip(),
            error_pattern=error_pattern.strip(),
            lesson_learned=lesson_learned.strip(),
            created_at=now,
        )

    def query_failed_attempts(self, action_query: Optional[str] = None, limit: int = 5) -> List[FailedAttempt]:
        """Retrieve recorded failure patterns to warn against anti-patterns."""
        sql = "SELECT * FROM failed_attempts WHERE 1=1"
        params: List[Any] = []

        if action_query:
            sql += " AND (action LIKE ? OR error_pattern LIKE ? OR lesson_learned LIKE ?)"
            wildcard = f"%{action_query.strip()}%"
            params.extend([wildcard, wildcard, wildcard])

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        attempts = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            for row in cursor.fetchall():
                attempts.append(
                    FailedAttempt(
                        id=row["id"],
                        action=row["action"],
                        error_pattern=row["error_pattern"],
                        lesson_learned=row["lesson_learned"],
                        created_at=row["created_at"],
                    )
                )
        return attempts

    # --- Episodic Notes API ---
    def record_episode(self, summary: str, tags: Optional[List[str]] = None) -> EpisodicNote:
        """Save a summary note from a completed conversation or milestone."""
        episode_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()
        tag_list = tags or []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO episodic_notes (id, interaction_summary, tags, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (episode_id, summary.strip(), json.dumps(tag_list), now),
            )
            conn.commit()

        return EpisodicNote(
            id=episode_id,
            interaction_summary=summary.strip(),
            tags=tag_list,
            created_at=now,
        )

    def get_recent_episodes(self, limit: int = 5) -> List[EpisodicNote]:
        """Fetch recent interaction episodes."""
        records = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM episodic_notes ORDER BY created_at DESC LIMIT ?", (limit,))
            for row in cursor.fetchall():
                records.append(
                    EpisodicNote(
                        id=row["id"],
                        interaction_summary=row["interaction_summary"],
                        tags=json.loads(row["tags"]),
                        created_at=row["created_at"],
                    )
                )
        return records

    # --- Proactive Context Brief Generator ---
    def build_context_brief(self, current_topic: Optional[str] = None, max_items: int = 6) -> str:
        """Generate a concise markdown memory brief to inject into LLM system prompts."""
        facts = self.recall_facts(query=current_topic, limit=max_items) if current_topic else []
        if len(facts) < 3:
            recent_facts = self.recall_facts(limit=3)
            seen_keys = {f.key for f in facts}
            for rf in recent_facts:
                if rf.key not in seen_keys and len(facts) < max_items:
                    facts.append(rf)
                    seen_keys.add(rf.key)

        failures = self.query_failed_attempts(action_query=current_topic, limit=3)
        if not failures:
            failures = self.query_failed_attempts(limit=2)

        decisions = self.query_decisions(query=current_topic, limit=3)
        if not decisions:
            decisions = self.query_decisions(limit=2)

        if not facts and not failures and not decisions:
            return ""

        lines = ["[MEMÓRIA EPISÓDICA E CONTEXTO DO SEGUNDO CÉREBRO]"]

        if facts:
            lines.append("Fatos e Preferências Relevantes:")
            for f in facts:
                lines.append(f"- ({f.category}) {f.key}: {f.value}")

        if decisions:
            lines.append("Decisões Arquiteturais Registradas:")
            for d in decisions:
                lines.append(f"- Decisão '{d.title}': {d.decision} (Razão: {d.rationale})")

        if failures:
            lines.append("⚠️ Lições de Falhas Anteriores (NÃO repita estas abordagens):")
            for fail in failures:
                lines.append(f"- Ao executar '{fail.action}': {fail.error_pattern} -> Lição: {fail.lesson_learned}")

        lines.append("[FIM DA MEMÓRIA EPISÓDICA]\n")
        return "\n".join(lines)
