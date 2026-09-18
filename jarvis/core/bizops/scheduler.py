"""Autonomous BizOps Execution Engine and Scheduler for Jarvis."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from jarvis.core.bizops.guardrails import PolicyGuardrail
from jarvis.core.bizops.models import (
    AutonomyLevel,
    BizOpsRunResult,
    BizTask,
    PendingAction,
)
from jarvis.core.config import JarvisConfig, get_config
from jarvis.core.darkfac import DarkFactoryClient
from jarvis.core.memory import EpisodicMemoryEngine

logger = logging.getLogger("jarvis.core.bizops.scheduler")


class BizOpsEngine:
    """Orchestrates scheduled and event-driven business operations with HITL guardrails."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        db_path: Path | str | None = None,
        memory_engine: Optional[EpisodicMemoryEngine] = None,
        darkfac_client: Optional[DarkFactoryClient] = None,
        guardrail: Optional[PolicyGuardrail] = None,
    ) -> None:
        self.config = config or get_config()
        if db_path is None or str(db_path) == ":memory:":
            self.db_path = ":memory:"
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
        else:
            self.db_path = str(Path(db_path).resolve())
            self._mem_conn = None
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.memory = memory_engine or EpisodicMemoryEngine(db_path=self.db_path)
        self.darkfac = darkfac_client or DarkFactoryClient(self.config)
        self.guardrail = guardrail or PolicyGuardrail(default_level=AutonomyLevel.LEVEL_2_HITL)

        self._init_tables()
        self._ensure_default_tasks()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS biz_tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    task_type TEXT NOT NULL,
                    schedule_interval_sec INTEGER,
                    autonomy_level INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    last_run_at TEXT,
                    last_result TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_actions (
                    id TEXT PRIMARY KEY,
                    task_id TEXT,
                    action_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    payload TEXT NOT NULL DEFAULT '{}',
                    autonomy_level_required INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending_approval',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolved_by TEXT,
                    execution_result TEXT
                )
                """
            )
            conn.commit()

    def _ensure_default_tasks(self) -> None:
        defaults = [
            BizTask(
                id="task_standup",
                title="Daily Standup Briefing",
                description="Consolida preferências ativas, backlog da Dark Factory e ações pendentes em um resumo executivo matinal.",
                task_type="daily_standup_briefing",
                schedule_interval_sec=86400,
                autonomy_level=AutonomyLevel.LEVEL_1_READ_ONLY,
            ),
            BizTask(
                id="task_backlog_hygiene",
                title="Dark Factory Backlog Hygiene",
                description="Inspeciona o estado do DarkHub e avalia itens em backlog que demandam atenção.",
                task_type="backlog_hygiene",
                schedule_interval_sec=43200,
                autonomy_level=AutonomyLevel.LEVEL_2_HITL,
            ),
            BizTask(
                id="task_health_pulse",
                title="System Ecosystem Health Pulse",
                description="Realiza sondagem de disponibilidade do Ollama, DarkHub e subsistemas.",
                task_type="health_pulse",
                schedule_interval_sec=3600,
                autonomy_level=AutonomyLevel.LEVEL_1_READ_ONLY,
            ),
        ]
        for t in defaults:
            if not self.get_task(t.id):
                self.register_task(t)

    # --- Task Management ---
    def register_task(self, task: BizTask) -> BizTask:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO biz_tasks (id, title, description, task_type, schedule_interval_sec, autonomy_level, status, last_run_at, last_result, enabled, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    task_type = excluded.task_type,
                    schedule_interval_sec = excluded.schedule_interval_sec,
                    autonomy_level = excluded.autonomy_level,
                    enabled = excluded.enabled,
                    metadata = excluded.metadata
                """,
                (
                    task.id,
                    task.title,
                    task.description,
                    task.task_type,
                    task.schedule_interval_sec,
                    int(task.autonomy_level),
                    task.status,
                    task.last_run_at,
                    task.last_result,
                    1 if task.enabled else 0,
                    json.dumps(task.metadata),
                ),
            )
            conn.commit()
        return task

    def get_task(self, task_id: str) -> Optional[BizTask]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM biz_tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                return BizTask(
                    id=row["id"],
                    title=row["title"],
                    description=row["description"] or "",
                    task_type=row["task_type"],
                    schedule_interval_sec=row["schedule_interval_sec"],
                    autonomy_level=AutonomyLevel(row["autonomy_level"]),
                    status=row["status"],
                    last_run_at=row["last_run_at"],
                    last_result=row["last_result"],
                    enabled=bool(row["enabled"]),
                    metadata=json.loads(row["metadata"]),
                )
        return None

    def list_tasks(self) -> List[BizTask]:
        tasks = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM biz_tasks ORDER BY title ASC")
            for row in cursor.fetchall():
                tasks.append(
                    BizTask(
                        id=row["id"],
                        title=row["title"],
                        description=row["description"] or "",
                        task_type=row["task_type"],
                        schedule_interval_sec=row["schedule_interval_sec"],
                        autonomy_level=AutonomyLevel(row["autonomy_level"]),
                        status=row["status"],
                        last_run_at=row["last_run_at"],
                        last_result=row["last_result"],
                        enabled=bool(row["enabled"]),
                        metadata=json.loads(row["metadata"]),
                    )
                )
        return tasks

    # --- Execution & Dispatch ---
    async def trigger_task(self, task_id: str) -> BizOpsRunResult:
        """Execute a specific operational task respecting its autonomy policy."""
        start = time.perf_counter()
        task = self.get_task(task_id)
        if not task:
            return BizOpsRunResult(
                task_id=task_id,
                success=False,
                message=f"Tarefa '{task_id}' não encontrada.",
                latency_ms=0.0,
            )

        now = datetime.now(timezone.utc).isoformat()
        self._update_task_status(task_id, status="running", last_run_at=now)

        try:
            if task.task_type == "daily_standup_briefing":
                res = await self._run_daily_standup_briefing(task)
            elif task.task_type == "backlog_hygiene":
                res = await self._run_backlog_hygiene(task)
            elif task.task_type == "health_pulse":
                res = await self._run_health_pulse(task)
            else:
                res = BizOpsRunResult(
                    task_id=task_id,
                    success=True,
                    message=f"Tarefa personalizada '{task.title}' executada com sucesso.",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            self._update_task_status(task_id, status="completed", last_result=res.message)
            return res

        except Exception as exc:
            logger.error("BizOps task '%s' failed: %s", task_id, exc)
            err_msg = f"Falha na execução: {exc}"
            self._update_task_status(task_id, status="failed", last_result=err_msg)
            return BizOpsRunResult(
                task_id=task_id,
                success=False,
                message=err_msg,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    async def _run_daily_standup_briefing(self, task: BizTask) -> BizOpsRunResult:
        """Generate daily briefing based on memory and pending actions."""
        facts = self.memory.recall_facts(limit=5)
        decisions = self.memory.query_decisions(limit=3)
        pending = self.list_pending_actions(status="pending_approval")
        hub_status = await self.darkfac.get_status()

        lines = [
            f"☀️ **Bom dia! Briefing Operacional do Jarvis ({datetime.now().strftime('%d/%m/%Y')})**\n",
            f"**Status DarkHub**: {'🟢 Online' if hub_status.online else '🔴 Offline'}",
            f"**Ações Pendentes de Revisão (HITL)**: {len(pending)} pendência(s)",
            f"**Fatos Ativos no Segundo Cérebro**: {len(facts)} itens recuperados",
            f"**Decisões Recentes Registradas**: {len(decisions)} decisão(ões)",
        ]
        briefing_text = "\n".join(lines)
        return BizOpsRunResult(
            task_id=task.id,
            success=True,
            message=briefing_text,
            data={"pending_count": len(pending), "facts_count": len(facts), "hub_online": hub_status.online},
        )

    async def _run_backlog_hygiene(self, task: BizTask) -> BizOpsRunResult:
        """Inspect backlog and propose tickets if necessary."""
        demands = await self.darkfac.list_demands()
        hub_status = await self.darkfac.get_status()

        # If Level 2, propose a review ticket or report
        payload = {
            "title": "Auditoria Semanal de Backlog",
            "problem_statement": f"Verificação de rotina: {len(demands)} demandas ativas no DarkHub.",
            "project_id": "darkfac",
        }
        allowed, msg, pending = self.guardrail.evaluate_action(
            action_type="create_dark_factory_demand",
            title="Criar Demanda de Auditoria de Backlog",
            payload=payload,
            task_id=task.id,
            effective_level=task.autonomy_level,
        )

        if not allowed and pending:
            self._save_pending_action(pending)
            return BizOpsRunResult(
                task_id=task.id,
                success=True,
                message=msg,
                pending_action_id=pending.id,
                data={"total_demands": len(demands), "hub_online": hub_status.online},
            )

        return BizOpsRunResult(
            task_id=task.id,
            success=True,
            message=f"Higiene concluída: {len(demands)} demandas catalogadas.",
            data={"total_demands": len(demands)},
        )

    async def _run_health_pulse(self, task: BizTask) -> BizOpsRunResult:
        """Probe ecosystem health."""
        hub = await self.darkfac.get_status()
        self.memory.store_fact(
            key="last_health_pulse",
            value=f"DarkHub: {hub.online} | Timestamp: {datetime.now(timezone.utc).isoformat()}",
            category="project_context",
        )
        return BizOpsRunResult(
            task_id=task.id,
            success=True,
            message=f"Health pulse registrado. DarkHub online: {hub.online}.",
            data={"hub_online": hub.online},
        )

    def _update_task_status(
        self,
        task_id: str,
        status: str,
        last_run_at: Optional[str] = None,
        last_result: Optional[str] = None,
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if last_run_at and last_result:
                cursor.execute(
                    "UPDATE biz_tasks SET status = ?, last_run_at = ?, last_result = ? WHERE id = ?",
                    (status, last_run_at, last_result, task_id),
                )
            elif last_run_at:
                cursor.execute(
                    "UPDATE biz_tasks SET status = ?, last_run_at = ? WHERE id = ?",
                    (status, last_run_at, task_id),
                )
            elif last_result:
                cursor.execute(
                    "UPDATE biz_tasks SET status = ?, last_result = ? WHERE id = ?",
                    (status, last_result, task_id),
                )
            else:
                cursor.execute("UPDATE biz_tasks SET status = ? WHERE id = ?", (status, task_id))
            conn.commit()

    # --- Pending Actions (HITL) ---
    def _save_pending_action(self, action: PendingAction) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO pending_actions (id, task_id, action_type, title, description, payload, autonomy_level_required, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action.id,
                    action.task_id,
                    action.action_type,
                    action.title,
                    action.description,
                    json.dumps(action.payload),
                    int(action.autonomy_level_required),
                    action.status,
                    action.created_at,
                ),
            )
            conn.commit()

    def list_pending_actions(self, status: Optional[str] = None) -> List[PendingAction]:
        sql = "SELECT * FROM pending_actions"
        params: List[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC"

        actions = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            for row in cursor.fetchall():
                actions.append(
                    PendingAction(
                        id=row["id"],
                        task_id=row["task_id"],
                        action_type=row["action_type"],
                        title=row["title"],
                        description=row["description"] or "",
                        payload=json.loads(row["payload"]),
                        autonomy_level_required=AutonomyLevel(row["autonomy_level_required"]),
                        status=row["status"],
                        created_at=row["created_at"],
                        resolved_at=row["resolved_at"],
                        resolved_by=row["resolved_by"],
                        execution_result=json.loads(row["execution_result"]) if row["execution_result"] else None,
                    )
                )
        return actions

    def get_pending_action(self, action_id: str) -> Optional[PendingAction]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM pending_actions WHERE id = ?", (action_id,))
            row = cursor.fetchone()
            if row:
                return PendingAction(
                    id=row["id"],
                    task_id=row["task_id"],
                    action_type=row["action_type"],
                    title=row["title"],
                    description=row["description"] or "",
                    payload=json.loads(row["payload"]),
                    autonomy_level_required=AutonomyLevel(row["autonomy_level_required"]),
                    status=row["status"],
                    created_at=row["created_at"],
                    resolved_at=row["resolved_at"],
                    resolved_by=row["resolved_by"],
                    execution_result=json.loads(row["execution_result"]) if row["execution_result"] else None,
                )
        return None

    async def approve_action(self, action_id: str, operator: str = "operator") -> BizOpsRunResult:
        """Approve and execute a pending HITL action."""
        action = self.get_pending_action(action_id)
        if not action:
            return BizOpsRunResult(
                task_id="unknown",
                success=False,
                message=f"Ação pendente '{action_id}' não encontrada.",
            )

        if action.status != "pending_approval":
            return BizOpsRunResult(
                task_id=action.task_id or "unknown",
                success=False,
                message=f"Ação já se encontra resolvida como '{action.status}'.",
            )

        now = datetime.now(timezone.utc).isoformat()

        # Execute payload action
        try:
            exec_res = {}
            if action.action_type == "create_dark_factory_demand":
                payload = action.payload
                exec_res = await self.darkfac.create_demand(
                    title=payload.get("title", action.title),
                    problem_statement=payload.get("problem_statement", ""),
                    project_id=payload.get("project_id", "darkfac"),
                )
            else:
                exec_res = {"status": "executed", "action": action.action_type}

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE pending_actions
                    SET status = 'executed', resolved_at = ?, resolved_by = ?, execution_result = ?
                    WHERE id = ?
                    """,
                    (now, operator, json.dumps(exec_res), action_id),
                )
                conn.commit()

            return BizOpsRunResult(
                task_id=action.task_id or "unknown",
                success=True,
                message=f"Ação '{action.title}' aprovada e executada com sucesso por {operator}.",
                data=exec_res,
            )
        except Exception as exc:
            logger.error("Failed to execute approved action %s: %s", action_id, exc)
            return BizOpsRunResult(
                task_id=action.task_id or "unknown",
                success=False,
                message=f"Erro ao executar ação aprovada: {exc}",
            )

    def reject_action(self, action_id: str, operator: str = "operator") -> bool:
        """Reject and cancel a pending HITL action."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE pending_actions
                SET status = 'rejected', resolved_at = ?, resolved_by = ?
                WHERE id = ? AND status = 'pending_approval'
                """,
                (now, operator, action_id),
            )
            conn.commit()
            return cursor.rowcount > 0
