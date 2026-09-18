"""Telemetry and financial governance engine for token metrics and budget enforcement (Milestone 5)."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jarvis.core.telemetry.models import (
    BudgetPolicy,
    BudgetStatus,
    TelemetrySummary,
    TokenUsageRecord,
)

logger = logging.getLogger("jarvis.core.telemetry.engine")

# Pricing matrix per 1,000,000 tokens: (prompt_price_usd, completion_price_usd)
PRICING_PER_1M: Dict[str, Tuple[float, float]] = {
    # Local models: strictly $0.00
    "ollama": (0.00, 0.00),
    "qwen": (0.00, 0.00),
    "llama": (0.00, 0.00),
    "deepseek-r1": (0.00, 0.00),
    "mistral": (0.00, 0.00),
    # Google Cloud
    "google/gemini-2.5-flash": (0.075, 0.30),
    "gemini-2.5-flash": (0.075, 0.30),
    "google/gemini-2.0-flash": (0.10, 0.40),
    # High-efficiency Pareto Cloud
    "deepseek/deepseek-v4.1-flash": (0.14, 0.28),
    "deepseek/deepseek-chat": (0.14, 0.28),
    # Specialist & Frontier Cloud
    "x-ai/grok-4.6": (0.80, 2.00),
    "x-ai/grok-beta": (0.80, 2.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "anthropic/claude-3-5-sonnet": (3.00, 15.00),
    "openai/gpt-4o": (2.50, 10.00),
}

# Commercial baseline used to calculate user savings (e.g. GPT-4o / Claude Opus baseline)
COMMERCIAL_BASELINE_PER_1M = (5.00, 15.00)


def calculate_cost(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    provider: str = "",
    tokens_prompt: Optional[int] = None,
    tokens_completion: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute estimated USD cost and counter-factual commercial baseline cost."""
    p_tok = tokens_prompt if tokens_prompt is not None else prompt_tokens
    c_tok = tokens_completion if tokens_completion is not None else completion_tokens

    lowered = model.lower()
    prov_lowered = provider.lower()
    rates: Optional[Tuple[float, float]] = None

    if "ollama" in lowered or "ollama" in prov_lowered or "local" in lowered:
        rates = (0.00, 0.00)
    else:
        for key, val in PRICING_PER_1M.items():
            if key in lowered:
                rates = val
                break

    if rates is None:
        rates = (0.50, 1.50)  # Standard average rate for unrecognized cloud models

    prompt_cost = (p_tok / 1_000_000) * rates[0]
    comp_cost = (c_tok / 1_000_000) * rates[1]
    cost_usd = round(prompt_cost + comp_cost, 6)

    base_prompt_cost = (p_tok / 1_000_000) * COMMERCIAL_BASELINE_PER_1M[0]
    base_comp_cost = (c_tok / 1_000_000) * COMMERCIAL_BASELINE_PER_1M[1]
    baseline_usd = round(base_prompt_cost + base_comp_cost, 6)

    return cost_usd, baseline_usd


class TelemetryEngine:
    """Manages token transactions, cost calculations, savings analytics, and budget guardrails."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        policy: Optional[BudgetPolicy] = None,
    ) -> None:
        if db_path is None or str(db_path) == ":memory:":
            self.db_path = ":memory:"
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        else:
            self.db_path = str(Path(db_path).resolve())
            self._mem_conn = None
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_tables()
        if policy is not None:
            self.update_policy(policy)

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage_records (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0.0,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    baseline_cost_usd REAL NOT NULL DEFAULT 0.0,
                    task_tag TEXT NOT NULL DEFAULT 'chat',
                    details TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS budget_policies (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    daily_limit_usd REAL NOT NULL DEFAULT 2.00,
                    monthly_limit_usd REAL NOT NULL DEFAULT 30.00,
                    alert_threshold_pct REAL NOT NULL DEFAULT 80.0,
                    auto_fallback_to_local INTEGER NOT NULL DEFAULT 1,
                    enforce_circuit_breaker INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            cur.execute(
                """
                INSERT OR IGNORE INTO budget_policies (id, daily_limit_usd, monthly_limit_usd, alert_threshold_pct, auto_fallback_to_local, enforce_circuit_breaker)
                VALUES (1, 2.00, 30.00, 80.0, 1, 1)
                """
            )
            conn.commit()

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        provider: str = "",
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute estimated USD cost and counter-factual commercial baseline cost."""
        return calculate_cost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider=provider,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
        )

    def record_usage(
        self,
        model: str,
        provider: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        task_tag: str = "chat",
        details: Optional[Dict[str, Any]] = None,
        cost_usd: Optional[float] = None,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> TokenUsageRecord:
        """Record an inference usage transaction."""
        p_tok = tokens_prompt if tokens_prompt is not None else prompt_tokens
        c_tok = tokens_completion if tokens_completion is not None else completion_tokens
        calc_cost, base_usd = self.calculate_cost(
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            provider=provider,
        )
        final_cost = cost_usd if cost_usd is not None else calc_cost
        total_tokens = p_tok + c_tok

        det = dict(details or {})
        if session_id:
            det["session_id"] = session_id

        record = TokenUsageRecord(
            model=model,
            provider=provider,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=final_cost,
            baseline_cost_usd=base_usd,
            task_tag=task_tag,
            details=det,
        )

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO token_usage_records (
                    id, timestamp, model, provider, prompt_tokens, completion_tokens,
                    total_tokens, latency_ms, cost_usd, baseline_cost_usd, task_tag, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.timestamp,
                    record.model,
                    record.provider,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.latency_ms,
                    record.cost_usd,
                    record.baseline_cost_usd,
                    record.task_tag,
                    json.dumps(record.details),
                ),
            )
            conn.commit()

        return record

    @property
    def policy(self) -> BudgetPolicy:
        """Convenience property accessing current budget policy."""
        return self.get_policy()

    def get_policy(self) -> BudgetPolicy:
        """Fetch current budget governance policy."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM budget_policies WHERE id = 1")
            row = cur.fetchone()
            if row:
                return BudgetPolicy(
                    daily_limit_usd=row["daily_limit_usd"],
                    monthly_limit_usd=row["monthly_limit_usd"],
                    alert_threshold_pct=row["alert_threshold_pct"],
                    auto_fallback_to_local=bool(row["auto_fallback_to_local"]),
                    enforce_circuit_breaker=bool(row["enforce_circuit_breaker"]),
                )
        return BudgetPolicy()

    def update_policy(
        self,
        policy: Optional[BudgetPolicy] = None,
        **kwargs: Any,
    ) -> BudgetPolicy:
        """Update budget thresholds and circuit breaker flags."""
        curr = self.get_policy()
        if policy is not None:
            new_daily = policy.daily_limit_usd
            new_monthly = policy.monthly_limit_usd
            new_alert = policy.alert_threshold_pct
            new_fallback = policy.auto_fallback_to_local
            new_cb = policy.enforce_circuit_breaker
        else:
            new_daily = kwargs.get("daily_limit_usd", curr.daily_limit_usd)
            new_monthly = kwargs.get("monthly_limit_usd", curr.monthly_limit_usd)
            new_alert = kwargs.get("alert_threshold_pct", curr.alert_threshold_pct)
            new_fallback = kwargs.get("auto_fallback_to_local", curr.auto_fallback_to_local)
            new_cb = kwargs.get("enforce_circuit_breaker", curr.enforce_circuit_breaker)

        updated = BudgetPolicy(
            daily_limit_usd=float(new_daily),
            monthly_limit_usd=float(new_monthly),
            alert_threshold_pct=float(new_alert),
            auto_fallback_to_local=bool(new_fallback),
            enforce_circuit_breaker=bool(new_cb),
        )

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE budget_policies SET
                    daily_limit_usd = ?,
                    monthly_limit_usd = ?,
                    alert_threshold_pct = ?,
                    auto_fallback_to_local = ?,
                    enforce_circuit_breaker = ?
                WHERE id = 1
                """,
                (
                    updated.daily_limit_usd,
                    updated.monthly_limit_usd,
                    updated.alert_threshold_pct,
                    1 if updated.auto_fallback_to_local else 0,
                    1 if updated.enforce_circuit_breaker else 0,
                ),
            )
            conn.commit()
        return updated

    def get_budget_status(self) -> BudgetStatus:
        """Evaluate real-time spend against daily and monthly limits."""
        policy = self.get_policy()
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        month_str = datetime.now(timezone.utc).strftime("%Y-%m")

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM token_usage_records WHERE timestamp LIKE ?",
                (f"{now_str}%",),
            )
            daily_spent = float(cur.fetchone()[0])

            cur.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM token_usage_records WHERE timestamp LIKE ?",
                (f"{month_str}%",),
            )
            monthly_spent = float(cur.fetchone()[0])

        daily_pct = round((daily_spent / policy.daily_limit_usd * 100), 1) if policy.daily_limit_usd > 0 else 0.0
        monthly_pct = round((monthly_spent / policy.monthly_limit_usd * 100), 1) if policy.monthly_limit_usd > 0 else 0.0

        is_exceeded = (daily_spent >= policy.daily_limit_usd) or (monthly_spent >= policy.monthly_limit_usd)
        is_warning = not is_exceeded and (
            (daily_pct >= policy.alert_threshold_pct) or (monthly_pct >= policy.alert_threshold_pct)
        )

        status_str = "exceeded" if is_exceeded else ("warning" if is_warning else "ok")
        circuit_active = is_exceeded and policy.enforce_circuit_breaker

        msg = "Orçamento dentro dos parâmetros nominais."
        if is_exceeded:
            msg = (
                f"Teto orçamentário excedido (${daily_spent:.4f} / ${policy.daily_limit_usd:.2f} diário). "
                f"{'Fallback para modelo local ativado.' if policy.auto_fallback_to_local else 'Requisições bloqueadas.'}"
            )
        elif is_warning:
            msg = f"Atenção: Consumo atingiu {max(daily_pct, monthly_pct):.1f}% do teto configurado."

        return BudgetStatus(
            status=status_str,
            daily_spent_usd=round(daily_spent, 4),
            daily_limit_usd=policy.daily_limit_usd,
            daily_percent=daily_pct,
            monthly_spent_usd=round(monthly_spent, 4),
            monthly_limit_usd=policy.monthly_limit_usd,
            monthly_percent=monthly_pct,
            circuit_breaker_active=circuit_active,
            message=msg,
        )

    def get_summary(self, days: int = 30) -> TelemetrySummary:
        """Aggregate system-wide financial and usage telemetry."""
        budget_st = self.get_budget_status()

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(total_tokens), 0),
                    COALESCE(SUM(prompt_tokens), 0),
                    COALESCE(SUM(completion_tokens), 0),
                    COALESCE(SUM(cost_usd), 0.0),
                    COALESCE(SUM(baseline_cost_usd), 0.0),
                    COUNT(*)
                FROM token_usage_records
                """
            )
            row = cur.fetchone()
            total_tok = int(row[0])
            prompt_tok = int(row[1])
            comp_tok = int(row[2])
            cost_usd = round(float(row[3]), 6)
            base_usd = round(float(row[4]), 6)
            rec_count = int(row[5])
            savings_usd = round(max(0.0, base_usd - cost_usd), 6)

            # Breakdown by model
            cur.execute(
                """
                SELECT model, COUNT(*), SUM(total_tokens), SUM(cost_usd), SUM(baseline_cost_usd)
                FROM token_usage_records
                GROUP BY model
                ORDER BY SUM(total_tokens) DESC
                """
            )
            by_model: Dict[str, Dict[str, Any]] = {}
            for r in cur.fetchall():
                m_name = str(r[0])
                m_calls = int(r[1])
                m_toks = int(r[2])
                m_cost = round(float(r[3]), 6)
                m_base = round(float(r[4]), 6)
                m_savings = round(max(0.0, m_base - m_cost), 6)
                by_model[m_name] = {
                    "calls": m_calls,
                    "tokens": m_toks,
                    "cost_usd": m_cost,
                    "savings_usd": m_savings,
                }
            tokens_by_model = {m: d["tokens"] for m, d in by_model.items()}

            # Breakdown by provider
            cur.execute(
                """
                SELECT provider, SUM(cost_usd)
                FROM token_usage_records
                GROUP BY provider
                """
            )
            cost_by_prov = {r[0]: round(float(r[1]), 4) for r in cur.fetchall()}

        return TelemetrySummary(
            total_tokens=total_tok,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_cost_usd=cost_usd,
            total_savings_usd=savings_usd,
            daily_spent_usd=budget_st.daily_spent_usd,
            tokens_by_model=tokens_by_model,
            by_model=by_model,
            cost_by_provider=cost_by_prov,
            record_count=rec_count,
            budget=budget_st,
        )

    def list_records(self, limit: int = 50) -> List[TokenUsageRecord]:
        """Fetch latest token transactions."""
        records: List[TokenUsageRecord] = []
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT * FROM token_usage_records
                ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            )
            for r in cur.fetchall():
                records.append(
                    TokenUsageRecord(
                        id=r["id"],
                        timestamp=r["timestamp"],
                        model=r["model"],
                        provider=r["provider"],
                        prompt_tokens=r["prompt_tokens"],
                        completion_tokens=r["completion_tokens"],
                        total_tokens=r["total_tokens"],
                        latency_ms=r["latency_ms"],
                        cost_usd=r["cost_usd"],
                        baseline_cost_usd=r["baseline_cost_usd"],
                        task_tag=r["task_tag"],
                        details=json.loads(r["details"]),
                    )
                )
        return records
