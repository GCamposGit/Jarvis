"""Domain models for token consumption telemetry and budget governance (Milestone 5)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, computed_field


class TokenUsageRecord(BaseModel):
    """Immutable record of an inference transaction."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: f"tok_{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0  # Equivalent cost if run on commercial proprietary models (GPT-4o)
    task_tag: str = "chat"
    details: Dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def savings_usd(self) -> float:
        """Estimated savings compared to proprietary baseline."""
        return max(0.0, self.baseline_cost_usd - self.cost_usd)


class BudgetPolicy(BaseModel):
    """Spending limits and circuit breaker policy."""

    model_config = ConfigDict(frozen=True, extra="allow")

    daily_limit_usd: float = 2.00
    monthly_limit_usd: float = 30.00
    alert_threshold_pct: float = 80.0
    auto_fallback_to_local: bool = True
    enforce_circuit_breaker: bool = True
    default_fallback_model: str = "qwen-code-deep:latest"

    @computed_field
    @property
    def circuit_breaker_enabled(self) -> bool:
        """Alias for enforce_circuit_breaker."""
        return self.enforce_circuit_breaker

    @computed_field
    @property
    def fallback_to_local_on_limit(self) -> bool:
        """Alias for auto_fallback_to_local."""
        return self.auto_fallback_to_local


class BudgetStatus(BaseModel):
    """Real-time budget consumption state."""

    model_config = ConfigDict(frozen=True)

    status: str  # "ok", "warning", "exceeded"
    daily_spent_usd: float = 0.0
    daily_limit_usd: float = 2.00
    daily_percent: float = 0.0
    monthly_spent_usd: float = 0.0
    monthly_limit_usd: float = 30.00
    monthly_percent: float = 0.0
    circuit_breaker_active: bool = False
    message: str = "Orçamento dentro dos limites operacionais."

    @computed_field
    @property
    def daily_spend_usd(self) -> float:
        """Convenience alias for daily_spent_usd serialized in JSON."""
        return self.daily_spent_usd

    @computed_field
    @property
    def monthly_spend_usd(self) -> float:
        """Convenience alias for monthly_spent_usd serialized in JSON."""
        return self.monthly_spent_usd

    @computed_field
    @property
    def circuit_breaker_enabled(self) -> bool:
        """Convenience alias for UI checks."""
        return self.circuit_breaker_active


class TelemetrySummary(BaseModel):
    """Aggregated financial and token metrics across sessions."""

    model_config = ConfigDict(frozen=True)

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_savings_usd: float = 0.0
    daily_spent_usd: float = 0.0
    tokens_by_model: Dict[str, int] = Field(default_factory=dict)
    by_model: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    cost_by_provider: Dict[str, float] = Field(default_factory=dict)
    record_count: int = 0
    budget: BudgetStatus

    @computed_field
    @property
    def total_calls(self) -> int:
        """Convenience alias for record_count."""
        return self.record_count

    @computed_field
    @property
    def total_prompt_tokens(self) -> int:
        """Convenience alias for prompt_tokens serialized in JSON."""
        return self.prompt_tokens

    @computed_field
    @property
    def total_completion_tokens(self) -> int:
        """Convenience alias for completion_tokens serialized in JSON."""
        return self.completion_tokens

    @computed_field
    @property
    def daily_spend_usd(self) -> float:
        """Convenience alias for daily_spent_usd."""
        return self.daily_spent_usd

    @computed_field
    @property
    def total_spend_usd(self) -> float:
        """Convenience alias for total_cost_usd."""
        return self.total_cost_usd
