"""Domain models and enumerations for Autonomous Business Operations (BizOps)."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class AutonomyLevel(int, enum.Enum):
    """Graduated Autonomy Levels for safe agent operation."""

    LEVEL_1_READ_ONLY = 1  # Observational, reports, read-only probes
    LEVEL_2_HITL = 2       # Human-in-the-Loop: Proposes changes, requires explicit user approval
    LEVEL_3_AUTONOMOUS = 3  # Fully autonomous execution within strict pre-authorized policy limits


class BizTask(BaseModel):
    """An autonomous operational or business task managed by Jarvis."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    title: str
    description: str = ""
    task_type: str  # "daily_standup_briefing", "backlog_hygiene", "memory_compaction", "health_pulse", "custom"
    schedule_interval_sec: Optional[int] = None
    autonomy_level: AutonomyLevel = AutonomyLevel.LEVEL_2_HITL
    status: str = "idle"  # "idle", "running", "completed", "failed"
    last_run_at: Optional[str] = None
    last_result: Optional[str] = None
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PendingAction(BaseModel):
    """A proposed external action awaiting Human-in-the-Loop review (Level 2)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: f"act_{uuid.uuid4().hex[:8]}")
    task_id: Optional[str] = None
    action_type: str  # "create_darkfac_demand", "notify_operator", "db_update", "api_call"
    title: str
    description: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    autonomy_level_required: AutonomyLevel = AutonomyLevel.LEVEL_2_HITL
    status: str = "pending_approval"  # "pending_approval", "approved", "rejected", "executed"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None


class BizOpsRunResult(BaseModel):
    """Result of running or triggering a BizOps task."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    success: bool
    message: str
    pending_action_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
