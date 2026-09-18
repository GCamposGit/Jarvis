"""Domain models and data structures for the Deterministic Agent Harness (Milestone 4)."""

from __future__ import annotations

import enum
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ExecutionRiskLevel(int, enum.Enum):
    """Risk classification for agent actions and code execution."""

    SAFE = 1      # Pure compute, read-only search, factual retrieval
    MEDIUM = 2    # Read/write within authorized workspace or temp areas
    HIGH = 3      # OS processes, file deletion, cross-repository access, shell execution


class SafetyPolicy(BaseModel):
    """Configuration constraints defining the blast radius and safety limits of the harness."""

    model_config = ConfigDict(frozen=True)

    forbidden_paths: List[str] = Field(
        default_factory=lambda: [
            "C:\\Windows",
            "C:\\Program Files",
            "C:\\Program Files (x86)",
            "C:\\dev\\MeetingRelator",  # Strictly protected production repo
            ".git",
        ]
    )
    allowed_write_roots: List[str] = Field(
        default_factory=lambda: [
            str(Path(os.getcwd()).resolve()),
            str(Path(os.environ.get("TEMP", "C:\\Temp")).resolve()),
        ]
    )
    forbidden_commands: List[str] = Field(
        default_factory=lambda: [
            "format",
            "rmdir /s",
            "del /f",
            "del /s",
            "Remove-Item -Recurse -Force",
            "mkfs",
            "shutdown",
        ]
    )
    forbidden_ast_calls: List[str] = Field(
        default_factory=lambda: [
            "os.system",
            "subprocess.Popen",
            "subprocess.call",
            "shutil.rmtree",
            "ctypes",
        ]
    )
    timeout_seconds: float = 5.0
    max_output_chars: int = 8000
    allow_network_in_sandbox: bool = False


class PreflightCheckResult(BaseModel):
    """Outcome of a pre-execution safety evaluation."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    risk_level: ExecutionRiskLevel
    reasons: List[str] = Field(default_factory=list)
    suggested_action: str = ""


class SandboxExecutionResult(BaseModel):
    """Telemetry and outcome of a sandboxed execution."""

    model_config = ConfigDict(frozen=True)

    success: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    violation_detected: bool = False
    error_type: Optional[str] = None
    retro_recorded: bool = False


class HarnessAuditEntry(BaseModel):
    """Auditable ledger record for every agent tool or code execution."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: f"audit_{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action_type: str
    target: str
    risk_level: ExecutionRiskLevel
    allowed: bool
    reasons: List[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    status: str = "completed"
    details: Dict[str, Any] = Field(default_factory=dict)
