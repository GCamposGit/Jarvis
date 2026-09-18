"""Isolated Deterministic Code Sandbox for the Agent Harness (Milestone 4)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from jarvis.core.harness.audit import HarnessAuditLedger
from jarvis.core.harness.guardrail import HarnessSafetyGuardrail
from jarvis.core.harness.models import (
    ExecutionRiskLevel,
    HarnessAuditEntry,
    PreflightCheckResult,
    SandboxExecutionResult,
)
from jarvis.core.memory import EpisodicMemoryEngine

logger = logging.getLogger("jarvis.core.harness.sandbox")


class DeterministicCodeSandbox:
    """Executes code snippets in an isolated, monitored Python environment with strict resource limits."""

    def __init__(
        self,
        guardrail: Optional[HarnessSafetyGuardrail] = None,
        memory_engine: Optional[EpisodicMemoryEngine] = None,
        audit_ledger: Optional[HarnessAuditLedger] = None,
    ) -> None:
        self.guardrail = guardrail or HarnessSafetyGuardrail()
        self.memory = memory_engine
        self.audit = audit_ledger or HarnessAuditLedger(":memory:")

    def execute_python(
        self,
        code: str,
        timeout_seconds: Optional[float] = None,
        record_retrospectives: bool = True,
    ) -> SandboxExecutionResult:
        """Run Python code deterministically in an isolated subprocess."""
        start_time = time.perf_counter()
        timeout = timeout_seconds or self.guardrail.policy.timeout_seconds
        max_chars = self.guardrail.policy.max_output_chars

        # 1. Preflight AST validation
        preflight: PreflightCheckResult = self.guardrail.validate_python_syntax(code)
        if not preflight.allowed:
            dur_ms = (time.perf_counter() - start_time) * 1000
            err_msg = "; ".join(preflight.reasons)
            res = SandboxExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=err_msg,
                execution_time_ms=dur_ms,
                violation_detected=True,
                error_type="PreflightSafetyViolation",
            )
            self.audit.record_event(
                HarnessAuditEntry(
                    action_type="sandboxed_python_preflight",
                    target="ast_code_validation",
                    risk_level=preflight.risk_level,
                    allowed=False,
                    reasons=preflight.reasons,
                    execution_time_ms=dur_ms,
                    status="blocked",
                    details={"code_snippet": code[:200]},
                )
            )
            return res

        # 2. Subprocess execution in an isolated temporary environment
        with tempfile.TemporaryDirectory(prefix="jarvis_sandbox_") as tmp_dir:
            script_path = Path(tmp_dir) / "sandbox_exec.py"
            script_path.write_text(code, encoding="utf-8")

            # Environment sanitation: remove sensitive variables if any
            clean_env = os.environ.copy()
            clean_env.pop("ANTHROPIC_API_KEY", None)
            clean_env.pop("OPENAI_API_KEY", None)
            clean_env.pop("GEMINI_API_KEY", None)
            clean_env.pop("GROQ_API_KEY", None)
            clean_env.pop("OPENROUTER_API_KEY", None)

            try:
                sub_proc = subprocess.run(
                    [sys.executable, "-I", str(script_path)],
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=clean_env,
                )
                dur_ms = (time.perf_counter() - start_time) * 1000
                stdout = sub_proc.stdout[:max_chars]
                stderr = sub_proc.stderr[:max_chars]
                is_success = sub_proc.returncode == 0
                error_type = None if is_success else "ProcessExitError"

                retro_saved = False
                if not is_success and record_retrospectives and self.memory:
                    try:
                        self.memory.record_failed_attempt(
                            action="sandboxed_python_code",
                            error_pattern=stderr.splitlines()[-1] if stderr.splitlines() else "Erro desconhecido",
                            lesson_learned=f"Código gerou erro (código {sub_proc.returncode}). Ajustar lógica antes de reexecutar.",
                        )
                        retro_saved = True
                    except Exception as exc:
                        logger.debug("Failed to record retro: %s", exc)

                res = SandboxExecutionResult(
                    success=is_success,
                    exit_code=sub_proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time_ms=dur_ms,
                    violation_detected=False,
                    error_type=error_type,
                    retro_recorded=retro_saved,
                )

                self.audit.record_event(
                    HarnessAuditEntry(
                        action_type="sandboxed_python_execution",
                        target="isolated_subprocess",
                        risk_level=ExecutionRiskLevel.SAFE if is_success else ExecutionRiskLevel.MEDIUM,
                        allowed=True,
                        reasons=[],
                        execution_time_ms=dur_ms,
                        status="success" if is_success else "failed",
                        details={"exit_code": sub_proc.returncode, "stdout_len": len(stdout)},
                    )
                )
                return res

            except subprocess.TimeoutExpired:
                dur_ms = (time.perf_counter() - start_time) * 1000
                timeout_msg = f"Tempo de execução excedeu o limite máximo de {timeout}s configurado pelo harness."
                retro_saved = False
                if record_retrospectives and self.memory:
                    try:
                        self.memory.record_failed_attempt(
                            action="sandboxed_python_code",
                            error_pattern=f"TimeoutExpired ({timeout}s)",
                            lesson_learned="Evitar loops não-terminantes ou operações intensivas sem paginação.",
                        )
                        retro_saved = True
                    except Exception:
                        pass

                res = SandboxExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=timeout_msg,
                    execution_time_ms=dur_ms,
                    violation_detected=True,
                    error_type="TimeoutExpired",
                    retro_recorded=retro_saved,
                )
                self.audit.record_event(
                    HarnessAuditEntry(
                        action_type="sandboxed_python_execution",
                        target="isolated_subprocess",
                        risk_level=ExecutionRiskLevel.HIGH,
                        allowed=True,
                        reasons=[timeout_msg],
                        execution_time_ms=dur_ms,
                        status="timeout",
                        details={"timeout_seconds": timeout},
                    )
                )
                return res

            except Exception as exc:
                dur_ms = (time.perf_counter() - start_time) * 1000
                res = SandboxExecutionResult(
                    success=False,
                    exit_code=-2,
                    stdout="",
                    stderr=str(exc),
                    execution_time_ms=dur_ms,
                    violation_detected=False,
                    error_type=type(exc).__name__,
                )
                return res
