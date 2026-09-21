"""Deterministic tests for Agent Harness Lifecycle Defense, Safety Guardrails and Code Sandbox (Milestone 4)."""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from jarvis.core.assistant import JarvisAssistant
from jarvis.core.config import JarvisConfig
from jarvis.core.harness import (
    DeterministicCodeSandbox,
    ExecutionRiskLevel,
    HarnessAuditEntry,
    HarnessAuditLedger,
    HarnessSafetyGuardrail,
    SafetyPolicy,
)
from jarvis.core.memory import EpisodicMemoryEngine
from jarvis.web.app import create_app


def test_harness_path_validation_and_blast_radius(tmp_path: Path):
    """Verify preflight path checks prevent path traversal and blast-radius escape."""
    policy = SafetyPolicy(
        forbidden_paths=["C:\\dev\\MeetingRelator", "C:\\Windows", str(tmp_path / "protected")],
        allowed_write_roots=[str(tmp_path / "allowed_workspace")],
    )
    guard = HarnessSafetyGuardrail(policy=policy)

    allowed_dir = tmp_path / "allowed_workspace"
    allowed_dir.mkdir(parents=True, exist_ok=True)
    allowed_file = allowed_dir / "output.txt"

    # 1. Allowed write inside workspace
    res1 = guard.validate_path(allowed_file, is_write=True)
    assert res1.allowed is True
    assert res1.risk_level == ExecutionRiskLevel.MEDIUM

    # 2. Path traversal attack with '..'
    traversal_path = str(allowed_dir) + "/../forbidden.txt"
    res2 = guard.validate_path(traversal_path, is_write=True)
    assert res2.allowed is False
    assert res2.risk_level == ExecutionRiskLevel.HIGH
    assert any("path traversal" in r.lower() for r in res2.reasons)

    # 3. Forbidden production repository path
    meeting_relator_file = "C:\\dev\\MeetingRelator\\meeting_database.db"
    res3 = guard.validate_path(meeting_relator_file, is_write=False)
    assert res3.allowed is False
    assert res3.risk_level == ExecutionRiskLevel.HIGH
    assert any("diretório protegido" in r.lower() for r in res3.reasons)

    # 4. Write outside allowed workspace
    outside_file = tmp_path / "somewhere_else" / "test.txt"
    res4 = guard.validate_path(outside_file, is_write=True)
    assert res4.allowed is False
    assert any("fora dos diretórios permitidos" in r.lower() for r in res4.reasons)


def test_harness_python_ast_validation():
    """Verify AST inspection detects syntax errors and prohibited system calls."""
    guard = HarnessSafetyGuardrail()

    # 1. Clean computational code
    valid_code = "total = sum(x**2 for x in range(10))\nprint(total)"
    res_valid = guard.validate_python_syntax(valid_code)
    assert res_valid.allowed is True
    assert res_valid.risk_level == ExecutionRiskLevel.SAFE

    # 2. Syntax error
    invalid_code = "def broken(\n  return 42"
    res_invalid = guard.validate_python_syntax(invalid_code)
    assert res_invalid.allowed is False
    assert any("Erro de sintaxe" in r for r in res_invalid.reasons)

    # 3. Dangerous system call os.system
    exploit_code = "import os\nos.system('del /f test.txt')"
    res_exploit = guard.validate_python_syntax(exploit_code)
    assert res_exploit.allowed is False
    assert any("os.system" in r for r in res_exploit.reasons)

    # 4. Prohibited ctypes import
    ctypes_code = "import ctypes\nctypes.CDLL('kernel32.dll')"
    res_ctypes = guard.validate_python_syntax(ctypes_code)
    assert res_ctypes.allowed is False
    assert any("ctypes" in r for r in res_ctypes.reasons)


def test_deterministic_sandbox_successful_run(tmp_path: Path):
    """Verify clean Python snippet runs isolated in sandbox returning stdout."""
    audit = HarnessAuditLedger(tmp_path / "audit.db")
    sandbox = DeterministicCodeSandbox(audit_ledger=audit)

    code = "numbers = [1, 2, 3, 4, 5]\nprint('RESULT=' + str(sum(numbers)))"
    res = sandbox.execute_python(code=code, timeout_seconds=5.0)

    assert res.success is True
    assert res.exit_code == 0
    assert "RESULT=15" in res.stdout.strip()
    assert res.violation_detected is False
    assert res.execution_time_ms > 0.0

    # Verify audit event
    events = audit.list_events(limit=5)
    assert len(events) >= 1
    assert events[0].status == "success"


def test_deterministic_sandbox_timeout_enforcement(tmp_path: Path):
    """Verify infinite loop is safely halted by the harness timeout."""
    audit = HarnessAuditLedger(tmp_path / "audit.db")
    sandbox = DeterministicCodeSandbox(audit_ledger=audit)

    infinite_loop_code = "import time\nwhile True:\n    time.sleep(0.05)"
    res = sandbox.execute_python(code=infinite_loop_code, timeout_seconds=0.6)

    assert res.success is False
    assert res.error_type == "TimeoutExpired"
    assert "excedeu o limite" in res.stderr
    assert res.violation_detected is True

    # Audit logged as timeout
    events = audit.list_events(limit=5)
    assert any(e.status == "timeout" for e in events)


def test_deterministic_sandbox_error_and_retro_recording(tmp_path: Path):
    """Verify runtime exception triggers automatic retrospective in EpisodicMemoryEngine."""
    mem_db = tmp_path / "mem.db"
    mem_engine = EpisodicMemoryEngine(db_path=mem_db)
    audit = HarnessAuditLedger(tmp_path / "audit.db")
    sandbox = DeterministicCodeSandbox(memory_engine=mem_engine, audit_ledger=audit)

    broken_script = "raise ZeroDivisionError('Division by zero in calculation')"
    res = sandbox.execute_python(code=broken_script, timeout_seconds=3.0)

    assert res.success is False
    assert res.exit_code != 0
    assert "ZeroDivisionError" in res.stderr
    assert res.retro_recorded is True

    # Check memory recorded failed attempt
    failures = mem_engine.query_failed_attempts(action_query="sandboxed_python_code")
    assert len(failures) >= 1
    assert "ZeroDivisionError" in failures[0].error_pattern or "código" in failures[0].lesson_learned


def test_harness_audit_ledger(tmp_path: Path):
    """Verify audit ledger event storage, querying, and stats."""
    audit = HarnessAuditLedger(tmp_path / "audit_test.db")

    # Record safe event
    audit.record_event(
        HarnessAuditEntry(
            action_type="tool_execution",
            target="search_second_brain",
            risk_level=ExecutionRiskLevel.SAFE,
            allowed=True,
            reasons=[],
            execution_time_ms=12.5,
            status="completed",
        )
    )

    # Record blocked event
    audit.record_event(
        HarnessAuditEntry(
            action_type="path_access",
            target="C:\\Windows\\System32",
            risk_level=ExecutionRiskLevel.HIGH,
            allowed=False,
            reasons=["Diretório de sistema proibido"],
            execution_time_ms=1.2,
            status="blocked",
        )
    )

    all_events = audit.list_events()
    assert len(all_events) == 2

    blocked = audit.list_events(only_blocked=True)
    assert len(blocked) == 1
    assert blocked[0].target == "C:\\Windows\\System32"

    stats = audit.get_stats()
    assert stats["total_events"] == 2
    assert stats["blocked_events"] == 1
    assert stats["allowed_events"] == 1
    assert stats["block_rate_pct"] == 50.0


@pytest.mark.anyio
async def test_assistant_harness_preflight_interception(tmp_path: Path):
    """Verify JarvisAssistant preflight intercepts dangerous tool arguments."""
    mem_db = tmp_path / "mem.db"
    mem_engine = EpisodicMemoryEngine(db_path=mem_db)
    audit = HarnessAuditLedger(tmp_path / "audit.db")
    guard = HarnessSafetyGuardrail()
    sandbox = DeterministicCodeSandbox(guardrail=guard, memory_engine=mem_engine, audit_ledger=audit)

    assistant = JarvisAssistant(
        memory_engine=mem_engine,
        harness_guardrail=guard,
        code_sandbox=sandbox,
        harness_audit=audit,
    )

    # 1. Test MCP tool run_sandboxed_python
    res_py = await assistant.mcp.execute_tool("run_sandboxed_python", {"code": "print('HELLO_SANDBOX')"})
    assert res_py.is_error is False
    assert "HELLO_SANDBOX" in res_py.output.get("stdout", "")

    # 2. Test MCP tool validate_execution_safety
    res_val = await assistant.mcp.execute_tool(
        "validate_execution_safety",
        {"target_path": "C:\\dev\\MeetingRelator\\secret.py", "is_write": True},
    )
    assert res_val.is_error is False
    assert res_val.output.get("allowed") is False

    # 3. Test MCP tool get_harness_audit_log
    res_log = await assistant.mcp.execute_tool("get_harness_audit_log", {"limit": 10})
    assert res_log.is_error is False
    assert len(res_log.output) >= 1


def test_harness_web_rest_endpoints(tmp_path: Path):
    """Verify REST endpoints under /api/harness/* in FastAPI web gateway."""
    mem_db = tmp_path / "mem.db"
    cfg = JarvisConfig(memory_db_path=mem_db)
    app = create_app(config=cfg)
    client = TestClient(app)

    # 1. POST /api/harness/sandbox/python (successful)
    r_code = client.post(
        "/api/harness/sandbox/python",
        json={"code": "res = 20 * 5\nprint(f'MATH={res}')"},
    )
    assert r_code.status_code == 200
    data = r_code.json()
    assert data["success"] is True
    assert "MATH=100" in data["stdout"]

    # 2. POST /api/harness/validate (blocked path)
    r_val = client.post(
        "/api/harness/validate",
        json={"target_path": "C:\\dev\\MeetingRelator\\main.py", "is_write": True},
    )
    assert r_val.status_code == 200
    val_data = r_val.json()
    assert val_data["allowed"] is False
    assert len(val_data["reasons"]) > 0

    # 3. GET /api/harness/audit
    r_audit = client.get("/api/harness/audit")
    assert r_audit.status_code == 200
    assert isinstance(r_audit.json(), list)

    # 4. GET /api/harness/status
    r_status = client.get("/api/harness/status")
    assert r_status.status_code == 200
    status_data = r_status.json()
    assert status_data["status"] == "operational"
    assert "stats" in status_data
    assert "policy" in status_data
