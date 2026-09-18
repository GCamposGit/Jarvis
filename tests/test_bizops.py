"""Deterministic tests for Autonomous Business Operations (BizOps) Engine & Guardrails."""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest
import httpx

from jarvis.core.bizops import (
    AutonomyLevel,
    BizOpsEngine,
    BizOpsRunResult,
    BizTask,
    PendingAction,
    PolicyGuardrail,
)
from jarvis.core.config import JarvisConfig
from jarvis.core.darkfac import DarkDemandSummary, DarkFactoryClient, DarkHubStatus
from jarvis.core.memory import EpisodicMemoryEngine
from jarvis.core.assistant import JarvisAssistant
from jarvis.core.models import UnifiedModelRouter
from jarvis.web.app import create_app


def test_guardrail_read_only_action():
    guard = PolicyGuardrail(default_level=AutonomyLevel.LEVEL_2_HITL)
    allowed, msg, pending = guard.evaluate_action(
        action_type="search_second_brain",
        title="Busca de Documento",
        payload={"query": "contrato"},
    )
    assert allowed is True
    assert pending is None
    assert "Nível 1" in msg


def test_guardrail_modifying_action_blocks_on_level_2():
    guard = PolicyGuardrail(default_level=AutonomyLevel.LEVEL_2_HITL)
    allowed, msg, pending = guard.evaluate_action(
        action_type="create_dark_factory_demand",
        title="Demanda de Auditoria",
        payload={"title": "Auditoria de Seguranca"},
    )
    assert allowed is False
    assert pending is not None
    assert isinstance(pending, PendingAction)
    assert pending.status == "pending_approval"
    assert "HITL" in msg


def test_guardrail_modifying_action_auto_approved_on_level_3():
    guard = PolicyGuardrail(default_level=AutonomyLevel.LEVEL_3_AUTONOMOUS)
    allowed, msg, pending = guard.evaluate_action(
        action_type="create_dark_factory_demand",
        title="Demanda de Auditoria",
        payload={"title": "Auditoria de Seguranca"},
        effective_level=AutonomyLevel.LEVEL_3_AUTONOMOUS,
    )
    assert allowed is True
    assert pending is None
    assert "Nível 3" in msg


@pytest.mark.anyio
async def test_bizops_engine_tasks_and_standup(tmp_path: Path):
    db_path = tmp_path / "bizops_test.db"
    mem_engine = EpisodicMemoryEngine(db_path=db_path)
    mem_engine.store_fact(key="prioridade_trimestral", value="Lançamento v1.0", category="project_context")

    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/cloud/status":
            return httpx.Response(200, json={"status": "active"})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    darkfac = DarkFactoryClient(http_client=client)
    bizops = BizOpsEngine(db_path=db_path, memory_engine=mem_engine, darkfac_client=darkfac)

    tasks = bizops.list_tasks()
    assert len(tasks) >= 3
    task_ids = [t.id for t in tasks]
    assert "task_standup" in task_ids

    # Trigger standup
    res = await bizops.trigger_task("task_standup")
    assert res.success is True
    assert "Briefing Operacional do Jarvis" in res.message
    assert "Status DarkHub" in res.message


@pytest.mark.anyio
async def test_bizops_engine_hitl_approval_flow(tmp_path: Path):
    db_path = tmp_path / "bizops_test.db"
    mem_engine = EpisodicMemoryEngine(db_path=db_path)

    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/cloud/status":
            return httpx.Response(200, json={"status": "active"})
        if request.url.path == "/api/demands/tickets":
            if request.method == "GET":
                return httpx.Response(200, json=[])
            elif request.method == "POST":
                return httpx.Response(200, json={"demand_id": "DEM-999", "status": "created"})
        if request.url.path == "/api/demands/next-id":
            return httpx.Response(200, json={"next_id": "DEM-999"})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    darkfac = DarkFactoryClient(http_client=client)
    bizops = BizOpsEngine(db_path=db_path, memory_engine=mem_engine, darkfac_client=darkfac)

    # 1. Trigger backlog hygiene (Level 2 HITL)
    res = await bizops.trigger_task("task_backlog_hygiene")
    assert res.success is True
    assert res.pending_action_id is not None

    # 2. Check pending actions
    pending_list = bizops.list_pending_actions(status="pending_approval")
    assert len(pending_list) >= 1
    action_id = res.pending_action_id

    # 3. Approve action
    appr_res = await bizops.approve_action(action_id, operator="TechLead")
    assert appr_res.success is True
    assert "aprovada e executada" in appr_res.message
    assert appr_res.data.get("demand_id") == "DEM-999"

    # 4. Verify status changed to executed
    act_after = bizops.get_pending_action(action_id)
    assert act_after.status == "executed"
    assert act_after.resolved_by == "TechLead"


def test_bizops_engine_rejection_flow(tmp_path: Path):
    db_path = tmp_path / "bizops_test.db"
    bizops = BizOpsEngine(db_path=db_path)

    # Manually save a pending action
    action = PendingAction(
        task_id="test_task",
        action_type="delete_critical_data",
        title="Operacao Arriscada",
        payload={"target": "all"},
    )
    bizops._save_pending_action(action)

    # Reject
    assert bizops.reject_action(action.id, operator="SecurityAuditor") is True
    rechecked = bizops.get_pending_action(action.id)
    assert rechecked.status == "rejected"
    assert rechecked.resolved_by == "SecurityAuditor"


@pytest.mark.anyio
async def test_assistant_bizops_mcp_tools():
    db_path = ":memory:"
    assistant = JarvisAssistant(
        memory_engine=EpisodicMemoryEngine(db_path=db_path),
        bizops_engine=BizOpsEngine(db_path=db_path),
    )

    tool_names = [t.name for t in assistant.mcp.list_tools()]
    assert "list_bizops_tasks" in tool_names
    assert "trigger_bizops_task" in tool_names
    assert "list_pending_bizops_actions" in tool_names
    assert "approve_bizops_action" in tool_names

    # Test executing list_bizops_tasks via MCP
    res = await assistant.mcp.execute_tool("list_bizops_tasks", {})
    assert not res.is_error
    assert len(res.output) >= 3


@pytest.mark.anyio
async def test_web_bizops_endpoints(tmp_path: Path):
    db_path = tmp_path / "web_bizops_test.db"
    cfg = JarvisConfig(memory_db_path=db_path)
    app = create_app(config=cfg)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Get tasks
        resp = await client.get("/api/bizops/tasks")
        assert resp.status_code == 200
        tasks = resp.json()
        assert len(tasks) >= 3

        # 2. Trigger task_standup
        resp = await client.post("/api/bizops/tasks/task_standup/trigger")
        assert resp.status_code == 200
        res_data = resp.json()
        assert res_data["success"] is True

        # 3. Get pending actions
        resp = await client.get("/api/bizops/pending")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
