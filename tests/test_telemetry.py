"""Deterministic tests for Real-time Token Consumption Telemetry, Budget Governance and Circuit Breaker (Milestone 5)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from jarvis.core.assistant import AssistantTurnResult, JarvisAssistant
from jarvis.core.config import JarvisConfig
from jarvis.core.models import ChatMessage, ModelResponse, UnifiedModelRouter
from jarvis.core.telemetry import (
    BudgetPolicy,
    BudgetStatus,
    TelemetryEngine,
    TelemetrySummary,
    TokenUsageRecord,
)
from jarvis.core.telemetry.engine import calculate_cost
from jarvis.web.app import create_app


def test_pricing_calculation_and_savings():
    """Verify deterministic pricing calculation across local and cloud models."""
    # 1. Local Ollama: always $0.00
    cost_local, base_local = calculate_cost(
        model="qwen-code-deep:latest",
        provider="ollama",
        tokens_prompt=1000,
        tokens_completion=500,
    )
    assert cost_local == 0.0
    savings_local = max(0.0, base_local - cost_local)
    # Baseline commercial rate: 1000*$5/1M + 500*$15/1M = $0.005 + $0.0075 = $0.0125
    assert savings_local == pytest.approx(0.0125, abs=1e-5)

    # 2. Gemini 2.5 Flash: $0.075 / $0.30 per 1M
    cost_gemini, base_gemini = calculate_cost(
        model="google/gemini-2.5-flash",
        provider="openrouter",
        tokens_prompt=1_000_000,
        tokens_completion=1_000_000,
    )
    assert cost_gemini == pytest.approx(0.375, abs=1e-5)
    # Baseline commercial is $20.00 per 2M, so savings = 20.00 - 0.375 = 19.625
    savings_gemini = max(0.0, base_gemini - cost_gemini)
    assert savings_gemini == pytest.approx(19.625, abs=1e-5)

    # 3. DeepSeek V4.1 Flash: $0.14 / $0.28 per 1M
    cost_deepseek, _ = calculate_cost(
        model="deepseek/deepseek-v4.1-flash",
        provider="openrouter",
        tokens_prompt=1_000_000,
        tokens_completion=1_000_000,
    )
    assert cost_deepseek == pytest.approx(0.420, abs=1e-5)

    # 4. Unknown model fallback
    cost_unknown, _ = calculate_cost(
        model="custom-model",
        provider="custom-provider",
        tokens_prompt=1000,
        tokens_completion=1000,
    )
    assert cost_unknown >= 0.0


def test_telemetry_engine_records_and_summary(tmp_path: Path):
    """Verify recording token transactions, persistence in SQLite, and summary roll-up."""
    db_file = tmp_path / "test_memory.db"
    engine = TelemetryEngine(db_file)

    # Record 1: Local Ollama turn
    rec1 = engine.record_usage(
        model="qwen-code-deep:latest",
        provider="ollama",
        tokens_prompt=500,
        tokens_completion=200,
        latency_ms=120.0,
        session_id="session-1",
    )
    assert rec1.cost_usd == 0.0
    assert rec1.savings_usd > 0.0
    assert rec1.total_tokens == 700

    # Record 2: Cloud Gemini turn
    rec2 = engine.record_usage(
        model="google/gemini-2.5-flash",
        provider="openrouter",
        tokens_prompt=2000,
        tokens_completion=800,
        latency_ms=450.0,
        session_id="session-1",
    )
    assert rec2.cost_usd > 0.0
    assert rec2.total_tokens == 2800

    # Verify Summary Aggregation
    summary = engine.get_summary()
    assert summary.total_calls == 2
    assert summary.total_prompt_tokens == 2500
    assert summary.total_completion_tokens == 1000
    assert summary.total_tokens == 3500
    assert summary.total_cost_usd == pytest.approx(rec2.cost_usd, abs=1e-5)
    assert summary.total_savings_usd == pytest.approx(rec1.savings_usd + rec2.savings_usd, abs=1e-5)

    assert "qwen-code-deep:latest" in summary.by_model
    assert summary.by_model["qwen-code-deep:latest"]["calls"] == 1
    assert summary.by_model["qwen-code-deep:latest"]["cost_usd"] == 0.0
    assert "google/gemini-2.5-flash" in summary.by_model

    # Verify List Records
    records = engine.list_records(limit=10)
    assert len(records) == 2
    assert records[0].id == rec2.id  # Latest first


def test_budget_status_and_circuit_breaker(tmp_path: Path):
    """Verify budget threshold transitions (ok -> warning -> exceeded) and policy updates."""
    db_file = tmp_path / "test_memory.db"
    engine = TelemetryEngine(
        db_path=db_file,
        policy=BudgetPolicy(
            daily_limit_usd=1.00,
            monthly_limit_usd=25.00,
            warning_threshold=0.80,
            circuit_breaker_enabled=True,
        ),
    )

    # Initial status: ok
    st0 = engine.get_budget_status()
    assert st0.status == "ok"
    assert st0.circuit_breaker_active is False
    assert st0.daily_spend_usd == 0.0

    # Record spend below warning (0.50 USD)
    engine.record_usage(
        model="custom-cloud",
        provider="openrouter",
        tokens_prompt=1000,
        tokens_completion=1000,
        cost_usd=0.50,
    )
    st1 = engine.get_budget_status()
    assert st1.status == "ok"
    assert st1.daily_spend_usd == 0.50
    assert st1.circuit_breaker_active is False

    # Record additional spend crossing 80% (0.35 USD -> 0.85 total)
    engine.record_usage(
        model="custom-cloud",
        provider="openrouter",
        tokens_prompt=1000,
        tokens_completion=1000,
        cost_usd=0.35,
    )
    st2 = engine.get_budget_status()
    assert st2.status == "warning"
    assert st2.daily_spend_usd == 0.85
    assert st2.circuit_breaker_active is False

    # Record additional spend crossing 100% (0.20 USD -> 1.05 total)
    engine.record_usage(
        model="custom-cloud",
        provider="openrouter",
        tokens_prompt=1000,
        tokens_completion=1000,
        cost_usd=0.20,
    )
    st3 = engine.get_budget_status()
    assert st3.status == "exceeded"
    assert st3.daily_spend_usd == 1.05
    assert st3.circuit_breaker_active is True

    # Increase daily limit via policy update
    new_policy = engine.update_policy(daily_limit_usd=5.00)
    assert new_policy.daily_limit_usd == 5.00

    st4 = engine.get_budget_status()
    assert st4.status == "ok"
    assert st4.circuit_breaker_active is False


@pytest.mark.anyio
async def test_assistant_telemetry_mcp_tools_and_circuit_breaker(tmp_path: Path):
    """Verify assistant MCP tools for telemetry and automated circuit breaker fallback."""
    cfg = JarvisConfig(memory_db_path=tmp_path / "assistant_telemetry.db")
    engine = TelemetryEngine(
        db_path=cfg.memory_db_path,
        policy=BudgetPolicy(
            daily_limit_usd=1.00,
            circuit_breaker_enabled=True,
            fallback_to_local_on_limit=True,
            default_fallback_model="qwen-code-deep:latest",
        ),
    )

    assistant = JarvisAssistant(config=cfg, telemetry_engine=engine)

    # 1. Check MCP Tools Registration
    tools = assistant.mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "get_telemetry_summary" in tool_names
    assert "get_budget_status" in tool_names
    assert "update_budget_policy" in tool_names

    # 2. Execute get_budget_status via MCP
    res_b = await assistant.mcp.execute_tool("get_budget_status", {})
    assert not res_b.is_error
    assert res_b.output["status"] == "ok"

    # 3. Execute update_budget_policy via MCP
    res_u = await assistant.mcp.execute_tool(
        "update_budget_policy",
        {"daily_limit_usd": 2.50, "alert_threshold_pct": 80.0},
    )
    assert not res_u.is_error
    assert res_u.output["daily_limit_usd"] == 2.50

    # 4. Simulate Assistant Chat and Telemetry Recording
    fake_response = ModelResponse(
        text="Jarvis responde com precisão.",
        model="google/gemini-2.5-flash",
        provider="openrouter",
        tokens_prompt=120,
        tokens_completion=45,
        latency_ms=250.0,
        cost_usd=0.0000225,
    )

    with patch.object(assistant.models, "generate", new=AsyncMock(return_value=fake_response)):
        turn = await assistant.chat(
            user_message="Qual o status do projeto?",
            model="google/gemini-2.5-flash",
            provider="openrouter",
        )

        assert turn.tokens_prompt == 120
        assert turn.tokens_completion == 45
        assert turn.cost_usd == pytest.approx(0.0000225, abs=1e-7)

        # Verify recorded in telemetry engine
        summary = assistant.telemetry.get_summary()
        assert summary.total_calls == 1
        assert summary.total_tokens == 165

    # 5. Circuit Breaker Fallback Test
    # Force budget exceeded
    assistant.telemetry.update_policy(daily_limit_usd=0.00001)

    captured_call = {}

    async def mock_generate_cb(*args, **kwargs):
        captured_call["model"] = kwargs.get("model")
        captured_call["provider"] = kwargs.get("provider")
        return ModelResponse(
            text="Resposta local via fallback.",
            model=kwargs.get("model", "qwen-code-deep:latest"),
            provider=kwargs.get("provider", "ollama"),
            tokens_prompt=50,
            tokens_completion=20,
            latency_ms=100.0,
            cost_usd=0.0,
        )

    with patch.object(assistant.models, "generate", side_effect=mock_generate_cb):
        # Request expensive model, but budget is exceeded
        turn_cb = await assistant.chat(
            user_message="Executar cálculo complexo",
            model="anthropic/claude-sonnet-4.6",
            provider="openrouter",
        )

        # Verified routed to local Ollama fallback at $0
        assert captured_call["model"] == "qwen-code-deep:latest"
        assert captured_call["provider"] == "ollama"
        assert turn_cb.cost_usd == 0.0


def test_telemetry_web_api_endpoints(tmp_path: Path):
    """Verify FastAPI REST endpoints for summary, budget, policy updates, and records."""
    cfg = JarvisConfig(memory_db_path=tmp_path / "api_telemetry.db")
    app = create_app(config=cfg)
    client = TestClient(app)

    # 1. GET /api/telemetry/summary
    res_s = client.get("/api/telemetry/summary")
    assert res_s.status_code == 200
    data_s = res_s.json()
    assert "total_tokens" in data_s
    assert "total_cost_usd" in data_s
    assert "total_savings_usd" in data_s

    # 2. GET /api/telemetry/budget
    res_b = client.get("/api/telemetry/budget")
    assert res_b.status_code == 200
    data_b = res_b.json()
    assert "status" in data_b
    assert "policy" in data_b
    assert data_b["status"]["status"] in ("ok", "warning", "exceeded")

    # 3. POST /api/telemetry/budget (update policy)
    res_u = client.post(
        "/api/telemetry/budget",
        json={
            "daily_limit_usd": 3.50,
            "monthly_limit_usd": 50.00,
            "circuit_breaker_enabled": True,
            "fallback_to_local_on_limit": True,
        },
    )
    assert res_u.status_code == 200
    data_u = res_u.json()
    assert data_u["policy"]["daily_limit_usd"] == 3.50
    assert data_u["policy"]["monthly_limit_usd"] == 50.00

    # 4. Record usage directly in engine and verify records endpoint
    app.state.telemetry.record_usage(
        model="google/gemini-2.5-flash",
        provider="openrouter",
        tokens_prompt=500,
        tokens_completion=150,
        latency_ms=200.0,
    )

    res_r = client.get("/api/telemetry/records?limit=10")
    assert res_r.status_code == 200
    records = res_r.json()
    assert len(records) >= 1
    assert records[0]["model"] == "google/gemini-2.5-flash"
    assert records[0]["prompt_tokens"] == 500
