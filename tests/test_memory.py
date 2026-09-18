"""Deterministic tests for Jarvis Episodic Memory Engine & MCP Integration."""

from __future__ import annotations

import pytest
import httpx
from pathlib import Path

from jarvis.core.config import JarvisConfig
from jarvis.core.memory import EpisodicMemoryEngine, MemoryFact, DecisionRecord, FailedAttempt
from jarvis.core.assistant import JarvisAssistant
from jarvis.core.models import UnifiedModelRouter
from jarvis.web.app import create_app


def test_memory_facts_crud(tmp_path: Path):
    db_path = tmp_path / "test_memory.db"
    engine = EpisodicMemoryEngine(db_path=db_path)

    # 1. Store facts
    f1 = engine.store_fact(
        key="user_name",
        value="Guilherme Campos",
        category="user_preference",
        tags=["user", "profile"],
    )
    assert f1.key == "user_name"
    assert f1.value == "Guilherme Campos"

    f2 = engine.store_fact(
        key="preferred_language",
        value="Python 3.12+ com Pydantic v2",
        category="tech_stack",
        tags=["python", "standards"],
    )

    # 2. Get fact
    fact = engine.get_fact("user_name")
    assert fact is not None
    assert fact.value == "Guilherme Campos"
    assert "user" in fact.tags

    # 3. Recall facts with query
    recalled = engine.recall_facts(query="Guilherme")
    assert len(recalled) == 1
    assert recalled[0].key == "user_name"

    # 4. Recall by category
    tech_facts = engine.recall_facts(category="tech_stack")
    assert len(tech_facts) == 1
    assert tech_facts[0].key == "preferred_language"

    # 5. Delete fact
    assert engine.delete_fact("user_name")
    assert engine.get_fact("user_name") is None


def test_memory_decisions_logging(tmp_path: Path):
    db_path = tmp_path / "test_memory.db"
    engine = EpisodicMemoryEngine(db_path=db_path)

    dec = engine.log_decision(
        title="Uso de SQLite para Memória Local",
        decision="Utilizar SQLite embutido ao invés de banco vetorial externo para memória primária.",
        alternatives=["ChromaDB em contêiner", "Pinecone em nuvem"],
        rationale="Elimina latência de rede, custo de hospedagem e preserva privacidade total.",
        problem_statement="Como persistir preferências e episódios sem vazar dados ou depender de infraestrutura pesada?",
    )
    assert isinstance(dec, DecisionRecord)
    assert dec.title == "Uso de SQLite para Memória Local"
    assert len(dec.alternatives) == 2

    # Query decisions
    found = engine.query_decisions(query="SQLite")
    assert len(found) == 1
    assert "privacidade total" in found[0].rationale


def test_memory_failed_attempts_retro(tmp_path: Path):
    db_path = tmp_path / "test_memory.db"
    engine = EpisodicMemoryEngine(db_path=db_path)

    engine.record_failed_attempt(
        action="Instalar chromadb compilando dependências C++ no Windows",
        error_pattern="error: Microsoft Visual C++ 14.0 or greater is required",
        lesson_learned="Preferir bibliotecas embutidas na stdlib (sqlite3) ou bindings pré-compilados sem build em tempo de execução.",
    )

    fails = engine.query_failed_attempts(action_query="chromadb")
    assert len(fails) == 1
    assert "Visual C++" in fails[0].error_pattern
    assert "sqlite3" in fails[0].lesson_learned


def test_memory_context_brief_generation(tmp_path: Path):
    db_path = tmp_path / "test_memory.db"
    engine = EpisodicMemoryEngine(db_path=db_path)

    engine.store_fact(key="project_goal", value="Construir o melhor assistente autônomo", category="project_context")
    engine.log_decision(title="Arquitetura Headless", decision="Manter core desacoplado de UI", rationale="Permite CLI e testes sem browser")
    engine.record_failed_attempt(action="acoplar logica na rota fastapi", error_pattern="Dificuldade de rodar CLI", lesson_learned="Manter logica em jarvis.core")

    brief = engine.build_context_brief(current_topic="project_goal")
    assert "[MEMÓRIA EPISÓDICA E CONTEXTO DO SEGUNDO CÉREBRO]" in brief
    assert "Construir o melhor assistente autônomo" in brief
    assert "Arquitetura Headless" in brief
    assert "⚠️ Lições de Falhas Anteriores" in brief


@pytest.mark.anyio
async def test_assistant_with_memory_tools_and_context(tmp_path: Path):
    db_path = tmp_path / "test_memory.db"
    mem_engine = EpisodicMemoryEngine(db_path=db_path)
    mem_engine.store_fact(key="diretriz_principal", value="Sempre validar código com pytest", category="business_rule")

    def mock_model_handler(request: httpx.Request):
        req_json = httpx.Response(200, json={
            "message": {
                "role": "assistant",
                "content": "Entendido. Memória consultada com sucesso.",
            },
            "prompt_eval_count": 20,
            "eval_count": 15,
        })
        return req_json

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_model_handler))
    models = UnifiedModelRouter(http_client=client)
    assistant = JarvisAssistant(model_router=models, memory_engine=mem_engine)

    # Check registered memory tools
    tool_names = [t.name for t in assistant.mcp.list_tools()]
    assert "store_memory_fact" in tool_names
    assert "recall_memory" in tool_names
    assert "log_decision" in tool_names
    assert "record_failed_approach" in tool_names

    res = await assistant.chat("Qual é a diretriz de testes do projeto?")
    assert "Memória consultada" in res.response_text


@pytest.mark.anyio
async def test_web_memory_endpoints(tmp_path: Path):
    db_path = tmp_path / "web_test_memory.db"
    cfg = JarvisConfig(memory_db_path=db_path)
    app = create_app(config=cfg)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Post fact
        resp = await client.post(
            "/api/memory/facts",
            json={
                "key": "framework_preferido",
                "value": "FastAPI + Pydantic v2",
                "category": "tech_stack",
                "tags": ["web", "api"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "framework_preferido"

        # 2. Get facts
        resp = await client.get("/api/memory/facts?query=FastAPI")
        assert resp.status_code == 200
        facts = resp.json()
        assert len(facts) == 1
        assert facts[0]["key"] == "framework_preferido"

        # 3. Post decision
        resp = await client.post(
            "/api/memory/decisions",
            json={
                "title": "Decisao de Design",
                "decision": "Manter tudo tipado",
                "problem_statement": "Evitar bugs em tempo de execucao",
                "rationale": "Pydantic valida em runtime",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "Decisao de Design"

        # 4. Get decisions
        resp = await client.get("/api/memory/decisions")
        assert resp.status_code == 200
        decisions = resp.json()
        assert len(decisions) >= 1

        # 5. Delete fact
        resp = await client.delete("/api/memory/facts/framework_preferido")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
