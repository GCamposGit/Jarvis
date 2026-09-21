"""Deterministic tests for Jarvis Assistant conversational engine."""

from __future__ import annotations

import asyncio
import json
import httpx

from jarvis.core.assistant import JarvisAssistant
from jarvis.core.config import JarvisConfig
from jarvis.core.darkfac import DarkFactoryClient
from jarvis.core.models import UnifiedModelRouter


def test_assistant_darkhub_status_shortcut():
    async def _run():
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/cloud/status":
                return httpx.Response(200, json={"status": "active"})
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        darkfac = DarkFactoryClient(http_client=client)
        assistant = JarvisAssistant(darkfac_client=darkfac)

        res = await assistant.chat("Qual é o status da Dark Factory?")
        assert "Online" in res.response_text
        assert len(res.tools_executed) == 1
        assert res.tools_executed[0]["tool"] == "check_dark_factory_status"

    asyncio.run(_run())


def test_assistant_regular_chat():
    async def _run():
        def mock_model_handler(request: httpx.Request):
            if request.url.path == "/api/chat":
                return httpx.Response(
                    200,
                    json={
                        "message": {"role": "assistant", "content": "Olá, eu sou o Jarvis. Como posso te ajudar hoje?"},
                        "prompt_eval_count": 10,
                        "eval_count": 12,
                    },
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_model_handler))
        models = UnifiedModelRouter(http_client=client)
        assistant = JarvisAssistant(model_router=models)

        res = await assistant.chat("Olá Jarvis")
        assert "Olá, eu sou o Jarvis" in res.response_text

    asyncio.run(_run())


def test_assistant_tool_calling_and_second_turn():
    async def _run():
        call_count = 0

        def mock_model_handler(request: httpx.Request):
            nonlocal call_count
            if request.url.path == "/api/chat":
                call_count += 1
                if call_count == 1:
                    return httpx.Response(
                        200,
                        json={
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "search_second_brain",
                                            "arguments": '{"query": "política de IA"}',
                                        }
                                    }
                                ],
                            },
                        },
                    )
                elif call_count == 2:
                    return httpx.Response(
                        200,
                        json={
                            "message": {
                                "role": "assistant",
                                "content": "O código oficial da Política de IA é PO-CORP-007.",
                            },
                        },
                    )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_model_handler))
        models = UnifiedModelRouter(http_client=client)
        assistant = JarvisAssistant(model_router=models)

        res = await assistant.chat("Qual o número da política de IA aprovada?")
        assert call_count == 2
        assert len(res.tools_executed) == 1
        assert res.tools_executed[0]["tool"] == "search_second_brain"
        assert "PO-CORP-007" in res.response_text

    asyncio.run(_run())


def test_assistant_pure_memory_store_directive():
    async def _run():
        assistant = JarvisAssistant()
        res = await assistant.chat("Jarvis, guarde o fato de que a expectativa de faturamento mensal da mansão do lago é de R$10.000")
        assert "Fato registrado com sucesso na sua memória episódica" in res.response_text
        assert len(res.tools_executed) == 1
        assert res.tools_executed[0]["tool"] == "store_memory_fact"

        recalled = assistant.memory.recall_facts(query="mansão do lago")
        assert len(recalled) >= 1
        assert "R$10.000" in recalled[0].value

    asyncio.run(_run())


def test_assistant_compound_store_and_question_with_episodic_memory():
    async def _run():
        prompt_received = ""

        def mock_model_handler(request: httpx.Request):
            nonlocal prompt_received
            if request.url.path == "/api/chat":
                body = json.loads(request.read().decode())
                prompt_received = json.dumps(body.get("messages", []), ensure_ascii=False)
                return httpx.Response(
                    200,
                    json={
                        "message": {
                            "role": "assistant",
                            "content": "A expectativa de receita mensal da mansão do Lago é de R$10.000 e a operação é conduzida pela Cintia, sua esposa.",
                        },
                    },
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_model_handler))
        models = UnifiedModelRouter(http_client=client)
        assistant = JarvisAssistant(model_router=models)

        user_input = (
            "Jarvis, guarde o fato de que a expectaiva de faturamento mensal da mansão do lago é de R$10.000 "
            "e quem toca a operação é a Cintia,  minha esposa..Qual a expectativa de receita mensal da mansão do Lago e quem toca a operação?"
        )
        res = await assistant.chat(user_input)

        # 1. Stored fact tool was recorded
        assert any(t["tool"] == "store_memory_fact" for t in res.tools_executed)
        # 2. Fact was persisted in SQLite
        recalled = assistant.memory.recall_facts(query="Cintia")
        assert len(recalled) >= 1
        assert "R$10.000" in recalled[0].value
        # 3. System prompt contained episodic memory block with the newly stored fact
        assert "MEMÓRIA EPISÓDICA" in prompt_received
        assert "Cintia" in prompt_received
        # 4. Response answered the question accurately
        assert "R$10.000" in res.response_text
        assert "Cintia" in res.response_text
        # 5. search_second_brain was NEVER called
        assert not any(t.get("tool") == "search_second_brain" for t in res.tools_executed)

    asyncio.run(_run())


def test_assistant_sua_memoria_resolution():
    async def _run():
        assistant = JarvisAssistant()
        assistant.memory.store_fact(
            key="mansao_do_lago_operacao",
            value="A mansão do lago fatura R$10.000/mês e a operação é tocada pela Cintia.",
            category="business_rule",
            tags=["mansao", "lago", "cintia"],
        )

        from jarvis.core.models import ChatMessage
        history = [
            ChatMessage(role="user", content="Qual a expectativa de receita mensal da mansão do Lago e quem toca a operação?"),
            ChatMessage(role="assistant", content="Para te responder, preciso consultar a fonte."),
        ]

        res = await assistant.chat("sua memória", history=history)
        assert len(res.tools_executed) == 1
        assert res.tools_executed[0]["tool"] == "recall_memory"
        assert "R$10.000" in res.response_text
        assert "Cintia" in res.response_text
        assert "memória episódica" in res.response_text

    asyncio.run(_run())


def test_assistant_qwen_raw_json_tool_calling_executes_and_synthesizes():
    async def _run():
        turn_count = 0

        def mock_handler(request: httpx.Request):
            nonlocal turn_count
            if request.url.path == "/api/chat":
                turn_count += 1
                if turn_count == 1:
                    # First turn: model outputs raw JSON tool call in content
                    raw_content = '{"name": "recall_memory", "arguments": {"query": "mansão do lago"}}'
                    return httpx.Response(
                        200,
                        json={
                            "message": {"role": "assistant", "content": raw_content},
                            "prompt_eval_count": 80,
                            "eval_count": 20,
                        },
                    )
                else:
                    # Second turn (synthesis): model receives tool outputs and provides clean answer
                    return httpx.Response(
                        200,
                        json={
                            "message": {
                                "role": "assistant",
                                "content": "A expectativa de receita mensal da Mansão do Lago é de R$10.000 e quem conduz a operação é a Cintia, sua esposa.",
                            },
                            "prompt_eval_count": 120,
                            "eval_count": 35,
                        },
                    )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        cfg = JarvisConfig(default_provider="ollama", default_local_model="qwen-code-deep:latest")
        router = UnifiedModelRouter(config=cfg, http_client=client)
        assistant = JarvisAssistant(config=cfg, model_router=router)

        # Store test fact
        assistant.memory.store_fact(
            key="mansao_do_lago_operacao",
            value="A mansão do lago fatura R$10.000/mês e a operação é tocada pela Cintia.",
            category="business_rule",
            tags=["mansao", "lago", "cintia"],
        )

        res = await assistant.chat("Qual é a expectativa de receita mensal da mansão do Lago e quem toca a operação?")

        assert len(res.tools_executed) == 1
        assert res.tools_executed[0]["tool"] == "recall_memory"
        assert "R$10.000" in res.response_text
        assert "Cintia" in res.response_text
        # Ensure no raw JSON leaked in response text
        assert '{"name":' not in res.response_text

    asyncio.run(_run())


def test_assistant_darkfac_mcp_tools():
    async def _run():
        cancelled_list = []

        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/demands/tickets" and request.method == "GET":
                return httpx.Response(
                    200,
                    json=[
                        {"id": "JRV-01", "title": "Demanda 1", "project_id": "jarvis", "status": "planned", "origin": "user"},
                        {"id": "JRV-02", "title": "Tabela config", "project_id": "jarvis", "status": "planned", "origin": "user"},
                        {"id": "JRV-03", "title": "Tabela config", "project_id": "jarvis", "status": "planned", "origin": "user"},
                    ],
                )
            if "/status" in request.url.path and request.method == "PATCH":
                tid = request.url.path.split("/")[4]
                cancelled_list.append(tid)
                return httpx.Response(200, json={"id": tid, "status": "cancelled"})
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        cfg = JarvisConfig()
        df = DarkFactoryClient(config=cfg, http_client=client)
        assistant = JarvisAssistant(config=cfg, darkfac_client=df)

        # Test list_dark_factory_demands
        list_res = await assistant.mcp.execute_tool("list_dark_factory_demands", {"project_id": "jarvis"})
        assert not list_res.is_error
        assert len(list_res.output) == 3

        # Test cancel_dark_factory_demand
        cancel_res = await assistant.mcp.execute_tool("cancel_dark_factory_demand", {"ticket_id": "JRV-03"})
        assert not cancel_res.is_error
        assert cancel_res.output.get("status") == "cancelled"

        # Test deduplicate_dark_factory_demands
        dedup_res = await assistant.mcp.execute_tool("deduplicate_dark_factory_demands", {"project_id": "jarvis"})
        assert not dedup_res.is_error
        assert dedup_res.output.get("cancelled_count") == 1

    asyncio.run(_run())

