"""Deterministic tests for Jarvis Assistant conversational engine."""

from __future__ import annotations

import asyncio
import httpx

from jarvis.core.assistant import JarvisAssistant
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
