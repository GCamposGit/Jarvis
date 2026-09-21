"""Deterministic tests for Jarvis Unified Model Router."""

from __future__ import annotations

import asyncio
import httpx

from jarvis.core.config import JarvisConfig
from jarvis.core.models import ChatMessage, UnifiedModelRouter


def test_ollama_generation_success():
    async def _run():
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/chat":
                return httpx.Response(
                    200,
                    json={
                        "message": {"role": "assistant", "content": "Olá! Sou o modelo local do Jarvis."},
                        "prompt_eval_count": 25,
                        "eval_count": 15,
                    },
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        cfg = JarvisConfig(default_provider="ollama", default_local_model="qwen2.5-coder:latest")
        router = UnifiedModelRouter(config=cfg, http_client=client)

        resp = await router.generate([ChatMessage(role="user", content="Oi")])
        assert resp.provider == "ollama"
        assert resp.model == "qwen2.5-coder:latest"
        assert "Olá! Sou o modelo local" in resp.text
        assert resp.cost_usd == 0.0

    asyncio.run(_run())


def test_openrouter_generation_success():
    async def _run():
        def mock_handler(request: httpx.Request):
            if "openrouter.ai" in request.url.host:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"role": "assistant", "content": "Resposta do Claude 3.7 Sonnet."}}
                        ],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 20},
                    },
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        cfg = JarvisConfig(openrouter_api_key="sk-or-v1-mock-key-123456")
        router = UnifiedModelRouter(config=cfg, http_client=client)

        resp = await router.generate(
            [ChatMessage(role="user", content="Escreva um plano")],
            model="anthropic/claude-sonnet-4.6",
            provider="openrouter",
        )
        assert resp.provider == "openrouter"
        assert resp.model == "anthropic/claude-sonnet-4.6"
        assert "Claude" in resp.text

    asyncio.run(_run())


def test_list_local_models():
    async def _run():
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/tags":
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {"name": "qwen2.5-coder:latest"},
                            {"name": "llama3.2:latest"},
                        ]
                    },
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        router = UnifiedModelRouter(http_client=client)

        models = await router.list_local_models()
        assert len(models) == 2
        assert "qwen2.5-coder:latest" in models

    asyncio.run(_run())


def test_google_gemini_generation_success():
    async def _run():
        def mock_handler(request: httpx.Request):
            if "generativelanguage.googleapis.com" in request.url.host:
                return httpx.Response(
                    200,
                    json={
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": "Resposta direta do Gemini Flash $0 marginal."}],
                                    "role": "model",
                                }
                            }
                        ],
                        "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 15},
                    },
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        cfg = JarvisConfig(gemini_api_key="AIzaSyMockKeyForTesting123456")
        router = UnifiedModelRouter(config=cfg, http_client=client)

        resp = await router.generate(
            [ChatMessage(role="user", content="Explique o conceito")],
            model="gemini-2.5-flash",
            provider="google",
        )
        assert resp.provider == "google"
        assert resp.model == "gemini-2.5-flash"
        assert "Gemini Flash" in resp.text
        assert resp.cost_usd == 0.0

    asyncio.run(_run())


def test_extract_tool_calls_raw_concatenated_json():
    from jarvis.core.models import extract_tool_calls_from_text

    raw_text = (
        '{"name": "search_second_brain", "arguments": {"query": "mansão do lago receita mensal"}} '
        '{"name": "recall_memory", "arguments": {"query": "operador mansão do lago"}} '
        '{"name": "get_telemetry_summary", "arguments": {}}'
    )
    tools, clean = extract_tool_calls_from_text(raw_text)
    assert tools is not None
    assert len(tools) == 3
    assert tools[0]["function"]["name"] == "search_second_brain"
    assert tools[1]["function"]["name"] == "recall_memory"
    assert tools[2]["function"]["name"] == "get_telemetry_summary"
    assert clean == ""


def test_extract_tool_calls_tags_and_markdown():
    from jarvis.core.models import extract_tool_calls_from_text

    tag_text = '<tool_call>{"name": "recall_memory", "arguments": {"query": "teste"}}</tool_call>'
    tools1, clean1 = extract_tool_calls_from_text(tag_text)
    assert tools1 is not None
    assert len(tools1) == 1
    assert tools1[0]["function"]["name"] == "recall_memory"

    md_text = '```json\n[{"name": "recall_memory", "arguments": {"query": "teste"}}]\n```'
    tools2, clean2 = extract_tool_calls_from_text(md_text)
    assert tools2 is not None
    assert len(tools2) == 1
    assert tools2[0]["function"]["name"] == "recall_memory"


def test_ollama_raw_json_tool_calls_auto_extracted():
    async def _run():
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/chat":
                # Emulate Qwen returning raw JSON in message content and tool_calls as None
                raw_json = (
                    '{"name": "search_second_brain", "arguments": {"query": "mansão do lago"}} '
                    '{"name": "recall_memory", "arguments": {"query": "operador"}}'
                )
                return httpx.Response(
                    200,
                    json={
                        "message": {"role": "assistant", "content": raw_json},
                        "prompt_eval_count": 50,
                        "eval_count": 30,
                    },
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        cfg = JarvisConfig(default_provider="ollama", default_local_model="qwen-code-deep:latest")
        router = UnifiedModelRouter(config=cfg, http_client=client)

        tools_schema = [
            {"type": "function", "function": {"name": "search_second_brain"}},
            {"type": "function", "function": {"name": "recall_memory"}},
        ]
        resp = await router.generate(
            [ChatMessage(role="user", content="Mansão do Lago")],
            tools=tools_schema,
        )
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0]["function"]["name"] == "search_second_brain"
        assert resp.tool_calls[1]["function"]["name"] == "recall_memory"
        # Raw JSON was consumed, so text should be empty
        assert resp.text == ""

    asyncio.run(_run())

