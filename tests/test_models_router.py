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

