"""Deterministic integration tests for Jarvis FastAPI Web API."""

from __future__ import annotations

import asyncio
import httpx

from jarvis.core.config import JarvisConfig
from jarvis.web.app import create_app


def test_health_endpoint():
    async def _run():
        cfg = JarvisConfig()
        app = create_app(cfg)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["version"] == "0.1.0"
            assert data["tools_count"] >= 3

    asyncio.run(_run())


def test_mcp_tools_and_call_endpoints():
    async def _run():
        cfg = JarvisConfig()
        app = create_app(cfg)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            # List tools
            tools_resp = await ac.get("/api/mcp/tools")
            assert tools_resp.status_code == 200
            tools = tools_resp.json()
            assert len(tools) >= 3

            # Call search tool
            call_resp = await ac.post(
                "/api/mcp/call",
                json={
                    "tool_name": "search_second_brain",
                    "arguments": {"query": "python"},
                },
            )
            assert call_resp.status_code == 200
            result = call_resp.json()
            assert not result["is_error"]
            assert len(result["output"]) > 0

    asyncio.run(_run())


def test_chat_endpoint():
    async def _run():
        cfg = JarvisConfig()
        app = create_app(cfg)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/chat",
                json={
                    "message": "Qual é o status da Dark Factory?",
                    "history": [],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "response_text" in data
            assert len(data["tools_executed"]) > 0

    asyncio.run(_run())


def test_static_index_serving():
    async def _run():
        cfg = JarvisConfig()
        app = create_app(cfg)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/")
            assert resp.status_code == 200
            assert "Jarvis" in resp.text
            assert "Assistente Pessoal" in resp.text

    asyncio.run(_run())


def test_audio_transcribe_endpoint():
    async def _run():
        from unittest.mock import MagicMock
        from jarvis.core.audio import TranscriptionResult

        cfg = JarvisConfig()
        app = create_app(cfg)

        mock_result = TranscriptionResult(
            text="Hello world test",
            language="en",
            engine_used="cloud_groq_whisper",
            fallback_triggered=True,
            confidence_score=0.99,
        )
        app.state.audio_engine.transcribe = MagicMock(return_value=mock_result)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            files = {"file": ("test.wav", b"RIFF1234WAVEfmt ", "audio/wav")}
            resp = await ac.post("/api/audio/transcribe?language=en", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["text"] == "Hello world test"
            assert data["language"] == "en"
            assert data["engine_used"] == "cloud_groq_whisper"

    asyncio.run(_run())

