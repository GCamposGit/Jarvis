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


def test_darkfac_cancel_and_deduplicate_endpoints():
    async def _run():
        from unittest.mock import AsyncMock

        cfg = JarvisConfig()
        app = create_app(cfg)

        app.state.darkfac.cancel_demand = AsyncMock(return_value={"id": "JRV-03", "status": "cancelled"})
        app.state.darkfac.deduplicate_demands = AsyncMock(return_value={"status": "success", "cancelled_count": 1})

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            cancel_resp = await ac.post("/api/darkfac/demands/JRV-03/cancel")
            assert cancel_resp.status_code == 200
            assert cancel_resp.json().get("status") == "cancelled"

            dedup_resp = await ac.post("/api/darkfac/demands/deduplicate?project_id=jarvis")
            assert dedup_resp.status_code == 200
            assert dedup_resp.json().get("cancelled_count") == 1

    asyncio.run(_run())


def test_darkfac_projects_endpoint_and_create_error_status():
    async def _run():
        from unittest.mock import AsyncMock
        app = create_app(JarvisConfig())
        app.state.darkfac.list_projects = AsyncMock(
            return_value=[{"id": "jarvis", "name": "Jarvis", "description": ""}]
        )
        app.state.darkfac.create_demand = AsyncMock(
            return_value={"error": "ControlStore indisponível", "status_code": 503}
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            projects = await ac.get("/api/darkfac/projects")
            assert projects.status_code == 200 and projects.json()[0]["id"] == "jarvis"
            response = await ac.post("/api/darkfac/demands", json={"title": "Demanda de teste", "project_id": "jarvis"})
            assert response.status_code == 503
            assert response.json()["detail"] == "ControlStore indisponível"
    asyncio.run(_run())
