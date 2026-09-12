"""Deterministic tests for Jarvis DarkFactory Client."""

from __future__ import annotations

import asyncio
import httpx

from jarvis.core.config import JarvisConfig
from jarvis.core.darkfac import DarkFactoryClient


def test_darkhub_status_online():
    async def _run():
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/cloud/status":
                return httpx.Response(
                    200,
                    json={
                        "service_name": "darkhub",
                        "status": "active",
                        "uptime_seconds": 3600,
                    },
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        df = DarkFactoryClient(http_client=client)

        status = await df.get_status()
        assert status.online is True
        assert status.status_code == 200
        assert status.cloud_status.get("status") == "active"

    asyncio.run(_run())


def test_darkhub_list_demands():
    async def _run():
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/demands/tickets":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": "USR-01",
                            "title": "Integração de busca",
                            "project_id": "jarvis",
                            "status": "planned",
                            "origin": "user",
                        },
                        {
                            "id": "USR-02",
                            "title": "Configuração de mic",
                            "project_id": "jarvis",
                            "status": "completed",
                            "origin": "user",
                        },
                    ],
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        df = DarkFactoryClient(http_client=client)

        demands = await df.list_demands(project_id="jarvis")
        assert len(demands) == 2
        assert demands[0].id == "USR-01"
        assert demands[0].title == "Integração de busca"

    asyncio.run(_run())


def test_darkhub_create_demand():
    async def _run():
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/demands/next-id":
                return httpx.Response(200, json={"next_id": "USR-42"})
            if request.url.path == "/api/demands/tickets" and request.method == "POST":
                return httpx.Response(
                    201,
                    json={
                        "id": "USR-42",
                        "title": "Criar novo módulo",
                        "status": "planned",
                    },
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        df = DarkFactoryClient(http_client=client)

        res = await df.create_demand(
            title="Criar novo módulo",
            problem_statement="Necessidade de expansão",
            project_id="jarvis",
        )
        assert res.get("id") == "USR-42"
        assert res.get("title") == "Criar novo módulo"

    asyncio.run(_run())
