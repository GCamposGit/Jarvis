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


def test_darkhub_create_demand_deduplication():
    async def _run():
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/demands/tickets" and request.method == "GET":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": "JRV-02",
                            "title": "Tabela para configurar parâmetros",
                            "project_id": "jarvis",
                            "status": "planned",
                            "origin": "user",
                        }
                    ],
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        df = DarkFactoryClient(http_client=client)

        res = await df.create_demand(
            title="Tabela para configurar parâmetros",
            problem_statement="Tentando criar duplicata",
            project_id="jarvis",
        )
        assert res.get("id") == "JRV-02"
        assert res.get("deduplicated") is True

    asyncio.run(_run())


def test_darkhub_cancel_demand():
    async def _run():
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/demands/tickets/JRV-03/status" and request.method == "PATCH":
                assert request.url.params.get("status") == "cancelled"
                return httpx.Response(
                    200,
                    json={"id": "JRV-03", "status": "cancelled"},
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        df = DarkFactoryClient(http_client=client)

        res = await df.cancel_demand("JRV-03", notes="Removendo duplicata")
        assert res.get("id") == "JRV-03"
        assert res.get("status") == "cancelled"

    asyncio.run(_run())


def test_darkhub_deduplicate_demands():
    async def _run():
        cancelled = []

        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/demands/tickets" and request.method == "GET":
                return httpx.Response(
                    200,
                    json=[
                        {"id": "JRV-02", "title": "Tabela para configurar parâmetros", "status": "planned"},
                        {"id": "JRV-03", "title": "Tabela para configurar parâmetros", "status": "planned"},
                        {"id": "JRV-04", "title": "Tabela para configurar parâmetros", "status": "planned"},
                        {"id": "JRV-05", "title": "Outra demanda", "status": "planned"},
                    ],
                )
            if "/status" in request.url.path and request.method == "PATCH":
                ticket_id = request.url.path.split("/")[4]
                cancelled.append(ticket_id)
                return httpx.Response(200, json={"id": ticket_id, "status": "cancelled"})
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        df = DarkFactoryClient(http_client=client)

        result = await df.deduplicate_demands(project_id="jarvis")
        assert result["status"] == "success"
        assert result["kept_count"] == 2
        assert result["cancelled_count"] == 2
        assert cancelled == ["JRV-03", "JRV-04"]

    asyncio.run(_run())
