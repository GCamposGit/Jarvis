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


def test_darkhub_list_projects():
    async def _run():
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/projects":
                return httpx.Response(200, json=[
                    {"id": "jarvis", "name": "Jarvis (AI Assistant)", "description": "Jarvis"},
                    {"id": "site-ggcampos", "name": "Site Pessoal (ATRIUM)", "description": "Site"},
                ])
            return httpx.Response(404)
        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        projects = await DarkFactoryClient(http_client=client).list_projects()
        assert [project.id for project in projects] == ["jarvis", "site-ggcampos"]
    asyncio.run(_run())


def test_darkhub_create_demand_uses_autonomous_intake_and_canonical_project():
    async def _run():
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/projects":
                return httpx.Response(200, json=[{"id": "site-ggcampos", "name": "Site Pessoal (ATRIUM)", "description": "Site"}])
            if request.url.path == "/api/demands/intake" and request.method == "POST":
                body = request.read().decode("utf-8")
                assert '"project_id":"site-ggcampos"' in body
                assert request.headers.get("Idempotency-Key", "").startswith("jarvis-")
                return httpx.Response(202, json={
                    "demand_id": "dem-42", "demand_version": "1.0",
                    "run_id": "run-42", "initial_job_id": "job-42",
                    "mode": "autonomous", "committed_at": "2026-09-25T12:00:00Z",
                })
            return httpx.Response(404)
        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        df = DarkFactoryClient(http_client=client)
        result = await df.create_demand("Criar novo módulo", "Necessidade de expansão", project_id="ATRIUM")
        assert result["id"] == "dem-42"
        assert result["project_id"] == "site-ggcampos"
        assert result["status"] == "queued"
        assert result["run_id"] == "run-42"
        assert result["initial_job_id"] == "job-42"
    asyncio.run(_run())


def test_darkhub_create_demand_reuses_stable_idempotency_key():
    async def _run():
        keys = []
        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/projects":
                return httpx.Response(200, json=[{"id": "jarvis", "name": "Jarvis", "description": ""}])
            if request.url.path == "/api/demands/intake":
                keys.append(request.headers["Idempotency-Key"])
                return httpx.Response(202, json={
                    "demand_id": "dem-43", "demand_version": "1.0",
                    "run_id": "run-43", "initial_job_id": "job-43",
                    "mode": "autonomous", "committed_at": "2026-09-25T12:00:00Z",
                })
            return httpx.Response(404)
        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        df = DarkFactoryClient(http_client=client)
        first = await df.create_demand("Nova capacidade", "Detalhes", project_id="jarvis")
        replay = await df.create_demand("Nova capacidade", "Detalhes", project_id="jarvis")
        assert first["run_id"] == replay["run_id"] == "run-43"
        assert len(keys) == 2 and keys[0] == keys[1]
    asyncio.run(_run())


def test_darkhub_create_demand_rejects_unknown_project():
    async def _run():
        submitted = False
        def mock_handler(request: httpx.Request):
            nonlocal submitted
            if request.url.path == "/api/projects":
                return httpx.Response(200, json=[{"id": "jarvis", "name": "Jarvis", "description": ""}])
            if request.url.path == "/api/demands/intake":
                submitted = True
            return httpx.Response(404)
        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        result = await DarkFactoryClient(http_client=client).create_demand("Nova capacidade", project_id="nao-existe")
        assert result["status_code"] == 422 and not submitted
    asyncio.run(_run())


def test_darkhub_create_demand_dispatches_same_title_from_legacy_backlog():
    async def _run():
        calls = []

        def mock_handler(request: httpx.Request):
            if request.url.path == "/api/projects":
                return httpx.Response(200, json=[{"id": "jarvis", "name": "Jarvis", "description": ""}])
            if request.url.path == "/api/demands/intake" and request.method == "POST":
                calls.append(request)
                return httpx.Response(202, json={
                    "demand_id": "dem-44", "demand_version": "1.0",
                    "run_id": "run-44", "initial_job_id": "job-44",
                    "mode": "autonomous", "committed_at": "2026-09-25T12:00:00Z",
                })
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        df = DarkFactoryClient(http_client=client)
        result = await df.create_demand(
            title="Tabela para configurar parâmetros",
            problem_statement="Demanda que já existia no backlog sem execução",
            project_id="jarvis",
        )
        assert result["status"] == "queued"
        assert result["run_id"] == "run-44"
        assert len(calls) == 1

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
