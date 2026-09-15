"""Asynchronous REST Client for DarkHub and Dark Factory Ecosystem."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.config import JarvisConfig, get_config

logger = logging.getLogger("jarvis.core.darkfac")


class DarkHubStatus(BaseModel):
    """Normalized status of the DarkHub cloud connection."""

    model_config = ConfigDict(frozen=True)

    online: bool
    url: str
    status_code: Optional[int] = None
    cloud_status: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DarkDemandSummary(BaseModel):
    """Summary of a ticket/demand in the Dark Factory backlog."""

    id: str
    title: str
    project_id: str = "darkfac"
    status: str = "planned"
    origin: str = "user"


class DarkFactoryClient:
    """Async client communicating with the DarkHub REST API."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.config = config or get_config()
        self.base_url = self.config.darkhub_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(timeout=15.0)

    async def get_status(self) -> DarkHubStatus:
        """Probe DarkHub /api/cloud/status and root availability."""
        try:
            resp = await self._client.get(f"{self.base_url}/api/cloud/status")
            if resp.status_code == 200:
                return DarkHubStatus(
                    online=True,
                    url=self.base_url,
                    status_code=resp.status_code,
                    cloud_status=resp.json(),
                )
            return DarkHubStatus(
                online=False,
                url=self.base_url,
                status_code=resp.status_code,
                error=f"Unexpected status HTTP {resp.status_code}",
            )
        except Exception as exc:
            logger.debug("DarkHub status check failed: %s", exc)
            return DarkHubStatus(
                online=False,
                url=self.base_url,
                error=str(exc),
            )

    async def list_demands(self, project_id: Optional[str] = None) -> List[DarkDemandSummary]:
        """Fetch backlog tickets from DarkHub."""
        params = {}
        if project_id:
            params["project_id"] = project_id

        try:
            resp = await self._client.get(f"{self.base_url}/api/demands/tickets", params=params)
            if resp.status_code == 200:
                items = resp.json()
                results: List[DarkDemandSummary] = []
                for item in items:
                    results.append(
                        DarkDemandSummary(
                            id=item.get("id", ""),
                            title=item.get("title", ""),
                            project_id=item.get("project_id", "darkfac"),
                            status=item.get("status", "planned"),
                            origin=item.get("origin", "user"),
                        )
                    )
                return results
        except Exception as exc:
            logger.warning("Failed to fetch demands from DarkHub: %s", exc)
        return []

    async def create_demand(
        self,
        title: str,
        problem_statement: str = "",
        core_journey: str = "",
        project_id: str = "darkfac",
        acceptance_criteria: Optional[List[str]] = None,
        non_goals: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new formal demand/ticket in the Dark Factory backlog."""
        # 1. Fetch next ticket ID
        next_id = f"USR-{int(httpx._utils.get_environment_proxies().get('dummy', 1))}"
        try:
            id_resp = await self._client.get(
                f"{self.base_url}/api/demands/next-id",
                params={"project_id": project_id},
            )
            if id_resp.status_code == 200:
                next_id = id_resp.json().get("next_id", next_id)
        except Exception:
            pass

        journey = [core_journey] if isinstance(core_journey, str) and core_journey else (core_journey or ["Submitted through Jarvis Assistant"])
        ticket_payload = {
            "id": next_id,
            "project_id": project_id,
            "title": title,
            "origin": "user",
            "status": "planned",
            "item_type": "feature",
            "lifecycle_stage": "execution",
            "horizon": "now",
            "problem_statement": problem_statement or title,
            "core_journey": journey,
            "acceptance_criteria": acceptance_criteria or ["Harness passes with deterministic verification"],
            "non_goals": non_goals or [],
        }

        try:
            resp = await self._client.post(
                f"{self.base_url}/api/demands/tickets",
                json=ticket_payload,
            )
            if resp.status_code in (200, 201):
                return resp.json()
            return {"error": f"Failed with HTTP {resp.status_code}: {resp.text}", "payload": ticket_payload}
        except Exception as exc:
            logger.error("Error creating demand in DarkHub: %s", exc)
            return {"error": str(exc), "payload": ticket_payload}

    async def get_telemetry_stats(self) -> Dict[str, Any]:
        """Fetch telemetry and resource stats from DarkHub."""
        try:
            resp = await self._client.get(f"{self.base_url}/api/telemetry/stats")
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            logger.debug("Failed to fetch telemetry stats: %s", exc)
        return {}
