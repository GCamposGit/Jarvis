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
        # Check if an active demand with identical title already exists in the project
        clean_title = title.strip().lower()
        try:
            existing_demands = await self.list_demands(project_id=project_id)
            for ed in existing_demands:
                if ed.title.strip().lower() == clean_title and ed.status not in ("cancelled", "completed"):
                    logger.info("Demand '%s' already exists as %s with status %s. Returning existing.", title, ed.id, ed.status)
                    return {
                        "id": ed.id,
                        "title": ed.title,
                        "project_id": ed.project_id,
                        "status": ed.status,
                        "deduplicated": True,
                        "message": f"Demanda já registrada no backlog ({ed.id}). Evitada duplicação.",
                    }
        except Exception as exc:
            logger.debug("Deduplication pre-check failed: %s", exc)

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

    async def update_demand_status(
        self,
        ticket_id: str,
        status: str = "cancelled",
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update status of a demand ticket in the DarkHub backlog (e.g. cancelled, planned, completed)."""
        params = {"status": status}
        if notes:
            params["notes"] = notes

        try:
            resp = await self._client.patch(
                f"{self.base_url}/api/demands/tickets/{ticket_id}/status",
                params=params,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"Failed with HTTP {resp.status_code}: {resp.text}", "status_code": resp.status_code}
        except Exception as exc:
            logger.error("Error updating demand status in DarkHub: %s", exc)
            return {"error": str(exc)}

    async def cancel_demand(self, ticket_id: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """Cancel or deactivate a demand ticket in DarkHub."""
        return await self.update_demand_status(ticket_id, status="cancelled", notes=notes or "Cancelada via Jarvis")

    async def deduplicate_demands(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Find active demands with identical titles and cancel duplicates, keeping the oldest instance."""
        demands = await self.list_demands(project_id=project_id)
        seen: Dict[str, str] = {}
        cancelled_tickets: List[Dict[str, str]] = []
        kept_tickets: List[Dict[str, str]] = []

        for d in demands:
            if d.status in ("cancelled", "completed"):
                continue
            normalized_title = d.title.strip().lower()
            if normalized_title in seen:
                primary_id = seen[normalized_title]
                res = await self.cancel_demand(
                    d.id, notes=f"Duplicata automática de {primary_id} cancelada pelo Jarvis"
                )
                cancelled_tickets.append({"id": d.id, "title": d.title, "result": str(res)})
            else:
                seen[normalized_title] = d.id
                kept_tickets.append({"id": d.id, "title": d.title})

        return {
            "status": "success",
            "kept_count": len(kept_tickets),
            "cancelled_count": len(cancelled_tickets),
            "kept": kept_tickets,
            "cancelled": cancelled_tickets,
        }

    async def get_telemetry_stats(self) -> Dict[str, Any]:
        """Fetch telemetry and resource stats from DarkHub."""
        try:
            resp = await self._client.get(f"{self.base_url}/api/telemetry/stats")
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            logger.debug("Failed to fetch telemetry stats: %s", exc)
        return {}
