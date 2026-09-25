"""Deterministic tests for MeetingRelator Production Bridge and Multimodal Ingestion (Milestone 3)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List
import pytest
import httpx
from fastapi.testclient import TestClient

from jarvis.core.bizops import AutonomyLevel, BizOpsEngine
from jarvis.core.config import JarvisConfig
from jarvis.core.darkfac import DarkFactoryClient
from jarvis.core.meeting_relator import (
    MeetingActionItem,
    MeetingChapter,
    MeetingDecision,
    MeetingDetail,
    MeetingParticipant,
    MeetingRelatorBridge,
    MeetingSummary,
    MeetingSyncResult,
)
from jarvis.core.memory import EpisodicMemoryEngine
from jarvis.core.assistant import JarvisAssistant
from jarvis.web.app import create_app


MEETING_RELATOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    date TEXT NOT NULL,
    meeting_name TEXT NOT NULL,
    meeting_type TEXT,
    duration_seconds REAL,
    speaker_count INTEGER,
    oneliner TEXT,
    executive_summary TEXT,
    detailed_summary TEXT,
    full_transcript TEXT,
    pdf_path TEXT,
    mp3_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    speaker_id TEXT NOT NULL,
    speaker_label TEXT,
    identified_name TEXT,
    email TEXT,
    role_inferred TEXT,
    talk_time_pct REAL,
    interventions INTEGER,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id)
);

CREATE TABLE IF NOT EXISTS action_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    description TEXT NOT NULL,
    owner TEXT,
    timestamp TEXT,
    priority TEXT,
    evidence TEXT,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    description TEXT NOT NULL,
    timestamp TEXT,
    participants TEXT,
    impact TEXT,
    evidence TEXT,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id)
);

CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    chapter_index INTEGER,
    title TEXT NOT NULL,
    time_start TEXT,
    time_end TEXT,
    summary TEXT,
    detailed_notes TEXT,
    key_points TEXT,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id)
);
"""


@pytest.fixture
def mock_meeting_relator_db(tmp_path: Path) -> Path:
    """Creates a temporary SQLite database populated with realistic MeetingRelator data."""
    db_file = tmp_path / "meeting_database.db"
    with sqlite3.connect(db_file) as conn:
        conn.executescript(MEETING_RELATOR_SCHEMA)
        cur = conn.cursor()

        # Insert Meeting 1
        cur.execute(
            """
            INSERT INTO meetings (
                id, session_id, date, meeting_name, meeting_type, duration_seconds,
                speaker_count, oneliner, executive_summary, detailed_summary,
                full_transcript, pdf_path, mp3_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "2026-09-18_10-00-00_Alinhamento_Arquitetura",
                "2026-09-18",
                "Alinhamento de Arquitetura Jarvis v2",
                "technical_review",
                1800.0,
                2,
                "Definida a estratégia de migração para arquitetura orientada a eventos e isolamento Dark Factory.",
                "Reunião de alinhamento com foco no isolamento do MeetingRelator e despacho de demandas ao DarkHub.",
                "Discussão aprofundada sobre os componentes de Segundo Cérebro, BizOps e pontes com sistemas em produção.",
                "Gui: Vamos integrar o MeetingRelator sem tocar nos arquivos de produção.\\nArquiteto: Perfeito, usamos PRAGMA query_only e gravamos syncs no Jarvis.",
                "/path/to/relatorio_1.pdf",
                "/path/to/audio_1.mp3",
            ),
        )

        # Participants Meeting 1
        cur.execute(
            """
            INSERT INTO participants (meeting_id, speaker_id, speaker_label, identified_name, role_inferred, talk_time_pct, interventions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "spk_0", "Canal Local", "Guilherme", "Tech Lead", 60.0, 15),
        )
        cur.execute(
            """
            INSERT INTO participants (meeting_id, speaker_id, speaker_label, identified_name, role_inferred, talk_time_pct, interventions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "spk_1", "Canal Remoto", "Arquiteto Cloud", "Staff Architect", 40.0, 10),
        )

        # Action Items Meeting 1
        cur.execute(
            """
            INSERT INTO action_items (meeting_id, description, owner, timestamp, priority, evidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "Implementar adaptador read-only com PRAGMA query_only no Jarvis", "Guilherme", "00:12:30", "alta", "Gui: Eu implemento o adaptador hoje."),
        )
        cur.execute(
            """
            INSERT INTO action_items (meeting_id, description, owner, timestamp, priority, evidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "Configurar tarefa periódica de sincronização no BizOps", "Guilherme", "00:25:00", "media", "Arquiteto: Precisamos de um cron para sync."),
        )

        # Decisions Meeting 1
        cur.execute(
            """
            INSERT INTO decisions (meeting_id, description, timestamp, participants, impact, evidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "Decidido manter C:\\\\dev\\\\MeetingRelator 100% isolado e intacto", "00:05:15", "Guilherme, Arquiteto", "alto", "Gui: Zero modificações no repo de gravação."),
        )

        # Chapters Meeting 1
        cur.execute(
            """
            INSERT INTO chapters (meeting_id, chapter_index, title, time_start, time_end, summary, detailed_notes, key_points)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                1,
                "Introdução e Restrições de Produção",
                "00:00:00",
                "00:10:00",
                "Revisão dos requisitos de segurança do MeetingRelator.",
                "Não podemos introduzir regressão ou concorrência de escrita.",
                json.dumps(["Zero lock", "SQLite read-only"]),
            ),
        )

        # Insert Meeting 2
        cur.execute(
            """
            INSERT INTO meetings (
                id, session_id, date, meeting_name, meeting_type, duration_seconds,
                speaker_count, oneliner, executive_summary, detailed_summary,
                full_transcript, pdf_path, mp3_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                2,
                "2026-09-17_15-00-00_Planejamento_Sprint",
                "2026-09-17",
                "Planejamento Semanal Sprint 42",
                "planning",
                2400.0,
                3,
                "Definição de metas da sprint e priorização de demandas da Dark Factory.",
                "Revisão de tickets e balanceamento de carga entre agentes autônomos.",
                "Priorizado o fechamento do Marco 3 de produtividade com foco em Segundo Cérebro.",
                "Equipe alinhando prazos e dependências de entrega.",
                "/path/to/relatorio_2.pdf",
                "/path/to/audio_2.mp3",
            ),
        )

        conn.commit()

    return db_file


def test_meeting_relator_bridge_read_and_search(tmp_path: Path, mock_meeting_relator_db: Path):
    """Verify that MeetingRelatorBridge queries meetings, details and search with zero write."""
    mem_db = tmp_path / "jarvis_mem.db"
    mem_engine = EpisodicMemoryEngine(db_path=mem_db)
    bridge = MeetingRelatorBridge(
        memory_engine=mem_engine,
        db_override_path=mock_meeting_relator_db,
    )

    # 1. List meetings
    meetings = bridge.list_meetings(limit=10)
    assert len(meetings) == 2
    assert meetings[0].id == 1
    assert "Alinhamento de Arquitetura Jarvis v2" in meetings[0].meeting_name
    assert meetings[0].speaker_count == 2
    assert meetings[1].id == 2
    assert "Planejamento Semanal Sprint 42" in meetings[1].meeting_name

    # 2. Get meeting detail
    detail = bridge.get_meeting(1)
    assert detail is not None
    assert isinstance(detail, MeetingDetail)
    assert detail.summary.meeting_name == "Alinhamento de Arquitetura Jarvis v2"
    assert len(detail.participants) == 2
    assert detail.participants[0].identified_name == "Guilherme"
    assert len(detail.action_items) == 2
    assert "PRAGMA query_only" in detail.action_items[0].description
    assert len(detail.decisions) == 1
    assert "100% isolado" in detail.decisions[0].description
    assert len(detail.chapters) == 1
    assert detail.chapters[0].title == "Introdução e Restrições de Produção"

    # Non-existent meeting
    assert bridge.get_meeting(999) is None

    # 3. Search meetings
    search_res = bridge.search_meetings("arquitetura")
    assert len(search_res) >= 1
    assert search_res[0].id == 1

    search_empty = bridge.search_meetings("inexistente_termo_xyz")
    assert len(search_empty) == 0


def test_meeting_relator_sync_to_second_brain(tmp_path: Path, mock_meeting_relator_db: Path):
    """Verify synchronization into EpisodicMemoryEngine (facts, decisions, episodes) with idempotency."""
    mem_db = tmp_path / "jarvis_mem.db"
    mem_engine = EpisodicMemoryEngine(db_path=mem_db)
    bridge = MeetingRelatorBridge(
        memory_engine=mem_engine,
        db_override_path=mock_meeting_relator_db,
    )

    # First sync: should sync 2 meetings
    res1 = bridge.sync_to_second_brain(limit=10)
    assert res1.synced_meetings_count == 2
    assert res1.facts_created >= 2
    assert res1.decisions_created >= 1
    assert res1.episodes_created >= 2

    # Verify facts were created in memory
    facts = mem_engine.recall_facts(category="meeting_relator")
    assert len(facts) >= 2
    assert any("meeting_1" in f.key for f in facts)

    # Verify decisions were recorded in memory
    decisions = mem_engine.query_decisions()
    assert len(decisions) >= 1
    assert any("100% isolado" in d.title or "100% isolado" in d.decision for d in decisions)

    # Second sync: should detect meetings as already synced (idempotent)
    res2 = bridge.sync_to_second_brain(limit=10)
    assert res2.synced_meetings_count == 0
    assert "Já sincronizadas" in res2.message or res2.synced_meetings_count == 0


@pytest.mark.anyio
async def test_meeting_relator_dispatch_action_items_and_hitl(tmp_path: Path, mock_meeting_relator_db: Path):
    """Verify action items conversion to Level 2 HITL proposals and approval into DarkHub."""
    mem_db = tmp_path / "jarvis_mem.db"
    mem_engine = EpisodicMemoryEngine(db_path=mem_db)

    def mock_handler(request: httpx.Request):
        if request.url.path == "/api/projects":
            return httpx.Response(200, json=[
                {"id": "jarvis", "name": "Jarvis", "description": ""},
                {"id": "darkfac", "name": "DarkFac", "description": ""},
            ])
        if request.url.path == "/api/demands/intake" and request.method == "POST":
            return httpx.Response(202, json={
                "demand_id": "DEM-MEETING-101",
                "run_id": "run-meeting-101",
                "initial_job_id": "job-meeting-101",
                "mode": "autonomous",
            })
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    darkfac = DarkFactoryClient(http_client=client)
    bizops = BizOpsEngine(db_path=mem_db, memory_engine=mem_engine, darkfac_client=darkfac)

    bridge = MeetingRelatorBridge(
        memory_engine=mem_engine,
        darkfac_client=darkfac,
        bizops_engine=bizops,
        db_override_path=mock_meeting_relator_db,
    )

    # Dispatch action items from meeting 1
    pending_actions = bridge.dispatch_action_items_to_darkfac(meeting_id=1, project_id="darkfac")
    assert len(pending_actions) == 2
    for p in pending_actions:
        assert p.status == "pending_approval"
        assert p.autonomy_level_required == AutonomyLevel.LEVEL_2_HITL
        assert p.action_type == "create_dark_factory_demand"
        assert "Alinhamento de Arquitetura" in p.title

    # Verify pending actions registered in BizOpsEngine
    stored_pending = bizops.list_pending_actions(status="pending_approval")
    assert len(stored_pending) == 2

    # Operator approves first action item
    first_action = pending_actions[0]
    appr_res = await bizops.approve_action(first_action.id, operator="LeadArchitect")
    assert appr_res.success is True
    assert "aprovada e executada" in appr_res.message
    assert appr_res.data.get("demand_id") == "DEM-MEETING-101"

    # Action is now executed
    resolved = bizops.get_pending_action(first_action.id)
    assert resolved.status == "executed"
    assert resolved.resolved_by == "LeadArchitect"


@pytest.mark.anyio
async def test_assistant_meeting_mcp_tools(tmp_path: Path, mock_meeting_relator_db: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify MCP tools exposed to the Assistant for MeetingRelator integration."""
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: None)
    mem_db = tmp_path / "jarvis_mem.db"
    mem_engine = EpisodicMemoryEngine(db_path=mem_db)
    bizops = BizOpsEngine(db_path=mem_db, memory_engine=mem_engine)
    bridge = MeetingRelatorBridge(
        memory_engine=mem_engine,
        bizops_engine=bizops,
        db_override_path=mock_meeting_relator_db,
    )
    assistant = JarvisAssistant(
        memory_engine=mem_engine,
        bizops_engine=bizops,
        meeting_bridge=bridge,
    )

    # 1. list_recent_meetings tool
    res_list = await assistant.mcp.execute_tool("list_recent_meetings", {"limit": 5})
    assert res_list.is_error is False
    assert len(res_list.output) == 2
    assert res_list.output[0]["id"] == 1

    # 2. get_meeting_details tool
    res_detail = await assistant.mcp.execute_tool("get_meeting_details", {"meeting_id": 1})
    assert res_detail.is_error is False
    assert res_detail.output["summary"]["meeting_name"] == "Alinhamento de Arquitetura Jarvis v2"
    assert len(res_detail.output["action_items"]) == 2

    # 3. search_meetings tool
    res_search = await assistant.mcp.execute_tool("search_meetings", {"query": "Sprint 42"})
    assert res_search.is_error is False
    assert len(res_search.output) == 1
    assert res_search.output[0]["id"] == 2

    # 4. sync_meetings_to_second_brain tool
    res_sync = await assistant.mcp.execute_tool("sync_meetings_to_second_brain", {"limit": 5})
    assert res_sync.is_error is False
    assert res_sync.output["synced_meetings_count"] == 2

    # 5. dispatch_meeting_demands tool
    res_disp = await assistant.mcp.execute_tool("dispatch_meeting_demands", {"meeting_id": 1, "project_id": "darkfac"})
    assert res_disp.is_error is False
    assert len(res_disp.output) == 2

    # 6. launch_meeting_recorder tool
    res_launch = await assistant.mcp.execute_tool("launch_meeting_recorder", {})
    assert res_launch.is_error is False
    assert "status" in res_launch.output


@pytest.mark.anyio
async def test_bizops_meeting_sync_task(tmp_path: Path, mock_meeting_relator_db: Path):
    """Verify task_meeting_sync execution in BizOpsEngine."""
    mem_db = tmp_path / "jarvis_mem.db"
    mem_engine = EpisodicMemoryEngine(db_path=mem_db)
    cfg = JarvisConfig(memory_db_path=mem_db, meeting_relator_output_folder=mock_meeting_relator_db.parent)
    bizops = BizOpsEngine(config=cfg, db_path=mem_db, memory_engine=mem_engine)

    # Check task registered in default tasks
    task = bizops.get_task("task_meeting_sync")
    assert task is not None
    assert task.task_type == "meeting_sync"

    # Trigger the task
    res = await bizops.trigger_task("task_meeting_sync")
    assert res.success is True
    assert "Sincronização concluída" in res.message or "Já sincronizadas" in res.message


def test_meeting_relator_rest_api_endpoints(tmp_path: Path, mock_meeting_relator_db: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify REST endpoints for meetings in the FastAPI web gateway."""
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: None)
    mem_db = tmp_path / "jarvis_mem.db"
    cfg = JarvisConfig(
        memory_db_path=mem_db,
        meeting_relator_output_folder=mock_meeting_relator_db.parent,
    )
    app = create_app(config=cfg)

    # Override bridge with mock db path
    bridge = MeetingRelatorBridge(
        config=cfg,
        memory_engine=app.state.memory,
        bizops_engine=app.state.bizops,
        db_override_path=mock_meeting_relator_db,
    )
    app.state.meetings = bridge
    app.state.assistant.meetings = bridge

    client = TestClient(app)

    # GET /api/meetings
    r_list = client.get("/api/meetings")
    assert r_list.status_code == 200
    meetings = r_list.json()
    assert len(meetings) == 2
    assert meetings[0]["id"] == 1

    # GET /api/meetings with search query
    r_search = client.get("/api/meetings?query=Arquitetura")
    assert r_search.status_code == 200
    search_items = r_search.json()
    assert len(search_items) >= 1
    assert search_items[0]["id"] == 1

    # GET /api/meetings/{id}
    r_detail = client.get("/api/meetings/1")
    assert r_detail.status_code == 200
    detail = r_detail.json()
    assert detail["summary"]["meeting_name"] == "Alinhamento de Arquitetura Jarvis v2"
    assert len(detail["participants"]) == 2

    # GET /api/meetings/999 (not found)
    r_404 = client.get("/api/meetings/999")
    assert r_404.status_code == 404

    # POST /api/meetings/sync
    r_sync = client.post("/api/meetings/sync?limit=5")
    assert r_sync.status_code == 200
    sync_data = r_sync.json()
    assert sync_data["synced_meetings_count"] == 2

    # POST /api/meetings/1/dispatch-demands
    r_disp = client.post("/api/meetings/1/dispatch-demands?project_id=darkfac")
    assert r_disp.status_code == 200
    dispatched = r_disp.json()
    assert len(dispatched) == 2

    # POST /api/meetings/launch-recorder
    r_launch = client.post("/api/meetings/launch-recorder")
    assert r_launch.status_code == 200
    assert "status" in r_launch.json()
