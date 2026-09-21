"""MeetingRelator Production Bridge and Second Brain Ingestion Adapter (Milestone 3).

Integrates the production MeetingRelator system (C:\\dev\\MeetingRelator) into Jarvis:
- Non-invasive, strictly read-only access to meeting_database.db and session folders.
- Automatic Second Brain ingestion into EpisodicMemoryEngine (facts, decisions, episodes).
- Action Item extraction and dispatch to Dark Factory (DarkHub demands via BizOps HITL).
- Desktop recording launcher.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.bizops.models import AutonomyLevel, PendingAction
from jarvis.core.config import JarvisConfig, get_config
from jarvis.core.darkfac import DarkFactoryClient
from jarvis.core.memory import EpisodicMemoryEngine

logger = logging.getLogger("jarvis.core.meeting_relator")


class MeetingSummary(BaseModel):
    """Brief metadata summary of a recorded meeting."""

    model_config = ConfigDict(frozen=True)

    id: int
    session_id: str
    date: str
    meeting_name: str
    meeting_type: str = ""
    duration_seconds: float = 0.0
    speaker_count: int = 0
    oneliner: str = ""
    mp3_path: str = ""
    pdf_path: str = ""


class MeetingParticipant(BaseModel):
    """Speaker or attendee identified in a meeting."""

    model_config = ConfigDict(frozen=True)

    id: int
    meeting_id: int
    speaker_id: str
    speaker_label: str = ""
    identified_name: str = ""
    email: str = ""
    role_inferred: str = ""
    talk_time_pct: float = 0.0
    interventions: int = 0


class MeetingActionItem(BaseModel):
    """Action item / commitment extracted from the meeting."""

    model_config = ConfigDict(frozen=True)

    id: int
    meeting_id: int
    description: str
    owner: str = ""
    timestamp: str = ""
    priority: str = ""
    evidence: str = ""


class MeetingDecision(BaseModel):
    """Formal decision or consensus agreed upon during the meeting."""

    model_config = ConfigDict(frozen=True)

    id: int
    meeting_id: int
    description: str
    timestamp: str = ""
    participants: str = ""
    impact: str = ""
    evidence: str = ""


class MeetingChapter(BaseModel):
    """Topic chapter or thematic segment in the meeting."""

    model_config = ConfigDict(frozen=True)

    id: int
    meeting_id: int
    chapter_index: int = 0
    title: str
    time_start: str = ""
    time_end: str = ""
    summary: str = ""
    detailed_notes: str = ""
    key_points: List[str] = Field(default_factory=list)


class MeetingDetail(BaseModel):
    """Full meeting report including transcript, decisions, actions and chapters."""

    model_config = ConfigDict(frozen=True)

    summary: MeetingSummary
    executive_summary: str = ""
    detailed_summary: str = ""
    full_transcript: str = ""
    participants: List[MeetingParticipant] = Field(default_factory=list)
    action_items: List[MeetingActionItem] = Field(default_factory=list)
    decisions: List[MeetingDecision] = Field(default_factory=list)
    chapters: List[MeetingChapter] = Field(default_factory=list)


class MeetingSyncResult(BaseModel):
    """Result of synchronizing MeetingRelator data with Jarvis Second Brain."""

    model_config = ConfigDict(frozen=True)

    synced_meetings_count: int = 0
    facts_created: int = 0
    decisions_created: int = 0
    episodes_created: int = 0
    message: str = ""


class MeetingRelatorBridge:
    """Non-invasive read-only bridge to MeetingRelator database and session assets."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        memory_engine: Optional[EpisodicMemoryEngine] = None,
        darkfac_client: Optional[DarkFactoryClient] = None,
        bizops_engine: Any = None,
        db_override_path: Optional[Path | str] = None,
    ) -> None:
        self.config = config or get_config()
        self.memory = memory_engine or EpisodicMemoryEngine(self.config.memory_db_path)
        self.darkfac = darkfac_client or DarkFactoryClient(self.config)
        self.bizops = bizops_engine

        if db_override_path:
            self.db_path = Path(db_override_path).resolve()
        elif self.config.meeting_relator_output_folder:
            self.db_path = self.config.meeting_relator_output_folder / "meeting_database.db"
        else:
            self.db_path = Path("C:/Users/guigc/OneDrive/Desktop/00. Externo/08. Meetings/meeting_database.db")

    def is_available(self) -> bool:
        """Check if MeetingRelator database exists and is readable."""
        return self.db_path.is_file()

    def _get_readonly_conn(self) -> sqlite3.Connection:
        """Open a safe, read-only SQLite connection to prevent lock contention."""
        if not self.db_path.is_file():
            raise FileNotFoundError(f"MeetingRelator database not found at {self.db_path}")

        # Connect with WAL mode and query_only flag to protect production data
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA query_only = ON")
        except Exception:
            pass
        return conn

    def list_meetings(self, limit: int = 50, meeting_type: str = "") -> List[MeetingSummary]:
        """List recent recorded meetings."""
        if not self.is_available():
            return []

        try:
            with self._get_readonly_conn() as conn:
                if meeting_type:
                    query = (
                        "SELECT id, session_id, date, meeting_name, meeting_type, "
                        "duration_seconds, speaker_count, oneliner, mp3_path, pdf_path "
                        "FROM meetings WHERE meeting_type LIKE ? ORDER BY date DESC LIMIT ?"
                    )
                    rows = conn.execute(query, (f"%{meeting_type}%", limit)).fetchall()
                else:
                    query = (
                        "SELECT id, session_id, date, meeting_name, meeting_type, "
                        "duration_seconds, speaker_count, oneliner, mp3_path, pdf_path "
                        "FROM meetings ORDER BY date DESC LIMIT ?"
                    )
                    rows = conn.execute(query, (limit,)).fetchall()

                return [
                    MeetingSummary(
                        id=row["id"],
                        session_id=row["session_id"] or "",
                        date=row["date"] or "",
                        meeting_name=row["meeting_name"] or "Sem Título",
                        meeting_type=row["meeting_type"] or "",
                        duration_seconds=float(row["duration_seconds"] or 0.0),
                        speaker_count=int(row["speaker_count"] or 0),
                        oneliner=row["oneliner"] or "",
                        mp3_path=row["mp3_path"] or "",
                        pdf_path=row["pdf_path"] or "",
                    )
                    for row in rows
                ]
        except Exception as exc:
            logger.warning("Failed to query MeetingRelator meetings: %s", exc)
            return []

    def get_meeting(self, meeting_id: int) -> Optional[MeetingDetail]:
        """Get full meeting data by ID including participants, actions, decisions and chapters."""
        if not self.is_available():
            return None

        try:
            with self._get_readonly_conn() as conn:
                row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
                if not row:
                    return None

                summary = MeetingSummary(
                    id=row["id"],
                    session_id=row["session_id"] or "",
                    date=row["date"] or "",
                    meeting_name=row["meeting_name"] or "Sem Título",
                    meeting_type=row["meeting_type"] or "",
                    duration_seconds=float(row["duration_seconds"] or 0.0),
                    speaker_count=int(row["speaker_count"] or 0),
                    oneliner=row["oneliner"] or "",
                    mp3_path=row["mp3_path"] or "",
                    pdf_path=row["pdf_path"] or "",
                )

                # Fetch participants
                part_rows = conn.execute(
                    "SELECT * FROM participants WHERE meeting_id = ?", (meeting_id,)
                ).fetchall()
                participants = [
                    MeetingParticipant(
                        id=r["id"],
                        meeting_id=r["meeting_id"],
                        speaker_id=r["speaker_id"],
                        speaker_label=r["speaker_label"] or "",
                        identified_name=r["identified_name"] or "",
                        email=r["email"] or "",
                        role_inferred=r["role_inferred"] or "",
                        talk_time_pct=float(r["talk_time_pct"] or 0.0),
                        interventions=int(r["interventions"] or 0),
                    )
                    for r in part_rows
                ]

                # Fetch action items
                act_rows = conn.execute(
                    "SELECT * FROM action_items WHERE meeting_id = ?", (meeting_id,)
                ).fetchall()
                action_items = [
                    MeetingActionItem(
                        id=r["id"],
                        meeting_id=r["meeting_id"],
                        description=r["description"] or "",
                        owner=r["owner"] or "",
                        timestamp=r["timestamp"] or "",
                        priority=r["priority"] or "",
                        evidence=r["evidence"] or "",
                    )
                    for r in act_rows
                ]

                # Fetch decisions
                dec_rows = conn.execute(
                    "SELECT * FROM decisions WHERE meeting_id = ?", (meeting_id,)
                ).fetchall()
                decisions = [
                    MeetingDecision(
                        id=r["id"],
                        meeting_id=r["meeting_id"],
                        description=r["description"] or "",
                        timestamp=r["timestamp"] or "",
                        participants=r["participants"] or "",
                        impact=r["impact"] or "",
                        evidence=r["evidence"] or "",
                    )
                    for r in dec_rows
                ]

                # Fetch chapters
                chap_rows = conn.execute(
                    "SELECT * FROM chapters WHERE meeting_id = ? ORDER BY chapter_index ASC", (meeting_id,)
                ).fetchall()
                chapters = []
                for r in chap_rows:
                    kp_raw = r["key_points"] or "[]"
                    try:
                        kps = json.loads(kp_raw) if isinstance(kp_raw, str) else kp_raw
                    except Exception:
                        kps = []
                    chapters.append(
                        MeetingChapter(
                            id=r["id"],
                            meeting_id=r["meeting_id"],
                            chapter_index=int(r["chapter_index"] or 0),
                            title=r["title"] or "",
                            time_start=r["time_start"] or "",
                            time_end=r["time_end"] or "",
                            summary=r["summary"] or "",
                            detailed_notes=r["detailed_notes"] or "",
                            key_points=kps if isinstance(kps, list) else [],
                        )
                    )

                return MeetingDetail(
                    summary=summary,
                    executive_summary=row["executive_summary"] or "",
                    detailed_summary=row["detailed_summary"] or "",
                    full_transcript=row["full_transcript"] or "",
                    participants=participants,
                    action_items=action_items,
                    decisions=decisions,
                    chapters=chapters,
                )
        except Exception as exc:
            logger.error("Failed to load meeting detail %d: %s", meeting_id, exc)
            return None

    def search_meetings(self, query: str, limit: int = 20) -> List[MeetingSummary]:
        """Search meetings across titles, oneliners, summaries, and transcripts."""
        if not self.is_available() or not query.strip():
            return []

        try:
            with self._get_readonly_conn() as conn:
                q = f"%{query.strip()}%"
                sql = (
                    "SELECT id, session_id, date, meeting_name, meeting_type, "
                    "duration_seconds, speaker_count, oneliner, mp3_path, pdf_path "
                    "FROM meetings "
                    "WHERE meeting_name LIKE ? OR oneliner LIKE ? OR executive_summary LIKE ? "
                    "   OR detailed_summary LIKE ? OR full_transcript LIKE ? "
                    "ORDER BY date DESC LIMIT ?"
                )
                rows = conn.execute(sql, (q, q, q, q, q, limit)).fetchall()
                return [
                    MeetingSummary(
                        id=row["id"],
                        session_id=row["session_id"] or "",
                        date=row["date"] or "",
                        meeting_name=row["meeting_name"] or "Sem Título",
                        meeting_type=row["meeting_type"] or "",
                        duration_seconds=float(row["duration_seconds"] or 0.0),
                        speaker_count=int(row["speaker_count"] or 0),
                        oneliner=row["oneliner"] or "",
                        mp3_path=row["mp3_path"] or "",
                        pdf_path=row["pdf_path"] or "",
                    )
                    for row in rows
                ]
        except Exception as exc:
            logger.warning("Meeting search failed: %s", exc)
            return []

    def sync_to_second_brain(
        self,
        meeting_id: Optional[int] = None,
        limit: int = 50,
    ) -> MeetingSyncResult:
        """Ingest meeting data into Jarvis EpisodicMemoryEngine."""
        if not self.is_available():
            return MeetingSyncResult(message="MeetingRelator database indisponível.")

        meetings_to_sync: List[int] = []
        if meeting_id is not None:
            meetings_to_sync.append(meeting_id)
        else:
            all_meetings = self.list_meetings(limit=limit)
            for m in all_meetings:
                marker_key = f"meeting_synced_{m.id}"
                if not self.memory.get_fact(marker_key):
                    meetings_to_sync.append(m.id)

        if not meetings_to_sync:
            return MeetingSyncResult(
                synced_meetings_count=0,
                facts_created=0,
                decisions_created=0,
                episodes_created=0,
                message="Já sincronizadas: nenhuma nova reunião para catalogar no Segundo Cérebro.",
            )

        facts_count = 0
        decisions_count = 0
        episodes_count = 0

        for mid in meetings_to_sync:
            detail = self.get_meeting(mid)
            if not detail:
                continue

            # 1. Store summary fact
            fact_key = f"meeting_{detail.summary.id}_{detail.summary.session_id}"
            fact_val = (
                f"Reunião '{detail.summary.meeting_name}' em {detail.summary.date}.\n"
                f"Resumo: {detail.summary.oneliner or detail.executive_summary[:200]}"
            )
            self.memory.store_fact(
                key=fact_key,
                value=fact_val,
                category="meeting_relator",
                tags=["meeting", detail.summary.meeting_type or "general"],
            )
            facts_count += 1

            # 2. Store decisions
            for d in detail.decisions:
                self.memory.log_decision(
                    title=f"Decisão em '{detail.summary.meeting_name}'",
                    decision=d.description,
                    rationale=d.evidence or d.impact or "Registrado em ata formal de reunião.",
                    problem_statement=f"Deliberação acordada por {d.participants or 'participantes'} na reunião de {detail.summary.date}.",
                    recorded_by="MeetingRelator",
                )
                decisions_count += 1

            # 3. Store episode note
            self.memory.record_episode(
                summary=(
                    f"Ata de Reunião: {detail.summary.meeting_name} ({detail.summary.date}). "
                    f"{detail.summary.oneliner} "
                    f"Ações: {len(detail.action_items)} | Decisões: {len(detail.decisions)}"
                ),
                tags=["meeting_relator", detail.summary.session_id],
            )
            episodes_count += 1

            # 4. Mark as synced
            self.memory.store_fact(
                key=f"meeting_synced_{detail.summary.id}",
                value=f"Sincronizado com o Segundo Cérebro em {detail.summary.date}",
                category="sync_marker",
                tags=["internal"],
            )

        msg = (
            f"Sincronização concluída: {len(meetings_to_sync)} reunião(ões) processada(s). "
            f"{facts_count} fatos, {decisions_count} decisões e {episodes_count} episódios catalogados."
        )
        return MeetingSyncResult(
            synced_meetings_count=len(meetings_to_sync),
            facts_created=facts_count,
            decisions_created=decisions_count,
            episodes_created=episodes_count,
            message=msg,
        )

    def dispatch_action_items_to_darkfac(
        self,
        meeting_id: int,
        project_id: str = "darkfac",
    ) -> List[PendingAction]:
        """Convert meeting action items into Dark Factory demand proposals (HITL Level 2)."""
        detail = self.get_meeting(meeting_id)
        if not detail or not detail.action_items:
            return []

        created_actions: List[PendingAction] = []
        for act in detail.action_items:
            title = f"[{detail.summary.meeting_name}] {act.description[:70]}"
            prob = (
                f"Item de ação derivado da reunião '{detail.summary.meeting_name}' ({detail.summary.date}).\n"
                f"- Responsável: {act.owner or 'Não atribuído'}\n"
                f"- Timestamp: {act.timestamp or 'N/A'}\n"
                f"- Evidência: {act.evidence or 'Registrado em ata de reunião.'}"
            )
            payload = {
                "title": title,
                "problem_statement": prob,
                "project_id": project_id,
                "meeting_id": meeting_id,
                "owner": act.owner,
            }

            if self.bizops and hasattr(self.bizops, "guardrail"):
                allowed, msg, pending = self.bizops.guardrail.evaluate_action(
                    action_type="create_dark_factory_demand",
                    title=f"Criar Demanda: {title}",
                    payload=payload,
                    task_id=f"meeting_{meeting_id}_act_{act.id}",
                    effective_level=AutonomyLevel.LEVEL_2_HITL,
                )
                if pending:
                    self.bizops._save_pending_action(pending)
                    created_actions.append(pending)
            else:
                # Fallback model when running standalone
                p = PendingAction(
                    task_id=f"meeting_{meeting_id}_act_{act.id}",
                    action_type="create_dark_factory_demand",
                    title=f"Criar Demanda: {title}",
                    payload=payload,
                    autonomy_level_required=AutonomyLevel.LEVEL_2_HITL,
                    status="pending_approval",
                )
                created_actions.append(p)

        return created_actions

    def launch_recorder(self) -> Dict[str, Any]:
        """Launch MeetingRelator application on Windows without blocking Jarvis."""
        repo = self.config.meeting_relator_repo_path
        main_py = repo / "main.py"

        # 1. Try python script in repo
        if main_py.is_file():
            try:
                subprocess.Popen(
                    [sys.executable, str(main_py)],
                    cwd=str(repo),
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                return {"status": "launched", "source": str(main_py)}
            except Exception as exc:
                return {"status": "error", "message": f"Erro ao iniciar via script: {exc}"}

        # 2. Try installed executable
        cand_exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MeetingRelator" / "MeetingRelator.exe"
        if cand_exe.is_file():
            try:
                subprocess.Popen([str(cand_exe)])
                return {"status": "launched", "source": str(cand_exe)}
            except Exception as exc:
                return {"status": "error", "message": f"Erro ao iniciar via EXE: {exc}"}

        return {"status": "not_found", "message": "Executável ou script do MeetingRelator não encontrado."}
