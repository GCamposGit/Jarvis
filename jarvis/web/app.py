"""FastAPI Web Application and REST/WebSocket Gateway for Jarvis."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jarvis.core.assistant import AssistantTurnResult, JarvisAssistant
from jarvis.core.audio import AudioTranscriptionEngine, TranscriptionResult
from jarvis.core.config import JarvisConfig, get_config
from jarvis.core.darkfac import DarkDemandSummary, DarkFactoryClient, DarkHubStatus
from jarvis.core.mcp import MCPManager, MCPTool, MCPToolResult
from jarvis.core.models import ChatMessage, UnifiedModelRouter

logger = logging.getLogger("jarvis.web")

STATIC_DIR = Path(__file__).resolve().parent / "static"


# Request / Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[Dict[str, str]] = Field(default_factory=list)
    model: Optional[str] = None
    provider: Optional[str] = None


class CreateDemandRequest(BaseModel):
    title: str = Field(..., min_length=3)
    problem_statement: str = ""
    project_id: str = "darkfac"
    acceptance_criteria: List[str] = Field(default_factory=list)


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    darkhub_online: bool
    ollama_online: bool
    models_available: List[str]
    tools_count: int


def create_app(config: Optional[JarvisConfig] = None) -> FastAPI:
    """Factory creating and configuring the Jarvis FastAPI application."""
    cfg = config or get_config()
    app = FastAPI(
        title="Jarvis Assistant",
        description="Autonomous Personal Productivity Assistant Web Gateway",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize engines
    model_router = UnifiedModelRouter(cfg)
    darkfac_client = DarkFactoryClient(cfg)
    mcp_manager = MCPManager(cfg)
    assistant = JarvisAssistant(
        config=cfg,
        model_router=model_router,
        darkfac_client=darkfac_client,
        mcp_manager=mcp_manager,
    )
    audio_engine = AudioTranscriptionEngine(cfg)

    # Store in app state for tests and handlers
    app.state.config = cfg
    app.state.assistant = assistant
    app.state.audio_engine = audio_engine
    app.state.darkfac = darkfac_client
    app.state.mcp = mcp_manager
    app.state.models = model_router

    # Ensure static directory exists
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # REST Routes
    @app.get("/api/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        darkhub_st = await darkfac_client.get_status()
        local_models = await model_router.list_local_models()
        return HealthResponse(
            status="ok",
            version="0.1.0",
            darkhub_online=darkhub_st.online,
            ollama_online=len(local_models) > 0,
            models_available=local_models,
            tools_count=len(mcp_manager.list_tools()),
        )

    @app.post("/api/chat", response_model=AssistantTurnResult)
    async def post_chat(req: ChatRequest) -> AssistantTurnResult:
        hist = [ChatMessage(role=m.get("role", "user"), content=m.get("content", "")) for m in req.history]
        return await assistant.chat(
            user_message=req.message,
            history=hist,
            model=req.model,
            provider=req.provider,
        )

    @app.post("/api/audio/transcribe", response_model=TranscriptionResult)
    async def transcribe_audio(
        file: UploadFile = File(...),
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        tmp_id = uuid.uuid4().hex[:8]
        ext = Path(file.filename or "audio.wav").suffix or ".wav"
        save_path = cfg.audio_upload_dir / f"upload_{tmp_id}{ext}"
        try:
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            return audio_engine.transcribe(save_path, language=language)
        finally:
            save_path.unlink(missing_ok=True)

    @app.get("/api/darkfac/status", response_model=DarkHubStatus)
    async def get_darkfac_status() -> DarkHubStatus:
        return await darkfac_client.get_status()

    @app.get("/api/darkfac/demands", response_model=List[DarkDemandSummary])
    async def get_darkfac_demands(project_id: Optional[str] = None) -> List[DarkDemandSummary]:
        return await darkfac_client.list_demands(project_id=project_id)

    @app.post("/api/darkfac/demands")
    async def create_darkfac_demand(req: CreateDemandRequest) -> Dict[str, Any]:
        return await darkfac_client.create_demand(
            title=req.title,
            problem_statement=req.problem_statement,
            project_id=req.project_id,
            acceptance_criteria=req.acceptance_criteria,
        )

    @app.get("/api/mcp/tools", response_model=List[MCPTool])
    async def get_mcp_tools() -> List[MCPTool]:
        return mcp_manager.list_tools()

    @app.post("/api/mcp/call", response_model=MCPToolResult)
    async def call_mcp_tool(req: ToolCallRequest) -> MCPToolResult:
        return await mcp_manager.execute_tool(req.tool_name, req.arguments)

    # Static UI routes
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        async def root_index() -> FileResponse:
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app
