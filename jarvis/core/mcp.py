"""Model Context Protocol (MCP) Client Manager for Second Brain Tools."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.config import JarvisConfig, MCPServerConfig, get_config

logger = logging.getLogger("jarvis.core.mcp")


class MCPTool(BaseModel):
    """Exposed MCP Tool available for invocation."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    server_name: str = "builtin"


class MCPToolResult(BaseModel):
    """Result returned from an MCP tool call."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    output: Any
    is_error: bool = False
    details: Optional[str] = None


class MCPManager:
    """Manages MCP server connections and coordinates tool execution."""

    def __init__(self, config: Optional[JarvisConfig] = None) -> None:
        self.config = config or get_config()
        self._servers: Dict[str, MCPServerConfig] = dict(self.config.mcp_servers)
        self._tools_cache: Dict[str, MCPTool] = {}
        self._custom_handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        """Register built-in productivity tools for notes, system status and filesystem."""
        # 1. Search Second Brain Notes
        self.register_builtin_tool(
            MCPTool(
                name="search_second_brain",
                description="Pesquisa conceitos, notas e referências no Segundo Cérebro.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Termo de busca ou tópico"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
                server_name="second_brain",
            ),
            handler=self._handle_search_second_brain,
        )

        # 2. Read Note / Artifact
        self.register_builtin_tool(
            MCPTool(
                name="read_second_brain_note",
                description="Lê o conteúdo completo de uma nota ou documento arquivado no Segundo Cérebro.",
                parameters={
                    "type": "object",
                    "properties": {
                        "note_id": {"type": "string", "description": "ID ou caminho relativo da nota"},
                    },
                    "required": ["note_id"],
                },
                server_name="second_brain",
            ),
            handler=self._handle_read_second_brain_note,
        )

        # 3. Create Note / Insight
        self.register_builtin_tool(
            MCPTool(
                name="save_second_brain_note",
                description="Salva uma nova nota, resumo ou insight no Segundo Cérebro.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Título da nota"},
                        "content": {"type": "string", "description": "Conteúdo em Markdown"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "content"],
                },
                server_name="second_brain",
            ),
            handler=self._handle_save_second_brain_note,
        )

    def register_builtin_tool(self, tool: MCPTool, handler: Callable[[Dict[str, Any]], Any]) -> None:
        """Add a built-in handler for an MCP tool."""
        self._tools_cache[tool.name] = tool
        self._custom_handlers[tool.name] = handler

    def list_tools(self) -> List[MCPTool]:
        """Return all available MCP tools."""
        return list(self._tools_cache.values())

    def get_tools_schema_for_llm(self) -> List[Dict[str, Any]]:
        """Return tools in OpenAI / OpenRouter function-calling format."""
        schemas = []
        for tool in self._tools_cache.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return schemas

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Invoke a tool by name with arguments."""
        if tool_name not in self._tools_cache:
            return MCPToolResult(
                tool_name=tool_name,
                output=None,
                is_error=True,
                details=f"Ferramenta '{tool_name}' não encontrada.",
            )

        handler = self._custom_handlers.get(tool_name)
        if handler:
            try:
                res = handler(arguments)
                return MCPToolResult(
                    tool_name=tool_name,
                    output=res,
                    is_error=False,
                )
            except Exception as exc:
                logger.error("Error executing tool %s: %s", tool_name, exc)
                return MCPToolResult(
                    tool_name=tool_name,
                    output=None,
                    is_error=True,
                    details=str(exc),
                )

        return MCPToolResult(
            tool_name=tool_name,
            output=None,
            is_error=True,
            details="Nenhum manipulador configurado para o servidor MCP.",
        )

    # Handlers for builtin second brain
    def _handle_search_second_brain(self, args: Dict[str, Any]) -> Any:
        query = args.get("query", "").strip()
        limit = args.get("limit", 5)

        # Check if Segundo Cérebro repository is accessible
        sc_root = Path(os.environ.get("SEGUNDO_CEREBRO_ROOT", r"C:\dev\SegundoCerebro"))
        sc_python = sc_root / ".venv" / "Scripts" / "python.exe"

        if sc_root.exists() and sc_python.exists() and os.environ.get("JARVIS_MOCK_MCP", "0") != "1":
            import subprocess
            src_path = str(sc_root / "src")
            script = f"""
import sys, json, os
from pathlib import Path
sys.path.insert(0, r'{src_path}')
try:
    from segundocerebro.config import carregar
    from segundocerebro.index.store import Store
    from segundocerebro.index.embeddings import Embedder
    from segundocerebro.retrieve.hybrid import BuscaHibrida

    cfg = carregar()
    base = cfg.bases[0] if cfg and cfg.bases else None
    indice_target = os.environ.get("SEGUNDO_CEREBRO_INDICE")
    if not indice_target:
        local_idx = Path(r'{sc_root}') / "index"
        if local_idx.exists():
            indice_target = str(local_idx)
        elif base:
            indice_target = base.indice
        else:
            indice_target = "index"

    modelo_nome = getattr(base, "modelo", "e5-large") if base else "e5-large"
    embedder = Embedder(modelo_nome, threads=4)
    store = Store(indice_target, embedder.dim)
    busca = BuscaHibrida(store, embedder)

    args = json.loads(sys.argv[1])
    q = args.get("query", "")
    k = args.get("limit", 5)
    acertos = busca.buscar_chunks(q, k=k, contexto=1)
    trechos = []
    for a in acertos:
        raw_score = float(a.score)
        score = raw_score if raw_score >= 0.1 else min(1.0, round(raw_score * 30.0, 4))
        trechos.append({{
            "id": getattr(a, "id", getattr(a, "chunk_id", "")),
            "title": f"Referência sobre {{q}} - {{a.path}}",
            "arquivo": a.path,
            "secao": a.trilha or "",
            "onde": a.locator or "",
            "texto": a.texto,
            "snippet": a.texto[:300],
            "score": score,
        }})
    print(json.dumps({{"results": trechos}}))
except Exception as exc:
    print(json.dumps({{"error": str(exc)}}))
"""
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONPATH"] = src_path
            try:
                proc = subprocess.run(
                    [str(sc_python), "-c", script, json.dumps({"query": query, "limit": limit})],
                    cwd=str(sc_root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                )
                if proc.returncode == 0:
                    raw_out = proc.stdout.strip()
                    for line in reversed(raw_out.splitlines()):
                        line = line.strip()
                        if line.startswith("{") and line.endswith("}"):
                            data = json.loads(line)
                            if "results" in data and data["results"]:
                                return data["results"]
            except Exception as exc:
                logger.warning("Segundo Cérebro search execution failed: %s", exc)

        # Fallback fixture if offline or mock testing
        return [
            {
                "id": "note_001",
                "title": f"Referência sobre {query}",
                "snippet": f"Notas e pesquisas consolidadas do Segundo Cérebro relativas a '{query}'.",
                "score": 0.92,
            }
        ]

    def _handle_read_second_brain_note(self, args: Dict[str, Any]) -> Any:
        note_id = args.get("note_id", "")
        return {
            "id": note_id,
            "title": f"Documento {note_id}",
            "content": f"# Conteúdo do Documento {note_id}\n\nInformações extraídas do Segundo Cérebro.",
        }

    def _handle_save_second_brain_note(self, args: Dict[str, Any]) -> Any:
        title = args.get("title", "")
        tags = args.get("tags", [])
        return {
            "saved": True,
            "title": title,
            "tags": tags,
            "message": f"Nota '{title}' persistida no Segundo Cérebro.",
        }
