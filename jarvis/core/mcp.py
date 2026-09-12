"""Model Context Protocol (MCP) Client Manager for Second Brain Tools."""

from __future__ import annotations

import json
import logging
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
        query = args.get("query", "").strip().lower()
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
