"""Conversational and Tool-Calling Assistant Engine for Jarvis."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.config import JarvisConfig, get_config
from jarvis.core.darkfac import DarkFactoryClient
from jarvis.core.mcp import MCPManager
from jarvis.core.models import ChatMessage, ModelResponse, UnifiedModelRouter

logger = logging.getLogger("jarvis.core.assistant")

SYSTEM_PROMPT = """Você é o Jarvis, um assistente pessoal de produtividade inteligente, executivo e proativo.
Você opera conectado a dois grandes ecossistemas:
1. **Segundo Cérebro**: através de MCPs, você pode buscar notas, ler referências e arquivar novos conhecimentos.
2. **Dark Factory**: você tem acesso ao DarkHub para consultar status, inspecionar tickets e registrar novas demandas.

Diretrizes de Resposta:
- Seja claro, objetivo, elegante e direto.
- Responda em Português do Brasil com excelente formatação em Markdown.
- Se o usuário pedir para criar um ticket ou demanda para a Dark Factory, processe ou sugira a chamada da ferramenta correspondente.
- Destaque links ou ações práticas de forma clara.
"""


class AssistantTurnResult(BaseModel):
    """Result of a single user-assistant interaction turn."""

    model_config = ConfigDict(frozen=True)

    response_text: str
    model_used: str
    provider_used: str
    tools_executed: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0


class JarvisAssistant:
    """Orchestrates conversations, tool invocations, and action triggers."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        model_router: Optional[UnifiedModelRouter] = None,
        darkfac_client: Optional[DarkFactoryClient] = None,
        mcp_manager: Optional[MCPManager] = None,
    ) -> None:
        self.config = config or get_config()
        self.models = model_router or UnifiedModelRouter(self.config)
        self.darkfac = darkfac_client or DarkFactoryClient(self.config)
        self.mcp = mcp_manager or MCPManager(self.config)
        self._register_darkfac_mcp_tools()

    def _register_darkfac_mcp_tools(self) -> None:
        """Expose Dark Factory actions as MCP tools inside the assistant."""
        from jarvis.core.mcp import MCPTool

        # Tool 1: Create Demand in DarkHub
        self.mcp.register_builtin_tool(
            MCPTool(
                name="create_dark_factory_demand",
                description="Cria uma nova demanda ou ticket de funcionalidade no backlog da Dark Factory.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Título claro e objetivo da demanda"},
                        "problem_statement": {"type": "string", "description": "Descrição do problema ou objetivo"},
                        "project_id": {"type": "string", "default": "darkfac", "description": "ID do projeto"},
                    },
                    "required": ["title"],
                },
                server_name="dark_factory",
            ),
            handler=self._handle_create_darkfac_demand,
        )

        # Tool 2: Check DarkHub Status
        self.mcp.register_builtin_tool(
            MCPTool(
                name="check_dark_factory_status",
                description="Verifica a disponibilidade da Dark Factory e do DarkHub na nuvem.",
                parameters={"type": "object", "properties": {}},
                server_name="dark_factory",
            ),
            handler=self._handle_check_darkfac_status,
        )

    def _handle_create_darkfac_demand(self, args: Dict[str, Any]) -> Any:
        import asyncio

        title = args.get("title", "")
        desc = args.get("problem_statement", title)
        proj = args.get("project_id", "darkfac")
        # Direct sync wrapper for the async call
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.darkfac.create_demand(title, desc, project_id=proj)).result()
            return loop.run_until_complete(self.darkfac.create_demand(title, desc, project_id=proj))
        except Exception as exc:
            return {"status": "error", "message": f"Erro ao criar demanda: {exc}"}

    def _handle_check_darkfac_status(self, args: Dict[str, Any]) -> Any:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    res = pool.submit(asyncio.run, self.darkfac.get_status()).result()
                    return res.model_dump()
            res = loop.run_until_complete(self.darkfac.get_status())
            return res.model_dump()
        except Exception as exc:
            return {"status": "offline", "error": str(exc)}

    async def chat(
        self,
        user_message: str,
        history: Optional[List[ChatMessage]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> AssistantTurnResult:
        """Process a conversation turn with tools support."""
        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT)
        ]
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_message))

        # Check for direct intent shortcuts (e.g. demand creation or status check)
        lowered = user_message.lower()
        tools_executed: List[Dict[str, Any]] = []

        if "status da dark factory" in lowered or "status do darkhub" in lowered:
            status_res = await self.darkfac.get_status()
            executed = {"tool": "check_dark_factory_status", "result": status_res.model_dump()}
            tools_executed.append(executed)
            st_text = "🟢 **Online**" if status_res.online else f"🔴 **Offline** ({status_res.error or 'Falha de rede'})"
            response_text = f"O DarkHub está atualmente {st_text}.\n- **URL**: `{status_res.url}`"
            return AssistantTurnResult(
                response_text=response_text,
                model_used="internal-intent-router",
                provider_used="builtin",
                tools_executed=tools_executed,
            )

        # Normal model completion with tool schemas
        tools_schema = self.mcp.get_tools_schema_for_llm()
        try:
            resp: ModelResponse = await self.models.generate(
                messages=messages,
                model=model,
                provider=provider,
                tools=tools_schema if tools_schema else None,
            )
            response_text = resp.text

            # Execute tool calls if returned by model
            if resp.tool_calls:
                for tcall in resp.tool_calls:
                    fn = tcall.get("function", {})
                    fn_name = fn.get("name")
                    fn_args = {}
                    try:
                        raw_args = fn.get("arguments", "{}")
                        fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        pass

                    if fn_name:
                        t_res = await self.mcp.execute_tool(fn_name, fn_args)
                        tools_executed.append({
                            "tool": fn_name,
                            "arguments": fn_args,
                            "output": t_res.output,
                            "is_error": t_res.is_error,
                        })

                # Append execution summary to response if text was empty
                if not response_text and tools_executed:
                    response_text = f"Ação executada com sucesso: `{tools_executed[0].get('tool')}`."

            return AssistantTurnResult(
                response_text=response_text,
                model_used=resp.model,
                provider_used=resp.provider,
                tools_executed=tools_executed,
                latency_ms=resp.latency_ms,
            )
        except Exception as exc:
            logger.error("Chat generation failed: %s", exc)
            return AssistantTurnResult(
                response_text=f"Desculpe, ocorreu uma falha ao contatar o provedor de IA: {exc}",
                model_used="error",
                provider_used="none",
                tools_executed=[],
            )
