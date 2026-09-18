"""Conversational and Tool-Calling Assistant Engine for Jarvis."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.bizops import BizOpsEngine
from jarvis.core.config import JarvisConfig, get_config
from jarvis.core.darkfac import DarkFactoryClient
from jarvis.core.harness import (
    DeterministicCodeSandbox,
    ExecutionRiskLevel,
    HarnessAuditEntry,
    HarnessAuditLedger,
    HarnessSafetyGuardrail,
    PreflightCheckResult,
    SandboxExecutionResult,
)
from jarvis.core.mcp import MCPManager
from jarvis.core.meeting_relator import MeetingRelatorBridge
from jarvis.core.memory import EpisodicMemoryEngine
from jarvis.core.models import ChatMessage, ModelResponse, UnifiedModelRouter
from jarvis.core.telemetry import (
    BudgetPolicy,
    BudgetStatus,
    TelemetryEngine,
    TelemetrySummary,
)

logger = logging.getLogger("jarvis.core.assistant")

SYSTEM_PROMPT = """Você é o Jarvis, um assistente pessoal executivo de alta inteligência, produtividade e engenharia.
Você opera conectado aos seguintes ecossistemas:
1. **Memória Episódica & Contexto Pessoal**: via ferramentas MCP (`store_memory_fact`, `recall_memory`, `log_decision`, `record_failed_approach`), contendo os fatos declarados pelo usuário, preferências, pessoas conhecidas, regras de negócio informadas e histórico operacional gravados no SQLite local.
2. **Segundo Cérebro (Acervo Corporativo)**: via ferramentas MCP (`search_second_brain`, `read_second_brain_note`, `save_second_brain_note`), contendo o acervo documental formal da empresa (políticas PO/PR/PL, contratos, propostas, relatórios e arquivos estáticos).
3. **Dark Factory**: via DarkHub, para telemetria da fábrica autônoma de software, inspeção de backlog e registro de demandas.
4. **MeetingRelator & Reuniões**: via ferramentas MCP (`list_recent_meetings`, `get_meeting_details`, `search_meetings`, `sync_meetings_to_second_brain`, `dispatch_meeting_demands`, `launch_meeting_recorder`), para consultar transcrições completas de reuniões, atas, decisões tomadas, participantes e despachar itens de ação como demandas para a Dark Factory.
5. **Agent Harness & Sandbox Determinístico**: via ferramentas MCP (`run_sandboxed_python`, `validate_execution_safety`, `get_harness_audit_log`), para executar cálculos e análises em Python em ambiente seguro com contenção de recursos, verificar segurança de caminhos/comandos e auditar eventos de segurança.
6. **Telemetria de Tokens & Governança Orçamentária**: via ferramentas MCP (`get_telemetry_summary`, `get_budget_status`, `update_budget_policy`), para consultar o consumo financeiro em tempo real, verificar a economia acumulada ($0 local vs comercial) e controlar tetos de gastos.

Instruções Mandatórias sobre Memória e Uso de Ferramentas:
- **Armazenamento de Fatos na Memória Episódica (`store_memory_fact`)**: Sempre que o usuário disser "guarde", "lembre-se", "armazene", "registre o fato" ou declarar dados importantes sobre sua vida, preferências, família, ou operações de negócios, você DEVE acionar a ferramenta `store_memory_fact` para persistir o fato.
- **Consulta à Memória Episódica (`recall_memory`)**: Sempre que o usuário perguntar sobre o que ele já te disse, sua memória ("sua memória", "o que você sabe sobre mim"), pessoas conhecidas, família, preferências pessoais ou métricas operacionais ditadas diretamente por ele, consulte a Memória Episódica (ou o bloco `[MEMÓRIA EPISÓDICA E CONTEXTO DO SEGUNDO CÉREBRO]` já injetado neste prompt). NUNCA faça busca no acervo corporativo (`search_second_brain`) para dados pessoais ou fatos operacionais declarados pelo usuário na conversa.
- **Consulta ao Segundo Cérebro Corporativo (`search_second_brain`)**: Apenas acione `search_second_brain` quando o usuário perguntar expressamente sobre documentos, políticas internas formais (ex: PO, PR, PL), normas corporativas, contratos, apresentações institucionais ou notas técnicas do acervo da empresa. Com base nos trechos reais recuperados, cite códigos de documentos (ex: PO-CORP-007) e seções.

Diretrizes de Comunicação e Resposta (Dual-Channel Output):
- **Resumo Falado Inicial**: Inicie sempre sua resposta com 1 ou 2 frases executivas, diretas e afirmativas. Esse primeiro trecho será sintetizado em voz para o operador.
- **Detalhamento Técnico (Visual)**: A seguir, forneça profundidade analítica, contexto arquitetural, justificativas de primeiros princípios, tabelas e citações formatadas em Markdown.
- Evite respostas vagas ou superficiais; responda no nível de um Staff Engineer / Principal Architect.
- Idioma padrão: Português do Brasil fluente e sofisticado.
"""


class AssistantTurnResult(BaseModel):
    """Result of a single user-assistant interaction turn."""

    model_config = ConfigDict(frozen=True)

    response_text: str
    model_used: str
    provider_used: str
    tools_executed: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    cost_usd: float = 0.0


class JarvisAssistant:
    """Orchestrates conversations, tool invocations, and action triggers."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        model_router: Optional[UnifiedModelRouter] = None,
        darkfac_client: Optional[DarkFactoryClient] = None,
        mcp_manager: Optional[MCPManager] = None,
        memory_engine: Optional[EpisodicMemoryEngine] = None,
        bizops_engine: Optional[BizOpsEngine] = None,
        meeting_bridge: Optional[MeetingRelatorBridge] = None,
        harness_guardrail: Optional[HarnessSafetyGuardrail] = None,
        code_sandbox: Optional[DeterministicCodeSandbox] = None,
        harness_audit: Optional[HarnessAuditLedger] = None,
        telemetry_engine: Optional[TelemetryEngine] = None,
    ) -> None:
        self.config = config or get_config()
        self.models = model_router or UnifiedModelRouter(self.config)
        self.darkfac = darkfac_client or DarkFactoryClient(self.config)
        self.mcp = mcp_manager or MCPManager(self.config)
        self.memory = memory_engine or EpisodicMemoryEngine(self.config.memory_db_path)
        self.bizops = bizops_engine or BizOpsEngine(
            config=self.config,
            memory_engine=self.memory,
            darkfac_client=self.darkfac,
        )
        self.meetings = meeting_bridge or MeetingRelatorBridge(
            config=self.config,
            memory_engine=self.memory,
            darkfac_client=self.darkfac,
            bizops_engine=self.bizops,
        )
        self.audit = harness_audit or HarnessAuditLedger(self.config.memory_db_path)
        self.guardrail = harness_guardrail or HarnessSafetyGuardrail()
        self.sandbox = code_sandbox or DeterministicCodeSandbox(
            guardrail=self.guardrail,
            memory_engine=self.memory,
            audit_ledger=self.audit,
        )
        self.telemetry = telemetry_engine or TelemetryEngine(self.config.memory_db_path)
        self._register_darkfac_mcp_tools()
        self._register_memory_mcp_tools()
        self._register_bizops_mcp_tools()
        self._register_meeting_mcp_tools()
        self._register_harness_mcp_tools()
        self._register_telemetry_mcp_tools()

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

    def _register_memory_mcp_tools(self) -> None:
        """Expose Episodic Memory and Second Brain actions as MCP tools."""
        from jarvis.core.mcp import MCPTool

        self.mcp.register_builtin_tool(
            MCPTool(
                name="store_memory_fact",
                description="Armazena ou atualiza um fato, preferência do usuário, pessoa da família, regra de negócio ou contexto operacional na Memória Episódica persistente.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Chave identificadora única do fato"},
                        "value": {"type": "string", "description": "Descrição detalhada do fato ou preferência"},
                        "category": {"type": "string", "default": "general", "description": "Categoria: user_preference, project_context, business_rule, tech_stack"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags para indexação"},
                    },
                    "required": ["key", "value"],
                },
                server_name="memory",
            ),
            handler=self._handle_store_memory_fact,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="recall_memory",
                description="Busca fatos, preferências, regras operacionais e pessoas gravadas na Memória Episódica do assistente. Use sempre que o usuário perguntar sobre sua memória, o que já foi dito, pessoas conhecidas ou regras customizadas.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Termo de busca"},
                        "category": {"type": "string", "description": "Filtro opcional de categoria"},
                        "limit": {"type": "integer", "default": 5},
                    },
                },
                server_name="memory",
            ),
            handler=self._handle_recall_memory,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="log_decision",
                description="Registra formalmente uma decisão de arquitetura ou de negócio para manter consistência perpétua.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Título da decisão"},
                        "decision": {"type": "string", "description": "A opção ou caminho decidido"},
                        "rationale": {"type": "string", "description": "Justificativa da escolha"},
                        "problem_statement": {"type": "string", "description": "Problema ou contexto resolvido"},
                        "alternatives": {"type": "array", "items": {"type": "string"}, "description": "Outras opções consideradas"},
                    },
                    "required": ["title", "decision"],
                },
                server_name="memory",
            ),
            handler=self._handle_log_decision,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="record_failed_approach",
                description="Registra uma abordagem que falhou ou gerou erro para evitar que futuros agentes repitam o mesmo erro.",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "description": "Ação tentada"},
                        "error_pattern": {"type": "string", "description": "Sintoma ou erro gerado"},
                        "lesson_learned": {"type": "string", "description": "Lição aprendida e como contornar"},
                    },
                    "required": ["action", "error_pattern", "lesson_learned"],
                },
                server_name="memory",
            ),
            handler=self._handle_record_failed_approach,
        )

    def _handle_store_memory_fact(self, args: Dict[str, Any]) -> Any:
        fact = self.memory.store_fact(
            key=args.get("key", ""),
            value=args.get("value", ""),
            category=args.get("category", "general"),
            tags=args.get("tags", []),
        )
        return {"status": "stored", "key": fact.key, "category": fact.category}

    def _handle_recall_memory(self, args: Dict[str, Any]) -> Any:
        facts = self.memory.recall_facts(
            query=args.get("query"),
            category=args.get("category"),
            limit=args.get("limit", 5),
        )
        return [f.model_dump() for f in facts]

    def _handle_log_decision(self, args: Dict[str, Any]) -> Any:
        dec = self.memory.log_decision(
            title=args.get("title", ""),
            decision=args.get("decision", ""),
            alternatives=args.get("alternatives", []),
            rationale=args.get("rationale", ""),
            problem_statement=args.get("problem_statement", ""),
        )
        return dec.model_dump()

    def _handle_record_failed_approach(self, args: Dict[str, Any]) -> Any:
        att = self.memory.record_failed_attempt(
            action=args.get("action", ""),
            error_pattern=args.get("error_pattern", ""),
            lesson_learned=args.get("lesson_learned", ""),
        )
        return att.model_dump()

    def _register_bizops_mcp_tools(self) -> None:
        """Expose Autonomous BizOps tasks and HITL actions as MCP tools."""
        from jarvis.core.mcp import MCPTool

        self.mcp.register_builtin_tool(
            MCPTool(
                name="list_bizops_tasks",
                description="Lista todas as tarefas de operações de negócios autônomas (BizOps) cadastradas no Jarvis.",
                parameters={"type": "object", "properties": {}},
                server_name="bizops",
            ),
            handler=self._handle_list_bizops_tasks,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="trigger_bizops_task",
                description="Dispara a execução imediata de uma tarefa operacional (ex: task_standup, task_backlog_hygiene, task_health_pulse).",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "ID da tarefa a executar"},
                    },
                    "required": ["task_id"],
                },
                server_name="bizops",
            ),
            handler=self._handle_trigger_bizops_task,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="list_pending_bizops_actions",
                description="Lista ações que exigem autorização humana (HITL Nível 2) antes de produzir efeitos externos.",
                parameters={"type": "object", "properties": {}},
                server_name="bizops",
            ),
            handler=self._handle_list_pending_bizops_actions,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="approve_bizops_action",
                description="Aprova e executa uma ação pendente sob controle humano (HITL).",
                parameters={
                    "type": "object",
                    "properties": {
                        "action_id": {"type": "string", "description": "ID da ação pendente a aprovar"},
                        "operator": {"type": "string", "default": "operator", "description": "Nome ou identificador do operador"},
                    },
                    "required": ["action_id"],
                },
                server_name="bizops",
            ),
            handler=self._handle_approve_bizops_action,
        )

    def _handle_list_bizops_tasks(self, args: Dict[str, Any]) -> Any:
        tasks = self.bizops.list_tasks()
        return [t.model_dump() for t in tasks]

    def _handle_trigger_bizops_task(self, args: Dict[str, Any]) -> Any:
        import asyncio
        task_id = args.get("task_id", "")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    res = pool.submit(asyncio.run, self.bizops.trigger_task(task_id)).result()
                    return res.model_dump()
            return loop.run_until_complete(self.bizops.trigger_task(task_id)).model_dump()
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def _handle_list_pending_bizops_actions(self, args: Dict[str, Any]) -> Any:
        pending = self.bizops.list_pending_actions()
        return [p.model_dump() for p in pending]

    def _handle_approve_bizops_action(self, args: Dict[str, Any]) -> Any:
        import asyncio
        action_id = args.get("action_id", "")
        op = args.get("operator", "operator")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    res = pool.submit(asyncio.run, self.bizops.approve_action(action_id, operator=op)).result()
                    return res.model_dump()
            return loop.run_until_complete(self.bizops.approve_action(action_id, operator=op)).model_dump()
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def _register_meeting_mcp_tools(self) -> None:
        """Expose MeetingRelator multimodal ingestion actions as MCP tools."""
        from jarvis.core.mcp import MCPTool

        self.mcp.register_builtin_tool(
            MCPTool(
                name="list_recent_meetings",
                description="Lista metadados das reuniões mais recentes gravadas e transcritas pelo MeetingRelator.",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10, "description": "Número máximo de reuniões a listar"},
                    },
                },
                server_name="meeting_relator",
            ),
            handler=self._handle_list_recent_meetings,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="get_meeting_details",
                description="Obtém a ata detalhada, transcrição completa, participantes, decisões e itens de ação de uma reunião específica.",
                parameters={
                    "type": "object",
                    "properties": {
                        "meeting_id": {"type": "integer", "description": "ID numérico da reunião no banco de dados"},
                    },
                    "required": ["meeting_id"],
                },
                server_name="meeting_relator",
            ),
            handler=self._handle_get_meeting_details,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="search_meetings",
                description="Pesquisa no histórico de reuniões por palavras-chave em transcrições, títulos ou resumos executivos.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Termo ou frase a pesquisar"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
                server_name="meeting_relator",
            ),
            handler=self._handle_search_meetings,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="sync_meetings_to_second_brain",
                description="Sincroniza reuniões do MeetingRelator com o Segundo Cérebro (armazenando fatos, decisões e episódios).",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10, "description": "Número máximo de reuniões recentes a sincronizar"},
                    },
                },
                server_name="meeting_relator",
            ),
            handler=self._handle_sync_meetings,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="dispatch_meeting_demands",
                description="Converte os itens de ação (action items) de uma reunião em demandas da Dark Factory, submetendo para aprovação HITL.",
                parameters={
                    "type": "object",
                    "properties": {
                        "meeting_id": {"type": "integer", "description": "ID da reunião"},
                        "project_id": {"type": "string", "default": "darkfac", "description": "Projeto destino na Dark Factory"},
                    },
                    "required": ["meeting_id"],
                },
                server_name="meeting_relator",
            ),
            handler=self._handle_dispatch_meeting_demands,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="launch_meeting_recorder",
                description="Inicia a aplicação de gravação do MeetingRelator no Windows em background.",
                parameters={"type": "object", "properties": {}},
                server_name="meeting_relator",
            ),
            handler=self._handle_launch_meeting_recorder,
        )

    def _handle_list_recent_meetings(self, args: Dict[str, Any]) -> Any:
        limit = args.get("limit", 10)
        meetings = self.meetings.list_meetings(limit=limit)
        return [m.model_dump() for m in meetings]

    def _handle_get_meeting_details(self, args: Dict[str, Any]) -> Any:
        meeting_id = args.get("meeting_id")
        if meeting_id is None:
            return {"status": "error", "message": "meeting_id é obrigatório."}
        detail = self.meetings.get_meeting(int(meeting_id))
        if not detail:
            return {"status": "not_found", "message": f"Reunião com ID {meeting_id} não encontrada."}
        return detail.model_dump()

    def _handle_search_meetings(self, args: Dict[str, Any]) -> Any:
        query = args.get("query", "")
        limit = args.get("limit", 10)
        results = self.meetings.search_meetings(query=query, limit=limit)
        return [m.model_dump() for m in results]

    def _handle_sync_meetings(self, args: Dict[str, Any]) -> Any:
        limit = args.get("limit", 10)
        res = self.meetings.sync_to_second_brain(limit=limit)
        return res.model_dump()

    def _handle_dispatch_meeting_demands(self, args: Dict[str, Any]) -> Any:
        meeting_id = args.get("meeting_id")
        if meeting_id is None:
            return {"status": "error", "message": "meeting_id é obrigatório."}
        proj = args.get("project_id", "darkfac")
        actions = self.meetings.dispatch_action_items_to_darkfac(int(meeting_id), project_id=proj)
        return [a.model_dump() for a in actions]

    def _handle_launch_meeting_recorder(self, args: Dict[str, Any]) -> Any:
        return self.meetings.launch_recorder()

    def _register_harness_mcp_tools(self) -> None:
        """Expose Agent Harness and Sandboxed Execution tools via MCP."""
        from jarvis.core.mcp import MCPTool

        self.mcp.register_builtin_tool(
            MCPTool(
                name="run_sandboxed_python",
                description="Executa código Python de cálculo, análise ou processamento de dados em um sandbox determinístico com timeout e isolamento de recursos.",
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Código Python completo a ser executado"},
                        "timeout_seconds": {"type": "number", "default": 5.0, "description": "Tempo limite de execução em segundos (padrão 5s)"},
                    },
                    "required": ["code"],
                },
                server_name="harness",
            ),
            handler=self._handle_run_sandboxed_python,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="validate_execution_safety",
                description="Avalia previamente se um caminho de arquivo, comando ou snippet de código atende às políticas de segurança e raio de impacto do harness.",
                parameters={
                    "type": "object",
                    "properties": {
                        "target_path": {"type": "string", "description": "Caminho de arquivo para validação (opcional)"},
                        "is_write": {"type": "boolean", "default": False, "description": "Indica se a intenção é escrever no caminho"},
                        "code": {"type": "string", "description": "Código Python para validação sintática e de segurança AST (opcional)"},
                    },
                },
                server_name="harness",
            ),
            handler=self._handle_validate_execution_safety,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="get_harness_audit_log",
                description="Recupera o histórico recente de auditoria e verificações de segurança realizadas pelo harness.",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 20, "description": "Número máximo de registros a listar"},
                        "only_blocked": {"type": "boolean", "default": False, "description": "Filtrar apenas tentativas bloqueadas por violação de segurança"},
                    },
                },
                server_name="harness",
            ),
            handler=self._handle_get_harness_audit_log,
        )

    def _handle_run_sandboxed_python(self, args: Dict[str, Any]) -> Any:
        code = args.get("code", "")
        timeout = args.get("timeout_seconds")
        if not code:
            return {"status": "error", "message": "Código Python é obrigatório."}
        res = self.sandbox.execute_python(code=code, timeout_seconds=float(timeout) if timeout else None)
        return res.model_dump()

    def _handle_validate_execution_safety(self, args: Dict[str, Any]) -> Any:
        target_path = args.get("target_path")
        code = args.get("code")
        is_write = bool(args.get("is_write", False))

        if code:
            res = self.guardrail.validate_python_syntax(code)
            return res.model_dump()
        if target_path:
            res = self.guardrail.validate_path(target_path, is_write=is_write)
            return res.model_dump()
        return {"allowed": True, "message": "Nenhum alvo especificado para validação."}

    def _handle_get_harness_audit_log(self, args: Dict[str, Any]) -> Any:
        limit = int(args.get("limit", 20))
        only_blocked = bool(args.get("only_blocked", False))
        events = self.audit.list_events(limit=limit, only_blocked=only_blocked)
        return [e.model_dump() for e in events]

    def _register_telemetry_mcp_tools(self) -> None:
        """Expose Token Telemetry and Budget Governance tools via MCP."""
        from jarvis.core.mcp import MCPTool

        self.mcp.register_builtin_tool(
            MCPTool(
                name="get_telemetry_summary",
                description="Retorna o resumo financeiro consolidado de tokens gastos, economia gerada vs modelos comerciais e consumo por modelo.",
                parameters={"type": "object", "properties": {}},
                server_name="telemetry",
            ),
            handler=self._handle_get_telemetry_summary,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="get_budget_status",
                description="Verifica o status atual do orçamento diário e mensal, gasto acumulado e se o circuit breaker está ativo.",
                parameters={"type": "object", "properties": {}},
                server_name="telemetry",
            ),
            handler=self._handle_get_budget_status,
        )

        self.mcp.register_builtin_tool(
            MCPTool(
                name="update_budget_policy",
                description="Atualiza os limites de gastos diário/mensal e configurações de circuit breaker.",
                parameters={
                    "type": "object",
                    "properties": {
                        "daily_limit_usd": {"type": "number", "description": "Novo limite diário em USD"},
                        "monthly_limit_usd": {"type": "number", "description": "Novo limite mensal em USD"},
                        "alert_threshold_pct": {"type": "number", "description": "Percentual para emissão de alertas"},
                    },
                },
                server_name="telemetry",
            ),
            handler=self._handle_update_budget_policy,
        )

    def _handle_get_telemetry_summary(self, args: Dict[str, Any]) -> Any:
        return self.telemetry.get_summary().model_dump()

    def _handle_get_budget_status(self, args: Dict[str, Any]) -> Any:
        return self.telemetry.get_budget_status().model_dump()

    def _handle_update_budget_policy(self, args: Dict[str, Any]) -> Any:
        curr = self.telemetry.get_policy()
        updated = BudgetPolicy(
            daily_limit_usd=float(args.get("daily_limit_usd", curr.daily_limit_usd)),
            monthly_limit_usd=float(args.get("monthly_limit_usd", curr.monthly_limit_usd)),
            alert_threshold_pct=float(args.get("alert_threshold_pct", curr.alert_threshold_pct)),
            auto_fallback_to_local=curr.auto_fallback_to_local,
            enforce_circuit_breaker=curr.enforce_circuit_breaker,
        )
        return self.telemetry.update_policy(updated).model_dump()

    def _extract_and_persist_facts(self, user_message: str) -> List[Dict[str, Any]]:
        """Deterministically extract and store facts if message starts with explicit memory directives."""
        clean = user_message.strip()
        pattern = re.compile(
            r"^(?:jarvis[,\s]+)?(?:guarde(?:\s+na\s+sua\s+memória|\s+na\s+memória)?|armazene|registre|salve(?:\s+na\s+sua\s+memória|\s+na\s+memória)?|lembre-?se)\s+(?:o\s+fato\s+(?:de\s+que|que)?\s*|de\s+que\s*|que\s*|:\s*)?(?P<body>.+)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.match(clean)
        if not match:
            return []

        body = match.group("body").strip()
        # Check if body contains a trailing question or secondary clause
        split_match = re.split(
            r"(?:\.{2,}|\n+|\?+|(?:[\.!?]\s+))(?=(?:qual|quem|como|onde|quando|quanto|quantos|quanta|quantas|o\s+que|por\s+que|será)\b|\?)",
            body,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        if len(split_match) > 1:
            fact_part = split_match[0].strip()
            question_part = split_match[1].strip()
        else:
            parts = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)
            if len(parts) > 1 and any(q_word in parts[1].lower() for q_word in ["qual", "quem", "como", "onde", "quando", "quanto", "?"]):
                fact_part = parts[0].strip()
                question_part = parts[1].strip()
            else:
                fact_part = body
                question_part = ""

        fact_part = fact_part.strip(" .,;:-")
        if not fact_part or len(fact_part) < 4:
            return []

        words = re.findall(r"[a-zA-Z0-9áéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]+", fact_part.lower())
        ignore_words = {
            "a", "o", "as", "os", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
            "e", "é", "que", "quem", "minha", "meu", "sua", "seu", "um", "uma", "uns", "umas", "para", "com",
            "pelo", "pela", "pelos", "pelas"
        }
        key_words = [w for w in words if w not in ignore_words and len(w) >= 3][:4]
        fact_key = "_".join(key_words) if key_words else "fato_usuario"

        category = "general"
        lower_fact = fact_part.lower()
        if any(term in lower_fact for term in ["faturamento", "receita", "lucro", "custo", "venda", "empresa", "operação", "operacao", "negócio", "negocio"]):
            category = "business_rule"
        elif any(term in lower_fact for term in ["esposa", "marido", "filho", "filha", "gosto", "prefiro", "nome", "aniversário"]):
            category = "user_preference"

        stored_fact = self.memory.store_fact(
            key=fact_key,
            value=fact_part,
            category=category,
            tags=key_words,
        )

        return [{
            "tool": "store_memory_fact",
            "arguments": {
                "key": stored_fact.key,
                "value": stored_fact.value,
                "category": stored_fact.category,
                "tags": stored_fact.tags,
            },
            "output": {
                "status": "stored",
                "key": stored_fact.key,
                "category": stored_fact.category,
            },
            "is_error": False,
            "question_part": question_part,
        }]

    async def chat(
        self,
        user_message: str,
        history: Optional[List[ChatMessage]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> AssistantTurnResult:
        """Process a conversation turn with tools support."""
        tools_executed: List[Dict[str, Any]] = []

        # 1. Proactive deterministic memory storage extraction
        extracted_facts = self._extract_and_persist_facts(user_message)
        question_followup = ""
        if extracted_facts:
            for ef in extracted_facts:
                tools_executed.append({
                    "tool": ef["tool"],
                    "arguments": ef["arguments"],
                    "output": ef["output"],
                    "is_error": False,
                })
                question_followup = ef.get("question_part", "")

            # If message is purely storage instruction without subsequent question, return direct confirmation
            if not question_followup:
                first_fact = extracted_facts[0]
                stored_key = first_fact["output"]["key"]
                stored_val = first_fact["arguments"]["value"]
                stored_cat = first_fact["output"]["category"]
                response_text = (
                    f"Fato registrado com sucesso na sua memória episódica persistente.\n\n"
                    f"- **Chave**: `{stored_key}`\n"
                    f"- **Categoria**: `{stored_cat}`\n"
                    f"- **Conteúdo**: \"{stored_val}\"\n\n"
                    f"Esta informação já está consolidada no seu perfil e pronta para recuperação instantânea."
                )
                return AssistantTurnResult(
                    response_text=response_text,
                    model_used="internal-memory-router",
                    provider_used="builtin",
                    tools_executed=tools_executed,
                    latency_ms=0.0,
                    tokens_prompt=0,
                    tokens_completion=0,
                    cost_usd=0.0,
                )

        # 2. Check for episodic memory resolution shortcuts (e.g. "sua memória")
        lowered_stripped = user_message.strip().lower()
        if lowered_stripped in [
            "sua memória", "sua memoria", "na sua memória", "na sua memoria",
            "da sua memória", "da sua memoria", "consulte sua memória", "consulte sua memoria",
            "olhe sua memória", "olhe na sua memória", "o que você tem na memória", "o que tem na sua memória"
        ]:
            ref_topic = ""
            if history:
                for msg in reversed(history):
                    if msg.role == "user" and msg.content.strip().lower() not in [
                        "sua memória", "sua memoria", "na sua memória", "na sua memoria"
                    ]:
                        ref_topic = msg.content
                        break

            recalled = self.memory.recall_facts(query=ref_topic if ref_topic else None, limit=5)
            if recalled:
                tools_executed.append({
                    "tool": "recall_memory",
                    "arguments": {"query": ref_topic, "limit": 5},
                    "output": [f.model_dump() for f in recalled],
                    "is_error": False,
                })
                bullet_lines = "\n".join([f"- **{f.key}** ({f.category}): {f.value}" for f in recalled])
                response_text = (
                    f"Consultando diretamente a sua memória episódica, recuperei as seguintes informações registradas:\n\n"
                    f"{bullet_lines}\n\n"
                    f"Estas informações estão consolidadas na sua base pessoal de conhecimento."
                )
                return AssistantTurnResult(
                    response_text=response_text,
                    model_used="internal-memory-router",
                    provider_used="builtin",
                    tools_executed=tools_executed,
                    latency_ms=0.0,
                    tokens_prompt=0,
                    tokens_completion=0,
                    cost_usd=0.0,
                )

        # 3. Check for Dark Factory status shortcut
        lowered = user_message.lower()
        if "status da dark factory" in lowered or "status do darkhub" in lowered:
            status_res = await self.darkfac.get_status()
            executed = {"tool": "check_dark_factory_status", "result": status_res.model_dump()}
            tools_executed.append(executed)
            st_text = "🟢 **Online**" if status_res.online else f"🔴 **Offline** ({status_res.error or 'Falha de rede'})"
            response_text = (
                f"A Dark Factory está operacional e o DarkHub está {st_text}. "
                f"A orquestração autônoma e as rotas de telemetria estão disponíveis para processamento de demandas.\n\n"
                f"### Detalhes de Telemetria\n"
                f"- **Endpoint**: `{status_res.url}`\n"
                f"- **Status HTTP**: `{status_res.status_code or 200}`\n"
                f"- **Ecosistema**: Handoffs e hooks sincronizados para execução contínua."
            )
            return AssistantTurnResult(
                response_text=response_text,
                model_used="internal-intent-router",
                provider_used="builtin",
                tools_executed=tools_executed,
                latency_ms=0.0,
                tokens_prompt=0,
                tokens_completion=0,
                cost_usd=0.0,
            )

        # 4. Inject memory brief (includes any newly saved fact from step 1)
        query_topic = question_followup if question_followup else user_message
        memory_brief = self.memory.build_context_brief(query_topic)
        system_content = SYSTEM_PROMPT
        if memory_brief:
            system_content = f"{SYSTEM_PROMPT}\n\n{memory_brief}"

        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=system_content)
        ]
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_message))

        # Check budget circuit breaker
        budget_st = self.telemetry.get_budget_status()
        if budget_st.circuit_breaker_active:
            logger.warning("Budget limit active. Enforcing fallback to local model.")
            if not model or model != self.config.default_local_model:
                model = self.config.default_local_model
                provider = "ollama"

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

            # Record turn token telemetry
            self.telemetry.record_usage(
                model=resp.model,
                provider=resp.provider,
                prompt_tokens=resp.tokens_prompt,
                completion_tokens=resp.tokens_completion,
                latency_ms=resp.latency_ms,
                task_tag="chat",
            )

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
                        # Preflight deterministic safety guardrail check
                        preflight = self.guardrail.validate_tool_call(fn_name, fn_args)
                        if not preflight.allowed:
                            self.audit.record_event(
                                HarnessAuditEntry(
                                    action_type=f"mcp_tool_call:{fn_name}",
                                    target=str(fn_args)[:200],
                                    risk_level=preflight.risk_level,
                                    allowed=False,
                                    reasons=preflight.reasons,
                                    status="blocked_by_harness",
                                )
                            )
                            tools_executed.append({
                                "tool": fn_name,
                                "arguments": fn_args,
                                "output": {
                                    "status": "blocked_by_safety_harness",
                                    "reasons": preflight.reasons,
                                    "suggested_action": preflight.suggested_action,
                                },
                                "is_error": True,
                            })
                            continue

                        t_res = await self.mcp.execute_tool(fn_name, fn_args)
                        self.audit.record_event(
                            HarnessAuditEntry(
                                action_type=f"mcp_tool_call:{fn_name}",
                                target=str(fn_args)[:200],
                                risk_level=preflight.risk_level,
                                allowed=True,
                                status="executed" if not t_res.is_error else "error",
                            )
                        )
                        tools_executed.append({
                            "tool": fn_name,
                            "arguments": fn_args,
                            "output": t_res.output,
                            "is_error": t_res.is_error,
                        })

                if tools_executed:
                    tool_context_blocks = []
                    for t in tools_executed:
                        tool_context_blocks.append(
                            f"=== Resultado da ferramenta '{t['tool']}' ===\n"
                            f"{json.dumps(t['output'], ensure_ascii=False, indent=2)}"
                        )
                    tool_names = [t.get("tool", "") for t in tools_executed]
                    is_memory_tool = any(tn in ["store_memory_fact", "recall_memory", "log_decision"] for tn in tool_names)
                    assistant_msg = "Acessando a memória episódica persistente..." if is_memory_tool else "Consultando o acervo do Segundo Cérebro..."
                    if is_memory_tool:
                        user_directive = (
                            f"[Evidências e registros da Memória Episódica]:\n"
                            f"{chr(10).join(tool_context_blocks)}\n\n"
                            f"Com base nos dados da memória episódica acima, responda à solicitação do usuário com clareza, "
                            f"fornecendo a informação solicitada ou confirmando o armazenamento com precisão."
                        )
                    else:
                        user_directive = (
                            f"[Evidências e trechos reais retornados pelas ferramentas]:\n"
                            f"{chr(10).join(tool_context_blocks)}\n\n"
                            f"Com base exclusiva nos dados acima, responda à pergunta do usuário com precisão, "
                            f"citando as fontes, nomes de arquivos e seções encontradas."
                        )
                    messages.append(ChatMessage(
                        role="assistant",
                        content=resp.text or assistant_msg,
                    ))
                    messages.append(ChatMessage(
                        role="user",
                        content=user_directive,
                    ))
                    try:
                        final_resp = await self.models.generate(
                            messages=messages,
                            model=model,
                            provider=provider,
                            tools=None,
                        )
                        response_text = final_resp.text
                        self.telemetry.record_usage(
                            model=final_resp.model,
                            provider=final_resp.provider,
                            prompt_tokens=final_resp.tokens_prompt,
                            completion_tokens=final_resp.tokens_completion,
                            latency_ms=final_resp.latency_ms,
                            task_tag="chat_synthesis",
                        )
                    except Exception as exc:
                        logger.warning("Second-turn synthesis failed: %s", exc)
                        if not response_text:
                            response_text = f"Ação executada com sucesso: `{tools_executed[0].get('tool')}`."

            try:
                self.memory.record_episode(
                    summary=f"User: {user_message[:100]} | Jarvis: {response_text[:120]}",
                    tags=["chat", resp.model],
                )
            except Exception as exc:
                logger.debug("Could not record episodic note: %s", exc)

            return AssistantTurnResult(
                response_text=response_text,
                model_used=resp.model,
                provider_used=resp.provider,
                tools_executed=tools_executed,
                latency_ms=resp.latency_ms,
                tokens_prompt=resp.tokens_prompt,
                tokens_completion=resp.tokens_completion,
                cost_usd=resp.cost_usd,
            )
        except Exception as exc:
            logger.error("Chat generation failed: %s", exc)
            return AssistantTurnResult(
                response_text=f"Desculpe, ocorreu uma falha ao contatar o provedor de IA: {exc}",
                model_used="error",
                provider_used="none",
                tools_executed=[],
                latency_ms=0.0,
                tokens_prompt=0,
                tokens_completion=0,
                cost_usd=0.0,
            )
