"""Deterministic preflight safety guardrails for the agent harness (Milestone 4)."""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from jarvis.core.harness.models import (
    ExecutionRiskLevel,
    PreflightCheckResult,
    SafetyPolicy,
)

logger = logging.getLogger("jarvis.core.harness.guardrail")


class HarnessSafetyGuardrail:
    """Evaluates the risk of tools, code snippets, and paths before execution."""

    def __init__(self, policy: Optional[SafetyPolicy] = None) -> None:
        self.policy = policy or SafetyPolicy()

    def validate_path(self, target_path: str | Path, is_write: bool = False) -> PreflightCheckResult:
        """Inspect a file path against blast-radius boundaries and path-traversal attacks."""
        raw_str = str(target_path).strip()
        reasons: List[str] = []

        # 1. Check for directory traversal sequences
        if ".." in raw_str:
            reasons.append(f"Tentativa de path traversal detectada ('..'): '{raw_str}'")

        try:
            resolved = Path(raw_str).resolve()
            resolved_str = str(resolved).lower()
        except Exception as exc:
            return PreflightCheckResult(
                allowed=False,
                risk_level=ExecutionRiskLevel.HIGH,
                reasons=[f"Caminho inválido ou inacessível: {exc}"],
                suggested_action="Forneça um caminho absoluto ou relativo válido.",
            )

        # 2. Check against strictly forbidden paths
        for forbidden in self.policy.forbidden_paths:
            try:
                forb_resolved = str(Path(forbidden).resolve()).lower()
                if resolved_str == forb_resolved or resolved_str.startswith(forb_resolved + os.sep) or resolved_str.startswith(forb_resolved + "/"):
                    reasons.append(f"Acesso bloqueado: o caminho '{resolved}' está sob o diretório protegido '{forbidden}'")
            except Exception:
                if forbidden.lower() in resolved_str:
                    reasons.append(f"Acesso bloqueado por conter padrão proibido '{forbidden}'")

        # 3. If it is a write operation, verify it is inside allowed write roots
        if is_write and not reasons:
            is_allowed_root = False
            for allowed_root in self.policy.allowed_write_roots:
                try:
                    root_resolved = str(Path(allowed_root).resolve()).lower()
                    if resolved_str == root_resolved or resolved_str.startswith(root_resolved + os.sep) or resolved_str.startswith(root_resolved + "/"):
                        is_allowed_root = True
                        break
                except Exception:
                    continue

            if not is_allowed_root:
                reasons.append(
                    f"Tentativa de escrita fora dos diretórios permitidos do workspace: '{resolved}'"
                )

        if reasons:
            return PreflightCheckResult(
                allowed=False,
                risk_level=ExecutionRiskLevel.HIGH,
                reasons=reasons,
                suggested_action="Limite as operações de escrita ao workspace do projeto ou diretório temporário.",
            )

        risk = ExecutionRiskLevel.MEDIUM if is_write else ExecutionRiskLevel.SAFE
        return PreflightCheckResult(
            allowed=True,
            risk_level=risk,
            reasons=[],
            suggested_action="Operação permitida.",
        )

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> PreflightCheckResult:
        """Inspect MCP tool invocation before execution."""
        lowered_tool = tool_name.lower()

        # Read-only search and recollection tools are SAFE
        safe_read_tools = {
            "search_second_brain",
            "read_second_brain_note",
            "recall_memory",
            "list_bizops_tasks",
            "list_pending_bizops_actions",
            "list_recent_meetings",
            "get_meeting_details",
            "search_meetings",
            "check_dark_factory_status",
            "validate_execution_safety",
            "get_harness_audit_log",
        }
        if lowered_tool in safe_read_tools:
            return PreflightCheckResult(
                allowed=True,
                risk_level=ExecutionRiskLevel.SAFE,
                reasons=[],
                suggested_action="Executar ferramenta de leitura.",
            )

        # Inspect any path argument passed to any tool
        for key, val in arguments.items():
            if isinstance(val, str) and ("path" in key.lower() or "file" in key.lower() or "dir" in key.lower()):
                # If tool modifies or writes
                is_write = "write" in lowered_tool or "delete" in lowered_tool or "store" in lowered_tool
                path_check = self.validate_path(val, is_write=is_write)
                if not path_check.allowed:
                    return path_check

        # Commands inspection
        for key, val in arguments.items():
            if isinstance(val, str):
                for cmd in self.policy.forbidden_commands:
                    if re.search(rf"\b{re.escape(cmd)}\b", val, re.IGNORECASE):
                        return PreflightCheckResult(
                            allowed=False,
                            risk_level=ExecutionRiskLevel.HIGH,
                            reasons=[f"Argumento '{key}' contém comando proibido: '{cmd}'"],
                            suggested_action="Remova comandos destrutivos do argumento.",
                        )

        # Modifying internal state (memory, bizops, demands)
        medium_tools = {
            "store_memory_fact",
            "log_decision",
            "record_failed_approach",
            "trigger_bizops_task",
            "approve_bizops_action",
            "sync_meetings_to_second_brain",
            "dispatch_meeting_demands",
            "create_dark_factory_demand",
        }
        if lowered_tool in medium_tools:
            return PreflightCheckResult(
                allowed=True,
                risk_level=ExecutionRiskLevel.MEDIUM,
                reasons=[],
                suggested_action="Operação modificadora governada.",
            )

        # Default fallback for unknown or external tools: MEDIUM
        return PreflightCheckResult(
            allowed=True,
            risk_level=ExecutionRiskLevel.MEDIUM,
            reasons=[],
            suggested_action="Ferramenta verificada.",
        )

    def validate_python_syntax(self, code: str) -> PreflightCheckResult:
        """Statically inspect Python code using AST before running in sandbox."""
        try:
            tree = ast.parse(code)
        except SyntaxError as syn_err:
            return PreflightCheckResult(
                allowed=False,
                risk_level=ExecutionRiskLevel.HIGH,
                reasons=[f"Erro de sintaxe Python na linha {syn_err.lineno}: {syn_err.msg}"],
                suggested_action="Corrija a sintaxe do código antes da execução.",
            )

        reasons: List[str] = []
        for node in ast.walk(tree):
            # Check for forbidden function calls or imports
            if isinstance(node, ast.Call):
                call_repr = ""
                if isinstance(node.func, ast.Attribute):
                    val_id = getattr(node.func.value, "id", "")
                    call_repr = f"{val_id}.{node.func.attr}"
                elif isinstance(node.func, ast.Name):
                    call_repr = node.func.id

                for forbidden_call in self.policy.forbidden_ast_calls:
                    if call_repr == forbidden_call or call_repr.endswith(f".{forbidden_call}"):
                        reasons.append(f"Chamada de função perigosa bloqueada: '{call_repr}'")

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("ctypes",):
                        reasons.append(f"Importação de módulo de baixo nível bloqueada: '{alias.name}'")

            elif isinstance(node, ast.ImportFrom):
                if node.module in ("ctypes",):
                    reasons.append(f"Importação de módulo de baixo nível bloqueada: '{node.module}'")

        if reasons:
            return PreflightCheckResult(
                allowed=False,
                risk_level=ExecutionRiskLevel.HIGH,
                reasons=reasons,
                suggested_action="Remova chamadas de sistema, subprocessos ou bibliotecas de memória direta.",
            )

        return PreflightCheckResult(
            allowed=True,
            risk_level=ExecutionRiskLevel.SAFE,
            reasons=[],
            suggested_action="Código sintaticamente seguro para sandbox.",
        )
