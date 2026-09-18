"""Policy Guardrails and Autonomy Level Governance for BizOps."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from jarvis.core.bizops.models import AutonomyLevel, PendingAction

logger = logging.getLogger("jarvis.core.bizops.guardrails")

# Classification of actions by risk tier
READ_ONLY_ACTIONS = {
    "read_second_brain_note",
    "search_second_brain",
    "recall_memory",
    "query_decisions",
    "check_dark_factory_status",
    "inspect_dark_factory_backlog",
    "generate_briefing",
}

MODIFYING_ACTIONS = {
    "create_dark_factory_demand",
    "store_memory_fact",
    "delete_memory_fact",
    "log_decision",
    "record_failed_approach",
    "external_webhook_post",
    "execute_system_command",
}


class PolicyGuardrail:
    """Enforces fail-closed autonomy boundaries and creates HITL review checkpoints."""

    def __init__(self, default_level: AutonomyLevel = AutonomyLevel.LEVEL_2_HITL) -> None:
        self.default_level = default_level

    def evaluate_action(
        self,
        action_type: str,
        title: str,
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
        effective_level: Optional[AutonomyLevel] = None,
    ) -> Tuple[bool, str, Optional[PendingAction]]:
        """Evaluate an action before execution.
        
        Returns:
            (allowed_immediately, reason, pending_action_if_blocked)
        """
        level = effective_level or self.default_level

        # 1. Read-only actions are always allowed immediately
        if action_type in READ_ONLY_ACTIONS:
            return True, "Ação somente leitura aprovada automaticamente (Nível 1).", None

        # 2. Level 3 Autonomous Execution: allowed if in standard modifying whitelist
        if level >= AutonomyLevel.LEVEL_3_AUTONOMOUS:
            if action_type in MODIFYING_ACTIONS:
                return True, f"Ação '{action_type}' aprovada sob política autônoma (Nível 3).", None

        # 3. Level 2 HITL: Modifying actions must be paused and presented for human approval
        pending = PendingAction(
            task_id=task_id,
            action_type=action_type,
            title=title,
            description=f"Ação com efeito colateral aguardando autorização do operador humano: {action_type}",
            payload=payload,
            autonomy_level_required=AutonomyLevel.LEVEL_2_HITL,
            status="pending_approval",
        )
        msg = (
            f"Ação '{action_type}' exige autorização humana sob política Nível 2 (HITL). "
            f"Pendente de aprovação sob ID: {pending.id}."
        )
        return False, msg, pending
