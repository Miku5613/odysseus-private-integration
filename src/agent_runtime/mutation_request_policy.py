from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from src.agent_runtime.tool_risk_mapper import gate_tool_execution


DESTRUCTIVE_REQUEST_PATTERN = re.compile(
    r"\b(?:delete|remove|purge|wipe|destroy|truncate|expunge|trash)\b",
    re.IGNORECASE,
)


def evaluate_mutation_request(message: str) -> dict[str, Any]:
    """Map a proposed mutation to the same gate used for concrete tool calls."""

    destructive = bool(DESTRUCTIVE_REQUEST_PATTERN.search(str(message or "")))
    proposed_tool = "delete_file" if destructive else "write_file"
    gate = gate_tool_execution(proposed_tool)

    if gate.decision.action == "propose_only":
        answer = (
            "I cannot execute this destructive request directly. The central "
            "policy engine requires a non-executing proposal for review. No "
            "files were deleted or changed."
        )
    elif gate.decision.action == "confirm":
        answer = (
            "I cannot execute this write automatically. The central policy "
            "engine requires explicit confirmation before a mutating tool can "
            "run. No files were changed; I can prepare a proposed edit for review."
        )
    else:
        answer = (
            "The mutation request was evaluated by the central policy engine. "
            "No tool was executed."
        )

    return {
        "proposed_tool": proposed_tool,
        "destructive": destructive,
        "allowed": gate.allowed,
        "blocked_reason": gate.blocked_reason,
        "decision": asdict(gate.decision),
        "answer": answer,
    }


__all__ = ["evaluate_mutation_request"]
