from __future__ import annotations

from typing import Any

from .hitl_policy import build_hitl_policy


CONTRACT_VERSION = "legacy_fallback_contract_v1"


def _decision_value(decision: Any, name: str, default: Any = "") -> Any:
    if isinstance(decision, dict):
        return decision.get(name, default)
    return getattr(decision, name, default)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x)]
    if str(value):
        return [str(value)]
    return []


def _compact_reason(*parts: Any) -> str:
    text = " ".join(" ".join(str(part or "").split()) for part in parts if str(part or "").strip())
    return text or "No fallback contract reason recorded."


def build_legacy_fallback_contract(
    decision: Any,
    *,
    hitl_policy: dict[str, Any] | None = None,
    evidence_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify runtime ownership for one turn.

    The contract is intentionally descriptive-only. It does not call tools,
    execute graphs, or send a response. It only declares which runtime should
    own the next step and why.

    Ownership values:
    - langgraph: safe deterministic graph lane can handle it
    - human_gate: ask the user for missing source/scope
    - blocked: do not proceed
    - legacy: delegate to legacy runtime
    """

    hitl_policy = dict(hitl_policy or build_hitl_policy(decision, evidence_summary=evidence_summary))
    evidence_summary = dict(evidence_summary or hitl_policy.get("evidence_summary") or {})

    intent = str(_decision_value(decision, "intent", hitl_policy.get("intent", "")))
    target_runtime = str(_decision_value(decision, "target_runtime", hitl_policy.get("target_runtime", "")))
    confidence = str(_decision_value(decision, "confidence", hitl_policy.get("confidence", "")))
    reason = _compact_reason(
        _decision_value(decision, "reason", ""),
        hitl_policy.get("reason", ""),
    )

    action = str(hitl_policy.get("action", "defer"))
    allowed_graphs = _as_list(_decision_value(decision, "allowed_graphs", hitl_policy.get("allowed_graphs", [])))
    blocked_graphs = _as_list(_decision_value(decision, "blocked_graphs", hitl_policy.get("blocked_graphs", [])))

    ownership = "legacy"
    next_runtime = "legacy_runtime"
    delegate_to_legacy = True
    execute_langgraph = False
    ask_user = False
    block = False
    risk_level = "medium"
    fallback_reason = reason
    requirements: list[str] = []

    if action == "execute_read_graph":
        ownership = "langgraph"
        next_runtime = target_runtime or "obsidian_read_graph"
        delegate_to_legacy = False
        execute_langgraph = True
        ask_user = False
        block = False
        risk_level = "low"
        requirements = ["read_only_graph", "evidence_recording"]
        fallback_reason = "Safe high-confidence read graph route."
    elif action == "ask_user":
        ownership = "human_gate"
        next_runtime = "human_in_the_loop"
        delegate_to_legacy = False
        execute_langgraph = False
        ask_user = True
        block = False
        risk_level = "low"
        requirements = ["clarify_source_or_scope"]
        fallback_reason = "Human clarification is required before any runtime owns execution."
    elif action == "block":
        ownership = "blocked"
        next_runtime = "blocked"
        delegate_to_legacy = False
        execute_langgraph = False
        ask_user = False
        block = True
        risk_level = "high"
        requirements = ["explicit_write_profile", "preview", "confirmation", "backup_or_rollback"]
        fallback_reason = "Blocked instead of delegating because the request is an unsupported mutation."
    elif intent in {"knowledge_research", "knowledge_propose"}:
        ownership = "legacy"
        next_runtime = target_runtime or "legacy_runtime"
        delegate_to_legacy = True
        execute_langgraph = False
        ask_user = False
        block = False
        risk_level = "medium"
        requirements = ["legacy_capability_boundary", "tool_safety_gate"]
        fallback_reason = "LangGraph does not yet own this research/proposal lane."
    elif intent == "not_knowledge":
        ownership = "legacy"
        next_runtime = target_runtime or "legacy_runtime"
        delegate_to_legacy = True
        execute_langgraph = False
        ask_user = False
        block = False
        risk_level = "low"
        requirements = ["general_runtime_or_legacy_default"]
        fallback_reason = "Request is outside the knowledge graph lane."

    return {
        "version": CONTRACT_VERSION,
        "intent": intent,
        "confidence": confidence,
        "target_runtime": target_runtime,
        "hitl_action": action,
        "ownership": ownership,
        "next_runtime": next_runtime,
        "delegate_to_legacy": delegate_to_legacy,
        "execute_langgraph": execute_langgraph,
        "ask_user": ask_user,
        "block": block,
        "risk_level": risk_level,
        "reason": fallback_reason,
        "requirements": requirements,
        "allowed_graphs": allowed_graphs,
        "blocked_graphs": blocked_graphs,
        "hitl_policy": hitl_policy,
        "evidence_summary": evidence_summary,
    }


__all__ = [
    "CONTRACT_VERSION",
    "build_legacy_fallback_contract",
]
