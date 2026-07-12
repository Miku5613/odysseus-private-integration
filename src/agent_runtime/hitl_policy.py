from __future__ import annotations

from typing import Any


POLICY_VERSION = "hitl_policy_v1"


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


def _clarifying_options(decision: Any) -> list[str]:
    ctx = _decision_value(decision, "clarifying_context", {}) or {}
    if isinstance(ctx, dict):
        return _as_list(ctx.get("options"))
    return []


def _compact_reason(value: str) -> str:
    text = " ".join(str(value or "").split())
    return text or "No policy reason recorded."


def build_clarification_question(decision: Any) -> str:
    """Return a concise user-facing clarification question.

    This is intentionally deterministic. It avoids executing any graph or
    guessing a source when the router says human input is needed.
    """

    ambiguity = _as_list(_decision_value(decision, "ambiguity", []))
    options = _clarifying_options(decision)
    reason = _compact_reason(str(_decision_value(decision, "reason", "")))

    lines = ["I need one detail before I continue."]

    if reason:
        lines.extend(["", reason])

    if ambiguity:
        lines.append("")
        lines.append("Unclear part:")
        lines.extend(f"- {item}" for item in ambiguity)

    if options:
        lines.append("")
        lines.append("Which source should I use?")
        lines.extend(f"- {item}" for item in options)
    else:
        lines.append("")
        lines.append("Which exact note, file, document, or source should I use?")

    return "\n".join(lines).strip()


def build_hitl_policy(
    decision: Any,
    *,
    evidence_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a formal HITL/fallback policy for one deterministic runtime decision.

    The policy does not execute tools. It only classifies the next safe action:
    - execute_read_graph
    - ask_user
    - block
    - defer
    - ignore_not_knowledge

    This lets graph bridges record and test the reason for asking the user,
    blocking, or falling back instead of scattering that logic through prose.
    """

    intent = str(_decision_value(decision, "intent", ""))
    confidence = str(_decision_value(decision, "confidence", ""))
    target_runtime = str(_decision_value(decision, "target_runtime", ""))
    reason = _compact_reason(str(_decision_value(decision, "reason", "")))
    needs_human = bool(_decision_value(decision, "needs_human", False))
    is_read_only = bool(_decision_value(decision, "is_read_only", True))
    write_allowed = bool(_decision_value(decision, "write_allowed", False))

    allowed_graphs = _as_list(_decision_value(decision, "allowed_graphs", []))
    blocked_graphs = _as_list(_decision_value(decision, "blocked_graphs", []))
    ambiguity = _as_list(_decision_value(decision, "ambiguity", []))
    evidence_summary = dict(evidence_summary or {})

    action = "defer"
    requires_human = False
    should_execute_graph = False
    should_block = False
    fallback_target = target_runtime
    answer_mode = intent or "unknown"
    sufficiency_status = "not_applicable"
    question = ""

    if intent == "unsupported_mutation":
        action = "block"
        requires_human = False
        should_execute_graph = False
        should_block = True
        fallback_target = "blocked"
        answer_mode = "blocked"
        sufficiency_status = "blocked"
    elif intent == "knowledge_read" and confidence == "high" and is_read_only and not write_allowed:
        action = "execute_read_graph"
        should_execute_graph = True
        requires_human = False
        fallback_target = ""
        answer_mode = "knowledge_read"
        sufficiency_status = "pending_read_evidence"
    elif needs_human or intent == "needs_clarification":
        action = "ask_user"
        requires_human = True
        should_execute_graph = False
        fallback_target = "human_in_the_loop"
        answer_mode = "clarification"
        sufficiency_status = "insufficient"
        question = build_clarification_question(decision)
    elif intent in {"knowledge_propose", "knowledge_research"}:
        action = "defer"
        requires_human = False
        should_execute_graph = False
        fallback_target = target_runtime
        answer_mode = intent
        sufficiency_status = "deferred"
    elif intent == "not_knowledge":
        action = "ignore_not_knowledge"
        requires_human = False
        should_execute_graph = False
        fallback_target = target_runtime
        answer_mode = "not_knowledge"
        sufficiency_status = "not_applicable"

    return {
        "version": POLICY_VERSION,
        "intent": intent,
        "confidence": confidence,
        "target_runtime": target_runtime,
        "action": action,
        "answer_mode": answer_mode,
        "requires_human": requires_human,
        "should_execute_graph": should_execute_graph,
        "should_block": should_block,
        "fallback_target": fallback_target,
        "sufficiency_status": sufficiency_status,
        "reason": reason,
        "question": question,
        "ambiguity": ambiguity,
        "allowed_graphs": allowed_graphs,
        "blocked_graphs": blocked_graphs,
        "read_only": is_read_only,
        "write_allowed": write_allowed,
        "evidence_summary": evidence_summary,
    }


__all__ = [
    "POLICY_VERSION",
    "build_clarification_question",
    "build_hitl_policy",
]
