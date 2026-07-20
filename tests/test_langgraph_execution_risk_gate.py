from pathlib import Path

from src.agent_runtime.tool_risk_mapper import (
    decide_tool_call_policy_action,
)


ROOT = Path(__file__).resolve().parents[1]

GRAPH = (
    ROOT / "src" / "agent_runtime" / "graph.py"
).read_text(
    encoding="utf-8",
    errors="replace",
)

LEGACY = (
    ROOT / "src" / "agent_loop.py"
).read_text(
    encoding="utf-8",
    errors="replace",
)


def test_langgraph_has_execution_time_risk_gate() -> None:
    assert (
        "LangGraph execution-time tool risk gate v1"
        in GRAPH
    )

    assert (
        "decide_tool_call_policy_action("
        in GRAPH
    )


def test_confirm_calls_do_not_reach_tool_invoke_branch() -> None:
    assert (
        '_risk_decision.action == "confirm"'
        in GRAPH
    )

    assert "CONFIRMATION_REQUIRED:" in GRAPH
    assert "tool_confirmation_required" in GRAPH


def test_destructive_calls_are_blocked() -> None:
    assert (
        '_risk_decision.action == "propose_only"'
        in GRAPH
    )

    assert "DESTRUCTIVE_ACTION_BLOCKED:" in GRAPH
    assert "tool_risk_blocked" in GRAPH


def test_safe_calls_still_reach_tool_execution() -> None:
    assert "tool_obj = TOOL_MAP.get(name)" in GRAPH
    assert "content = tool_obj.invoke(args)" in GRAPH


def test_action_aware_read_is_allowed() -> None:
    decision = decide_tool_call_policy_action(
        "manage_notes",
        {"action": "list"},
    )

    assert decision.action == "allow"


def test_action_aware_mutation_requires_confirmation() -> None:
    decision = decide_tool_call_policy_action(
        "manage_notes",
        {"action": "add"},
    )

    assert decision.action == "confirm"


def test_action_aware_delete_is_proposal_only() -> None:
    decision = decide_tool_call_policy_action(
        "manage_notes",
        {"action": "delete"},
    )

    assert decision.action == "propose_only"


def test_legacy_has_no_langgraph_execution_gate() -> None:
    assert (
        "LangGraph execution-time tool risk gate v1"
        not in LEGACY
    )

    assert (
        "decide_tool_call_policy_action("
        not in LEGACY
    )

    assert "tool_confirmation_required" not in LEGACY
    assert "tool_risk_blocked" not in LEGACY
