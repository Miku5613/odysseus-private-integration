from __future__ import annotations

import sys
from typing import Any

from src.agent_runtime.knowledge_chat_bridge import run_knowledge_chat_turn


def _frames_are_not_done_only(frames: list[str]) -> bool:
    return not any(frame.strip() == "data: [DONE]" for frame in frames)


def _metric(metrics: dict[str, Any], key: str) -> Any:
    if key not in metrics:
        raise AssertionError(f"Missing metric: {key}")
    return metrics[key]


def _assert_common_governance(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    tracker = _metric(metrics, "evidence_tracker")
    hitl_policy = _metric(metrics, "hitl_policy")
    fallback_contract = _metric(metrics, "legacy_fallback_contract")

    assert tracker["version"] == "evidence_tracker_v1"
    assert hitl_policy["version"] == "hitl_policy_v1"
    assert fallback_contract["version"] == "legacy_fallback_contract_v1"

    assert fallback_contract["hitl_policy"] == hitl_policy
    assert fallback_contract["evidence_summary"] == hitl_policy["evidence_summary"]
    assert tracker["route"]["intent"] == hitl_policy["intent"]
    assert hitl_policy["intent"] == fallback_contract["intent"]

    return tracker, hitl_policy, fallback_contract


def _case_clarification() -> None:
    final, frames, events, metrics = run_knowledge_chat_turn("Read the doc.")
    tracker, hitl_policy, fallback_contract = _assert_common_governance(metrics)

    assert "I need a bit more context" in final
    assert _frames_are_not_done_only(frames)
    assert any(event.get("tool") == "knowledge_decision" for event in events)

    assert tracker["route"]["intent"] == "needs_clarification"
    assert tracker["human_gate"]["required"] is True
    assert tracker["fallback"]["used"] is True
    assert tracker["final"]["answer_mode"] == "clarification"

    assert hitl_policy["action"] == "ask_user"
    assert hitl_policy["requires_human"] is True
    assert hitl_policy["should_execute_graph"] is False

    assert fallback_contract["ownership"] == "human_gate"
    assert fallback_contract["ask_user"] is True
    assert fallback_contract["delegate_to_legacy"] is False
    assert fallback_contract["execute_langgraph"] is False
    assert fallback_contract["block"] is False


def _case_blocked_mutation() -> None:
    final, frames, events, metrics = run_knowledge_chat_turn("Delete my daily notes from Ideaverse.")
    tracker, hitl_policy, fallback_contract = _assert_common_governance(metrics)

    assert final
    assert _frames_are_not_done_only(frames)
    assert any(event.get("tool") == "knowledge_decision" for event in events)

    assert tracker["route"]["intent"] == "unsupported_mutation"
    assert tracker["human_gate"]["required"] is True
    assert tracker["final"]["answer_mode"] == "clarification"

    assert hitl_policy["action"] == "block"
    assert hitl_policy["should_block"] is True
    assert hitl_policy["should_execute_graph"] is False

    assert fallback_contract["ownership"] == "blocked"
    assert fallback_contract["block"] is True
    assert fallback_contract["delegate_to_legacy"] is False
    assert fallback_contract["execute_langgraph"] is False
    assert fallback_contract["ask_user"] is False


def _case_research_delegates_boundary() -> None:
    final, frames, events, metrics = run_knowledge_chat_turn("Search the web and compare my note with latest docs.")
    tracker, hitl_policy, fallback_contract = _assert_common_governance(metrics)

    assert final
    assert _frames_are_not_done_only(frames)
    assert any(event.get("tool") == "knowledge_decision" for event in events)

    assert tracker["route"]["intent"] == "knowledge_research"
    assert hitl_policy["action"] == "defer"
    assert hitl_policy["should_execute_graph"] is False

    assert fallback_contract["ownership"] == "legacy"
    assert fallback_contract["delegate_to_legacy"] is True
    assert fallback_contract["execute_langgraph"] is False
    assert fallback_contract["block"] is False
    assert "legacy_capability_boundary" in fallback_contract["requirements"]


def _case_high_confidence_read_contract_visible_without_live_execution() -> None:
    # This eval checks the governance decision shape through the router-backed
    # policy/contract modules indirectly via the bridge. It intentionally uses a
    # likely high-confidence note read phrase, but does not require Obsidian REST
    # availability to pass. If the read graph cannot produce live observations,
    # the governance objects must still be present and internally consistent.
    final, frames, events, metrics = run_knowledge_chat_turn("Read my Ideaverse note about Odysseus.")
    tracker, hitl_policy, fallback_contract = _assert_common_governance(metrics)

    assert final
    assert _frames_are_not_done_only(frames)
    assert any(event.get("tool") == "knowledge_decision" for event in events)

    assert tracker["route"]["intent"] in {"knowledge_read", "needs_clarification"}
    assert hitl_policy["intent"] == tracker["route"]["intent"]
    assert fallback_contract["intent"] == hitl_policy["intent"]

    if tracker["route"]["intent"] == "knowledge_read":
        assert hitl_policy["action"] == "execute_read_graph"
        assert fallback_contract["ownership"] == "langgraph"
        assert fallback_contract["execute_langgraph"] is True
        assert fallback_contract["delegate_to_legacy"] is False
    else:
        assert hitl_policy["action"] == "ask_user"
        assert fallback_contract["ownership"] == "human_gate"
        assert fallback_contract["ask_user"] is True


CASES = [
    ("clarification_governance_chain", _case_clarification),
    ("blocked_mutation_governance_chain", _case_blocked_mutation),
    ("research_legacy_boundary", _case_research_delegates_boundary),
    ("high_confidence_read_contract_visible", _case_high_confidence_read_contract_visible_without_live_execution),
]


def main() -> int:
    passed = 0
    failed = 0

    for name, case in CASES:
        try:
            case()
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
        else:
            passed += 1
            print(f"PASS  {name}")

    print("=" * 40)
    print(f"Passed: {passed}/{len(CASES)}")
    print(f"Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
