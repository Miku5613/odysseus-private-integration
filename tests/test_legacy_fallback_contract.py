from __future__ import annotations

from src.agent_runtime.hitl_policy import build_hitl_policy
from src.agent_runtime.knowledge_chat_bridge import run_knowledge_chat_turn
from src.agent_runtime.knowledge_graph_router import evaluate_knowledge_request
from src.agent_runtime.legacy_fallback_contract import (
    CONTRACT_VERSION,
    build_legacy_fallback_contract,
)


def test_contract_assigns_high_confidence_read_to_langgraph():
    decision = evaluate_knowledge_request("Read my Ideaverse note about Odysseus.")

    contract = build_legacy_fallback_contract(decision)

    assert contract["version"] == CONTRACT_VERSION
    assert contract["intent"] == "knowledge_read"
    assert contract["ownership"] == "langgraph"
    assert contract["execute_langgraph"] is True
    assert contract["delegate_to_legacy"] is False
    assert contract["ask_user"] is False
    assert contract["block"] is False
    assert contract["risk_level"] == "low"
    assert "read_only_graph" in contract["requirements"]


def test_contract_assigns_vague_read_to_human_gate():
    decision = evaluate_knowledge_request("Read the doc.")

    contract = build_legacy_fallback_contract(decision)

    assert contract["intent"] == "needs_clarification"
    assert contract["ownership"] == "human_gate"
    assert contract["next_runtime"] == "human_in_the_loop"
    assert contract["ask_user"] is True
    assert contract["delegate_to_legacy"] is False
    assert contract["execute_langgraph"] is False
    assert contract["block"] is False
    assert "clarify_source_or_scope" in contract["requirements"]


def test_contract_blocks_unsupported_mutation_instead_of_legacy_delegate():
    decision = evaluate_knowledge_request("Delete my daily notes from Ideaverse.")

    contract = build_legacy_fallback_contract(decision)

    assert contract["intent"] == "unsupported_mutation"
    assert contract["ownership"] == "blocked"
    assert contract["next_runtime"] == "blocked"
    assert contract["block"] is True
    assert contract["delegate_to_legacy"] is False
    assert contract["execute_langgraph"] is False
    assert contract["ask_user"] is False
    assert contract["risk_level"] == "high"
    assert "backup_or_rollback" in contract["requirements"]


def test_contract_delegates_research_lane_to_legacy_boundary():
    decision = evaluate_knowledge_request("Search the web and compare my note with latest docs.")

    contract = build_legacy_fallback_contract(decision)

    assert contract["intent"] == "knowledge_research"
    assert contract["ownership"] == "legacy"
    assert contract["delegate_to_legacy"] is True
    assert contract["execute_langgraph"] is False
    assert contract["ask_user"] is False
    assert contract["block"] is False
    assert "legacy_capability_boundary" in contract["requirements"]


def test_contract_delegates_non_knowledge_to_legacy_default():
    decision = {
        "intent": "not_knowledge",
        "confidence": "medium",
        "target_runtime": "legacy_runtime",
        "reason": "Outside knowledge graph scope.",
        "needs_human": False,
        "allowed_graphs": [],
        "blocked_graphs": [],
        "is_read_only": True,
        "write_allowed": False,
    }

    contract = build_legacy_fallback_contract(decision)

    assert contract["intent"] == "not_knowledge"
    assert contract["ownership"] == "legacy"
    assert contract["next_runtime"] == "legacy_runtime"
    assert contract["delegate_to_legacy"] is True
    assert contract["risk_level"] == "low"
    assert "general_runtime_or_legacy_default" in contract["requirements"]


def test_contract_accepts_existing_hitl_policy_and_evidence_summary():
    decision = evaluate_knowledge_request("Read the doc.")
    policy = build_hitl_policy(decision)
    summary = {"intent": "needs_clarification", "needs_human": True}

    contract = build_legacy_fallback_contract(
        decision,
        hitl_policy=policy,
        evidence_summary=summary,
    )

    assert contract["hitl_policy"] == policy
    assert contract["evidence_summary"] == summary
    assert contract["ownership"] == "human_gate"


def test_knowledge_chat_bridge_metrics_include_legacy_contract_for_clarification():
    final, frames, events, metrics = run_knowledge_chat_turn("Read the doc.")

    contract = metrics.get("legacy_fallback_contract")

    assert "I need a bit more context" in final
    assert contract
    assert contract["version"] == CONTRACT_VERSION
    assert contract["intent"] == "needs_clarification"
    assert contract["ownership"] == "human_gate"
    assert contract["ask_user"] is True
    assert contract["delegate_to_legacy"] is False
    assert contract["execute_langgraph"] is False
    assert contract["block"] is False
    assert contract["hitl_policy"]["action"] == "ask_user"
    assert contract["evidence_summary"]["needs_human"] is True
    assert any(ev.get("tool") == "knowledge_decision" for ev in events)
    assert not any(frame.strip() == "data: [DONE]" for frame in frames)


def test_knowledge_chat_bridge_metrics_include_legacy_contract_for_blocked_mutation():
    final, frames, events, metrics = run_knowledge_chat_turn("Delete my daily notes from Ideaverse.")

    contract = metrics.get("legacy_fallback_contract")

    assert contract
    assert contract["version"] == CONTRACT_VERSION
    assert contract["intent"] == "unsupported_mutation"
    assert contract["ownership"] == "blocked"
    assert contract["block"] is True
    assert contract["delegate_to_legacy"] is False
    assert contract["execute_langgraph"] is False
    assert contract["ask_user"] is False
    assert contract["hitl_policy"]["action"] == "block"
    assert contract["evidence_summary"]["intent"] == "unsupported_mutation"
    assert any(ev.get("tool") == "knowledge_decision" for ev in events)
    assert not any(frame.strip() == "data: [DONE]" for frame in frames)
