from __future__ import annotations

from src.agent_runtime.hitl_policy import (
    POLICY_VERSION,
    build_clarification_question,
    build_hitl_policy,
)
from src.agent_runtime.knowledge_graph_router import evaluate_knowledge_request


def test_hitl_policy_allows_high_confidence_read_graph():
    decision = evaluate_knowledge_request("Read my Ideaverse note about Odysseus.")

    policy = build_hitl_policy(decision)

    assert policy["version"] == POLICY_VERSION
    assert policy["intent"] == "knowledge_read"
    assert policy["action"] == "execute_read_graph"
    assert policy["should_execute_graph"] is True
    assert policy["requires_human"] is False
    assert policy["answer_mode"] == "knowledge_read"
    assert policy["sufficiency_status"] == "pending_read_evidence"


def test_hitl_policy_asks_user_for_vague_doc():
    decision = evaluate_knowledge_request("Read the doc.")

    policy = build_hitl_policy(decision)

    assert policy["intent"] == "needs_clarification"
    assert policy["action"] == "ask_user"
    assert policy["requires_human"] is True
    assert policy["should_execute_graph"] is False
    assert policy["fallback_target"] == "human_in_the_loop"
    assert policy["answer_mode"] == "clarification"
    assert policy["sufficiency_status"] == "insufficient"
    assert "Which source should I use?" in policy["question"]
    assert "Ideaverse/Obsidian note" in policy["question"]


def test_hitl_policy_blocks_destructive_knowledge_mutation():
    decision = evaluate_knowledge_request("Delete my daily notes from Ideaverse.")

    policy = build_hitl_policy(decision)

    assert policy["intent"] == "unsupported_mutation"
    assert policy["action"] == "block"
    assert policy["should_block"] is True
    assert policy["should_execute_graph"] is False
    assert policy["fallback_target"] == "blocked"
    assert policy["sufficiency_status"] == "blocked"


def test_hitl_policy_defers_research_without_graph_execution():
    decision = evaluate_knowledge_request("Search the web and compare my note with latest docs.")

    policy = build_hitl_policy(decision)

    assert policy["intent"] == "knowledge_research"
    assert policy["action"] == "defer"
    assert policy["should_execute_graph"] is False
    assert policy["requires_human"] is False
    assert policy["sufficiency_status"] == "deferred"


def test_clarification_question_has_fallback_when_options_missing():
    decision = {
        "intent": "needs_clarification",
        "confidence": "low",
        "target_runtime": "human_in_the_loop",
        "reason": "The target is ambiguous.",
        "needs_human": True,
        "ambiguity": ["missing note name"],
        "clarifying_context": {},
        "allowed_graphs": [],
        "blocked_graphs": ["obsidian_read_graph"],
        "is_read_only": True,
        "write_allowed": False,
    }

    question = build_clarification_question(decision)

    assert "I need one detail" in question
    assert "missing note name" in question
    assert "Which exact note, file, document, or source should I use?" in question
