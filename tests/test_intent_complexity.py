from __future__ import annotations

import asyncio
import json

import pytest

import src.agent_runtime.intent_complexity as intent_complexity


def _conversation_payload():
    return {
        "execution_class": "conversation",
        "goals": [],
        "needs_planning": False,
        "requested_sources": ["model_knowledge"],
        "excluded_sources": [],
        "missing_information": [],
        "clarification_needed": False,
        "reasoning_summary": (
            "No external execution is required."
        ),
    }


def _single_payload():
    return {
        "execution_class": "single_graph",
        "goals": [
            {
                "id": "goal_1",
                "objective": "Perform one bounded execution unit.",
                "work_domains": [
                    "persisted_state"
                ],
                "requires_external_state": True,
                "requires_mutation": False,
            }
        ],
        "needs_planning": False,
        "requested_sources": ["authorized_workspace_files"],
        "excluded_sources": [],
        "missing_information": [],
        "clarification_needed": False,
        "reasoning_summary": (
            "One bounded execution unit is sufficient."
        ),
    }


def _multi_payload():
    return {
        "execution_class": "multi_graph",
        "goals": [
            {
                "id": "goal_1",
                "objective": (
                    "Coordinate multiple distinct state systems."
                ),
                "work_domains": [
                    "workspace_knowledge",
                    "calendar_state",
                    "cross_source_prioritization",
                ],
                "requires_external_state": True,
                "requires_mutation": False,
            }
        ],
        "needs_planning": True,
        "requested_sources": [
            "authorized_workspace_files",
            "personal_memory",
        ],
        "excluded_sources": [],
        "missing_information": [],
        "clarification_needed": False,
        "reasoning_summary": (
            "Dependency-aware planning is required."
        ),
    }


def test_conversation_is_zero_execution():
    result = (
        intent_complexity
        .validate_intent_complexity_payload(
            _conversation_payload()
        )
    )

    assert result.execution_class == "conversation"
    assert result.goals == ()
    assert result.needs_planning is False


def test_conversation_allows_descriptive_nonexecution_goal():
    payload = _conversation_payload()
    payload["goals"] = [
        {
            "id": "goal_1",
            "objective": (
                "Explain a general concept from model knowledge."
            ),
            "work_domains": [
                "conceptual_explanation"
            ],
            "requires_external_state": False,
            "requires_mutation": False,
        }
    ]

    result = (
        intent_complexity
        .validate_intent_complexity_payload(
            payload
        )
    )

    assert result.execution_class == "conversation"
    assert len(result.goals) == 1
    assert result.goals[0].work_domains == (
        "conceptual_explanation",
    )


def test_conversation_rejects_execution_bearing_goal():
    payload = _conversation_payload()
    payload["goals"] = [
        {
            "id": "goal_1",
            "objective": "Retrieve persisted knowledge.",
            "work_domains": [
                "persisted_knowledge"
            ],
            "requires_external_state": True,
            "requires_mutation": False,
        }
    ]

    with pytest.raises(ValueError):
        (
            intent_complexity
            .validate_intent_complexity_payload(
                payload
            )
        )


def test_single_graph_requires_goal_without_capability_selection():
    result = (
        intent_complexity
        .validate_intent_complexity_payload(
            _single_payload()
        )
    )

    assert result.execution_class == "single_graph"
    assert result.needs_planning is False
    assert result.goals[0].work_domains == (
        "persisted_state",
    )

    payload = _single_payload()
    payload["goals"] = []

    with pytest.raises(ValueError):
        (
            intent_complexity
            .validate_intent_complexity_payload(
                payload
            )
        )


def test_single_graph_must_not_require_task_planner():
    payload = _single_payload()
    payload["needs_planning"] = True

    with pytest.raises(ValueError):
        (
            intent_complexity
            .validate_intent_complexity_payload(
                payload
            )
        )


def test_multi_graph_requires_task_planning():
    result = (
        intent_complexity
        .validate_intent_complexity_payload(
            _multi_payload()
        )
    )

    assert result.execution_class == "multi_graph"
    assert result.needs_planning is True

    payload = _multi_payload()
    payload["needs_planning"] = False

    with pytest.raises(ValueError):
        (
            intent_complexity
            .validate_intent_complexity_payload(
                payload
            )
        )


def test_work_domains_are_backend_independent_semantic_labels():
    payload = _multi_payload()
    payload["goals"][0]["work_domains"] = [
        "future_system_not_in_langgraph_registry",
        "legacy_only_state_domain",
        "cross_source_synthesis",
    ]

    result = (
        intent_complexity
        .validate_intent_complexity_payload(
            payload
        )
    )

    assert result.goals[0].work_domains == (
        "future_system_not_in_langgraph_registry",
        "legacy_only_state_domain",
        "cross_source_synthesis",
    )


def test_prompt_owns_complexity_not_capability_selection():
    prompt = (
        intent_complexity
        .build_intent_complexity_prompt(
            "Assess this request."
        )
    )

    assert "conversation:" in prompt
    assert "single_graph:" in prompt
    assert "multi_graph:" in prompt
    assert "Do not use keyword matching" in prompt

    assert "AVAILABLE_CAPABILITIES:" not in prompt
    assert "selected_capabilities" not in prompt
    assert "capability_actions" not in prompt
    assert "tool_budget_hint" not in prompt

    assert "Missing native LangGraph coverage" in prompt
    assert "Legacy runtime" in prompt
    assert "must NEVER downgrade multi_graph" in prompt
    assert "Descriptive work_domains do not by themselves imply" in prompt
    assert "specific prior, saved, generated, or persisted artifact" in prompt
    assert "Multiple independently acquired provenance" in prompt
    assert "acquire A + acquire B + combine A and B" in prompt
    assert "Semantic topic count is not the same thing" in prompt
    assert "source-of-truth or execution surface" in prompt
    assert "Count acquisition boundaries, not topic labels" in prompt
    assert "persisted/private state are a distinct acquisition domain" in prompt
    assert "Possessive or deictic references to user-owned state" in prompt
    assert "Execution class is determined by prerequisite acquisition" in prompt
    assert "one report or one answer" in prompt
    assert "does not know a filename" in prompt
    assert "authorized_workspace_files" in prompt
    assert "Never substitute public_web" in prompt
    assert "Do not treat unsupplied repository evidence" in prompt


def test_model_call_reuses_canonical_candidate_chain(
    monkeypatch,
):
    captured = {}

    async def fake_fallback(
        candidates,
        messages,
        **kwargs,
    ):
        captured["candidates"] = candidates
        captured["messages"] = messages
        captured["kwargs"] = kwargs

        return json.dumps(
            _conversation_payload()
        )

    monkeypatch.setattr(
        intent_complexity,
        "llm_call_async_with_fallback",
        fake_fallback,
    )

    candidates = [
        (
            "http://primary.example/v1/chat/completions",
            "primary-model",
            {"X-Test": "primary"},
        ),
        (
            "http://fallback.example/v1/chat/completions",
            "fallback-model",
            {"X-Test": "fallback"},
        ),
    ]

    result = asyncio.run(
        intent_complexity.assess_intent_complexity(
            "Have a normal conversation.",
            llm_candidates=candidates,
            prompt_type="test",
            max_tokens=1200,
        )
    )

    assert result.execution_class == "conversation"
    assert captured["candidates"] == candidates
    assert captured["kwargs"]["temperature"] == 0.0
    assert captured["kwargs"]["max_tokens"] == 3000
    assert captured["kwargs"]["prompt_type"] == "test"
    validator = captured["kwargs"]["response_validator"]
    assert callable(validator)
    assert validator(
        json.dumps(_conversation_payload())
    ).execution_class == "conversation"


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (3000, 3000),
        (3500, 3500),
        (5000, 4096),
    ],
)
def test_model_output_token_bounds(
    monkeypatch,
    requested,
    expected,
):
    captured = {}

    async def fake_fallback(
        candidates,
        messages,
        **kwargs,
    ):
        captured["kwargs"] = kwargs
        return json.dumps(
            _conversation_payload()
        )

    monkeypatch.setattr(
        intent_complexity,
        "llm_call_async_with_fallback",
        fake_fallback,
    )

    asyncio.run(
        intent_complexity.assess_intent_complexity(
            "Have a normal conversation.",
            llm_candidates=[
                (
                    "http://example/v1/chat/completions",
                    "model",
                    {},
                )
            ],
            max_tokens=requested,
        )
    )

    assert captured["kwargs"]["max_tokens"] == expected


def test_missing_information_is_deduplicated():
    payload = _single_payload()
    payload["missing_information"] = [
        "calendar authorization",
        "calendar authorization",
    ]

    result = (
        intent_complexity
        .validate_intent_complexity_payload(
            payload
        )
    )

    assert result.missing_information == (
        "calendar authorization",
    )


def test_assessment_retries_structured_chain_after_validation_exhaustion(
    monkeypatch,
):
    calls = []

    async def fake_fallback(
        candidates,
        *,
        messages,
        response_validator=None,
        **kwargs,
    ):
        calls.append(messages[0]["content"])
        response = (
            "not valid json"
            if len(calls) == 1
            else json.dumps(_conversation_payload())
        )
        if response_validator is not None:
            response_validator(response)
        return response

    monkeypatch.setattr(
        intent_complexity,
        "llm_call_async_with_fallback",
        fake_fallback,
    )

    result = asyncio.run(
        intent_complexity.assess_intent_complexity(
            "Explain dependency graphs conceptually.",
            llm_candidates=[
                ("url", "model", {}),
            ],
        )
    )

    assert result.execution_class == "conversation"
    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert "STRUCTURED OUTPUT RETRY:" in calls[1]
