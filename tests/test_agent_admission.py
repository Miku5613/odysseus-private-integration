from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.agent_runtime import agent_admission


def _assessment(
    execution_class: str,
):
    return SimpleNamespace(
        execution_class=execution_class,
    )


def test_langgraph_disabled_preserves_legacy_agent_without_model_call(
    monkeypatch,
):
    calls = []

    async def forbidden(
        *args,
        **kwargs,
    ):
        calls.append(
            (args, kwargs)
        )
        raise AssertionError(
            "Disabled LangGraph must not invoke semantic admission"
        )

    monkeypatch.setattr(
        agent_admission,
        "assess_intent_complexity",
        forbidden,
    )

    result = asyncio.run(
        agent_admission.assess_explicit_agent_admission(
            "Inspect this workspace.",
            langgraph_enabled=False,
            llm_candidates=[
                ("url", "model", {}),
            ],
        )
    )

    assert result.path == "legacy_agent"
    assert result.assessment is None
    assert calls == []


def test_conversation_class_reuses_normal_chat(
    monkeypatch,
):
    captured = {}

    async def fake(
        query,
        *,
        llm_candidates,
        prompt_type=None,
        max_tokens=1400,
        active_workspace=False,
        workspace_label=None,
    ):
        captured["query"] = query
        captured["candidates"] = llm_candidates
        captured["prompt_type"] = prompt_type
        return _assessment(
            "conversation"
        )

    monkeypatch.setattr(
        agent_admission,
        "assess_intent_complexity",
        fake,
    )

    candidates = [
        (
            "primary",
            "selected",
            {"Authorization": "x"},
        ),
        (
            "fallback",
            "backup",
            {},
        ),
    ]

    result = asyncio.run(
        agent_admission.assess_explicit_agent_admission(
            "Explain a binary tree.",
            langgraph_enabled=True,
            llm_candidates=candidates,
            prompt_type="default",
        )
    )

    assert result.path == "normal_chat"
    assert result.use_normal_chat is True
    assert captured["candidates"] is candidates
    assert captured["prompt_type"] == "default"


def test_single_graph_class_is_preserved_for_existing_agent_runtime(
    monkeypatch,
):
    async def fake(
        *args,
        **kwargs,
    ):
        return _assessment(
            "single_graph"
        )

    monkeypatch.setattr(
        agent_admission,
        "assess_intent_complexity",
        fake,
    )

    result = asyncio.run(
        agent_admission.assess_explicit_agent_admission(
            "Retrieve my latest saved deep research report.",
            langgraph_enabled=True,
            llm_candidates=[
                ("url", "model", {}),
            ],
        )
    )

    assert result.path == "single_graph"
    assert result.use_single_graph is True
    assert result.use_normal_chat is False


def test_multi_graph_class_is_preserved_for_p3c5b_planning(
    monkeypatch,
):
    async def fake(
        *args,
        **kwargs,
    ):
        return _assessment(
            "multi_graph"
        )

    monkeypatch.setattr(
        agent_admission,
        "assess_intent_complexity",
        fake,
    )

    result = asyncio.run(
        agent_admission.assess_explicit_agent_admission(
            "Compare my notes, inspect the project, then write a report.",
            langgraph_enabled=True,
            llm_candidates=[
                ("url", "model", {}),
            ],
        )
    )

    assert result.path == "multi_graph"
    assert result.use_multi_graph is True
    assert result.use_normal_chat is False


def test_unknown_execution_class_is_rejected(
    monkeypatch,
):
    async def fake(
        *args,
        **kwargs,
    ):
        return _assessment(
            "unsupported"
        )

    monkeypatch.setattr(
        agent_admission,
        "assess_intent_complexity",
        fake,
    )

    try:
        asyncio.run(
            agent_admission.assess_explicit_agent_admission(
                "Do something.",
                langgraph_enabled=True,
                llm_candidates=[
                    ("url", "model", {}),
                ],
            )
        )
    except ValueError as exc:
        assert (
            "Unsupported Intent/Complexity execution_class"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Unknown execution class was accepted"
        )
