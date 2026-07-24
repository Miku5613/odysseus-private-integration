from __future__ import annotations

import asyncio

import pytest

from src.agent_runtime.step_action_resolver import (
    parse_capability_action_response,
    resolve_capability_action_async,
)
from src.agent_runtime.supervisor_contracts import CapabilityDescriptor


class AsyncModel:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        async def value():
            return self.response

        return value()


def _descriptor(actions=("start", "read")):
    return CapabilityDescriptor(
        name="deep_research",
        description="Start or retrieve persistent research.",
        kind="worker",
        provider="test",
        supported_actions=actions,
    )


def test_action_resolver_selects_only_advertised_action():
    model = AsyncModel({
        "action": "read",
        "clarification_needed": False,
        "missing_information": [],
        "reasoning_summary": "Retrieve the existing artifact.",
    })

    result = asyncio.run(
        resolve_capability_action_async(
            "Retrieve the latest saved report.",
            _descriptor(),
            model=model,
            context={
                "dependency_outputs": {
                    "find": {
                        "summary": "A report exists.",
                    }
                }
            },
        )
    )

    assert result.action == "read"
    assert result.clarification_needed is False
    assert "completed dependency outputs" in model.prompts[0].lower()


def test_action_resolver_skips_model_when_capability_has_no_actions():
    class Forbidden:
        def invoke(self, prompt):
            raise AssertionError("Model must not be called")

    result = asyncio.run(
        resolve_capability_action_async(
            "Inspect code.",
            _descriptor(actions=()),
            model=Forbidden(),
        )
    )

    assert result.action is None


def test_action_parser_rejects_unadvertised_action():
    with pytest.raises(ValueError, match="Unsupported action"):
        parse_capability_action_response(
            {
                "action": "delete",
                "clarification_needed": False,
                "missing_information": [],
                "reasoning_summary": "",
            },
            capability=_descriptor(),
        )
