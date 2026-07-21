import json

import pytest

from src.agent_runtime.supervisor_contracts import CapabilityDescriptor
from src.agent_runtime.supervisor_reasoner import (
    build_supervisor_prompt,
    reason_about_capabilities,
)


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeModel:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)

        if isinstance(self.payload, str):
            return FakeResponse(self.payload)

        return FakeResponse(json.dumps(self.payload))


def test_prompt_exposes_all_four_core_workers():
    prompt = build_supervisor_prompt("Inspect this project.")

    assert "code_inspection" in prompt
    assert "knowledge_search" in prompt
    assert "deep_research" in prompt
    assert "mcp_specialized" in prompt


def test_prompt_explicitly_allows_multiple_capabilities():
    prompt = build_supervisor_prompt("Compare my notes with current research.")

    assert "zero, one, or multiple capabilities" in prompt
    assert "Do not force every request into exactly one worker" in prompt


def test_prompt_does_not_encode_example_keyword_routes():
    prompt = build_supervisor_prompt("Example request.")

    assert 'if "note"' not in prompt
    assert 'if "football"' not in prompt


def test_reasoner_returns_multi_capability_decision():
    model = FakeModel(
        {
            "selected_capabilities": [
                "knowledge_search",
                "deep_research",
            ],
            "missing_information": [],
            "clarification_needed": False,
            "reasoning_summary": "Needs personal knowledge and current research.",
        }
    )

    decision = reason_about_capabilities(
        model,
        "Compare my project notes with current research.",
    )

    assert decision.selected_capabilities == (
        "knowledge_search",
        "deep_research",
    )
    assert decision.clarification_needed is False
    assert len(model.prompts) == 1


def test_reasoner_can_return_no_capabilities():
    model = FakeModel(
        {
            "selected_capabilities": [],
            "missing_information": [],
            "clarification_needed": False,
            "reasoning_summary": "No external capability required.",
        }
    )

    decision = reason_about_capabilities(model, "Hello there.")

    assert decision.selected_capabilities == ()


def test_reasoner_preserves_generic_missing_information():
    model = FakeModel(
        {
            "selected_capabilities": ["knowledge_search"],
            "missing_information": ["which of several equally plausible targets is intended"],
            "clarification_needed": True,
            "reasoning_summary": "Target remains materially ambiguous.",
        }
    )

    decision = reason_about_capabilities(model, "Summarize that.")

    assert decision.clarification_needed is True
    assert decision.missing_information == (
        "which of several equally plausible targets is intended",
    )


def test_reasoner_rejects_unknown_model_capability():
    model = FakeModel(
        {
            "selected_capabilities": ["invented_magic_worker"],
        }
    )

    with pytest.raises(ValueError, match="Unknown capability"):
        reason_about_capabilities(model, "Do something.")


def test_reasoner_accepts_dynamic_capability_registry_entry():
    extra = CapabilityDescriptor(
        name="dynamic_mcp_weather",
        description="Retrieve specialized weather data from a connected MCP.",
        kind="worker",
        provider="mcp_worker",
        supported_actions=("weather_lookup",),
        tool_names=("weather_lookup",),
    )

    model = FakeModel(
        {
            "selected_capabilities": ["dynamic_mcp_weather"],
            "missing_information": [],
            "clarification_needed": False,
            "reasoning_summary": "Connected specialized capability matches the request.",
        }
    )

    decision = reason_about_capabilities(
        model,
        "Check the specialized weather service.",
        extra_capabilities=(extra,),
    )

    assert decision.selected_capabilities == ("dynamic_mcp_weather",)
    assert "dynamic_mcp_weather" in model.prompts[0]


def test_reasoner_accepts_plain_string_model_response():
    model = FakeModel(
        json.dumps(
            {
                "selected_capabilities": ["code_inspection"],
                "missing_information": [],
                "clarification_needed": False,
                "reasoning_summary": "Programming workspace inspection required.",
            }
        )
    )

    decision = reason_about_capabilities(
        model,
        "Find where this function is implemented.",
    )

    assert decision.selected_capabilities == ("code_inspection",)
