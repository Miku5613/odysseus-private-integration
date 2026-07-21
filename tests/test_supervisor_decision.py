import json

import pytest

from src.agent_runtime.supervisor_decision import (
    build_supervisor_capability_context,
    parse_supervisor_decision,
    validate_supervisor_decision,
)


def test_decision_supports_multiple_capabilities():
    decision = validate_supervisor_decision(
        {
            "selected_capabilities": [
                "knowledge_search",
                "deep_research",
                "code_inspection",
            ],
            "missing_information": [],
            "clarification_needed": False,
            "reasoning_summary": "Needs personal context, current research, and code inspection.",
        }
    )

    assert decision.selected_capabilities == (
        "knowledge_search",
        "deep_research",
        "code_inspection",
    )


def test_decision_can_select_no_capabilities():
    decision = validate_supervisor_decision(
        {
            "selected_capabilities": [],
            "missing_information": [],
            "clarification_needed": False,
            "reasoning_summary": "No external capability required.",
        }
    )

    assert decision.selected_capabilities == ()


def test_decision_deduplicates_capabilities_preserving_order():
    decision = validate_supervisor_decision(
        {
            "selected_capabilities": [
                "knowledge_search",
                "web_search",
                "knowledge_search",
            ]
        }
    )

    assert decision.selected_capabilities == (
        "knowledge_search",
        "web_search",
    )


def test_unknown_capability_is_rejected():
    with pytest.raises(ValueError, match="Unknown capability"):
        validate_supervisor_decision(
            {"selected_capabilities": ["magic_unknown_worker"]}
        )


def test_missing_information_is_generic_not_parameter_schema_locked():
    decision = validate_supervisor_decision(
        {
            "selected_capabilities": ["mcp_specialized"],
            "missing_information": ["which connected service the user intends"],
            "clarification_needed": True,
        }
    )

    assert decision.clarification_needed is True
    assert decision.missing_information == (
        "which connected service the user intends",
    )


def test_parse_requires_valid_json():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_supervisor_decision("not-json")


def test_parse_accepts_valid_json_object():
    payload = json.dumps(
        {
            "selected_capabilities": ["code_inspection"],
            "missing_information": [],
            "clarification_needed": False,
            "reasoning_summary": "Inspect the programming workspace.",
        }
    )

    decision = parse_supervisor_decision(payload)

    assert decision.selected_capabilities == ("code_inspection",)


def test_capability_context_exposes_semantics_not_keyword_rules():
    context = build_supervisor_capability_context()
    names = {item["name"] for item in context}

    assert "code_inspection" in names
    assert "knowledge_search" in names
    assert "deep_research" in names
    assert "mcp_specialized" in names

    for item in context:
        assert "keywords" not in item


def test_custom_known_capability_set_supports_future_dynamic_mcp_entries():
    decision = validate_supervisor_decision(
        {
            "selected_capabilities": ["dynamic_mcp_weather"],
        },
        known_capabilities={"dynamic_mcp_weather"},
    )

    assert decision.selected_capabilities == ("dynamic_mcp_weather",)
