from dataclasses import FrozenInstanceError

import pytest

from src.agent_runtime.supervisor_contracts import (
    CapabilityDescriptor,
    WorkerHandoff,
    WorkerResult,
)


def test_capability_descriptor_supports_worker_capability():
    capability = CapabilityDescriptor(
        name="code_inspection",
        description="Inspect arbitrary programming codebases.",
        kind="worker",
        provider="code_worker",
        discoverable_inputs=("workspace", "code_target"),
        supported_actions=("inspect", "trace", "explain"),
        tool_names=("grep", "read_file", "read_file_around"),
    )

    assert capability.name == "code_inspection"
    assert capability.kind == "worker"
    assert "code_target" in capability.discoverable_inputs


def test_capability_descriptor_is_immutable():
    capability = CapabilityDescriptor(
        name="knowledge_search",
        description="Search indexed semantic knowledge.",
        kind="worker",
        provider="knowledge_worker",
    )

    with pytest.raises(FrozenInstanceError):
        capability.name = "changed"


def test_worker_handoff_is_domain_neutral():
    handoff = WorkerHandoff(
        capability="knowledge_search",
        objective="Find relevant project decisions.",
        context={"query": "Atlas write mode"},
    )

    assert handoff.context["query"] == "Atlas write mode"
    assert not hasattr(handoff, "claim_support")
    assert not hasattr(handoff, "code_location")


def test_worker_result_does_not_require_code_evidence():
    result = WorkerResult(
        capability="deep_research",
        status="completed",
        summary="Research completed.",
        data={"research_id": "example"},
    )

    assert result.status == "completed"
    assert result.evidence == ()


def test_worker_result_can_request_clarification():
    result = WorkerResult(
        capability="mcp_specialized",
        status="needs_clarification",
        summary="More information is required.",
        missing_information=("target_service",),
    )

    assert result.missing_information == ("target_service",)


def test_worker_result_supports_specialist_evidence():
    result = WorkerResult(
        capability="code_inspection",
        status="completed",
        summary="Implementation located.",
        evidence=(
            {
                "kind": "code_location",
                "file": "routes/chat_routes.py",
                "line": 100,
            },
        ),
    )

    assert result.evidence[0]["kind"] == "code_location"
