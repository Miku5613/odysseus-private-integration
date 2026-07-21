import pytest

from src.agent_runtime.capability_registry import (
    build_capability_registry,
    builtin_capabilities,
    worker_capabilities,
)
from src.agent_runtime.supervisor_contracts import CapabilityDescriptor


def test_builtin_registry_contains_four_core_workers():
    registry = build_capability_registry()

    assert registry["code_inspection"].provider == "code_worker"
    assert registry["knowledge_search"].provider == "knowledge_worker"
    assert registry["deep_research"].provider == "research_worker"
    assert registry["mcp_specialized"].provider == "mcp_worker"


def test_code_worker_is_not_odysseus_specific():
    capability = build_capability_registry()["code_inspection"]

    assert "arbitrary" in capability.description.lower()
    assert capability.metadata["supports_active_doc"] is True


def test_knowledge_worker_is_semantic_and_source_agnostic():
    capability = build_capability_registry()["knowledge_search"]

    assert capability.metadata["semantic_backend"] == "chromadb"
    assert capability.metadata["source_agnostic"] is True


def test_research_worker_reuses_existing_builtin_research_tools():
    capability = build_capability_registry()["deep_research"]

    assert "trigger_research" in capability.tool_names
    assert "manage_research" in capability.tool_names


def test_mcp_worker_is_dynamic_not_bound_to_one_example():
    capability = build_capability_registry()["mcp_specialized"]

    assert capability.metadata["dynamic"] is True
    assert capability.metadata["registry_extensible"] is True
    assert "sports" not in capability.name


def test_shared_capabilities_are_not_workers():
    registry = build_capability_registry()

    assert registry["ask_user"].kind == "interaction"
    assert registry["web_search"].kind == "direct_tool"


def test_worker_capability_filter_returns_only_worker_backends():
    names = {capability.name for capability in worker_capabilities()}

    assert names == {
        "code_inspection",
        "knowledge_search",
        "deep_research",
        "mcp_specialized",
    }


def test_registry_accepts_dynamic_capabilities():
    extra = CapabilityDescriptor(
        name="mcp_example_dynamic",
        description="Example dynamically discovered MCP capability.",
        kind="worker",
        provider="mcp_worker",
        tool_names=("example_tool",),
    )

    registry = build_capability_registry((extra,))

    assert registry["mcp_example_dynamic"] == extra


def test_registry_rejects_duplicate_capability_names():
    duplicate = CapabilityDescriptor(
        name="code_inspection",
        description="Duplicate.",
        kind="worker",
        provider="other",
    )

    with pytest.raises(ValueError, match="Duplicate capability name"):
        build_capability_registry((duplicate,))


def test_builtin_capability_order_is_stable():
    names = [capability.name for capability in builtin_capabilities()]

    assert names[:4] == [
        "code_inspection",
        "knowledge_search",
        "deep_research",
        "mcp_specialized",
    ]
