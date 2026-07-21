from __future__ import annotations

from collections.abc import Iterable

from src.agent_runtime.supervisor_contracts import CapabilityDescriptor


_BUILTIN_CAPABILITIES: tuple[CapabilityDescriptor, ...] = (
    CapabilityDescriptor(
        name="code_inspection",
        description=(
            "Inspect, understand, trace, explain, and reason about arbitrary "
            "programming codebases and coding workspaces. Use for source-code "
            "location, implementation analysis, debugging context, architecture "
            "inspection, and coding workflows."
        ),
        kind="worker",
        provider="code_worker",
        discoverable_inputs=(
            "workspace",
            "code_target",
            "file",
            "symbol",
            "implementation_goal",
        ),
        supported_actions=(
            "inspect",
            "search",
            "trace",
            "explain",
            "analyze",
            "prepare_code_output",
        ),
        metadata={
            "supports_active_doc": True,
            "specialist_evidence": True,
        },
    ),
    CapabilityDescriptor(
        name="knowledge_search",
        description=(
            "Semantically retrieve and synthesize indexed knowledge from the "
            "current authorized workspace, vault, notes collection, or other "
            "ChromaDB-backed knowledge source."
        ),
        kind="worker",
        provider="knowledge_worker",
        discoverable_inputs=(
            "topic",
            "note_title",
            "path",
            "project",
            "date",
            "workspace",
        ),
        supported_actions=(
            "search",
            "retrieve",
            "list",
            "read",
            "summarize",
            "synthesize",
        ),
        metadata={
            "semantic_backend": "chromadb",
            "source_agnostic": True,
        },
    ),
    CapabilityDescriptor(
        name="deep_research",
        description=(
            "Start, inspect, manage, retrieve, and reuse structured deep research "
            "using the built-in research runtime and its persisted research results."
        ),
        kind="worker",
        provider="research_worker",
        discoverable_inputs=(
            "research_question",
            "research_id",
            "topic",
            "existing_research",
        ),
        supported_actions=(
            "trigger_research",
            "manage_research",
            "retrieve_research",
            "reuse_research_context",
        ),
        tool_names=(
            "trigger_research",
            "manage_research",
        ),
    ),
    CapabilityDescriptor(
        name="mcp_specialized",
        description=(
            "Use specialized capabilities exposed by connected MCP servers when "
            "a request maps naturally to an external or domain-specific MCP tool. "
            "This capability is semantic and extensible; it is not tied to any "
            "single MCP server, sport, service, or example."
        ),
        kind="worker",
        provider="mcp_worker",
        discoverable_inputs=(
            "requested_capability",
            "target_service",
            "domain",
            "resource",
        ),
        supported_actions=(
            "discover_mcp_capability",
            "invoke_mcp_tool",
            "combine_mcp_results",
        ),
        metadata={
            "dynamic": True,
            "registry_extensible": True,
        },
    ),
    CapabilityDescriptor(
        name="ask_user",
        description=(
            "Ask one focused clarification question when ambiguity cannot be "
            "safely or reasonably resolved through available discovery capabilities."
        ),
        kind="interaction",
        provider="supervisor",
        supported_actions=("clarify",),
        tool_names=("ask_user",),
    ),
    CapabilityDescriptor(
        name="web_search",
        description=(
            "Search current public web information as a shared capability that may "
            "support multiple workers or direct Supervisor reasoning."
        ),
        kind="direct_tool",
        provider="shared_tools",
        discoverable_inputs=("query", "topic", "url"),
        supported_actions=("search", "retrieve_current_information"),
        tool_names=("web_search",),
        metadata={"shared": True},
    ),
)


def builtin_capabilities() -> tuple[CapabilityDescriptor, ...]:
    """Return the immutable built-in Supervisor capability descriptors."""
    return _BUILTIN_CAPABILITIES


def build_capability_registry(
    extra_capabilities: Iterable[CapabilityDescriptor] = (),
) -> dict[str, CapabilityDescriptor]:
    """Build a capability registry, allowing dynamic providers such as MCP.

    Duplicate names are rejected instead of silently replacing an existing
    capability. Dynamic discovery must therefore choose stable unique names.
    """

    registry: dict[str, CapabilityDescriptor] = {}

    for capability in (*_BUILTIN_CAPABILITIES, *tuple(extra_capabilities)):
        if capability.name in registry:
            raise ValueError(f"Duplicate capability name: {capability.name}")

        registry[capability.name] = capability

    return registry


def worker_capabilities(
    registry: dict[str, CapabilityDescriptor] | None = None,
) -> tuple[CapabilityDescriptor, ...]:
    """Return capabilities backed by specialized workers."""

    source = registry or build_capability_registry()
    return tuple(
        capability
        for capability in source.values()
        if capability.kind == "worker"
    )


__all__ = [
    "build_capability_registry",
    "builtin_capabilities",
    "worker_capabilities",
]
