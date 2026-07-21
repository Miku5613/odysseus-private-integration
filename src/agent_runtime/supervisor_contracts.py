from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


SupervisorCapabilityKind = Literal["worker", "direct_tool", "interaction"]
WorkerStatus = Literal["completed", "needs_clarification", "failed", "partial"]


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Semantic capability exposed to the general Supervisor."""

    name: str
    description: str
    kind: SupervisorCapabilityKind
    provider: str
    discoverable_inputs: tuple[str, ...] = ()
    supported_actions: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerHandoff:
    """Generic Supervisor-to-worker boundary payload."""

    capability: str
    objective: str
    context: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    source_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkerResult:
    """Normalized result returned by any specialized worker."""

    capability: str
    status: WorkerStatus
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[dict[str, Any], ...] = ()
    missing_information: tuple[str, ...] = ()
    error: str | None = None


class SupervisorState(TypedDict, total=False):
    """General orchestration state without worker-specific state pollution."""

    messages: list[Any]
    raw_query: str
    available_capabilities: list[CapabilityDescriptor]
    active_capabilities: list[str]
    completed_capabilities: list[str]
    missing_information: list[str]
    worker_context: dict[str, Any]
    worker_results: list[WorkerResult]
    final_user_narrative: str | None


__all__ = [
    "CapabilityDescriptor",
    "SupervisorCapabilityKind",
    "SupervisorState",
    "WorkerHandoff",
    "WorkerResult",
    "WorkerStatus",
]
