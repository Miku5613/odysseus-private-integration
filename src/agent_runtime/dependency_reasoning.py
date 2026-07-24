from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any, Awaitable, Protocol

from src.agent_runtime.supervisor_contracts import (
    CapabilityDescriptor,
    WorkerHandoff,
    WorkerResult,
)


DEPENDENCY_REASONING_CAPABILITY = "dependency_reasoning"


class DependencyReasoningModel(Protocol):
    """Minimal model seam for dependency-only PlanStep reasoning."""

    def invoke(self, prompt: str) -> Any | Awaitable[Any]:
        ...


def dependency_reasoning_capability() -> CapabilityDescriptor:
    """Return the coordinator-scoped capability descriptor.

    This capability is intentionally not part of the global Supervisor registry.
    It exists only inside the multi-graph execution catalog so a dependent
    synthesis/comparison step can reason over already-completed PlanStep outputs
    without pretending to retrieve new external state.
    """

    return CapabilityDescriptor(
        name=DEPENDENCY_REASONING_CAPABILITY,
        description=(
            "Reason, compare, reconcile, synthesize, or decide using only "
            "already-completed PlanStep dependency outputs. This surface does "
            "not retrieve new external state, call source systems, or mutate."
        ),
        kind="worker",
        provider="multi_graph_coordinator",
        discoverable_inputs=("dependency_outputs",),
        supported_actions=(),
        metadata={
            "supports_external_state": False,
            "supports_mutation": False,
            "execution_priority": -100,
            "internal_multi_graph": True,
        },
    )


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()

    return str(response or "").strip()


def build_dependency_reasoning_prompt(
    objective: str,
    dependency_outputs: Mapping[str, Any],
) -> str:
    """Build one grounded reasoning prompt from completed dependency outputs."""

    normalized_objective = str(objective or "").strip()
    if not normalized_objective:
        raise ValueError(
            "Dependency reasoning requires a non-empty PlanStep objective"
        )

    normalized_dependencies = dict(dependency_outputs or {})
    if not normalized_dependencies:
        raise ValueError(
            "Dependency reasoning requires completed dependency outputs"
        )

    return "\n".join([
        "You are the dependency-reasoning execution surface of Odysseus.",
        "",
        "Complete only the current PlanStep using the validated dependency ",
        "outputs below.",
        "Do not retrieve new external information, call tools, invent missing ",
        "evidence, or claim that a dependency succeeded unless its payload says ",
        "it completed.",
        "When dependency evidence is incomplete, say exactly what remains ",
        "unsupported instead of filling the gap.",
        "Return only the user-facing result for this PlanStep.",
        "",
        "PLAN STEP OBJECTIVE:",
        normalized_objective,
        "",
        "COMPLETED DEPENDENCY OUTPUTS:",
        json.dumps(
            normalized_dependencies,
            ensure_ascii=False,
            default=str,
        ),
    ])


async def run_dependency_reasoning_worker(
    handoff: WorkerHandoff,
    *,
    model: DependencyReasoningModel,
    session_id: str | None = None,
) -> WorkerResult:
    """Execute a no-new-source PlanStep from completed dependency outputs."""

    if handoff.capability != DEPENDENCY_REASONING_CAPABILITY:
        raise ValueError(
            "Dependency reasoning worker requires "
            f"capability={DEPENDENCY_REASONING_CAPABILITY}"
        )

    dependency_outputs = handoff.context.get(
        "dependency_outputs",
        {},
    )
    if not isinstance(dependency_outputs, Mapping) or not dependency_outputs:
        return WorkerResult(
            capability=DEPENDENCY_REASONING_CAPABILITY,
            status="needs_clarification",
            summary=(
                "Dependency reasoning cannot run until its prerequisite "
                "PlanSteps have produced completed outputs."
            ),
            missing_information=(
                "completed dependency outputs",
            ),
        )

    prompt = build_dependency_reasoning_prompt(
        handoff.objective,
        dependency_outputs,
    )
    response = model.invoke(prompt)
    if inspect.isawaitable(response):
        response = await response

    result_text = _response_text(response)
    if not result_text:
        return WorkerResult(
            capability=DEPENDENCY_REASONING_CAPABILITY,
            status="failed",
            summary="Dependency reasoning produced no result.",
            data={
                "dependency_ids": list(
                    dependency_outputs
                ),
            },
            error=(
                "Dependency reasoning model returned an empty response."
            ),
        )

    return WorkerResult(
        capability=DEPENDENCY_REASONING_CAPABILITY,
        status="completed",
        summary=result_text,
        data={
            "dependency_ids": list(
                dependency_outputs
            ),
            "uses_external_state": False,
        },
    )


def build_dependency_reasoning_runner(
    model: DependencyReasoningModel,
) -> Callable[..., Awaitable[WorkerResult]]:
    """Bind the current conversational model to the coordinator-scoped worker."""

    async def run(
        handoff: WorkerHandoff,
        *,
        session_id: str | None = None,
    ) -> WorkerResult:
        return await run_dependency_reasoning_worker(
            handoff,
            model=model,
            session_id=session_id,
        )

    return run


__all__ = [
    "DEPENDENCY_REASONING_CAPABILITY",
    "DependencyReasoningModel",
    "build_dependency_reasoning_prompt",
    "build_dependency_reasoning_runner",
    "dependency_reasoning_capability",
    "run_dependency_reasoning_worker",
]
