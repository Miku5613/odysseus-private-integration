from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Iterable, Literal, Protocol

from src.agent_runtime.capability_registry import (
    build_capability_registry,
)
from src.agent_runtime.supervisor_contracts import (
    CapabilityDescriptor,
)
from src.agent_runtime.task_plan import PlanStep
from src.agent_runtime.worker_dispatcher import (
    builtin_worker_runners,
)


ExecutionSurfaceKind = Literal[
    "langgraph",
    "legacy",
    "registered",
]

ResolutionStatus = Literal[
    "resolved",
    "clarification",
    "unsupported",
]


@dataclass(frozen=True)
class ExecutionSurfaceDescriptor:
    """One executable surface visible to the step resolver.

    This descriptor belongs to the execution layer, not the planner. A PlanStep
    remains backend-independent and never stores one of these descriptors.
    """

    id: str
    kind: ExecutionSurfaceKind
    description: str
    capability_name: str | None = None
    supports_external_state: bool = True
    supports_mutation: bool = False
    priority: int = 0


@dataclass(frozen=True)
class ExecutionCompatibilityAssessment:
    """Model-proposed semantic compatibility, before deterministic selection."""

    compatible_surface_ids: tuple[str, ...]
    clarification_needed: bool = False
    missing_information: tuple[str, ...] = ()
    reasoning_summary: str = ""


@dataclass(frozen=True)
class ExecutionResolution:
    """Validated execution-surface resolution for one semantic PlanStep."""

    step_id: str
    status: ResolutionStatus
    selected_surface: ExecutionSurfaceDescriptor | None
    compatible_surface_ids: tuple[str, ...] = ()
    rejected_surface_ids: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    reasoning_summary: str = ""


class ExecutionResolverModel(Protocol):
    """Minimal model seam for semantic surface compatibility."""

    def invoke(self, prompt: str) -> Any | Awaitable[Any]:
        ...


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")

    return normalized


def _clean_string_tuple(
    value: Any,
    *,
    field: str,
) -> tuple[str, ...]:
    if value is None:
        return ()

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")

    result: list[str] = []
    seen: set[str] = set()

    for item in value:
        normalized = _required_text(
            item,
            field=f"{field} item",
        )
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)

    return tuple(result)


def _validate_surface(
    surface: ExecutionSurfaceDescriptor,
) -> ExecutionSurfaceDescriptor:
    surface_id = _required_text(
        surface.id,
        field="Execution surface id",
    )
    description = _required_text(
        surface.description,
        field=f"Execution surface {surface_id} description",
    )

    if surface.kind not in {
        "langgraph",
        "legacy",
        "registered",
    }:
        raise ValueError(
            f"Unsupported execution surface kind: {surface.kind}"
        )

    capability_name = surface.capability_name

    if capability_name is not None:
        capability_name = _required_text(
            capability_name,
            field=(
                f"Execution surface {surface_id} "
                "capability_name"
            ),
        )

    if (
        surface.kind == "langgraph"
        and capability_name is None
    ):
        raise ValueError(
            f"LangGraph surface {surface_id} "
            "requires capability_name"
        )

    if not isinstance(surface.priority, int):
        raise ValueError(
            f"Execution surface {surface_id} priority "
            "must be an integer"
        )

    return ExecutionSurfaceDescriptor(
        id=surface_id,
        kind=surface.kind,
        description=description,
        capability_name=capability_name,
        supports_external_state=bool(
            surface.supports_external_state
        ),
        supports_mutation=bool(
            surface.supports_mutation
        ),
        priority=surface.priority,
    )


def _native_surface_from_capability(
    capability: CapabilityDescriptor,
) -> ExecutionSurfaceDescriptor:
    metadata = dict(capability.metadata or {})

    return ExecutionSurfaceDescriptor(
        id=f"langgraph:{capability.name}",
        kind="langgraph",
        description=capability.description,
        capability_name=capability.name,
        supports_external_state=bool(
            metadata.get(
                "supports_external_state",
                True,
            )
        ),
        supports_mutation=bool(
            metadata.get(
                "supports_mutation",
                False,
            )
        ),
        priority=int(
            metadata.get(
                "execution_priority",
                0,
            )
        ),
    )


def _legacy_surface() -> ExecutionSurfaceDescriptor:
    return ExecutionSurfaceDescriptor(
        id="legacy:agent_loop",
        kind="legacy",
        description=(
            "The existing Odysseus legacy agent loop with broad tool "
            "coverage. Selection does not bypass tool policy, confirmation, "
            "write policy, or other runtime safety gates."
        ),
        capability_name=None,
        supports_external_state=True,
        supports_mutation=True,
        priority=0,
    )


def build_execution_surface_catalog(
    *,
    extra_surfaces: Iterable[
        ExecutionSurfaceDescriptor
    ] = (),
    include_legacy: bool = True,
    capability_registry: dict[
        str,
        CapabilityDescriptor,
    ] | None = None,
    worker_runners: dict[str, Any] | None = None,
) -> tuple[ExecutionSurfaceDescriptor, ...]:
    """Build the executable surface catalog without executing anything.

    Native LangGraph surfaces are created only for capabilities that are both
    registered and backed by an implemented worker runner. Legacy is exposed as
    a deliberate execution surface. Additional registered surfaces can be
    supplied by future integrations such as external providers.
    """

    registry = (
        build_capability_registry()
        if capability_registry is None
        else dict(capability_registry)
    )
    runners = (
        builtin_worker_runners()
        if worker_runners is None
        else dict(worker_runners)
    )

    surfaces: list[ExecutionSurfaceDescriptor] = []

    for capability_name in runners:
        capability = registry.get(capability_name)
        if capability is None:
            continue

        surfaces.append(
            _native_surface_from_capability(
                capability
            )
        )

    if include_legacy:
        surfaces.append(
            _legacy_surface()
        )

    surfaces.extend(
        tuple(extra_surfaces)
    )

    validated: list[ExecutionSurfaceDescriptor] = []
    seen: set[str] = set()

    for surface in surfaces:
        normalized = _validate_surface(surface)

        if normalized.id in seen:
            raise ValueError(
                "Duplicate execution surface id: "
                + normalized.id
            )

        seen.add(normalized.id)
        validated.append(normalized)

    return tuple(validated)


def _surface_payload(
    surface: ExecutionSurfaceDescriptor,
) -> dict[str, Any]:
    return {
        "id": surface.id,
        "kind": surface.kind,
        "description": surface.description,
        "capability_name": surface.capability_name,
        "supports_external_state": (
            surface.supports_external_state
        ),
        "supports_mutation": (
            surface.supports_mutation
        ),
    }


def build_execution_resolver_prompt(
    step: PlanStep,
    surfaces: tuple[
        ExecutionSurfaceDescriptor,
        ...,
    ],
) -> str:
    """Ask the model only for semantic compatibility, not final selection."""

    if not surfaces:
        raise ValueError(
            "Execution Resolver requires at least one surface"
        )

    schema = {
        "compatible_surface_ids": [
            "surface-id"
        ],
        "clarification_needed": False,
        "missing_information": [],
        "reasoning_summary": (
            "brief semantic compatibility explanation"
        ),
    }

    rules = [
        "You are the Odysseus PlanStep execution compatibility assessor.",
        "",
        "You do NOT execute the step and you do NOT make the final runtime "
        "selection.",
        "Your only job is to identify every listed execution surface that is "
        "semantically capable of fulfilling the PlanStep objective.",
        "",
        "RULES:",
        "1. Evaluate semantic capability from the PlanStep objective, work "
        "domains, and the surface descriptions.",
        "2. Return ALL semantically compatible surface ids, not just your "
        "favorite one.",
        "3. Do not prefer Legacy merely because it is broad; include a native "
        "LangGraph surface whenever it can genuinely perform the step.",
        "4. Missing native LangGraph parity is not a reason to erase or alter "
        "the PlanStep. Legacy or another registered surface may still be "
        "compatible.",
        "5. supports_mutation and supports_external_state are hard capability "
        "facts. Do not claim compatibility when a required fact is false.",
        "6. Selection never grants mutation permission. Downstream policy and "
        "confirmation gates still apply.",
        "7. Set clarification_needed only when missing user/source/scope "
        "information prevents determining a viable execution surface.",
        "8. Return only one JSON object matching the schema. No markdown.",
        "",
        "PLAN STEP:",
        json.dumps(
            {
                "id": step.id,
                "objective": step.objective,
                "work_domains": list(
                    step.work_domains
                ),
                "requires_external_state": (
                    step.requires_external_state
                ),
                "requires_mutation": (
                    step.requires_mutation
                ),
            },
            ensure_ascii=False,
            default=str,
        ),
        "",
        "AVAILABLE EXECUTION SURFACES:",
        json.dumps(
            [
                _surface_payload(surface)
                for surface in surfaces
            ],
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "",
        "REQUIRED JSON SCHEMA:",
        json.dumps(
            schema,
            ensure_ascii=False,
            indent=2,
        ),
    ]

    return "\n".join(rules)


def _response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response

    text: str

    if isinstance(response, str):
        text = response.strip()
    else:
        content = getattr(
            response,
            "content",
            None,
        )
        if isinstance(content, str):
            text = content.strip()
        else:
            text_value = getattr(
                response,
                "text",
                None,
            )
            if isinstance(text_value, str):
                text = text_value.strip()
            else:
                text = str(
                    response or ""
                ).strip()

    if not text:
        raise ValueError(
            "Execution Resolver model returned an empty response"
        )

    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start < 0 or end <= start:
            raise ValueError(
                "Execution Resolver model did not return a JSON object"
            )

        try:
            payload = json.loads(
                text[start : end + 1]
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Execution Resolver model returned invalid JSON"
            ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Execution Resolver response must be a JSON object"
        )

    return payload


def parse_execution_compatibility_response(
    response: Any,
    *,
    known_surface_ids: set[str],
) -> ExecutionCompatibilityAssessment:
    """Validate model compatibility output without trusting surface ids."""

    payload = _response_payload(response)

    allowed = {
        "compatible_surface_ids",
        "clarification_needed",
        "missing_information",
        "reasoning_summary",
    }
    unknown_keys = sorted(
        set(payload) - allowed
    )

    if unknown_keys:
        raise ValueError(
            "Execution Resolver response contains unsupported fields: "
            + ", ".join(unknown_keys)
        )

    compatible = _clean_string_tuple(
        payload.get(
            "compatible_surface_ids",
            [],
        ),
        field="compatible_surface_ids",
    )

    unknown_ids = [
        surface_id
        for surface_id in compatible
        if surface_id not in known_surface_ids
    ]

    if unknown_ids:
        raise ValueError(
            "Execution Resolver selected unknown surfaces: "
            + ", ".join(unknown_ids)
        )

    clarification_needed = payload.get(
        "clarification_needed",
        False,
    )

    if not isinstance(
        clarification_needed,
        bool,
    ):
        raise ValueError(
            "clarification_needed must be a boolean"
        )

    missing_information = _clean_string_tuple(
        payload.get(
            "missing_information",
            [],
        ),
        field="missing_information",
    )

    reasoning_summary = payload.get(
        "reasoning_summary",
        "",
    )

    if not isinstance(reasoning_summary, str):
        raise ValueError(
            "reasoning_summary must be a string"
        )

    return ExecutionCompatibilityAssessment(
        compatible_surface_ids=compatible,
        clarification_needed=clarification_needed,
        missing_information=missing_information,
        reasoning_summary=reasoning_summary.strip(),
    )


_KIND_RANK = {
    "langgraph": 0,
    "legacy": 1,
    "registered": 2,
}


def _hard_compatible(
    step: PlanStep,
    surface: ExecutionSurfaceDescriptor,
) -> bool:
    if (
        step.requires_external_state
        and not surface.supports_external_state
    ):
        return False

    if (
        step.requires_mutation
        and not surface.supports_mutation
    ):
        return False

    return True


def resolve_plan_step(
    step: PlanStep,
    *,
    model: ExecutionResolverModel,
    surfaces: tuple[
        ExecutionSurfaceDescriptor,
        ...,
    ] | None = None,
) -> ExecutionResolution:
    """Resolve one semantic PlanStep to a validated execution surface.

    The model proposes semantic compatibility. Deterministic runtime code then
    validates hard requirements and applies the permanent priority order:
    native LangGraph, then Legacy, then other registered surfaces.
    """

    catalog = (
        build_execution_surface_catalog()
        if surfaces is None
        else tuple(
            _validate_surface(surface)
            for surface in surfaces
        )
    )

    if not catalog:
        return ExecutionResolution(
            step_id=step.id,
            status="unsupported",
            selected_surface=None,
            reasoning_summary=(
                "No execution surfaces are available."
            ),
        )

    ids = [surface.id for surface in catalog]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Execution Resolver surface ids must be unique"
        )

    prompt = build_execution_resolver_prompt(
        step,
        catalog,
    )
    response = model.invoke(prompt)

    if inspect.isawaitable(response):
        if inspect.iscoroutine(response):
            response.close()
        raise TypeError(
            "ExecutionResolverModel returned an awaitable; use "
            "resolve_plan_step_async"
        )

    assessment = (
        parse_execution_compatibility_response(
            response,
            known_surface_ids=set(ids),
        )
    )

    if assessment.clarification_needed:
        return ExecutionResolution(
            step_id=step.id,
            status="clarification",
            selected_surface=None,
            compatible_surface_ids=(
                assessment.compatible_surface_ids
            ),
            missing_information=(
                assessment.missing_information
            ),
            reasoning_summary=(
                assessment.reasoning_summary
            ),
        )

    by_id = {
        surface.id: surface
        for surface in catalog
    }

    compatible: list[
        ExecutionSurfaceDescriptor
    ] = []
    rejected: list[str] = []

    for surface_id in (
        assessment.compatible_surface_ids
    ):
        surface = by_id[surface_id]

        if _hard_compatible(
            step,
            surface,
        ):
            compatible.append(surface)
        else:
            rejected.append(surface.id)

    if not compatible:
        reason = assessment.reasoning_summary

        if rejected:
            reason = (
                reason
                or (
                    "Semantically proposed surfaces failed "
                    "hard execution requirements."
                )
            )
        else:
            reason = (
                reason
                or (
                    "No known execution surface is semantically "
                    "compatible with this PlanStep."
                )
            )

        return ExecutionResolution(
            step_id=step.id,
            status="unsupported",
            selected_surface=None,
            compatible_surface_ids=(
                assessment.compatible_surface_ids
            ),
            rejected_surface_ids=tuple(
                rejected
            ),
            missing_information=(
                assessment.missing_information
            ),
            reasoning_summary=reason,
        )

    original_order = {
        surface.id: index
        for index, surface in enumerate(
            catalog
        )
    }

    selected = min(
        compatible,
        key=lambda surface: (
            _KIND_RANK[surface.kind],
            surface.priority,
            original_order[surface.id],
        ),
    )

    return ExecutionResolution(
        step_id=step.id,
        status="resolved",
        selected_surface=selected,
        compatible_surface_ids=(
            assessment.compatible_surface_ids
        ),
        rejected_surface_ids=tuple(
            rejected
        ),
        missing_information=(
            assessment.missing_information
        ),
        reasoning_summary=(
            assessment.reasoning_summary
        ),
    )


class _StaticExecutionResolverModel:
    """Replay one already-awaited compatibility response through validation."""

    def __init__(
        self,
        response: Any,
    ) -> None:
        self._response = response

    def invoke(
        self,
        prompt: str,
    ) -> Any:
        return self._response


async def resolve_plan_step_async(
    step: PlanStep,
    *,
    model: ExecutionResolverModel,
    surfaces: tuple[
        ExecutionSurfaceDescriptor,
        ...,
    ] | None = None,
) -> ExecutionResolution:
    """Resolve one PlanStep while supporting async production model adapters.

    Deterministic catalog validation and final priority selection remain owned
    by :func:`resolve_plan_step`; only the model invocation is awaited here.
    """

    catalog = (
        build_execution_surface_catalog()
        if surfaces is None
        else tuple(
            _validate_surface(surface)
            for surface in surfaces
        )
    )

    if not catalog:
        return resolve_plan_step(
            step,
            model=_StaticExecutionResolverModel(
                {}
            ),
            surfaces=catalog,
        )

    ids = [
        surface.id
        for surface in catalog
    ]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Execution Resolver surface ids must be unique"
        )

    prompt = build_execution_resolver_prompt(
        step,
        catalog,
    )
    response = model.invoke(prompt)

    if inspect.isawaitable(response):
        response = await response

    return resolve_plan_step(
        step,
        model=_StaticExecutionResolverModel(
            response
        ),
        surfaces=catalog,
    )


__all__ = [
    "ExecutionCompatibilityAssessment",
    "ExecutionResolution",
    "ExecutionResolverModel",
    "ExecutionSurfaceDescriptor",
    "ExecutionSurfaceKind",
    "ResolutionStatus",
    "build_execution_resolver_prompt",
    "build_execution_surface_catalog",
    "parse_execution_compatibility_response",
    "resolve_plan_step",
    "resolve_plan_step_async",
]
