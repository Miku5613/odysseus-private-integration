from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


_PLAN_KEYS = {
    "objective",
    "steps",
    "reasoning_summary",
}

_STEP_KEYS = {
    "id",
    "objective",
    "depends_on",
    "work_domains",
    "uses_dependency_state",
    "requires_external_state",
    "requires_mutation",
}


@dataclass(frozen=True)
class PlanStep:
    """Backend-independent semantic unit of work inside a TaskPlan.

    A PlanStep describes what must be accomplished and which prior semantic
    steps it depends on. It intentionally does not select a worker, tool,
    capability, action, provider, runtime, or backend. Execution resolution is
    a downstream responsibility.
    """

    id: str
    objective: str
    depends_on: tuple[str, ...] = ()
    work_domains: tuple[str, ...] = ()
    uses_dependency_state: bool = False
    requires_external_state: bool = False
    requires_mutation: bool = False


@dataclass(frozen=True)
class TaskPlan:
    """Validated dependency-aware semantic plan.

    TaskPlan owns decomposition and dependency structure only. Worker handoff,
    worker results, execution backend selection, status mutation, and artifact
    transport remain owned by downstream runtime layers.
    """

    objective: str
    steps: tuple[PlanStep, ...]
    reasoning_summary: str = ""

    def step_by_id(self, step_id: str) -> PlanStep:
        normalized = str(step_id or "").strip()
        for step in self.steps:
            if step.id == normalized:
                return step
        raise KeyError(normalized)


def _clean_required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _clean_optional_string(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value.strip()


def _clean_string_tuple(
    value: Any,
    *,
    field: str,
    reject_duplicates: bool = False,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")

    result: list[str] = []
    seen: set[str] = set()

    for item in value:
        normalized = _clean_required_string(
            item,
            field=f"{field} item",
        )
        if normalized in seen:
            if reject_duplicates:
                raise ValueError(
                    f"{field} must not contain duplicate values: "
                    f"{normalized}"
                )
            continue
        seen.add(normalized)
        result.append(normalized)

    return tuple(result)


def _clean_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a boolean")


def _reject_unknown_keys(
    payload: dict[str, Any],
    *,
    allowed: set[str],
    field: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"{field} contains unsupported fields: "
            + ", ".join(unknown)
        )


def _validate_dependencies(steps: tuple[PlanStep, ...]) -> None:
    ids = [step.id for step in steps]
    known = set(ids)

    if len(ids) != len(known):
        duplicates = sorted(
            step_id
            for step_id in known
            if ids.count(step_id) > 1
        )
        raise ValueError(
            "TaskPlan step ids must be unique: "
            + ", ".join(duplicates)
        )

    for step in steps:
        for dependency in step.depends_on:
            if dependency == step.id:
                raise ValueError(
                    f"PlanStep {step.id} cannot depend on itself"
                )
            if dependency not in known:
                raise ValueError(
                    f"PlanStep {step.id} depends on unknown step "
                    f"{dependency}"
                )


def _validate_dependency_state_usage(
    steps: tuple[PlanStep, ...],
) -> None:
    for step in steps:
        if (
            step.uses_dependency_state
            and not step.depends_on
        ):
            raise ValueError(
                f"PlanStep {step.id} uses_dependency_state=true "
                "requires at least one depends_on step"
            )


def topological_step_ids(plan: TaskPlan) -> tuple[str, ...]:
    """Return a stable topological ordering, rejecting dependency cycles."""

    by_id = {step.id: step for step in plan.steps}
    indegree = {
        step.id: len(step.depends_on)
        for step in plan.steps
    }
    children: dict[str, list[str]] = {
        step.id: []
        for step in plan.steps
    }

    for step in plan.steps:
        for dependency in step.depends_on:
            children[dependency].append(step.id)

    order_index = {
        step.id: index
        for index, step in enumerate(plan.steps)
    }

    ready = [
        step.id
        for step in plan.steps
        if indegree[step.id] == 0
    ]
    result: list[str] = []

    while ready:
        ready.sort(key=order_index.__getitem__)
        current = ready.pop(0)
        result.append(current)

        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    if len(result) != len(by_id):
        cyclic = [
            step_id
            for step_id, degree in indegree.items()
            if degree > 0
        ]
        raise ValueError(
            "TaskPlan dependencies must be acyclic; cycle involves: "
            + ", ".join(cyclic)
        )

    return tuple(result)


def validate_task_plan_payload(
    payload: dict[str, Any],
) -> TaskPlan:
    """Validate model/planner output into a backend-independent TaskPlan."""

    if not isinstance(payload, dict):
        raise ValueError("TaskPlan payload must be an object")

    _reject_unknown_keys(
        payload,
        allowed=_PLAN_KEYS,
        field="TaskPlan",
    )

    objective = _clean_required_string(
        payload.get("objective"),
        field="TaskPlan objective",
    )

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(
            "TaskPlan steps must be a non-empty list"
        )

    steps: list[PlanStep] = []

    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise ValueError(
                f"TaskPlan step {index} must be an object"
            )

        _reject_unknown_keys(
            raw_step,
            allowed=_STEP_KEYS,
            field=f"TaskPlan step {index}",
        )

        steps.append(
            PlanStep(
                id=_clean_required_string(
                    raw_step.get("id"),
                    field=f"TaskPlan step {index} id",
                ),
                objective=_clean_required_string(
                    raw_step.get("objective"),
                    field=f"TaskPlan step {index} objective",
                ),
                depends_on=_clean_string_tuple(
                    raw_step.get("depends_on", []),
                    field=(
                        f"TaskPlan step {index} depends_on"
                    ),
                    reject_duplicates=True,
                ),
                work_domains=_clean_string_tuple(
                    raw_step.get("work_domains", []),
                    field=(
                        f"TaskPlan step {index} work_domains"
                    ),
                ),
                uses_dependency_state=_clean_bool(
                    raw_step.get(
                        "uses_dependency_state",
                        False,
                    ),
                    field=(
                        f"TaskPlan step {index} "
                        "uses_dependency_state"
                    ),
                ),
                requires_external_state=_clean_bool(
                    raw_step.get(
                        "requires_external_state",
                        False,
                    ),
                    field=(
                        f"TaskPlan step {index} "
                        "requires_external_state"
                    ),
                ),
                requires_mutation=_clean_bool(
                    raw_step.get(
                        "requires_mutation",
                        False,
                    ),
                    field=(
                        f"TaskPlan step {index} "
                        "requires_mutation"
                    ),
                ),
            )
        )

    plan = TaskPlan(
        objective=objective,
        steps=tuple(steps),
        reasoning_summary=_clean_optional_string(
            payload.get("reasoning_summary", ""),
            field="TaskPlan reasoning_summary",
        ),
    )

    _validate_dependencies(plan.steps)
    _validate_dependency_state_usage(plan.steps)
    topological_step_ids(plan)
    return plan


def _normalize_runtime_ids(
    plan: TaskPlan,
    values: Iterable[str],
    *,
    field: str,
) -> set[str]:
    known = {step.id for step in plan.steps}
    normalized: set[str] = set()

    for value in values:
        step_id = _clean_required_string(
            value,
            field=f"{field} item",
        )
        if step_id not in known:
            raise ValueError(
                f"{field} references unknown step {step_id}"
            )
        normalized.add(step_id)

    return normalized


def blocked_step_ids(
    plan: TaskPlan,
    *,
    completed_step_ids: Iterable[str] = (),
    failed_step_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return pending steps transitively blocked by failed prerequisites."""

    completed = _normalize_runtime_ids(
        plan,
        completed_step_ids,
        field="completed_step_ids",
    )
    failed = _normalize_runtime_ids(
        plan,
        failed_step_ids,
        field="failed_step_ids",
    )

    overlap = completed & failed
    if overlap:
        raise ValueError(
            "A step cannot be both completed and failed: "
            + ", ".join(sorted(overlap))
        )

    blocked_or_failed = set(failed)
    changed = True

    while changed:
        changed = False
        for step in plan.steps:
            if (
                step.id in completed
                or step.id in blocked_or_failed
            ):
                continue

            if any(
                dependency in blocked_or_failed
                for dependency in step.depends_on
            ):
                blocked_or_failed.add(step.id)
                changed = True

    return tuple(
        step.id
        for step in plan.steps
        if (
            step.id in blocked_or_failed
            and step.id not in failed
        )
    )


def ready_step_ids(
    plan: TaskPlan,
    *,
    completed_step_ids: Iterable[str] = (),
    failed_step_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return pending steps whose dependencies are all completed.

    Failed prerequisites block their downstream dependents transitively.
    This function does not execute, select, or resolve any backend.
    """

    completed = _normalize_runtime_ids(
        plan,
        completed_step_ids,
        field="completed_step_ids",
    )
    failed = _normalize_runtime_ids(
        plan,
        failed_step_ids,
        field="failed_step_ids",
    )

    overlap = completed & failed
    if overlap:
        raise ValueError(
            "A step cannot be both completed and failed: "
            + ", ".join(sorted(overlap))
        )

    blocked = set(
        blocked_step_ids(
            plan,
            completed_step_ids=completed,
            failed_step_ids=failed,
        )
    )

    return tuple(
        step.id
        for step in plan.steps
        if (
            step.id not in completed
            and step.id not in failed
            and step.id not in blocked
            and all(
                dependency in completed
                for dependency in step.depends_on
            )
        )
    )


__all__ = [
    "PlanStep",
    "TaskPlan",
    "blocked_step_ids",
    "ready_step_ids",
    "topological_step_ids",
    "validate_task_plan_payload",
]
