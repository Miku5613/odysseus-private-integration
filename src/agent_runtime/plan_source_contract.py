from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from src.agent_runtime.intent_complexity import (
    EXTERNAL_SOURCES,
    IntentComplexityAssessment,
)
from src.agent_runtime.task_plan import PlanStep, TaskPlan


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(value).strip()
        for value in values
        if str(value).strip()
    ))


def _relevant_concepts(
    step: PlanStep,
    assessment: IntentComplexityAssessment,
) -> tuple[str, ...]:
    concepts = _dedupe(
        concept
        for goal in assessment.goals
        for concept in goal.retrieval_concepts
    )
    if step.retrieval_concepts:
        selected = _dedupe(step.retrieval_concepts)
        unsupported = set(selected) - set(concepts)
        if unsupported:
            raise ValueError(
                f"PlanStep {step.id} has retrieval concepts outside "
                "admission: " + ", ".join(sorted(unsupported))
            )
        return selected

    matching_goals = tuple(
        goal
        for goal in assessment.goals
        if goal.id == step.id and goal.retrieval_concepts
    )
    if len(matching_goals) == 1:
        return _dedupe(matching_goals[0].retrieval_concepts)

    external_goals = tuple(
        goal
        for goal in assessment.goals
        if goal.requires_external_state and goal.retrieval_concepts
    )
    if len(external_goals) == 1:
        return _dedupe(external_goals[0].retrieval_concepts)
    if concepts:
        raise ValueError(
            f"PlanStep {step.id} requires explicit retrieval_concepts "
            "because admission contains multiple external goals"
        )
    return ()


def bind_admitted_plan_sources(
    plan: TaskPlan,
    assessment: IntentComplexityAssessment,
) -> TaskPlan:
    """Bind model-planned steps to the already-admitted source envelope."""

    admitted = _dedupe(assessment.requested_sources)
    admitted_external = tuple(
        source
        for source in admitted
        if source in EXTERNAL_SOURCES
    )
    globally_excluded = _dedupe(assessment.excluded_sources)
    bound: list[PlanStep] = []

    for step in plan.steps:
        requested = _dedupe(step.requested_sources)
        excluded = _dedupe(
            (*globally_excluded, *step.excluded_sources)
        )

        if step.requires_external_state:
            if not requested:
                if len(admitted_external) != 1:
                    raise ValueError(
                        f"PlanStep {step.id} requires an explicit requested "
                        "source because admission authorized multiple or no "
                        "external sources"
                    )
                requested = admitted_external

            unauthorized = set(requested) - set(admitted_external)
            if unauthorized:
                raise ValueError(
                    f"PlanStep {step.id} requested sources outside admission: "
                    + ", ".join(sorted(unauthorized))
                )
            if set(requested) & set(excluded):
                raise ValueError(
                    f"PlanStep {step.id} requests an excluded source"
                )
        elif set(requested) & EXTERNAL_SOURCES:
            raise ValueError(
                f"PlanStep {step.id} requests external sources while "
                "requires_external_state=false"
            )

        bound.append(replace(
            step,
            requested_sources=requested,
            excluded_sources=excluded,
            retrieval_concepts=_relevant_concepts(
                step,
                assessment,
            ) if step.requires_external_state else (),
        ))

    return replace(plan, steps=tuple(bound))


def constrain_replanned_sources(
    previous_plan: TaskPlan,
    revised_plan: TaskPlan,
) -> TaskPlan:
    """Prevent a replan from widening beyond previously admitted sources."""

    admitted = _dedupe(
        source
        for step in previous_plan.steps
        for source in step.requested_sources
    )
    admitted_external = tuple(
        source for source in admitted if source in EXTERNAL_SOURCES
    )
    globally_excluded = _dedupe(
        source
        for step in previous_plan.steps
        for source in step.excluded_sources
    )
    bound: list[PlanStep] = []

    for step in revised_plan.steps:
        requested = _dedupe(step.requested_sources)
        excluded = _dedupe(
            (*globally_excluded, *step.excluded_sources)
        )
        if step.requires_external_state and not requested:
            if len(admitted_external) != 1:
                raise ValueError(
                    f"Replanned PlanStep {step.id} requires an explicit "
                    "requested source"
                )
            requested = admitted_external
        unauthorized = set(requested) - set(admitted_external)
        if unauthorized:
            raise ValueError(
                f"Replanned PlanStep {step.id} widens admitted sources: "
                + ", ".join(sorted(unauthorized))
            )
        if set(requested) & set(excluded):
            raise ValueError(
                f"Replanned PlanStep {step.id} requests an excluded source"
            )
        if not step.requires_external_state and (
            set(requested) & EXTERNAL_SOURCES
        ):
            raise ValueError(
                f"Replanned PlanStep {step.id} requests external sources "
                "while requires_external_state=false"
            )
        bound.append(replace(
            step,
            requested_sources=requested,
            excluded_sources=excluded,
        ))

    return replace(revised_plan, steps=tuple(bound))


def step_source_contract(step: PlanStep) -> dict[str, list[str]]:
    return {
        "requested_sources": list(step.requested_sources),
        "excluded_sources": list(step.excluded_sources),
        "retrieval_concepts": list(step.retrieval_concepts),
    }


__all__ = [
    "bind_admitted_plan_sources",
    "constrain_replanned_sources",
    "step_source_contract",
]
