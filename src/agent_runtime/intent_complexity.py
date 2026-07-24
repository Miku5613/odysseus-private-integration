from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from src.agent_runtime.structured_llm import (
    invoke_validated_fallback_async,
)
from src.llm_core import llm_call_async_with_fallback


ExecutionClass = Literal[
    "conversation",
    "single_graph",
    "multi_graph",
]


@dataclass(frozen=True)
class IntentGoal:
    """One semantic objective discovered before detailed task planning.

    work_domains are backend-independent semantic domains such as persisted
    knowledge, calendar state, current public research, or mutation work.
    They are not LangGraph capability names and are never validated against
    the current Capability Registry.
    """

    id: str
    objective: str
    work_domains: tuple[str, ...] = ()
    requires_external_state: bool = False
    requires_mutation: bool = False


@dataclass(frozen=True)
class IntentComplexityAssessment:
    """Validated 0 / 1 / multi-graph execution assessment.

    This contract owns semantic execution complexity only.

    It does not choose capabilities, actions, workers, tools, or execution
    backends. Downstream orchestration owns those decisions.

    A missing native LangGraph capability must never reduce the assessed
    complexity. The downstream Execution Resolver may deliberately delegate a
    step to the Legacy runtime (or another registered backend), return that
    result into planner state, and continue dependent steps.
    """

    execution_class: ExecutionClass
    goals: tuple[IntentGoal, ...]
    needs_planning: bool
    missing_information: tuple[str, ...]
    clarification_needed: bool
    reasoning_summary: str


def build_intent_complexity_prompt(raw_query: str) -> str:
    """Build the backend-independent semantic pre-planning prompt."""

    schema = {
        "execution_class": (
            "conversation | single_graph | multi_graph"
        ),
        "goals": [
            {
                "id": "goal_1",
                "objective": "Semantic objective.",
                "work_domains": [
                    "backend-independent semantic work domain"
                ],
                "requires_external_state": False,
                "requires_mutation": False,
            }
        ],
        "needs_planning": False,
        "missing_information": [],
        "clarification_needed": False,
        "reasoning_summary": "Brief complexity rationale.",
    }

    instructions = [
        "You are the Intent and Complexity Definer for the Odysseus "
        "LangGraph runtime.",
        "",
        "Your job is semantic pre-planning only.",
        "Do not answer the user.",
        "Do not execute tools.",
        "Do not choose capabilities, workers, tools, actions, or backends.",
        "Do not create detailed PlanSteps.",
        "Do not grant permissions.",
        "",
        "Classify the minimum execution structure actually required:",
        "",
        "conversation:",
        "- No external, persisted, mutable, or tool-backed execution is "
        "required.",
        "- The caller can reuse Odysseus's existing normal-chat behavior.",
        "- needs_planning must be false.",
        "",
        "single_graph:",
        "- One bounded execution unit can satisfy the request.",
        "- A single bounded research, retrieval, inspection, or mutation job "
        "can still be single_graph even if that unit later performs several "
        "internal tool calls within one acquisition/provenance domain.",
        "- Multiple subjects, versions, subtopics, communities, documents, or "
        "public sources inside one bounded research workflow do NOT by "
        "themselves create multiple graphs. Semantic topic count is not the "
        "same thing as provenance/state-system count.",
        "- Recovering one prior, saved, generated, or persisted artifact that "
        "is not actually supplied in the current request/context is execution, "
        "not conversation; if that retrieval plus transformation is one bounded "
        "unit, classify it as single_graph.",
        "- Exact capability/action/backend selection belongs downstream.",
        "- needs_planning must be false.",
        "",
        "multi_graph:",
        "- The request genuinely requires multiple distinct execution units, "
        "state systems, semantic work domains, dependencies, ordered handoffs, "
        "or parallelizable work.",
        "- When a final result depends on independently acquiring information "
        "from multiple provenance/state domains and then combining those "
        "outputs, classify multi_graph even if one model could perform the "
        "final synthesis after the acquisitions complete.",
        "- A provenance/state domain means an independently accessed source-of-"
        "truth or execution surface, not merely another semantic topic. "
        "Examples include current public information versus user-specific "
        "persisted notes, a saved artifact versus fresh external data, or two "
        "separate state systems that each require their own retrieval.",
        "- Do not collapse 'acquire A + acquire B + combine A and B' into one "
        "graph merely because the final comparison, recommendation, or report "
        "is a single response.",
        "- needs_planning must be true.",
        "- Do not invent exact PlanSteps here. A separate Task Planner owns "
        "decomposition, dependencies, and the final graph count.",
        "",
        "Backend-independence invariant:",
        "- Classify the work the user's objective requires, not only the work "
        "currently represented by native LangGraph capabilities.",
        "- Missing native LangGraph coverage must NEVER downgrade multi_graph "
        "to single_graph or conversation.",
        "- A downstream Execution Resolver may deliberately delegate an "
        "uncovered planned step to the Legacy runtime or another registered "
        "backend, return that output into planner state, and continue.",
        "",
        "Rules:",
        "1. Reason semantically from the user's objective. Do not use keyword "
        "matching as the decision method.",
        "2. Prefer the smallest sufficient execution class without hiding "
        "required work.",
        "3. Agent availability does not imply execution is required.",
        "4. General knowledge, conceptual explanation, opinion, and reasoning "
        "answerable from the conversational model's own knowledge are "
        "conversation. Descriptive work_domains do not by themselves imply "
        "execution.",
        "5. Access to user-specific persisted state, external/current state, "
        "or mutation counts as execution even when phrased conversationally.",
        "6. A reference to prior/session/history content is NOT automatically "
        "available conversational context. If the answer depends on recovering "
        "a specific prior, saved, generated, or persisted artifact whose "
        "contents are not supplied in the current request/context, retrieval "
        "is execution.",
        "7. Distinct required state systems count as distinct work domains. "
        "One bounded acquisition domain may use many internal sources/tools, "
        "topics, comparisons, or subquestions and remain single_graph. Do not "
        "promote to multi_graph just because research spans several semantic "
        "subtopics.",
        "8. User-specific facts, constraints, goals, notes, or saved context "
        "that must be retrieved from persisted/private state are a distinct "
        "acquisition domain from current/public research. If the final answer "
        "depends on independently acquiring both and then combining them, "
        "classify multi_graph.",
        "9. Possessive or deictic references to user-owned state -- such as "
        "their goals, plans, preferences, setup, constraints, notes, or saved "
        "context -- imply a user-specific acquisition domain when the actual "
        "referenced content is not embedded in the current request. Do not "
        "reinterpret that retrieval as mere semantic analysis just because "
        "the final deliverable is one report or one answer.",
        "10. Execution class is determined by prerequisite acquisition "
        "boundaries and dependencies, not by final artifact count. One final "
        "report can still require multi_graph when it depends on separately "
        "retrieved private/user state plus independently researched external "
        "information.",
        "11. Multiple independently acquired provenance domains whose outputs "
        "are prerequisites for a combined downstream result require "
        "multi_graph. Count acquisition boundaries, not topic labels.",
        "12. work_domains are short semantic labels only. They are NOT "
        "capability names, they do not themselves imply execution, and they "
        "must not depend on the current registry.",
        "13. Set clarification_needed=true only when ambiguity materially "
        "prevents safe or correct progress.",
        "14. Return only one JSON object matching the required schema.",
        "",
        "USER_QUERY:",
        raw_query,
        "",
        "REQUIRED_OUTPUT_SCHEMA:",
        json.dumps(
            schema,
            ensure_ascii=False,
            indent=2,
        ),
    ]

    return "\n".join(instructions)


def _load_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object while tolerating harmless model wrappers."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Intent/complexity model returned no textual JSON"
        )

    candidate = text.strip()

    if candidate.startswith("```"):
        lines = candidate.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        candidate = "\n".join(lines).strip()

    try:
        payload = json.loads(candidate)

        if not isinstance(payload, dict):
            raise ValueError(
                "Intent/complexity response must be one JSON object"
            )

        return payload

    except json.JSONDecodeError:
        decoder = json.JSONDecoder()

        for index, character in enumerate(candidate):
            if character != "{":
                continue

            try:
                payload, _ = decoder.raw_decode(
                    candidate[index:]
                )
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict):
                return payload

        raise ValueError(
            "Intent/complexity response did not contain valid JSON"
        )


def _strict_bool(
    value: Any,
    *,
    field: str,
) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"{field} must be a boolean"
        )

    return value


def _dedupe_strings(
    value: Any,
    *,
    field: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(
            f"{field} must be a list"
        )

    result: list[str] = []

    for entry in value:
        if (
            not isinstance(entry, str)
            or not entry.strip()
        ):
            raise ValueError(
                f"{field} must contain non-empty strings"
            )

        normalized = entry.strip()

        if normalized not in result:
            result.append(normalized)

    return tuple(result)


def _parse_goals(
    raw_goals: Any,
) -> tuple[IntentGoal, ...]:
    if not isinstance(raw_goals, list):
        raise ValueError(
            "goals must be a list"
        )

    goals: list[IntentGoal] = []
    seen_ids: set[str] = set()

    for index, raw_goal in enumerate(
        raw_goals,
        start=1,
    ):
        if not isinstance(raw_goal, dict):
            raise ValueError(
                f"goals[{index}] must be an object"
            )

        goal_id = raw_goal.get("id")
        objective = raw_goal.get("objective")

        if (
            not isinstance(goal_id, str)
            or not goal_id.strip()
        ):
            raise ValueError(
                f"goals[{index}].id must be a non-empty string"
            )

        goal_id = goal_id.strip()

        if goal_id in seen_ids:
            raise ValueError(
                f"Duplicate goal id: {goal_id}"
            )

        seen_ids.add(goal_id)

        if (
            not isinstance(objective, str)
            or not objective.strip()
        ):
            raise ValueError(
                f"goals[{index}].objective must be a "
                "non-empty string"
            )

        work_domains = _dedupe_strings(
            raw_goal.get(
                "work_domains",
                [],
            ),
            field=f"goals[{index}].work_domains",
        )

        requires_external_state = _strict_bool(
            raw_goal.get(
                "requires_external_state",
                False,
            ),
            field=(
                f"goals[{index}].requires_external_state"
            ),
        )

        requires_mutation = _strict_bool(
            raw_goal.get(
                "requires_mutation",
                False,
            ),
            field=f"goals[{index}].requires_mutation",
        )

        goals.append(
            IntentGoal(
                id=goal_id,
                objective=objective.strip(),
                work_domains=work_domains,
                requires_external_state=(
                    requires_external_state
                ),
                requires_mutation=requires_mutation,
            )
        )

    return tuple(goals)


def validate_intent_complexity_payload(
    payload: dict[str, Any],
) -> IntentComplexityAssessment:
    """Validate backend-independent semantic complexity invariants."""

    if not isinstance(payload, dict):
        raise ValueError(
            "Intent/complexity payload must be an object"
        )

    execution_class = payload.get(
        "execution_class"
    )

    if execution_class not in {
        "conversation",
        "single_graph",
        "multi_graph",
    }:
        raise ValueError(
            "execution_class must be conversation, "
            "single_graph, or multi_graph"
        )

    needs_planning = _strict_bool(
        payload.get(
            "needs_planning",
            False,
        ),
        field="needs_planning",
    )

    clarification_needed = _strict_bool(
        payload.get(
            "clarification_needed",
            False,
        ),
        field="clarification_needed",
    )

    missing_information = _dedupe_strings(
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

    goals = _parse_goals(
        payload.get("goals", [])
    )

    if execution_class == "conversation":
        for goal in goals:
            if (
                goal.requires_external_state
                or goal.requires_mutation
            ):
                raise ValueError(
                    "conversation goals must remain descriptive and "
                    "must not require external state or mutation"
                )

        if needs_planning:
            raise ValueError(
                "conversation assessment cannot require planning"
            )

        if clarification_needed:
            raise ValueError(
                "conversation clarification belongs to the normal "
                "chat path, not agent execution assessment"
            )

    elif execution_class == "single_graph":
        if not goals:
            raise ValueError(
                "single_graph assessment requires at least one goal"
            )

        if needs_planning:
            raise ValueError(
                "single_graph assessment must not require the "
                "multi-graph Task Planner"
            )

    else:
        if not goals:
            raise ValueError(
                "multi_graph assessment requires at least one goal"
            )

        if not needs_planning:
            raise ValueError(
                "multi_graph assessment must require planning"
            )

    return IntentComplexityAssessment(
        execution_class=execution_class,
        goals=goals,
        needs_planning=needs_planning,
        missing_information=missing_information,
        clarification_needed=clarification_needed,
        reasoning_summary=reasoning_summary.strip(),
    )


def parse_intent_complexity_response(
    text: str,
) -> IntentComplexityAssessment:
    """Parse and validate one model-produced assessment."""

    return validate_intent_complexity_payload(
        _load_json_object(text)
    )


async def assess_intent_complexity(
    raw_query: str,
    *,
    llm_candidates: Sequence[Any],
    prompt_type: str | None = None,
    max_tokens: int = 1400,
) -> IntentComplexityAssessment:
    """Run one model-backed semantic complexity assessment.

    The caller owns current-model and fallback resolution.

    This function consumes the already-resolved candidate chain and uses
    Odysseus's canonical fallback executor.
    """

    if (
        not isinstance(raw_query, str)
        or not raw_query.strip()
    ):
        raise ValueError(
            "raw_query must be a non-empty string"
        )

    prompt = build_intent_complexity_prompt(
        raw_query.strip()
    )

    token_limit = (
        max_tokens
        if isinstance(max_tokens, int)
        and not isinstance(max_tokens, bool)
        and max_tokens > 0
        else 1400
    )

    # Small reasoning models may consume a shorter allowance before emitting
    # structured JSON. Preserve the empirically validated bounded floor.
    token_limit = min(
        max(token_limit, 3000),
        4096,
    )

    response = await invoke_validated_fallback_async(
        llm_call_async_with_fallback,
        list(llm_candidates),
        prompt,
        parse_intent_complexity_response,
        call_kwargs={
            "temperature": 0.0,
            "max_tokens": token_limit,
            "prompt_type": prompt_type,
        },
        max_rounds=2,
    )

    if not isinstance(response, str):
        raise ValueError(
            "Intent/complexity LLM response must be text"
        )

    return parse_intent_complexity_response(
        response
    )


__all__ = [
    "ExecutionClass",
    "IntentGoal",
    "IntentComplexityAssessment",
    "build_intent_complexity_prompt",
    "validate_intent_complexity_payload",
    "parse_intent_complexity_response",
    "assess_intent_complexity",
]
