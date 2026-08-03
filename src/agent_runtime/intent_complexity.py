from __future__ import annotations

import json
import re
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
SideEffectIntent = Literal[
    "none",
    "mutation_requested",
    "approval_gated_mutation",
]

RequestedSource = Literal[
    "model_knowledge",
    "current_conversation",
    "authorized_workspace_files",
    "indexed_workspace_knowledge",
    "personal_memory",
    "completed_research",
    "public_web",
    "attached_document",
    "clarification_required",
]

REQUESTED_SOURCES = {
    "model_knowledge",
    "current_conversation",
    "authorized_workspace_files",
    "indexed_workspace_knowledge",
    "personal_memory",
    "completed_research",
    "public_web",
    "attached_document",
    "clarification_required",
}

EXTERNAL_SOURCES = REQUESTED_SOURCES - {
    "model_knowledge",
    "current_conversation",
    "clarification_required",
}


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
    retrieval_concepts: tuple[str, ...] = ()


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
    requested_sources: tuple[RequestedSource, ...] = ()
    excluded_sources: tuple[RequestedSource, ...] = ()
    side_effect_intent: SideEffectIntent = "none"


def build_intent_complexity_prompt(
    raw_query: str,
    *,
    active_workspace: bool = False,
    workspace_label: str | None = None,
) -> str:
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
                "retrieval_concepts": [
                    "short source-native search concept"
                ],
            }
        ],
        "needs_planning": False,
        "side_effect_intent": (
            "none | mutation_requested | approval_gated_mutation"
        ),
        "requested_sources": ["model_knowledge"],
        "excluded_sources": [],
        "missing_information": [],
        "clarification_needed": False,
        "reasoning_summary": "Brief complexity rationale.",
    }
    execution_context = {
        "active_workspace_available": bool(active_workspace),
        "active_workspace_label": (
            str(workspace_label or "").strip() or None
        ),
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
        "Identify the information sources the user is asking the runtime to "
        "use, and sources the user explicitly excludes.",
        "",
        "HIGHEST-PRIORITY ACTIVE WORKSPACE RULE:",
        "- If ACTIVE_EXECUTION_CONTEXT says an authorized workspace is "
        "available, a request for the actual source path, current "
        "implementation location, or the real logic of project behavior "
        "requires authorized_workspace_files and external-state execution.",
        "- The user does not need to literally say 'read files', name a file, "
        "or repeat the workspace name. Locating the actual current "
        "implementation necessarily requires workspace discovery.",
        "- Never claim model training knowledge is sufficient to identify the "
        "actual current source path or current repository implementation.",
        "- An active project normally contains multiple configuration "
        "surfaces. A request to inspect 'the configuration' or judge 'its "
        "defaults' without identifying which subsystem/configuration is not a "
        "bounded target. It MUST request clarification rather than choosing "
        "files on the user's behalf.",
        "- Tracing how a named current project setting, control, request, "
        "event, or policy moves between UI and backend layers is current "
        "implementation analysis, not a generic architecture explanation. It "
        "requires authorized_workspace_files even when the user does not ask "
        "for a particular filename or source path.",
        "- If the user explicitly excludes a source, excluded_sources MUST "
        "record it even when that source would not otherwise have been "
        "selected.",
        "- A requested side effect remains an execution goal when the user "
        "requires approval before it occurs. The approval gate is part of the "
        "execution lifecycle; it does not turn the request into conversation. "
        "Set requires_mutation=true and select the target state source, while "
        "leaving permission enforcement to deterministic runtime policy.",
        "- side_effect_intent MUST be approval_gated_mutation when the user "
        "requests a side effect but requires approval before execution. Use "
        "mutation_requested for a requested side effect without that explicit "
        "gate, otherwise none.",
        "- Workspace availability does not resolve an underspecified target. "
        "If a request refers only to an unspecified configuration, document, "
        "setting, or other target whose identity materially changes the "
        "inspection, ask a focused clarification first. Use "
        "clarification_required as the sole requested source until the target "
        "is identified.",
        "",
        "Classify the minimum execution structure actually required:",
        "",
        "conversation:",
        "- No external, persisted, mutable, or tool-backed execution is "
        "required.",
        "- The caller can reuse Odysseus's existing normal-chat behavior.",
        "- needs_planning must be false.",
        "- requested_sources may only be model_knowledge, "
        "current_conversation, or attached_document.",
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
        "- Frontend and backend files inside the same authorized workspace are "
        "one filesystem provenance domain and can be inspected by one Code "
        "specialist. Multiple files or application layers do not by themselves "
        "require multi_graph.",
        "- Treat several file inspections as one bounded unit only when they "
        "serve one coupled investigation whose useful result stands or falls "
        "together. A shared filesystem source or specialist capability does "
        "not erase independently useful deliverables.",
        "- A single cross-layer trace through one feature remains one Code "
        "goal. In contrast, a request to independently inspect several "
        "distinct protection subsystems and then compare their behavior has "
        "separate acquisition goals plus a synthesis dependency; represent "
        "those goals explicitly and classify multi_graph when one bounded "
        "specialist run cannot reliably establish every subsystem.",
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
        "- Independently useful goals may share authorized_workspace_files and "
        "the same downstream specialist. When one goal can be partial, "
        "unavailable, or blocked while the others should still produce useful "
        "results, preserve them as separate goals and classify multi_graph so "
        "the planner can isolate that failure.",
        "- An explicit request to continue independent goals after one goal is "
        "partial is direct evidence that one all-or-nothing execution unit is "
        "insufficient. This remains multi_graph even when every goal reads the "
        "same authorized workspace.",
        "- needs_planning must be true.",
        "- Do not invent exact PlanSteps here. A separate Task Planner owns "
        "decomposition, dependencies, and the final graph count.",
        "",
        "Source-selection contract:",
        "- model_knowledge means the answer can come from the model's general "
        "knowledge without retrieving user or external state.",
        "- current_conversation means the required evidence is already present "
        "in the supplied conversation.",
        "- authorized_workspace_files means direct read-only inspection of the "
        "active authorized filesystem workspace. It does not require an index.",
        "- indexed_workspace_knowledge means retrieval from a workspace index.",
        "- personal_memory means retrieval from the user's persisted memory.",
        "- completed_research means retrieval of a previously completed "
        "research artifact.",
        "- public_web means current public internet retrieval.",
        "- attached_document means the supplied attachment is the evidence.",
        "- clarification_required means the requested source or target cannot "
        "be determined safely enough to proceed.",
        "- A question asking where or how behavior, policy, configuration, a "
        "subsystem, or an implementation exists in the active project requires "
        "authorized_workspace_files even when the user does not know a "
        "filename. Do not treat unsupplied repository evidence as model "
        "knowledge or current conversation.",
        "- When an authorized active workspace is available, resolve an "
        "otherwise unqualified request about where project behavior is checked "
        "or implemented against that workspace. Do not reinterpret it as a "
        "generic operating-system or programming-language explanation.",
        "- ACTIVE WORKSPACE RESOLUTION INVARIANT: model training knowledge "
        "cannot establish where current workspace behavior actually lives. "
        "When the active workspace is available, an unqualified request to "
        "locate a check, rule, behavior, configuration, or implementation and "
        "summarize its actual logic must use authorized_workspace_files. This "
        "remains true when a generic conceptual answer would be possible. Use "
        "model_knowledge alone only when the user asks for a general concept "
        "or explicitly says not to access the workspace.",
        "- A read-only restriction limits side effects; it does not turn "
        "required source retrieval into model knowledge.",
        "- Select current_conversation only when the evidence needed to answer "
        "is actually included in the supplied query/context. A mere topic "
        "reference is not evidence.",
        "- Record explicit source restrictions in excluded_sources. Never "
        "substitute public_web for an unavailable workspace source.",
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
        "conversation only when the requested truth does not depend on the "
        "current workspace or another retrieved source. Descriptive "
        "work_domains do not by themselves imply execution.",
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
        "13. For a goal that retrieves evidence, provide retrieval_concepts: "
        "four to eight concise source-search tokens, common API terms, or "
        "plausible existing symbols that a specialist can search for. Prefer "
        "simple domain terms and established code vocabulary over invented "
        "compound identifiers. Include useful semantic alternatives. Exclude "
        "generic request verbs, prose glue, and actions the user explicitly "
        "prohibited. These are semantic search inputs, not routing rules.",
        "14. Set clarification_needed=true only when ambiguity materially "
        "prevents safe or correct progress.",
        "15. requested_sources and excluded_sources must contain only source "
        "names from the source-selection contract and must not overlap.",
        "16. If clarification_needed is true, include clarification_required "
        "in requested_sources. Otherwise do not include it.",
        "17. Return only one JSON object matching the required schema.",
        "",
        "USER_QUERY:",
        raw_query,
        "",
        "ACTIVE_EXECUTION_CONTEXT:",
        json.dumps(execution_context, ensure_ascii=False, indent=2),
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
        retrieval_concepts = _dedupe_strings(
            raw_goal.get(
                "retrieval_concepts",
                [],
            ),
            field=f"goals[{index}].retrieval_concepts",
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
                retrieval_concepts=retrieval_concepts,
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
    side_effect_intent = payload.get(
        "side_effect_intent",
        "none",
    )
    if side_effect_intent not in {
        "none",
        "mutation_requested",
        "approval_gated_mutation",
    }:
        raise ValueError(
            "side_effect_intent must be none, mutation_requested, "
            "or approval_gated_mutation"
        )

    requested_sources = _dedupe_strings(
        payload.get("requested_sources"),
        field="requested_sources",
    )
    excluded_sources = _dedupe_strings(
        payload.get("excluded_sources"),
        field="excluded_sources",
    )

    unknown_sources = (
        set(requested_sources) | set(excluded_sources)
    ) - REQUESTED_SOURCES
    if unknown_sources:
        raise ValueError(
            "Unknown requested or excluded source: "
            + ", ".join(sorted(unknown_sources))
        )

    overlap = set(requested_sources) & set(excluded_sources)
    if overlap:
        raise ValueError(
            "requested_sources and excluded_sources must not overlap"
        )

    if clarification_needed:
        requested_sources = ("clarification_required",)
        execution_class = "single_graph"
        needs_planning = False
    elif side_effect_intent != "none":
        execution_class = "single_graph"
        needs_planning = False
        goals = tuple(
            IntentGoal(
                id=goal.id,
                objective=goal.objective,
                work_domains=goal.work_domains,
                requires_external_state=goal.requires_external_state,
                requires_mutation=True,
                retrieval_concepts=goal.retrieval_concepts,
            )
            for goal in goals
        )
    elif "clarification_required" in requested_sources:
        raise ValueError(
            "clarification_required source requires clarification_needed"
        )

    if execution_class == "conversation":
        invalid_conversation_sources = set(requested_sources) - {
            "model_knowledge",
            "current_conversation",
            "attached_document",
        }
        if invalid_conversation_sources:
            raise ValueError(
                "conversation cannot require retrieved or mutable sources"
            )
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

        if not requested_sources:
            raise ValueError(
                "single_graph assessment requires a requested source"
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

        if not requested_sources:
            raise ValueError(
                "multi_graph assessment requires requested sources"
            )

        external_requested = (
            set(requested_sources) & EXTERNAL_SOURCES
        )
        if len(goals) < 2 and len(external_requested) < 2:
            raise ValueError(
                "multi_graph requires multiple semantic goals or "
                "multiple independently acquired source domains"
            )

    if set(requested_sources) & EXTERNAL_SOURCES:
        if not any(goal.requires_external_state for goal in goals):
            raise ValueError(
                "retrieved sources require an external-state goal"
            )

    return IntentComplexityAssessment(
        execution_class=execution_class,
        goals=goals,
        needs_planning=needs_planning,
        missing_information=missing_information,
        clarification_needed=clarification_needed,
        reasoning_summary=reasoning_summary.strip(),
        requested_sources=requested_sources,
        excluded_sources=excluded_sources,
        side_effect_intent=side_effect_intent,
    )


def parse_intent_complexity_response(
    text: str,
) -> IntentComplexityAssessment:
    """Parse and validate one model-produced assessment."""

    return validate_intent_complexity_payload(
        _load_json_object(text)
    )


def _explicit_independent_goal_clauses(
    raw_query: str,
) -> tuple[str, ...]:
    """Return user-authored numbered goals with an explicit isolation rule.

    Numbered prose alone is not enough to change semantic complexity.  This
    narrow guard applies only when the user explicitly says the goals are
    independent *and* directs the runtime to continue after an unavailable or
    partial goal.  That is an execution contract the model must not collapse
    into one all-or-nothing specialist call.
    """

    matches = list(re.finditer(r"(?<!\w)\((\d+)\)\s+", raw_query))
    if len(matches) < 2:
        return ()

    prefix = raw_query[: matches[0].start()]
    if not re.search(r"\bindependent(?:ly)?\b", prefix, re.IGNORECASE):
        return ()
    if not re.search(r"\bcontinue\b", raw_query, re.IGNORECASE):
        return ()
    if not re.search(
        r"\b(?:absent|blocked|fail(?:s|ed|ure)?|missing|partial|unavailable)\b",
        raw_query,
        re.IGNORECASE,
    ):
        return ()

    clauses: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_query)
        clause = raw_query[match.end() : end].strip()
        if clause:
            clauses.append(clause)
    return tuple(clauses)


def _preserve_explicit_independent_goals(
    raw_query: str,
    assessment: IntentComplexityAssessment,
) -> IntentComplexityAssessment:
    clauses = _explicit_independent_goal_clauses(raw_query)
    if (
        assessment.execution_class != "single_graph"
        or assessment.side_effect_intent != "none"
        or len(clauses) < 2
    ):
        return assessment

    requires_external_state = bool(
        set(assessment.requested_sources) & EXTERNAL_SOURCES
    )
    goals = tuple(
        IntentGoal(
            id=f"goal_{index}",
            objective=clause,
            work_domains=("independent_execution_unit",),
            requires_external_state=requires_external_state,
            requires_mutation=False,
        )
        for index, clause in enumerate(clauses, start=1)
    )
    return IntentComplexityAssessment(
        execution_class="multi_graph",
        goals=goals,
        needs_planning=True,
        missing_information=assessment.missing_information,
        clarification_needed=False,
        reasoning_summary=(
            "The user explicitly required independent numbered goals to "
            "continue when one goal is unavailable or partial."
        ),
        requested_sources=assessment.requested_sources,
        excluded_sources=assessment.excluded_sources,
        side_effect_intent="none",
    )


async def assess_intent_complexity(
    raw_query: str,
    *,
    llm_candidates: Sequence[Any],
    prompt_type: str | None = None,
    max_tokens: int = 1400,
    active_workspace: bool = False,
    workspace_label: str | None = None,
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
        raw_query.strip(),
        active_workspace=active_workspace,
        workspace_label=workspace_label,
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

    return _preserve_explicit_independent_goals(
        raw_query,
        parse_intent_complexity_response(response),
    )


__all__ = [
    "ExecutionClass",
    "SideEffectIntent",
    "RequestedSource",
    "REQUESTED_SOURCES",
    "IntentGoal",
    "IntentComplexityAssessment",
    "build_intent_complexity_prompt",
    "validate_intent_complexity_payload",
    "parse_intent_complexity_response",
    "assess_intent_complexity",
]
