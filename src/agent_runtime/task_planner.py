from __future__ import annotations

import inspect
import json
from typing import Any, Awaitable, Protocol

from src.agent_runtime.intent_complexity import (
    IntentComplexityAssessment,
)
from src.agent_runtime.task_plan import (
    TaskPlan,
    validate_task_plan_payload,
)


class TaskPlannerModel(Protocol):
    """Minimal model seam for semantic task planning.

    Production adapters may return either an immediate model response or an
    awaitable response. The synchronous and asynchronous planner entry points
    enforce the matching invocation contract.
    """

    def invoke(self, prompt: str) -> Any | Awaitable[Any]:
        ...


def _assessment_payload(
    assessment: IntentComplexityAssessment,
) -> dict[str, Any]:
    return {
        "execution_class": assessment.execution_class,
        "goals": [
            {
                "id": goal.id,
                "objective": goal.objective,
                "work_domains": list(goal.work_domains),
                "requires_external_state": (
                    goal.requires_external_state
                ),
                "requires_mutation": goal.requires_mutation,
            }
            for goal in assessment.goals
        ],
        "needs_planning": assessment.needs_planning,
        "missing_information": list(
            assessment.missing_information
        ),
        "clarification_needed": (
            assessment.clarification_needed
        ),
        "reasoning_summary": assessment.reasoning_summary,
    }


def build_task_planner_prompt(
    query: str,
    assessment: IntentComplexityAssessment,
) -> str:
    """Build a backend-independent dependency-planning prompt."""

    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("Task Planner query must not be empty")

    if assessment.execution_class != "multi_graph":
        raise ValueError(
            "Task Planner requires a multi_graph assessment"
        )

    if not assessment.needs_planning:
        raise ValueError(
            "multi_graph assessment must require planning"
        )

    schema = {
        "objective": "overall user-facing objective",
        "steps": [
            {
                "id": "stable_short_step_id",
                "objective": (
                    "semantic unit of work to accomplish"
                ),
                "depends_on": [
                    "prior_step_id"
                ],
                "work_domains": [
                    "semantic_domain_label"
                ],
                "requires_external_state": False,
                "requires_mutation": False,
            }
        ],
        "reasoning_summary": (
            "brief explanation of decomposition and dependencies"
        ),
    }

    rules = [
        "You are the Odysseus Task Planner.",
        "",
        "Your job is semantic decomposition only.",
        "Convert the validated multi-graph assessment into a dependency-aware "
        "TaskPlan.",
        "",
        "PERMANENT OWNERSHIP RULES:",
        "1. Describe WHAT work must be accomplished and WHICH prior steps each "
        "step depends on.",
        "2. Do NOT select workers, tools, capabilities, actions, providers, "
        "models, runtimes, backends, MCP servers, LangGraph, or Legacy.",
        "3. Missing native LangGraph coverage must NEVER remove, merge, or "
        "downgrade a semantically necessary step. A downstream Execution "
        "Resolver may deliberately choose Legacy or another compatible backend.",
        "4. Keep independently acquirable prerequisite work as separate steps "
        "when their outputs are independently useful or can execute in "
        "parallel.",
        "5. Add an explicit dependent synthesis/decision/action step when the "
        "final result genuinely depends on combining outputs from earlier "
        "steps.",
        "6. Do not split one bounded acquisition workflow merely because it "
        "contains multiple topics, documents, searches, or internal tool "
        "calls.",
        "7. Dependencies must reference step ids from the same plan. The plan "
        "must be acyclic.",
        "8. requires_external_state means the step needs information not "
        "already contained in the current request/context.",
        "9. requires_mutation means the step changes persistent or external "
        "state. Planning never grants permission to perform that mutation.",
        "10. work_domains are semantic labels only. They are not capability or "
        "backend names.",
        "11. Return only one JSON object matching the schema. Do not include "
        "markdown fences or commentary.",
        "",
        "USER REQUEST:",
        normalized_query,
        "",
        "VALIDATED INTENT / COMPLEXITY ASSESSMENT:",
        json.dumps(
            _assessment_payload(assessment),
            ensure_ascii=False,
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


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        for key in ("content", "text", "response"):
            value = response.get(key)
            if isinstance(value, str):
                return value.strip()

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()

    return str(response or "").strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    normalized = str(text or "").strip()
    if not normalized:
        raise ValueError(
            "Task Planner model returned an empty response"
        )

    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        start = normalized.find("{")
        end = normalized.rfind("}")

        if start < 0 or end <= start:
            raise ValueError(
                "Task Planner model did not return a JSON object"
            )

        try:
            payload = json.loads(
                normalized[start : end + 1]
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Task Planner model returned invalid JSON"
            ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Task Planner model response must be a JSON object"
        )

    return payload


def parse_task_plan_response(response: Any) -> TaskPlan:
    """Parse and validate model output through the TaskPlan contract.

    Already-decoded plan objects are validated directly. Model response
    wrappers and text responses still flow through the JSON extraction path.
    """

    if isinstance(response, dict) and (
        "objective" in response
        or "steps" in response
        or "reasoning_summary" in response
    ):
        payload = response
    else:
        payload = _extract_json_object(
            _response_text(response)
        )

    return validate_task_plan_payload(payload)


def _validate_multi_graph_plan(
    plan: TaskPlan,
) -> TaskPlan:
    if len(plan.steps) < 2:
        raise ValueError(
            "multi_graph TaskPlan must contain at least two "
            "semantic steps"
        )
    return plan


def plan_task_from_assessment(
    query: str,
    assessment: IntentComplexityAssessment,
    *,
    model: TaskPlannerModel,
) -> TaskPlan:
    """Create a validated semantic plan through a synchronous model seam.

    The caller owns model/provider resolution. Async production adapters must
    use :func:`plan_task_from_assessment_async` so the event loop is never
    blocked by a synchronous compatibility shim.
    """

    prompt = build_task_planner_prompt(
        query,
        assessment,
    )
    response = model.invoke(prompt)

    if inspect.isawaitable(response):
        if inspect.iscoroutine(response):
            response.close()
        raise TypeError(
            "TaskPlannerModel returned an awaitable; use "
            "plan_task_from_assessment_async"
        )

    return _validate_multi_graph_plan(
        parse_task_plan_response(response)
    )


async def plan_task_from_assessment_async(
    query: str,
    assessment: IntentComplexityAssessment,
    *,
    model: TaskPlannerModel,
) -> TaskPlan:
    """Create a validated semantic plan without blocking async runtimes."""

    prompt = build_task_planner_prompt(
        query,
        assessment,
    )
    response = model.invoke(prompt)

    if inspect.isawaitable(response):
        response = await response

    return _validate_multi_graph_plan(
        parse_task_plan_response(response)
    )


__all__ = [
    "TaskPlannerModel",
    "build_task_planner_prompt",
    "parse_task_plan_response",
    "plan_task_from_assessment",
    "plan_task_from_assessment_async",
]
