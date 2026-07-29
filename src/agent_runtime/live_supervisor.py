from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import PurePosixPath
from typing import Any

from src.agent_runtime.supervisor_decision import parse_supervisor_decision
from src.agent_runtime.supervisor_reasoner import build_supervisor_prompt
from src.agent_runtime.structured_llm import invoke_validated_fallback_async
from src.agent_runtime.legacy_sse_adapter import to_legacy_payload
from src.agent_runtime.worker_dispatcher import (
    DispatchResult,
    builtin_worker_runners,
    dispatch_supervisor_decision_async,
)
from src.agent_runtime.tool_confirmation_store import (
    register_pending_confirmation,
)
from src.agent_runtime.tool_risk_mapper import (
    decide_tool_call_policy_action,
)
from src.llm_core import (
    llm_call_async,
    llm_call_async_with_fallback,
)


RUNTIME_VERSION = "general-supervisor-v1"
TASK_PLANNER_RUNTIME = "general-supervisor-v1"

AsyncLlmCaller = Callable[..., Awaitable[Any]]
AsyncDispatcher = Callable[..., Awaitable[DispatchResult]]


def _latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages or []):
        if str(message.get("role") or "").lower() != "user":
            continue

        content = message.get("content")

        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in ("content", "response", "output", "text"):
            item = value.get(key)
            if isinstance(item, str):
                return item

    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content

    raise ValueError("LLM response did not contain text")


def _response_model(value: Any, selected_model: str) -> str:
    if isinstance(value, dict):
        for key in ("actual_model", "model", "answered_by"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    candidate = getattr(value, "model", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return selected_model


def _frame(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _dispatch_payload(dispatch: DispatchResult) -> dict[str, Any]:
    return {
        "requested_capabilities": list(dispatch.requested_capabilities),
        "completed_capabilities": list(dispatch.completed_capabilities),
        "undispatched_capabilities": list(dispatch.undispatched_capabilities),
        "worker_results": [
            {
                "capability": result.capability,
                "status": result.status,
                "summary": result.summary,
                "data": result.data,
                "evidence": list(result.evidence),
                "missing_information": list(result.missing_information),
                "error": result.error,
            }
            for result in dispatch.worker_results
        ],
    }


def _evidence_references(
    evidence: tuple[dict[str, Any], ...],
) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        reference = {
            key: str(item[key])[:1000]
            for key in (
                "path",
                "source_path",
                "file",
                "url",
                "line",
                "line_start",
            )
            if item.get(key) not in (None, "")
        }
        if reference:
            references.append(reference)
    return references[:24]


def _source_contract(
    admission_assessment: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    requested = tuple(
        str(value)
        for value in (
            getattr(admission_assessment, "requested_sources", ())
            or ()
        )
    )
    excluded = tuple(
        str(value)
        for value in (
            getattr(admission_assessment, "excluded_sources", ())
            or ()
        )
    )
    return requested, excluded


def _validate_source_capabilities(
    decision: Any,
    *,
    requested_sources: tuple[str, ...],
    excluded_sources: tuple[str, ...],
) -> Any:
    selected = set(decision.selected_capabilities)
    required = {
        "authorized_workspace_files": "code_inspection",
        "indexed_workspace_knowledge": "knowledge_search",
        "personal_memory": "knowledge_search",
        "completed_research": "deep_research",
        "public_web": "web_search",
        "clarification_required": "ask_user",
    }

    for source in requested_sources:
        capability = required.get(source)
        if capability and capability not in selected:
            raise ValueError(
                f"Requested source {source} requires capability {capability}"
            )

    if (
        "web_search" in selected
        and (
            "public_web" not in requested_sources
            or "public_web" in excluded_sources
        )
    ):
        raise ValueError(
            "web_search is outside the admitted source contract"
        )

    if (
        "knowledge_search" in selected
        and not set(requested_sources) & {
            "indexed_workspace_knowledge",
            "personal_memory",
        }
    ):
        raise ValueError(
            "knowledge_search is outside the admitted source contract"
        )

    if (
        "deep_research" in selected
        and "completed_research" not in requested_sources
    ):
        raise ValueError(
            "deep_research is outside the admitted source contract"
        )

    return decision


def _admitted_objective(
    admission_assessment: Any,
    fallback: str,
) -> str:
    goals = tuple(
        getattr(admission_assessment, "goals", ())
        or ()
    )
    for goal in goals:
        objective = str(
            getattr(goal, "objective", "") or ""
        ).strip()
        if objective:
            return objective
    return fallback


def _admitted_retrieval_concepts(
    admission_assessment: Any,
) -> tuple[str, ...]:
    result: list[str] = []
    for goal in tuple(
        getattr(admission_assessment, "goals", ())
        or ()
    ):
        for concept in tuple(
            getattr(goal, "retrieval_concepts", ())
            or ()
        ):
            value = str(concept).strip()
            if value and value not in result:
                result.append(value)
    return tuple(result)


def _single_graph_plan(
    *,
    query: str,
    requested_sources: tuple[str, ...],
) -> dict[str, Any]:
    source_titles = {
        "authorized_workspace_files": "Inspect the authorized workspace",
        "indexed_workspace_knowledge": "Search indexed workspace knowledge",
        "personal_memory": "Retrieve relevant saved context",
        "completed_research": "Retrieve completed research",
        "public_web": "Research current public sources",
        "attached_document": "Review the attached document",
    }
    acquisition_title = next(
        (
            source_titles[source]
            for source in requested_sources
            if source in source_titles
        ),
        "Complete the requested specialist work",
    )
    return {
        "runtime": TASK_PLANNER_RUNTIME,
        "plan_id": f"single:{abs(hash(query))}",
        "revision": 0,
        "status": "running",
        "objective": "Complete the requested task",
        "detail": "Using the admitted source and one specialist.",
        "steps": [
            {
                "id": "source_acquisition",
                "objective": acquisition_title,
                "worker_label": "Specialist",
                "depends_on": [],
                "status": "pending",
            },
            {
                "id": "final_synthesis",
                "objective": "Prepare the final answer",
                "worker_label": "Selected model",
                "depends_on": ["source_acquisition"],
                "status": "pending",
            },
        ],
    }


def _parse_workspace_mutation_proposal(
    value: Any,
) -> dict[str, Any]:
    text = _text(value).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Mutation proposal must be valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Mutation proposal must be an object")
    if payload.get("tool") != "write_file":
        raise ValueError(
            "Only the approval-gated write_file action is supported"
        )
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("Mutation arguments must be an object")
    path = str(arguments.get("path") or "").strip().replace("\\", "/")
    content = arguments.get("content")
    if (
        not path
        or PurePosixPath(path).is_absolute()
        or ":" in path
        or ".." in PurePosixPath(path).parts
    ):
        raise ValueError(
            "Mutation path must remain relative to the authorized workspace"
        )
    if not isinstance(content, str):
        raise ValueError("write_file content must be text")
    normalized = {
        "tool": "write_file",
        "arguments": {
            "path": path,
            "content": content,
        },
    }
    decision = decide_tool_call_policy_action(
        normalized["tool"],
        normalized["arguments"],
    )
    if decision.action != "confirm":
        raise ValueError(
            "Mutation proposal is not eligible for exact confirmation"
        )
    return normalized


def _fallback_summary(
    dispatch: DispatchResult,
    reasoning_summary: str,
) -> str:
    summaries = [
        result.summary.strip()
        for result in dispatch.worker_results
        if result.summary and result.summary.strip()
    ]

    if summaries:
        return "\n\n".join(summaries)

    if dispatch.undispatched_capabilities:
        names = ", ".join(dispatch.undispatched_capabilities)
        return f"The Supervisor selected capabilities that are not executable yet: {names}."

    return reasoning_summary or "The Supervisor completed without producing a final response."


async def stream_general_supervisor_sse(
    endpoint_url: str,
    model: str,
    messages: list[dict[str, Any]],
    headers: dict[str, Any] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    prompt_type: str | None = None,
    session_id: str | None = None,
    owner: str | None = None,
    workspace: str | None = None,
    active_document: Any = None,
    disabled_tools: set[str] | None = None,
    tool_policy: Any = None,
    admission_assessment: Any = None,
    fallbacks: list[Any] | tuple[Any, ...] | None = None,
    llm_caller: AsyncLlmCaller = llm_call_async,
    dispatcher: AsyncDispatcher = dispatch_supervisor_decision_async,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """Run the model-backed General Supervisor through the existing SSE seam."""

    started_at = time.time()
    query = _latest_user_message(messages)

    if not query:
        raise ValueError("General Supervisor could not find a user message")

    runtime_llm_candidates = [
        (endpoint_url, model, headers),
        *list(fallbacks or ()),
    ]
    requested_sources, excluded_sources = _source_contract(
        admission_assessment
    )
    worker_objective = _admitted_objective(
        admission_assessment,
        query,
    )
    retrieval_concepts = _admitted_retrieval_concepts(
        admission_assessment
    )

    yield _frame({
        "type": "plan_update",
        "data": _single_graph_plan(
            query=query,
            requested_sources=requested_sources,
        ),
    })

    if (
        "authorized_workspace_files" in requested_sources
        and not workspace
    ):
        limitation = (
            "I cannot inspect the requested workspace because no authorized "
            "workspace is available for this turn. I did not substitute public "
            "web search."
        )
        yield _frame({"delta": limitation})
        yield _frame({
            "type": "metrics",
            "data": {
                "runtime": RUNTIME_VERSION,
                "selected_model": model,
                "presenting_model": None,
                "model": None,
                "response_origin": "workspace_unavailable",
                "requested_sources": list(requested_sources),
                "excluded_sources": list(excluded_sources),
                "fallback_used": False,
                "fallback_scope": "none",
                "task_planner": True,
                "plan_status": "failed",
                "langgraph_runtime": True,
            },
        })
        yield "data: [DONE]\n\n"
        return

    async def call_runtime_llm(
        llm_messages: list[dict[str, Any]],
        **call_kwargs: Any,
    ) -> Any:
        # Explicit injection remains available for tests/internal callers.
        # Production follows Odysseus's canonical fallback executor.
        if llm_caller is not llm_call_async:
            return await llm_caller(
                endpoint_url,
                model,
                llm_messages,
                headers=headers,
                **call_kwargs,
            )

        return await llm_call_async_with_fallback(
            runtime_llm_candidates,
            messages=llm_messages,
            **call_kwargs,
        )

    async def call_structured_runtime_llm(
        candidates: list[Any],
        *,
        messages: list[dict[str, Any]],
        **call_kwargs: Any,
    ) -> Any:
        return await call_runtime_llm(messages, **call_kwargs)

    side_effect_intent = str(
        getattr(
            admission_assessment,
            "side_effect_intent",
            "none",
        )
        or "none"
    )
    if side_effect_intent == "approval_gated_mutation":
        if not session_id or not workspace:
            yield _frame({
                "delta": (
                    "I cannot prepare an exact durable approval because this "
                    "turn has no authorized workspace or session identity. "
                    "Nothing was written."
                ),
            })
            yield _frame({
                "type": "metrics",
                "data": {
                    "runtime": RUNTIME_VERSION,
                    "response_origin": "approval_unavailable",
                    "fallback_used": False,
                    "fallback_scope": "none",
                    "task_planner": True,
                    "plan_status": "failed",
                    "langgraph_runtime": True,
                },
            })
            yield "data: [DONE]\n\n"
            return

        mutation_prompt = "\n".join([
            "You are the bounded action planner for an already-admitted "
            "approval-gated workspace mutation.",
            "Do not execute anything and do not answer the user.",
            "Translate the request into exactly one write_file proposal.",
            "The path must be relative to the authorized workspace.",
            "Preserve the requested filename and content exactly.",
            "Return only JSON with this shape:",
            '{"tool":"write_file","arguments":{"path":"relative/path",'
            '"content":"exact text"}}',
            "",
            "USER REQUEST:",
            query,
        ])

        proposal_response = await invoke_validated_fallback_async(
            call_structured_runtime_llm,
            runtime_llm_candidates,
            mutation_prompt,
            _parse_workspace_mutation_proposal,
            call_kwargs={
                "temperature": 0.0,
                "max_tokens": 1200,
                "prompt_type": prompt_type,
            },
            max_rounds=2,
        )
        proposal = _parse_workspace_mutation_proposal(
            proposal_response
        )
        call_id = (
            "workspace_mutation_"
            f"{abs(hash((session_id, query, json.dumps(proposal, sort_keys=True))))}"
        )
        record = register_pending_confirmation(
            session_id=session_id,
            call_id=call_id,
            tool_name=proposal["tool"],
            arguments=proposal["arguments"],
            reason="Exact workspace mutation requires user approval.",
            policy_action="confirm",
            owner=owner,
            side_effect_risk="guided",
        )
        exact_action = (
            f"write_file({json.dumps(proposal['arguments'], ensure_ascii=False, sort_keys=True)})"
        )
        yield _frame({
            "type": "plan_update",
            "data": {
                **_single_graph_plan(
                    query=query,
                    requested_sources=requested_sources,
                ),
                "revision": 1,
                "detail": "Waiting for approval of the exact action.",
                "steps": [
                    {
                        "id": "source_acquisition",
                        "objective": "Confirm the exact workspace write",
                        "worker_label": "Approval gate",
                        "status": "blocked",
                    },
                    {
                        "id": "final_synthesis",
                        "objective": "Report the approved result",
                        "worker_label": "Selected model",
                        "depends_on": ["source_acquisition"],
                        "status": "pending",
                    },
                ],
            },
        })
        yield _frame({
            "type": "ask_user",
            "data": {
                "interaction_mode": "tool_confirmation",
                "question": f"Approve this exact action once?\n{exact_action}",
                "options": [
                    {
                        "label": "Yes",
                        "description": "Approve this exact action once.",
                    },
                    {
                        "label": "No",
                        "description": "Reject it without writing.",
                    },
                ],
                "multi": False,
                "session_id": session_id,
                "call_id": call_id,
                "fingerprint": record.fingerprint,
                "tool": proposal["tool"],
                "reason": record.reason,
                "confirmation_available": True,
            },
        })
        yield _frame({
            "type": "metrics",
            "data": {
                "runtime": RUNTIME_VERSION,
                "selected_model": model,
                "presenting_model": None,
                "model": None,
                "response_origin": "approval_required",
                "requested_sources": list(requested_sources),
                "excluded_sources": list(excluded_sources),
                "fallback_used": False,
                "fallback_scope": "none",
                "task_planner": True,
                "plan_status": "approval_required",
                "langgraph_runtime": True,
            },
        })
        yield "data: [DONE]\n\n"
        return

    decision_prompt = build_supervisor_prompt(
        query,
        requested_sources=requested_sources,
        excluded_sources=excluded_sources,
    )

    def parse_source_constrained_decision(value: Any) -> Any:
        decision = parse_supervisor_decision(
            _text(value).strip()
        )
        return _validate_source_capabilities(
            decision,
            requested_sources=requested_sources,
            excluded_sources=excluded_sources,
        )

    decision_response = await invoke_validated_fallback_async(
        call_structured_runtime_llm,
        runtime_llm_candidates,
        decision_prompt,
        parse_source_constrained_decision,
        call_kwargs={
            "temperature": 0.0,
            "max_tokens": min(max_tokens, 1400),
            "prompt_type": prompt_type,
        },
        max_rounds=2,
    )

    decision = parse_source_constrained_decision(decision_response)

    if decision.clarification_needed:
        missing = ", ".join(decision.missing_information)
        clarification = (
            f"I need one detail before I can continue: {missing}"
            if missing
            else "I need one clarification before I can continue safely."
        )

        yield _frame({"delta": clarification})
        yield _frame({
            "type": "metrics",
            "data": {
                "runtime": RUNTIME_VERSION,
                "selected_model": model,
                "presenting_model": None,
                "model": None,
                "worker": None,
                "fallback_used": False,
                "fallback_scope": "none",
                "response_origin": "runtime_clarification",
                "elapsed_s": round(time.time() - started_at, 3),
                "selected_capabilities": list(decision.selected_capabilities),
                "requested_sources": list(requested_sources),
                "excluded_sources": list(excluded_sources),
                "clarification_needed": True,
                "task_planner": True,
                "plan_status": "failed",
                "langgraph_runtime": True,
            },
        })
        yield "data: [DONE]\n\n"
        return

    supervisor_state = {
        "missing_information": list(decision.missing_information),
        "capability_actions": dict(decision.capability_actions or {}),
        "worker_context": {
            "query": query,
            "topic": worker_objective,
            "research_question": worker_objective,
            "requested_capability": worker_objective,
            "arguments": {},
            "owner": owner,
            "workspace": workspace,
            "active_doc": active_document,
            "root_id": kwargs.get("root_id"),
            "disabled_tools": tuple(disabled_tools or ()),
            "tool_policy": tool_policy,
            "requested_sources": requested_sources,
            "excluded_sources": excluded_sources,
            "retrieval_concepts": retrieval_concepts,
        },
    }

    dispatch_kwargs = {
        "supervisor_state": supervisor_state,
        "session_id": session_id,
    }
    if dispatcher is dispatch_supervisor_decision_async:
        dispatch_kwargs["worker_runners"] = builtin_worker_runners(
            code_finalizer_candidates=runtime_llm_candidates,
            code_finalizer_prompt_type=prompt_type,
        )

    dispatch = await dispatcher(
        decision,
        worker_objective,
        **dispatch_kwargs,
    )

    evidence_payload = _dispatch_payload(dispatch)
    for index, result in enumerate(dispatch.worker_results, start=1):
        for runtime_event in result.data.get("runtime_events", ()):
            if runtime_event.get("type") != "tool_confirmation_required":
                continue
            projected = to_legacy_payload(runtime_event)
            if isinstance(projected, dict):
                yield _frame(projected)

        call_id = f"supervisor_{result.capability}_{index}"
        yield _frame({
            "type": "tool_start",
            "id": call_id,
            "call_id": call_id,
            "tool": result.capability,
            "command": query,
            "plan_step_id": "source_acquisition",
        })
        yield _frame({
            "type": "tool_output",
            "id": call_id,
            "call_id": call_id,
            "tool": result.capability,
            "command": query,
            "output": str(result.summary or result.error or "")[:8000],
            "success": result.status == "completed",
            "status": result.status,
            "plan_step_id": "source_acquisition",
        })

    synthesis_prompt = "\n".join([
        "You are the final-response stage of the Odysseus General Supervisor.",
        "",
        "Answer the user using only the worker results and evidence below.",
        "Do not claim an operation succeeded unless a worker status says completed.",
        "Clearly state when confirmation, clarification, or additional execution is required.",
        "Do not mention internal class names, routing code, or JSON schemas.",
        "State which admitted source supplied the evidence. Do not claim that "
        "a source was inspected unless a completed worker result contains "
        "evidence from it.",
        "",
        "USER REQUEST:",
        query,
        "",
        "SUPERVISOR RATIONALE:",
        decision.reasoning_summary,
        "",
        "WORKER RESULTS:",
        json.dumps(
            {
                "requested_capabilities": list(
                    dispatch.requested_capabilities
                ),
                "worker_results": [
                    {
                        "capability": result.capability,
                        "status": result.status,
                        "summary": str(
                            result.summary or ""
                        )[:10000],
                        "missing_information": list(
                            result.missing_information
                        ),
                        "evidence_references": _evidence_references(
                            result.evidence
                        ),
                        "error": result.error,
                    }
                    for result in dispatch.worker_results
                ],
            },
            ensure_ascii=False,
            default=str,
        ),
        "",
        "SOURCE PROVENANCE:",
        json.dumps(
            {
                "requested_sources": requested_sources,
                "excluded_sources": excluded_sources,
            },
            ensure_ascii=False,
        ),
    ])

    response_origin = "selected_model"
    presenting_model: str | None = model
    synthesis_error: str | None = None
    try:
        final_response = await call_runtime_llm(
            [{"role": "user", "content": synthesis_prompt}],
            temperature=temperature,
            max_tokens=(
                max_tokens
                if max_tokens and max_tokens >= 8192
                else 8192
            ),
            prompt_type=prompt_type,
        )
        final_text = _text(final_response).strip()
        presenting_model = _response_model(final_response, model)
        if not final_text:
            compact_results = [
                {
                    "capability": result.capability,
                    "status": result.status,
                    "summary": str(result.summary or "")[:8000],
                }
                for result in dispatch.worker_results
            ]
            retry_prompt = "\n".join([
                "Answer the user's request concisely from the worker summaries.",
                "Do not discuss internal routing or hidden reasoning.",
                "",
                "USER REQUEST:",
                query,
                "",
                "WORKER SUMMARIES:",
                json.dumps(
                    compact_results,
                    ensure_ascii=False,
                    default=str,
                ),
            ])
            final_response = await call_runtime_llm(
                [{"role": "user", "content": retry_prompt}],
                temperature=min(temperature, 0.3),
                max_tokens=8192,
                prompt_type=prompt_type,
            )
            final_text = _text(final_response).strip()
            presenting_model = _response_model(final_response, model)
            if not final_text:
                raise ValueError(
                    "final synthesis returned an empty response "
                    "after one selected-model retry"
                )
    except Exception as exc:
        synthesis_error = f"{type(exc).__name__}: {exc}"
        response_origin = "controlled_runtime_failure"
        presenting_model = None
        final_text = (
            "The LangGraph workers finished, but the selected final-synthesis "
            "model did not produce a usable answer. The runtime stopped without "
            "claiming that a model authored the worker output."
        )

    yield _frame({"delta": final_text})
    worker_names = list(dispatch.requested_capabilities)
    fallback_used = any(
        bool(result.data.get("fallback_used"))
        for result in dispatch.worker_results
    )
    yield _frame({
        "type": "metrics",
        "data": {
            "runtime": RUNTIME_VERSION,
            "selected_model": model,
            "presenting_model": presenting_model,
            "model": presenting_model,
            "worker": (
                worker_names[0]
                if len(worker_names) == 1
                else worker_names
            ),
            "fallback_used": fallback_used,
            "fallback_scope": (
                "legacy_plan_step_only" if fallback_used else "none"
            ),
            "response_origin": response_origin,
            "synthesis_error": synthesis_error,
            "elapsed_s": round(time.time() - started_at, 3),
            "selected_capabilities": list(decision.selected_capabilities),
            "requested_sources": list(requested_sources),
            "excluded_sources": list(excluded_sources),
            "completed_capabilities": list(dispatch.completed_capabilities),
            "undispatched_capabilities": list(dispatch.undispatched_capabilities),
            "clarification_needed": False,
            "task_planner": True,
            "plan_status": (
                "completed"
                if (
                    response_origin == "selected_model"
                    and dispatch.worker_results
                    and all(
                        result.status == "completed"
                        for result in dispatch.worker_results
                    )
                )
                else "failed"
            ),
            "execution_order": [
                "source_acquisition",
                "final_synthesis",
            ],
            "langgraph_runtime": True,
        },
    })
    yield "data: [DONE]\n\n"


__all__ = ["RUNTIME_VERSION", "stream_general_supervisor_sse"]
