from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from src.agent_runtime.capability_registry import build_capability_registry
from src.agent_runtime.dependency_reasoning import (
    DEPENDENCY_REASONING_CAPABILITY,
    build_dependency_reasoning_runner,
    dependency_reasoning_capability,
)
from src.agent_runtime.execution_resolver import (
    ExecutionResolverModel,
    ExecutionSurfaceDescriptor,
    build_execution_surface_catalog,
)
from src.agent_runtime.intent_complexity import IntentComplexityAssessment
from src.agent_runtime.legacy_step_executor import (
    LEGACY_SURFACE_ID,
    LegacyStepExecutor,
    LegacyStreamFactory,
)
from src.agent_runtime.replan_controller import (
    ReplanContextProvider,
    ReplanControlResult,
    run_task_plan_with_replanning,
)
from src.agent_runtime.run_lifecycle import (
    LangGraphCancellationScope,
    LangGraphRunLifecycle,
)
from src.agent_runtime.step_action_resolver import (
    CapabilityActionModel,
    resolve_capability_action_async,
)
from src.agent_runtime.step_execution import (
    StepExecutionContext,
    StepSurfaceExecutor,
)
from src.agent_runtime.structured_llm import (
    invoke_validated_fallback_async,
)
from src.agent_runtime.supervisor_contracts import (
    CapabilityDescriptor,
    SupervisorState,
    WorkerHandoff,
    WorkerResult,
)
from src.agent_runtime.task_plan import TaskPlan
from src.agent_runtime.task_planner import (
    TaskPlannerModel,
    plan_task_from_assessment_async,
)
from src.agent_runtime.worker_dispatcher import (
    WorkerRunner,
    builtin_worker_runners,
)
from src.llm_core import llm_call_async_with_fallback


RUNTIME_VERSION = "p3c5b2-multi-graph-v1"
MAX_PLAN_STEPS = 16
AsyncLlmCaller = Callable[..., Awaitable[Any]]


def _frame(payload: dict[str, Any]) -> str:
    return (
        "data: "
        + json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
        + "\n\n"
    )


def _latest_user_message(
    messages: Sequence[Mapping[str, Any]],
) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


class _RuntimeFallbackModel:
    """Task-runtime model adapter backed by Odysseus's canonical fallbacks."""

    def __init__(
        self,
        candidates: Sequence[tuple],
        *,
        prompt_type: str | None,
        max_tokens: int,
        temperature: float = 0.0,
        llm_caller: AsyncLlmCaller = llm_call_async_with_fallback,
    ) -> None:
        self._candidates = list(candidates)
        self._prompt_type = prompt_type
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._llm_caller = llm_caller

    def _invoke(
        self,
        prompt: str,
        *,
        response_validator: Callable[[Any], Any] | None = None,
    ) -> Awaitable[Any]:
        call_kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "prompt_type": self._prompt_type,
        }

        if response_validator is not None:
            call_kwargs["response_validator"] = response_validator

        return self._llm_caller(
            self._candidates,
            **call_kwargs,
        )

    def invoke(self, prompt: str) -> Awaitable[Any]:
        return self._invoke(prompt)

    def invoke_validated(
        self,
        prompt: str,
        validator: Callable[[Any], Any],
    ) -> Awaitable[Any]:
        """Run validated fallbacks with one bounded repair round."""

        return invoke_validated_fallback_async(
            self._llm_caller,
            self._candidates,
            prompt,
            validator,
            call_kwargs={
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "prompt_type": self._prompt_type,
            },
            max_rounds=2,
        )


def _plan_payload(plan: TaskPlan) -> dict[str, Any]:
    return {
        "objective": plan.objective,
        "steps": [
            {
                "id": step.id,
                "objective": step.objective,
                "depends_on": list(step.depends_on),
                "work_domains": list(step.work_domains),
                "uses_dependency_state": step.uses_dependency_state,
                "requires_external_state": step.requires_external_state,
                "requires_mutation": step.requires_mutation,
            }
            for step in plan.steps
        ],
        "reasoning_summary": plan.reasoning_summary,
    }


def _bounded_value(
    value: Any,
    *,
    max_chars: int = 12000,
) -> Any:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        serialized = str(value)

    if len(serialized) <= max_chars:
        return value

    return {
        "truncated": True,
        "preview": serialized[:max_chars],
        "original_chars": len(serialized),
    }


def _worker_payload(result: WorkerResult) -> dict[str, Any]:
    return {
        "capability": result.capability,
        "status": result.status,
        "summary": result.summary,
        "data": _bounded_value(result.data),
        "evidence": _bounded_value(list(result.evidence), max_chars=8000),
        "missing_information": list(result.missing_information),
        "error": result.error,
    }


def _execution_payload(
    control: ReplanControlResult,
) -> dict[str, Any]:
    orchestration = control.orchestration
    step_payloads: list[dict[str, Any]] = []

    for step in orchestration.state.plan.steps:
        execution = next(
            item
            for item in orchestration.state.step_states
            if item.step_id == step.id
        )
        output = (
            _worker_payload(execution.output)
            if isinstance(execution.output, WorkerResult)
            else _bounded_value(execution.output)
        )
        step_payloads.append(
            {
                "id": step.id,
                "objective": step.objective,
                "depends_on": list(step.depends_on),
                "uses_dependency_state": step.uses_dependency_state,
                "status": execution.status,
                "output": output,
                "artifact_refs": list(execution.artifact_refs),
                "error": execution.error,
            }
        )

    return {
        "status": orchestration.status,
        "message": orchestration.message,
        "stopped_reason": control.stopped_reason,
        "revisions_applied": control.revisions_applied,
        "execution_order": list(orchestration.execution_order),
        "steps": step_payloads,
    }


def _fallback_final_text(
    query: str,
    control: ReplanControlResult,
) -> str:
    orchestration = control.orchestration
    summaries = [
        result.summary.strip()
        for result in orchestration.worker_results
        if result.summary.strip()
    ]

    if orchestration.status == "completed":
        if summaries:
            return "\n\n".join(summaries)
        return "The planned work completed, but no user-facing summary was produced."

    request = orchestration.replan_request
    if request is not None:
        reason = request.reason.strip()
        if reason:
            return (
                "I completed the work that was currently possible, but I need "
                f"one clarification before continuing: {reason}"
            )

    detail = orchestration.message.strip()
    if summaries:
        return (
            "\n\n".join(summaries)
            + "\n\n"
            + (
                "The remaining planned work did not complete."
                + (f" {detail}" if detail else "")
            )
        )

    return (
        "I could not complete the full multi-step request."
        + (f" {detail}" if detail else "")
    )


def _synthesis_prompt(
    query: str,
    plan: TaskPlan,
    control: ReplanControlResult,
) -> str:
    return "\n".join([
        "You are the final-response stage of the Odysseus multi-graph coordinator.",
        "",
        "Answer the user using only the validated plan execution results below.",
        "Do not claim a step or mutation succeeded unless its status and worker",
        "result explicitly say it completed.",
        "Preserve useful partial results, and clearly state any clarification,",
        "confirmation, execution failure, or remaining work.",
        "Do not mention internal Python classes, routing code, or JSON schemas.",
        "",
        "USER REQUEST:",
        query,
        "",
        "VALIDATED TASK PLAN:",
        json.dumps(
            _plan_payload(plan),
            ensure_ascii=False,
            default=str,
        ),
        "",
        "EXECUTION RESULTS:",
        json.dumps(
            _execution_payload(control),
            ensure_ascii=False,
            default=str,
        ),
    ])


def _clarification_question(
    control: ReplanControlResult,
) -> str | None:
    request = control.orchestration.replan_request
    if request is None:
        return None

    reason = str(request.reason or "").strip()
    return (
        reason
        or "I need one clarification before the remaining plan can continue."
    )


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _action_aware_worker_runners(
    base_runners: Mapping[str, WorkerRunner],
    *,
    action_model: CapabilityActionModel,
    capability_registry: Mapping[
        str,
        CapabilityDescriptor,
    ] | None = None,
) -> dict[str, WorkerRunner]:
    """Wrap native workers with validated action selection and failure normalization."""

    registry = (
        build_capability_registry()
        if capability_registry is None
        else dict(capability_registry)
    )
    wrapped: dict[str, WorkerRunner] = {}

    for capability_name, base_runner in base_runners.items():
        descriptor = registry.get(capability_name)

        async def run(
            handoff: WorkerHandoff,
            *,
            session_id: str | None = None,
            _capability_name: str = capability_name,
            _descriptor=descriptor,
            _base_runner=base_runner,
        ) -> WorkerResult:
            effective_handoff = handoff
            selected_action: str | None = None

            try:
                if (
                    _descriptor is not None
                    and _descriptor.supported_actions
                ):
                    action_resolution = await resolve_capability_action_async(
                        handoff.objective,
                        _descriptor,
                        model=action_model,
                        context=handoff.context,
                    )

                    if action_resolution.clarification_needed:
                        return WorkerResult(
                            capability=_capability_name,
                            status="needs_clarification",
                            summary=(
                                action_resolution.reasoning_summary
                                or (
                                    "The selected capability requires "
                                    "clarification before execution."
                                )
                            ),
                            missing_information=(
                                action_resolution.missing_information
                            ),
                        )

                    selected_action = action_resolution.action
                    context = dict(handoff.context)
                    if selected_action:
                        context["action"] = selected_action
                    effective_handoff = replace(
                        handoff,
                        context=context,
                    )

                value = _base_runner(
                    effective_handoff,
                    session_id=session_id,
                )
                result = await _await_if_needed(value)

                if not isinstance(result, WorkerResult):
                    raise TypeError(
                        "Worker runner returned "
                        f"{type(result).__name__}, expected WorkerResult"
                    )

                data = dict(result.data)
                data.setdefault(
                    "execution_surface",
                    f"langgraph:{_capability_name}",
                )
                if selected_action:
                    data.setdefault(
                        "resolved_action",
                        selected_action,
                    )

                return replace(
                    result,
                    data=data,
                )

            except Exception as exc:
                return WorkerResult(
                    capability=_capability_name,
                    status="failed",
                    summary=(
                        f"{_capability_name} execution failed."
                    ),
                    data={
                        "execution_surface": (
                            f"langgraph:{_capability_name}"
                        ),
                        "resolved_action": selected_action,
                    },
                    error=f"{type(exc).__name__}: {exc}",
                )

        wrapped[capability_name] = run

    return wrapped


def _base_supervisor_state(
    query: str,
    *,
    owner: str | None,
    workspace: str | None,
    active_document: Any,
    active_email: dict[str, str] | None,
    disabled_tools: set[str] | None,
    tool_policy: Any,
    root_id: str | None,
    uploaded_files: Sequence[dict[str, Any]],
) -> SupervisorState:
    return {
        "raw_query": query,
        "missing_information": [],
        "worker_context": {
            "query": query,
            "topic": query,
            "research_question": query,
            "requested_capability": query,
            "arguments": {},
            "owner": owner,
            "workspace": workspace,
            "active_doc": active_document,
            "active_email": active_email,
            "root_id": root_id,
            "disabled_tools": tuple(disabled_tools or ()),
            "tool_policy": tool_policy,
            "uploaded_files": list(uploaded_files),
        },
    }


def _tool_frames_and_events(
    control: ReplanControlResult,
) -> tuple[list[str], list[dict[str, Any]]]:
    frames: list[str] = []
    events: list[dict[str, Any]] = []

    for index, step in enumerate(
        control.orchestration.state.plan.steps,
        start=1,
    ):
        execution = next(
            item
            for item in control.orchestration.state.step_states
            if item.step_id == step.id
        )
        result = (
            execution.output
            if isinstance(execution.output, WorkerResult)
            else None
        )
        capability = (
            result.capability
            if result is not None
            else "plan_step"
        )
        summary = (
            result.summary
            if result is not None
            else execution.error
            or f"PlanStep status: {execution.status}"
        )
        success = (
            execution.status == "completed"
            and result is not None
            and result.status == "completed"
        )
        call_id = f"multi_graph_{step.id}_{index}"

        frames.append(
            _frame({
                "type": "tool_start",
                "id": call_id,
                "call_id": call_id,
                "tool": capability,
                "command": step.objective,
                "plan_step_id": step.id,
            })
        )
        frames.append(
            _frame({
                "type": "tool_output",
                "id": call_id,
                "call_id": call_id,
                "tool": capability,
                "command": step.objective,
                "output": str(summary or "")[:8000],
                "success": success,
                "status": (
                    result.status
                    if result is not None
                    else execution.status
                ),
                "plan_step_id": step.id,
            })
        )
        events.append({
            "round": index,
            "tool": capability,
            "command": step.objective,
            "output": str(summary or "")[:8000],
            "exit_code": 0 if success else 1,
            "plan_step_id": step.id,
        })

        if result is not None:
            nested = result.data.get("tool_events")
            if isinstance(nested, list):
                events.extend(
                    dict(item)
                    for item in nested
                    if isinstance(item, dict)
                )

    return frames, events


async def stream_multi_graph_coordinator_sse(
    endpoint_url: str,
    model: str,
    messages: list[dict[str, Any]],
    headers: dict[str, Any] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    prompt_type: str | None = None,
    max_rounds: int = 8,
    max_tool_calls: int = 0,
    context_length: int = 0,
    active_document: Any = None,
    active_email: dict[str, str] | None = None,
    session_id: str | None = None,
    disabled_tools: set[str] | None = None,
    owner: str | None = None,
    fallbacks: Sequence[tuple] | None = None,
    tool_policy: Any = None,
    workspace: str | None = None,
    forced_tools: set[str] | None = None,
    uploaded_files: Sequence[dict[str, Any]] | None = None,
    admission_assessment: IntentComplexityAssessment | None = None,
    root_id: str | None = None,
    planner_model: TaskPlannerModel | None = None,
    resolver_model: ExecutionResolverModel | None = None,
    replanner_model: TaskPlannerModel | None = None,
    action_model: CapabilityActionModel | None = None,
    final_model: TaskPlannerModel | None = None,
    worker_runners: Mapping[str, WorkerRunner] | None = None,
    surfaces: tuple[ExecutionSurfaceDescriptor, ...] | None = None,
    surface_executors: Mapping[str, StepSurfaceExecutor] | None = None,
    replan_context_provider: ReplanContextProvider | None = None,
    legacy_stream_factory: LegacyStreamFactory | None = None,
    llm_caller: AsyncLlmCaller = llm_call_async_with_fallback,
    max_plan_steps: int = MAX_PLAN_STEPS,
    max_step_executions: int = 64,
    max_revisions: int = 3,
    run_id: str | None = None,
    thread_id: str | None = None,
    checkpoint_ns: str | None = None,
    cancellation_scope: LangGraphCancellationScope | None = None,
    checkpoint_store: Any = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """Execute an admitted multi-graph turn through permanent runtime owners."""

    started_at = time.time()
    lifecycle = LangGraphRunLifecycle.create(
        session_id=session_id,
        run_id=run_id,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        owner=owner,
        cancellation=cancellation_scope,
        checkpoint_store=checkpoint_store,
    )
    query = _latest_user_message(messages)

    if not query:
        raise ValueError(
            "Multi-graph coordinator could not find a user message"
        )

    assessment = admission_assessment
    if (
        assessment is None
        or assessment.execution_class != "multi_graph"
        or not assessment.needs_planning
    ):
        raise ValueError(
            "Multi-graph coordinator requires an admitted multi_graph assessment"
        )

    yield _frame(lifecycle.event({
        "type": "runtime_run",
        "data": {
            "status": lifecycle.status,
            "admission_result": "multi_graph",
            "fallback_policy": "plan_step_only",
            "cancellation_scope": lifecycle.identity.run_id,
        },
    }))

    if assessment.clarification_needed:
        missing = ", ".join(assessment.missing_information)
        question = (
            f"I need one detail before I can plan this safely: {missing}"
            if missing
            else "I need one clarification before I can plan this safely."
        )
        yield _frame({
            "type": "ask_user",
            "data": {
                "question": question,
                "options": [],
                "multi": False,
            },
        })
        yield _frame({"delta": question})
        yield _frame({
            "type": "metrics",
            "data": {
                "runtime": RUNTIME_VERSION,
                "model": model,
                "requested_model": model,
                "elapsed_s": round(time.time() - started_at, 3),
                "multi_graph": True,
                "clarification_needed": True,
                "tool_calls": 0,
                "tool_events": [],
                "langgraph_runtime": True,
            },
        })
        terminal = lifecycle.finish(
            "AWAITING_APPROVAL",
            stage="admission",
            reason=question,
            completed_steps=[],
            whole_query_legacy_fallback=False,
        )
        if terminal is not None:
            yield _frame(terminal)
        yield "data: [DONE]\n\n"
        return

    candidates = [
        (endpoint_url, model, headers),
        *list(fallbacks or ()),
    ]

    planner = planner_model or _RuntimeFallbackModel(
        candidates,
        prompt_type=prompt_type,
        max_tokens=2400,
        llm_caller=llm_caller,
    )
    resolver = resolver_model or _RuntimeFallbackModel(
        candidates,
        prompt_type=prompt_type,
        max_tokens=1400,
        llm_caller=llm_caller,
    )
    replanner = replanner_model or _RuntimeFallbackModel(
        candidates,
        prompt_type=prompt_type,
        max_tokens=2400,
        llm_caller=llm_caller,
    )
    action_selector = action_model or _RuntimeFallbackModel(
        candidates,
        prompt_type=prompt_type,
        max_tokens=900,
        llm_caller=llm_caller,
    )
    synthesizer = final_model or _RuntimeFallbackModel(
        candidates,
        prompt_type=prompt_type,
        max_tokens=max_tokens,
        temperature=temperature,
        llm_caller=llm_caller,
    )

    try:
        plan = await plan_task_from_assessment_async(
            query,
            assessment,
            model=planner,
        )

        if len(plan.steps) > max_plan_steps:
            raise ValueError(
                f"Task Planner produced {len(plan.steps)} steps; "
                f"the safety limit is {max_plan_steps}"
            )
    except asyncio.CancelledError:
        lifecycle.cancellation.cancel()
        raise
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        error_text = (
            "I could not construct a valid bounded plan for this multi-step "
            "request. No work was started and no Legacy fallback was used. "
            f"{reason}"
        )
        yield _frame({"delta": error_text})
        terminal = lifecycle.finish(
            "FAILED",
            stage="planner",
            reason=reason,
            retryable=True,
            completed_steps=[],
            whole_query_legacy_fallback=False,
        )
        if terminal is not None:
            yield _frame(terminal)
        yield _frame({
            "type": "metrics",
            "data": {
                "runtime": RUNTIME_VERSION,
                "model": model,
                "requested_model": model,
                "elapsed_s": round(time.time() - started_at, 3),
                "multi_graph": True,
                "plan_status": "planner_failed",
                "terminal_status": "FAILED",
                "whole_query_legacy_fallback": False,
                "tool_calls": 0,
                "tool_events": [],
                "langgraph_runtime": True,
            },
        })
        yield "data: [DONE]\n\n"
        return

    yield _frame(lifecycle.event({
        "type": "plan_update",
        "data": {
            "runtime": RUNTIME_VERSION,
            "revision": 1,
            "status": "planned",
            **_plan_payload(plan),
        },
    }))

    dependency_reasoning_descriptor = (
        dependency_reasoning_capability()
    )
    capability_registry = build_capability_registry(
        extra_capabilities=(
            dependency_reasoning_descriptor,
        ),
    )

    base_runners = (
        builtin_worker_runners(
            code_finalizer_candidates=candidates,
            code_finalizer_prompt_type=prompt_type,
        )
        if worker_runners is None
        else dict(worker_runners)
    )
    base_runners.setdefault(
        DEPENDENCY_REASONING_CAPABILITY,
        build_dependency_reasoning_runner(
            synthesizer
        ),
    )
    action_aware_runners = _action_aware_worker_runners(
        base_runners,
        action_model=action_selector,
        capability_registry=capability_registry,
    )

    legacy_executor_kwargs = {
        "endpoint_url": endpoint_url,
        "model": model,
        "original_query": query,
        "headers": headers,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_type": prompt_type,
        "max_rounds": max_rounds,
        "max_tool_calls": max_tool_calls,
        "context_length": context_length,
        "active_document": active_document,
        "active_email": active_email,
        "session_id": session_id,
        "disabled_tools": disabled_tools,
        "owner": owner,
        "fallbacks": tuple(fallbacks or ()),
        "tool_policy": tool_policy,
        "workspace": workspace,
        "forced_tools": forced_tools,
        "uploaded_files": list(uploaded_files or ()),
    }
    if legacy_stream_factory is not None:
        legacy_executor_kwargs["stream_factory"] = legacy_stream_factory

    legacy_executor = LegacyStepExecutor(
        **legacy_executor_kwargs,
    )

    executors = dict(surface_executors or {})
    executors.setdefault(
        LEGACY_SURFACE_ID,
        legacy_executor,
    )

    execution_surfaces = (
        build_execution_surface_catalog(
            capability_registry=capability_registry,
            worker_runners=action_aware_runners,
        )
        if surfaces is None
        else tuple(surfaces)
    )

    context = StepExecutionContext(
        supervisor_state=_base_supervisor_state(
            query,
            owner=owner,
            workspace=workspace,
            active_document=active_document,
            active_email=active_email,
            disabled_tools=disabled_tools,
            tool_policy=tool_policy,
            root_id=root_id or kwargs.get("root_id"),
            uploaded_files=uploaded_files or (),
        ),
        session_id=session_id,
    )

    try:
        control = await run_task_plan_with_replanning(
            query,
            plan,
            resolver_model=resolver,
            replanner_model=replanner,
            replan_context_provider=replan_context_provider,
            surfaces=execution_surfaces,
            context=context,
            worker_runners=action_aware_runners,
            surface_executors=executors,
            max_step_executions=max_step_executions,
            max_revisions=max_revisions,
        )
    except asyncio.CancelledError:
        lifecycle.cancellation.cancel()
        raise
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        error_text = (
            "The multi-step plan started but its execution coordinator failed: "
            f"{reason}"
        )
        yield _frame({"delta": error_text})
        terminal = lifecycle.finish(
            "FAILED",
            stage="coordinator",
            reason=reason,
            retryable=True,
            completed_steps=[],
            whole_query_legacy_fallback=False,
        )
        if terminal is not None:
            yield _frame(terminal)
        yield _frame({
            "type": "metrics",
            "data": {
                "runtime": RUNTIME_VERSION,
                "model": model,
                "requested_model": model,
                "elapsed_s": round(time.time() - started_at, 3),
                "multi_graph": True,
                "plan_status": "coordinator_error",
                "tool_calls": 0,
                "tool_events": [],
                "langgraph_runtime": True,
            },
        })
        yield "data: [DONE]\n\n"
        return

    frames, tool_events = _tool_frames_and_events(
        control
    )
    completed_step_ids = [
        step_id
        for step_id in control.orchestration.execution_order
        if next(
            item
            for item in control.orchestration.state.step_states
            if item.step_id == step_id
        ).status == "completed"
    ]
    dependency_results = {
        step_id: {
            "status": result.status,
            "summary": result.summary,
            "data": result.data,
            "evidence": list(result.evidence),
            "error": result.error,
        }
        for step_id, result in zip(
            control.orchestration.execution_order,
            control.orchestration.worker_results,
        )
    }
    artifact_references = [
        artifact
        for result in control.orchestration.worker_results
        for artifact in (
            result.data.get("artifacts")
            if isinstance(result.data.get("artifacts"), list)
            else []
        )
        if isinstance(artifact, dict)
    ]
    yield _frame(lifecycle.update_runtime_state(
        completed_steps=completed_step_ids,
        dependency_results=dependency_results,
        artifact_references=artifact_references,
    ))
    for frame in frames:
        yield frame

    question = _clarification_question(control)
    if question:
        yield _frame({
            "type": "ask_user",
            "data": {
                "question": question,
                "options": [],
                "multi": False,
            },
        })

    fallback_text = _fallback_final_text(
        query,
        control,
    )

    try:
        response = synthesizer.invoke(
            _synthesis_prompt(
                query,
                control.orchestration.state.plan,
                control,
            )
        )
        response = await _await_if_needed(response)
        final_text = str(response or "").strip()
    except Exception:
        final_text = fallback_text

    if not final_text:
        final_text = fallback_text

    yield _frame({"delta": final_text})
    yield _frame({
        "type": "metrics",
        "data": {
            "runtime": RUNTIME_VERSION,
            "model": model,
            "requested_model": model,
            "elapsed_s": round(time.time() - started_at, 3),
            "multi_graph": True,
            "plan_status": control.orchestration.status,
            "stopped_reason": control.stopped_reason,
            "revisions_applied": control.revisions_applied,
            "plan_step_count": len(
                control.orchestration.state.plan.steps
            ),
            "execution_order": list(
                control.orchestration.execution_order
            ),
            "tool_calls": len(tool_events),
            "tool_events": tool_events,
            "clarification_needed": question is not None,
            "langgraph_runtime": True,
        },
    })
    terminal_status = (
        "COMPLETED"
        if control.orchestration.status == "completed"
        else (
            "AWAITING_APPROVAL"
            if question is not None
            else "FAILED"
        )
    )
    terminal = lifecycle.finish(
        terminal_status,
        stage="coordinator",
        reason=(
            control.orchestration.message
            or control.stopped_reason
        ),
        completed_steps=completed_step_ids,
        whole_query_legacy_fallback=False,
    )
    if terminal is not None:
        yield _frame(terminal)
    yield "data: [DONE]\n\n"


__all__ = [
    "MAX_PLAN_STEPS",
    "RUNTIME_VERSION",
    "stream_multi_graph_coordinator_sse",
]
