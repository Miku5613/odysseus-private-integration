from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.agent_loop import stream_agent_loop
from src.agent_runtime.execution_resolver import ExecutionResolution
from src.agent_runtime.step_execution import StepExecutionContext
from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.task_plan import PlanStep


LEGACY_SURFACE_ID = "legacy:agent_loop"
LegacyStreamFactory = Callable[..., AsyncIterator[str]]


def _frame_payload(frame: str) -> tuple[str, Any]:
    text = str(frame or "")

    if text == "data: [DONE]\n\n":
        return "done", None

    if text.startswith("event: error"):
        marker = "\ndata: "
        payload_text = text.split(marker, 1)[1].strip() if marker in text else ""
        try:
            return "error", json.loads(payload_text)
        except Exception:
            return "error", {"error": payload_text or text}

    if not text.startswith("data: "):
        return "other", text

    payload_text = text[6:].strip()
    if payload_text == "[DONE]":
        return "done", None

    try:
        return "data", json.loads(payload_text)
    except json.JSONDecodeError:
        return "other", payload_text


def _step_messages(
    step: PlanStep,
    context: StepExecutionContext,
    *,
    original_query: str,
) -> list[dict[str, Any]]:
    supervisor_state = dict(
        context.supervisor_state
        or {}
    )
    worker_context = dict(
        supervisor_state.get("worker_context", {})
    )

    payload = {
        "original_user_request": original_query,
        "plan_step": {
            "id": step.id,
            "objective": step.objective,
            "depends_on": list(step.depends_on),
            "work_domains": list(step.work_domains),
            "requires_external_state": step.requires_external_state,
            "requires_mutation": step.requires_mutation,
        },
        "completed_dependency_outputs": worker_context.get(
            "dependency_outputs",
            {},
        ),
    }

    return [
        {
            "role": "system",
            "content": (
                "You are executing exactly one bounded PlanStep from a "
                "validated dependency-aware plan. Complete only this step. "
                "Use completed dependency outputs as input, do not redo prior "
                "steps, and do not produce the overall final user response. "
                "All existing Odysseus tool, confirmation, mutation, and "
                "workspace policies remain authoritative."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        },
    ]


@dataclass
class LegacyStepExecutor:
    """Normalize one bounded Legacy agent-loop execution into WorkerResult."""

    endpoint_url: str
    model: str
    original_query: str
    headers: dict[str, Any] | None = None
    temperature: float = 0.3
    max_tokens: int = 4096
    prompt_type: str | None = None
    max_rounds: int = 8
    max_tool_calls: int = 0
    context_length: int = 0
    active_document: Any = None
    active_email: dict[str, str] | None = None
    session_id: str | None = None
    disabled_tools: set[str] | None = None
    owner: str | None = None
    fallbacks: Sequence[tuple] = ()
    tool_policy: Any = None
    workspace: str | None = None
    forced_tools: set[str] | None = None
    uploaded_files: list[dict[str, Any]] = field(default_factory=list)
    stream_factory: LegacyStreamFactory = stream_agent_loop
    _remaining_tool_calls: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.max_tool_calls > 0:
            self._remaining_tool_calls = self.max_tool_calls

    async def __call__(
        self,
        step: PlanStep,
        resolution: ExecutionResolution,
        context: StepExecutionContext,
    ) -> WorkerResult:
        surface = resolution.selected_surface
        if (
            surface is None
            or surface.kind != "legacy"
            or surface.id != LEGACY_SURFACE_ID
        ):
            raise ValueError(
                "LegacyStepExecutor requires legacy:agent_loop resolution"
            )

        if (
            self._remaining_tool_calls is not None
            and self._remaining_tool_calls <= 0
        ):
            return WorkerResult(
                capability="legacy_agent",
                status="needs_clarification",
                summary=(
                    "The configured Agent tool budget was exhausted before "
                    f"PlanStep {step.id} could run."
                ),
                data={
                    "plan_step_id": step.id,
                    "execution_surface": LEGACY_SURFACE_ID,
                },
                missing_information=("additional tool budget",),
            )

        effective_tool_budget = (
            self._remaining_tool_calls
            if self._remaining_tool_calls is not None
            else 0
        )

        full_response = ""
        metrics: dict[str, Any] = {}
        tool_events: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        ask_user: dict[str, Any] | None = None
        error: str | None = None
        rounds_exhausted = False
        completed_stream = False
        tool_calls = 0

        try:
            async for frame in self.stream_factory(
                self.endpoint_url,
                self.model,
                _step_messages(
                    step,
                    context,
                    original_query=self.original_query,
                ),
                headers=self.headers,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                prompt_type=self.prompt_type,
                max_rounds=self.max_rounds,
                max_tool_calls=effective_tool_budget,
                context_length=self.context_length,
                active_document=self.active_document,
                active_email=self.active_email,
                session_id=self.session_id or context.session_id,
                disabled_tools=self.disabled_tools,
                owner=self.owner,
                fallbacks=list(self.fallbacks),
                plan_mode=False,
                approved_plan=None,
                tool_policy=self.tool_policy,
                workspace=self.workspace,
                forced_tools=self.forced_tools,
                uploaded_files=list(self.uploaded_files),
            ):
                kind, payload = _frame_payload(frame)

                if kind == "done":
                    completed_stream = True
                    continue

                if kind == "error":
                    if isinstance(payload, Mapping):
                        error = str(
                            payload.get("error")
                            or payload.get("message")
                            or "Legacy agent step failed."
                        )
                    else:
                        error = str(payload or "Legacy agent step failed.")
                    continue

                if kind != "data" or not isinstance(payload, dict):
                    continue

                if "delta" in payload and not payload.get("thinking"):
                    full_response += str(payload.get("delta") or "")

                event_type = str(payload.get("type") or "")
                if event_type == "tool_start":
                    tool_calls += 1
                elif event_type == "tool_output":
                    event = {
                        "round": payload.get("round") or 1,
                        "tool": payload.get("tool") or "legacy_tool",
                        "command": payload.get("command") or "",
                        "output": payload.get("output") or "",
                        "exit_code": payload.get("exit_code", 0),
                    }
                    tool_events.append(event)
                    evidence.append(
                        {
                            "kind": "legacy_tool_output",
                            **event,
                        }
                    )
                elif event_type == "ask_user":
                    value = payload.get("data")
                    ask_user = dict(value) if isinstance(value, dict) else {}
                elif event_type == "rounds_exhausted":
                    rounds_exhausted = True
                elif event_type == "metrics":
                    value = payload.get("data")
                    if isinstance(value, dict):
                        metrics = dict(value)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        if self._remaining_tool_calls is not None:
            self._remaining_tool_calls = max(
                0,
                self._remaining_tool_calls - tool_calls,
            )

        data = {
            "plan_step_id": step.id,
            "execution_surface": LEGACY_SURFACE_ID,
            "tool_events": tool_events,
            "metrics": metrics,
            "stream_completed": completed_stream,
            "tool_calls": tool_calls,
        }

        if ask_user is not None:
            question = str(
                ask_user.get("question")
                or full_response
                or "The Legacy step requires a user response."
            ).strip()
            return WorkerResult(
                capability="legacy_agent",
                status="needs_clarification",
                summary=question,
                data={
                    **data,
                    "ask_user": ask_user,
                },
                evidence=tuple(evidence),
                missing_information=("user response",),
            )

        if error is not None:
            return WorkerResult(
                capability="legacy_agent",
                status="failed",
                summary=(
                    full_response.strip()
                    or "Legacy agent step failed."
                ),
                data=data,
                evidence=tuple(evidence),
                error=error,
            )

        if rounds_exhausted:
            return WorkerResult(
                capability="legacy_agent",
                status="partial",
                summary=(
                    full_response.strip()
                    or "Legacy agent step reached its round limit."
                ),
                data=data,
                evidence=tuple(evidence),
            )

        final_text = full_response.strip()
        if final_text or tool_events:
            if not final_text:
                final_text = "Legacy agent step completed with tool output."
            return WorkerResult(
                capability="legacy_agent",
                status="completed",
                summary=final_text,
                data=data,
                evidence=tuple(evidence),
            )

        return WorkerResult(
            capability="legacy_agent",
            status="failed",
            summary="Legacy agent step produced no usable result.",
            data=data,
            error="No response text or tool output was produced.",
        )


__all__ = [
    "LEGACY_SURFACE_ID",
    "LegacyStepExecutor",
    "LegacyStreamFactory",
]
