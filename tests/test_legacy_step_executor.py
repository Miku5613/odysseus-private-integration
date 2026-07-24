from __future__ import annotations

import asyncio
import json

from src.agent_runtime.execution_resolver import (
    ExecutionResolution,
    ExecutionSurfaceDescriptor,
)
from src.agent_runtime.legacy_step_executor import LegacyStepExecutor
from src.agent_runtime.step_execution import StepExecutionContext
from src.agent_runtime.task_plan import PlanStep


def _resolution():
    return ExecutionResolution(
        step_id="legacy_step",
        status="resolved",
        selected_surface=ExecutionSurfaceDescriptor(
            id="legacy:agent_loop",
            kind="legacy",
            description="Legacy agent.",
            supports_external_state=True,
            supports_mutation=True,
        ),
    )


def test_legacy_step_executor_normalizes_sse_and_dependency_context():
    captured = {}

    async def fake_stream(*args, **kwargs):
        captured["messages"] = args[2]
        captured["max_tool_calls"] = kwargs["max_tool_calls"]
        yield 'data: {"type":"tool_start","tool":"read_file","command":"a"}\n\n'
        yield 'data: {"type":"tool_output","tool":"read_file","command":"a","output":"A","exit_code":0}\n\n'
        yield 'data: {"delta":"Step result"}\n\n'
        yield 'data: {"type":"metrics","data":{"model":"test"}}\n\n'
        yield "data: [DONE]\n\n"

    executor = LegacyStepExecutor(
        endpoint_url="http://example",
        model="model",
        original_query="Complete both phases.",
        max_tool_calls=3,
        stream_factory=fake_stream,
    )

    result = asyncio.run(
        executor(
            PlanStep(
                id="legacy_step",
                objective="Read the required source.",
                depends_on=("prior",),
            ),
            _resolution(),
            StepExecutionContext(
                supervisor_state={
                    "worker_context": {
                        "dependency_outputs": {
                            "prior": {
                                "summary": "Prior result",
                            }
                        }
                    }
                }
            ),
        )
    )

    assert result.status == "completed"
    assert result.capability == "legacy_agent"
    assert result.summary == "Step result"
    assert result.data["tool_calls"] == 1
    assert result.data["execution_surface"] == "legacy:agent_loop"
    assert captured["max_tool_calls"] == 3
    payload = json.loads(captured["messages"][-1]["content"])
    assert payload["plan_step"]["id"] == "legacy_step"
    assert payload["completed_dependency_outputs"]["prior"]["summary"] == "Prior result"


def test_legacy_step_executor_converts_ask_user_to_clarification():
    async def fake_stream(*args, **kwargs):
        yield 'data: {"type":"ask_user","data":{"question":"Which file?"}}\n\n'
        yield "data: [DONE]\n\n"

    result = asyncio.run(
        LegacyStepExecutor(
            endpoint_url="http://example",
            model="model",
            original_query="Inspect a file.",
            stream_factory=fake_stream,
        )(
            PlanStep(
                id="legacy_step",
                objective="Inspect the selected file.",
            ),
            _resolution(),
            StepExecutionContext(),
        )
    )

    assert result.status == "needs_clarification"
    assert result.summary == "Which file?"
    assert result.missing_information == ("user response",)

def test_legacy_step_executor_shares_tool_budget_across_plan_steps():
    calls = []

    async def fake_stream(*args, **kwargs):
        calls.append(args[2])
        yield 'data: {"type":"tool_start","tool":"read_file","command":"a"}\n\n'
        yield 'data: {"type":"tool_output","tool":"read_file","command":"a","output":"A","exit_code":0}\n\n'
        yield 'data: {"delta":"First step"}\n\n'
        yield "data: [DONE]\n\n"

    executor = LegacyStepExecutor(
        endpoint_url="http://example",
        model="model",
        original_query="Complete two legacy phases.",
        max_tool_calls=1,
        stream_factory=fake_stream,
    )

    first = asyncio.run(
        executor(
            PlanStep(
                id="legacy_step",
                objective="Read the first source.",
            ),
            _resolution(),
            StepExecutionContext(),
        )
    )
    second = asyncio.run(
        executor(
            PlanStep(
                id="legacy_step",
                objective="Read another source.",
            ),
            _resolution(),
            StepExecutionContext(),
        )
    )

    assert first.status == "completed"
    assert second.status == "needs_clarification"
    assert second.missing_information == ("additional tool budget",)
    assert len(calls) == 1

