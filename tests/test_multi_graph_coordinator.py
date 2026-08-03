from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.agent_runtime.execution_resolver import (
    ExecutionSurfaceDescriptor,
)
from src.agent_runtime.intent_complexity import (
    IntentComplexityAssessment,
    IntentGoal,
)
from src.agent_runtime.multi_graph_coordinator import (
    _RuntimeFallbackModel,
    stream_multi_graph_coordinator_sse,
)
from src.agent_runtime.run_lifecycle import (
    LangGraphCancellationScope,
)
from src.agent_runtime.research_worker import run_research_worker
from src.agent_runtime.supervisor_contracts import WorkerResult


class Model:
    def __init__(self, responder):
        self.responder = responder
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        value = self.responder(prompt)

        async def response():
            return value

        return response()


def _assessment():
    return IntentComplexityAssessment(
        execution_class="multi_graph",
        goals=(
            IntentGoal(
                id="gather",
                objective="Gather evidence.",
                work_domains=("knowledge",),
                requires_external_state=True,
                retrieval_concepts=("runtime admission", "workspace source"),
            ),
            IntentGoal(
                id="combine",
                objective="Combine evidence.",
                work_domains=("synthesis",),
            ),
        ),
        needs_planning=True,
        missing_information=(),
        clarification_needed=False,
        reasoning_summary="Two dependent goals.",
        requested_sources=("attached_document",),
        excluded_sources=("public_web",),
    )


def _plan():
    return {
        "objective": "Gather and combine.",
        "steps": [
            {
                "id": "gather",
                "objective": "Gather evidence.",
                "depends_on": [],
                "work_domains": ["knowledge"],
                "uses_dependency_state": False,
                "requires_external_state": True,
                "requires_mutation": False,
            },
            {
                "id": "combine",
                "objective": "Combine evidence.",
                "depends_on": ["gather"],
                "work_domains": ["synthesis"],
                "uses_dependency_state": True,
                "requires_external_state": False,
                "requires_mutation": False,
            },
        ],
        "reasoning_summary": "Gather before combining.",
    }


async def _collect(generator):
    return [frame async for frame in generator]


def test_multi_graph_coordinator_executes_dependencies_and_emits_one_done():
    surfaces = (
        ExecutionSurfaceDescriptor(
            id="langgraph:gather_cap",
            kind="langgraph",
            description="Gather evidence.",
            capability_name="gather_cap",
        ),
        ExecutionSurfaceDescriptor(
            id="langgraph:combine_cap",
            kind="langgraph",
            description="Combine evidence.",
            capability_name="combine_cap",
        ),
    )

    def resolver(prompt):
        selected = (
            "langgraph:gather_cap"
            if '"id": "gather"' in prompt
            else "langgraph:combine_cap"
        )
        return {
            "compatible_surface_ids": [selected],
            "clarification_needed": False,
            "missing_information": [],
            "reasoning_summary": "Compatible.",
        }

    captured = {}

    def gather(handoff, *, session_id=None):
        captured["gather_context"] = dict(handoff.context)
        return WorkerResult(
            capability="gather_cap",
            status="completed",
            summary="Evidence A",
            data={"items": ["A"]},
        )

    def combine(handoff, *, session_id=None):
        captured.update(handoff.context)
        return WorkerResult(
            capability="combine_cap",
            status="completed",
            summary="Combined A",
        )

    frames = asyncio.run(
        _collect(
            stream_multi_graph_coordinator_sse(
                "http://example",
                "model",
                [{"role": "user", "content": "Gather then combine."}],
                admission_assessment=_assessment(),
                planner_model=Model(lambda prompt: _plan()),
                resolver_model=Model(resolver),
                replanner_model=Model(lambda prompt: _plan()),
                action_model=Model(lambda prompt: {}),
                final_model=Model(lambda prompt: "Final answer"),
                worker_runners={
                    "gather_cap": gather,
                    "combine_cap": combine,
                },
                surfaces=surfaces,
                session_id="session-1",
                run_id="run-1",
                thread_id="thread-1",
            )
        )
    )

    assert frames.count("data: [DONE]\n\n") == 1
    payloads = [
        json.loads(frame[6:])
        for frame in frames
        if frame.startswith("data: ")
        and frame != "data: [DONE]\n\n"
    ]
    assert any(item.get("type") == "plan_update" for item in payloads)
    assert any(item.get("delta") == "Final answer" for item in payloads)
    runtime_run = next(
        item["data"]
        for item in payloads
        if item.get("type") == "runtime_run"
    )
    assert runtime_run["run_id"] == "run-1"
    assert runtime_run["session_id"] == "session-1"
    assert runtime_run["thread_id"] == "thread-1"
    metrics = next(
        item["data"]
        for item in payloads
        if item.get("type") == "metrics"
    )
    assert metrics["plan_status"] == "completed"
    assert metrics["execution_order"] == ["gather", "combine"]
    assert metrics["multi_graph"] is True
    assert captured["dependency_outputs"]["gather"]["summary"] == "Evidence A"
    assert captured["gather_context"]["source_contract"] == {
        "requested_sources": ["attached_document"],
        "excluded_sources": ["public_web"],
        "retrieval_concepts": [
            "runtime admission",
            "workspace source",
        ],
    }
    assert captured["gather_context"]["retrieval_concepts"] == [
        "runtime admission",
        "workspace source",
    ]


def test_planner_exhaustion_emits_one_controlled_terminal_result():
    invalid = _plan()
    invalid["steps"][1]["depends_on"] = ["missing"]
    planner = Model(lambda prompt: invalid)

    frames = asyncio.run(
        _collect(
            stream_multi_graph_coordinator_sse(
                "http://example",
                "model",
                [{"role": "user", "content": "Gather then combine."}],
                admission_assessment=_assessment(),
                planner_model=planner,
            )
        )
    )
    payloads = [
        json.loads(frame[6:])
        for frame in frames
        if frame.startswith("data: ")
        and frame != "data: [DONE]\n\n"
    ]
    terminals = [
        item
        for item in payloads
        if item.get("type") == "runtime_terminal"
    ]

    assert len(planner.prompts) == 2
    assert frames.count("data: [DONE]\n\n") == 1
    assert len(terminals) == 1
    assert terminals[0]["data"]["status"] == "FAILED"
    assert terminals[0]["data"]["stage"] == "planner"
    assert terminals[0]["data"]["completed_steps"] == []
    assert terminals[0]["data"]["whole_query_legacy_fallback"] is False


def test_coordinator_cancellation_cancels_only_its_run_scope():
    async def scenario():
        blocker = asyncio.Event()

        class BlockingPlanner:
            def invoke(self, prompt):
                async def wait_forever():
                    await blocker.wait()
                    return _plan()

                return wait_forever()

        owned_calls = []
        scope = LangGraphCancellationScope("run-cancel")
        scope.own_cancellation(
            "owned-research-job",
            lambda: owned_calls.append("owned"),
        )
        generator = stream_multi_graph_coordinator_sse(
            "http://example",
            "model",
            [{"role": "user", "content": "Gather then combine."}],
            admission_assessment=_assessment(),
            planner_model=BlockingPlanner(),
            run_id="run-cancel",
            cancellation_scope=scope,
        )

        first = await generator.__anext__()
        assert '"type": "runtime_run"' in first

        pending = asyncio.create_task(generator.__anext__())
        await asyncio.sleep(0)
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("planner cancellation did not propagate")

        assert scope.cancelled is True
        assert owned_calls == ["owned"]

    asyncio.run(scenario())


def test_completed_research_report_enters_dependency_and_final_synthesis():
    report = (
        "LangGraph durable research coordination uses typed job states, "
        "stable ownership, final-report retrieval, and dependency handoff. "
        "This complete report explains how isolated cancellation and evidence "
        "references support reliable coordinator synthesis."
    )
    plan = {
        "objective": "Research and synthesize.",
        "steps": [
            {
                "id": "research",
                "objective": "Research LangGraph coordination.",
                "depends_on": [],
                "work_domains": ["research"],
                "uses_dependency_state": False,
                "requires_external_state": True,
                "requires_mutation": False,
            },
            {
                "id": "combine",
                "objective": "Synthesize the research report.",
                "depends_on": ["research"],
                "work_domains": ["synthesis"],
                "uses_dependency_state": True,
                "requires_external_state": False,
                "requires_mutation": False,
            },
        ],
        "reasoning_summary": "Research before synthesis.",
    }
    surfaces = (
        ExecutionSurfaceDescriptor(
            id="langgraph:deep_research",
            kind="langgraph",
            description="Run deep research.",
            capability_name="deep_research",
        ),
        ExecutionSurfaceDescriptor(
            id="langgraph:combine_cap",
            kind="langgraph",
            description="Combine dependency evidence.",
            capability_name="combine_cap",
        ),
    )
    captured = {}

    def resolver(prompt):
        selected = (
            "langgraph:deep_research"
            if '"id": "research"' in prompt
            else "langgraph:combine_cap"
        )
        return {
            "compatible_surface_ids": [selected],
            "clarification_needed": False,
            "missing_information": [],
            "reasoning_summary": "Compatible.",
        }

    async def research_runner(handoff, *, session_id=None):
        async def trigger(content, owner=None):
            return {
                "research_session_id": "research-final",
                "exit_code": 0,
            }

        async def status(job_id, owner=None):
            return {"status": "completed", "exit_code": 0}

        async def result(job_id, owner=None):
            return {
                "result": report,
                "sources": [{"url": "https://example.test/source"}],
                "raw_findings": [],
                "exit_code": 0,
            }

        return await run_research_worker(
            handoff,
            session_id=session_id,
            trigger_backend=trigger,
            status_backend=status,
            result_backend=result,
        )

    def combine_runner(handoff, *, session_id=None):
        captured["dependency"] = handoff.context[
            "dependency_outputs"
        ]["research"]
        return WorkerResult(
            capability="combine_cap",
            status="completed",
            summary="Research synthesized.",
        )

    def final_response(prompt):
        captured["final_prompt"] = prompt
        return "Final research synthesis."

    frames = asyncio.run(
        _collect(
            stream_multi_graph_coordinator_sse(
                "http://example",
                "model",
                [{"role": "user", "content": "Research then synthesize."}],
                admission_assessment=_assessment(),
                planner_model=Model(lambda prompt: plan),
                resolver_model=Model(resolver),
                replanner_model=Model(lambda prompt: plan),
                    action_model=Model(
                        lambda prompt: {
                            "action": "trigger_research",
                        "clarification_needed": False,
                        "missing_information": [],
                        "reasoning_summary": "Start new requested research.",
                    }
                ),
                final_model=Model(final_response),
                worker_runners={
                    "deep_research": research_runner,
                    "combine_cap": combine_runner,
                },
                surfaces=surfaces,
            )
        )
    )

    assert frames.count("data: [DONE]\n\n") == 1
    assert "dependency" in captured, frames
    research_data = captured["dependency"]["data"]
    assert research_data["dependency_state"]["report"] == report
    assert research_data["artifacts"][0]["research_id"] == "research-final"
    assert report in captured["final_prompt"]


def test_partial_worker_result_stops_at_replan_boundary_instead_of_crashing():
    surface = (
        ExecutionSurfaceDescriptor(
            id="langgraph:partial_cap",
            kind="langgraph",
            description="Return partial evidence.",
            capability_name="partial_cap",
        ),
    )

    result_frames = asyncio.run(
        _collect(
            stream_multi_graph_coordinator_sse(
                "http://example",
                "model",
                [{"role": "user", "content": "Do two phases."}],
                admission_assessment=_assessment(),
                planner_model=Model(lambda prompt: _plan()),
                resolver_model=Model(
                    lambda prompt: {
                        "compatible_surface_ids": [
                            "langgraph:partial_cap"
                        ],
                        "clarification_needed": False,
                        "missing_information": [],
                        "reasoning_summary": "Compatible.",
                    }
                ),
                replanner_model=Model(lambda prompt: _plan()),
                action_model=Model(lambda prompt: {}),
                final_model=Model(lambda prompt: "Need more information."),
                worker_runners={
                    "partial_cap": lambda handoff, session_id=None: WorkerResult(
                        capability="partial_cap",
                        status="partial",
                        summary="Only partial evidence was available.",
                    )
                },
                surfaces=surface,
            )
        )
    )

    assert result_frames[-1] == "data: [DONE]\n\n"
    payloads = [
        json.loads(frame[6:])
        for frame in result_frames
        if frame.startswith("data: ")
        and frame != "data: [DONE]\n\n"
    ]
    metrics = next(
        item["data"]
        for item in payloads
        if item.get("type") == "metrics"
    )
    assert metrics["plan_status"] == "needs_replan"
    assert metrics["clarification_needed"] is True
    assert any(item.get("type") == "ask_user" for item in payloads)

def test_multi_graph_coordinator_leaves_persistence_to_chat_route():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "agent_runtime"
        / "multi_graph_coordinator.py"
    ).read_text(encoding="utf-8")

    assert "save_assistant_response" not in source
    assert '"type": "message_saved"' not in source
    assert 'yield "data: [DONE]\\n\\n"' in source



def test_default_multi_graph_catalog_executes_dependency_reasoning_without_action_selection():
    action_model = Model(
        lambda prompt: {
            "action": "inspect",
            "clarification_needed": False,
            "missing_information": [],
            "reasoning_summary": "Inspect the requested source evidence.",
        }
    )

    def resolver(prompt):
        if '"id": "gather"' in prompt:
            return {
                "compatible_surface_ids": [
                    "langgraph:code_inspection"
                ],
                "clarification_needed": False,
                "missing_information": [],
                "reasoning_summary": "Code inspection owns the acquisition.",
            }

        assert "langgraph:dependency_reasoning" in prompt
        return {
            "compatible_surface_ids": [
                "langgraph:dependency_reasoning"
            ],
            "clarification_needed": False,
            "missing_information": [],
            "reasoning_summary": (
                "The dependent synthesis uses completed outputs only."
            ),
        }

    frames = asyncio.run(
        _collect(
            stream_multi_graph_coordinator_sse(
                "http://example",
                "model",
                [
                    {
                        "role": "user",
                        "content": "Gather then compare the completed result.",
                    }
                ],
                admission_assessment=_assessment(),
                planner_model=Model(lambda prompt: _plan()),
                resolver_model=Model(resolver),
                replanner_model=Model(lambda prompt: _plan()),
                action_model=action_model,
                final_model=Model(
                    lambda prompt: (
                        "Dependency comparison completed."
                        if "COMPLETED DEPENDENCY OUTPUTS:" in prompt
                        else "Final answer"
                    )
                ),
                worker_runners={
                    "code_inspection": (
                        lambda handoff, session_id=None: WorkerResult(
                            capability="code_inspection",
                            status="completed",
                            summary="Evidence A",
                            data={"items": ["A"]},
                        )
                    )
                },
            )
        )
    )

    payloads = [
        json.loads(frame[6:])
        for frame in frames
        if frame.startswith("data: ")
        and frame != "data: [DONE]\n\n"
    ]
    metrics = next(
        item["data"]
        for item in payloads
        if item.get("type") == "metrics"
    )

    assert metrics["plan_status"] == "completed"
    assert metrics["execution_order"] == [
        "gather",
        "combine",
    ]
    assert frames.count("data: [DONE]\n\n") == 1
    assert len(action_model.prompts) == 1
    assert "SELECTED CAPABILITY:" in action_model.prompts[0]
    assert '"name": "code_inspection"' in action_model.prompts[0]
    assert "dependency_reasoning" not in action_model.prompts[0]

def test_runtime_fallback_model_forwards_structured_validator():
    captured = {}

    async def fake_llm_caller(candidates, messages, **kwargs):
        captured["candidates"] = candidates
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        response = '{"ok": true}'
        kwargs["response_validator"](response)
        return response

    def validator(response):
        payload = json.loads(response)
        if payload != {"ok": True}:
            raise ValueError("invalid structured response")
        return payload

    model = _RuntimeFallbackModel(
        [
            (
                "http://primary.example/v1/chat/completions",
                "primary-model",
                {},
            )
        ],
        prompt_type="test",
        max_tokens=900,
        llm_caller=fake_llm_caller,
    )

    response = asyncio.run(
        model.invoke_validated(
            "Return JSON.",
            validator,
        )
    )

    assert response == '{"ok": true}'
    forwarded_validator = captured["kwargs"]["response_validator"]
    assert callable(forwarded_validator)
    assert forwarded_validator('{"ok": true}') == {"ok": True}
    assert captured["kwargs"]["prompt_type"] == "test"



def test_runtime_fallback_model_retries_rejected_structured_chain_once():
    prompts = []

    async def fake_llm_caller(
        candidates,
        *,
        messages,
        response_validator=None,
        **kwargs,
    ):
        prompt = messages[0]["content"]
        prompts.append(prompt)
        response = "invalid" if len(prompts) == 1 else "valid"
        if response_validator is not None:
            response_validator(response)
        return response

    def validator(value):
        if value != "valid":
            raise ValueError("invalid structured response")
        return value

    model = _RuntimeFallbackModel(
        [("url", "model", {})],
        prompt_type=None,
        max_tokens=200,
        llm_caller=fake_llm_caller,
    )

    result = asyncio.run(
        model.invoke_validated(
            "Return structured output.",
            validator,
        )
    )

    assert result == "valid"
    assert len(prompts) == 2
    assert prompts[0] == "Return structured output."
    assert "STRUCTURED OUTPUT RETRY:" in prompts[1]

def test_multi_graph_configures_code_worker_grounded_finalizer_from_chat_candidate_chain():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "agent_runtime"
        / "multi_graph_coordinator.py"
    ).read_text(encoding="utf-8")

    assert (
        'code_finalizer_candidates=role_candidates("code_worker")'
        in source
    )
    assert "code_finalizer_prompt_type=prompt_type" in source
