from types import SimpleNamespace

import json

import pytest

from src.agent_runtime.code_worker import run_code_worker
from src.agent_runtime.finalizer import (
    parse_grounded_worker_finalizer_response,
)
from src.agent_runtime.worker_dispatcher import builtin_worker_runners
from src.agent_runtime.general_graph import run_code_inspection_recovery
from src.agent_runtime.supervisor_contracts import WorkerHandoff


def test_code_worker_rejects_wrong_capability():
    handoff = WorkerHandoff(
        capability="knowledge_search",
        objective="Inspect this code.",
    )

    with pytest.raises(ValueError, match="code_inspection"):
        run_code_worker(handoff, stream_factory=lambda *args, **kwargs: [])


def test_code_worker_requires_objective():
    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="   ",
    )

    with pytest.raises(ValueError, match="non-empty objective"):
        run_code_worker(handoff, stream_factory=lambda *args, **kwargs: [])


def test_code_worker_passes_workspace_and_session_to_existing_runtime():
    captured = {}

    def fake_stream(question, workspace=None, session_id=None):
        captured["question"] = question
        captured["workspace"] = workspace
        captured["session_id"] = session_id
        yield {
            "type": "evidence_hit",
            "kind": "code_location",
            "file": "src/example.py",
            "line": 1,
        }
        yield {
            "type": "stream_done",
            "final_answer": "Done.",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Find the implementation.",
        context={"workspace": "D:/Projects/example"},
    )

    result = run_code_worker(
        handoff,
        session_id="session-1",
        stream_factory=fake_stream,
    )

    assert captured == {
        "question": "Find the implementation.",
        "workspace": "D:/Projects/example",
        "session_id": "session-1",
    }
    assert result.status == "completed"


def test_code_worker_collects_specialist_evidence():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "evidence_hit",
            "kind": "code_location",
            "file": "src/example.py",
            "line": 42,
        }
        yield {
            "type": "final_answer",
            "content": "Implementation found.",
        }
        yield {
            "type": "stream_done",
            "final_answer": "Implementation found.",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Find the implementation.",
    )

    result = run_code_worker(handoff, stream_factory=fake_stream)

    assert result.status == "completed"
    assert result.summary == "Implementation found."
    assert len(result.evidence) == 1
    assert result.evidence[0]["file"] == "src/example.py"
    assert result.evidence[0]["line"] == 42


def test_code_worker_reports_stream_error_as_failed_result():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_error",
            "error": "runtime exploded",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Inspect the project.",
    )

    result = run_code_worker(handoff, stream_factory=fake_stream)

    assert result.status == "failed"
    assert result.error == "runtime exploded"


def test_code_worker_partial_when_no_final_answer_is_emitted():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_start",
            "scope": "file_inspection",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Inspect the project.",
    )

    result = run_code_worker(handoff, stream_factory=fake_stream)

    assert result.status == "partial"


def test_code_worker_declares_active_doc_support_without_invoking_it():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "evidence_hit",
            "kind": "code_location",
            "file": "src/example.py",
            "line": 2,
        }
        yield {
            "type": "stream_done",
            "final_answer": "Prepared code guidance.",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Prepare a coding fix.",
    )

    result = run_code_worker(handoff, stream_factory=fake_stream)

    assert result.data["supports_active_doc"] is True


def test_code_worker_does_not_mark_evidence_free_final_answer_completed():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_done",
            "final_answer": "Evidence is incomplete.",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Inspect the implementation.",
    )

    result = run_code_worker(
        handoff,
        stream_factory=fake_stream,
        recovery_factory=lambda *args, **kwargs: SimpleNamespace(
            task=SimpleNamespace(should_fallback=True),
            final_answer="",
            observations=[],
        ),
    )

    assert result.status == "partial"
    assert result.summary == "Evidence is incomplete."
    assert result.data["evidence_complete"] is False


def test_code_worker_reuses_capability_locked_source_inspection_for_recovery():
    captured = {}

    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_done",
            "final_answer": "Evidence is incomplete.",
        }

    def fake_recovery(query, workspace):
        captured["query"] = query
        captured["workspace"] = workspace
        return SimpleNamespace(
            task=SimpleNamespace(should_fallback=False),
            final_answer=(
                "Relevant source evidence:\n"
                "- src/agent_runtime/multi_graph_coordinator.py"
            ),
            observations=[
                SimpleNamespace(
                    tool="read_file_around",
                    args={
                        "path": (
                            "src/agent_runtime/"
                            "multi_graph_coordinator.py"
                        ),
                        "line": 531,
                        "context": 36,
                    },
                    output=(
                        "500: class MultiGraphCoordinator:\n"
                        "531: async def stream_multi_graph_coordinator_sse"
                        "(...):\n"
                        "540:     # Multi-graph implementation coordinates "
                        "planned graph execution."
                    ),
                    success=True,
                )
            ],
        )

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Inspect the multi-graph implementation.",
        context={"workspace": "D:/AI/odysseus"},
    )

    result = run_code_worker(
        handoff,
        stream_factory=fake_stream,
        recovery_factory=fake_recovery,
        finalizer_candidates=[
            ("http://primary", "model", {}),
        ],
        finalizer_caller=lambda *args, **kwargs: (
            '{"answer":"src/agent_runtime/multi_graph_coordinator.py '
            'coordinates planned multi-graph execution.",'
            '"fulfills_objective":true,'
            '"missing_requirements":[],'
            '"evidence_paths":["src/agent_runtime/'
            'multi_graph_coordinator.py"]}'
        ),
    )

    assert captured == {
        "query": "Inspect the multi-graph implementation.",
        "workspace": "D:/AI/odysseus",
    }
    assert result.status == "completed"
    assert result.data["evidence_complete"] is True
    assert result.data["task_fulfillment_complete"] is True
    assert result.data["grounded_finalizer_used"] is True
    assert (
        result.data["evidence_backend"]
        == "deterministic_code_inspection_recovery"
    )
    assert len(result.evidence) == 1
    assert result.evidence[0]["tool"] == "read_file_around"


def test_code_worker_keeps_partial_when_evidence_recovery_has_no_source_evidence():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_done",
            "final_answer": "Evidence is incomplete.",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Inspect the implementation.",
        context={"workspace": "D:/AI/odysseus"},
    )

    result = run_code_worker(
        handoff,
        stream_factory=fake_stream,
        recovery_factory=lambda *args, **kwargs: SimpleNamespace(
            task=SimpleNamespace(should_fallback=False),
            final_answer="No reliable source evidence found.",
            observations=[
                SimpleNamespace(
                    tool="get_workspace",
                    args={},
                    output="D:/AI/odysseus",
                    success=True,
                )
            ],
        ),
    )

    assert result.status == "partial"
    assert result.evidence == ()


def test_code_worker_uses_specialist_semantic_support_for_completion():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "evidence_hit",
            "kind": "code_location",
            "file": "src/example.py",
            "line": 42,
        }
        yield {
            "type": "finalizer_start",
            "claim_support": "partial",
            "strong_evidence": True,
        }
        yield {
            "type": "stream_done",
            "final_answer": "A related source location was found.",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Explain the implementation responsibility.",
    )

    result = run_code_worker(
        handoff,
        stream_factory=fake_stream,
        recovery_factory=lambda *args, **kwargs: SimpleNamespace(
            task=SimpleNamespace(should_fallback=True),
            final_answer="",
            observations=[],
        ),
    )

    assert result.status == "partial"
    assert result.data["claim_support"] == "partial"
    assert result.data["semantic_support_seen"] is True
    assert result.data["evidence_complete"] is False


def test_code_worker_accepts_strong_specialist_support_without_narrow_evidence_hit():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "finalizer_start",
            "claim_support": "strong",
            "strong_evidence": True,
        }
        yield {
            "type": "stream_done",
            "final_answer": (
                "The implementation is in "
                "src/agent_runtime/multi_graph_coordinator.py."
            ),
        }

    def forbidden_recovery(*args, **kwargs):
        raise AssertionError(
            "Strong specialist claim support must not invoke recovery."
        )

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Locate the multi-graph coordinator implementation.",
    )

    result = run_code_worker(
        handoff,
        stream_factory=fake_stream,
        recovery_factory=forbidden_recovery,
    )

    assert result.status == "completed"
    assert result.data["claim_support"] == "strong"
    assert result.data["semantic_support_seen"] is True
    assert result.data["evidence_complete"] is True


def test_code_worker_does_not_promote_glob_only_recovery_to_completed():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_done",
            "final_answer": "Evidence is incomplete.",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective=(
            "Inspect the repository and explain the files implementing "
            "multi-graph planning and coordination."
        ),
        context={"workspace": "D:/AI/odysseus"},
    )

    result = run_code_worker(
        handoff,
        stream_factory=fake_stream,
        recovery_factory=lambda *args, **kwargs: SimpleNamespace(
            task=SimpleNamespace(should_fallback=False),
            final_answer=(
                "Likely files related to the request:\\n"
                "- tests/test_multi_graph_coordinator.py"
            ),
            observations=[
                SimpleNamespace(
                    tool="glob_files",
                    args={"pattern": "**/*.py"},
                    output=(
                        "tests/test_multi_graph_coordinator.py\\n"
                        "integrations/codex/scripts/odysseus_api.py"
                    ),
                    success=True,
                )
            ],
        ),
    )

    assert result.status == "partial"
    assert result.summary == "Evidence is incomplete."
    assert result.data["evidence_complete"] is False


def test_code_worker_partial_semantic_result_attempts_bounded_recovery():
    captured = {"recovery_calls": 0}

    def fake_stream(*args, **kwargs):
        yield {
            "type": "evidence_hit",
            "kind": "code_location",
            "file": "tests/test_multi_graph_coordinator.py",
            "line": 1,
        }
        yield {
            "type": "finalizer_start",
            "claim_support": "partial",
            "strong_evidence": True,
        }
        yield {
            "type": "stream_done",
            "final_answer": "Only a related test file was found.",
        }

    def fake_recovery(*args, **kwargs):
        captured["recovery_calls"] += 1
        return SimpleNamespace(
            task=SimpleNamespace(should_fallback=True),
            final_answer="",
            observations=[],
        )

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Inspect the multi-graph implementation.",
        context={"workspace": "D:/AI/odysseus"},
    )

    result = run_code_worker(
        handoff,
        stream_factory=fake_stream,
        recovery_factory=fake_recovery,
    )

    assert captured["recovery_calls"] == 1
    assert result.status == "partial"
    assert result.data["claim_support"] == "partial"

def test_code_inspection_recovery_preserves_resolved_capability_without_reclassification(
    monkeypatch,
    tmp_path,
):
    import src.agent_runtime.general_graph as general_graph

    source_dir = tmp_path / "src" / "agent_runtime"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "multi_graph_coordinator.py"
    source_file.write_text(
        "class MultiGraphCoordinator:\n"
        "    # Multi graph planning coordination implementation.\n"
        "    def coordinate_plan(self):\n"
        "        return 'replan'\n",
        encoding="utf-8",
    )

    def forbidden_plan_task(*args, **kwargs):
        raise AssertionError(
            "Capability-locked Code Worker recovery must not reclassify intent."
        )

    monkeypatch.setattr(
        general_graph,
        "plan_task",
        forbidden_plan_task,
    )

    state = run_code_inspection_recovery(
        (
            "Identify the files implementing multi-graph planning and "
            "coordination and explain their responsibilities."
        ),
        str(tmp_path),
    )

    assert state.task is not None
    assert state.task.kind == "code_lookup"
    assert state.task.target == ""
    assert state.task.should_fallback is False
    assert state.metrics["semantic_capability"] == "code_inspection"
    assert state.metrics["intent_reclassified"] is False
    assert any(
        observation.tool == "grep"
        for observation in state.observations
    )
    assert any(
        observation.tool == "read_file_around"
        for observation in state.observations
    )
    assert "multi_graph_coordinator.py" in state.final_answer


def test_code_worker_recovery_receives_only_step_objective_and_workspace():
    captured = {}

    def fake_stream(*args, **kwargs):
        yield {
            "type": "finalizer_start",
            "claim_support": "partial",
            "strong_evidence": False,
        }
        yield {
            "type": "stream_done",
            "final_answer": "Primary evidence is incomplete.",
        }

    def fake_recovery(query, workspace):
        captured["query"] = query
        captured["workspace"] = workspace
        return SimpleNamespace(
            task=SimpleNamespace(should_fallback=True),
            final_answer="",
            observations=[],
        )

    step_objective = (
        "Inspect the repository implementation of multi-graph replanning."
    )
    handoff = WorkerHandoff(
        capability="code_inspection",
        objective=step_objective,
        context={
            "workspace": "D:/AI/odysseus",
            "unrelated_original_query": (
                "Research a report, inspect code, compare both, and email me."
            ),
        },
    )

    result = run_code_worker(
        handoff,
        stream_factory=fake_stream,
        recovery_factory=fake_recovery,
    )

    assert captured == {
        "query": step_objective,
        "workspace": "D:/AI/odysseus",
    }
    assert result.status == "partial"

def test_code_recovery_source_scope_includes_agent_runtime_without_weakening_general_scope(
    tmp_path,
):
    from src.agent_runtime import general_tools

    runtime_dir = tmp_path / "src" / "agent_runtime"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "example_runtime.py").write_text(
        "def coordinate_replan():\n"
        "    return 'multi graph planning'\n",
        encoding="utf-8",
    )

    app_dir = tmp_path / "src" / "app"
    app_dir.mkdir(parents=True)
    (app_dir / "example_app.py").write_text(
        "def application_route():\n"
        "    return 'app'\n",
        encoding="utf-8",
    )

    ordinary = general_tools.glob_files(
        str(tmp_path),
        "src/**/*.py",
    )
    code_scope = general_tools.glob_files(
        str(tmp_path),
        "src/**/*.py",
        include_agent_runtime=True,
    )

    assert "src/agent_runtime/example_runtime.py" not in ordinary
    assert "src/app/example_app.py" in ordinary
    assert "src/agent_runtime/example_runtime.py" in code_scope

def test_code_recovery_search_terms_normalize_execution_concepts():
    from src.agent_runtime.general_graph import _inspection_terms

    terms = _inspection_terms(
        (
            "Inspect the repository workspace to identify files implementing "
            "multi-graph planning, coordination, and replanning with their "
            "responsibilities."
        )
    )

    assert "multi" in terms
    assert "graph" in terms
    assert "plan" in terms
    assert "coordinat" in terms
    assert "replan" in terms
    assert "repository" not in terms
    assert "workspace" not in terms
    assert "identify" not in terms
    assert "implementing" not in terms
    assert "responsibilities" not in terms


def test_code_recovery_prioritizes_semantic_paths_before_bounded_content_scan():
    from src.agent_runtime.general_graph import _prioritize_inspection_candidates

    filler = [
        f"src/misc/file_{index:04d}.py"
        for index in range(400)
    ]
    relevant = [
        "src/agent_runtime/multi_graph_coordinator.py",
        "src/agent_runtime/task_planner.py",
        "src/agent_runtime/replan_controller.py",
        "src/agent_runtime/plan_orchestrator.py",
    ]

    selected = _prioritize_inspection_candidates(
        filler + relevant,
        ["multi", "graph", "plan", "coordinat", "replan"],
        limit=40,
    )

    assert set(relevant).issubset(set(selected))
    assert selected.index(
        "src/agent_runtime/multi_graph_coordinator.py"
    ) < 4


def test_code_worker_recovery_expands_source_context_before_semantic_completion(
    tmp_path,
):
    runtime_dir = tmp_path / "src" / "agent_runtime"
    runtime_dir.mkdir(parents=True)

    source = runtime_dir / "multi_graph_coordinator.py"
    source.write_text(
        "\n".join(
            ["# filler"] * 80
            + [
                "class MultiGraphCoordinator:",
                "    def coordinate_multi_graph_planning(self):",
                "        # Coordinates planning and replanning responsibilities.",
                "        return 'multi graph coordination replanning'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_stream(*args, **kwargs):
        yield {
            "type": "finalizer_start",
            "claim_support": "partial",
            "strong_evidence": False,
        }
        yield {
            "type": "stream_done",
            "final_answer": "Primary inspection needs stronger source context.",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective=(
            "Inspect the repository workspace to identify files implementing "
            "multi-graph planning, coordination, and replanning with their "
            "responsibilities."
        ),
        context={"workspace": str(tmp_path)},
    )

    result = run_code_worker(
        handoff,
        stream_factory=fake_stream,
        finalizer_candidates=[
            ("http://primary", "model", {}),
        ],
        finalizer_caller=lambda *args, **kwargs: (
            '{"answer":"src/agent_runtime/multi_graph_coordinator.py '
            'coordinates multi-graph planning and replanning.",'
            '"fulfills_objective":true,'
            '"missing_requirements":[],'
            '"evidence_paths":["src/agent_runtime/'
            'multi_graph_coordinator.py"]}'
        ),
    )

    assert result.status == "completed"
    assert result.data["evidence_backend"] == (
        "deterministic_code_inspection_recovery"
    )
    assert result.data["claim_support"] == "strong"
    assert result.data["task_fulfillment_complete"] is True
    assert any(
        item.get("tool") == "read_file_around"
        for item in result.evidence
    )

def test_code_recovery_strong_evidence_without_grounded_finalizer_stays_partial():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_done",
            "final_answer": "Evidence is incomplete.",
        }

    def fake_recovery(*args, **kwargs):
        return SimpleNamespace(
            task=SimpleNamespace(should_fallback=False),
            final_answer=(
                "Relevant source evidence:\n"
                "- src/agent_runtime/multi_graph_coordinator.py"
            ),
            observations=[
                SimpleNamespace(
                    tool="read_file_around",
                    args={
                        "path": (
                            "src/agent_runtime/"
                            "multi_graph_coordinator.py"
                        ),
                        "line": 500,
                        "radius": 36,
                    },
                    output=(
                        "500: async def "
                        "stream_multi_graph_coordinator_sse(...):\n"
                        "520:     # coordinates planned multi graph execution"
                    ),
                    success=True,
                )
            ],
        )

    result = run_code_worker(
        WorkerHandoff(
            capability="code_inspection",
            objective="Inspect the multi-graph implementation.",
            context={"workspace": "D:/AI/odysseus"},
        ),
        stream_factory=fake_stream,
        recovery_factory=fake_recovery,
    )

    assert result.status == "partial"
    assert result.data["evidence_complete"] is True
    assert result.data["task_fulfillment_complete"] is False
    assert result.data["grounded_finalizer_unavailable"] is True


def test_code_recovery_requires_semantic_step_fulfillment_after_strong_evidence():
    captured = {}

    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_done",
            "final_answer": "Evidence is incomplete.",
        }

    def fake_recovery(*args, **kwargs):
        return SimpleNamespace(
            task=SimpleNamespace(should_fallback=False),
            final_answer="Raw evidence summary.",
            observations=[
                SimpleNamespace(
                    tool="read_file_around",
                    args={
                        "path": (
                            "src/agent_runtime/"
                            "multi_graph_coordinator.py"
                        ),
                        "line": 500,
                        "radius": 36,
                    },
                    output=(
                        "500: async def "
                        "stream_multi_graph_coordinator_sse(...):\n"
                        "520:     # coordinates planned multi graph execution"
                    ),
                    success=True,
                )
            ],
        )

    def fake_finalizer(candidates, messages, **kwargs):
        captured["candidates"] = candidates
        captured["prompt"] = messages[0]["content"]
        captured["kwargs"] = kwargs
        response = (
            '{"answer":"The evidence locates the coordinator but does not '
            'establish every requested component responsibility.",'
            '"fulfills_objective":false,'
            '"missing_requirements":["responsibilities of all requested components"],'
            '"evidence_paths":["src/agent_runtime/multi_graph_coordinator.py"]}'
        )
        kwargs["response_validator"](response)
        return response

    objective = (
        "Identify the files implementing multi-graph planning, coordination, "
        "and replanning with their responsibilities."
    )
    result = run_code_worker(
        WorkerHandoff(
            capability="code_inspection",
            objective=objective,
            context={
                "workspace": "D:/AI/odysseus",
                "query": (
                    "Research something else, inspect code, then compare both."
                ),
            },
        ),
        stream_factory=fake_stream,
        recovery_factory=fake_recovery,
        finalizer_candidates=[
            ("http://primary", "selected-model", {}),
            ("http://fallback", "fallback-model", {}),
        ],
        finalizer_prompt_type="agent",
        finalizer_caller=fake_finalizer,
    )

    assert result.status == "partial"
    assert result.data["evidence_complete"] is True
    assert result.data["task_fulfillment_complete"] is False
    assert result.missing_information == (
        "responsibilities of all requested components",
    )
    assert objective in captured["prompt"]
    assert "Research something else" not in captured["prompt"]
    assert captured["candidates"][0][1] == "selected-model"
    assert captured["kwargs"]["prompt_type"] == "agent"


def test_code_recovery_completed_summary_is_grounded_step_answer_not_retrieval_debris():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_done",
            "final_answer": "Evidence is incomplete.",
        }

    def fake_recovery(*args, **kwargs):
        return SimpleNamespace(
            task=SimpleNamespace(should_fallback=False),
            final_answer=(
                "grep: src/agent_runtime/multi_graph_coordinator.py:500"
            ),
            observations=[
                SimpleNamespace(
                    tool="read_file_around",
                    args={
                        "path": (
                            "src/agent_runtime/"
                            "multi_graph_coordinator.py"
                        ),
                        "line": 500,
                        "radius": 36,
                    },
                    output=(
                        "500: async def "
                        "stream_multi_graph_coordinator_sse(...):\n"
                        "520:     # coordinates planned multi graph execution"
                    ),
                    success=True,
                )
            ],
        )

    answer = (
        "src/agent_runtime/multi_graph_coordinator.py — coordinates planned "
        "multi-graph execution."
    )
    response = json.dumps(
        {
            "answer": answer,
            "fulfills_objective": True,
            "missing_requirements": [],
            "evidence_paths": [
                "src/agent_runtime/multi_graph_coordinator.py",
            ],
        }
    )

    result = run_code_worker(
        WorkerHandoff(
            capability="code_inspection",
            objective="Inspect the multi-graph implementation.",
            context={"workspace": "D:/AI/odysseus"},
        ),
        stream_factory=fake_stream,
        recovery_factory=fake_recovery,
        finalizer_candidates=[
            ("http://primary", "model", {}),
        ],
        finalizer_caller=lambda *args, **kwargs: response,
    )

    assert result.status == "completed"
    assert result.summary == answer
    assert result.data["task_fulfillment_complete"] is True
    assert result.data["grounded_finalizer_used"] is True
    assert "grep:" not in result.summary


def test_grounded_worker_finalizer_parser_rejects_false_completion_contract():
    with pytest.raises(ValueError, match="cannot also report missing"):
        parse_grounded_worker_finalizer_response(
            '{"answer":"partial",'
            '"fulfills_objective":true,'
            '"missing_requirements":["responsibility mapping"],'
            '"evidence_paths":["src/example.py"]}'
        )


def test_builtin_worker_runners_can_configure_code_finalizer_without_affecting_others():
    candidates = [
        ("http://primary", "selected", {}),
        ("http://fallback", "fallback", {}),
    ]

    runners = builtin_worker_runners(
        code_finalizer_candidates=candidates,
        code_finalizer_prompt_type="agent",
    )

    code_runner = runners["code_inspection"]
    assert code_runner.keywords["finalizer_candidates"] == tuple(candidates)
    assert code_runner.keywords["finalizer_prompt_type"] == "agent"
    assert runners["knowledge_search"].__name__ == "run_knowledge_worker"

def test_code_recovery_blocks_finalizer_paths_not_present_in_runtime_evidence():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_done",
            "final_answer": "Evidence is incomplete.",
        }

    def fake_recovery(*args, **kwargs):
        return SimpleNamespace(
            task=SimpleNamespace(should_fallback=False),
            final_answer="Raw evidence summary.",
            observations=[
                SimpleNamespace(
                    tool="read_file_around",
                    args={
                        "path": (
                            "src/agent_runtime/"
                            "multi_graph_coordinator.py"
                        ),
                        "line": 500,
                        "radius": 36,
                    },
                    output=(
                        "500: async def "
                        "stream_multi_graph_coordinator_sse(...):\n"
                        "520:     # coordinates planned multi graph execution"
                    ),
                    success=True,
                )
            ],
        )

    response = json.dumps(
        {
            "answer": "A different file owns this behavior.",
            "fulfills_objective": True,
            "missing_requirements": [],
            "evidence_paths": [
                "src/agent_runtime/not_observed.py",
            ],
        }
    )

    result = run_code_worker(
        WorkerHandoff(
            capability="code_inspection",
            objective="Inspect the multi-graph implementation.",
            context={"workspace": "D:/AI/odysseus"},
        ),
        stream_factory=fake_stream,
        recovery_factory=fake_recovery,
        finalizer_candidates=[
            ("http://primary", "model", {}),
        ],
        finalizer_caller=lambda *args, **kwargs: response,
    )

    assert result.status == "partial"
    assert result.data["evidence_complete"] is True
    assert result.data["grounded_finalizer_used"] is True
    assert result.data["task_fulfillment_complete"] is False
    assert "not present in the runtime evidence" in (
        result.missing_information[0]
    )

def test_code_worker_reverifies_strong_primary_completion_against_planstep_objective():
    captured = {
        "recovery_calls": 0,
        "finalizer_calls": 0,
    }

    def fake_stream(*args, **kwargs):
        yield {
            "type": "finalizer_start",
            "claim_support": "strong",
            "strong_evidence": True,
        }
        yield {
            "type": "stream_done",
            "final_answer": (
                "The code includes general_graph.py, task_planner.py, "
                "and multi_graph_coordinator.py."
            ),
        }

    def fake_recovery(query, workspace):
        captured["recovery_calls"] += 1
        return SimpleNamespace(
            task=SimpleNamespace(should_fallback=False),
            final_answer="Recovered source evidence.",
            observations=[
                SimpleNamespace(
                    tool="read_file_around",
                    args={
                        "path": (
                            "src/agent_runtime/"
                            "multi_graph_coordinator.py"
                        ),
                        "line": 500,
                        "radius": 36,
                    },
                    output=(
                        "500: async def "
                        "stream_multi_graph_coordinator_sse(...):\n"
                        "520:     # coordinates planned multi graph execution"
                    ),
                    success=True,
                )
            ],
        )

    def fake_finalizer(candidates, messages, **kwargs):
        captured["finalizer_calls"] += 1
        response = json.dumps(
            {
                "answer": (
                    "The evidence supports coordination, but does not establish "
                    "the requested replanning ownership and responsibilities."
                ),
                "fulfills_objective": False,
                "missing_requirements": [
                    "replanning implementation ownership and responsibilities",
                ],
                "evidence_paths": [
                    "src/agent_runtime/multi_graph_coordinator.py",
                ],
            }
        )
        kwargs["response_validator"](response)
        return response

    result = run_code_worker(
        WorkerHandoff(
            capability="code_inspection",
            objective=(
                "Identify files implementing multi-graph planning, coordination, "
                "and replanning with their responsibilities."
            ),
            context={"workspace": "D:/AI/odysseus"},
        ),
        stream_factory=fake_stream,
        recovery_factory=fake_recovery,
        finalizer_candidates=[
            ("http://primary", "selected-model", {}),
        ],
        finalizer_prompt_type="agent",
        finalizer_caller=fake_finalizer,
    )

    assert captured == {
        "recovery_calls": 1,
        "finalizer_calls": 1,
    }
    assert result.status == "partial"
    assert result.data["grounded_finalizer_used"] is True
    assert result.data["task_fulfillment_complete"] is False
    assert result.data["primary_specialist_completion_reverified"] is True
    assert result.missing_information == (
        "replanning implementation ownership and responsibilities",
    )
