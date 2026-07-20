from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

graph = (ROOT / "src/agent_runtime/graph.py").read_text(
    encoding="utf-8", errors="replace"
)
streaming = (ROOT / "src/agent_runtime/streaming.py").read_text(
    encoding="utf-8", errors="replace"
)
runner = (ROOT / "src/agent_runtime/runner.py").read_text(
    encoding="utf-8", errors="replace"
)
legacy = (ROOT / "src/agent_loop.py").read_text(
    encoding="utf-8", errors="replace"
)


def test_graph_accepts_resolved_runtime_scope():
    assert 'def build_graph(scope: str = "file_inspection"):' in graph
    assert 'resolved_scope = str(scope or "file_inspection")' in graph
    assert "tools = get_tools_for_scope(resolved_scope)" in graph


def test_streaming_passes_inferred_scope_to_graph():
    infer_pos = streaming.index(
        "scope = infer_tool_scope(question, workspace_path)"
    )
    build_pos = streaming.index("graph = build_graph(scope=scope)")
    assert infer_pos < build_pos


def test_runner_passes_inferred_scope_to_graph():
    infer_pos = runner.index(
        "scope = infer_tool_scope(question, workspace_path)"
    )
    build_pos = runner.index("graph = build_graph(scope=scope)")
    assert infer_pos < build_pos


def test_live_graph_does_not_hardcode_file_inspection_scope():
    assert 'tools = get_tools_for_scope("file_inspection")' not in graph


def test_resolved_scope_precedes_risk_visibility_and_binding():
    scope_pos = graph.index("tools = get_tools_for_scope(resolved_scope)")
    risk_pos = graph.index("LangGraph tool risk visibility v1")
    bind_pos = graph.index("llm_with_tools = llm.bind_tools(tools)")
    assert scope_pos < risk_pos < bind_pos


def test_execution_time_risk_gate_remains_active():
    assert "decide_tool_call_policy_action(" in graph
    assert "CONFIRMATION_REQUIRED:" in graph
    assert "DESTRUCTIVE_ACTION_BLOCKED:" in graph


def test_legacy_runtime_is_not_part_of_scope_handoff():
    assert "build_graph(scope=scope)" not in legacy