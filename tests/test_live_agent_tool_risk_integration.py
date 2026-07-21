from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

AGENT_LOOP = (ROOT / "src" / "agent_loop.py").read_text(
    encoding="utf-8", errors="replace"
)
CHAT_ROUTES = (ROOT / "routes" / "chat_routes.py").read_text(
    encoding="utf-8", errors="replace"
)
LIVE_SUPERVISOR = (
    ROOT / "src" / "agent_runtime" / "live_supervisor.py"
).read_text(encoding="utf-8", errors="replace")
DISPATCHER = (
    ROOT / "src" / "agent_runtime" / "worker_dispatcher.py"
).read_text(encoding="utf-8", errors="replace")


def test_legacy_agent_loop_has_no_langgraph_policy_instrumentation() -> None:
    assert "build_tool_gate_payload" not in AGENT_LOOP
    assert "gate_tool_execution(" not in AGENT_LOOP
    assert "split_tools_for_direct_execution(" not in AGENT_LOOP


def test_live_route_uses_general_supervisor() -> None:
    assert "stream_general_supervisor_sse" in CHAT_ROUTES
    assert "stream_langgraph_agent_legacy_sse" not in CHAT_ROUTES


def test_live_route_preserves_pre_output_legacy_fallback() -> None:
    assert "_stream_langgraph_first_with_legacy_fallback" in CHAT_ROUTES
    assert "stream_agent_loop" in CHAT_ROUTES


def test_agent_tools_toggle_remains_the_runtime_boundary() -> None:
    assert "agent_langgraph_enabled" in CHAT_ROUTES


def test_general_supervisor_uses_async_dispatcher_and_evidence() -> None:
    assert "dispatch_supervisor_decision_async" in LIVE_SUPERVISOR
    assert "WORKER RESULTS:" in LIVE_SUPERVISOR


def test_all_specialist_workers_are_registered() -> None:
    assert '"code_inspection": run_code_worker' in DISPATCHER
    assert '"knowledge_search": run_knowledge_worker' in DISPATCHER
    assert '"deep_research": run_research_worker' in DISPATCHER
    assert '"mcp_specialized": run_mcp_worker' in DISPATCHER
