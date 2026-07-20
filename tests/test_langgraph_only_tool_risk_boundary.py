from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LEGACY = (
    ROOT / "src" / "agent_loop.py"
).read_text(
    encoding="utf-8",
    errors="replace",
)

GRAPH = (
    ROOT / "src" / "agent_runtime" / "graph.py"
).read_text(
    encoding="utf-8",
    errors="replace",
)


def test_patch4i_observer_is_removed_from_legacy() -> None:
    assert (
        "from src.agent_runtime.tool_risk_mapper "
        "import build_tool_gate_payload"
        not in LEGACY
    )

    assert (
        "Tool risk visibility v1 — observational only"
        not in LEGACY
    )

    assert "[tool-risk] live_surface" not in LEGACY
    assert "_live_tool_schema_map" not in LEGACY


def test_langgraph_owns_tool_risk_observation() -> None:
    assert (
        "from src.agent_runtime.tool_risk_mapper "
        "import build_tool_gate_payload"
        in GRAPH
    )

    assert (
        "LangGraph tool risk visibility v1 — "
        "LangGraph-only observational policy."
        in GRAPH
    )

    assert "[langgraph-tool-risk]" in GRAPH


def test_langgraph_observes_actual_bound_tool_scope() -> None:
    tools_pos = GRAPH.index(
        'tools = get_tools_for_scope("file_inspection")'
    )

    risk_pos = GRAPH.index(
        "# LangGraph tool risk visibility v1"
    )

    bind_pos = GRAPH.index(
        "llm_with_tools = llm.bind_tools(tools)"
    )

    assert tools_pos < risk_pos < bind_pos


def test_langgraph_observation_uses_real_tool_objects() -> None:
    assert "for tool in (tools or [])" in GRAPH
    assert 'getattr(tool, "name", None)' in GRAPH
    assert "build_tool_gate_payload(" in GRAPH


def test_relocation_is_observational_not_enforcement() -> None:
    assert "gate_tool_execution(" not in GRAPH


def test_legacy_existing_functionality_is_not_reset() -> None:
    # These pre-existing local behaviors are intentionally preserved.
    assert "WEB_TOOL_NAMES" in LEGACY

    assert (
        "_FORCE_LOCAL_NATIVE_TOOLS_AFTER_DECISION_20260630"
        in LEGACY
    )
