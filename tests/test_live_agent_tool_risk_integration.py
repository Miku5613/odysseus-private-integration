from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

AGENT_LOOP = (
    ROOT / "src" / "agent_loop.py"
).read_text(
    encoding="utf-8",
    errors="replace",
)


def test_live_agent_imports_tool_gate_builder() -> None:
    assert (
        "from src.agent_runtime.tool_risk_mapper "
        "import build_tool_gate_payload"
        in AGENT_LOOP
    )


def test_live_agent_classifies_final_schema_surface() -> None:
    assert (
        "# Tool risk visibility v1 — observational only; "
        "no execution filtering."
        in AGENT_LOOP
    )

    assert "for _tool_schema in (all_tool_schemas or [])" in AGENT_LOOP
    assert "_live_tool_schema_map[str(_name)]" in AGENT_LOOP

    assert (
        "build_tool_gate_payload("
        in AGENT_LOOP
    )

    assert "_live_tool_schema_map" in AGENT_LOOP


def test_live_agent_uses_schema_parameters_for_risk() -> None:
    assert '_function.get("parameters")' in AGENT_LOOP

    # Important: schema-aware classification means multi-action tools can
    # expose destructive actions to the mapper instead of relying only on names.
    assert "_parameters" in AGENT_LOOP


def test_live_agent_emits_policy_counts() -> None:
    assert "[tool-risk] live_surface " in AGENT_LOOP

    for key in (
        "direct",
        "gated",
        "allow",
        "confirm",
        "propose_only",
    ):
        assert (
            f'_tool_gate_payload["counts"]["{key}"]'
            in AGENT_LOOP
        )


def test_patch4i_remains_observational() -> None:
    assert (
        "observational only; no execution filtering"
        in AGENT_LOOP
    )

    # No execution gate is invoked by agent_loop yet.
    assert "gate_tool_execution(" not in AGENT_LOOP
    assert "split_tools_for_direct_execution(" not in AGENT_LOOP


def test_risk_observation_occurs_after_final_schema_build() -> None:
    schema_pos = AGENT_LOOP.index(
        "_tool_names_sent = "
        '[t.get("function", {}).get("name")'
    )

    risk_pos = AGENT_LOOP.index(
        "# Tool risk visibility v1"
    )

    stream_pos = AGENT_LOOP.index(
        "async for chunk in stream_llm_with_fallback(",
        risk_pos,
    )

    assert schema_pos < risk_pos < stream_pos
