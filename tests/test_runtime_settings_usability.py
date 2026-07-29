from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (ROOT / "static" / "js" / "runtimeControl.js").read_text(
    encoding="utf-8"
)
STYLE = (ROOT / "static" / "style.css").read_text(encoding="utf-8")


def test_runtime_settings_have_concise_connected_help():
    required = [
        "Choose which Agent runtime newly created chats use by default.",
        "Keeps one native OpenCode conversation attached to each Odysseus chat",
        "Controls whether LangGraph answers directly, uses one specialist",
        "Controls what LangGraph may do without approval.",
        "It never falls back the entire query.",
        "Selects the model responsible for planning, worker selection, and replanning.",
        "Selects the model that writes the final user-facing answer",
        "Controls what Odysseus offers when a LangGraph run was interrupted",
        "Uses a suitable completed report before starting a duplicate research job.",
        "Allows LangGraph to read from the currently selected authorized workspace.",
        "Shows live plan and step progress while LangGraph is working.",
    ]
    for sentence in required:
        assert sentence in RUNTIME

    assert 'aria-describedby="${id}-help"' in RUNTIME
    assert 'aria-describedby="runtime-langgraph-context-${key}-help"' in RUNTIME


def test_runtime_settings_collapse_inside_a_floating_container():
    assert "container-type: inline-size" in STYLE
    assert "@container (max-width: 620px)" in STYLE
    assert "grid-template-columns: minmax(0, 1fr)" in STYLE
    assert "width: min(100%, 260px)" in STYLE
    assert ".runtime-select-row" in STYLE
    assert "min-width: 0" in STYLE
    assert "max-width: 100%" in STYLE
