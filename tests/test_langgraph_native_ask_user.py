from pathlib import Path

from src.agent_runtime.tool_adapters import TOOL_MAP, get_tools_for_scope
from src.agent_runtime.tool_risk_mapper import decide_tool_call_policy_action

ROOT = Path(__file__).resolve().parents[1]

graph = (ROOT / "src/agent_runtime/graph.py").read_text(
    encoding="utf-8", errors="replace"
)
sse = (ROOT / "src/agent_runtime/legacy_sse_adapter.py").read_text(
    encoding="utf-8", errors="replace"
)
legacy = (ROOT / "src/agent_loop.py").read_text(
    encoding="utf-8", errors="replace"
)


def test_ask_user_is_native_and_always_available():
    assert "ask_user" in TOOL_MAP
    assert any(t.name == "ask_user" for t in get_tools_for_scope("file_inspection"))
    assert any(t.name == "ask_user" for t in get_tools_for_scope("knowledge"))


def test_ask_user_is_safe_direct_interaction():
    decision = decide_tool_call_policy_action(
        "ask_user",
        {
            "question": "Which source?",
            "options": [
                {"label": "Notes"},
                {"label": "Files"},
            ],
        },
    )
    assert decision.action == "allow"


def test_graph_handles_ask_user_as_turn_ending_interaction():
    assert 'if name == "ask_user":' in graph
    assert "ask_user ends this turn" in graph
    assert 'getattr(last_message, "name", None) == "ask_user"' in graph


def test_sse_reuses_existing_ask_user_contract():
    assert 'event_type == "ask_user"' in sse
    assert '"type": "ask_user"' in sse
    assert '"options": event.get("options") or []' in sse


def test_legacy_runtime_remains_unmodified():
    assert "_ask_user_marker" not in legacy
    assert 'if name == "ask_user":' not in legacy
