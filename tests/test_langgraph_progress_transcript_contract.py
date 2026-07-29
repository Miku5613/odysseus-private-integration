from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_live_progress_is_removed_only_after_successful_llm_finalization():
    source = (ROOT / "static" / "js" / "chat.js").read_text(encoding="utf-8")

    assert "metrics?.response_origin === 'selected_model'" in source
    assert "cursor.classList?.contains('agent-thread')" in source
    assert "cursor.remove()" in source
    assert "controlled_runtime_failure" in source
    assert "LangGraph runtime failure" in source


def test_completed_history_does_not_reconstruct_raw_runtime_decision():
    source = (ROOT / "routes" / "chat_helpers.py").read_text(encoding="utf-8")

    assert 'md.pop("tool_events", None)' in source
    assert 'md.pop("round_texts", None)' in source
    assert 'md.get("response_origin") == "selected_model"' in source


def test_active_and_interrupted_durable_replay_routes_remain_available():
    source = (ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert '"/api/chat/stream_status/{session_id}"' in source
    assert '"/api/chat/resume/{session_id}"' in source
    assert "runtime_replay_complete" in source
