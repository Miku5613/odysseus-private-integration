from __future__ import annotations

from src.agent_runtime.knowledge_llm_response_bridge import (
    BRIDGE_VERSION,
    build_knowledge_response_messages,
    knowledge_llm_config_from_runtime_locals,
    try_knowledge_llm_response,
)


def test_missing_model_falls_back_to_deterministic_answer() -> None:
    result = try_knowledge_llm_response(
        llm_config={},
        user_message="read my vault note",
        deterministic_answer="deterministic fallback",
        runtime_decision={"intent": "knowledge_read"},
        evidence_summary={},
        hitl_policy={},
        legacy_fallback_contract={},
        llm_call_func=lambda *a, **k: "should not be called",
    )

    assert result["version"] == BRIDGE_VERSION
    assert result["used_llm"] is False
    assert result["fallback_used"] is True
    assert result["final_answer"] == "deterministic fallback"
    assert result["status"] == "skipped_missing_selected_model"


def test_llm_response_replaces_deterministic_answer_when_available() -> None:
    calls = []

    def fake_llm_call(url, model, messages, **kwargs):
        calls.append((url, model, messages, kwargs))
        return "LLM-authored answer from runtime evidence."

    result = try_knowledge_llm_response(
        llm_config={"url": "http://127.0.0.1:11434/v1/chat/completions", "model": "qwen3:8b"},
        user_message="what did I write in my note?",
        deterministic_answer="raw deterministic answer",
        runtime_decision={"intent": "knowledge_read", "write_allowed": False},
        evidence_summary={"observations_count": 1},
        hitl_policy={"action": "execute_read_graph"},
        legacy_fallback_contract={"ownership": "langgraph"},
        llm_call_func=fake_llm_call,
    )

    assert result["used_llm"] is True
    assert result["fallback_used"] is False
    assert result["status"] == "ok"
    assert result["final_answer"] == "LLM-authored answer from runtime evidence."
    assert calls
    assert calls[0][1] == "qwen3:8b"


def test_llm_error_falls_back_safely() -> None:
    def broken_llm(*args, **kwargs):
        raise RuntimeError("ollama unavailable")

    result = try_knowledge_llm_response(
        llm_config={"url": "http://127.0.0.1:11434/v1/chat/completions", "model": "qwen3:8b"},
        user_message="read note",
        deterministic_answer="safe fallback",
        runtime_decision={},
        evidence_summary={},
        hitl_policy={},
        legacy_fallback_contract={},
        llm_call_func=broken_llm,
    )

    assert result["used_llm"] is False
    assert result["fallback_used"] is True
    assert result["final_answer"] == "safe fallback"
    assert result["status"] == "llm_error"
    assert "RuntimeError" in result["error"]


def test_prompt_contains_runtime_boundaries() -> None:
    messages = build_knowledge_response_messages(
        user_message="delete and summarize my note",
        deterministic_answer="No note was edited.",
        runtime_decision={"intent": "unsupported_mutation", "write_allowed": False},
        evidence_summary={"sufficiency": "blocked"},
        hitl_policy={"action": "block"},
        legacy_fallback_contract={"ownership": "blocked"},
    )

    joined = "\n".join(m["content"] for m in messages)
    assert "runtime, not you, owns permissions" in joined.lower()
    assert "write_allowed" in joined
    assert "No note was edited." in joined


def test_runtime_locals_extractor_uses_context_and_session_objects() -> None:
    class Ctx:
        endpoint_url = "http://127.0.0.1:11434/v1/chat/completions"
        headers = {"x-test": "1"}

    class Sess:
        model = "llama3:latest"

    config = knowledge_llm_config_from_runtime_locals({"ctx": Ctx(), "sess": Sess()})

    assert config["url"].endswith("/chat/completions")
    assert config["model"] == "llama3:latest"
    assert config["headers"] == {"x-test": "1"}
