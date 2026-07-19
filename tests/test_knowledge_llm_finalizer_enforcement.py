from __future__ import annotations

from src.agent_runtime.knowledge_llm_response_bridge import enforce_knowledge_llm_final_answer


RAW_NOTE = """Active Obsidian note content:

Created: 2026-06-19
type: documentation
Area: local-ai

Purpose
This note is the central troubleshooting reference for my Odysseus local AI setup.
"""


def test_successful_llm_finalizer_uses_llm_answer() -> None:
    answer = enforce_knowledge_llm_final_answer(
        final_answer=RAW_NOTE,
        llm_response={
            "status": "ok",
            "used_llm": True,
            "model": "qwen3:8b",
            "final_answer": "I read the active note. It is your Odysseus troubleshooting reference.",
        },
        llm_config={"url": "http://127.0.0.1:11434/v1/chat/completions", "model": "qwen3:8b"},
    )

    assert answer == "I read the active note. It is your Odysseus troubleshooting reference."
    assert "Active Obsidian note content" not in answer


def test_failed_llm_finalizer_does_not_silently_return_raw_note() -> None:
    answer = enforce_knowledge_llm_final_answer(
        final_answer=RAW_NOTE,
        llm_response={
            "status": "llm_error",
            "used_llm": False,
            "model": "qwen3:8b",
            "final_answer": RAW_NOTE,
        },
        llm_config={"url": "http://127.0.0.1:11434/v1/chat/completions", "model": "qwen3:8b"},
    )

    assert "LLM finalizer did not produce" in answer
    assert "Bridge status: llm_error" in answer
    assert "Active Obsidian note content" not in answer


def test_missing_model_config_is_explicit() -> None:
    answer = enforce_knowledge_llm_final_answer(
        final_answer=RAW_NOTE,
        llm_response={
            "status": "skipped_missing_selected_model",
            "used_llm": False,
            "final_answer": RAW_NOTE,
        },
        llm_config=None,
    )

    assert "selected model configuration was not available" in answer
    assert "Bridge status: skipped_missing_selected_model" in answer
    assert "Active Obsidian note content" not in answer


def test_chat_bridge_uses_enforced_finalizer_and_logs_status() -> None:
    text = open("src/agent_runtime/knowledge_chat_bridge.py", "r", encoding="utf-8").read()

    assert "enforce_knowledge_llm_final_answer(" in text
    assert "metrics[\"llm_response_bridge\"] = llm_response" in text
    assert "llm_response_bridge status=%s used_llm=%s model=%s" in text
    assert "if llm_response.get(\"final_answer\"):" not in text
