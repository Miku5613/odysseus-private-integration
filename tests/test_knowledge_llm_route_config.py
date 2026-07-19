from __future__ import annotations

from pathlib import Path


def test_chat_routes_passes_live_llm_config_to_knowledge_bridge() -> None:
    text = Path("routes/chat_routes.py").read_text(encoding="utf-8")

    assert "knowledge_llm_config_from_runtime_locals" in text
    assert "_knowledge_llm_config = knowledge_llm_config_from_runtime_locals(locals())" in text
    assert "run_knowledge_chat_turn(message, llm_config=_knowledge_llm_config)" in text


def test_knowledge_bridge_accepts_llm_config_parameter() -> None:
    text = Path("src/agent_runtime/knowledge_chat_bridge.py").read_text(encoding="utf-8")

    assert "llm_config: dict[str, Any] | None = None" in text
    assert "try_knowledge_llm_response(" in text
    assert "metrics[\"llm_response_bridge\"]" in text
