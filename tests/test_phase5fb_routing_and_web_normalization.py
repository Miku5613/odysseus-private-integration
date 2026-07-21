from pathlib import Path

from src.agent_runtime.runtime_router import route_agent_runtime


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_live_runtime_worker_inspection_routes_to_general() -> None:
    decision = route_agent_runtime(
        "Inspect the current Odysseus project and report which specialist "
        "runtime workers are available. Use read-only inspection only."
    )

    assert decision.intent == "general_readonly"
    assert decision.target_runtime == "general"
    assert decision.is_safe_readonly is True


def test_streaming_chat_uses_normalized_web_boolean() -> None:
    source = (
        REPO_ROOT / "routes" / "chat_routes.py"
    ).read_text(encoding="utf-8")

    normalized_block = (
        "att_ids=att_ids,\n"
        "            use_web=_search_enabled,\n"
        "            use_rag=use_rag,\n"
        "            time_filter=time_filter,"
    )

    raw_block = (
        "att_ids=att_ids,\n"
        "            use_web=use_web,\n"
        "            use_rag=use_rag,\n"
        "            time_filter=time_filter,"
    )

    assert normalized_block in source
    assert raw_block not in source
