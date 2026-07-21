from pathlib import Path

from src.agent_runtime.runtime_router import route_agent_runtime
from src.agent_runtime.general_graph import plan_task
from src.agent_runtime import tool_adapters


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


def test_runtime_word_does_not_trigger_write_intent() -> None:
    task = plan_task(
        "Inspect the current Odysseus project and report which specialist "
        "runtime workers are available. Use read-only inspection only."
    )

    assert task.kind == "general_readonly"
    assert task.is_readonly is True
    assert task.should_fallback is False


def test_module_listing_preserves_file_lookup_semantics() -> None:
    task = plan_task("List the modules related to memory in Odysseus.")

    assert task.kind == "file_lookup"
    assert task.is_readonly is True
    assert task.should_fallback is False


def test_contract_tool_default_workspace_is_portable_repo_root() -> None:
    assert tool_adapters.DEFAULT_WORKSPACE == REPO_ROOT


def test_contract_tool_default_workspace_is_not_old_test_worktree() -> None:
    normalized = str(tool_adapters.DEFAULT_WORKSPACE).replace("\\", "/").lower()
    assert "odysseus-upstream-test" not in normalized



def test_worker_inventory_uses_supervisor_capability_registry() -> None:
    from src.agent_runtime.general_graph import (
        _format_worker_inventory,
        _is_worker_inventory_request,
    )

    query = (
        "Inspect the current Odysseus project and report which specialist "
        "runtime workers are available. Use read-only inspection only."
    )

    assert _is_worker_inventory_request(query) is True

    answer = _format_worker_inventory().lower()

    assert "code_worker" in answer
    assert "knowledge_worker" in answer
    assert "research_worker" in answer
    assert "mcp_worker" in answer
    assert ".dockerignore" not in answer


def test_backend_match_rejects_non_source_files() -> None:
    from src.agent_runtime.general_graph import (
        GeneralState,
        GeneralTask,
        choose_best_match,
    )

    state = GeneralState(
        raw_query=(
            "Inspect backend route code related to chat sessions and "
            "summarize what you find. Read only."
        ),
        workspace=r"D:\AI\odysseus",
    )
    state.task = GeneralTask(
        raw_query=state.raw_query,
        kind="route_lookup",
        target="",
        reason="test",
        is_readonly=True,
        should_fallback=False,
    )

    grep_output = "\n".join(
        [
            ".env:28: # RESEARCH_LLM_ENDPOINT=http://localhost:8000/v1/chat/completions",
            "routes/chat_routes.py:643: @router.post(\"/api/chat_stream\")",
        ]
    )

    best = choose_best_match(state, grep_output)

    assert best is not None
    assert best[0].replace("\\", "/") == "routes/chat_routes.py"
