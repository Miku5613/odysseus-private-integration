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


def test_negated_mutation_constraint_remains_readonly() -> None:
    decision = route_agent_runtime(
        "Inspect the current Odysseus repository and list the runtime files. "
        "Do not modify any file."
    )

    assert decision.intent == "file_lookup"
    assert decision.target_runtime == "general"
    assert decision.is_safe_readonly is True


def test_negated_mutation_does_not_hide_later_positive_write() -> None:
    decision = route_agent_runtime(
        "Do not edit the existing source, but create a new report file."
    )

    assert decision.intent == "write_mutation"
    assert decision.target_runtime == "legacy"
    assert decision.is_safe_readonly is False


def test_multiple_negated_mutation_terms_remain_readonly() -> None:
    decision = route_agent_runtime(
        "List the project modules. Do not edit, rename, or delete any files."
    )

    assert decision.intent == "file_lookup"
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


def test_backend_route_lookup_uses_routes_scope(monkeypatch) -> None:
    from src.agent_runtime.general_graph import (
        GeneralState,
        execute_readonly_plan,
        plan_task,
    )
    from src.agent_runtime import general_graph

    query = (
        "Inspect backend route code related to chat sessions and "
        "summarize what you find. Read only."
    )

    task = plan_task(query)

    assert task.kind == "route_lookup"

    calls = []

    monkeypatch.setattr(
        general_graph.tools,
        "get_workspace",
        lambda workspace: workspace,
    )

    def fake_grep(workspace, pattern, path=".", max_results=80):
        calls.append(
            {
                "workspace": workspace,
                "pattern": pattern,
                "path": path,
                "max_results": max_results,
            }
        )
        return (
            "routes/chat_routes.py:643: "
            '@router.post("/api/chat_stream")'
        )

    monkeypatch.setattr(
        general_graph.tools,
        "grep",
        fake_grep,
    )

    monkeypatch.setattr(
        general_graph.tools,
        "read_file_around",
        lambda workspace, path, line, radius=16: (
            "# File: routes/chat_routes.py\n"
            ">> 643: @router.post(\"/api/chat_stream\")"
        ),
    )

    state = GeneralState(
        raw_query=query,
        workspace=r"D:\AI\odysseus",
    )
    state.task = task

    execute_readonly_plan(state)

    assert calls

    searched_paths = [
        call["path"].replace("\\", "/")
        for call in calls
    ]

    assert all(
        path.startswith("routes/")
        for path in searched_paths
    )

    assert not any(
        path.startswith((".env", "docs/", "tests/", "Exclusions/"))
        for path in searched_paths
    )



def test_phase5fp_file_first_semantic_inspection(monkeypatch) -> None:
    from src.agent_runtime import general_graph
    from src.agent_runtime.general_graph import (
        GeneralState,
        execute_readonly_plan,
        plan_task,
    )

    query = (
        "Inspect backend route code related to chat sessions "
        "and summarize what you find. Read only."
    )

    task = plan_task(query)
    assert task.kind == "route_lookup"

    files = "\n".join(
        [
            "routes/admin_wipe_routes.py",
            "routes/chat_routes.py",
            "routes/session_routes.py",
        ]
    )

    def fake_glob(workspace, pattern, max_results=120):
        return files if pattern.startswith("routes/") else ""

    def fake_grep(workspace, pattern, path="", max_results=12):
        data = {
            "routes/admin_wipe_routes.py": (
                "routes/admin_wipe_routes.py:5: wipe chats\n"
                "routes/admin_wipe_routes.py:6: /api/sessions/all"
            ),
            "routes/chat_routes.py": (
                "routes/chat_routes.py:643: @router.post('/api/chat_stream')\n"
                "routes/chat_routes.py:700: session = session_manager.get_session(session_id)\n"
                "routes/chat_routes.py:710: chat_handler.preprocess_message(session)"
            ),
            "routes/session_routes.py": (
                "routes/session_routes.py:20: @router.get('/api/sessions')\n"
                "routes/session_routes.py:25: return session_manager.sessions"
            ),
        }
        return data.get(path, "")

    monkeypatch.setattr(
        general_graph.tools,
        "get_workspace",
        lambda workspace: workspace,
    )
    monkeypatch.setattr(
        general_graph.tools,
        "glob_files",
        fake_glob,
    )
    monkeypatch.setattr(
        general_graph.tools,
        "grep",
        fake_grep,
    )

    state = GeneralState(
        raw_query=query,
        workspace=r"D:\AI\odysseus",
    )
    state.task = task

    execute_readonly_plan(state)

    assert "routes/chat_routes.py" in state.final_answer
    assert "routes/session_routes.py" in state.final_answer
    assert ".env" not in state.final_answer


def test_phase5fp_email_polling_prefers_email_sources(monkeypatch) -> None:
    from src.agent_runtime import general_graph
    from src.agent_runtime.general_graph import (
        GeneralState,
        execute_readonly_plan,
        plan_task,
    )

    query = (
        "Inspect backend route code related to email polling "
        "and summarize what you find. Read only."
    )

    task = plan_task(query)

    files = "\n".join(
        [
            "routes/chat_routes.py",
            "routes/email_pollers.py",
            "routes/email_routes.py",
        ]
    )

    def fake_glob(workspace, pattern, max_results=120):
        return files if pattern.startswith("routes/") else ""

    def fake_grep(workspace, pattern, path="", max_results=12):
        data = {
            "routes/chat_routes.py": (
                "routes/chat_routes.py:800: active_email_context = True"
            ),
            "routes/email_pollers.py": (
                "routes/email_pollers.py:40: async def poll_email_accounts():\n"
                "routes/email_pollers.py:55: email polling interval"
            ),
            "routes/email_routes.py": (
                "routes/email_routes.py:100: @router.get('/api/email')"
            ),
        }
        return data.get(path, "")

    monkeypatch.setattr(
        general_graph.tools,
        "get_workspace",
        lambda workspace: workspace,
    )
    monkeypatch.setattr(
        general_graph.tools,
        "glob_files",
        fake_glob,
    )
    monkeypatch.setattr(
        general_graph.tools,
        "grep",
        fake_grep,
    )

    state = GeneralState(
        raw_query=query,
        workspace=r"D:\AI\odysseus",
    )
    state.task = task

    execute_readonly_plan(state)

    assert "routes/email_pollers.py" in state.final_answer


def test_phase5fp_frontend_inspection_stays_in_frontend(monkeypatch) -> None:
    from src.agent_runtime import general_graph
    from src.agent_runtime.general_graph import (
        GeneralState,
        execute_readonly_plan,
        plan_task,
    )

    query = (
        "Inspect frontend code related to chat streaming "
        "and summarize what you find. Read only."
    )

    task = plan_task(query)

    files = "\n".join(
        [
            "static/js/chat.js",
            "static/js/chatStream.js",
            "static/js/streamingRenderer.js",
        ]
    )

    def fake_glob(workspace, pattern, max_results=120):
        return files if pattern.startswith("static/js/") else ""

    def fake_grep(workspace, pattern, path="", max_results=12):
        data = {
            "static/js/chat.js": (
                "static/js/chat.js:100: start chat stream"
            ),
            "static/js/chatStream.js": (
                "static/js/chatStream.js:20: export async function streamChat()\n"
                "static/js/chatStream.js:30: streaming response reader"
            ),
            "static/js/streamingRenderer.js": (
                "static/js/streamingRenderer.js:10: render streaming segment"
            ),
        }
        return data.get(path, "")

    monkeypatch.setattr(
        general_graph.tools,
        "get_workspace",
        lambda workspace: workspace,
    )
    monkeypatch.setattr(
        general_graph.tools,
        "glob_files",
        fake_glob,
    )
    monkeypatch.setattr(
        general_graph.tools,
        "grep",
        fake_grep,
    )

    state = GeneralState(
        raw_query=query,
        workspace=r"D:\AI\odysseus",
    )
    state.task = task

    execute_readonly_plan(state)

    assert "static/js/chatStream.js" in state.final_answer
    assert ".env" not in state.final_answer



def test_phase5fq_runtime_logic_cleanup() -> None:
    from src.agent_runtime.general_graph import (
        _inspection_terms,
        plan_task,
    )

    terms = _inspection_terms(
        "Which backend source files are responsible for "
        "chat session handling? Inspect read-only."
    )

    assert "chat" in terms
    assert "session" in terms
    assert "files" not in terms
    assert "responsible" not in terms
    assert "handling" not in terms

    task = plan_task(
        "Inspect backend code related to calendar events "
        "and summarize what you find. Read only."
    )

    assert task.kind == "code_lookup"


def test_phase5fq_file_lookup_semantic_fallback(
    monkeypatch,
) -> None:
    from src.agent_runtime import general_graph
    from src.agent_runtime.general_graph import (
        GeneralState,
        execute_readonly_plan,
        plan_task,
    )

    query = (
        "Which frontend files implement streaming chat "
        "responses? Inspect the source read-only."
    )

    task = plan_task(query)
    assert task.kind == "file_lookup"

    files = "\n".join(
        [
            "static/js/chat.js",
            "static/js/chatStream.js",
            "static/js/streamingRenderer.js",
            "static/js/cookbook.js",
        ]
    )

    calls = {"count": 0}

    def fake_glob(
        workspace,
        pattern,
        max_results=40,
    ):
        calls["count"] += 1

        # Deterministic topic globs miss the paraphrase.
        if calls["count"] <= 2:
            return ""

        if pattern.startswith("static/js/"):
            return files

        return ""

    monkeypatch.setattr(
        general_graph.tools,
        "get_workspace",
        lambda workspace: workspace,
    )

    monkeypatch.setattr(
        general_graph.tools,
        "glob_files",
        fake_glob,
    )

    state = GeneralState(
        raw_query=query,
        workspace=r"D:\AI\odysseus",
    )

    state.task = task

    execute_readonly_plan(state)

    answer = state.final_answer.lower()

    assert (
        "chat.js" in answer
        or "chatstream.js" in answer
    )

    assert "cookbook.js" not in answer


def test_phase5fq_path_relevance_beats_incidental_mentions(
    monkeypatch,
) -> None:
    from src.agent_runtime import general_graph
    from src.agent_runtime.general_graph import (
        GeneralState,
        execute_readonly_plan,
        plan_task,
    )

    query = (
        "Inspect backend route code related to chat sessions "
        "and summarize what you find. Read only."
    )

    task = plan_task(query)

    files = "\n".join(
        [
            "routes/admin_wipe_routes.py",
            "routes/session_routes.py",
            "routes/chat_routes.py",
            "routes/chat_helpers.py",
        ]
    )

    def fake_glob(
        workspace,
        pattern,
        max_results=120,
    ):
        return (
            files
            if pattern.startswith("routes/")
            else ""
        )

    def fake_grep(
        workspace,
        pattern,
        path="",
        max_results=12,
    ):
        data = {
            "routes/admin_wipe_routes.py": (
                "routes/admin_wipe_routes.py:5: "
                "chats sessions wipe\n"
                "routes/admin_wipe_routes.py:6: "
                "chats sessions delete"
            ),
            "routes/session_routes.py": (
                "routes/session_routes.py:1: "
                "session routes\n"
                "routes/session_routes.py:10: "
                "SessionManager"
            ),
            "routes/chat_routes.py": (
                "routes/chat_routes.py:1: "
                "chat routes\n"
                "routes/chat_routes.py:643: "
                "/api/chat_stream"
            ),
            "routes/chat_helpers.py": (
                "routes/chat_helpers.py:1: "
                "chat helpers"
            ),
        }

        return data.get(path, "")

    monkeypatch.setattr(
        general_graph.tools,
        "get_workspace",
        lambda workspace: workspace,
    )

    monkeypatch.setattr(
        general_graph.tools,
        "glob_files",
        fake_glob,
    )

    monkeypatch.setattr(
        general_graph.tools,
        "grep",
        fake_grep,
    )

    state = GeneralState(
        raw_query=query,
        workspace=r"D:\AI\odysseus",
    )

    state.task = task

    execute_readonly_plan(state)

    assert "session_routes.py" in state.final_answer
    assert "chat_routes.py" in state.final_answer
    assert "admin_wipe_routes.py" not in state.final_answer



def test_phase5fs_contextual_source_retrieval(
    monkeypatch,
) -> None:
    from src.agent_runtime import general_graph
    from src.agent_runtime.general_graph import (
        GeneralState,
        execute_readonly_plan,
        plan_task,
    )

    query = (
        "Inspect backend route code related to chat sessions "
        "and summarize what you find. Read only."
    )

    task = plan_task(query)

    files = "\n".join(
        [
            "Exclusions/Backups/routes/chat_routes.py.before_patch",
            "Exclusions/Scripts/test_prompt_harness.py",
            "routes/chat_routes.py",
            "routes/session_routes.py",
        ]
    )

    def fake_glob(
        workspace,
        pattern,
        max_results=120,
    ):
        return (
            files
            if pattern.startswith("routes/")
            else ""
        )

    # Phase 5FS must not require real Git metadata in unit tests.
    def fake_context(workspace, path, repository_context=None):
        if path.startswith("Exclusions/Backups/"):
            return {
                "path": path,
                "role": "historical_backup",
                "tracked": False,
                "ignored": True,
                "modified": False,
                "mtime": 10,
                "caveats": [
                    "appears to be a backup or historical copy",
                    "not tracked by the current Git index",
                ],
            }

        if path.startswith("Exclusions/Scripts/"):
            return {
                "path": path,
                "role": "local_script",
                "tracked": False,
                "ignored": True,
                "modified": False,
                "mtime": 30,
                "caveats": [
                    "appears to be a local script or harness",
                ],
            }

        return {
            "path": path,
            "role": "source",
            "tracked": True,
            "ignored": False,
            "modified": False,
            "mtime": 20,
            "caveats": [],
        }

    def fake_grep(
        workspace,
        pattern,
        path="",
        max_results=12,
    ):
        data = {
            "Exclusions/Backups/routes/chat_routes.py.before_patch": (
                "Exclusions/Backups/routes/chat_routes.py.before_patch:"
                "10: chat session handler"
            ),
            "Exclusions/Scripts/test_prompt_harness.py": (
                "Exclusions/Scripts/test_prompt_harness.py:"
                "1: Inspect backend route code related to chat sessions"
            ),
            "routes/chat_routes.py": (
                "routes/chat_routes.py:"
                "643: @router.post('/api/chat_stream')\n"
                "routes/chat_routes.py:"
                "700: session_manager.get_session(session_id)"
            ),
            "routes/session_routes.py": (
                "routes/session_routes.py:"
                "10: from core.session_manager import SessionManager"
            ),
        }

        return data.get(path, "")

    monkeypatch.setattr(
        general_graph.tools,
        "get_workspace",
        lambda workspace: workspace,
    )

    monkeypatch.setattr(
        general_graph.tools,
        "glob_files",
        fake_glob,
    )

    monkeypatch.setattr(
        general_graph.tools,
        "grep",
        fake_grep,
    )

    monkeypatch.setattr(
        general_graph,
        "_source_provenance_context",
        fake_context,
    )

    state = GeneralState(
        raw_query=query,
        workspace=r"D:\AI\odysseus",
    )
    state.task = task

    execute_readonly_plan(state)

    answer = state.final_answer

    assert "routes/chat_routes.py" in answer
    assert "routes/session_routes.py" in answer

    # The test-harness literal match must not beat live semantic source.
    first_source = next(
        line
        for line in answer.splitlines()
        if line.startswith("- ")
    )

    assert "Exclusions/Scripts" not in first_source


def test_phase5fs_unusual_sources_are_not_blocked() -> None:
    from src.agent_runtime.general_graph import (
        _contextual_source_score,
    )

    provenance = {
        "role": "historical_backup",
        "tracked": False,
        "ignored": True,
        "modified": False,
        "mtime": 1,
        "caveats": [
            "appears to be a backup or historical copy",
        ],
    }

    score, details = _contextual_source_score(
        path=(
            "Exclusions/Backups/routes/"
            "chat_routes.py.before_patch"
        ),
        output=(
            "tool_output chat session route"
        ),
        terms=["chat", "session"],
        raw_query=(
            "Inspect backend route code related "
            "to chat sessions"
        ),
        provenance=provenance,
    )

    # It receives a real score rather than being excluded.
    assert isinstance(score, int)
    assert score > 0
    assert details["coverage"] >= 1



def test_phase5fsc_non_git_workspace_skips_git(
    tmp_path,
    monkeypatch,
) -> None:
    import subprocess

    from src.agent_runtime.general_graph import (
        _build_repository_context,
    )

    calls = []

    def forbidden_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(
            "Git subprocess must not run for a non-Git workspace."
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        forbidden_run,
    )

    context = _build_repository_context(
        str(tmp_path),
    )

    assert context == {
        "tracked": set(),
        "ignored": set(),
        "modified": set(),
    }

    assert calls == []
