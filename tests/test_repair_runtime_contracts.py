from __future__ import annotations

import json

import pytest


def test_exact_lookup_requires_a_precise_location_shape():
    from src.agent_runtime.runtime_router import route_agent_runtime

    exact = route_agent_runtime("Find where doc_stream is handled")
    broad = route_agent_runtime(
        "Inspect the backend and tell me which code handles conversations."
    )

    assert exact.intent == "exact_lookup"
    assert exact.target_runtime == "contract"
    assert broad.intent == "general_readonly"
    assert broad.target_runtime == "general"


def test_workspace_authority_covers_contextual_questions_not_other_domains():
    from src.agent_runtime.knowledge_semantic_chat_bridge import (
        should_intercept_workspace_knowledge,
    )

    assert should_intercept_workspace_knowledge(
        "What should I work on next?"
    )
    assert not should_intercept_workspace_knowledge(
        "Find where doc_stream is handled"
    )
    assert not should_intercept_workspace_knowledge(
        "Research the latest local AI news"
    )
    assert not should_intercept_workspace_knowledge(
        "Hey, how are you doing today?"
    )


def test_rewrite_of_a_note_is_policy_gated_knowledge_mutation():
    from src.agent_runtime.knowledge_graph_router import evaluate_knowledge_request

    decision = evaluate_knowledge_request(
        "Rewrite goals.md to change its primary objective."
    )

    assert decision.intent == "knowledge_propose"
    assert decision.write_allowed is False
    assert decision.needs_human is True


def test_contract_filesystem_search_excludes_artifact_trees(tmp_path):
    from src.agent_runtime import tool_adapters

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "live.py").write_text(
        "TARGET_CONTRACT_SYMBOL = True\n",
        encoding="utf-8",
    )
    (tmp_path / "Exclusions" / "Scripts").mkdir(parents=True)
    (tmp_path / "Exclusions" / "Scripts" / "matrix.py").write_text(
        "TARGET_CONTRACT_SYMBOL = 'prompt copy'\n",
        encoding="utf-8",
    )
    (tmp_path / "Backups").mkdir()
    (tmp_path / "Backups" / "live.py.before_patch").write_text(
        "TARGET_CONTRACT_SYMBOL = 'historical'\n",
        encoding="utf-8",
    )

    token = tool_adapters.bind_workspace(tmp_path)
    try:
        output = tool_adapters.grep.invoke({
            "pattern": "TARGET_CONTRACT_SYMBOL",
            "path": ".",
            "max_matches": 20,
        })
    finally:
        tool_adapters.reset_workspace(token)

    normalized = output.replace("\\", "/")
    assert "src/live.py" in normalized
    assert "Exclusions/" not in normalized
    assert "Backups/" not in normalized


def test_semantic_workspace_stream_emits_common_runtime_contract(monkeypatch):
    from src.agent_runtime import knowledge_semantic_chat_bridge as bridge
    from src.agent_runtime.supervisor_contracts import WorkerResult

    monkeypatch.setattr(
        bridge,
        "run_knowledge_worker",
        lambda handoff: WorkerResult(
            capability="knowledge_search",
            status="completed",
            summary="No match.",
            data={"root_id": "root", "results": []},
        ),
    )
    monkeypatch.setattr(bridge, "get_semantic_vault_stats", lambda **kwargs: {})
    monkeypatch.setattr(bridge, "generate_intent_preface", lambda **kwargs: {})

    _answer, frames, _events, _metrics = bridge.run_knowledge_semantic_chat_turn(
        "What should I do next?",
        workspace="D:/workspace",
    )
    payloads = [
        json.loads(frame[len("data: "):])
        for frame in frames
        if frame.startswith("data: ")
    ]
    runtime = next(
        item for item in payloads
        if item.get("tool") == "runtime_decision"
        and item.get("type") == "tool_output"
    )

    assert runtime["runtime_decision"]["runtime"] == "knowledge"
    assert runtime["runtime_decision"]["runtime_family"] == "workspace_semantic"


@pytest.mark.asyncio
async def test_web_search_capability_dispatches_through_tool_gate(monkeypatch):
    from src.agent_runtime.shared_tool_worker import run_web_search_worker
    from src.agent_runtime.supervisor_contracts import WorkerHandoff
    import src.tool_execution as tool_execution

    async def fake_execute(block, **kwargs):
        assert block.tool_type == "web_search"
        assert json.loads(block.content)["query"] == "current release notes"
        return "web_search", {"output": "grounded result", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "execute_tool_block", fake_execute)

    result = await run_web_search_worker(
        WorkerHandoff(
            capability="web_search",
            objective="current release notes",
            context={"query": "current release notes"},
        )
    )

    assert result.status == "completed"
    assert result.summary == "grounded result"


def test_builtin_dispatcher_registers_shared_web_search():
    from src.agent_runtime.worker_dispatcher import builtin_worker_runners

    assert "web_search" in builtin_worker_runners()


def test_mutation_requests_use_central_gate_without_execution():
    from src.agent_runtime.mutation_request_policy import evaluate_mutation_request

    destructive = evaluate_mutation_request("Delete the obsolete source file.")
    write = evaluate_mutation_request("Edit the backend route implementation.")

    assert destructive["allowed"] is False
    assert destructive["decision"]["action"] == "propose_only"
    assert write["allowed"] is False
    assert write["decision"]["action"] == "confirm"


def test_contract_ranking_prefers_operational_use_over_router_alias(monkeypatch):
    from src.agent_runtime import contract_runtime
    from src.agent_runtime.contract_models import SubTask

    output = "\n".join([
        "src/agent_runtime/contract_runtime.py:104: ([\"document stream\"], \"doc_stream_open\"),",
        "src/agent_loop.py:3443: yield event({\"type\": \"doc_stream_open\"})",
    ])
    task = SubTask(
        id="task_001",
        kind="exact_lookup",
        target="doc_stream",
        source_scope=["."],
        required_evidence="Find operational evidence.",
    )
    monkeypatch.setattr(
        contract_runtime,
        "_contextual_match_bonus",
        lambda *args, **kwargs: 0.0,
    )

    match = contract_runtime._first_match(output, task)

    assert match == ("src/agent_loop.py", 3443)


def test_deterministic_bridges_default_on_but_keep_kill_switches(monkeypatch):
    from src.agent_runtime.contract_bridge import contract_runtime_enabled
    from src.agent_runtime.general_bridge import general_runtime_enabled

    monkeypatch.delenv("ODYSSEUS_ENABLE_CONTRACT_RUNTIME", raising=False)
    monkeypatch.delenv("ODYSSEUS_ENABLE_GENERAL_RUNTIME", raising=False)
    assert contract_runtime_enabled() is True
    assert general_runtime_enabled() is True

    monkeypatch.setenv("ODYSSEUS_ENABLE_CONTRACT_RUNTIME", "false")
    monkeypatch.setenv("ODYSSEUS_ENABLE_GENERAL_RUNTIME", "false")
    assert contract_runtime_enabled() is False
    assert general_runtime_enabled() is False


def test_general_bridge_keeps_runner_importable_from_external_workspace(tmp_path):
    import asyncio

    from src.agent_runtime.general_bridge import stream_general_runtime_legacy_sse

    (tmp_path / "travel.md").write_text("# Travel\nKyoto itinerary", encoding="utf-8")
    (tmp_path / "packing.md").write_text("# Packing\nPassport", encoding="utf-8")

    async def collect():
        return [
            frame
            async for frame in stream_general_runtime_legacy_sse(
                "List the markdown files related to travel and packing in this workspace. Read only.",
                workspace=str(tmp_path),
            )
        ]

    combined = "".join(asyncio.run(collect()))
    assert "travel.md" in combined
    assert "packing.md" in combined
    assert "No module named 'src'" not in combined


def test_workspace_knowledge_uses_grounded_llm_synthesis(monkeypatch):
    from src.agent_runtime import knowledge_semantic_chat_bridge as bridge
    from src.agent_runtime.supervisor_contracts import WorkerResult

    monkeypatch.setattr(bridge, "generate_intent_preface", lambda **kwargs: {})
    monkeypatch.setattr(bridge, "get_semantic_vault_stats", lambda **kwargs: {})
    monkeypatch.setattr(
        bridge,
        "run_knowledge_worker",
        lambda handoff: WorkerResult(
            capability="knowledge_search",
            status="completed",
            summary="Evidence retrieved.",
            data={
                "root_id": "workspace_root",
                "results": [{
                    "title": "Budget",
                    "path": "budget.md",
                    "heading": "Totals",
                    "preview": "The grounded total is 42.",
                    "similarity": 0.9,
                }],
            },
        ),
    )
    monkeypatch.setattr(
        bridge,
        "try_knowledge_llm_response",
        lambda **kwargs: {
            "used_llm": True,
            "status": "ok",
            "final_answer": "The grounded answer is 42.",
        },
    )

    answer, _frames, _events, metrics = bridge.run_knowledge_semantic_chat_turn(
        "What is the total?",
        workspace="D:/workspace",
        llm_config={"url": "http://model", "model": "test"},
    )

    assert answer == "The grounded answer is 42."
    assert metrics["llm_synthesis"] is True
    assert metrics["source_authority"] == "workspace"


def test_mutation_policy_path_does_not_generate_read_preface(monkeypatch):
    from src.agent_runtime import knowledge_semantic_chat_bridge as bridge

    def forbidden_preface(**kwargs):
        raise AssertionError("mutation policy must not generate a read preface")

    monkeypatch.setattr(bridge, "generate_intent_preface", forbidden_preface)

    answer, _frames, _events, _metrics = bridge.run_knowledge_semantic_chat_turn(
        "Rewrite goals.md with a different objective.",
        workspace="D:/workspace",
    )

    assert "cannot apply" in answer.lower()
    assert "no files were changed" in answer.lower()
