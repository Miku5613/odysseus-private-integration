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
    assert should_intercept_workspace_knowledge(
        "Add up the electricity and internet bills in this workspace."
    )
    assert should_intercept_workspace_knowledge(
        "Review this workspace and suggest the next actions."
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


def test_file_discovery_paraphrase_routes_to_general_runtime():
    from src.agent_runtime.runtime_router import route_agent_runtime

    decision = route_agent_runtime(
        "Show the travel and packing Markdown files available here. Read only."
    )

    assert decision.intent == "file_lookup"
    assert decision.target_runtime == "general"


def test_rewrite_of_a_note_is_policy_gated_knowledge_mutation():
    from src.agent_runtime.knowledge_graph_router import evaluate_knowledge_request

    decision = evaluate_knowledge_request(
        "Rewrite goals.md to change its primary objective."
    )

    assert decision.intent == "knowledge_propose"
    assert decision.write_allowed is False
    assert decision.needs_human is True


def test_contract_filesystem_search_ranks_artifacts_instead_of_discarding_them(tmp_path):
    from src.agent_runtime import tool_adapters
    from src.agent_runtime.contract_models import SubTask
    from src.agent_runtime.contract_runtime import _rank_matches_with_provenance

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
            "include_low_authority": True,
        })
    finally:
        tool_adapters.reset_workspace(token)

    normalized = output.replace("\\", "/")
    assert "src/live.py" in normalized
    assert "Exclusions/Scripts/matrix.py" in normalized
    assert "Backups/live.py.before_patch" in normalized

    task = SubTask(
        id="task_001",
        kind="exact_lookup",
        target="TARGET_CONTRACT_SYMBOL",
        source_scope=["."],
        dependencies=[],
        is_lookup_only=True,
        required_evidence="Find the exact implementation source.",
    )
    ranked = _rank_matches_with_provenance(
        output,
        task,
        str(tmp_path),
    )

    assert ranked[0]["path"].replace("\\", "/") == "src/live.py"
    assert ranked[0]["provenance"]["role"] == "source"
    assert {
        item["provenance"].get("role")
        for item in ranked
    } >= {"source", "historical_backup", "local_script"}
    assert ranked[0]["total_score"] > ranked[-1]["total_score"]


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


def test_broad_workspace_planning_uses_domain_neutral_retrieval_facets():
    from src.agent_runtime.knowledge_semantic_chat_bridge import (
        _planning_retrieval_queries,
    )

    queries = _planning_retrieval_queries(
        "Based on my workspace, what should I focus on next?"
    )

    assert len(queries) >= 3
    assert any("deadline" in query for query in queries)
    assert any("bill" in query for query in queries)
    assert any("next action" in query for query in queries)
    assert _planning_retrieval_queries(
        "What are the three most important things I should focus on next?"
    )
    assert _planning_retrieval_queries("What is my dentist appointment time?") == []
    assert _planning_retrieval_queries(
        "What were my 3D learning priorities?"
    ) == []
    assert _planning_retrieval_queries(
        "Give me my trip priorities and the most important packing items."
    ) == []
    assert _planning_retrieval_queries(
        "What should I do about the electricity bill next?"
    ) == []


def test_comprehensive_planning_answer_keeps_distinct_obligations_and_actions():
    from src.agent_runtime.knowledge_semantic_chat_bridge import (
        _format_comprehensive_planning_answer,
    )

    answer = _format_comprehensive_planning_answer([
        {"fact": "Electricity bill is due 24 July 2026", "path": "bills.md", "dated": True},
        {"fact": "Return library books by 26 July 2026", "path": "library.md", "dated": True},
        {"fact": "Next action: finish the modular asset kit", "path": "project.md", "dated": False},
        {"fact": "Priority: complete the Python exercise", "path": "learning.md", "dated": False},
    ])

    assert "Electricity bill" in answer
    assert "Return library books" in answer
    assert "modular asset kit" in answer
    assert "Python exercise" in answer
    assert "No workspace files were changed" in answer


def test_knowledge_worker_merges_multi_query_workspace_evidence(monkeypatch):
    from src.agent_runtime import knowledge_worker as worker
    from src.agent_runtime.supervisor_contracts import WorkerHandoff

    monkeypatch.setattr(
        worker,
        "find_root_by_path",
        lambda path: {"root_id": "workspace_root"},
    )

    def fake_search(root_id, query, k=8):
        assert root_id == "workspace_root"
        if "deadline" in query:
            return [{
                "path": "deadlines.md",
                "title": "Deadlines",
                "heading": "Deadlines",
                "preview": "Assignment due Friday",
                "similarity": 0.8,
            }]
        if "bill" in query:
            return [{
                "path": "bills.md",
                "title": "Bills",
                "heading": "Bills",
                "preview": "Electricity payment due Thursday",
                "similarity": 0.9,
            }]
        return [{
            "path": "deadlines.md",
            "title": "Deadlines",
            "heading": "Deadlines",
            "preview": "Assignment due Friday",
            "similarity": 0.2,
        }]

    monkeypatch.setattr(worker, "search_root", fake_search)
    monkeypatch.setattr(
        worker,
        "search_root_queries",
        lambda root_id, queries, k=8: [
            fake_search(root_id, query, k=k)
            for query in queries
        ],
    )
    result = worker.run_knowledge_worker(
        WorkerHandoff(
            capability="knowledge_search",
            objective="What should I do next?",
            context={
                "query": "What should I do next?",
                "workspace": "D:/workspace",
                "retrieval_queries": [
                    "deadlines and obligations",
                    "bills and payments",
                ],
            },
        )
    )

    assert result.status == "completed"
    assert result.data["result_count"] == 2
    assert [item["path"] for item in result.data["results"]] == [
        "bills.md",
        "deadlines.md",
    ]
    assert len(result.data["retrieval_queries"]) == 3


def test_currency_total_fact_uses_only_query_named_entities():
    from src.agent_runtime.knowledge_semantic_chat_bridge import (
        _build_currency_total_fact,
    )

    fact = _build_currency_total_fact(
        "How much are hosting and electricity together?",
        [{
            "path": "expenses.md",
            "preview": (
                "Hosting: ₹999\n"
                "Electricity: ₹2,340\n"
                "Optional course: ₹1,499\n"
            ),
        }],
    )

    assert fact is not None
    assert fact["total_plain"] == "3339"
    assert fact["total_formatted"] == "3,339"
    assert [item["entity"] for item in fact["operands"]] == [
        "hosting",
        "electricity",
    ]
    assert _build_currency_total_fact(
        "What did electricity cost?",
        [{"preview": "Electricity: ₹2,340"}],
    ) is None


def test_planning_agenda_extracts_dated_obligations_and_next_actions():
    from src.agent_runtime.knowledge_semantic_chat_bridge import (
        _build_planning_agenda,
    )

    agenda = _build_planning_agenda([
        {
            "path": "deadlines.md",
            "preview": (
                "Library books:\n- Return: 25 July 2026\n"
                "Python assignment:\n- Submit: 26 July 2026\n"
            ),
        },
        {
            "path": "project.md",
            "preview": "Next steps:\n1. finish reusable asset kit\n",
        },
    ])

    facts = "\n".join(item["fact"] for item in agenda).lower()
    assert "library books: return: 25 july 2026" in facts
    assert "python assignment: submit: 26 july 2026" in facts
    assert "finish reusable asset kit" in facts


def test_balanced_priorities_mix_urgent_and_distinct_action_sources():
    from src.agent_runtime.knowledge_semantic_chat_bridge import (
        _select_balanced_planning_priorities,
    )

    selected = _select_balanced_planning_priorities(
        "What are the three most important things I should focus on next?",
        [
            {"path": "calendar.md", "fact": "Renew permit: Due: 24 July 2026", "dated": True},
            {"path": "calendar.md", "fact": "Appointment: Date: 25 July 2026", "dated": True},
            {"path": "learning.md", "fact": "Priority: Finish the current chapter", "dated": False},
            {"path": "project.md", "fact": "Next steps: finish reusable asset kit", "dated": False},
        ],
    )

    assert [item["path"] for item in selected] == [
        "calendar.md",
        "learning.md",
        "project.md",
    ]

    from src.agent_runtime.knowledge_semantic_chat_bridge import (
        _format_balanced_priority_answer,
    )
    answer = _format_balanced_priority_answer(selected)
    assert "renew permit" in answer.lower()
    assert "finish the current chapter" in answer.lower()
    assert "finish reusable asset kit" in answer.lower()
    assert "no files were changed" in answer.lower()


def test_knowledge_metrics_use_common_sse_data_contract(monkeypatch):
    import json

    from src.agent_runtime import knowledge_semantic_chat_bridge as bridge

    answer, frames, _events, metrics = bridge.run_knowledge_semantic_chat_turn(
        "Rewrite planning.md with a new priority.",
        workspace="D:/workspace",
    )
    payloads = [
        json.loads(frame.removeprefix("data: ").strip())
        for frame in frames
        if frame.startswith("data: {")
    ]
    metric_frame = next(item for item in payloads if item.get("type") == "metrics")
    assert metric_frame["data"] == metrics
    assert "metrics" not in metric_frame
    assert "cannot apply" in answer.lower()


def test_semantic_multi_query_batches_embeddings_and_decodes_rows(monkeypatch):
    from src.agent_runtime import knowledge_vault_semantic as semantic

    class FakeEmbedding:
        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts):
            assert texts == ["deadlines", "active projects"]
            return type("Vectors", (), {"tolist": lambda self: [[1, 0], [0, 1]]})()

    class FakeCollection:
        def query(self, **kwargs):
            assert len(kwargs["query_embeddings"]) == 2
            return {
                "ids": [["a"], ["b"]],
                "documents": [["Due Friday"], ["Finish asset kit"]],
                "metadatas": [[{"path": "dates.md"}], [{"path": "project.md"}]],
                "distances": [[0.1], [0.2]],
            }

    monkeypatch.setattr(semantic, "_embedding_client", lambda **kwargs: FakeEmbedding())
    monkeypatch.setattr(semantic, "_collection", lambda: FakeCollection())
    batches = semantic.search_semantic_vault_index_many(
        ["deadlines", "active projects"],
        index_namespace="root",
    )
    assert batches[0][0]["path"] == "dates.md"
    assert batches[1][0]["path"] == "project.md"


def test_negative_constraint_enforcement_is_generic():
    from src.agent_runtime.knowledge_semantic_chat_bridge import (
        _enforce_negative_constraints,
        _extract_negative_constraints,
    )

    constraints = _extract_negative_constraints([{
        "path": "instructions.md",
        "preview": "Do not include private token in the report.",
    }])
    answer, report = _enforce_negative_constraints(
        "Recommended items:\n- public summary, private token\n- public appendix",
        constraints,
    )
    assert "private token" not in answer.lower().split("source constraints applied:")[0]
    assert "public summary" in answer.lower()
    assert "public appendix" in answer.lower()
    assert "do not include private token" in answer.lower()
    assert report["changed"] is True

    compliant, compliant_report = _enforce_negative_constraints(
        "The plan explicitly avoids using private token.",
        constraints,
    )
    assert compliant == "The plan explicitly avoids using private token."
    assert compliant_report["changed"] is False

    markdown_compliant, markdown_report = _enforce_negative_constraints(
        "You do **not** need **private token** for this report.",
        constraints,
    )
    assert markdown_compliant == (
        "You do **not** need **private token** for this report."
    )
    assert markdown_report["changed"] is False

    formatted_conflict, formatted_report = _enforce_negative_constraints(
        "Shopping list:\n- **private token** (listed as need to buy)\n- public summary",
        constraints,
    )
    assert "****" not in formatted_conflict
    assert "listed as need to buy" not in formatted_conflict
    assert "- public summary" in formatted_conflict
    assert formatted_report["changed"] is True


def test_certification_normalizes_equivalent_dates_numbers_and_contractions():
    from src.agent_runtime.certification_text import (
        positive_forbidden_hit,
        semantic_contains,
    )

    answer = "The deadline is July 28, 2026. Total: ₹3,339. I couldn't find more."
    assert semantic_contains(answer, "28 july")
    assert semantic_contains(answer, "3339")
    assert semantic_contains(answer, "cannot")
    assert positive_forbidden_hit("Do not use sesame.", "sesame") is False
    assert positive_forbidden_hit("Add sesame.", "sesame") is True


def test_large_result_reranking_is_bounded_and_path_diverse():
    from src.agent_runtime.knowledge_semantic_chat_bridge import (
        _rerank_and_diversify_results,
    )

    results = [
        {
            "path": f"notes/note-{index}.md",
            "title": f"Note {index}",
            "heading": "Actions",
            "preview": "Priority and next action",
            "similarity": 1 - (index / 2000),
        }
        for index in range(1500)
    ]
    selected = _rerank_and_diversify_results(
        results,
        "What should I focus on next?",
        max_results=10,
    )
    assert len(selected) == 10
    assert len({item["path"] for item in selected}) == 10
