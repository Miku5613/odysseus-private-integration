import pytest

from src.agent_runtime.knowledge_worker import run_knowledge_worker
from src.agent_runtime.supervisor_contracts import WorkerHandoff


def test_knowledge_worker_rejects_wrong_capability():
    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Search project knowledge.",
    )

    with pytest.raises(ValueError, match="knowledge_search"):
        run_knowledge_worker(handoff, retriever=lambda *args, **kwargs: [])


def test_knowledge_worker_uses_context_query_when_present():
    captured = {}

    def fake_retriever(query, k=8):
        captured["query"] = query
        captured["k"] = k
        return []

    handoff = WorkerHandoff(
        capability="knowledge_search",
        objective="Find relevant project knowledge.",
        context={"query": "Atlas write mode decisions"},
    )

    run_knowledge_worker(handoff, retriever=fake_retriever)

    assert captured["query"] == "Atlas write mode decisions"
    assert captured["k"] == 8


def test_knowledge_worker_falls_back_to_objective_as_query():
    captured = {}

    def fake_retriever(query, k=8):
        captured["query"] = query
        return []

    handoff = WorkerHandoff(
        capability="knowledge_search",
        objective="What did I decide about Atlas write mode?",
    )

    run_knowledge_worker(handoff, retriever=fake_retriever)

    assert captured["query"] == "What did I decide about Atlas write mode?"


def test_knowledge_worker_returns_semantic_results_and_evidence():
    def fake_retriever(query, k=8):
        return [
            {
                "path": "Atlas/Odysseus.md",
                "title": "Odysseus",
                "heading": "Write Mode",
                "similarity": 0.91,
                "preview": "Write mode creates a proposal copy.",
                "metadata": {"root_id": "example"},
            }
        ]

    handoff = WorkerHandoff(
        capability="knowledge_search",
        objective="Find Atlas write mode decisions.",
    )

    result = run_knowledge_worker(
        handoff,
        session_id="session-1",
        retriever=fake_retriever,
    )

    assert result.status == "completed"
    assert result.data["result_count"] == 1
    assert result.data["session_id"] == "session-1"
    assert result.evidence[0]["kind"] == "knowledge_retrieval"
    assert result.evidence[0]["path"] == "Atlas/Odysseus.md"


def test_knowledge_worker_returns_partial_for_no_results():
    handoff = WorkerHandoff(
        capability="knowledge_search",
        objective="Find something absent.",
    )

    result = run_knowledge_worker(
        handoff,
        retriever=lambda *args, **kwargs: [],
    )

    assert result.status == "partial"
    assert result.data["result_count"] == 0


def test_knowledge_worker_converts_backend_failure_to_worker_result():
    def broken_retriever(*args, **kwargs):
        raise RuntimeError("Chroma unavailable")

    handoff = WorkerHandoff(
        capability="knowledge_search",
        objective="Search project knowledge.",
    )

    result = run_knowledge_worker(
        handoff,
        retriever=broken_retriever,
    )

    assert result.status == "failed"
    assert result.error == "Chroma unavailable"
