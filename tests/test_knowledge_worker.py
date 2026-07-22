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
        context={
            "query": "Atlas write mode decisions",
            "root_id": "example",
        },
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
        context={"root_id": "example"},
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
        context={"root_id": "example"},
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
        context={"root_id": "example"},
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
        context={"root_id": "example"},
    )

    result = run_knowledge_worker(
        handoff,
        retriever=broken_retriever,
    )

    assert result.status == "failed"
    assert result.error == "Chroma unavailable"


def test_knowledge_worker_does_not_use_an_implicit_global_root():
    called = False

    def fake_retriever(*args, **kwargs):
        nonlocal called
        called = True
        return []

    result = run_knowledge_worker(
        WorkerHandoff(
            capability="knowledge_search",
            objective="Search my notes.",
        ),
        retriever=fake_retriever,
    )

    assert result.status == "partial"
    assert result.data["root_id"] is None
    assert "fallback is disabled" in result.summary
    assert called is False


def test_workspace_source_precedes_an_explicit_root_id(monkeypatch):
    import src.agent_runtime.knowledge_worker as worker

    captured = {}
    monkeypatch.setattr(
        worker,
        "find_root_by_path",
        lambda path: {"root_id": "workspace_root"},
    )

    def fake_workspace_search(root_id, query, k=8):
        captured.update(root_id=root_id, query=query, k=k)
        return [{"path": "dentist.md", "title": "Dentist"}]

    monkeypatch.setattr(worker, "search_root", fake_workspace_search)
    result = worker.run_knowledge_worker(
        WorkerHandoff(
            capability="knowledge_search",
            objective="When is the dentist appointment?",
            context={
                "workspace": "C:/Selected/Workspace",
                "root_id": "different_global_root",
            },
        )
    )

    assert result.status == "completed"
    assert result.data["root_id"] == "workspace_root"
    assert captured["root_id"] == "workspace_root"


def test_unindexed_workspace_fails_without_calling_global_retriever(monkeypatch):
    import src.agent_runtime.knowledge_worker as worker

    called = False
    monkeypatch.setattr(worker, "find_root_by_path", lambda path: None)

    def forbidden_global_search(*args, **kwargs):
        nonlocal called
        called = True
        return []

    result = worker.run_knowledge_worker(
        WorkerHandoff(
            capability="knowledge_search",
            objective="What are my Kyoto priorities?",
            context={"workspace": "C:/Selected/Workspace"},
        ),
        retriever=forbidden_global_search,
    )

    assert result.status == "partial"
    assert result.data["index_required"] is True
    assert called is False
