import pytest

from src.agent_runtime.code_worker import run_code_worker
from src.agent_runtime.supervisor_contracts import WorkerHandoff


def test_code_worker_rejects_wrong_capability():
    handoff = WorkerHandoff(
        capability="knowledge_search",
        objective="Inspect this code.",
    )

    with pytest.raises(ValueError, match="code_inspection"):
        run_code_worker(handoff, stream_factory=lambda *args, **kwargs: [])


def test_code_worker_requires_objective():
    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="   ",
    )

    with pytest.raises(ValueError, match="non-empty objective"):
        run_code_worker(handoff, stream_factory=lambda *args, **kwargs: [])


def test_code_worker_passes_workspace_and_session_to_existing_runtime():
    captured = {}

    def fake_stream(question, workspace=None, session_id=None):
        captured["question"] = question
        captured["workspace"] = workspace
        captured["session_id"] = session_id
        yield {
            "type": "stream_done",
            "final_answer": "Done.",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Find the implementation.",
        context={"workspace": "D:/Projects/example"},
    )

    result = run_code_worker(
        handoff,
        session_id="session-1",
        stream_factory=fake_stream,
    )

    assert captured == {
        "question": "Find the implementation.",
        "workspace": "D:/Projects/example",
        "session_id": "session-1",
    }
    assert result.status == "completed"


def test_code_worker_collects_specialist_evidence():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "evidence_hit",
            "kind": "code_location",
            "file": "src/example.py",
            "line": 42,
        }
        yield {
            "type": "final_answer",
            "content": "Implementation found.",
        }
        yield {
            "type": "stream_done",
            "final_answer": "Implementation found.",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Find the implementation.",
    )

    result = run_code_worker(handoff, stream_factory=fake_stream)

    assert result.status == "completed"
    assert result.summary == "Implementation found."
    assert len(result.evidence) == 1
    assert result.evidence[0]["file"] == "src/example.py"
    assert result.evidence[0]["line"] == 42


def test_code_worker_reports_stream_error_as_failed_result():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_error",
            "error": "runtime exploded",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Inspect the project.",
    )

    result = run_code_worker(handoff, stream_factory=fake_stream)

    assert result.status == "failed"
    assert result.error == "runtime exploded"


def test_code_worker_partial_when_no_final_answer_is_emitted():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_start",
            "scope": "file_inspection",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Inspect the project.",
    )

    result = run_code_worker(handoff, stream_factory=fake_stream)

    assert result.status == "partial"


def test_code_worker_declares_active_doc_support_without_invoking_it():
    def fake_stream(*args, **kwargs):
        yield {
            "type": "stream_done",
            "final_answer": "Prepared code guidance.",
        }

    handoff = WorkerHandoff(
        capability="code_inspection",
        objective="Prepare a coding fix.",
    )

    result = run_code_worker(handoff, stream_factory=fake_stream)

    assert result.data["supports_active_doc"] is True
