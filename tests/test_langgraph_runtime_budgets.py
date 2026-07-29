from __future__ import annotations

from src.agent_runtime.durable_checkpoint import DurableRunCheckpointStore
from src.agent_runtime.run_lifecycle import LangGraphRunLifecycle


def test_large_dependency_is_offloaded_without_evidence_loss(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "budget.sqlite3")
    lifecycle = LangGraphRunLifecycle.create(
        session_id="budget-session",
        run_id="budget-run",
        owner="alice",
        checkpoint_store=store,
    )
    large_report = "research-evidence-" * 10_000
    lifecycle.update_runtime_state(
        completed_steps=["research"],
        dependency_results={"research": {"report": large_report}},
    )

    recovered = store.recover_latest(run_id="budget-run")
    reference = recovered["dependency_results"]["research"]
    assert reference["offloaded"] is True
    assert reference["bytes"] > 64_000
    artifact = store.artifact_store.root / f"{reference['sha256']}.json"
    assert artifact.exists()
    assert large_report in artifact.read_text(encoding="utf-8")
    assert reference in recovered["artifact_references"]
    store.close()


def test_large_event_is_replayed_as_an_artifact_reference(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "event-budget.sqlite3")
    lifecycle = LangGraphRunLifecycle.create(
        session_id="event-session",
        run_id="event-run",
        checkpoint_store=store,
    )
    lifecycle.event({
        "type": "tool_result",
        "data": {"output": "x" * 100_000},
    })
    events = store.events_after(run_id="event-run")
    assert len(events) == 1
    assert events[0]["data"]["payload_offloaded"] is True
    reference = events[0]["data"]["artifact"]
    assert (store.artifact_store.root / f"{reference['sha256']}.json").exists()
    store.close()
