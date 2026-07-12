from __future__ import annotations

from src.agent_runtime.knowledge_chat_bridge import run_knowledge_chat_turn
from src.agent_runtime.evidence_tracker import (
    TRACKER_VERSION,
    compact_tracker_summary,
    new_evidence_tracker,
    record_fallback,
    record_final,
    record_human_gate,
    record_observations,
    record_route,
    record_sufficiency,
)


def test_evidence_tracker_records_source_backed_answer():
    tracker = new_evidence_tracker(
        user_request="Find where message_saved is emitted.",
        runtime="deterministic_contract",
        run_id="test_run",
    )

    record_route(
        tracker,
        intent="exact_lookup",
        target_runtime="contract",
        reason="Precise event lookup.",
        confidence="high",
        safe_readonly=True,
    )

    record_observations(
        tracker,
        [
            {
                "observation_id": "obs_1",
                "kind": "context_window",
                "source_path": "routes/chat_routes.py",
                "line_start": 120,
                "line_end": 180,
                "content": "yield data: message_saved",
                "source_tool": "read_file_around",
                "source_authority": "source_code",
                "confidence": 42,
            }
        ],
    )

    record_sufficiency(
        tracker,
        status="sufficient",
        reason="Source-code evidence satisfied the query.",
    )
    record_human_gate(tracker, required=False)
    record_fallback(tracker, used=False)
    record_final(
        tracker,
        answer_mode="source_backed_answer",
        source_basis="source_code_observation",
        final_answer="message_saved is emitted in routes/chat_routes.py.",
    )

    assert tracker["version"] == TRACKER_VERSION
    assert tracker["route"]["target_runtime"] == "contract"
    assert tracker["evidence"]["strength"] == "strong"
    assert tracker["evidence"]["source_paths"] == ["routes/chat_routes.py"]
    assert tracker["sufficiency"]["status"] == "sufficient"
    assert tracker["human_gate"]["required"] is False
    assert tracker["fallback"]["used"] is False

    summary = compact_tracker_summary(tracker)

    assert summary["intent"] == "exact_lookup"
    assert summary["evidence_strength"] == "strong"
    assert summary["sufficiency"] == "sufficient"
    assert summary["fallback_used"] is False


def test_evidence_tracker_records_human_gate_and_fallback():
    tracker = new_evidence_tracker(
        user_request="Read the doc.",
        runtime="knowledge_chat_bridge",
        run_id="test_human",
    )

    record_route(
        tracker,
        intent="needs_clarification",
        target_runtime="human_in_the_loop",
        reason="The referenced document is ambiguous.",
        confidence="medium",
        safe_readonly=True,
        needs_human=True,
    )

    record_observations(tracker, [])
    record_sufficiency(
        tracker,
        status="insufficient",
        reason="No concrete target document was identified.",
    )
    record_human_gate(
        tracker,
        required=True,
        reason="Ambiguous document target.",
        ambiguity=["which document"],
    )
    record_fallback(
        tracker,
        used=True,
        target_runtime="human_in_the_loop",
        reason="Clarification required before graph execution.",
    )
    record_final(
        tracker,
        answer_mode="clarification",
        source_basis="router_decision",
        final_answer="I need a bit more context.",
    )

    assert tracker["human_gate"]["required"] is True
    assert tracker["fallback"]["used"] is True
    assert tracker["fallback"]["target_runtime"] == "human_in_the_loop"
    assert tracker["sufficiency"]["status"] == "insufficient"

    summary = compact_tracker_summary(tracker)

    assert summary["needs_human"] is True
    assert summary["fallback_used"] is True
    assert summary["answer_mode"] == "clarification"


def test_evidence_tracker_handles_weak_empty_and_path_only_evidence():
    tracker = new_evidence_tracker(
        user_request="Find related files.",
        runtime="general",
        run_id="test_paths",
    )

    record_observations(
        tracker,
        [
            {"kind": "file_candidate", "path": "src/example.py", "confidence": 0},
            {"kind": "note", "content": "A weak observation with no path.", "confidence": 0},
        ],
    )

    assert tracker["evidence"]["strength"] == "partial"
    assert tracker["evidence"]["source_paths"] == ["src/example.py"]
    assert tracker["evidence"]["source_count"] == 1
    assert tracker["evidence"]["items"][0]["source_path"] == "src/example.py"


def test_knowledge_chat_bridge_metrics_include_evidence_tracker_for_clarification():
    final, frames, events, metrics = run_knowledge_chat_turn("Read the doc.")

    tracker = metrics.get("evidence_tracker")
    summary = metrics.get("evidence_tracker_summary")

    assert "I need a bit more context" in final
    assert tracker
    assert summary

    assert tracker["runtime"] == "knowledge_chat_bridge"
    assert tracker["route"]["intent"] == "needs_clarification"
    assert tracker["human_gate"]["required"] is True
    assert tracker["fallback"]["used"] is True
    assert tracker["final"]["answer_mode"] == "clarification"

    assert summary["intent"] == "needs_clarification"
    assert summary["needs_human"] is True
    assert summary["fallback_used"] is True

    assert any(ev.get("tool") == "knowledge_decision" for ev in events)
    assert not any(frame.strip() == "data: [DONE]" for frame in frames)
