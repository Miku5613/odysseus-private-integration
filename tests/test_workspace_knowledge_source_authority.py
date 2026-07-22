from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.agent_runtime.supervisor_contracts import WorkerResult


class _ReadDecision:
    intent = "knowledge_read"
    confidence = "high"
    needs_human = False

    def to_dict(self):
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "needs_human": self.needs_human,
        }


def test_semantic_bridge_passes_explicit_workspace_to_worker(monkeypatch):
    import src.agent_runtime.knowledge_semantic_chat_bridge as bridge

    captured = {}
    monkeypatch.setattr(bridge, "evaluate_knowledge_request", lambda message: _ReadDecision())
    monkeypatch.setattr(bridge, "generate_intent_preface", lambda **kwargs: {})
    monkeypatch.setattr(
        bridge,
        "run_knowledge_worker",
        lambda handoff: (
            captured.update(context=handoff.context)
            or WorkerResult(
                capability="knowledge_search",
                status="completed",
                summary="Workspace match found.",
                data={
                    "root_id": "workspace_daily_notes",
                    "results": [{
                        "path": "dentist.md",
                        "title": "Dentist",
                        "heading": "Appointment",
                        "similarity": 0.95,
                        "preview": "Friday at 10 AM; bring insurance card.",
                    }],
                },
            )
        ),
    )
    monkeypatch.setattr(
        bridge,
        "get_semantic_vault_stats",
        lambda **kwargs: {"collection_name": "knowledge", "namespace_count": 1},
    )
    monkeypatch.setattr(
        bridge,
        "search_semantic_vault_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("global vault search must not run")
        ),
    )

    answer, _frames, _events, metrics = bridge.run_knowledge_semantic_chat_turn(
        "When was that dentist appointment again?",
        workspace="C:/Selected/Workspace",
    )

    assert captured["context"]["workspace"] == "C:/Selected/Workspace"
    assert "dentist.md" in answer
    assert metrics["source_authority"] == "workspace"
    assert metrics["root_id"] == "workspace_daily_notes"


def test_unindexed_workspace_bridge_fails_closed(monkeypatch):
    import src.agent_runtime.knowledge_semantic_chat_bridge as bridge

    monkeypatch.setattr(bridge, "evaluate_knowledge_request", lambda message: _ReadDecision())
    monkeypatch.setattr(bridge, "generate_intent_preface", lambda **kwargs: {})
    monkeypatch.setattr(
        bridge,
        "run_knowledge_worker",
        lambda handoff: WorkerResult(
            capability="knowledge_search",
            status="partial",
            summary="The active workspace is not indexed yet.",
            data={"root_id": None, "index_required": True},
        ),
    )
    monkeypatch.setattr(
        bridge,
        "get_semantic_vault_stats",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("global vault stats must not run")
        ),
    )
    monkeypatch.setattr(
        bridge,
        "search_semantic_vault_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("global vault search must not run")
        ),
    )

    answer, _frames, _events, _metrics = bridge.run_knowledge_semantic_chat_turn(
        "What are my Kyoto priorities?",
        workspace="C:/Unindexed/Workspace",
    )

    assert "not indexed yet" in answer
    assert "without searching Ideaverse" in answer


def test_chat_route_workspace_knowledge_path_has_no_global_fallback():
    source = Path("routes/chat_routes.py").read_text(encoding="utf-8")

    assert "if workspace or is_semantic_vault_search_request(message):" in source
    assert "workspace=workspace or None" in source
    assert "workspace bridge failed; stopping without a global fallback" in source
    assert "if workspace_rejected:" in source
    rejected_block = source.split("if workspace_rejected:", 1)[1].split(
        "if ctx.preprocessed.attachment_meta:", 1
    )[0]
    assert 'yield "data: [DONE]\\n\\n"' in rejected_block
    assert "return" in rejected_block


def test_workspace_knowledge_path_has_no_mojibake():
    suspicious = ("Ãƒ", "Ã‚", "Ã¢â‚¬", "Ã¢â€", "ï¿½", "\x00")
    for relative in (
        "routes/chat_routes.py",
        "routes/workspace_routes.py",
        "src/agent_runtime/knowledge_semantic_chat_bridge.py",
        "src/agent_runtime/workspace_index_manager.py",
    ):
        source = Path(relative).read_text(encoding="utf-8")
        assert not [token for token in suspicious if token in source], relative
