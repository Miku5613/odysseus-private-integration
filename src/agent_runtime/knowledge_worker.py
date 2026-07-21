from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.agent_runtime.knowledge_vault_semantic import search_semantic_vault_index
from src.agent_runtime.supervisor_contracts import WorkerHandoff, WorkerResult


KNOWLEDGE_WORKER_CAPABILITY = "knowledge_search"

SemanticRetriever = Callable[..., list[dict[str, Any]]]


def _result_evidence(results: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    evidence: list[dict[str, Any]] = []

    for result in results:
        evidence.append(
            {
                "kind": "knowledge_retrieval",
                "path": result.get("path"),
                "title": result.get("title"),
                "heading": result.get("heading"),
                "similarity": result.get("similarity"),
                "preview": result.get("preview"),
                "metadata": result.get("metadata", {}),
            }
        )

    return tuple(evidence)


def run_knowledge_worker(
    handoff: WorkerHandoff,
    *,
    session_id: str | None = None,
    retriever: SemanticRetriever = search_semantic_vault_index,
) -> WorkerResult:
    """Run semantic knowledge retrieval behind the common worker boundary."""

    if handoff.capability != KNOWLEDGE_WORKER_CAPABILITY:
        raise ValueError(
            "Knowledge Worker requires capability=knowledge_search"
        )

    objective = str(handoff.objective or "").strip()

    if not objective:
        raise ValueError("Knowledge Worker requires a non-empty objective")

    query = str(handoff.context.get("query") or objective).strip()

    if not query:
        return WorkerResult(
            capability=KNOWLEDGE_WORKER_CAPABILITY,
            status="needs_clarification",
            summary="A knowledge query is required.",
            missing_information=("knowledge query",),
        )

    try:
        results = retriever(query, k=8)
    except Exception as exc:
        return WorkerResult(
            capability=KNOWLEDGE_WORKER_CAPABILITY,
            status="failed",
            summary="Semantic knowledge retrieval failed.",
            data={
                "query": query,
                "session_id": session_id,
            },
            error=str(exc),
        )

    evidence = _result_evidence(results)

    if not results:
        return WorkerResult(
            capability=KNOWLEDGE_WORKER_CAPABILITY,
            status="partial",
            summary="No matching indexed knowledge was found.",
            data={
                "query": query,
                "result_count": 0,
                "session_id": session_id,
            },
        )

    return WorkerResult(
        capability=KNOWLEDGE_WORKER_CAPABILITY,
        status="completed",
        summary=f"Retrieved {len(results)} relevant knowledge result(s).",
        data={
            "query": query,
            "result_count": len(results),
            "results": results,
            "session_id": session_id,
        },
        evidence=evidence,
    )


__all__ = [
    "KNOWLEDGE_WORKER_CAPABILITY",
    "run_knowledge_worker",
]
