from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any

from src.agent_runtime.knowledge_vault_semantic import search_semantic_vault_index
from src.agent_runtime.workspace_index_manager import find_root_by_path, search_root
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

    workspace = str(
        handoff.context.get("workspace") or ""
    ).strip()

    explicit_root_id = str(
        handoff.context.get("root_id") or ""
    ).strip()

    # Source authority invariant:
    # 1. An explicit validated workspace is authoritative.
    # 2. An explicit root_id is a separate authorized source selector.
    # 3. There is no implicit global/default root.
    #
    # A workspace path is not automatically interchangeable with a
    # Chroma root_id. Until that workspace has a scoped index, the
    # worker must fail honestly rather than silently query another root.
    if workspace:
        root = find_root_by_path(workspace)

        if root is None:
            return WorkerResult(
                capability=KNOWLEDGE_WORKER_CAPABILITY,
                status="partial",
                summary=(
                    "The active workspace is not indexed yet."
                ),
                data={
                    "query": query,
                    "workspace": workspace,
                    "root_id": None,
                    "result_count": 0,
                    "session_id": session_id,
                    "index_required": True,
                },
                missing_information=(
                    "workspace-scoped index",
                ),
            )

        root_id = str(root.get("root_id") or "").strip()

        try:
            results = search_root(
                root_id,
                query,
                k=8,
            )
        except Exception as exc:
            return WorkerResult(
                capability=KNOWLEDGE_WORKER_CAPABILITY,
                status="failed",
                summary="Workspace knowledge retrieval failed.",
                data={
                    "query": query,
                    "workspace": workspace,
                    "root_id": root_id,
                    "session_id": session_id,
                },
                error=str(exc),
            )

        evidence = _result_evidence(results)

        if not results:
            return WorkerResult(
                capability=KNOWLEDGE_WORKER_CAPABILITY,
                status="partial",
                summary=(
                    "No matching information was found in the active workspace."
                ),
                data={
                    "query": query,
                    "workspace": workspace,
                    "root_id": root_id,
                    "result_count": 0,
                    "session_id": session_id,
                },
            )

        return WorkerResult(
            capability=KNOWLEDGE_WORKER_CAPABILITY,
            status="completed",
            summary=(
                f"Retrieved {len(results)} relevant workspace result(s)."
            ),
            data={
                "query": query,
                "workspace": workspace,
                "root_id": root_id,
                "result_count": len(results),
                "results": results,
                "session_id": session_id,
            },
            evidence=evidence,
        )

    if not explicit_root_id:
        return WorkerResult(
            capability=KNOWLEDGE_WORKER_CAPABILITY,
            status="partial",
            summary=(
                "No explicit knowledge root is active. "
                "Global knowledge fallback is disabled."
            ),
            data={
                "query": query,
                "workspace": None,
                "root_id": None,
                "result_count": 0,
                "session_id": session_id,
            },
        )

    root_id = explicit_root_id

    try:
        parameters = inspect.signature(retriever).parameters
        supports_namespace = (
            "index_namespace" in parameters
            or any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in parameters.values()
            )
        )

        if supports_namespace:
            results = retriever(
                query,
                k=8,
                index_namespace=root_id,
            )
        else:
            results = retriever(query, k=8)
    except Exception as exc:
        return WorkerResult(
            capability=KNOWLEDGE_WORKER_CAPABILITY,
            status="failed",
            summary="Semantic knowledge retrieval failed.",
            data={
                "query": query,
                "root_id": root_id,
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
                "root_id": root_id,
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
            "root_id": root_id,
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
