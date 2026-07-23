from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any

from src.agent_runtime.knowledge_vault_semantic import search_semantic_vault_index
from src.agent_runtime.workspace_index_manager import (
    find_root_by_path,
    search_root,
    search_root_queries,
)
from src.agent_runtime.supervisor_contracts import WorkerHandoff, WorkerResult


KNOWLEDGE_WORKER_CAPABILITY = "knowledge_search"

SemanticRetriever = Callable[..., list[dict[str, Any]]]


def _retrieval_queries(
    primary_query: str,
    context: dict[str, Any],
) -> list[str]:
    """Return a bounded, de-duplicated set of runtime-provided queries."""

    out = [primary_query]
    expansions = context.get("retrieval_queries")
    if isinstance(expansions, (list, tuple)):
        for value in expansions:
            clean = str(value or "").strip()
            if clean and clean.casefold() not in {item.casefold() for item in out}:
                out.append(clean)
            if len(out) >= 6:
                break
    return out


def _merge_query_results(
    batches: list[tuple[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Merge multi-query evidence while preserving its strongest score."""

    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []

    for retrieval_query, results in batches:
        for raw in results:
            item = dict(raw)
            key = (
                str(item.get("path") or item.get("id") or ""),
                str(item.get("heading") or ""),
                str(item.get("title") or ""),
            )
            if not any(key):
                continue

            current = merged.get(key)
            current_score = float((current or {}).get("similarity") or -1.0)
            candidate_score = float(item.get("similarity") or -1.0)
            matched_queries = list((current or {}).get("matched_queries") or [])
            if retrieval_query not in matched_queries:
                matched_queries.append(retrieval_query)

            if current is None or candidate_score > current_score:
                item["retrieval_query"] = retrieval_query
                item["matched_queries"] = matched_queries
                merged[key] = item
                if current is None:
                    order.append(key)
            else:
                current["matched_queries"] = matched_queries

    return sorted(
        (merged[key] for key in order),
        key=lambda item: float(item.get("similarity") or -1.0),
        reverse=True,
    )


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
    retrieval_queries = _retrieval_queries(query, dict(handoff.context))

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
            if len(retrieval_queries) > 1:
                query_batches = search_root_queries(
                    root_id,
                    retrieval_queries,
                    k=8,
                )
            else:
                query_batches = [
                    search_root(root_id, retrieval_queries[0], k=8)
                ]
            results = _merge_query_results(list(zip(
                retrieval_queries,
                query_batches,
            )))
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
                "retrieval_queries": retrieval_queries,
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
            results = _merge_query_results([
                (
                    retrieval_query,
                    retriever(
                        retrieval_query,
                        k=8,
                        index_namespace=root_id,
                    ),
                )
                for retrieval_query in retrieval_queries
            ])
        else:
            results = _merge_query_results([
                (retrieval_query, retriever(retrieval_query, k=8))
                for retrieval_query in retrieval_queries
            ])
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
            "retrieval_queries": retrieval_queries,
            "session_id": session_id,
        },
        evidence=evidence,
    )


__all__ = [
    "KNOWLEDGE_WORKER_CAPABILITY",
    "run_knowledge_worker",
]
