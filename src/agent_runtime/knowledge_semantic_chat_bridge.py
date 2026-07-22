
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from .knowledge_graph_router import (
    build_human_clarification_prompt,
    evaluate_knowledge_request,
)
from .runtime_router import (
    is_casual_conversation_request,
    route_agent_runtime,
)
from .knowledge_vault_semantic import (
    get_semantic_vault_stats,
    search_semantic_vault_index,
)
from .daily_driver_preface import generate_intent_preface
from .knowledge_worker import run_knowledge_worker
from .knowledge_llm_response_bridge import try_knowledge_llm_response
from .supervisor_contracts import WorkerHandoff


def sse_frame(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def is_semantic_vault_search_request(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False

    # Do not steal exact/active/path reads from the existing read graph.
    exact_read_markers = [
        "read the active note",
        "active note",
        "list vault folders",
        "list my vault folders",
        "vault folders",
        "read note ",
        "read file ",
        "read path ",
    ]
    if any(marker in text for marker in exact_read_markers):
        return False

    semantic_markers = [
        "search my notes",
        "search the notes",
        "search notes",
        "search my vault",
        "find my notes",
        "find notes",
        "notes about",
        "note about",
        "what did i write about",
        "what have i written about",
        "what do my notes say about",
        "in my notes",
        "from my notes",
        "within my notes",
        "based on my notes",
        "ideaverse about",
        "knowledge about",
    ]

    return any(marker in text for marker in semantic_markers)


def should_intercept_workspace_knowledge(message: str) -> bool:
    """Treat an active workspace as authoritative context for safe questions.

    The workspace is request-scoped context, so users should not have to repeat
    "in my notes" on every follow-up. Clear code inspection, current-world
    research, and short social turns remain with their dedicated runtimes.
    """

    text = str(message or "").strip()
    if not text:
        return False

    decision = evaluate_knowledge_request(text)
    if decision.intent != "not_knowledge":
        return True

    if is_casual_conversation_request(text):
        return False

    runtime = route_agent_runtime(text)
    if runtime.target_runtime in {"contract", "general"}:
        return False

    low = text.lower()
    if re.search(
        r"\b(?:research|latest|news|web|online|browser|google|search online|look up online)\b",
        low,
    ):
        return False

    if re.search(
        r"\b(?:backend|frontend|source code|codebase|implementation|route|function|class|module|callsite|handler)\b|"
        r"(?:^|\s)(?:src|routes|static/js)/|\.(?:py|js|ts|tsx|jsx)\b",
        low,
    ):
        return False

    # A question or contextual read request inside an active workspace should
    # consult that workspace before claiming the information is unavailable.
    return bool(
        "?" in text
        or re.search(
            r"^\s*(?:what|when|which|where|who|how|should|do|did|was|were|is|are|can|could|would|remind|summarize|list|help|give|tell)\b",
            low,
        )
    )


def extract_semantic_query(message: str) -> str:
    raw = str(message or "").strip()
    text = raw

    patterns = [
        r"(?i)^\s*search\s+(?:my\s+)?(?:vault|notes)\s+(?:for|about)\s+",
        r"(?i)^\s*find\s+(?:my\s+)?notes\s+(?:for|about|on|related to)\s+",
        r"(?i)^\s*what\s+(?:did|have)\s+i\s+(?:write|written)\s+about\s+",
        r"(?i)^\s*what\s+do\s+my\s+notes\s+say\s+about\s+",
        r"(?i)\s+in\s+my\s+notes\s*$",
        r"(?i)\s+from\s+my\s+notes\s*$",
        r"(?i)\s+based\s+on\s+my\s+notes\s*$",
    ]

    for pat in patterns:
        text = re.sub(pat, "", text).strip()

    text = text.strip(" .?!\"'")
    return text or raw



def _query_terms(query: str) -> list[str]:
    text = str(query or "").lower()
    terms = re.findall(r"[a-zA-Z0-9_]+", text)
    stop = {
        "what", "did", "have", "written", "write", "about", "the", "and", "for",
        "with", "from", "notes", "note", "my", "your", "in", "on", "of", "to",
    }
    out: list[str] = []
    for t in terms:
        if len(t) >= 3 and t not in stop and t not in out:
            out.append(t)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _semantic_rerank_score(result: dict[str, Any], query: str) -> float:
    base = _safe_float(result.get("similarity"), 0.0)
    terms = _query_terms(query)

    title = str(result.get("title") or "").lower()
    path = str(result.get("path") or "").lower()
    heading = str(result.get("heading") or "").lower()
    preview = str(result.get("preview") or "").lower()

    title_path = f"{title} {path}".replace("_", " ").replace("-", " ")
    all_blob = f"{title_path} {heading} {preview}".replace("_", " ").replace("-", " ")

    score = base

    # Strongly prefer exact title/path intent matches.
    if terms and all(t in title_path for t in terms):
        score += 0.22
    if terms and all(t in title for t in terms):
        score += 0.16
    if terms and all(t in path for t in terms):
        score += 0.14

    # Heading/preview are useful, but weaker than note identity.
    if terms and all(t in heading for t in terms):
        score += 0.08
    if terms and all(t in all_blob for t in terms):
        score += 0.05

    # Troubleshooting-type queries should promote troubleshooting docs over generic setup docs.
    if "troubleshooting" in terms or "troubleshoot" in terms:
        if "troubleshooting" in title_path or "troubleshoot" in title_path:
            score += 0.20
        elif "troubleshooting" in heading or "troubleshoot" in heading:
            score += 0.08
        else:
            score -= 0.06

    # Prefer AIOS/WIP agent guidance slightly over old generic Atlas pages when both match.
    if "aios" in path and ("troubleshooting" in title_path or "troubleshoot" in title_path):
        score += 0.06

    return round(score, 4)


def _rerank_and_diversify_results(
    results: list[dict[str, Any]],
    query: str,
    *,
    max_results: int = 6,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for r in results:
        item = dict(r)
        item["rerank_score"] = _semantic_rerank_score(item, query)
        enriched.append(item)

    enriched.sort(
        key=lambda r: (
            _safe_float(r.get("rerank_score")),
            _safe_float(r.get("similarity")),
        ),
        reverse=True,
    )

    # First pass: avoid one generic note taking 4 slots.
    picked: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for r in enriched:
        path = str(r.get("path") or "")
        if path and path in seen_paths:
            continue
        picked.append(r)
        if path:
            seen_paths.add(path)
        if len(picked) >= max_results:
            return picked

    # Second pass: fill remaining slots with additional chunks.
    for r in enriched:
        if r in picked:
            continue
        picked.append(r)
        if len(picked) >= max_results:
            break

    return picked


def _format_search_output(results: list[dict[str, Any]], query: str) -> str:
    lines = []
    lines.append(f"Semantic vault search query: {query}")
    lines.append(f"Results: {len(results)}")

    for i, r in enumerate(results, start=1):
        lines.append("")
        lines.append(f"{i}. {r.get('title') or '(untitled)'}")
        lines.append(f"   path: {r.get('path')}")
        lines.append(f"   heading: {r.get('heading')}")
        lines.append(f"   similarity: {r.get('similarity')}")
        preview = str(r.get("preview") or "").replace("\n", " ")
        lines.append(f"   preview: {preview[:420]}")

    return "\n".join(lines).strip()


def _build_final_answer(
    results: list[dict[str, Any]],
    query: str,
    *,
    source_label: str = "Ideaverse",
) -> str:
    if not results:
        return (
            f"I searched the {source_label} semantic index for **{query}**, but I did not find matching note chunks.\n\n"
            "The search was read-only. No notes were changed."
        )

    lines = []
    lines.append(f"I searched the {source_label} notes for **{query}** and found these strongest matches:")
    lines.append("")

    for i, r in enumerate(results[:6], start=1):
        title = r.get("title") or "(untitled)"
        path = r.get("path") or "(unknown path)"
        heading = r.get("heading") or "(no heading)"
        similarity = r.get("similarity")
        rerank_score = r.get("rerank_score")
        preview = str(r.get("preview") or "").replace("\n", " ").strip()

        lines.append(f"{i}. **{title}**")
        lines.append(f"   Path: `{path}`")
        lines.append(f"   Heading: `{heading}`")
        if similarity is not None:
            lines.append(f"   Similarity: `{similarity}`")
        if rerank_score is not None:
            lines.append(f"   Rerank: `{rerank_score}`")
        if preview:
            lines.append(f"   Preview: {preview[:360]}")
        lines.append("")

    lines.append("This was a read-only semantic search over the local Chroma vault index. No Obsidian notes were edited.")

    return "\n".join(lines).strip()


def run_knowledge_semantic_chat_turn(
    message: str,
    *,
    workspace: str | None = None,
    root_id: str | None = None,
    llm_config: dict[str, Any] | None = None,
) -> tuple[str, list[str], list[dict[str, Any]], dict[str, Any]]:
    started = time.time()
    frames: list[str] = []
    tool_events: list[dict[str, Any]] = []

    decision = evaluate_knowledge_request(message)
    query = extract_semantic_query(message)
    explicit_workspace = str(workspace or "").strip()
    explicit_root_id = str(root_id or "").strip()
    source_label = "active workspace" if explicit_workspace else "Ideaverse"

    frames.append(sse_frame({
        "type": "agent_prep",
        "title": "Knowledge runtime",
        "content": f"Preparing a read-only semantic search of the {source_label}.",
    }))
    frames.append(sse_frame({
        "type": "agent_step",
        "title": "Routing knowledge request",
        "content": "Checking intent, confidence, and graph safety before semantic search.",
    }))

    runtime_status = {
        "intent": f"workspace_{getattr(decision, 'intent', 'knowledge')}",
        "runtime": "knowledge",
        "runtime_family": "workspace_semantic",
        "safe_readonly": bool(getattr(decision, "is_read_only", True)),
        "safety_profile": (
            "read_only"
            if getattr(decision, "is_read_only", True)
            else "policy_gated"
        ),
        "reason": (
            "The validated active workspace is authoritative for this "
            "knowledge request."
        ),
    }

    runtime_call_id = "runtime_decision_" + uuid.uuid4().hex[:10]
    runtime_output = json.dumps(runtime_status, indent=2, ensure_ascii=False)
    frames.append(sse_frame({
        "type": "tool_start",
        "id": runtime_call_id,
        "call_id": runtime_call_id,
        "tool": "runtime_decision",
        "command": "Runtime decision",
    }))
    frames.append(sse_frame({
        "type": "tool_output",
        "id": runtime_call_id,
        "call_id": runtime_call_id,
        "tool": "runtime_decision",
        "command": "Runtime decision",
        "output": runtime_output,
        "exit_code": 0,
        "runtime_decision": runtime_status,
    }))
    tool_events.append({
        "type": "tool_output",
        "tool": "runtime_decision",
        "command": "Runtime decision",
        "output": runtime_output,
        "exit_code": 0,
        "runtime_decision": runtime_status,
    })

    # DAILY_DRIVER_PREFACE_V016F1C_START
    if getattr(decision, "is_read_only", True):
        try:
            _preface = generate_intent_preface(
                user_request=message,
                intent="vault_search",
                resource=f"{source_label} notes",
                next_action="semantic_vault_search",
                safety="read-only",
                timeout_s=8.0,
            )
            _preface_text = str(_preface.get("text") or "").strip()
            if _preface_text:
                frames.append(sse_frame({"delta": _preface_text + "\n\n"}))
        except Exception:
            # Preface is UX-only. Never block semantic search.
            pass
    # DAILY_DRIVER_PREFACE_V016F1C_END

    decision_call_id = "knowledge_decision_" + uuid.uuid4().hex[:10]
    decision_output = json.dumps(decision.to_dict(), indent=2, ensure_ascii=False)

    frames.append(sse_frame({
        "type": "tool_start",
        "id": decision_call_id,
        "call_id": decision_call_id,
        "tool": "knowledge_decision",
        "command": "Knowledge graph decision",
    }))
    frames.append(sse_frame({
        "type": "tool_output",
        "id": decision_call_id,
        "call_id": decision_call_id,
        "tool": "knowledge_decision",
        "command": "Knowledge graph decision",
        "output": decision_output,
    }))
    tool_events.append({
        "type": "tool_output",
        "tool": "knowledge_decision",
        "command": "Knowledge graph decision",
        "output": decision_output,
    })

    if getattr(decision, "intent", "") in {"unsupported_mutation", "knowledge_propose", "knowledge_research"}:
        if decision.intent == "unsupported_mutation":
            final_answer = (
                "I cannot execute this destructive workspace request directly. "
                "The central policy path blocks delete, move, rename, purge, and "
                "cleanup operations from automatic execution.\n\n"
                "No files were changed. I can prepare a non-executing proposal; "
                "any later mutating action would require explicit confirmation."
            )
        elif decision.intent == "knowledge_propose":
            final_answer = (
                "I cannot apply this workspace write automatically. No files were "
                "changed. I can prepare a reviewable proposal or copy, and execution "
                "would require explicit confirmation through the policy gate."
            )
        else:
            final_answer = (
                "This request requires the isolated research capability rather "
                "than a workspace write. No notes were changed."
            )
        frames.append(sse_frame({"delta": final_answer}))
        metrics = {
            "runtime": "knowledge_semantic_chat",
            "knowledge_runtime": "semantic_vault_search",
            "knowledge_intent": getattr(decision, "intent", ""),
            "knowledge_confidence": getattr(decision, "confidence", ""),
            "needs_human": getattr(decision, "needs_human", False),
            "read_only": True,
            "write_capable": False,
            "elapsed_s": round(time.time() - started, 3),
        }
        frames.append(sse_frame({"type": "metrics", "metrics": metrics}))
        return final_answer, frames, tool_events, metrics

    # Respect HITL/blocks before any search.
    if getattr(decision, "needs_human", False) and getattr(decision, "intent", "") not in {"knowledge_read"}:
        final_answer = build_human_clarification_prompt(decision)
        frames.append(sse_frame({"delta": final_answer}))
        metrics = {
            "runtime": "knowledge_semantic_chat",
            "knowledge_runtime": "semantic_vault_search",
            "knowledge_intent": getattr(decision, "intent", ""),
            "knowledge_confidence": getattr(decision, "confidence", ""),
            "needs_human": True,
            "read_only": True,
            "write_capable": False,
            "elapsed_s": round(time.time() - started, 3),
        }
        frames.append(sse_frame({"type": "metrics", "metrics": metrics}))
        return final_answer, frames, tool_events, metrics

    search_call_id = "knowledge_semantic_search_" + uuid.uuid4().hex[:10]

    frames.append(sse_frame({
        "type": "agent_step",
        "title": "Searching semantic vault index",
        "content": f"Searching the {source_label} Chroma index for: {query}",
    }))
    frames.append(sse_frame({
        "type": "tool_start",
        "id": search_call_id,
        "call_id": search_call_id,
        "tool": "knowledge_semantic_search",
        "command": "Search semantic vault index",
    }))

    source_root_id = explicit_root_id or None
    worker_result = None
    if explicit_workspace or explicit_root_id:
        worker_result = run_knowledge_worker(
            WorkerHandoff(
                capability="knowledge_search",
                objective=message,
                context={
                    "query": query,
                    "workspace": explicit_workspace,
                    "root_id": explicit_root_id,
                },
            )
        )
        source_root_id = worker_result.data.get("root_id") or source_root_id
        raw_results = list(worker_result.data.get("results") or [])
    else:
        raw_results = search_semantic_vault_index(query, k=18)
    results = _rerank_and_diversify_results(raw_results, query, max_results=6)
    if source_root_id:
        stats = get_semantic_vault_stats(index_namespace=source_root_id)
    elif explicit_workspace or explicit_root_id:
        # An explicit but unresolved source must not even consult metadata for
        # the default/global namespace. Keep the failure fully fail-closed.
        stats = {}
    else:
        stats = get_semantic_vault_stats()
    output = _format_search_output(results, query)
    if worker_result is not None:
        output = (
            f"Source authority: active workspace\n"
            f"Workspace: {explicit_workspace}\n"
            f"Root ID: {source_root_id or '(not indexed)'}\n"
            f"Worker status: {worker_result.status}\n"
            f"Worker summary: {worker_result.summary}\n\n"
            f"{output}"
        )

    frames.append(sse_frame({
        "type": "tool_output",
        "id": search_call_id,
        "call_id": search_call_id,
        "tool": "knowledge_semantic_search",
        "command": "Search semantic vault index",
        "output": output,
    }))
    tool_events.append({
        "type": "tool_output",
        "tool": "knowledge_semantic_search",
        "command": "Search semantic vault index",
        "output": output,
    })

    if worker_result is not None and worker_result.status == "failed":
        final_answer = (
            "Workspace knowledge retrieval failed, so I stopped without "
            "searching Ideaverse or another global root."
        )
    elif worker_result is not None and worker_result.data.get("index_required"):
        final_answer = (
            "The active workspace is not indexed yet, so I stopped without "
            "searching Ideaverse or another global root."
        )
    else:
        final_answer = _build_final_answer(
            results,
            query,
            source_label=source_label,
        )

        llm_response = try_knowledge_llm_response(
            llm_config=llm_config,
            user_message=message,
            deterministic_answer=final_answer,
            runtime_decision=decision.to_dict(),
            evidence_summary={
                "source_authority": (
                    "workspace" if explicit_workspace else "explicit_vault_request"
                ),
                "workspace": explicit_workspace or None,
                "root_id": source_root_id,
                "results": results,
                "formatted_evidence": output,
            },
            hitl_policy={
                "read_only": True,
                "write_allowed": False,
            },
            legacy_fallback_contract={
                "global_fallback_allowed": not bool(explicit_workspace),
                "fail_closed": bool(explicit_workspace),
            },
        )
        if llm_response.get("used_llm") and str(
            llm_response.get("final_answer") or ""
        ).strip():
            final_answer = str(llm_response["final_answer"]).strip()

    frames.append(sse_frame({
        "type": "agent_step",
        "title": "Finalizing grounded note answer",
        "content": "Preparing a read-only answer with source paths and headings.",
    }))

    frames.append(sse_frame({"delta": final_answer}))

    metrics = {
        "runtime": "knowledge_semantic_chat",
        "knowledge_runtime": "semantic_vault_search",
        "knowledge_intent": getattr(decision, "intent", ""),
        "knowledge_confidence": getattr(decision, "confidence", ""),
        "needs_human": False,
        "read_only": True,
        "write_capable": False,
        "semantic_query": query,
        "semantic_results": len(results),
        "semantic_collection": stats.get("collection_name"),
        "semantic_namespace_count": stats.get("namespace_count"),
        "workspace": explicit_workspace or None,
        "root_id": source_root_id,
        "source_authority": "workspace" if explicit_workspace else "explicit_vault_request",
        "elapsed_s": round(time.time() - started, 3),
        "llm_synthesis": bool(
            'llm_response' in locals() and llm_response.get("used_llm")
        ),
        "llm_synthesis_status": (
            llm_response.get("status")
            if 'llm_response' in locals()
            else "not_applicable"
        ),
    }
    frames.append(sse_frame({"type": "metrics", "metrics": metrics}))

    return final_answer, frames, tool_events, metrics
