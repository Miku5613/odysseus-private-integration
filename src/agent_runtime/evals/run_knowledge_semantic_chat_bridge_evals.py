
from __future__ import annotations

import os

from src.agent_runtime.knowledge_semantic_chat_bridge import (
    extract_semantic_query,
    is_semantic_vault_search_request,
    run_knowledge_semantic_chat_turn,
)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(("PASS  " if ok else "FAIL  ") + name)
    if detail and not ok:
        print("      " + detail)
    return ok


def main() -> int:
    passed = 0
    failed = 0

    static_cases = [
        ("detect_search_my_notes", is_semantic_vault_search_request("Search my notes for ChromaDB"), ""),
        ("detect_what_did_i_write", is_semantic_vault_search_request("What did I write about LangGraph runtime progress?"), ""),
        ("do_not_steal_active_note", not is_semantic_vault_search_request("Read the active note."), ""),
        ("extract_search_query", extract_semantic_query("Search my notes for ChromaDB") == "ChromaDB", extract_semantic_query("Search my notes for ChromaDB")),
        ("extract_written_query", "langgraph" in extract_semantic_query("What did I write about LangGraph runtime progress?").lower(), extract_semantic_query("What did I write about LangGraph runtime progress?")),
    ]

    for name, ok, detail in static_cases:
        if check(name, ok, detail):
            passed += 1
        else:
            failed += 1

    if not os.environ.get("OBSIDIAN_API_KEY"):
        print("SKIP  live_semantic_chat_cases — OBSIDIAN_API_KEY not set")
        print("=" * 40)
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        return 1 if failed else 0

    final, frames, tool_events, metrics = run_knowledge_semantic_chat_turn("Search my notes for ChromaDB document RAG")

    live_cases = [
        ("semantic_final_mentions_search", "searched your ideaverse notes" in final.lower(), final[:300]),
        ("semantic_final_has_path", "Path:" in final, final[:500]),
        ("semantic_tool_event_present", any(e.get("tool") == "knowledge_semantic_search" for e in tool_events), str(tool_events)),
        ("semantic_metrics_results", metrics.get("semantic_results", 0) > 0, str(metrics)),
        ("semantic_live_delta_shape", any('"delta"' in f for f in frames), str(frames[-3:])),
    ]

    for name, ok, detail in live_cases:
        if check(name, ok, detail):
            passed += 1
        else:
            failed += 1

    print("=" * 40)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
