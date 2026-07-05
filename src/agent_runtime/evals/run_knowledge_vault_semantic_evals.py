
from __future__ import annotations

import os

from src.agent_runtime.knowledge_vault_semantic import (
    build_semantic_vault_index,
    get_semantic_vault_stats,
    search_semantic_vault_index,
)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(("PASS  " if ok else "FAIL  ") + name)
    if detail and not ok:
        print("      " + detail)
    return ok


def main() -> int:
    passed = 0
    failed = 0

    if not os.environ.get("OBSIDIAN_API_KEY"):
        print("SKIP  semantic_vault_index_cases — OBSIDIAN_API_KEY not set")
        return 0

    stats = build_semantic_vault_index(rebuild=True, max_notes=1000, batch_size=12)
    stats_dict = stats.to_dict()

    cases = [
        ("semantic_index_has_notes", stats.note_count > 0, str(stats_dict)),
        ("semantic_index_has_chunks", stats.chunk_count > 0, str(stats_dict)),
    ]

    for name, ok, detail in cases:
        if check(name, ok, detail):
            passed += 1
        else:
            failed += 1

    chroma_stats = get_semantic_vault_stats()
    if check("semantic_chroma_namespace_count", chroma_stats.get("namespace_count", 0) > 0, str(chroma_stats)):
        passed += 1
    else:
        failed += 1

    results = search_semantic_vault_index("Odysseus LangGraph runtime progress", k=5)
    if check("semantic_search_odysseus_langgraph", len(results) > 0, str(results[:2])):
        passed += 1
    else:
        failed += 1

    if results:
        top = results[0]
        if check("semantic_result_has_source_path", bool(top.get("path")), str(top)):
            passed += 1
        else:
            failed += 1
        if check("semantic_result_has_preview", bool(top.get("preview")), str(top)):
            passed += 1
        else:
            failed += 1

    print("=" * 40)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
