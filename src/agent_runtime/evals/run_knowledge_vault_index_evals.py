
from __future__ import annotations

import os

from src.agent_runtime.knowledge_vault_index import (
    build_structural_vault_index,
    load_cached_vault_index,
    search_structural_vault_index,
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
        print("SKIP  live_vault_index_cases — OBSIDIAN_API_KEY not set")
        return 0

    index = build_structural_vault_index(max_depth=8, max_notes=500)

    cases = [
        ("index_has_notes", index.note_count > 0, f"note_count={index.note_count}"),
        ("index_has_folders", index.folder_count > 0, f"folder_count={index.folder_count}"),
        ("index_has_aios_or_root", any("AIOS" in f or f == "/" for f in index.folders), str(index.folders[:10])),
        ("cache_loads", load_cached_vault_index() is not None, "cache did not load"),
    ]

    for name, ok, detail in cases:
        if check(name, ok, detail):
            passed += 1
        else:
            failed += 1

    results = search_structural_vault_index(index, "Odysseus", limit=5)
    if check("search_odysseus_returns_results", len(results) > 0, str(results[:2])):
        passed += 1
    else:
        failed += 1

    print("=" * 40)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
