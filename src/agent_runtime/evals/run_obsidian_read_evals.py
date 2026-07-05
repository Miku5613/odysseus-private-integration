
from __future__ import annotations

import os
import sys

from src.agent_runtime.obsidian_read_graph import plan_read_task, run_obsidian_read_graph


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(("PASS  " if ok else "FAIL  ") + name)
    if detail and not ok:
        print("      " + detail)
    return ok


def main() -> int:
    passed = 0
    failed = 0

    static_cases = [
        ("plan_status", "Check Obsidian status.", "status"),
        ("plan_active", "Read the active note.", "active_note"),
        ("plan_list_root", "List the vault folders.", "list_dir"),
        ("plan_list_quoted", 'List folder "AIOS/".', "list_dir"),
        ("plan_read_quoted", 'Read note "AIOS/Index.md".', "read_note"),
        ("block_write", 'Append this to "AIOS/Index.md".', "unsupported"),
        ("block_delete", 'Delete "AIOS/Index.md".', "unsupported"),
        ("block_rename", 'Rename "AIOS/Index.md".', "unsupported"),
    ]

    for name, query, expected in static_cases:
        kind, target = plan_read_task(query)
        ok = kind == expected
        if check(name, ok, f"expected {expected}, got {kind} target={target!r}"):
            passed += 1
        else:
            failed += 1

    if os.environ.get("OBSIDIAN_API_KEY"):
        live_cases = [
            ("live_status", "Check Obsidian status.", ["Obsidian REST status: OK"]),
            ("live_list_root", "List the vault folders.", ["Obsidian vault listing", "- AIOS/"]),
            ("live_active", "Read the active note.", ["Active Obsidian note content"]),
        ]

        for name, query, contains in live_cases:
            state = run_obsidian_read_graph(query)
            ok = all(x in state.final_answer for x in contains)
            ok = ok and state.metrics.get("read_only") is True and state.metrics.get("write_capable") is False
            if check(name, ok, state.final_answer[:500]):
                passed += 1
            else:
                failed += 1
    else:
        print("SKIP  live_obsidian_cases — OBSIDIAN_API_KEY not set")

    print("=" * 40)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
