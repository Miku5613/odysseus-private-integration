
from __future__ import annotations

from src.agent_runtime.knowledge_semantic_chat_bridge import (
    _rerank_and_diversify_results,
    run_knowledge_semantic_chat_turn,
)
from src.agent_runtime.knowledge_vault_semantic import search_semantic_vault_index


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(("PASS  " if ok else "FAIL  ") + name)
    if detail and not ok:
        print("      " + detail)
    return ok


def main() -> int:
    passed = 0
    failed = 0

    query = "Troubleshooting Odysseus"
    raw = search_semantic_vault_index(query, k=18)
    ranked = _rerank_and_diversify_results(raw, query, max_results=6)

    paths = [str(r.get("path") or "") for r in ranked]
    joined = "\n".join(paths)

    tests = [
        (
            "has_results",
            bool(ranked),
            joined,
        ),
        (
            "promotes_troubleshooting_note_top3",
            any("Odysseus Troubleshooting" in p or "Odysseus_Troubleshooting" in p for p in paths[:3]),
            joined,
        ),
        (
            "diversifies_paths",
            len(set(paths[:4])) >= min(3, len(paths[:4])),
            joined,
        ),
        (
            "generic_odysseus_not_two_top_slots",
            not (
                len(paths) >= 2
                and paths[0].endswith("Atlas/Odysseus/Odysseus.md")
                and paths[1].endswith("Atlas/Odysseus/Odysseus.md")
            ),
            joined,
        ),
    ]

    answer, frames, tool_events, metrics = run_knowledge_semantic_chat_turn(
        "What did I write about Troubleshooting Odysseus?"
    )

    frame_text = "".join(frames)

    tests.extend([
        (
            "live_frames_include_preface_delta_before_tools",
            frame_text.find('"delta"') != -1 and frame_text.find('"delta"') < frame_text.find('"tool_start"'),
            frame_text[:1200],
        ),
        (
            "final_mentions_troubleshooting_note",
            "Odysseus Troubleshooting" in answer or "Odysseus_Troubleshooting" in answer,
            answer[:1200],
        ),
    ])

    print("Ranked paths:")
    for i, p in enumerate(paths, start=1):
        print(f"{i}. {p}")

    print()
    for name, ok, detail in tests:
        if check(name, ok, detail):
            passed += 1
        else:
            failed += 1

    print("=" * 40)
    print("Passed:", passed)
    print("Failed:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
