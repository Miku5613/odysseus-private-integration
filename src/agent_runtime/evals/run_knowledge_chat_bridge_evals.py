
from __future__ import annotations

from src.agent_runtime.knowledge_chat_bridge import run_knowledge_chat_turn, should_intercept_knowledge_chat


CASES = [
    {
        "id": "intercept_active_note",
        "query": "Read the active note.",
        "expect_intercept": True,
        "expect_text": ["Active Obsidian note content"],
        "expect_event_tool": "knowledge_decision",
    },
    {
        "id": "hitl_read_doc",
        "query": "Read the doc.",
        "expect_intercept": True,
        "expect_text": ["I need a bit more context", "uploaded file", "Odysseus code/docs"],
        "expect_event_tool": "knowledge_decision",
    },
    {
        "id": "do_not_intercept_codebase",
        "query": "Find where document streaming is handled.",
        "expect_intercept": False,
    },
    {
        "id": "propose_no_write",
        "query": "Append this to my daily note.",
        "expect_intercept": True,
        "expect_text": ["direct writes are disabled", "will not edit"],
        "expect_event_tool": "knowledge_decision",
    },
    {
        "id": "destructive_block",
        "query": "Delete old daily notes.",
        "expect_intercept": True,
        "expect_text": ["blocking", "No note was edited"],
        "expect_event_tool": "knowledge_decision",
    },
]


def check(case: dict) -> bool:
    intercepted = should_intercept_knowledge_chat(case["query"])
    failures = []

    if intercepted != case["expect_intercept"]:
        failures.append(f"intercept expected {case['expect_intercept']}, got {intercepted}")

    if intercepted:
        final, frames, events, metrics = run_knowledge_chat_turn(case["query"])

        for needle in case.get("expect_text", []):
            if needle not in final:
                failures.append(f"final missing {needle!r}")

        expected_tool = case.get("expect_event_tool")
        if expected_tool and not any(ev.get("tool") == expected_tool for ev in events):
            failures.append(f"missing tool event {expected_tool!r}")

        if any(frame.strip() == "data: [DONE]" for frame in frames):
            failures.append("bridge should not emit standalone [DONE]; chat route owns final DONE")

        if not metrics:
            failures.append("missing metrics")

    if failures:
        print(f"FAIL  {case['id']}")
        for f in failures:
            print("      " + f)
        return False

    print(f"PASS  {case['id']}")
    return True


def main() -> int:
    passed = 0
    failed = 0
    for case in CASES:
        if check(case):
            passed += 1
        else:
            failed += 1
    print("=" * 40)
    print(f"Passed: {passed}/{len(CASES)}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
