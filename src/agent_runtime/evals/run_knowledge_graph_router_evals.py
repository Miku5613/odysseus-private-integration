
from __future__ import annotations

from src.agent_runtime.knowledge_graph_router import (
    evaluate_knowledge_request,
    build_human_clarification_prompt,
)


CASES = [
    # High-confidence read
    {
        "id": "read_active_note",
        "query": "Read the active note.",
        "expect_intent": "knowledge_read",
        "expect_confidence": "high",
        "expect_runtime": "obsidian_read_graph",
        "expect_needs_human": False,
    },
    {
        "id": "list_vault_folders",
        "query": "List my vault folders.",
        "expect_intent": "knowledge_read",
        "expect_confidence": "high",
        "expect_runtime": "obsidian_read_graph",
        "expect_needs_human": False,
    },
    {
        "id": "my_notes_odysseus",
        "query": "Summarize my notes about Odysseus.",
        "expect_intent": "knowledge_read",
        "expect_confidence": "high",
        "expect_needs_human": False,
    },
    {
        "id": "what_did_i_write",
        "query": "What did I write about LangGraph?",
        "expect_intent": "knowledge_read",
        "expect_confidence": "high",
        "expect_needs_human": False,
    },
    {
        "id": "known_folder_aios",
        "query": "List AIOS folder.",
        "expect_intent": "knowledge_read",
        "expect_confidence": "high",
        "expect_runtime": "obsidian_read_graph",
    },

    # Human-in-the-loop ambiguity
    {
        "id": "read_the_doc",
        "query": "Read the doc.",
        "expect_intent": "needs_clarification",
        "expect_needs_human": True,
        "expect_contains": ["uploaded file", "Odysseus code/docs"],
    },
    {
        "id": "summarize_the_note",
        "query": "Summarize the note.",
        "expect_intent": "needs_clarification",
        "expect_needs_human": True,
        "expect_contains": ["source is unclear"],
    },
    {
        "id": "find_that_progress_report",
        "query": "Find that progress report.",
        "expect_intent": "needs_clarification",
        "expect_needs_human": True,
    },

    # Recent context can resolve vague notes
    {
        "id": "context_resolves_note",
        "query": "Summarize the note.",
        "context": "We are working in Obsidian/Ideaverse and looking at the active note.",
        "expect_intent": "knowledge_read",
        "expect_confidence": "high",
        "expect_runtime": "obsidian_read_graph",
    },

    # False positives must stay out
    {
        "id": "codebase_document_stream",
        "query": "Find where document streaming is handled.",
        "expect_intent": "not_knowledge",
        "expect_runtime": "legacy_or_existing_router",
    },
    {
        "id": "codebase_note_routes",
        "query": "Find files related to note_routes.py.",
        "expect_intent": "not_knowledge",
    },
    {
        "id": "uploaded_doc",
        "query": "Summarize the document I uploaded.",
        "expect_intent": "not_knowledge",
    },

    # Propose, not write
    {
        "id": "append_daily_note",
        "query": "Append this to my daily note.",
        "expect_intent": "knowledge_propose",
        "expect_needs_human": True,
        "expect_write_allowed": False,
    },
    {
        "id": "create_note",
        "query": "Create a note about v0.16.",
        "expect_intent": "knowledge_propose",
        "expect_needs_human": True,
        "expect_write_allowed": False,
    },
    {
        "id": "update_aios_index",
        "query": "Update AIOS Index with this.",
        "expect_intent": "knowledge_propose",
        "expect_needs_human": True,
        "expect_write_allowed": False,
    },

    # Destructive blocked
    {
        "id": "delete_daily_notes",
        "query": "Delete old daily notes.",
        "expect_intent": "unsupported_mutation",
        "expect_needs_human": True,
        "expect_write_allowed": False,
        "expect_runtime": "blocked",
    },
    {
        "id": "rename_vault_folder",
        "query": "Rename my vault folder.",
        "expect_intent": "unsupported_mutation",
        "expect_needs_human": True,
        "expect_runtime": "blocked",
    },
    {
        "id": "move_wip_notes",
        "query": "Move all WIP notes into AIOS.",
        "expect_intent": "unsupported_mutation",
        "expect_needs_human": True,
        "expect_runtime": "blocked",
    },

    # Research separated
    {
        "id": "web_compare_notes",
        "query": "Search the web and compare with my notes.",
        "expect_intent": "knowledge_research",
        "expect_write_allowed": False,
    },
    {
        "id": "latest_docs_note_draft",
        "query": "Look up current LangGraph docs and make a note draft.",
        "expect_intent": "knowledge_research",
        "expect_write_allowed": False,
    },
]


def check_case(case: dict) -> bool:
    d = evaluate_knowledge_request(case["query"], recent_context=case.get("context", ""))

    failures = []
    if "expect_intent" in case and d.intent != case["expect_intent"]:
        failures.append(f"intent expected {case['expect_intent']}, got {d.intent}")
    if "expect_confidence" in case and d.confidence != case["expect_confidence"]:
        failures.append(f"confidence expected {case['expect_confidence']}, got {d.confidence}")
    if "expect_runtime" in case and d.target_runtime != case["expect_runtime"]:
        failures.append(f"runtime expected {case['expect_runtime']}, got {d.target_runtime}")
    if "expect_needs_human" in case and d.needs_human != case["expect_needs_human"]:
        failures.append(f"needs_human expected {case['expect_needs_human']}, got {d.needs_human}")
    if "expect_write_allowed" in case and d.write_allowed != case["expect_write_allowed"]:
        failures.append(f"write_allowed expected {case['expect_write_allowed']}, got {d.write_allowed}")

    if "expect_contains" in case:
        text = d.to_trace_text() + "\n" + build_human_clarification_prompt(d)
        for needle in case["expect_contains"]:
            if needle not in text:
                failures.append(f"missing text {needle!r}")

    if failures:
        print(f"FAIL  {case['id']}")
        for f in failures:
            print("      " + f)
        print("      decision:", d.to_dict())
        return False

    print(f"PASS  {case['id']}")
    return True


def main() -> int:
    passed = 0
    failed = 0

    for case in CASES:
        if check_case(case):
            passed += 1
        else:
            failed += 1

    print("=" * 40)
    print(f"Passed: {passed}/{len(CASES)}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
