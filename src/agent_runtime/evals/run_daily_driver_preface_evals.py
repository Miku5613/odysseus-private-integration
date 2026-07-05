
from __future__ import annotations

from src.agent_runtime.daily_driver_preface import (
    build_preface_prompt,
    generate_intent_preface,
    is_simple_conversation_fast_path,
)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(("PASS  " if ok else "FAIL  ") + name)
    if detail and not ok:
        print("      " + detail)
    return ok


def main() -> int:
    passed = 0
    failed = 0

    cases = [
        ("fast_hello", is_simple_conversation_fast_path("hello"), ""),
        ("fast_thanks", is_simple_conversation_fast_path("thanks"), ""),
        ("not_fast_vault_question", not is_simple_conversation_fast_path("What did I write about Troubleshooting Odysseus?"), ""),
        ("prompt_has_user_request", "Troubleshooting Odysseus" in str(build_preface_prompt(
            user_request="What did I write about Troubleshooting Odysseus?",
            intent="vault_search",
            resource="Ideaverse notes",
            next_action="semantic_vault_search",
            safety="read-only",
        )), ""),
    ]

    for name, ok, detail in cases:
        if check(name, ok, detail):
            passed += 1
        else:
            failed += 1

    result = generate_intent_preface(
        user_request="What did I write about Troubleshooting Odysseus?",
        intent="vault_search",
        resource="Ideaverse notes",
        next_action="semantic_vault_search",
        safety="read-only",
        timeout_s=2.5,
    )

    text = result.get("text") or ""
    source = result.get("source") or ""
    is_llm = source in {"llm", "llm_native_ollama", "llm_openai_compat"}
    is_honest_technical_fallback = source.startswith("technical_fallback_") and (
        "failed" in text.lower()
        or "disabled" in text.lower()
        or "took too long" in text.lower()
    )

    live_cases = [
        ("preface_has_text", bool(text), str(result)),
        ("preface_is_short", len(text) <= 260, text),
        ("preface_not_internal_node_dump", "KNOWLEDGE_" not in text and "semantic_vault_search" not in text, text),
        ("preface_source_valid", is_llm or is_honest_technical_fallback, str(result)),
        ("preface_not_fake_natural_fallback", source != "fallback_error" and source != "fallback_disabled", str(result)),
    ]

    for name, ok, detail in live_cases:
        if check(name, ok, detail):
            passed += 1
        else:
            failed += 1

    print("=" * 40)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("preface:", text)
    print("source:", result.get("source"), "model:", result.get("model"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
