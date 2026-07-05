from __future__ import annotations

from src.agent_runtime.runtime_router import route_agent_runtime, explain_runtime_route


CASES = [
    {
        "id": "architecture_summary_trace",
        "query": "Summarize how memory is wired in Odysseus.",
        "expect": [
            "Runtime decision:",
            "intent: architecture_summary",
            "runtime: general",
            "runtime_family: deterministic_general",
            "safety_profile: read_only",
            "safe_readonly: True",
        ],
    },
    {
        "id": "file_lookup_trace",
        "query": "Find files related to memory.",
        "expect": [
            "Runtime decision:",
            "intent: file_lookup",
            "runtime: general",
            "runtime_family: deterministic_general",
            "safety_profile: read_only",
        ],
    },
    {
        "id": "exact_lookup_trace",
        "query": "Find where the frontend handles tool_output events.",
        "expect": [
            "Runtime decision:",
            "intent: exact_lookup",
            "runtime: contract",
            "runtime_family: deterministic_contract",
            "safety_profile: read_only",
        ],
    },
    {
        "id": "write_mutation_trace",
        "query": "Patch chat_routes.py to change the stream.",
        "expect": [
            "Runtime decision:",
            "intent: write_mutation",
            "runtime: legacy",
            "runtime_family: legacy_agent_loop",
            "safety_profile: write_blocked_or_legacy",
            "safe_readonly: False",
        ],
    },
    {
        "id": "research_trace",
        "query": "Research the latest Ollama release notes.",
        "expect": [
            "Runtime decision:",
            "intent: legacy_exploratory",
            "runtime: legacy",
            "runtime_family: legacy_agent_loop",
            "safety_profile: legacy_unverified",
            "safe_readonly: False",
        ],
    },
]


def main() -> int:
    print("Odysseus Runtime Trace Eval")
    print("=" * 40)

    passed = 0
    failed = 0

    for case in CASES:
        trace = explain_runtime_route(case["query"])
        missing = [item for item in case["expect"] if item not in trace]

        # Also verify status dict is populated and stable.
        status = route_agent_runtime(case["query"]).to_status()
        required_status_keys = {
            "intent",
            "runtime",
            "runtime_family",
            "safe_readonly",
            "safety_profile",
            "reason",
        }
        missing_status = sorted(required_status_keys - set(status.keys()))

        if missing or missing_status:
            failed += 1
            print(f"FAIL  {case['id']}")
            if missing:
                print("  Missing trace fragments:")
                for item in missing:
                    print(f"   - {item}")
            if missing_status:
                print("  Missing status keys:")
                for key in missing_status:
                    print(f"   - {key}")
            print("  Trace:")
            print(trace)
        else:
            passed += 1
            print(f"PASS  {case['id']}")

    print("=" * 40)
    print(f"Passed: {passed}/{passed + failed}")
    print(f"Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
