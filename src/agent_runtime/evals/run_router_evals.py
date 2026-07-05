from __future__ import annotations

import json
from pathlib import Path
import sys

from src.agent_runtime.runtime_router import route_agent_runtime


def main() -> int:
    root = Path(__file__).resolve().parent
    cases_path = root / "router_eval_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8-sig"))

    print("Odysseus Runtime Router Eval")
    print("=" * 40)

    passed = 0
    failed = 0

    for case in cases:
        decision = route_agent_runtime(case["query"])
        ok = True
        reasons = []

        if decision.intent != case["expect_intent"]:
            ok = False
            reasons.append(f"intent expected={case['expect_intent']} got={decision.intent}")

        if decision.target_runtime != case["expect_runtime"]:
            ok = False
            reasons.append(f"runtime expected={case['expect_runtime']} got={decision.target_runtime}")

        if decision.is_safe_readonly != case["expect_safe_readonly"]:
            ok = False
            reasons.append(
                f"safe_readonly expected={case['expect_safe_readonly']} got={decision.is_safe_readonly}"
            )

        if ok:
            passed += 1
            print(f"PASS  {case['id']}")
        else:
            failed += 1
            print(f"FAIL  {case['id']}")
            print(f"      query={case['query']}")
            print(f"      decision={decision.to_dict()}")
            for reason in reasons:
                print(f"      {reason}")

    print("=" * 40)
    print(f"Passed: {passed}/{len(cases)}")
    print(f"Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
