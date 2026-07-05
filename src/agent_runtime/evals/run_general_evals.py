
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CASES = Path(__file__).with_name("general_eval_cases.json")


def run_case(case: dict) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "src.agent_runtime.general_runner",
        case["query"],
        "--workspace",
        str(ROOT),
    ]
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )

    if p.returncode != 0:
        return False, f"runner failed: {p.stderr[-1000:]}"

    try:
        data = json.loads(p.stdout)
    except Exception as e:
        return False, f"bad json: {e}; stdout={p.stdout[:500]}"

    final = data.get("final_answer", "")
    metrics = data.get("metrics", {})
    fallback = bool(metrics.get("fallback_required"))

    if fallback != bool(case["expect_fallback"]):
        return False, f"fallback mismatch expected={case['expect_fallback']} got={fallback}; final={final}"

    low = final.lower().replace("\\", "/")
    for needle in case.get("expect_contains", []):
        if needle.lower().replace("\\", "/") not in low:
            return False, f"missing {needle!r}; final={final}"

    return True, final


def main() -> int:
    cases = json.loads(CASES.read_text(encoding="utf-8-sig"))
    passed = 0
    print("Odysseus General Runtime Eval")
    print("=" * 40)
    for case in cases:
        ok, msg = run_case(case)
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {case['id']}")
        if not ok:
            print(f"      {msg}")
        passed += int(ok)
    print("=" * 40)
    print(f"Passed: {passed}/{len(cases)}")
    print(f"Failed: {len(cases)-passed}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
