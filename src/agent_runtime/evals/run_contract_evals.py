from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = Path(__file__).with_name("contract_eval_cases.json")


def run_case(query: str) -> str:
    cmd = [
        sys.executable,
        "-m",
        "src.agent_runtime.contract_runner",
        "--legacy-wrapper",
        query,
    ]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return proc.stdout


def extract_frames(output: str) -> list[dict]:
    frames: list[dict] = []

    for line in output.splitlines():
        marker = "[legacy-wrapper-frame] data:"
        if marker not in line:
            continue

        payload = line.split(marker, 1)[1].strip()
        payload = payload.replace("\\n\\n", "").strip()

        if payload == "[DONE]":
            continue

        try:
            frames.append(json.loads(payload))
        except Exception:
            pass

    return frames


def frame_text(frames: list[dict]) -> str:
    return json.dumps(frames, ensure_ascii=False, indent=2)


def get_contract(frames: list[dict]) -> dict:
    for frame in frames:
        if frame.get("type") == "task_contract":
            subtasks = frame.get("data", {}).get("subtasks", [])
            if subtasks:
                return subtasks[0]
    return {}


def get_metrics(frames: list[dict]) -> dict:
    for frame in frames:
        if frame.get("type") == "metrics":
            return frame.get("data", {})
    return {}


def get_final_text(frames: list[dict]) -> str:
    bits: list[str] = []
    for frame in frames:
        if "delta" in frame:
            bits.append(str(frame["delta"]))
    return "\n".join(bits)


def get_tool_outputs(frames: list[dict]) -> str:
    bits: list[str] = []
    for frame in frames:
        if frame.get("type") == "tool_output":
            bits.append(str(frame.get("output", "")))
    return "\n".join(bits)


def normalize_for_match(value: object) -> str:
    """Normalize eval text for Windows path / JSON escaping differences."""
    text = str(value)
    text = text.replace("\\\\", "\\")
    text = text.replace("/", "\\")
    return text.lower()


def check_contains(haystack: object, needle: str) -> bool:
    if not needle:
        return True
    return normalize_for_match(needle) in normalize_for_match(haystack)


def main() -> int:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8-sig"))

    passed = 0
    failed = 0
    results = []

    for case in cases:
        name = case["name"]
        output = run_case(case["query"])
        frames = extract_frames(output)

        contract = get_contract(frames)
        metrics = get_metrics(frames)
        final = get_final_text(frames)
        tool_outputs = get_tool_outputs(frames)
        all_text = frame_text(frames) + "\n" + output

        expected_status = case.get("expected_status", "completed")
        completed = int(metrics.get("completed_subtasks", 0) or 0)
        unresolved = int(metrics.get("unresolved_subtasks", 0) or 0)

        checks = []

        if expected_status == "completed":
            checks.append(("status_completed", completed >= 1 and unresolved == 0))
        elif expected_status == "unresolved":
            checks.append(("status_unresolved", unresolved >= 1))

        checks.append(("kind", contract.get("kind") == case.get("expected_kind")))
        checks.append(("target", contract.get("target") == case.get("expected_target")))
        checks.append(("scope", check_contains(contract.get("source_scope", []), case.get("expected_scope_contains", ""))))
        checks.append(("file", check_contains(all_text, case.get("expected_file_contains", ""))))
        checks.append(("text", check_contains(final + "\n" + tool_outputs, case.get("expected_text_contains", ""))))

        ok = all(v for _, v in checks)
        if ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        results.append({
            "name": name,
            "status": status,
            "checks": checks,
            "contract": contract,
            "metrics": metrics,
            "final": final,
            "raw_output": output,
        })

    print()
    print("Odysseus Contract Runtime Eval")
    print("=" * 36)

    for item in results:
        print(f"{item['status']:4}  {item['name']}")
        if item["status"] == "FAIL":
            for check_name, ok in item["checks"]:
                print(f"      {check_name:18} {'ok' if ok else 'FAIL'}")
            print(f"      contract: {item['contract']}")
            print(f"      metrics:  {item['metrics']}")
            print(f"      final:    {item['final'][:500]}")
            if not item.get("metrics"):
                print("      note:     metrics frame missing or failed to parse")
                print("      raw output preview:")
                preview = str(item.get("raw_output", ""))[:2000]
                for line in preview.splitlines():
                    print("        " + line)

    print("=" * 36)
    print(f"Passed: {passed}/{len(cases)}")
    print(f"Failed: {failed}/{len(cases)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

