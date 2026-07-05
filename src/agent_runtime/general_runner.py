
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .general_graph import run_general_runtime, state_to_legacy_frames


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--legacy-wrapper", action="store_true")
    args = parser.parse_args()

    state = run_general_runtime(args.query, args.workspace)

    if args.legacy_wrapper:
        for frame in state_to_legacy_frames(state):
            print("[legacy-wrapper-frame] data: " + json.dumps(frame, ensure_ascii=False) + r"\n\n", flush=True)
        print("[legacy-wrapper-frame] data: [DONE]" + r"\n\n", flush=True)
    else:
        print(json.dumps({
            "final_answer": state.final_answer,
            "metrics": state.metrics,
            "task": state.task.__dict__ if state.task else None,
            "observations": [o.__dict__ for o in state.observations],
        }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
