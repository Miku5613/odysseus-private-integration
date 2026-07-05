
from __future__ import annotations

import argparse
import json

from .obsidian_read_graph import run_obsidian_read_graph


def main() -> int:
    ap = argparse.ArgumentParser(description="Run standalone Obsidian Read Graph.")
    ap.add_argument("query", help="Read-only Obsidian query.")
    args = ap.parse_args()

    state = run_obsidian_read_graph(args.query)
    print(json.dumps({
        "final_answer": state.final_answer,
        "metrics": state.metrics,
        "observations": [
            {
                "tool": o.tool,
                "args": o.args,
                "success": o.success,
                "output_preview": o.output_preview,
            }
            for o in state.observations
        ],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
