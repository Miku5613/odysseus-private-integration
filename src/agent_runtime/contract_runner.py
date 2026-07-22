from __future__ import annotations

from pathlib import Path

import argparse

from src.agent_runtime.contract_runtime import run_contract_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Odysseus LangGraph Contract Runtime v0.11")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--legacy-wrapper", action="store_true")
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()

    question = " ".join(args.question)
    workspace = str(
        Path(args.workspace).resolve()
    )

    print(f"Workspace: {workspace}")
    print(f"Question: {question}")
    print()
    print("--- Running Odysseus LangGraph Runtime v0.11.3-contract ---")
    print()

    run_contract_runtime(
        question,
        legacy_wrapper=args.legacy_wrapper,
        workspace=workspace,
    )


if __name__ == "__main__":
    main()

