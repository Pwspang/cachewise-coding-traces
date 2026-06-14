#!/usr/bin/env python3
"""Run the full tool duration prediction pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(PACKAGE_DIR / script), *(extra or [])]
    subprocess.run(cmd, check=True, cwd=str(PACKAGE_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parsed-traces",
        type=Path,
        default=None,
        help="Input parsed_traces directory",
    )
    parser.add_argument("--skip-insert", action="store_true")
    args = parser.parse_args()

    insert_args: list[str] = []
    if args.parsed_traces:
        insert_args = ["--input", str(args.parsed_traces)]

    if not args.skip_insert:
        print("\n=== Insert HumanPause events ===")
        run("insert_human_pauses.py", insert_args)

    print("\n=== Extract tool call dataset ===")
    run("extract_tool_calls.py")

    print("\n=== Train TF-IDF duration clusters ===")
    run("build_clusters.py")

    print("\nDone. Models in release/tool_duration_prediction/models/")


if __name__ == "__main__":
    main()
