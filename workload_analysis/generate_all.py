#!/usr/bin/env python3
"""Generate all workload analysis plots from release/parsed_traces."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVENTS_DIR = ROOT / "release/parsed_traces"
DEFAULT_OUTPUT_DIR = ROOT / "release/workload_analysis"

PLOTS = [
    ("plot_session_duration.py", "session_duration_cdf.png"),
    ("plot_prefill_decode_cdf.py", "prefill_decode_cdf_prefill.png + prefill_decode_cdf_decode.png"),
    ("plot_context_evolution.py", "context_evolution.png"),
    ("plot_turns_distribution.py", "turns_distribution.png"),
    ("plot_tool_durations.py", "tool_call_duration_distribution.png"),
    ("plot_user_vs_tool_cdf.py", "user_vs_tool_cdf.png"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()

    for script, outputs in PLOTS:
        script_path = SCRIPT_DIR / script
        print(f"Running {script} -> {outputs}")
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--input",
                str(args.input),
                "--output",
                str(args.output),
            ],
            check=True,
            cwd=str(SCRIPT_DIR),
        )
        print()

    print("All plots written to", args.output)


if __name__ == "__main__":
    main()
