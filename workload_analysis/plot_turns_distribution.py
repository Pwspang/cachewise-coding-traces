#!/usr/bin/env python3
"""Distribution of number of turns per session from parsed traces"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_events import DEFAULT_EVENTS_DIR, load_all_events

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "release/workload_analysis"


def load_turn_counts(events_dir: Path) -> list[int]:
    turn_counts: list[int] = []
    for _, events in load_all_events(events_dir):
        llm_calls = [e for e in events if e.get("event_type") == "llm_call"]
        if llm_calls:
            turn_counts.append(len(llm_calls))
    return turn_counts


def plot_turns_cdf(turn_counts: list[int], output_file: Path) -> None:
    sorted_turns = np.sort(turn_counts)
    y = np.arange(1, len(sorted_turns) + 1) / len(sorted_turns)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        sorted_turns,
        y,
        label="CATraces",
        color="#2e7d32",
        linewidth=2.5,
        alpha=0.85,
        zorder=10,
    )

    ax.set_xlabel("Number of Turns", fontsize=18)
    ax.set_ylabel("CDF", fontsize=18)
    ax.set_xscale("log")
    ax.set_xlim(left=1)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, alpha=0.3, linestyle="--", zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=14, framealpha=0.9)

    plt.tight_layout(pad=0.8)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    turn_counts = load_turn_counts(args.input)
    print(f"Loaded {len(turn_counts)} sessions")
    if turn_counts:
        print(
            f"  mean={np.mean(turn_counts):.1f}, median={np.median(turn_counts):.1f}, "
            f"max={max(turn_counts)}"
        )
    plot_turns_cdf(turn_counts, args.output / "turns_distribution.png")


if __name__ == "__main__":
    main()
