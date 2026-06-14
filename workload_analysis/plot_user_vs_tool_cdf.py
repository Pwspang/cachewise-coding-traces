#!/usr/bin/env python3
"""CDF of user-initiated vs tool-initiated LLM invocations per session."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from human_pauses import count_human_pauses
from load_events import DEFAULT_EVENTS_DIR, load_all_events

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "release/workload_analysis"


def count_tool_calls_and_pauses(events_dir: Path) -> list[dict]:
    session_data: list[dict] = []
    for path, events in load_all_events(events_dir):
        tool_calls = sum(
            1 for e in events if e.get("event_type") == "tool_call"
        )
        human_pauses = count_human_pauses(events)
        session_data.append(
            {
                "tool_calls": tool_calls,
                "human_pauses": human_pauses,
                "session_id": path.parent.name,
            }
        )
    return session_data


def plot_user_vs_tool_cdf(session_data: list[dict], output_file: Path) -> None:
    tool_initiated = [s["tool_calls"] for s in session_data]
    user_initiated = [s["human_pauses"] for s in session_data]

    tool_p50 = np.percentile(tool_initiated, 50)
    user_p50 = np.percentile(user_initiated, 50)
    ratio = tool_p50 / user_p50 if user_p50 > 0 else 0
    print(f"Tool P50: {tool_p50:.1f}, User P50: {user_p50:.1f}, Ratio: {ratio:.2f}x")

    tool_sorted = np.sort(tool_initiated)
    user_sorted = np.sort(user_initiated)
    tool_cdf_y = np.arange(1, len(tool_sorted) + 1) / len(tool_sorted)
    user_cdf_y = np.arange(1, len(user_sorted) + 1) / len(user_sorted)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        tool_sorted,
        tool_cdf_y,
        color="#ff7f0e",
        linestyle="-",
        linewidth=2.5,
        marker="o",
        markevery=0.05,
        markersize=10,
        label="Tool-initiated",
    )
    ax.plot(
        user_sorted,
        user_cdf_y,
        color="#1f77b4",
        linestyle="-",
        linewidth=2.5,
        marker="s",
        markevery=0.05,
        markersize=10,
        label="User-initiated",
    )

    ax.set_xlabel("# LLM Requests", fontsize=18)
    ax.set_ylabel("CDF", fontsize=18)
    ax.set_xscale("symlog", linthresh=1, linscale=0.5)
    ax.grid(True, alpha=0.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(
        fontsize=18,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.0),
    )
    ax.tick_params(axis="both", labelsize=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    session_data = count_tool_calls_and_pauses(args.input)
    total_tools = sum(s["tool_calls"] for s in session_data)
    total_pauses = sum(s["human_pauses"] for s in session_data)
    print(f"Sessions: {len(session_data)}")
    print(f"Total tool calls: {total_tools:,}, human pauses: {total_pauses:,}")
    if total_pauses:
        print(f"Overall ratio: {total_tools / total_pauses:.2f}x")
    plot_user_vs_tool_cdf(session_data, args.output / "user_vs_tool_cdf.png")


if __name__ == "__main__":
    main()
