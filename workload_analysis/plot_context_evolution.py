#!/usr/bin/env python3
"""Plot context length growth vs turn number"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from load_events import DEFAULT_EVENTS_DIR, load_all_events

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "release/workload_analysis"


def load_sessions(events_dir: Path) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for _, events in load_all_events(events_dir):
        llm_calls = [e for e in events if e.get("event_type") == "llm_call"]
        if not llm_calls:
            continue
        llm_calls.sort(key=lambda x: x.get("timestamp", ""))
        turn_numbers: list[int] = []
        context_lengths: list[int] = []
        current_context = 0
        for turn_idx, call in enumerate(llm_calls, 1):
            input_tokens = call.get("input_tokens", 0) or 0
            cache_read = call.get("cache_read_input_tokens", 0) or 0
            call_context = input_tokens + cache_read
            current_context = max(current_context, call_context)
            turn_numbers.append(turn_idx)
            context_lengths.append(current_context)
        if context_lengths:
            sessions.append(
                {
                    "turn_numbers": turn_numbers,
                    "context_lengths": context_lengths,
                    "max_context": max(context_lengths),
                    "total_calls": len(llm_calls),
                }
            )
    return sessions


def calculate_percentiles(sessions: list[dict[str, Any]], max_turns: int = 500) -> dict[str, np.ndarray]:
    if not sessions:
        return {}
    turn_grid = np.linspace(1, max_turns, min(500, max_turns))
    aligned_contexts = []
    for session in sessions:
        aligned = np.interp(
            turn_grid,
            session["turn_numbers"],
            session["context_lengths"],
            left=0,
            right=session["context_lengths"][-1],
        )
        aligned_contexts.append(aligned)
    aligned_contexts_arr = np.array(aligned_contexts)
    return {
        "turn_grid": turn_grid,
        "p50": np.percentile(aligned_contexts_arr, 50, axis=0),
        "p75": np.percentile(aligned_contexts_arr, 75, axis=0),
        "p90": np.percentile(aligned_contexts_arr, 90, axis=0),
    }


def plot_context_evolution(sessions: list[dict[str, Any]], output_file: Path) -> None:
    style = {"color": "#1976d2", "linewidth": 4.5, "alpha": 0.85}
    max_turns = min(max(s["total_calls"] for s in sessions), 500)
    percentiles = calculate_percentiles(sessions, max_turns)
    if not percentiles:
        raise ValueError("No session data to plot")

    turn_grid = percentiles["turn_grid"]
    p50 = percentiles["p50"]
    p75 = percentiles["p75"]
    p90 = percentiles["p90"]
    valid_mask = turn_grid <= max_turns

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        turn_grid[valid_mask],
        p50[valid_mask],
        label="CATraces",
        color=style["color"],
        linestyle="-",
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        zorder=10,
    )
    ax.plot(
        turn_grid[valid_mask],
        p75[valid_mask],
        color=style["color"],
        linestyle="--",
        linewidth=style["linewidth"] * 0.8,
        alpha=style["alpha"] * 0.7,
        zorder=9,
    )
    ax.plot(
        turn_grid[valid_mask],
        p90[valid_mask],
        color=style["color"],
        linestyle=":",
        linewidth=style["linewidth"] * 0.8,
        alpha=style["alpha"] * 0.7,
        zorder=8,
    )
    ax.fill_between(
        turn_grid[valid_mask],
        p50[valid_mask],
        p75[valid_mask],
        alpha=0.15,
        color=style["color"],
        zorder=5,
    )
    ax.plot([], [], color="black", linestyle="--", linewidth=3.5, label="75th percentile")
    ax.plot([], [], color="black", linestyle=":", linewidth=3.5, label="90th percentile")

    ax.set_xlabel("Turn Number", fontsize=18, fontweight="bold")
    ax.set_ylabel("Context Length", fontsize=18, fontweight="bold")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _p: f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}")
    )
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=1, right=max_turns)
    ax.set_ylim(bottom=0)
    ax.legend()

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    sessions = load_sessions(args.input)
    print(f"Loaded {len(sessions)} sessions")
    plot_context_evolution(sessions, args.output / "context_evolution.png")


if __name__ == "__main__":
    main()
