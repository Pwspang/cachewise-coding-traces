#!/usr/bin/env python3
"""Plot expected remaining tool duration vs elapsed time from trained cluster models.
"""

from __future__ import annotations

import argparse
import bisect
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.curves import overall_curve, point_prob_still_running
from shared.paths import DEFAULT_MODELS

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "plots/tool_aging_dist_integration.png"
SELECTED_TOOLS = ["Bash", "Glob", "Grep", "Read", "Edit", "NotebookEdit", "HumanPause", "WebFetch"]

def lookup_prob_still_running(curve: list[dict[str, float]], t_ms: float) -> float:
    if not curve:
        return 0.0
    times = [pt["t_ms"] for pt in curve]
    probs = [point_prob_still_running(pt) for pt in curve]
    idx = bisect.bisect_left(times, t_ms)
    if idx <= 0:
        return probs[0]
    if idx >= len(times):
        return 0.0
    t0, t1 = times[idx - 1], times[idx]
    p0, p1 = probs[idx - 1], probs[idx]
    if t1 == t0:
        return p1
    w = (t_ms - t0) / (t1 - t0)
    return p0 + w * (p1 - p0)


def expected_remaining_time(curve: list[dict[str, float]], elapsed_ms: float) -> float:
    if not curve:
        return 0.0

    p_t = lookup_prob_still_running(curve, elapsed_ms)
    if p_t <= 0:
        return 0.0

    times = np.array([pt["t_ms"] for pt in curve])
    probs = np.array([point_prob_still_running(pt) for pt in curve])
    idx = bisect.bisect_left(times, elapsed_ms)
    if idx >= len(times):
        return 0.0

    if idx == 0:
        start_prob = probs[0]
        start_idx = 0
    else:
        t0, t1 = times[idx - 1], times[idx]
        p0, p1 = probs[idx - 1], probs[idx]
        if t1 == t0:
            start_prob = p1
        else:
            w = (elapsed_ms - t0) / (t1 - t0)
            start_prob = p0 + w * (p1 - p0)
        start_idx = idx

    integral = 0.0
    if start_idx < len(times) and elapsed_ms < times[start_idx]:
        t_start = elapsed_ms
        t_end = times[start_idx]
        integral += (t_end - t_start) * (start_prob + probs[start_idx]) / 2.0

    for i in range(start_idx, len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        integral += (t1 - t0) * (probs[i] + probs[i + 1]) / 2.0

    return float(integral / p_t) if p_t > 0 else 0.0


def load_models(models_dir: Path) -> dict[str, Any]:
    bundle_path = models_dir / "all_models.pkl"
    if bundle_path.exists():
        with bundle_path.open("rb") as f:
            return pickle.load(f)

    stats_path = models_dir / "cluster_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"No models at {models_dir} (missing all_models.pkl and cluster_stats.json)")
    with stats_path.open() as f:
        stats = json.load(f)
    return {
        tool: {
            "overall_prob_still_running": data["overall_prob_still_running"],
            "overall_stats": data.get("overall_stats", {}),
        }
        for tool, data in stats.items()
    }


def plot_tool_aging(
    models: dict[str, Any],
    tool_names: list[str],
    output_file: Path,
) -> dict[str, list[tuple[float, float]]]:
    if len(tool_names) > 10:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    max_time_s = 600
    time_points_s = np.logspace(-2, np.log10(max_time_s), 200)
    colors = [f"C{i % 10}" for i in range(len(tool_names))]
    marker_styles = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "P", "<", ">", "d", "H", "+", "x", "|", "_", "1", "2"]

    curves: dict[str, list[tuple[float, float]]] = {}

    for idx, tool_name in enumerate(tool_names):
        if tool_name not in models:
            print(f"Warning: {tool_name} not found in models")
            continue

        prob_curve = overall_curve(models[tool_name])
        if not prob_curve:
            print(f"Warning: No prob_still_running curve for {tool_name}")
            continue

        max_curve_time_s = max(pt["t_ms"] for pt in prob_curve) / 1000.0
        expected_remaining_times: list[float] = []
        valid_times: list[float] = []

        for t_s in time_points_s:
            if t_s > max_curve_time_s:
                break
            elapsed_ms = t_s * 1000.0
            expected_remaining_ms = expected_remaining_time(prob_curve, elapsed_ms)
            expected_remaining_s = expected_remaining_ms / 1000.0
            if lookup_prob_still_running(prob_curve, elapsed_ms) > 0 and expected_remaining_s > 0:
                expected_remaining_times.append(expected_remaining_s)
                valid_times.append(t_s)

        if not valid_times:
            print(f"Warning: No valid data points for {tool_name}")
            continue

        curves[tool_name] = list(zip(valid_times, expected_remaining_times))
        marker_style = marker_styles[idx % len(marker_styles)]
        markevery = max(1, len(valid_times) // 15)
        ax.plot(
            valid_times,
            expected_remaining_times,
            label=tool_name,
            linewidth=2.0,
            marker=marker_style,
            markersize=5,
            markevery=markevery,
            color=colors[idx],
        )

    ax.set_xlabel("Time Elapsed Since Last Access (s)", fontsize=12)
    ax.set_ylabel("Exp. Remaining Duration (s)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", labelsize=9)
    ax.grid(True, which="both", alpha=0.07, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        bbox_transform=ax.transAxes,
        ncol=4,
        fontsize=10,
        framealpha=0.9,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    leg.set_in_layout(False)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.82)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, facecolor="white")
    print(f"Saved to {output_file}")
    plt.close()
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tools", nargs="*", default=SELECTED_TOOLS)
    args = parser.parse_args()

    print("Loading models from", args.models)
    models = load_models(args.models)
    plot_tool_aging(models, args.tools, args.output)


if __name__ == "__main__":
    main()
