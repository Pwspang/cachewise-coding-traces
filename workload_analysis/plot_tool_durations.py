#!/usr/bin/env python3
"""Tool execution duration distribution from parsed_traces"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from human_pauses import extract_human_pause_durations
from load_events import DEFAULT_EVENTS_DIR, load_all_events, session_id_from_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "release/workload_analysis"


def load_tool_data(events_dir: Path) -> list[dict]:
    all_events: list[dict] = []
    for path, events in load_all_events(events_dir):
        session_id = session_id_from_path(path)
        tool_calls = [
            e
            for e in events
            if e.get("event_type") == "tool_call" and e.get("execution_time_ms") is not None
        ]
        for tool_call in tool_calls:
            tool_call = dict(tool_call)
            tool_call["execution_time_ms"] = tool_call["execution_time_ms"] / 100.0
            all_events.append(tool_call)

        for pause_ms in extract_human_pause_durations(events):
            if pause_ms > 1000:
                all_events.append(
                    {
                        "tool_name": "Human Pause",
                        "execution_time_ms": pause_ms / 100.0,
                        "session_id": session_id,
                    }
                )
    return all_events


def consolidate_mcp_tools(tool_calls: list[dict]) -> list[dict]:
    tool_durations: dict[str, list[float]] = defaultdict(list)
    for tool in tool_calls:
        tool_name = tool["tool_name"]
        if tool_name in ("KillBash", "KillShell"):
            continue
        if "mcp_" in tool_name.lower() or tool_name.startswith("mcp"):
            consolidated_name = "MCP Tools"
        else:
            consolidated_name = tool_name
        tool_durations[consolidated_name].append(tool["execution_time_ms"])

    tool_stats = []
    for name, durations in tool_durations.items():
        durations_arr = np.array(durations)
        tool_stats.append(
            {
                "name": name,
                "durations": durations_arr,
                "median": float(np.median(durations_arr)),
                "count": len(durations_arr),
            }
        )
    tool_stats.sort(key=lambda x: x["median"], reverse=True)
    return tool_stats


def plot_tool_durations(tool_stats: list[dict], output_file: Path) -> None:
    tool_stats_filtered = [t for t in tool_stats if t["name"] != "Human Pause"]
    tool_names = [t["name"] for t in tool_stats_filtered]
    durations_data = [t["durations"] * 100 / 1000.0 for t in tool_stats_filtered]

    fig, ax = plt.subplots(figsize=(9, 6))
    box_plot = ax.boxplot(
        durations_data,
        tick_labels=tool_names,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2.5, color="#8B4A4A"),
    )
    for patch in box_plot["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)

    ax.set_ylabel("Execution Time (s)", fontsize=16.5)
    ax.set_xlabel("Tool Name", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.01)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_facecolor("#ffffff")

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

    tool_calls = load_tool_data(args.input)
    tool_stats = consolidate_mcp_tools(tool_calls)
    print(f"Tool categories: {len(tool_stats)}, total calls: {len(tool_calls)}")
    plot_tool_durations(tool_stats, args.output / "tool_call_duration_distribution.png")


if __name__ == "__main__":
    main()
