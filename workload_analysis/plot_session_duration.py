#!/usr/bin/env python3
"""Session duration CDF from parsed_traces"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_events import DEFAULT_EVENTS_DIR, load_all_events

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "release/workload_analysis"


def load_session_durations(events_dir: Path) -> list[dict]:
    session_data = []
    for path, events in load_all_events(events_dir):
        timestamps = []
        for event in events:
            ts = event.get("timestamp")
            if not ts:
                continue
            try:
                timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                continue
        if len(timestamps) < 2:
            continue
        duration_min = (max(timestamps) - min(timestamps)).total_seconds() / 60
        session_data.append(
            {
                "session_id": path.parent.name,
                "trace_length_minutes": duration_min,
                "num_events": len(events),
            }
        )
    return session_data


def plot_session_duration_rectangular(session_data: list[dict], output_file: Path) -> None:
    session_data = sorted(session_data, key=lambda x: x["trace_length_minutes"])
    trace_lengths = [s["trace_length_minutes"] for s in session_data]
    sorted_lengths = np.sort(trace_lengths)
    y = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

    p25_value = np.percentile(trace_lengths, 25)
    p50_value = np.percentile(trace_lengths, 50)
    p75_value = np.percentile(trace_lengths, 75)
    x_80_value = sorted_lengths[np.where(y >= 0.8)[0][0]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sorted_lengths, y, linewidth=5.0, color="steelblue")

    for label, p_val, y_pos in [
        ("P25", p25_value, 0.25),
        ("P50", p50_value, 0.5),
        ("P75", p75_value, 0.75),
    ]:
        ax.annotate(
            f"{label}: {p_val:.0f} min",
            xy=(p_val, y_pos),
            xytext=(p_val + x_80_value * 0.15, y_pos - 0.15),
            fontsize=18,
            color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            ha="left",
        )

    ax.set_xlabel("Session Duration (min)", fontsize=18)
    ax.set_ylabel("CDF", fontsize=18)
    ax.set_xlim(0, x_80_value * 1.05)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.8)
    ax.set_facecolor("#ffffff")
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches=None, facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    session_data = load_session_durations(args.input)
    print(f"Loaded {len(session_data)} sessions")
    plot_session_duration_rectangular(
        session_data,
        args.output / "session_duration_cdf.png",
    )


if __name__ == "__main__":
    main()
