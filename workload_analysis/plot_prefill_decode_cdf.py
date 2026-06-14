#!/usr/bin/env python3
"""Prefill and decode length CDFs parsed_traces"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from load_events import DEFAULT_EVENTS_DIR, load_all_events

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "release/workload_analysis"

DATASET_NAME = "CATraces"
DATASET_STYLE = {
    "color": "#E63946",
    "linestyle": "-",
    "linewidth": 2.5,
    "label": DATASET_NAME,
    "marker": "*",
    "markevery": 0.05,
    "markersize": 10,
}


def load_prefill_decode_stats(events_dir: Path) -> dict[str, list[int]]:
    prefill: list[int] = []
    decode: list[int] = []
    for _, events in load_all_events(events_dir):
        for event in events:
            if event.get("event_type") != "llm_call":
                continue
            input_t = event.get("input_tokens", 0) or 0
            cache_read = event.get("cache_read_input_tokens", 0) or 0
            output_t = event.get("output_tokens", 0) or 0
            p = input_t + cache_read
            if p > 0:
                prefill.append(p)
            if output_t > 0:
                decode.append(output_t)
    return {"prefill": prefill, "decode": decode}


def _plot_cdf_on_ax(ax, values: list[int], *, xlabel: str) -> None:
    if not values:
        return
    sorted_vals = np.sort(np.array(values))
    y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(
        sorted_vals,
        y,
        color=DATASET_STYLE["color"],
        linestyle=DATASET_STYLE["linestyle"],
        linewidth=DATASET_STYLE["linewidth"],
        marker=DATASET_STYLE["marker"],
        markevery=DATASET_STYLE["markevery"],
        markersize=DATASET_STYLE["markersize"],
        label=DATASET_STYLE["label"],
    )
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("CDF", fontsize=18)
    ax.set_xscale("symlog", linthresh=1, linscale=0.5)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=15, frameon=True)


def plot_prefill_cdf(stats: dict[str, list[int]], output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _plot_cdf_on_ax(ax, stats["prefill"], xlabel="Prefill Length (tokens)")
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {output_file}")


def plot_decode_cdf(stats: dict[str, list[int]], output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _plot_cdf_on_ax(ax, stats["decode"], xlabel="Decode Length (tokens)")
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    stats = load_prefill_decode_stats(args.input)
    print(
        f"Prefill calls: {len(stats['prefill'])}, "
        f"Decode calls: {len(stats['decode'])}"
    )
    plot_prefill_cdf(stats, args.output / "prefill_decode_cdf_prefill.png")
    plot_decode_cdf(stats, args.output / "prefill_decode_cdf_decode.png")


if __name__ == "__main__":
    main()
