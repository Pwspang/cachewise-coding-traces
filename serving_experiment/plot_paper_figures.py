"""Plot audited measurements using the layouts of CacheWise Figures 13 and 15b.

Only measured loads and policies are plotted. No paper values or unmeasured
oracle results are substituted for experimental measurements.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = "https://arxiv.org/html/2606.16824v1"
STYLES = {
    "infercept": ("InferCept adaptation", "#8c8c8c", "oo"),
    "baseline": ("vLLM baseline (FCFS/LRU)", "#c9c9c9", ".."),
    "cachewise_no_aging": ("CacheWise", "#91499e", "//"),
    "infercept_prefix_no_aging": ("InferCept adaptation (prefix)", "#8c8c8c", "oo"),
    "prefix_only_no_aging": ("vLLM baseline (prefix/LRU)", "#c9c9c9", ".."),
}


def plot(reports, output):
    output.mkdir(parents=True, exist_ok=True)
    loads = {}
    for path in reports:
        report = json.loads(path.read_text())
        assert report["status"] == "complete", "Use completed, audited results"
        assert report["common_server_arguments_verified"]
        assert report["trace_audit"]["identical_request_hashes_and_lengths_all_runs"]
        counts = {r["sessions"] for r in report["runs"].values()}
        assert len(counts) == 1
        count = counts.pop()
        assert count not in loads, "Duplicate offered-load result"
        loads[count] = report
    loads = dict(sorted(loads.items()))
    exports = []
    for count, report in loads.items():
        for policy, data in report["policies"].items():
            low, high = data["observed_mean_time_range_s"]
            exports.append(
                {
                    "sessions": count,
                    "policy": policy,
                    "repeats": sum(
                        r["policy"] == policy for r in report["runs"].values()
                    ),
                    "mean_session_completion_s": data["mean_session_llm_time_s"],
                    "min_repeat_mean_s": low,
                    "max_repeat_mean_s": high,
                }
            )
    with (output / "paper-figure-data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(exports[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(exports)
    repeat_counts = {r["repeats"] for r in exports}
    assert len(repeat_counts) == 1, "Unequal repeat counts across plotted cases"
    repeats = repeat_counts.pop()

    for name, title, policies in (
        (
            "figure13-session-completion",
            "Mean Session Completion Time",
            ("infercept", "baseline", "cachewise_no_aging"),
        ),
        (
            "figure15b-eviction-comparison",
            "Mean Session Completion Time — Same Prefix-Aware Scheduler",
            ("infercept_prefix_no_aging", "prefix_only_no_aging", "cachewise_no_aging"),
        ),
    ):
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        fig.subplots_adjust(top=0.76, bottom=0.29, left=0.10, right=0.98)
        positions = np.arange(len(loads))
        width = 0.24
        maximum = 0
        for i, policy in enumerate(policies):
            data = [r["policies"][policy] for r in loads.values()]
            means = np.array([r["mean_session_llm_time_s"] for r in data])
            lows = np.array([r["observed_mean_time_range_s"][0] for r in data])
            highs = np.array([r["observed_mean_time_range_s"][1] for r in data])
            label, color, hatch = STYLES[policy]
            bars = ax.bar(
                positions + (i - 1) * width,
                means,
                width,
                color=color,
                edgecolor="#252525",
                hatch=hatch,
                label=label,
                yerr=np.array([means - lows, highs - means]),
                capsize=4,
                zorder=3,
            )
            ax.bar_label(
                bars, labels=[f"{x:.1f}" for x in means], padding=6, fontsize=10
            )
            maximum = max(maximum, max(highs))
        ax.set_ylim(0, maximum * 1.20)
        ax.set_xticks(positions, [str(n) for n in loads])
        ax.set_xlabel("Offered Load (# Sessions)", fontsize=12)
        ax.set_ylabel("Seconds", fontsize=12)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.99, 0.97, "Lower is better", ha="right", va="top", transform=ax.transAxes
        )
        fig.suptitle(title, fontsize=16, y=0.98)
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=3,
            fontsize=9,
        )
        fig.text(
            0.5,
            0.19,
            "Measured adaptation: Qwen2.5-Coder-14B-Instruct · one RTX Pro 6000\n"
            "Session time sums LLM request latencies; tool durations are excluded.\n"
            f"{repeats} repeats per policy; whiskers show observed repeat ranges (not confidence intervals).",
            ha="center",
            va="top",
            fontsize=9,
        )
        fig.text(
            0.5,
            0.085,
            "Prefix-aware aging disabled; CacheWise uses tool-name predictions.\n"
            "InferCept uses preserve/discard without swap. CacheWise* oracle was not measured.\n"
            f"Measured loads only: {', '.join(map(str, loads))}. Layout adapted from CacheWise ({name.split('-')[0]}).",
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
        )
        for extension in ("png", "pdf", "svg"):
            destination = output / f"{name}.{extension}"
            fig.savefig(
                destination,
                dpi=220,
                metadata={"Creator": "CacheWise reproduction"},
            )
            if extension == "svg":
                destination.write_text(
                    "\n".join(
                        line.rstrip() for line in destination.read_text().splitlines()
                    )
                    + "\n"
                )
        plt.close(fig)
    (output / "paper-figure-provenance.json").write_text(
        json.dumps(
            {
                "paper": PAPER,
                "figures": ["13", "15(b)"],
                "measured_loads": list(loads),
                "input_reports": [str(p.resolve()) for p in reports],
                "input_report_sha256": [
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in reports
                ],
                "unmeasured_oracle_omitted": True,
                "interpolation_or_extrapolation": False,
                "metric": "Mean over repeats of mean per-segment summed request latency, excluding external tool delays.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.reports, args.output)
