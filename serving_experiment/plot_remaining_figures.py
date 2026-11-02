"""Render the measurable N=10 panels of CacheWise Figures 14–16.

Extract per-run statistics from audited request traces, or replot the published
CSV. Unsupported experiments remain explicitly absent, never estimated.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_paper_figures import PAPER, STYLES

MAIN = ("infercept", "baseline", "cachewise_no_aging")
PREFIX = ("infercept_prefix_no_aging", "prefix_only_no_aging", "cachewise_no_aging")


def extract(root, report):
    rows, hashes = [], {}
    assert (root / "COMPLETE").exists()
    for name, expected in report["runs"].items():
        path = root / name / "requests.jsonl"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        requests = [json.loads(s) for s in path.read_text().splitlines()]
        assert not any(r["event"] == "request_error" for r in requests)
        requests = [r for r in requests if r["event"] == "request_end"]
        assert len(requests) == len({r["request_key"] for r in requests}) == 83
        audit = report["trace_audit"]["runs"][name]
        assert (
            sum(r["usage"]["prompt_tokens"] for r in requests) == audit["prompt_tokens"]
        )
        output_tokens = sum(r["usage"]["completion_tokens"] for r in requests)
        assert output_tokens == audit["output_tokens"]
        sessions = defaultdict(list)
        for request in requests:
            sessions[request["session"]].append(request)
        assert len(sessions) == expected["sessions"] == 10
        useful_tokens = 0
        for group in sessions.values():
            previous_prefix = 0
            for request in sorted(group, key=lambda r: r["turn"]):
                usage = request["usage"]
                prompt, output = usage["prompt_tokens"], usage["completion_tokens"]
                assert prompt >= previous_prefix
                useful_tokens += prompt - previous_prefix + output
                previous_prefix = prompt + output
        latencies = [r["latency_s"] for r in requests]
        mean_session = sum(latencies) / len(sessions)
        assert np.isclose(
            mean_session, expected["mean_session_llm_time_s"], rtol=0, atol=1e-8
        )
        elapsed = expected["elapsed_s"]
        rows.append(
            {
                "run": name,
                "policy": expected["policy"],
                "sessions": 10,
                "requests": len(requests),
                "elapsed_s": elapsed,
                "useful_tokens": useful_tokens,
                "output_tokens": output_tokens,
                "useful_token_goodput_k_s": useful_tokens / elapsed / 1000,
                "output_token_goodput_s": output_tokens / elapsed,
                "request_throughput_s": len(requests) / elapsed,
                "p50_latency_s": float(np.percentile(latencies, 50, method="linear")),
                "p90_latency_s": float(np.percentile(latencies, 90, method="linear")),
                "p99_latency_s": float(np.percentile(latencies, 99, method="linear")),
                "mean_session_completion_s": mean_session,
            }
        )
    assert len({r["useful_tokens"] for r in rows}) == 1
    return rows, hashes


def save(fig, output, name):
    for extension in ("png", "pdf", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=200, metadata={"Creator": "CacheWise reproduction"})
        if extension == "svg":
            path.write_text(
                "\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def bars(ax, rows, policies, field, title, unit, styles):
    width = 0.72 / len(policies)
    maximum = 0
    for i, policy in enumerate(policies):
        values = np.array([float(r[field]) for r in rows if r["policy"] == policy])
        assert len(values) == 3
        mean = values.mean()
        label, color, hatch = styles[policy]
        bar = ax.bar(
            (i - (len(policies) - 1) / 2) * width,
            mean,
            width,
            color=color,
            edgecolor="#252525",
            hatch=hatch,
            label=label,
            yerr=[[mean - values.min()], [values.max() - mean]],
            capsize=3,
            zorder=3,
        )
        ax.bar_label(
            bar,
            labels=[f"{mean:.3f}" if mean < 1 else f"{mean:.1f}"],
            padding=4,
            fontsize=9,
        )
        maximum = max(maximum, values.max())
    ax.set(
        title=title,
        ylabel=unit,
        xlabel="Offered load (# sessions)",
        ylim=(0, maximum * 1.25),
    )
    ax.set_xticks([0], ["10"])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def decorate(fig, ax, title, note, *, columns=3):
    fig.suptitle(title, fontsize=16, y=0.985)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=columns,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.16 if fig.get_figheight() < 7 else 0.10,
        "N=10 only · Qwen2.5-Coder-14B-Instruct · one RTX Pro 6000 · 3 repeats per policy\n"
        "Bars average per-run metrics; whiskers show observed repeat ranges, not confidence intervals.\n"
        "Useful tokens = new prompt tokens + generated tokens, counted once per segment; rates use whole-run elapsed time.",
        ha="center",
        va="top",
        fontsize=8.5,
    )
    fig.text(0.5, 0.035, note, ha="center", va="top", fontsize=8, color="#444444")


def plot(rows, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.subplots_adjust(
        top=0.81, bottom=0.20, left=0.07, right=0.98, hspace=0.65, wspace=0.35
    )
    panels = [
        ("useful_token_goodput_k_s", "(a) Useful-token goodput ↑", "K tokens/s"),
        None,
        ("p50_latency_s", "(c) P50 request latency ↓", "Seconds"),
        ("p90_latency_s", "(d) P90 request latency ↓", "Seconds"),
        ("p99_latency_s", "(e) P99 request latency ↓", "Seconds"),
        ("request_throughput_s", "(f) Request throughput ↑", "Requests/s"),
    ]
    for ax, panel in zip(axes.flat, panels):
        if panel is None:
            ax.set_title("(b) Full block-eviction count")
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "Not recorded\n\nOnly 1% block-lifetime sampling was enabled.\nA sampled count is not a full eviction total.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
        else:
            field, title, unit = panel
            bars(ax, rows, MAIN, field, title, unit, STYLES)
    decorate(
        fig,
        axes[0, 0],
        "Figure 14-style serving efficiency — measured panels",
        "Latency percentiles use linear interpolation within each run. InferCept is the preserve/discard adaptation; no oracle was measured.",
    )
    save(fig, output, "figure14-serving-efficiency")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2))
    fig.subplots_adjust(top=0.76, bottom=0.28, left=0.08, right=0.98, wspace=0.30)
    bars(
        axes[0],
        rows,
        PREFIX,
        "useful_token_goodput_k_s",
        "(a) Useful-token goodput ↑",
        "K tokens/s",
        STYLES,
    )
    bars(
        axes[1],
        rows,
        PREFIX,
        "mean_session_completion_s",
        "(b) Mean session completion time ↓",
        "Seconds",
        STYLES,
    )
    decorate(
        fig,
        axes[0],
        "Figure 15-style eviction comparison — same prefix-aware scheduler",
        "Aging disabled in all cases; CacheWise uses tool-name prediction. InferCept adaptation has no swap; oracle not measured.",
    )
    save(fig, output, "figure15-eviction-ablation")

    policies = (
        "baseline",
        "prefix_only_no_aging",
        "infercept",
        "infercept_prefix_no_aging",
    )
    styles = {
        "baseline": ("LRU / FCFS", "#c9c9c9", ".."),
        "prefix_only_no_aging": ("LRU / prefix-aware", "#7ba3cf", "//"),
        "infercept": ("InferCept / FCFS", "#8c8c8c", "oo"),
        "infercept_prefix_no_aging": ("InferCept / prefix-aware", "#91499e", "\\\\"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2))
    fig.subplots_adjust(top=0.76, bottom=0.28, left=0.08, right=0.98, wspace=0.30)
    bars(
        axes[0],
        rows,
        policies,
        "useful_token_goodput_k_s",
        "(a) Useful-token goodput ↑",
        "K tokens/s",
        styles,
    )
    bars(
        axes[1],
        rows,
        policies,
        "mean_session_completion_s",
        "(b) Mean session completion time ↓",
        "Seconds",
        styles,
    )
    decorate(
        fig,
        axes[0],
        "Figure 16-style scheduling ablation — measured policy pairs",
        "Prefix-aware aging disabled. No matched CacheWise/FCFS run exists in job 10807, so that pair is omitted.",
        columns=4,
    )
    save(fig, output, "figure16-scheduling-ablation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path)
    source.add_argument("--data", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "complete" and report["common_server_arguments_verified"]
    assert report["trace_audit"]["identical_request_hashes_and_lengths_all_runs"]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.root:
        rows, hashes = extract(args.root, report)
        with (args.output / "performance-figure-data.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        (args.output / "performance-figure-provenance.json").write_text(
            json.dumps(
                {
                    "paper": PAPER,
                    "figures": ["14 (partial)", "15", "16 (partial)"],
                    "report_sha256": hashlib.sha256(
                        args.report.read_bytes()
                    ).hexdigest(),
                    "request_trace_sha256": hashes,
                    "measured_loads": [10],
                    "goodput_definition": "New prompt tokens and generated output tokens counted once per segment, divided by whole-run elapsed time, including external waits. Output-only goodput is also provided separately in the CSV.",
                    "percentile_definition": "NumPy linear interpolation within each run; plot mean and observed range of the three per-run percentiles, not a percentile of per-request averages.",
                    "missing": {
                        "14(b),17": "Full eviction counts were not recorded; block-lifetime sampling was 1%.",
                        "18": "No CPU offload experiment was run.",
                        "19": "No matched predictor-granularity sweep or oracle was run.",
                        "20": "No exclusive CPU scheduler/GPU execution profiling; request residence times cannot substitute for these.",
                        "16 CacheWise FCFS pair": "Not run in the matched experiment.",
                    },
                },
                indent=2,
            )
            + "\n"
        )
    else:
        with args.data.open() as handle:
            rows = list(csv.DictReader(handle))
    assert len(rows) == 15 and {r["run"] for r in rows} == set(report["runs"])
    assert all(int(r["sessions"]) == 10 for r in rows)
    for row in rows:
        expected = report["runs"][row["run"]]
        assert row["policy"] == expected["policy"]
        assert int(row["requests"]) == expected["requests"]
        assert float(row["elapsed_s"]) == expected["elapsed_s"]
        assert np.isclose(
            float(row["mean_session_completion_s"]),
            expected["mean_session_llm_time_s"],
            rtol=0,
            atol=1e-8,
        )
        assert (
            0
            <= float(row["p50_latency_s"])
            <= float(row["p90_latency_s"])
            <= float(row["p99_latency_s"])
            <= expected["max_request_latency_s"]
        )
    plot(rows, args.output)


if __name__ == "__main__":
    main()
