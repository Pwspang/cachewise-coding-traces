"""Summarize controlled policy ablations without mixing full and trimmed runs."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze(root, additional_roots=()):
    rows, signature = [], None
    for policy in (
        "baseline",
        "prefix_only",
        "eviction_only",
        "cachewise",
        "human_first",
        "queued_first",
    ):
        matches = [
            candidate / f"{policy}-r0"
            for candidate in (root, *additional_roots)
            if (candidate / f"{policy}-r0").exists()
        ]
        if not matches:
            continue
        assert len(matches) == 1, f"Ambiguous source runs for {policy}"
        path = matches[0]
        manifest = json.loads((path / "manifest.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        assert manifest["status"] == summary["status"] == "complete"
        starts = [
            json.loads(line)
            for line in (path / "requests.jsonl").read_text().splitlines()
        ]
        current = {
            r["request_key"]: (
                r["prompt_sha256"],
                r["prompt_tokens"],
                r["output_tokens"],
            )
            for r in starts
            if r["event"] == "request_start"
        }
        if signature is not None:
            assert current == signature, "Ablations received different inputs"
        signature = current
        requests = [r for s in summary["sessions"] for r in s["requests_detail"]]
        assert len(requests) == len(current)
        prompt = sum(r["usage"]["prompt_tokens"] for r in requests)
        cached = sum(
            r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in requests
        )
        rows.append(
            {
                "policy": policy,
                "source_run": path.parent.name,
                "requests": len(requests),
                "elapsed_s": summary["elapsed_s"],
                "mean_segment_llm_s": statistics.mean(
                    s["llm_time_s"] for s in summary["sessions"]
                ),
                "prompt_tokens": prompt,
                "uncached_prompt_tokens": prompt - cached,
                "cache_hit_fraction": cached / prompt,
                "median_ttft_s": statistics.median(r["ttft_s"] for r in requests),
                "segments": {s["id"]: s["llm_time_s"] for s in summary["sessions"]},
            }
        )
    assert len(rows) >= 2
    (root / "diagnostic_comparison.json").write_text(
        json.dumps(
            {
                "identical_request_inputs": True,
                "results": rows,
                "limitation": "One run per variant on a diagnostic subset selected to retain the original regression; not a representative benchmark.",
            },
            indent=2,
        )
        + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    names = {
        "baseline": "Baseline",
        "prefix_only": "Prefix scheduling\nonly",
        "eviction_only": "Predictive eviction\nonly",
        "cachewise": "CacheWise",
        "human_first": "Evict idle\nsessions first",
        "queued_first": "Protect at\narrival",
    }
    labels = [names[r["policy"]] for r in rows]
    for ax, field, ylabel in zip(
        axes,
        ["mean_segment_llm_s", "uncached_prompt_tokens"],
        ["Mean LLM time per segment (s)", "Uncached prompt tokens"],
    ):
        bars = ax.bar(labels, [r[field] for r in rows], color="#2563eb")
        ax.bar_label(bars, fmt="%.1f" if field.endswith("_s") else "%.0f", padding=3)
        ax.set_ylim(0, max(r[field] for r in rows) * 1.16)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Controlled diagnostic replay · same inputs and KV-cache capacity")
    fig.savefig(root / "diagnostic_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--additional-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    analyze(args.root, args.additional_root)
