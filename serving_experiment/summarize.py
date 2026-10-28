"""Compare successful runs with identical workload and generation settings."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(root):
    rows, expected = [], None
    for path in sorted(root.glob("*-r*/manifest.json")):
        manifest = json.loads(path.read_text())
        summary = json.loads((path.parent / "summary.json").read_text())
        if manifest["status"] != "complete" or summary["status"] != "complete":
            raise ValueError(f"Incomplete run: {path.parent}")
        signature = [
            manifest[k]
            for k in ("selected_workload_sha256", "model", "sampling", "delay_scale")
        ]
        if expected is not None and expected != signature:
            raise ValueError("Run configurations or selected workloads differ")
        expected = signature
        requests = [r for s in summary["sessions"] for r in s["requests_detail"]]
        cached = [
            (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
            for r in requests
        ]
        rows.append(
            {
                "run": path.parent.name,
                "policy": manifest["policy"],
                "requests": len(requests),
                "sessions": len(summary["sessions"]),
                "elapsed_s": summary["elapsed_s"],
                "mean_session_llm_time_s": statistics.mean(
                    s["llm_time_s"] for s in summary["sessions"]
                ),
                "median_request_latency_s": statistics.median(
                    r["latency_s"] for r in requests
                ),
                "median_ttft_s": statistics.median(
                    r["ttft_s"] for r in requests if r["ttft_s"] is not None
                ),
                "output_tokens_per_s": summary["output_tokens_per_s"],
                "prompt_cache_hit_fraction": sum(cached)
                / sum(r["usage"]["prompt_tokens"] for r in requests)
                if all(v is not None for v in cached)
                else None,
            }
        )
    if not rows or {r["policy"] for r in rows} != {"baseline", "cachewise"}:
        raise ValueError("Need successful baseline and CacheWise runs")
    (root / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    by_policy = {
        policy: [r for r in rows if r["policy"] == policy]
        for policy in ("baseline", "cachewise")
    }
    if len(by_policy["baseline"]) != len(by_policy["cachewise"]):
        raise ValueError("Unbalanced policy repeat counts")
    aggregate = {
        policy: {
            "runs": len(group),
            "mean_segment_llm_time_s": statistics.mean(
                r["mean_session_llm_time_s"] for r in group
            ),
            "range_segment_llm_time_s": [
                min(r["mean_session_llm_time_s"] for r in group),
                max(r["mean_session_llm_time_s"] for r in group),
            ],
            "mean_prompt_cache_hit_fraction": statistics.mean(
                r["prompt_cache_hit_fraction"] for r in group
            )
            if all(r["prompt_cache_hit_fraction"] is not None for r in group)
            else None,
        }
        for policy, group in by_policy.items()
    }
    aggregate["baseline_over_cachewise_llm_time_ratio"] = (
        aggregate["baseline"]["mean_segment_llm_time_s"]
        / aggregate["cachewise"]["mean_segment_llm_time_s"]
    )
    aggregate["limitation"] = (
        "Paired repeats on one selected workload; ranges are observed run ranges, not confidence intervals."
    )
    (root / "comparison-summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    summarize(p.parse_args().root)
