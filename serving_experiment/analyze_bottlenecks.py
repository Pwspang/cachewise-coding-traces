"""Attribute recorded request latency and characterize cache pressure offline.

Phase metrics are cumulative request residence times, not GPU/CPU profiler
durations. Concurrent request times overlap. No causal speedup is inferred.
"""

import argparse
import json
import re
import statistics
from pathlib import Path


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def metric(text, name, labels=None):
    pattern = re.compile(r"^" + re.escape("vllm:" + name) + r"\{([^}]*)\} (\S+)$")
    values = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match and all(
            f'{key}="{value}"' in match[1] for key, value in (labels or {}).items()
        ):
            values.append(float(match[2]))
    if not values:
        raise ValueError(f"Missing metric: {name}, labels={labels}")
    return sum(values)


def analyze(root, workload):
    source = json.loads(workload.read_text())
    by_id = {s["id"]: s for s in source["sessions"]}
    results = {}
    signatures = []
    for path in sorted(root.glob("*-r*/manifest.json")):
        manifest = json.loads(path.read_text())
        assert manifest["status"] == "complete"
        signatures.append(manifest["selected_workload_sha256"])
        sessions = [by_id[sid] for sid in manifest["session_ids"]]
        events = records(path.parent / "requests.jsonl")
        ends = [r for r in events if r["event"] == "request_end"]
        starts = [r for r in events if r["event"] == "request_start"]
        snapshots = [
            r for r in records(path.parent / "metrics.jsonl") if r.get("status") == 200
        ]
        final = snapshots[-1]["text"]
        phases = {
            phase: metric(final, f"request_{phase}_time_seconds_sum")
            for phase in ("queue", "prefill", "decode")
        }
        total = metric(final, "e2e_request_latency_seconds_sum")
        for phase in phases:
            assert metric(final, f"request_{phase}_time_seconds_count") == len(ends)
        assert metric(final, "e2e_request_latency_seconds_count") == len(ends)
        assert metric(final, "request_generation_tokens_sum") == sum(
            r["usage"]["completion_tokens"] for r in ends
        )
        phases["unattributed"] = total - sum(phases.values())
        assert phases["unattributed"] >= -0.001
        prompt = sum(r["usage"]["prompt_tokens"] for r in ends)
        uncached = sum(
            r["usage"]["prompt_tokens"]
            - r["usage"]["prompt_tokens_details"]["cached_tokens"]
            for r in ends
        )
        assert metric(final, "request_prefill_kv_computed_tokens_sum") == uncached
        cold = sum(s["turns"][0]["prompt_tokens"] for s in sessions)
        growth = sum(
            cur["prompt_tokens"] - prev["prompt_tokens"] - prev["output_tokens"]
            for s in sessions
            for prev, cur in zip(s["turns"], s["turns"][1:])
        )
        delays = [t["delay_after_s"] for s in sessions for t in s["turns"][:-1]]
        count = len(ends)
        # Trapezoid estimates over the observed metric window (roughly 2 s sampling).
        window = snapshots[-1]["elapsed_s"] - snapshots[0]["elapsed_s"]
        occupancy = {}
        for name in (
            "num_requests_running",
            "num_requests_waiting",
            "kv_cache_usage_perc",
        ):
            values = [metric(r["text"], name) for r in snapshots]
            integral = sum(
                (b["elapsed_s"] - a["elapsed_s"]) * (va + vb) / 2
                for a, b, va, vb in zip(snapshots, snapshots[1:], values, values[1:])
            )
            occupancy[name] = {
                "time_weighted_mean": integral / window,
                "maximum": max(values),
            }
        command = json.loads((root / f"{path.parent.name}-command.json").read_text())
        blocks = int(command[command.index("--num-gpu-blocks-override") + 1])
        block_config = next(
            line
            for line in final.splitlines()
            if line.startswith("vllm:cache_config_info{")
        )
        block_size = int(re.search(r'block_size="(\d+)"', block_config)[1])
        results[path.parent.name] = {
            "policy": manifest["policy"],
            "requests": count,
            "segments": len(sessions),
            "server_request_time_sum_s": total,
            "client_request_time_sum_s": sum(r["latency_s"] for r in ends),
            "phase_request_time_sum_s": phases,
            "phase_fraction_of_server_request_time": {
                k: v / total for k, v in phases.items()
            },
            "requests_queued_over_30s": count
            - metric(final, "request_queue_time_seconds_bucket", {"le": "30.0"}),
            "preemptions": metric(final, "num_preemptions_total"),
            "sampled_eviction_events": metric(final, "kv_block_lifetime_seconds_count"),
            "prompt_tokens": prompt,
            "uncached_prompt_tokens": uncached,
            "cold_initial_prompt_tokens": cold,
            "fresh_growth_prompt_tokens": growth,
            "uncached_above_fresh_token_floor": uncached - cold - growth,
            "kv_token_capacity": blocks * block_size,
            "initial_context_to_capacity_ratio": cold / (blocks * block_size),
            "prompt_length_range": [
                min(r["prompt_tokens"] for r in starts),
                max(r["prompt_tokens"] for r in starts),
            ],
            "transitions": len(delays),
            "transitions_delay_at_most_1s": sum(d <= 1 for d in delays),
            "external_delay_sum_s": sum(delays),
            "occupancy": occupancy,
        }
    assert len(set(signatures)) == 1
    aggregate = {}
    for policy in ("baseline", "cachewise"):
        group = [r for r in results.values() if r["policy"] == policy]
        total = sum(r["server_request_time_sum_s"] for r in group)
        aggregate[policy] = {
            "runs": len(group),
            "phase_fraction": {
                k: sum(r["phase_request_time_sum_s"][k] for r in group) / total
                for k in ("queue", "prefill", "decode", "unattributed")
            },
            "mean_phase_time_per_segment_s": {
                k: statistics.mean(
                    r["phase_request_time_sum_s"][k] / r["segments"] for r in group
                )
                for k in ("queue", "prefill", "decode", "unattributed")
            },
        }
    result = {
        "interpretation": "Overlapping request residence times, not exclusive CPU/GPU execution times. Unattributed is the residual between e2e and phase sums. Cache usage measures active blocks, not all retained cached prefixes. Excess uncached tokens include repeated computation and block-alignment/last-token overhead; not an exact eviction count. Queue waits above 30s show exposure to the aging threshold, not measured counts of FCFS overrides.",
        "runs": results,
        "aggregate": aggregate,
    }
    (root / "bottleneck-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(root, aggregate)
    print(json.dumps(result, indent=2))


def plot(root, aggregate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    policies = ["baseline", "cachewise"]
    left = [0.0, 0.0]
    for phase, color in zip(
        ("queue", "prefill", "decode", "unattributed"),
        ("#64748b", "#2563eb", "#d97706", "#d1d5db"),
    ):
        values = [
            aggregate[p]["mean_phase_time_per_segment_s"][phase] for p in policies
        ]
        ax.barh([1, 0], values, left=left, height=0.5, color=color, label=phase.title())
        for i, value in enumerate(values):
            if value >= 20:
                ax.text(
                    left[i] + value / 2,
                    1 - i,
                    f"{value:.1f}s",
                    ha="center",
                    va="center",
                    color="white",
                )
            left[i] += value
    ax.set_yticks([1, 0], ["Baseline", "Fixed CacheWise"])
    ax.set_xlabel("Mean summed server request time per segment (seconds)")
    ax.set_title("Queue waiting dominates the measured session latency")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    ax.set_xlim(0, max(left) * 1.05)
    fig.suptitle(
        "Two repeats · ten segments · Qwen-14B · one RTX Pro 6000", fontsize=10
    )
    fig.savefig(root / "bottleneck-breakdown.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workload", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.workload)
