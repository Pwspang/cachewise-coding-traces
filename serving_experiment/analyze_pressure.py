"""Audit the prespecified five-run scheduling/eviction/capacity ablation."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from analyze_bottlenecks import metric, records
from inspect_traces import audit_repeats


def tool_wait_overlap(path):
    changes = []
    for event in json.loads(path.read_text())["traceEvents"]:
        if event["name"] == "Tool/human delay" and event["args"]["tools"]:
            changes.extend(
                [
                    (event["ts"] / 1e6, 1),
                    ((event["ts"] + event["dur"]) / 1e6, -1),
                ]
            )
    active, previous = 0, 0
    durations = {}
    for time, delta in sorted(changes):
        durations[active] = durations.get(active, 0) + time - previous
        active += delta
        previous = time
    assert active == 0
    return {
        "any_tool_wait_s": sum(v for k, v in durations.items() if k >= 1),
        "at_least_two_sessions_waiting_for_tools_s": sum(
            v for k, v in durations.items() if k >= 2
        ),
    }


def analyze(root):
    rows, signatures, command_bases, patches = {}, [], [], []
    request_sets = {}
    expected_requests = None
    for capacity, policies, blocks in (
        ("full", ("baseline", "prefix_only_no_aging", "cachewise_no_aging"), 17981),
        ("half", ("cachewise_no_aging", "baseline"), 8990),
    ):
        group = root / capacity
        assert (group / "COMPLETE").exists(), f"Incomplete group: {capacity}"
        audit_repeats(group)
        patches.append(
            hashlib.sha256((group / "vllm-source.patch").read_bytes()).hexdigest()
        )
        for policy in policies:
            name = f"{policy}-r0"
            path = group / name
            manifest = json.loads((path / "manifest.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
            assert manifest["status"] == summary["status"] == "complete"
            signatures.append(
                [
                    manifest[k]
                    for k in (
                        "selected_workload_sha256",
                        "source_workload_sha256",
                        "model",
                        "sampling",
                        "delay_scale",
                    )
                ]
            )
            events = records(path / "requests.jsonl")
            requests = [r for r in events if r["event"] == "request_end"]
            request_sets[f"{capacity}/{name}"] = {r["request_key"]: r for r in requests}
            request_signature = {
                r["request_key"]: (
                    r["prompt_sha256"],
                    r["prompt_tokens"],
                    r["output_tokens"],
                )
                for r in events
                if r["event"] == "request_start"
            }
            if expected_requests is not None:
                assert request_signature == expected_requests, (
                    "Inputs differ across capacities"
                )
            expected_requests = request_signature
            snapshots = records(path / "metrics.jsonl")
            assert all(r.get("status") == 200 for r in snapshots)
            final = snapshots[-1]["text"]
            assert metric(final, "e2e_request_latency_seconds_count") == len(requests)
            phases = {
                phase: metric(final, f"request_{phase}_time_seconds_sum")
                for phase in ("queue", "prefill", "decode")
            }
            for phase in phases:
                assert metric(final, f"request_{phase}_time_seconds_count") == len(
                    requests
                )
            total = metric(final, "e2e_request_latency_seconds_sum")
            phases["unattributed"] = total - sum(phases.values())
            assert phases["unattributed"] >= -0.001
            prompt = sum(r["usage"]["prompt_tokens"] for r in requests)
            cached = sum(
                r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in requests
            )
            assert (
                metric(final, "request_prefill_kv_computed_tokens_sum")
                == prompt - cached
            )
            command = json.loads((group / f"{name}-command.json").read_text())
            assert command[command.index("--num-gpu-blocks-override") + 1] == str(
                blocks
            )
            assert command[command.index("--scheduling-policy") + 1] == (
                "fcfs" if policy == "baseline" else "prefix_aware"
            )
            assert command[command.index("--kv-cache-eviction-policy") + 1] == (
                "predictive" if policy == "cachewise_no_aging" else "lru"
            )
            if policy != "baseline":
                assert command[command.index("--prefix-aware-max-wait-s") + 1] == "0"
            # These are the only allowed differences in this controlled experiment.
            for option in (
                "--num-gpu-blocks-override",
                "--scheduling-policy",
                "--kv-cache-eviction-policy",
                "--prefix-aware-max-wait-s",
                "--cachewise-predictor",
                "--cachewise-bootstrap-path",
            ):
                if option in command:
                    index = command.index(option)
                    del command[index : index + 2]
            command_bases.append(command)
            decisions = [
                r for file in group.glob(f"{name}-queue-*.jsonl") for r in records(file)
            ]
            if policy != "baseline":
                assert len(decisions) == len(requests)
                assert len({r["request_id"] for r in decisions}) == len(requests)
                assert not any(r["aging_choice"] for r in decisions)
            rows[f"{capacity}/{name}"] = {
                "capacity": capacity,
                "policy": policy,
                "kv_blocks": blocks,
                "requests": len(requests),
                "segments": len(summary["sessions"]),
                "mean_segment_llm_time_s": statistics.mean(
                    s["llm_time_s"] for s in summary["sessions"]
                ),
                "wall_time_s": summary["elapsed_s"],
                "median_ttft_s": statistics.median(r["ttft_s"] for r in requests),
                "max_request_latency_s": max(r["latency_s"] for r in requests),
                "prompt_cache_hit_fraction": cached / prompt,
                "uncached_prompt_tokens": prompt - cached,
                "phase_time_per_segment_s": {
                    k: v / len(summary["sessions"]) for k, v in phases.items()
                },
                "preemptions": metric(final, "num_preemptions_total"),
                "sampled_evictions": metric(final, "kv_block_lifetime_seconds_count"),
                "metric_snapshots": len(snapshots),
                "tool_wait_overlap": tool_wait_overlap(path / "timeline.json"),
            }
    assert all(s == signatures[0] for s in signatures)
    assert all(c == command_bases[0] for c in command_bases)
    assert len(set(patches)) == 1
    contrasts = {}
    for label, control, intervention in (
        ("full_cache_total", "full/baseline-r0", "full/cachewise_no_aging-r0"),
        ("full_cache_scheduling", "full/baseline-r0", "full/prefix_only_no_aging-r0"),
        (
            "full_cache_predictive_eviction",
            "full/prefix_only_no_aging-r0",
            "full/cachewise_no_aging-r0",
        ),
        ("half_cache_total", "half/baseline-r0", "half/cachewise_no_aging-r0"),
    ):
        a, b = rows[control], rows[intervention]
        contrasts[label] = {
            "control": control,
            "intervention": intervention,
            "control_over_intervention_mean_time_ratio": a["mean_segment_llm_time_s"]
            / b["mean_segment_llm_time_s"],
            "mean_time_reduction_fraction": 1
            - b["mean_segment_llm_time_s"] / a["mean_segment_llm_time_s"],
            "uncached_token_reduction_fraction": 1
            - b["uncached_prompt_tokens"] / a["uncached_prompt_tokens"],
        }
    differences = []
    for key, lru in request_sets["full/prefix_only_no_aging-r0"].items():
        cw = request_sets["full/cachewise_no_aging-r0"][key]
        lru_cached = lru["usage"]["prompt_tokens_details"]["cached_tokens"]
        cw_cached = cw["usage"]["prompt_tokens_details"]["cached_tokens"]
        differences.append(
            {
                "request": key,
                "lru_cached_tokens": lru_cached,
                "cachewise_cached_tokens": cw_cached,
                "cachewise_minus_lru_cached_tokens": cw_cached - lru_cached,
                "lru_latency_s": lru["latency_s"],
                "cachewise_latency_s": cw["latency_s"],
            }
        )
    result = {
        "runs": rows,
        "contrasts": contrasts,
        "identical_request_hashes_and_lengths_across_all_five_runs": True,
        "controlled_server_arguments_verified": True,
        "engine_patch_sha256": patches[0],
        "predictive_eviction_request_differences": differences,
        "limitations": "One run per case, fixed prespecified order. Half-capacity is a pressure intervention, not the paper's concurrency sweep. All prefix-aware cases disable aging. Phase times overlap and are not exclusive resource costs. Evictions are sampled, not exact totals. No InferCept or oracle baseline.",
    }
    (root / "pressure-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(root, rows)
    print(json.dumps(result, indent=2))


def plot(root, rows):
    import matplotlib.pyplot as plt

    labels = [
        "LRU + FCFS",
        "LRU + prefix scheduling",
        "CacheWise",
        "CacheWise",
        "LRU + FCFS",
    ]
    colors = ["#64748b", "#d97706", "#2563eb", "#2563eb", "#64748b"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    values = list(rows.values())
    for ax, indices, title in (
        (axes[0], range(3), "287,696-token KV capacity"),
        (axes[1], range(3, 5), "143,840-token KV capacity"),
    ):
        for pos, i in enumerate(indices):
            value = values[i]["mean_segment_llm_time_s"]
            ax.barh(pos, value, color=colors[i], height=0.6)
            ax.text(value + 5, pos, f"{value:.1f}s", va="center")
        ax.set_yticks(range(len(indices)), [labels[i] for i in indices])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Mean summed LLM time per segment (seconds)")
        ax.set_xlim(0, max(values[i]["mean_segment_llm_time_s"] for i in indices) * 1.2)
    fig.suptitle("Same ten segments; all prefix-aware cases disable the aging override")
    fig.savefig(root / "pressure-comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root)
