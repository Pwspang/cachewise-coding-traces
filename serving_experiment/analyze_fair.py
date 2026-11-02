"""Validate and compare the matched five-policy, three-repeat experiment."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from analyze_bottlenecks import metric, records
from inspect_traces import audit_repeats
from summarize import summarize

POLICIES = {
    "baseline": ("fcfs", "lru"),
    "infercept": ("fcfs", "infercept"),
    "prefix_only_no_aging": ("prefix_aware", "lru"),
    "cachewise_no_aging": ("prefix_aware", "predictive"),
    "infercept_prefix_no_aging": ("prefix_aware", "infercept"),
}


def analyze(root):
    assert (root / "COMPLETE").exists(), "Experiment has not completed"
    expected = {f"{policy}-r{repeat}" for policy in POLICIES for repeat in range(3)}
    actual = {p.parent.name for p in root.glob("*-r*/manifest.json")}
    assert actual == expected, "Missing or unexpected policy/repeat"
    audit = audit_repeats(root)
    summarize(root)
    rows = {r["run"]: r for r in json.loads((root / "comparison.json").read_text())}
    inputs = json.loads((root / "prepared-inputs.json").read_text())
    for name, digest in inputs.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert (
        "+++ b/vllm/v1/core/cachewise/infercept.py"
        in (root / "vllm-source.patch").read_text()
    )
    commands, manifests = [], []
    for name, row in rows.items():
        path = root / name
        manifest = json.loads((path / "manifest.json").read_text())
        manifests.append(
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
        assert manifest["source_workload_sha256"] == inputs["workload.json"]
        command = json.loads((root / f"{name}-command.json").read_text())

        def option(flag, command=command):
            return command[command.index(flag) + 1]

        scheduling, eviction = POLICIES[row["policy"]]
        assert option("--scheduling-policy") == scheduling
        assert option("--kv-cache-eviction-policy") == eviction
        assert option("--num-gpu-blocks-override") == "17981"
        if eviction == "infercept":
            assert (
                Path(option("--infercept-recompute-profile")).name
                == "recompute-profile.json"
            )
        if eviction == "predictive":
            assert option("--cachewise-predictor") == "tool_name"
            assert Path(option("--cachewise-bootstrap-path")).name == "bootstrap.jsonl"
        if scheduling == "prefix_aware":
            assert option("--prefix-aware-max-wait-s") == "0"
            files = list(root.glob(f"{name}-queue-*.jsonl"))
            assert files, "Missing queue observer output"
            decisions = [r for file in files for r in records(file)]
            assert all(
                r["max_wait_s"] == 0 and not r["aging_choice"] for r in decisions
            )
            assert len({r["request_id"] for r in decisions}) == row["requests"]
            row["queue_pop_count"] = len(decisions)
            row["queue_observer_time_s"] = sum(
                records(file)[-1]["observer_seconds"] for file in files
            )
        for flag in (
            "--scheduling-policy",
            "--kv-cache-eviction-policy",
            "--prefix-aware-max-wait-s",
            "--cachewise-predictor",
            "--cachewise-bootstrap-path",
            "--infercept-recompute-profile",
        ):
            if flag in command:
                index = command.index(flag)
                del command[index : index + 2]
        commands.append(command)
        snapshots = records(path / "metrics.jsonl")
        assert all(r.get("status") == 200 for r in snapshots)
        final = snapshots[-1]["text"]
        assert metric(final, "e2e_request_latency_seconds_count") == row["requests"]
        requests = [
            r for r in records(path / "requests.jsonl") if r["event"] == "request_end"
        ]
        uncached = sum(
            r["usage"]["prompt_tokens"]
            - r["usage"]["prompt_tokens_details"]["cached_tokens"]
            for r in requests
        )
        assert metric(final, "request_prefill_kv_computed_tokens_sum") == uncached
        row["uncached_prompt_tokens"] = uncached
        row["preemptions"] = metric(final, "num_preemptions_total")
        row["max_request_latency_s"] = max(r["latency_s"] for r in requests)
        row["mean_phase_residence_time_per_segment_s"] = {
            phase: metric(final, f"request_{phase}_time_seconds_sum") / row["sessions"]
            for phase in ("queue", "prefill", "decode")
        }
    assert all(c == commands[0] for c in commands)
    assert all(m == manifests[0] for m in manifests)
    aggregate = {}
    for policy in POLICIES:
        group = [r for r in rows.values() if r["policy"] == policy]
        aggregate[policy] = {
            field: statistics.mean(r[field] for r in group)
            for field in (
                "mean_session_llm_time_s",
                "elapsed_s",
                "prompt_cache_hit_fraction",
                "uncached_prompt_tokens",
            )
        }
        aggregate[policy]["observed_mean_time_range_s"] = [
            min(r["mean_session_llm_time_s"] for r in group),
            max(r["mean_session_llm_time_s"] for r in group),
        ]
    contrasts = {}
    for a, b in (
        ("baseline", "infercept"),
        ("baseline", "cachewise_no_aging"),
        ("baseline", "prefix_only_no_aging"),
        ("infercept", "cachewise_no_aging"),
        ("infercept", "infercept_prefix_no_aging"),
        ("prefix_only_no_aging", "cachewise_no_aging"),
        ("prefix_only_no_aging", "infercept_prefix_no_aging"),
        ("infercept_prefix_no_aging", "cachewise_no_aging"),
    ):
        paired = [
            rows[f"{a}-r{i}"]["mean_session_llm_time_s"]
            / rows[f"{b}-r{i}"]["mean_session_llm_time_s"]
            for i in range(3)
        ]
        contrasts[f"{a}_over_{b}"] = {
            "ratio_of_means": aggregate[a]["mean_session_llm_time_s"]
            / aggregate[b]["mean_session_llm_time_s"],
            "paired_ratios": paired,
        }
    result = {
        "status": "complete",
        "policies": aggregate,
        "contrasts": contrasts,
        "runs": rows,
        "trace_audit": audit,
        "prepared_inputs_sha256": inputs,
        "recompute_profile_sha256": hashlib.sha256(
            (root / "recompute-profile.json").read_bytes()
        ).hexdigest(),
        "engine_patch_sha256": hashlib.sha256(
            (root / "vllm-source.patch").read_bytes()
        ).hexdigest(),
        "common_server_arguments_verified": True,
        "limitation": "Three rotated repeats on one physical GPU and selected workload. Ranges are observed repeats, not confidence intervals. Prefix-aware cases disable aging. InferCept is the preserve/discard adaptation without swap. Phase times overlap and are not exclusive GPU/CPU costs.",
    }
    (root / "fair-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(root, aggregate)
    print(json.dumps({"policies": aggregate, "contrasts": contrasts}, indent=2))


def plot(root, aggregate):
    import matplotlib.pyplot as plt

    names = [
        "FCFS + LRU",
        "FCFS + InferCept",
        "Prefix + LRU",
        "Prefix + CacheWise",
        "Prefix + InferCept",
    ]
    rows = [aggregate[p] for p in POLICIES]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for i, row in enumerate(rows):
        mean = row["mean_session_llm_time_s"]
        low, high = row["observed_mean_time_range_s"]
        ax.barh(
            i, mean, color=["#64748b", "#059669", "#d97706", "#2563eb", "#059669"][i]
        )
        ax.errorbar(
            mean, i, xerr=[[mean - low], [high - mean]], color="black", capsize=3
        )
        ax.text(high + 5, i, f"{mean:.1f}s", va="center")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(r["observed_mean_time_range_s"][1] for r in rows) * 1.15)
    ax.set_xlabel("Mean summed LLM time per segment (seconds; lower is better)")
    ax.set_title(
        "Same GPU and workload; three repeats; prefix-aware aging disabled\nWhiskers show observed repeat range"
    )
    fig.savefig(root / "fair-comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root)
