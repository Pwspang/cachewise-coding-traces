"""Audit the matched LRU / learned / tool-duration-oracle experiment."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from inspect_traces import audit_repeats, metric, records
from replay import digest, replay_hints, select_sessions
from summarize import summarize

POLICIES = (
    "prefix_only_no_aging",
    "cachewise_no_aging",
    "cachewise_oracle_no_aging",
)


def analyze(root, smoke=False):
    assert (root / "COMPLETE").exists(), "Experiment incomplete"
    repeats, sessions = (1, 2) if smoke else (3, 10)
    expected = {f"{p}-r{r}" for p in POLICIES for r in range(repeats)}
    assert {p.parent.name for p in root.glob("*-r*/manifest.json")} == expected
    inputs = json.loads((root / "prepared-inputs.json").read_text())
    for name, sha in inputs.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
    workload = json.loads((root / "workload.json").read_text())
    selected = select_sessions(
        workload, sessions, 2 if smoke else 0, 32768 if smoke else 0
    )
    turns = {
        f"{s['id']}:{i}": (t, i + 1 == len(s["turns"]))
        for s in selected
        for i, t in enumerate(s["turns"])
    }
    if not smoke:
        assert len(turns) == 83
    audit = audit_repeats(root)
    summarize(root, reference_policy=POLICIES[0])
    rows = json.loads((root / "comparison.json").read_text())
    uuids, common_commands, trace_hashes = set(), [], {}
    for row in rows:
        name, policy = row["run"], row["policy"]
        path = root / name
        oracle = policy == POLICIES[2]
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["selected_workload_sha256"] == digest(selected)
        assert manifest["source_workload_sha256"] == inputs["workload.json"]
        assert manifest["mark_terminal_idle"] and manifest["oracle"] == oracle
        assert manifest["delay_scale"] == 1
        command = json.loads((root / f"{name}-command.json").read_text())

        def option(flag, command=command):
            return command[command.index(flag) + 1]

        assert option("--scheduling-policy") == "prefix_aware"
        assert option("--prefix-aware-max-wait-s") == "0"
        assert option("--num-gpu-blocks-override") == "17981"
        assert option("--kv-cache-eviction-policy") == (
            "lru" if policy == POLICIES[0] else "predictive"
        )
        if policy != POLICIES[0]:
            assert option("--cachewise-predictor") == (
                "oracle" if oracle else "tool_name"
            )
            assert Path(option("--cachewise-bootstrap-path")).name == "bootstrap.jsonl"
        for flag in (
            "--kv-cache-eviction-policy",
            "--cachewise-predictor",
            "--cachewise-bootstrap-path",
        ):
            if flag in command:
                i = command.index(flag)
                del command[i : i + 2]
        common_commands.append(command)
        hinted_durations, seen = [], set()
        with gzip.open(path / "payloads.jsonl.gz", "rt") as handle:
            for line in handle:
                record = json.loads(line)
                key = record["request_key"]
                assert key not in seen
                seen.add(key)
                turn, terminal = turns[key]
                hints = replay_hints(
                    turn,
                    terminal=terminal,
                    mark_terminal_idle=True,
                    oracle=oracle,
                    delay_scale=1,
                )
                assert record["body"]["vllm_xargs"] == hints
                if "cachewise_oracle_duration_s" in hints:
                    hinted_durations.append(hints["cachewise_oracle_duration_s"])
        assert seen == set(turns)
        server_log = (root / f"{name}-server.log").read_text()
        consumed = [
            float(s)
            for s in re.findall(
                r"CacheWise oracle duration: request_id=\S+ duration_s=([\d.]+)",
                server_log,
            )
        ]
        assert len(consumed) == len(hinted_durations)
        assert all(
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9)
            for actual, expected in zip(sorted(consumed), sorted(hinted_durations))
        ), "Engine did not consume the expected oracle hints"
        row["oracle_hints_consumed"] = len(consumed)
        with (root / f"{name}-gpu.csv").open() as handle:
            gpu = list(csv.DictReader(handle, skipinitialspace=True))
        assert gpu
        uuids.update(r["uuid"].strip() for r in gpu)
        queue_files = list(root.glob(f"{name}-queue-*.jsonl"))
        assert queue_files
        pops = [r for p in queue_files for r in records(p)]
        assert all(r["max_wait_s"] == 0 and not r["aging_choice"] for r in pops)
        assert len({r["request_id"] for r in pops}) == len(turns)
        snapshots = records(path / "metrics.jsonl")
        assert all(r.get("status") == 200 for r in snapshots)
        final = snapshots[-1]["text"]
        events = records(path / "requests.jsonl")
        starts = {r["request_key"]: r for r in events if r["event"] == "request_start"}
        ends = {r["request_key"]: r for r in events if r["event"] == "request_end"}
        uncached = sum(
            r["usage"]["prompt_tokens"]
            - r["usage"]["prompt_tokens_details"]["cached_tokens"]
            for r in ends.values()
        )
        assert metric(final, "vllm:request_prefill_kv_computed_tokens_sum") == uncached
        assert metric(final, "vllm:e2e_request_latency_seconds_count") == len(turns)
        row["uncached_prompt_tokens"] = uncached
        row["preemptions"] = metric(final, "vllm:num_preemptions_total")
        row["max_request_latency_s"] = max(r["latency_s"] for r in ends.values())
        # Tool-duration truth excludes transport and client bookkeeping. Measure
        # that residual instead of claiming exact future server-arrival times.
        residuals = []
        for s in selected:
            for i, turn in enumerate(s["turns"][:-1]):
                key, next_key = f"{s['id']}:{i}", f"{s['id']}:{i + 1}"
                actual_gap = (
                    starts[next_key]["elapsed_s"]
                    - ends[key]["start_s"]
                    - ends[key]["latency_s"]
                )
                residuals.append(actual_gap - turn["delay_after_s"])
        row["client_gap_minus_tool_delay_mean_s"] = statistics.mean(residuals)
        row["client_gap_minus_tool_delay_max_s"] = max(residuals)
        trace_hashes[name] = hashlib.sha256(
            (path / "requests.jsonl").read_bytes()
        ).hexdigest()
    assert len(uuids) == 1
    assert all(c == common_commands[0] for c in common_commands)
    aggregate = {}
    for policy in POLICIES:
        group = [r for r in rows if r["policy"] == policy]
        aggregate[policy] = {
            key: statistics.mean(r[key] for r in group)
            for key in (
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
    ratios = {
        f"{a}_over_{b}": aggregate[a]["mean_session_llm_time_s"]
        / aggregate[b]["mean_session_llm_time_s"]
        for a, b in (
            (POLICIES[0], POLICIES[1]),
            (POLICIES[0], POLICIES[2]),
            (POLICIES[1], POLICIES[2]),
        )
    }
    report = {
        "scope": "Recorded tool-duration oracle, not an omniscient future-arrival/Belady policy. All policies mark terminal turns idle. One cohort; repeat ranges are not confidence intervals.",
        "smoke": smoke,
        "policies": aggregate,
        "time_ratios": ratios,
        "runs": rows,
        "audit": audit,
        "input_hashes": inputs,
        "physical_gpu_uuids": sorted(uuids),
        "request_trace_hashes": trace_hashes,
    }
    (root / "oracle-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"policies": aggregate, "time_ratios": ratios}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    analyze(args.root, args.smoke)
