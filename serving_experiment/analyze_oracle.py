"""Audit the oracle diagnostic or consolidated scheduling/eviction experiment."""

import argparse
import csv
import gzip
import hashlib
import io
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
CONSOLIDATED_POLICIES = (
    "baseline",
    "eviction_only",
    "prefix_only_no_aging",
    "cachewise_no_aging",
    "cachewise_oracle_no_aging",
    "cachewise",
)


def policy_settings(policy):
    """Expected scheduler, eviction, predictor, and aging limit for each case."""
    scheduling = "fcfs" if policy in {"baseline", "eviction_only"} else "prefix_aware"
    eviction = "lru" if policy in {"baseline", "prefix_only_no_aging"} else "predictive"
    predictor = "oracle" if policy == "cachewise_oracle_no_aging" else "tool_name"
    return scheduling, eviction, predictor, 30 if policy == "cachewise" else 0


def normalize_parallel_entrypoint(command):
    """Ignore directory aliases after validating runtime and wrapper source."""
    assert Path(command[0]).name == "python"
    assert Path(command[1]).name == "server_queue_diagnostics.py"
    return ["python", "server_queue_diagnostics.py", *command[2:]]


def analyze(root, smoke=False, consolidated=False, repeat_index=None, parallel=False):
    assert not parallel or (consolidated and not smoke and repeat_index is None)
    policies = CONSOLIDATED_POLICIES if consolidated else POLICIES
    assert (root / "COMPLETE").exists(), "Experiment incomplete"
    repeats, sessions = (1, 2) if smoke else (3, 10)
    indices = range(repeats) if repeat_index is None else [repeat_index]
    assert repeat_index is None or (not smoke and repeat_index >= 0)
    expected = {f"{p}-r{r}" for p in policies for r in indices}
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
    summarize(root, reference_policy=policies[0])
    rows = json.loads((root / "comparison.json").read_text())
    uuids, common_commands, trace_hashes = set(), [], {}
    gpu_by_repeat = {}
    if parallel:
        hardware = []
        for repeat in range(3):
            source = root / f"source-r{repeat}"
            env = json.loads((source / "environment.json").read_text())
            devices = list(
                csv.DictReader(io.StringIO(env["gpu"]), skipinitialspace=True)
            )
            assert len(devices) == 1
            hardware.append(
                [
                    env["vllm_revision"],
                    env["model"],
                    env["python"],
                    devices[0]["name"],
                    devices[0]["driver_version"],
                    devices[0]["memory.total [MiB]"],
                ]
            )
            for name in (
                "workload.json",
                "bootstrap.jsonl",
                "vllm-source.patch",
                "packages.txt",
                "model_snapshot.txt",
                "runner_source/replay.py",
                "runner_source/server_queue_diagnostics.py",
                "runner_source/queue_observer.py",
            ):
                assert (source / name).read_bytes() == (
                    root / "source-r0" / name
                ).read_bytes(), name
        assert all(h == hardware[0] for h in hardware), (
            "GPU/runtime configuration differs across repeat blocks"
        )
    for row in rows:
        name, policy = row["run"], row["policy"]
        path = root / name
        oracle = policy == "cachewise_oracle_no_aging"
        scheduling, eviction, predictor, aging = policy_settings(policy)
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["selected_workload_sha256"] == digest(selected)
        assert manifest["source_workload_sha256"] == inputs["workload.json"]
        assert manifest["mark_terminal_idle"] and manifest["oracle"] == oracle
        assert manifest["delay_scale"] == 1
        command = json.loads((root / f"{name}-command.json").read_text())

        def option(flag, command=command):
            return command[command.index(flag) + 1]

        assert option("--scheduling-policy") == scheduling
        if scheduling == "prefix_aware":
            assert option("--prefix-aware-max-wait-s") == str(aging)
        assert option("--num-gpu-blocks-override") == "17981"
        assert option("--kv-cache-eviction-policy") == eviction
        if eviction == "predictive":
            assert option("--cachewise-predictor") == predictor
            assert Path(option("--cachewise-bootstrap-path")).name == "bootstrap.jsonl"
        for flag in (
            "--scheduling-policy",
            "--prefix-aware-max-wait-s",
            "--kv-cache-eviction-policy",
            "--cachewise-predictor",
            "--cachewise-bootstrap-path",
        ):
            if flag in command:
                i = command.index(flag)
                del command[i : i + 2]
        if parallel:
            command = normalize_parallel_entrypoint(command)
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
        run_uuids = {r["uuid"].strip() for r in gpu}
        assert len(run_uuids) == 1, "Each run must use exactly one physical GPU"
        uuids.update(run_uuids)
        repeat = name.rsplit("-r", 1)[1]
        gpu_by_repeat.setdefault(repeat, set()).update(run_uuids)
        queue_files = list(root.glob(f"{name}-queue-*.jsonl"))
        if scheduling == "prefix_aware":
            assert queue_files
            pops = [r for p in queue_files for r in records(p)]
            assert all(r["max_wait_s"] == aging for r in pops)
            if aging == 0:
                assert not any(r["aging_choice"] for r in pops)
            assert len({r["request_id"] for r in pops}) == len(turns)
            row["queue_pop_count"] = len(pops)
            row["aging_pop_count"] = sum(r["aging_choice"] for r in pops)
        else:
            assert not queue_files, "FCFS case unexpectedly used a prefix-aware queue"
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
    assert all(len(values) == 1 for values in gpu_by_repeat.values())
    assert parallel or len(uuids) == 1
    assert all(c == common_commands[0] for c in common_commands)
    aggregate = {}
    for policy in policies:
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
    contrasts = [
        (POLICIES[0], POLICIES[1]),
        (POLICIES[0], POLICIES[2]),
        (POLICIES[1], POLICIES[2]),
    ]
    if consolidated:
        contrasts += [
            ("baseline", "eviction_only"),
            ("baseline", "prefix_only_no_aging"),
            ("baseline", "cachewise_no_aging"),
            ("eviction_only", "cachewise_no_aging"),
            ("cachewise", "cachewise_no_aging"),
        ]
    ratios = {
        f"{a}_over_{b}": aggregate[a]["mean_session_llm_time_s"]
        / aggregate[b]["mean_session_llm_time_s"]
        for a, b in contrasts
    }
    report = {
        "scope": "Recorded tool-duration oracle, not an omniscient future-arrival/Belady policy. All policies mark terminal turns idle. One cohort; repeat ranges are not confidence intervals.",
        "smoke": smoke,
        "study": "consolidated" if consolidated else "oracle",
        "comparison_design": "All six policies share one GPU within each repeat; repeat blocks run on separate GPUs."
        if parallel
        else "All runs share one physical GPU.",
        "physical_gpu_by_repeat": {k: sorted(v) for k, v in gpu_by_repeat.items()},
        "policies": aggregate,
        "time_ratios": ratios,
        "paired_time_ratios": {
            f"{a}_over_{b}": [
                next(
                    r["mean_session_llm_time_s"]
                    for r in rows
                    if r["run"] == f"{a}-r{repeat}"
                )
                / next(
                    r["mean_session_llm_time_s"]
                    for r in rows
                    if r["run"] == f"{b}-r{repeat}"
                )
                for repeat in indices
            ]
            for a, b in contrasts
        },
        "runs": rows,
        "audit": audit,
        "input_hashes": inputs,
        "physical_gpu_uuids": sorted(uuids),
        "request_trace_hashes": trace_hashes,
    }
    filename = (
        "consolidated-comparison.json" if consolidated else "oracle-comparison.json"
    )
    (root / filename).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"policies": aggregate, "time_ratios": ratios}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--consolidated", action="store_true")
    parser.add_argument("--repeat-index", type=int)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()
    analyze(args.root, args.smoke, args.consolidated, args.repeat_index, args.parallel)
