"""Compare paired CacheWise runs with and without the FCFS aging override."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_bottlenecks import metric, records


def analyze(root):
    rows = []
    expected = None
    commands = []
    for policy, threshold in (("cachewise", "30"), ("cachewise_no_aging", "0")):
        name = f"{policy}-r0"
        path = root / name
        manifest = json.loads((path / "manifest.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        assert manifest["status"] == summary["status"] == "complete"
        signature = [
            manifest[k]
            for k in ("selected_workload_sha256", "model", "sampling", "delay_scale")
        ]
        if expected is not None:
            assert signature == expected
        expected = signature
        command = json.loads((root / f"{name}-command.json").read_text())
        index = command.index("--prefix-aware-max-wait-s") + 1
        assert command[index] == threshold
        command[index] = "<threshold>"
        commands.append(command)
        requests = [
            r for r in records(path / "requests.jsonl") if r["event"] == "request_end"
        ]
        snapshots = [
            r for r in records(path / "metrics.jsonl") if r.get("status") == 200
        ]
        final = snapshots[-1]["text"]
        assert metric(final, "e2e_request_latency_seconds_count") == len(requests)
        decisions = [
            r for file in root.glob(f"{name}-queue-*.jsonl") for r in records(file)
        ]
        assert len(decisions) == len(requests), (
            "Unexpected queue reentries or missing observations"
        )
        assert len({r["request_id"] for r in decisions}) == len(requests)
        assert all(r["max_wait_s"] == float(threshold) for r in decisions)
        final_counters = [
            records(file)[-1] for file in root.glob(f"{name}-queue-*.jsonl")
        ]
        prompt = sum(r["usage"]["prompt_tokens"] for r in requests)
        cached = sum(
            r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in requests
        )
        rows.append(
            {
                "run": name,
                "threshold_s": float(threshold),
                "requests": len(requests),
                "mean_segment_llm_time_s": statistics.mean(
                    s["llm_time_s"] for s in summary["sessions"]
                ),
                "max_segment_llm_time_s": max(
                    s["llm_time_s"] for s in summary["sessions"]
                ),
                "wall_time_s": summary["elapsed_s"],
                "median_ttft_s": statistics.median(r["ttft_s"] for r in requests),
                "max_request_latency_s": max(r["latency_s"] for r in requests),
                "prompt_cache_hit_fraction": cached / prompt,
                "uncached_prompt_tokens": prompt - cached,
                "queue_pop_observations": len(decisions),
                "aging_selected_pops": sum(r["aging_choice"] for r in decisions),
                "fresh_selection_calls": sum(
                    r["fresh_choices"] for r in final_counters
                ),
                "aging_selection_calls": sum(
                    r["aging_choices"] for r in final_counters
                ),
                "observer_seconds_before_final_write": sum(
                    r["observer_seconds"] for r in final_counters
                ),
                "queue_time_sum_s": metric(final, "request_queue_time_seconds_sum"),
                "preemptions": metric(final, "num_preemptions_total"),
            }
        )
    assert commands[0] == commands[1], "Server arguments differ beyond the threshold"
    assert rows[1]["aging_selected_pops"] == rows[1]["aging_selection_calls"] == 0
    a, b = rows
    result = {
        "runs": rows,
        "mean_segment_llm_time_reduction_fraction": 1
        - b["mean_segment_llm_time_s"] / a["mean_segment_llm_time_s"],
        "enabled_over_disabled_time_ratio": a["mean_segment_llm_time_s"]
        / b["mean_segment_llm_time_s"],
        "server_arguments_differ_only_in_aging_threshold": True,
        "limitation": "One paired run on one GPU and selected workload, 30s first then disabled. Not a comparison to LRU, a load sweep, or a statistical replication. Observer counts aging-rule eligibility for selected choices without computing counterfactual prefix scores; a counted override need not change the winning request.",
    }
    (root / "aging-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root)
