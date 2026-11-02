"""Audit reproducibility gaps without changing the workload or training data."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from replay import select_sessions


def audit(root, prepared, output):
    workload_path = prepared / "workload.json"
    bootstrap_path = prepared / "bootstrap.jsonl"
    workload = json.loads(workload_path.read_text())
    bootstrap = [json.loads(line) for line in bootstrap_path.read_text().splitlines()]
    training = set(workload["training_sessions"])
    sources = list((root / "parsed_traces").glob("*/*/events.json"))
    bootstrap_keys = {r["tool"] for r in bootstrap}
    selected = select_sessions(workload, 10, 0, 0)
    transitions = [t for s in selected for t in s["turns"][:-1]]
    keys = [
        "+".join(sorted(t["name"] for t in turn["next_tools"])) for turn in transitions
    ]
    missing = Counter(key for key in keys if key not in bootstrap_keys)
    loads = {}
    for count in (10, 20, 30, 40, 50):
        sessions = select_sessions(workload, count, 0, 0)
        delays = [t["delay_after_s"] for s in sessions for t in s["turns"][:-1]]
        loads[count] = {
            "segments": len(sessions),
            "source_sessions": len({s["source_session"] for s in sessions}),
            "requests": sum(len(s["turns"]) for s in sessions),
            "initial_prompt_tokens": sum(
                s["turns"][0]["prompt_tokens"] for s in sessions
            ),
            "output_tokens": sum(
                t["output_tokens"] for s in sessions for t in s["turns"]
            ),
            "external_delay_sum_s": sum(delays),
            "max_external_delay_s": max(delays),
            "delays_over_300s": [
                {
                    "segment": s["id"],
                    "message": t["message_id"],
                    "delay_s": t["delay_after_s"],
                }
                for s in sessions
                for t in s["turns"][:-1]
                if t["delay_after_s"] > 300
            ],
        }
    result = {
        "workload_sha256": hashlib.sha256(workload_path.read_bytes()).hexdigest(),
        "bootstrap_sha256": hashlib.sha256(bootstrap_path.read_bytes()).hexdigest(),
        "parsed_source_sessions": len(sources),
        "training_source_sessions": len(training),
        "training_fraction_of_parsed_sources": len(training) / len(sources),
        "prepared_segments": len(workload["sessions"]),
        "eligible_segments_at_least_two_turns": sum(
            len(s["turns"]) >= 2 for s in workload["sessions"]
        ),
        "preparation_audit": workload["audit"],
        "bootstrap_samples": len(bootstrap),
        "unique_bootstrap_argument_strings": len({r["args"] for r in bootstrap}),
        "all_bootstrap_arguments_are_redacted": all(
            json.loads(r["args"]) == {"content": "<redacted>"} for r in bootstrap
        ),
        "bootstrap_durations_over_3600s": sum(
            r["duration_s"] > 3600 for r in bootstrap
        ),
        "ten_segment_transitions": len(transitions),
        "transitions_at_most_1s": sum(t["delay_after_s"] <= 1 for t in transitions),
        "missing_initial_bootstrap_key_count": sum(missing.values()),
        "missing_initial_bootstrap_keys": dict(missing),
        "load_characteristics": loads,
        "limitations": [
            "The paper's exact serving split and selected sessions are unavailable here; the denominator is this repository's parsed source sessions.",
            "Missing keys describe bootstrap coverage, before online learning; they do not measure prediction error or latency impact.",
            "Redaction prevents reconstruction of argument-specific tool distributions. Raising the cluster count cannot restore missing information.",
            "Long recorded tool intervals are preserved, not clipped or dropped. The 178164.206-second interval also exists in the raw tool-use/result timestamps.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "load_characteristics"}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.prepared, args.output)
