"""Relate engine eviction decisions to actual arrivals and idle sessions."""

import argparse
import json
from pathlib import Path


def analyze(root, policy):
    client = [
        json.loads(line)
        for line in (root / f"{policy}-r0/requests.jsonl").read_text().splitlines()
    ]
    keys = {
        r["response_id"]: r["request_key"]
        for r in client
        if r["event"] == "request_end"
    }

    def key(request_id):
        return next(
            (value for rid, value in keys.items() if request_id.startswith(rid)),
            request_id,
        )

    events = []
    for path in root.glob(f"{policy}-r0-decisions-*.jsonl"):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    arrivals = sorted(
        [r for r in events if r["event"] == "admitted" and r["sid"] is not None],
        key=lambda r: r["time"],
    )
    counts = {
        "total_evicted_block_events": 0,
        "evicted_while_next_request_already_waiting": 0,
        "evicted_within_3_seconds_before_next_arrival": 0,
    }
    examples = {"already_waiting": [], "imminent": []}
    for event in events:
        if event["event"] != "eviction":
            continue
        for sid, count in event["victims"].items():
            sid = int(sid)
            counts["total_evicted_block_events"] += count
            nxt = next(
                (r for r in arrivals if r["sid"] == sid and r["time"] > event["time"]),
                None,
            )
            if nxt is None:
                continue
            remaining = nxt["arrival_time"] - event["time"]
            category = (
                "already_waiting"
                if remaining < 0
                else "imminent"
                if remaining < 3
                else None
            )
            if category is None:
                continue
            field = (
                "evicted_while_next_request_already_waiting"
                if remaining < 0
                else "evicted_within_3_seconds_before_next_arrival"
            )
            counts[field] += count
            victim = next(s for s in event["sessions"] if s["sid"] == sid)
            human = [
                {
                    "last_request": key(s["request_id"]),
                    "priority_s": s["priority_s"],
                    "blocks": s["blocks"],
                }
                for s in event["sessions"]
                if s["tools"] == [] and not s["in_flight"] and s["sid"] != sid
            ]
            if human and len(examples[category]) < 20:
                examples[category].append(
                    {
                        "next_request": key(nxt["request_id"]),
                        "evicted_blocks": count,
                        "predicted_remaining_s": victim["priority_s"],
                        "actual_seconds_until_next_arrival": remaining,
                        "idle_human_sessions_retained": human,
                    }
                )
    result = {
        "policy": policy,
        "counts": counts,
        "examples": examples,
        "logging_seconds": max(
            (r.get("logging_seconds", 0) for r in events), default=0
        ),
        "note": "Counts are session-tagged block eviction events, not distinct physical blocks; unowned/default-segment evictions are not observed by this hook. A negative arrival interval means the request was already waiting; engine arrival timestamps are used.",
    }
    (root / f"{policy}-eviction-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("policy")
    args = parser.parse_args()
    analyze(args.root, args.policy)
