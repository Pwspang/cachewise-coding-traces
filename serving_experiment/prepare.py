"""Prepare a deterministic, sanitized length-and-timing replay workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def human_boundary_messages(raw_path):
    """Find assistant turns followed by actual user input, excluding tool results."""
    previous, boundaries = None, set()
    for line in raw_path.read_text().splitlines():
        entry = json.loads(line)
        message = entry.get("message", {})
        if entry.get("type") == "assistant":
            previous = message.get("id", previous)
        elif entry.get("type") == "user" and previous is not None:
            content = message.get("content", [])
            human_input = (isinstance(content, str) and bool(content)) or (
                isinstance(content, list)
                and any(
                    isinstance(item, dict) and item.get("type") != "tool_result"
                    for item in content
                )
            )
            if human_input:
                boundaries.add(previous)
    return boundaries


def prepare(
    root: Path, output: Path, seed=42, max_model_len=131072, split_human_pauses=True
):
    output.mkdir(parents=True, exist_ok=False)
    train_path = root / "tool_duration_prediction/datasets/splits/train_sessions.json"
    train = set(json.loads(train_path.read_text())["train_sessions"])
    bootstrap, sessions, audit = [], [], defaultdict(int)
    source_hashes = {}
    for path in sorted((root / "parsed_traces").glob("*/*/events.json")):
        source_hashes[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        events = json.loads(path.read_text())
        sid = path.parent.name
        human_boundaries = set()
        if split_human_pauses:
            raw_path = root / "raw_traces" / path.parent.parent.name / f"{sid}.jsonl"
            human_boundaries = human_boundary_messages(raw_path)
            source_hashes[str(raw_path.relative_to(root))] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        groups, tools = {}, defaultdict(list)
        for event in events:
            if event["event_type"] == "tool_call":
                tools[event["associated_llm_call_id"]].append(event)
                duration = event.get("execution_time_ms")
                if sid in train and duration is not None and duration >= 0:
                    bootstrap.append(
                        {
                            "tool": event["tool_name"],
                            "args": json.dumps(
                                event.get("tool_input", {}), sort_keys=True
                            ),
                            "duration_s": duration / 1000,
                        }
                    )
            elif event["event_type"] == "llm_call":
                audit["llm_records"] += 1
                groups.setdefault(event["message_id"], []).append(event)
        ordered = sorted(
            groups.items(), key=lambda item: timestamp(item[1][0]["timestamp"])
        )
        audit["unique_messages"] += len(ordered)
        if sid in train:
            continue
        segment, segment_id = [], 0

        def flush(sid=sid, path=path):
            nonlocal segment, segment_id
            if segment:
                segment[-1]["delay_after_s"] = 0
                sessions.append(
                    {
                        "id": f"{sid}_{segment_id:03d}",
                        "source_session": sid,
                        "project": path.parent.parent.name,
                        "turns": segment,
                    }
                )
                segment_id += 1
                segment = []

        for index, (mid, records) in enumerate(ordered):
            if any(r.get("model") == "<synthetic>" for r in records):
                audit["synthetic_messages_skipped"] += 1
                flush()
                continue
            prompt = max(
                sum(
                    r.get(k) or 0
                    for k in (
                        "input_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    )
                )
                for r in records
            )
            decode = max(r.get("output_tokens") or 0 for r in records)
            calls = tools[mid]
            if prompt <= 0 or decode <= 0 or prompt + decode > max_model_len:
                audit["invalid_or_overlength_messages_skipped"] += 1
                flush()
                continue
            if any(t.get("execution_time_ms") is None for t in calls):
                audit["messages_with_unknown_tool_duration_skipped"] += 1
                flush()
                continue
            delay = 0.0
            boundary = False
            if calls:
                # Concurrent tools finish at the latest result; do not sum durations.
                delay = max(timestamp(t["result_timestamp"]) for t in calls) - min(
                    timestamp(t["timestamp"]) for t in calls
                )
            elif index + 1 < len(ordered):
                delay = max(
                    0,
                    timestamp(ordered[index + 1][1][0]["timestamp"])
                    - max(timestamp(r["timestamp"]) for r in records),
                )
                boundary = delay >= 300 if not split_human_pauses else False
            if mid in human_boundaries:
                boundary = True
            if (
                segment
                and prompt < segment[-1]["prompt_tokens"] + segment[-1]["output_tokens"]
            ):
                audit["context_reset_boundaries"] += 1
                flush()
            segment.append(
                {
                    "message_id": mid,
                    "request_ids": sorted({r.get("request_id") or "" for r in records}),
                    "prompt_tokens": prompt,
                    "output_tokens": decode,
                    "delay_after_s": max(0, delay),
                    "next_tools": [
                        {
                            "name": t["tool_name"],
                            "args": json.dumps(t.get("tool_input", {}), sort_keys=True),
                        }
                        for t in calls
                    ],
                }
            )
            if boundary:
                audit["human_pause_boundaries"] += 1
                flush()
        flush()
    random.Random(seed).shuffle(sessions)
    manifest = {
        "seed": seed,
        "max_model_len": max_model_len,
        "training_sessions": sorted(train),
        "source_sha256": source_hashes,
        "audit": dict(audit),
        "human_pause_policy": "raw_user_boundaries"
        if split_human_pauses
        else "no_tool_gap_300s",
        "methodology": "Synthetic token IDs preserving within-segment prefix growth; max output usage per message; tools overlap until latest result; "
        + (
            "raw user-input boundaries"
            if split_human_pauses
            else "non-tool gaps >=300s"
        )
        + " and context shrink split segments. Redacted text is not reconstructed. Published split retained; not an exact paper reproduction.",
        "sessions": sessions,
    }
    (output / "workload.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "bootstrap.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in bootstrap)
    )
    print(
        json.dumps(
            {
                "evaluation_segments": len(sessions),
                "evaluation_turns": sum(len(s["turns"]) for s in sessions),
                "bootstrap_samples": len(bootstrap),
                "audit": dict(audit),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-model-len", type=int, default=131072)
    boundary_args = p.add_mutually_exclusive_group()
    boundary_args.add_argument(
        "--split-human-pauses",
        action="store_true",
        default=True,
        help="Split at actual user messages (default); tool results are excluded.",
    )
    boundary_args.add_argument(
        "--legacy-human-pauses",
        dest="split_human_pauses",
        action="store_false",
        help="Reproduce the old 300-second heuristic for historical comparisons only.",
    )
    a = p.parse_args()
    prepare(a.root, a.output, a.seed, a.max_model_len, a.split_human_pauses)
