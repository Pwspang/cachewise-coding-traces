#!/usr/bin/env python3
"""Compare release/parsed_traces events.json against example_logs originals."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LLM_FIELDS = [
    "timestamp",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "stop_reason",
    "has_tool_use",
    "num_tools_called",
]
TOOL_FIELDS = [
    "timestamp",
    "tool_name",
    "has_result",
    "is_error",
    "execution_time_ms",
    "result_timestamp",
]


def orig_events_path(meta: dict) -> Path:
    project = meta["source_project"]
    session = meta["source_session_id"]
    if project == "_root":
        return ROOT / "example_logs" / session / "events.json"
    return ROOT / "example_logs" / project / session / "events.json"


def summarize(events: list[dict]) -> tuple[int, int, int]:
    llm = sum(1 for e in events if e.get("event_type") == "llm_call")
    tool = sum(1 for e in events if e.get("event_type") == "tool_call")
    return len(events), llm, tool


def main() -> None:
    session_maps = json.loads((ROOT / "release/id_maps/session_maps.json").read_text())
    parsed_manifest = json.loads((ROOT / "release/parsed_traces/manifest.json").read_text())

    rows: list[dict] = []
    missing_orig: list[tuple[str, str]] = []

    for entry in parsed_manifest["sessions"]:
        sid = entry["session_id"]
        meta = session_maps[sid]
        orig_path = orig_events_path(meta)
        new_path = ROOT / entry["output_path"]
        if not orig_path.exists():
            missing_orig.append((sid, str(orig_path)))
            continue

        orig = json.loads(orig_path.read_text())
        new = json.loads(new_path.read_text())
        o_total, o_llm, o_tool = summarize(orig)
        n_total, n_llm, n_tool = summarize(new)
        rows.append(
            {
                "sid": sid,
                "orig_path": str(orig_path.relative_to(ROOT)),
                "o_total": o_total,
                "n_total": n_total,
                "o_llm": o_llm,
                "n_llm": n_llm,
                "o_tool": o_tool,
                "n_tool": n_tool,
                "d_llm": n_llm - o_llm,
                "d_tool": n_tool - o_tool,
                "meta": meta,
                "output_path": entry["output_path"],
            }
        )

    perfect = [r for r in rows if r["d_llm"] == 0 and r["d_tool"] == 0]
    llm_only = [r for r in rows if r["d_llm"] != 0 and r["d_tool"] == 0]

    print("=== OVERVIEW ===")
    print(f"Comparable sessions: {len(rows)} / {parsed_manifest['num_sessions']}")
    print(f"Missing original events.json: {len(missing_orig)}")
    if missing_orig:
        for sid, path in missing_orig:
            print(f"  {sid}: {path}")

    print("\n=== AGGREGATE COUNTS (mapped sessions only) ===")
    print(
        f"Original: llm={sum(r['o_llm'] for r in rows)}, "
        f"tool={sum(r['o_tool'] for r in rows)}, "
        f"total={sum(r['o_total'] for r in rows)}"
    )
    print(
        f"Release:  llm={sum(r['n_llm'] for r in rows)}, "
        f"tool={sum(r['n_tool'] for r in rows)}, "
        f"total={sum(r['n_total'] for r in rows)}"
    )

    print("\n=== MATCH SUMMARY ===")
    print(f"Perfect (llm + tool counts match): {len(perfect)}")
    print(f"LLM-only diff (tool counts still match): {len(llm_only)}")
    print(f"Tool count differs: {sum(1 for r in rows if r['d_tool'] != 0)}")

    print("\nLLM delta distribution (new - orig):")
    print(dict(sorted(Counter(r["d_llm"] for r in rows).items())))

    field_issues = 0
    for r in perfect:
        orig = json.loads((ROOT / r["orig_path"]).read_text())
        new = json.loads((ROOT / r["output_path"]).read_text())
        for o, n in zip(orig, new, strict=True):
            keys = LLM_FIELDS if o["event_type"] == "llm_call" else TOOL_FIELDS
            for k in keys:
                if o.get(k) != n.get(k):
                    field_issues += 1
                    break
    print(f"\nPerfect sessions comparable-field mismatches: {field_issues}")

    print("\n=== SUBSET CHECK (sessions with fewer LLM calls) ===")
    subset_ok = 0
    for r in llm_only:
        orig = json.loads((ROOT / r["orig_path"]).read_text())
        new = json.loads((ROOT / r["output_path"]).read_text())
        o_tools = [
            (e["timestamp"], e["tool_name"], e.get("has_result"), e.get("execution_time_ms"))
            for e in orig
            if e["event_type"] == "tool_call"
        ]
        n_tools = [
            (e["timestamp"], e["tool_name"], e.get("has_result"), e.get("execution_time_ms"))
            for e in new
            if e["event_type"] == "tool_call"
        ]
        o_llm = {
            e["timestamp"]: e
            for e in orig
            if e["event_type"] == "llm_call"
        }
        n_llm = {
            e["timestamp"]: e
            for e in new
            if e["event_type"] == "llm_call"
        }
        tools_match = o_tools == n_tools
        shared = set(o_llm) & set(n_llm)
        llm_fields_match = all(
            all(o_llm[ts].get(k) == n_llm[ts].get(k) for k in LLM_FIELDS)
            for ts in shared
        )
        if tools_match and llm_fields_match and set(n_llm) <= set(o_llm):
            subset_ok += 1
    print(f"  {subset_ok}/{len(llm_only)} llm-diff sessions are exact subsets of originals")
    outlier_sids = []
    for r in llm_only:
        orig = json.loads((ROOT / r["orig_path"]).read_text())
        new = json.loads((ROOT / r["output_path"]).read_text())
        o_tools = [
            (e["timestamp"], e["tool_name"], e.get("has_result"), e.get("execution_time_ms"))
            for e in orig
            if e["event_type"] == "tool_call"
        ]
        n_tools = [
            (e["timestamp"], e["tool_name"], e.get("has_result"), e.get("execution_time_ms"))
            for e in new
            if e["event_type"] == "tool_call"
        ]
        o_llm = {e["timestamp"]: e for e in orig if e["event_type"] == "llm_call"}
        n_llm = {e["timestamp"]: e for e in new if e["event_type"] == "llm_call"}
        shared = set(o_llm) & set(n_llm)
        llm_fields_match = all(
            all(o_llm[ts].get(k) == n_llm[ts].get(k) for k in LLM_FIELDS)
            for ts in shared
        )
        if not (o_tools == n_tools and llm_fields_match and set(n_llm) <= set(o_llm)):
            outlier_sids.append(r["sid"])
    if outlier_sids:
        print(f"  Outliers (not exact subsets): {outlier_sids}")

    print("\n=== SPOT CHECK session_0001 (perfect match session) ===")
    r = next(x for x in rows if x["sid"] == "session_0001")
    orig = json.loads((ROOT / r["orig_path"]).read_text())
    new = json.loads((ROOT / r["output_path"]).read_text())
    print(f"  counts: orig={r['o_total']} new={r['n_total']}")
    print(f"  first llm_call match: {orig[0]['timestamp'] == new[0]['timestamp']}")
    print(f"  all events byte-identical on comparable fields: ", end="")
    identical = True
    for o, n in zip(orig, new, strict=True):
        keys = LLM_FIELDS if o["event_type"] == "llm_call" else TOOL_FIELDS
        if any(o.get(k) != n.get(k) for k in keys):
            identical = False
            break
    print(identical)


if __name__ == "__main__":
    main()
