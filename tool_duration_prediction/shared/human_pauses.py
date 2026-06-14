"""HumanPause extraction"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def parse_timestamp(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def extract_human_pause_events(
    events: list[dict[str, Any]],
    *,
    min_duration_seconds: float = 2.0,
    source_file: str = "",
) -> list[dict[str, Any]]:
    """Build HumanPause tool_call records for gaps between LLM calls."""
    llm_calls = [e for e in events if e.get("event_type") == "llm_call"]
    llm_calls.sort(key=lambda x: parse_timestamp(x["timestamp"]))

    session_id = next((e.get("session_id") for e in events if e.get("session_id")), None)
    pauses: list[dict[str, Any]] = []

    for i in range(len(llm_calls) - 1):
        current = llm_calls[i]
        nxt = llm_calls[i + 1]
        current_id = current.get("message_id")
        next_id = nxt.get("message_id")

        if current_id == next_id:
            continue
        if current.get("has_tool_use") or current.get("num_tools_called", 0):
            continue

        current_ts = parse_timestamp(current["timestamp"])
        next_ts = parse_timestamp(nxt["timestamp"])
        duration_ms = (next_ts - current_ts).total_seconds() * 1000
        duration_s = duration_ms / 1000.0

        if duration_s < min_duration_seconds or duration_ms >= 86_400_000:
            continue

        has_tool_between = any(
            e.get("event_type") == "tool_call"
            and current_ts < parse_timestamp(e["timestamp"]) < next_ts
            for e in events
        )
        if has_tool_between:
            continue

        pauses.append(
            {
                "event_type": "tool_call",
                "tool_name": "HumanPause",
                "timestamp": current.get("timestamp"),
                "session_id": session_id,
                "tool_use_id": f"human_pause_{current_id}_{next_id}",
                "tool_input": {
                    "description": "Human pause between LLM responses",
                    "previous_llm_call_id": current_id,
                    "next_llm_call_id": next_id,
                },
                "associated_llm_call_id": current_id,
                "execution_time_ms": duration_ms,
                "result_timestamp": nxt.get("timestamp"),
                "has_result": True,
                "is_error": False,
                "source_file": source_file,
            }
        )

    return pauses


def insert_human_pauses(
    events: list[dict[str, Any]],
    human_pauses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pause_map: dict[tuple[str, str], dict[str, Any]] = {}
    for pause in human_pauses:
        prev_id = pause.get("tool_input", {}).get("previous_llm_call_id")
        next_id = pause.get("tool_input", {}).get("next_llm_call_id")
        if prev_id and next_id:
            pause_map[(prev_id, next_id)] = pause

    events_sorted = sorted(events, key=lambda x: parse_timestamp(x.get("timestamp", "")))
    result: list[dict[str, Any]] = []

    for i, event in enumerate(events_sorted):
        result.append(event)
        if event.get("event_type") != "llm_call":
            continue
        if event.get("has_tool_use") or event.get("num_tools_called", 0):
            continue

        current_llm_id = event.get("message_id")
        next_llm = None
        next_llm_idx = None
        for j in range(i + 1, len(events_sorted)):
            if events_sorted[j].get("event_type") == "llm_call":
                next_llm = events_sorted[j]
                next_llm_idx = j
                break

        if not next_llm or not current_llm_id:
            continue
        next_llm_id = next_llm.get("message_id")
        key = (current_llm_id, next_llm_id)
        if key not in pause_map:
            continue

        current_ts = parse_timestamp(event["timestamp"])
        next_ts = parse_timestamp(next_llm["timestamp"])
        has_tool_between = False
        for k in range(i + 1, next_llm_idx):
            mid = events_sorted[k]
            if mid.get("event_type") == "tool_call":
                mid_ts = parse_timestamp(mid["timestamp"])
                if current_ts < mid_ts < next_ts:
                    has_tool_between = True
                    break

        if not has_tool_between:
            pause = pause_map[key]
            result.append(
                {
                    "event_type": "tool_call",
                    "timestamp": pause.get("timestamp"),
                    "session_id": pause.get("session_id"),
                    "tool_use_id": pause.get("tool_use_id"),
                    "tool_name": "HumanPause",
                    "tool_input": pause.get("tool_input", {}),
                    "associated_llm_call_id": pause.get("associated_llm_call_id"),
                    "has_result": pause.get("has_result", True),
                    "is_error": pause.get("is_error", False),
                    "execution_time_ms": pause.get("execution_time_ms"),
                    "result_timestamp": pause.get("result_timestamp"),
                }
            )
            del pause_map[key]

    return sorted(result, key=lambda x: parse_timestamp(x.get("timestamp", "")))
