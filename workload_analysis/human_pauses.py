"""HumanPause detection"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def parse_timestamp(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def count_human_pauses(
    events: list[dict[str, Any]],
    *,
    min_duration_seconds: float = 2.0,
) -> int:
    return len(
        extract_human_pause_durations(events, min_duration_seconds=min_duration_seconds)
    )


def extract_human_pause_durations(
    events: list[dict[str, Any]],
    *,
    min_duration_seconds: float = 2.0,
) -> list[float]:
    llm_calls = [e for e in events if e.get("event_type") == "llm_call"]
    llm_calls.sort(key=lambda x: parse_timestamp(x["timestamp"]))

    durations_ms: list[float] = []
    for i in range(len(llm_calls) - 1):
        current = llm_calls[i]
        nxt = llm_calls[i + 1]

        if current.get("message_id") == nxt.get("message_id"):
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
        if not has_tool_between:
            durations_ms.append(duration_ms)

    return durations_ms
