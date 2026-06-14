#!/usr/bin/env python3
"""Extract all tool calls (including HumanPause) from events with pauses inserted"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.load_events import iter_events_files, load_events
from shared.paths import DEFAULT_EVENTS_WITH_PAUSES, DEFAULT_TOOL_CALLS


def calculate_context_at_point(events: list[dict[str, Any]], event_index: int) -> dict[str, Any]:
    context: dict[str, Any] = {
        "total_events_before": event_index,
        "llm_calls_before": 0,
        "tool_calls_before": 0,
        "total_input_tokens": 0,
        "total_cache_read_tokens": 0,
        "current_context_length": 0,
        "session_progress": 0.0,
    }
    total_events = len(events)
    if total_events > 0:
        context["session_progress"] = event_index / total_events

    current_context = 0
    for i in range(min(event_index, len(events))):
        event = events[i]
        if event.get("event_type") == "llm_call":
            context["llm_calls_before"] += 1
            input_tokens = event.get("input_tokens", 0) or 0
            cache_read = event.get("cache_read_input_tokens", 0) or 0
            context["total_input_tokens"] += input_tokens
            context["total_cache_read_tokens"] += cache_read
            current_context = max(current_context, input_tokens + cache_read)
        elif event.get("event_type") == "tool_call":
            context["tool_calls_before"] += 1

    context["current_context_length"] = current_context
    return context


def extract_tool_call_context(events: list[dict[str, Any]], tool_call_index: int) -> dict[str, Any]:
    tool_call = events[tool_call_index]
    associated_llm_id = tool_call.get("associated_llm_call_id")
    associated_llm = None
    for i in range(tool_call_index - 1, -1, -1):
        if (
            events[i].get("event_type") == "llm_call"
            and events[i].get("message_id") == associated_llm_id
        ):
            associated_llm = events[i]
            break

    tool_input = tool_call.get("tool_input", {}) or {}
    command = None
    if tool_call.get("tool_name") in ("Bash", "Shell") and isinstance(tool_input, dict):
        command = tool_input.get("command")

    result: dict[str, Any] = {
        "tool_name": tool_call.get("tool_name"),
        "tool_input": tool_input,
        "command": command,
        "execution_time_ms": tool_call.get("execution_time_ms"),
        "execution_time_seconds": (tool_call.get("execution_time_ms") or 0) / 1000,
        "is_error": tool_call.get("is_error"),
        "has_result": tool_call.get("has_result"),
        "associated_llm_model": associated_llm.get("model") if associated_llm else None,
        "associated_llm_input_tokens": associated_llm.get("input_tokens") if associated_llm else None,
        "associated_llm_cache_read": (
            associated_llm.get("cache_read_input_tokens") if associated_llm else None
        ),
    }
    result.update(calculate_context_at_point(events, tool_call_index))
    return result


def extract_all_tool_calls(events_dir: Path) -> list[dict[str, Any]]:
    all_tool_calls: list[dict[str, Any]] = []
    for path in iter_events_files(events_dir):
        events = load_events(path)
        if not events:
            continue
        session_id = events[0].get("session_id") or path.parent.name
        project_id = path.parent.parent.name
        sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))

        for i, event in enumerate(sorted_events):
            if event.get("event_type") != "tool_call":
                continue
            if event.get("execution_time_ms") is None:
                continue
            all_tool_calls.append(
                {
                    "session_id": session_id,
                    "project_id": project_id,
                    "event_index": i,
                    **extract_tool_call_context(sorted_events, i),
                }
            )
    return all_tool_calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EVENTS_WITH_PAUSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_TOOL_CALLS)
    args = parser.parse_args()

    tool_calls = extract_all_tool_calls(args.input)
    real = sum(1 for t in tool_calls if t.get("tool_name") != "HumanPause")
    pauses = sum(1 for t in tool_calls if t.get("tool_name") == "HumanPause")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(tool_calls, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(tool_calls)} tool calls to {args.output}")
    print(f"real tools: {real}, HumanPause: {pauses}")


if __name__ == "__main__":
    main()
