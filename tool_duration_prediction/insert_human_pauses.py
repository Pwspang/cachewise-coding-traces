#!/usr/bin/env python3
"""Insert filtered HumanPause tool_call events into per-session event logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.human_pauses import extract_human_pause_events, insert_human_pauses
from shared.load_events import iter_events_files, load_events
from shared.paths import DEFAULT_EVENTS_WITH_PAUSES, DEFAULT_PARSED_TRACES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_PARSED_TRACES)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVENTS_WITH_PAUSES)
    args = parser.parse_args()

    total_pauses = 0
    total_sessions = 0

    for in_path in iter_events_files(args.input):
        rel = in_path.relative_to(args.input)
        out_path = args.output / rel.parent / "events.json"
        events = load_events(in_path)
        if not events:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("[]\n")
            continue

        pauses = extract_human_pause_events(events, source_file=str(rel))
        enriched = insert_human_pauses(events, pauses) if pauses else events

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(enriched, f, indent=2)
            f.write("\n")

        total_sessions += 1
        total_pauses += len(pauses)

    print(f"Wrote {total_sessions} sessions to {args.output}")
    print(f"Inserted {total_pauses} HumanPause events")


if __name__ == "__main__":
    main()
