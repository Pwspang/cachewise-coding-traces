#!/usr/bin/env python3
"""Extract events.json from sanitized raw traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from event_parser import (
    events_to_dict,
    extract_events_from_transcript,
    parse_transcript_entry,
)

ROOT = SCRIPT_DIR.parent


def discover_jsonl_files(input_dir: Path) -> list[Path]:
    """Find sanitized session JSONL files (skip manifest.json)."""
    files: list[Path] = []
    for path in sorted(input_dir.rglob("*.jsonl")):
        if path.name == "manifest.json":
            continue
        files.append(path)
    return files


def repair_sanitized_line(obj: dict[str, Any], project_id: str) -> dict[str, Any]:
    """Add parser-required fields dropped or redacted during sanitization."""
    out = dict(obj)
    out.setdefault("userType", "external")
    out.setdefault("version", "0.0.0")
    out["cwd"] = f"{project_id}/"
    return out


def load_sanitized_transcript(jsonl_path: Path, project_id: str) -> list[Any]:
    """Load sanitized JSONL and parse into transcript entries."""
    entries: list[Any] = []
    with jsonl_path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj = repair_sanitized_line(obj, project_id)
                entries.append(parse_transcript_entry(obj))
            except Exception as exc:
                print(f"  Warning: {jsonl_path.name}:{line_no} skipped: {exc}")
    return entries


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def extract_events_file(
    jsonl_path: Path, output_path: Path, project_id: str, root: Path
) -> dict[str, Any]:
    """Extract events for one sanitized JSONL file."""
    transcript_entries = load_sanitized_transcript(jsonl_path, project_id)
    events = extract_events_from_transcript(transcript_entries)
    events_dict = events_to_dict(events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(events_dict, f, indent=2)
        f.write("\n")

    llm_calls = sum(1 for e in events if e.event_type == "llm_call")
    tool_calls = sum(1 for e in events if e.event_type == "tool_call")
    return {
        "input_path": _display_path(jsonl_path, root),
        "output_path": _display_path(output_path, root),
        "num_events": len(events),
        "num_llm_calls": llm_calls,
        "num_tool_calls": tool_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract events.json from release/raw_traces JSONL files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "release/raw_traces",
        help="Directory containing sanitized .jsonl traces",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "release/parsed_traces",
        help="Output directory for per-session events.json files",
    )
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    jsonl_files = discover_jsonl_files(input_dir)
    if not jsonl_files:
        raise SystemExit(f"No .jsonl files found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sessions: list[dict[str, Any]] = []
    errors: list[str] = []

    print(f"Extracting events from {len(jsonl_files)} JSONL files")
    print(f"  input:  {input_dir}")
    print(f"  output: {output_dir}")

    for jsonl_path in jsonl_files:
        rel = jsonl_path.relative_to(input_dir)
        # raw_traces/project_001/session_0001.jsonl
        # -> parsed_traces/project_001/session_0001/events.json
        if len(rel.parts) < 2:
            project_id = "_root"
            session_id = rel.stem
        else:
            project_id = rel.parts[0]
            session_id = rel.stem

        out_path = output_dir / project_id / session_id / "events.json"
        try:
            info = extract_events_file(jsonl_path, out_path, project_id, ROOT)
            info["project_id"] = project_id
            info["session_id"] = session_id
            manifest_sessions.append(info)
            print(
                f"  {rel} -> {out_path} "
                f"({info['num_llm_calls']} llm, {info['num_tool_calls']} tool)"
            )
        except Exception as exc:
            msg = f"{rel}: {exc}"
            errors.append(msg)
            print(f"  ERROR {msg}")

    manifest = {
        "num_sessions": len(manifest_sessions),
        "total_llm_calls": sum(s["num_llm_calls"] for s in manifest_sessions),
        "total_tool_calls": sum(s["num_tool_calls"] for s in manifest_sessions),
        "sessions": manifest_sessions,
        "errors": errors,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\nWrote {len(manifest_sessions)} events.json files to {output_dir}")
    print(f"Manifest: {manifest_path}")
    if errors:
        print(f"Errors: {len(errors)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
