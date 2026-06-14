from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_DIR = ROOT / "release/parsed_traces"


def iter_events_files(events_dir: Path | None = None) -> Iterator[Path]:
    """Yield every session events.json under parsed_traces."""
    base = events_dir or DEFAULT_EVENTS_DIR
    if not base.is_dir():
        raise FileNotFoundError(f"Events directory not found: {base}")
    for path in sorted(base.rglob("events.json")):
        if path.name == "events.json" and path.parent.name.startswith("session_"):
            yield path


def load_all_events(events_dir: Path | None = None) -> list[tuple[Path, list[dict[str, Any]]]]:
    """Load all session event lists. Returns (path, events) pairs."""
    sessions: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in iter_events_files(events_dir):
        with path.open() as f:
            events = json.load(f)
        if events:
            sessions.append((path, events))
    return sessions


def session_id_from_path(path: Path) -> str:
    return path.parent.name
