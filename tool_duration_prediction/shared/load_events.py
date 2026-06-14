"""Load per-session events.json from a release tree."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_events_files(events_dir: Path) -> Iterator[Path]:
    if not events_dir.is_dir():
        raise FileNotFoundError(f"Events directory not found: {events_dir}")
    for path in sorted(events_dir.rglob("events.json")):
        if path.parent.name.startswith("session_"):
            yield path


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return json.load(f)
