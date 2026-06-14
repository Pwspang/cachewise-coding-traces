from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_PARSED_TRACES = ROOT / "release/parsed_traces"
DEFAULT_EVENTS_WITH_PAUSES = PACKAGE_DIR / "datasets/events_with_pauses"
DEFAULT_TOOL_CALLS = PACKAGE_DIR / "datasets/all_tool_calls.json"
DEFAULT_SPLITS = PACKAGE_DIR / "datasets/splits/train_sessions.json"
DEFAULT_MODELS = PACKAGE_DIR / "models"
ID_MAPS = ROOT / "release/id_maps/session_maps.json"
ORIG_TRAIN_SESSIONS = ROOT / "example_logs/duration_bucket_classifier/training_sessions.json"
