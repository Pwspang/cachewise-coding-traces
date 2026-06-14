"""Train session split"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.paths import DEFAULT_PARSED_TRACES, ID_MAPS, ORIG_TRAIN_SESSIONS, ROOT


def _reverse_session_map(session_maps: dict[str, Any]) -> dict[str, str]:
    """Map source session UUID -> release session_id."""
    return {
        meta["source_session_id"]: release_id
        for release_id, meta in session_maps.items()
    }


def load_train_sessions(
    splits_path: Path | None = None,
    *,
    fallback_seed: int = 42,
    train_fraction: float = 0.5,
) -> tuple[list[str], dict[str, Any]]:
    """Return release session IDs for training.

    Prefers mapping ``example_logs/.../training_sessions.json`` UUIDs through
    private ``release/id_maps/session_maps.json``. Falls back to a deterministic
    fraction split on sorted release session IDs when maps are unavailable.
    """
    if splits_path and splits_path.is_file():
        data = json.loads(splits_path.read_text())
        return data["train_sessions"], data

    meta: dict[str, Any] = {"source": "generated", "seed": fallback_seed}

    if ID_MAPS.is_file() and ORIG_TRAIN_SESSIONS.is_file():
        session_maps = json.loads(ID_MAPS.read_text())
        orig = json.loads(ORIG_TRAIN_SESSIONS.read_text())
        uuid_to_release = _reverse_session_map(session_maps)
        train_sessions = [
            uuid_to_release[u]
            for u in orig.get("train_sessions", [])
            if u in uuid_to_release
        ]
        meta.update(
            {
                "source": "mapped_from_example_logs",
                "orig_train_count": len(orig.get("train_sessions", [])),
                "mapped_train_count": len(train_sessions),
            }
        )
        return sorted(train_sessions), meta

    # Fallback: use all release sessions from parsed_traces manifest
    manifest = ROOT / "release/parsed_traces/manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            "No train split found. Provide datasets/splits/train_sessions.json "
            "or ensure release/id_maps and parsed_traces/manifest exist."
        )
    entries = json.loads(manifest.read_text()).get("sessions", [])
    all_sessions = sorted({e["session_id"] for e in entries})
    import random

    rng = random.Random(fallback_seed)
    shuffled = list(all_sessions)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_fraction))
    train_sessions = sorted(shuffled[:n_train])
    meta.update(
        {
            "source": "random_fraction",
            "train_fraction": train_fraction,
            "total_sessions": len(all_sessions),
            "train_session_count": len(train_sessions),
        }
    )
    return train_sessions, meta


def write_train_sessions(path: Path, train_sessions: list[str], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_sessions": sorted(train_sessions),
        "train_session_count": len(train_sessions),
        **meta,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
