#!/usr/bin/env python3
"""Inference for expected remaining tool-call duration E[T - t | T > t]."""

from __future__ import annotations

import argparse
import bisect
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.curves import cluster_curve, overall_curve, point_prob_still_running
from shared.paths import DEFAULT_MODELS

SIM_THRESHOLD = 0.3


class ToolDurationModel:
    def __init__(self, tool_name: str, data: dict[str, Any]):
        self.tool_name = tool_name
        self.vectorizer = data["vectorizer"]
        self.cluster_model = data["cluster_model"]
        self.overall_stats = data["overall_stats"]
        self.overall_prob_still_running = overall_curve(data)
        self.cluster_stats = data["cluster_stats"]
        self.centroids = (
            self.cluster_model.cluster_centers_
            if self.cluster_model is not None
            and hasattr(self.cluster_model, "cluster_centers_")
            else None
        )

    def prob_still_running_curve_for_cluster(self, cluster_id: int) -> list[dict[str, float]]:
        for cluster in self.cluster_stats:
            if cluster["cluster_id"] == cluster_id:
                return cluster_curve(cluster)
        return self.overall_prob_still_running

    def nearest_cluster(self, text_vector: np.ndarray) -> tuple[int, float]:
        if self.cluster_model is None or self.centroids is None:
            return 0, 1.0
        sim = cosine_similarity(text_vector, self.centroids)[0]
        best = int(sim.argmax())
        return best, float(sim[best])


def load_models(models_dir: Path) -> dict[str, ToolDurationModel]:
    bundle_path = models_dir / "all_models.pkl"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
    with bundle_path.open("rb") as f:
        bundled = pickle.load(f)
    return {name: ToolDurationModel(name, data) for name, data in bundled.items()}


def lookup_prob_still_running(curve: list[dict[str, float]], t_ms: float) -> float:
    if not curve:
        return 0.0
    times = [pt["t_ms"] for pt in curve]
    probs = [point_prob_still_running(pt) for pt in curve]
    idx = bisect.bisect_left(times, t_ms)
    if idx <= 0:
        return probs[0]
    if idx >= len(times):
        return 0.0
    t0, t1 = times[idx - 1], times[idx]
    p0, p1 = probs[idx - 1], probs[idx]
    if t1 == t0:
        return p1
    w = (t_ms - t0) / (t1 - t0)
    return p0 + w * (p1 - p0)


def expected_remaining_time(curve: list[dict[str, float]], elapsed_ms: float) -> float:
    if not curve:
        return 0.0
    p_t = lookup_prob_still_running(curve, elapsed_ms)
    if p_t <= 0:
        return 0.0

    times = np.array([pt["t_ms"] for pt in curve])
    probs = np.array([point_prob_still_running(pt) for pt in curve])
    idx = bisect.bisect_left(times, elapsed_ms)
    if idx >= len(times):
        return 0.0

    if idx == 0:
        start_prob = probs[0]
        start_idx = 0
    else:
        t0, t1 = times[idx - 1], times[idx]
        p0, p1 = probs[idx - 1], probs[idx]
        if t1 == t0:
            start_prob = p1
        else:
            w = (elapsed_ms - t0) / (t1 - t0)
            start_prob = p0 + w * (p1 - p0)
        start_idx = idx

    integral = 0.0
    if start_idx < len(times) and elapsed_ms < times[start_idx]:
        integral += (times[start_idx] - elapsed_ms) * (start_prob + probs[start_idx]) / 2.0
    for i in range(start_idx, len(times) - 1):
        integral += (times[i + 1] - times[i]) * (probs[i] + probs[i + 1]) / 2.0

    return float(integral / p_t) if p_t > 0 else 0.0


def canonicalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def select_curve(
    model: ToolDurationModel, text: str
) -> tuple[list[dict[str, float]], float, int]:
    text_vec = model.vectorizer.transform([text])
    cluster_id, similarity = model.nearest_cluster(text_vec)
    if similarity < SIM_THRESHOLD:
        return model.overall_prob_still_running, similarity, -1
    return model.prob_still_running_curve_for_cluster(cluster_id), similarity, cluster_id


def run_single(
    models: dict[str, ToolDurationModel], tool: str, elapsed_ms: float, text: str
) -> dict[str, Any]:
    model = models.get(tool) or models.get(tool.lower()) or models.get(tool.capitalize())
    if not model:
        raise ValueError(f"No model found for tool '{tool}'")
    curve, similarity, cluster_id = select_curve(model, text)
    expected_remaining_ms = expected_remaining_time(curve, elapsed_ms)
    return {
        "tool": tool,
        "elapsed_ms": elapsed_ms,
        "elapsed_seconds": elapsed_ms / 1000.0,
        "expected_remaining_ms": expected_remaining_ms,
        "expected_remaining_seconds": expected_remaining_ms / 1000.0,
        "similarity": similarity,
        "cluster_id": cluster_id,
    }


def demo(models: dict[str, ToolDurationModel]) -> None:
    tests = [
        {"tool": "Bash", "text": "", "elapsed_ms": 30_000},
        {"tool": "Read", "text": "", "elapsed_ms": 2_000},
        {"tool": "HumanPause", "text": "(human pause)", "elapsed_ms": 60_000},
    ]
    print("=== Demo: expected remaining time ===")
    for test in tests:
        result = run_single(models, test["tool"], test["elapsed_ms"], test["text"])
        print(json.dumps({**test, **result}, indent=2))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--tool", help="Tool name (e.g. Bash, HumanPause)")
    parser.add_argument("--elapsed", type=float, help="Elapsed time in seconds")
    parser.add_argument("--text", default="", help="Argument text for TF-IDF cluster match")
    parser.add_argument("--demo", action="store_true", help="Run demo queries")
    args = parser.parse_args()

    models = load_models(args.models)
    if args.demo or not args.tool:
        demo(models)
        return

    result = run_single(
        models,
        args.tool,
        args.elapsed * 1000.0,
        canonicalize_text(args.text),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()