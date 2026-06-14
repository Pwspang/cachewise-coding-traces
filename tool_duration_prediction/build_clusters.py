#!/usr/bin/env python3
"""Build per-tool TF-IDF duration clusters and prob-still-running models."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.paths import DEFAULT_MODELS, DEFAULT_SPLITS, DEFAULT_TOOL_CALLS
from shared.session_split import load_train_sessions, write_train_sessions

MIN_CLUSTER_SIZE = 30
MAX_CLUSTERS = 1000
DURATION_THRESHOLD_60 = 60_000
DURATION_THRESHOLD_100 = 100_000
OUTLIER_THRESHOLDS = {"Edit": 60_000, "Read": 60_000}
HUMAN_PAUSE_TEXT = "(human pause)"


def normalize_command(command: str | None) -> str:
    if not command:
        return ""
    return " ".join(command.strip().lower().split())


def flatten_tool_input(tool_input: Any) -> list[str]:
    tokens: list[str] = []

    def _rec(value: Any) -> None:
        if isinstance(value, dict):
            for key, val in value.items():
                tokens.append(str(key))
                _rec(val)
        elif isinstance(value, list):
            for item in value:
                _rec(item)
        elif value is not None:
            tokens.append(str(value))

    _rec(tool_input)
    return tokens


def build_text_representation(entry: dict[str, Any]) -> str:
    if entry.get("tool_name") == "HumanPause":
        return HUMAN_PAUSE_TEXT
    command_text = normalize_command(entry.get("command"))
    tool_input_tokens = flatten_tool_input(entry.get("tool_input"))
    combined = " ".join([command_text] + tool_input_tokens)
    return combined.strip() or "(empty)"


def compute_duration_stats(durations_ms: np.ndarray) -> dict[str, float]:
    if durations_ms.size == 0:
        return {
            "count": 0,
            "mean_s": 0.0,
            "median_s": 0.0,
            "p90_s": 0.0,
            "pct_ge_60s": 0.0,
            "pct_ge_100s": 0.0,
        }
    durations_s = durations_ms / 1000.0
    return {
        "count": int(durations_ms.size),
        "mean_s": float(durations_s.mean()),
        "median_s": float(np.median(durations_s)),
        "p90_s": float(np.percentile(durations_s, 90)),
        "pct_ge_60s": float((durations_ms >= DURATION_THRESHOLD_60).mean() * 100),
        "pct_ge_100s": float((durations_ms >= DURATION_THRESHOLD_100).mean() * 100),
    }


def compute_prob_still_running_curve(
    durations_ms: np.ndarray, max_points: int = 200
) -> list[dict[str, float]]:
    if durations_ms.size == 0:
        return []
    sorted_ms = np.sort(durations_ms)
    n = sorted_ms.size
    unique_times, first_idx = np.unique(sorted_ms, return_index=True)
    prob_still_running = 1.0 - first_idx / n
    points = list(zip(unique_times, prob_still_running))
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = [points[i] for i in idx]
    return [{"t_ms": float(t), "prob_still_running": float(p)} for t, p in points]


def extract_top_terms(vectorizer: TfidfVectorizer, centroid: np.ndarray, top_k: int = 10) -> list[str]:
    if centroid.ndim > 1:
        centroid = centroid.A1
    feature_names = np.array(vectorizer.get_feature_names_out())
    if centroid.size != feature_names.size:
        return []
    top_indices = centroid.argsort()[::-1][:top_k]
    return [feature_names[idx] for idx in top_indices if centroid[idx] > 0]


def build_clusters(
    dataset_path: Path,
    train_sessions: set[str],
    output_dir: Path,
) -> dict[str, Any]:
    with dataset_path.open() as f:
        dataset = json.load(f)

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "cluster_stats.json"

    tool_data: dict[str, list[dict[str, Any]]] = defaultdict(list)
    filtered_outliers = 0

    for idx, entry in enumerate(dataset):
        if entry.get("session_id") not in train_sessions:
            continue
        duration_ms = entry.get("execution_time_ms", 0) or 0
        tool_name = entry.get("tool_name", "Unknown")
        if tool_name in OUTLIER_THRESHOLDS and duration_ms > OUTLIER_THRESHOLDS[tool_name]:
            filtered_outliers += 1
            continue
        tool_data[tool_name].append(
            {
                "dataset_index": idx,
                "text": build_text_representation(entry),
                "duration_ms": duration_ms,
            }
        )

    cluster_summary: dict[str, Any] = {}
    all_models: dict[str, Any] = {}

    for tool_name, info in sorted(tool_data.items(), key=lambda kv: kv[0]):
        texts = [rec["text"] for rec in info]
        durations = np.array([rec["duration_ms"] for rec in info], dtype=float)
        n_samples = len(texts)
        print(f"Processing {tool_name}: {n_samples} samples")

        overall_stats = compute_duration_stats(durations)
        overall_prob_still_running = compute_prob_still_running_curve(durations)

        if tool_name == "HumanPause":
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=1)
            tfidf_matrix = vectorizer.fit_transform(texts)
            labels = np.zeros(n_samples, dtype=int)
            cluster_model = None
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=1)
            tfidf_matrix = vectorizer.fit_transform(texts)
            if n_samples < MIN_CLUSTER_SIZE * 2:
                labels = np.zeros(n_samples, dtype=int)
                cluster_model = None
            else:
                n_clusters = min(MAX_CLUSTERS, max(2, n_samples // MIN_CLUSTER_SIZE))
                cluster_model = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    batch_size=min(512, max(32, n_clusters * 10)),
                    n_init="auto",
                )
                labels = cluster_model.fit_predict(tfidf_matrix)

        clusters_info = []
        for label in np.unique(labels):
            idxs = np.where(labels == label)[0]
            cluster_durations = durations[idxs]
            stats = compute_duration_stats(cluster_durations)
            prob_still_running_curve = compute_prob_still_running_curve(cluster_durations)
            centroid = tfidf_matrix[idxs].mean(axis=0)
            top_terms = extract_top_terms(vectorizer, centroid)
            clusters_info.append(
                {
                    "cluster_id": int(label),
                    "count": stats["count"],
                    "mean_s": stats["mean_s"],
                    "median_s": stats["median_s"],
                    "p90_s": stats["p90_s"],
                    "pct_ge_60s": stats["pct_ge_60s"],
                    "pct_ge_100s": stats["pct_ge_100s"],
                    "top_terms": top_terms,
                    "prob_still_running_curve": prob_still_running_curve,
                }
            )

        model_blob = {
            "tool_name": tool_name,
            "vectorizer": vectorizer,
            "cluster_model": cluster_model,
            "overall_stats": overall_stats,
            "overall_prob_still_running": overall_prob_still_running,
            "cluster_stats": clusters_info,
            "labels": labels,
            "dataset_indices": [rec["dataset_index"] for rec in info],
        }

        model_path = models_dir / f"{tool_name.replace('/', '_')}.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model_blob, f)
        all_models[tool_name] = model_blob
        cluster_summary[tool_name] = {
            "sample_count": n_samples,
            "cluster_count": len(clusters_info),
            "overall_stats": overall_stats,
            "overall_prob_still_running": overall_prob_still_running,
            "clusters": clusters_info,
            "model_path": str(model_path.relative_to(output_dir)),
        }

    with stats_path.open("w") as f:
        json.dump(cluster_summary, f, indent=2)
        f.write("\n")

    bundle_path = output_dir / "all_models.pkl"
    with bundle_path.open("wb") as f:
        pickle.dump(all_models, f)

    return {
        "tools": len(all_models),
        "filtered_outliers": filtered_outliers,
        "stats_path": str(stats_path),
        "bundle_path": str(bundle_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TOOL_CALLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    args = parser.parse_args()

    train_sessions, meta = load_train_sessions(args.splits)
    write_train_sessions(args.splits, train_sessions, meta)
    print(f"Training on {len(train_sessions)} sessions ({meta.get('source')})")

    summary = build_clusters(args.dataset, set(train_sessions), args.output)
    print(f"Built models for {summary['tools']} tools")
    if summary["filtered_outliers"]:
        print(f"Filtered {summary['filtered_outliers']} Edit/Read outliers > 60s")
    print(f"Stats: {summary['stats_path']}")
    print(f"Bundle: {summary['bundle_path']}")


if __name__ == "__main__":
    main()
