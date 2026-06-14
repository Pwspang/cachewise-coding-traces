"""Helpers for prob-still-running curves (P(execution time > t))."""

from __future__ import annotations

from typing import Any


def point_prob_still_running(pt: dict[str, float]) -> float:
    return pt["prob_still_running"]


def overall_curve(data: dict[str, Any]) -> list[dict[str, float]]:
    return data["overall_prob_still_running"]


def cluster_curve(cluster: dict[str, Any]) -> list[dict[str, float]]:
    return cluster["prob_still_running_curve"]
