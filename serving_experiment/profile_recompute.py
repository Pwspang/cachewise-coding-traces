"""Profile uncached prefill latency and fit InferCept's recompute-time model.

Each probe is a fresh random prompt (no prefix-cache hit) with one output
token, so latency approximates recomputing that many context tokens under
the server's chunked-prefill settings. The fit T_rec(C) = a*C^2 + b*C + c is
written as {"recompute_s_coeffs": [a, b, c]} for --infercept-recompute-profile.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

SIZES = [512, 2048, 8192, 16384, 32768, 65536, 98304, 126976]


def fit_quadratic(points):
    """Least-squares a*x^2 + b*x + c, clamping a and b to be non-negative."""
    sums = [sum(x**k for x, _ in points) for k in range(5)]
    rhs = [sum(y * x**k for x, y in points) for k in range(3)]
    # Normal equations, unknowns ordered (c, b, a).
    matrix = [[sums[i + j] for j in range(3)] + [rhs[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(matrix[r][col]))
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        for row in range(3):
            if row != col:
                factor = matrix[row][col] / matrix[col][col]
                matrix[row] = [v - factor * p for v, p in zip(matrix[row], matrix[col])]
    c, b, a = (matrix[i][3] / matrix[i][i] for i in range(3))
    if a < 0 or b < 0:
        # Fall back to a non-negative linear fit through the measurements.
        n, sx, sxx = sums[0], sums[1], sums[2]
        b = max(0.0, (n * rhs[1] - sx * rhs[0]) / (n * sxx - sx * sx))
        a, c = 0.0, (rhs[0] - b * sx) / n
    return a, b, c


def probe(base_url, model, tokens, rng):
    payload = {
        "model": model,
        "prompt": [rng.randrange(1000, 20000) for _ in range(tokens)],
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
    }
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read())
    elapsed = time.perf_counter() - started
    usage = body["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    if usage["prompt_tokens"] != tokens or cached >= tokens // 2:
        raise RuntimeError(f"Probe was not an uncached {tokens}-token prefill: {usage}")
    return elapsed, cached


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:18080")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=20240201)
    a = p.parse_args()
    if a.output.exists():
        p.error(f"{a.output} already exists")
    rng = random.Random(a.seed)
    probe(a.base_url, a.model, min(a.sizes), rng)  # Warm-up, not recorded.
    samples = []
    for tokens in a.sizes:
        for _ in range(a.repeats):
            elapsed, cached = probe(a.base_url, a.model, tokens, rng)
            samples.append({"tokens": tokens, "latency_s": elapsed, "cached": cached})
            print(f"{tokens} tokens: {elapsed:.3f} s", flush=True)
    points = [
        (x, statistics.median(s["latency_s"] for s in samples if s["tokens"] == x))
        for x in a.sizes
    ]
    coeffs = fit_quadratic(points)
    a_, b_, c_ = coeffs
    a.output.write_text(
        json.dumps(
            {
                "recompute_s_coeffs": list(coeffs),
                "median_points": [
                    {"tokens": x, "latency_s": y, "fit_s": a_ * x * x + b_ * x + c_}
                    for x, y in points
                ],
                "samples": samples,
                "method": (
                    "Uncached random-token prompts, max_tokens=1, client wall "
                    "time; one warm-up probe discarded."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"recompute_s_coeffs": list(coeffs)}))


if __name__ == "__main__":
    main()
