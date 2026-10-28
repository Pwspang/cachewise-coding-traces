"""Validate paired request traces and plot client timelines and server metrics."""

import argparse
import gzip
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def metric(text, name):
    pattern = re.compile(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+(\S+)$")
    values = [float(m[1]) for line in text.splitlines() if (m := pattern.match(line))]
    return sum(values) if values else None


def audit_repeats(root):
    """Check every repeat, including archived stream integrity and actual usage."""
    expected = None
    runs = {}
    for path in sorted(root.glob("*-r*/manifest.json")):
        manifest = json.loads(path.read_text())
        assert manifest["status"] == "complete", f"Incomplete {path.parent.name}"
        events = records(path.parent / "requests.jsonl")
        starts = [r for r in events if r["event"] == "request_start"]
        ends = [r for r in events if r["event"] == "request_end"]
        signature = {
            r["request_key"]: (
                r["prompt_sha256"],
                r["prompt_tokens"],
                r["output_tokens"],
            )
            for r in starts
        }
        assert len(signature) == len(starts) == len(ends), "Duplicate/missing requests"
        assert set(signature) == {r["request_key"] for r in ends}
        assert not any(r["event"] == "request_error" for r in events)
        for r in ends:
            _, prompt, output = signature[r["request_key"]]
            assert r["usage"]["prompt_tokens"] == prompt
            assert r["usage"]["completion_tokens"] == output
        if expected is not None:
            assert signature == expected, "Different request tokens across repeats"
        expected = signature
        archive_counts = {}
        for archive in ("payloads.jsonl.gz", "responses.jsonl.gz"):
            with gzip.open(path.parent / archive, "rt") as stream:
                archive_counts[archive] = sum(1 for line in stream if json.loads(line))
        snapshots = records(path.parent / "metrics.jsonl")
        healthy = [r for r in snapshots if r.get("status") == 200]
        final = healthy[-1]["text"] if healthy else ""
        runs[path.parent.name] = {
            "requests": len(ends),
            "output_tokens": sum(r["usage"]["completion_tokens"] for r in ends),
            "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in ends),
            "metric_snapshots": len(snapshots),
            "failed_metric_snapshots": len(snapshots) - len(healthy),
            "preemptions": metric(final, "vllm:num_preemptions_total"),
            "archive_records": archive_counts,
        }
    assert len(runs) >= 2, "Need at least two completed runs"
    result = {"identical_request_hashes_and_lengths_all_runs": True, "runs": runs}
    (root / "repeat-trace-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def inspect(root):
    audit_repeats(root)
    comparison = json.loads((root / "comparison.json").read_text())
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 10), layout="constrained", sharex=True, sharey="row"
    )
    evidence, signatures, combined_timeline = {}, [], []
    for col, policy in enumerate(("baseline", "cachewise")):
        path = root / f"{policy}-r0"
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["status"] == "complete", f"Incomplete {policy}"
        events = records(path / "requests.jsonl")
        starts = {r["request_key"]: r for r in events if r["event"] == "request_start"}
        ends = {r["request_key"]: r for r in events if r["event"] == "request_end"}
        assert starts.keys() == ends.keys(), f"Unfinished requests in {policy}"
        assert not any(r["event"] == "request_error" for r in events)
        signatures.append(
            {
                key: (r["prompt_sha256"], r["prompt_tokens"], r["output_tokens"])
                for key, r in starts.items()
            }
        )
        timeline = json.loads((path / "timeline.json").read_text())["traceEvents"]
        combined_timeline.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": col + 1,
                "tid": 0,
                "args": {"name": policy + " (elapsed time aligned to run start)"},
            }
        )
        for lane, sid in enumerate(manifest["session_ids"]):
            combined_timeline.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": col + 1,
                    "tid": lane,
                    "args": {"name": sid},
                }
            )
        combined_timeline.extend({**event, "pid": col + 1} for event in timeline)
        for event in timeline:
            color = "#2563eb" if event["name"] == "LLM request" else "#d1d5db"
            axes[0, col].broken_barh(
                [(event["ts"] / 1e6, event["dur"] / 1e6)],
                (event["tid"] - 0.32, 0.64),
                facecolors=color,
            )
        axes[0, col].set_title("Baseline" if policy == "baseline" else "CacheWise")
        axes[0, col].set_yticks(
            range(len(manifest["session_ids"])), manifest["session_ids"]
        )
        axes[0, col].set_xlabel("Elapsed seconds (blue: LLM; gray: external delay)")
        snapshots = records(path / "metrics.jsonl")
        healthy = [r for r in snapshots if r.get("status") == 200]
        cache = [
            (r["elapsed_s"], metric(r["text"], "vllm:kv_cache_usage_perc"))
            for r in healthy
        ]
        cache = [(t, value) for t, value in cache if value is not None]
        if cache:
            axes[1, col].plot(
                [t for t, _ in cache], [100 * v for _, v in cache], color="#2563eb"
            )
        axes[1, col].set(
            xlabel="Elapsed seconds", ylabel="Active KV-cache usage (%)", ylim=(0, 105)
        )
        ordered = sorted(ends.values(), key=lambda r: r["start_s"])
        axes[2, col].scatter(
            [r["start_s"] for r in ordered], [r["ttft_s"] for r in ordered], s=14
        )
        axes[2, col].set(
            xlabel="Request start (seconds)", ylabel="Time to first token (seconds)"
        )
        final = healthy[-1]["text"] if healthy else ""
        evidence[policy] = {
            "requests": len(ends),
            "metric_snapshots": len(snapshots),
            "failed_metric_snapshots": len(snapshots) - len(healthy),
            "max_active_kv_cache_fraction": max((v for _, v in cache), default=None),
            "preemptions": metric(final, "vllm:num_preemptions_total"),
            "sampled_evicted_blocks": metric(
                final, "vllm:kv_block_lifetime_seconds_count"
            ),
            "prefix_cache_queries": metric(final, "vllm:prefix_cache_queries_total"),
            "prefix_cache_hits": metric(final, "vllm:prefix_cache_hits_total"),
            "output_tokens": sum(
                r["usage"]["completion_tokens"] for r in ends.values()
            ),
        }
        for ax in axes[:, col]:
            ax.grid(alpha=0.2)
    assert signatures[0] == signatures[1], "Policies received different request tokens"
    evidence["identical_request_hashes_and_lengths"] = True
    fig.suptitle(
        "Qwen2.5-Coder-14B-Instruct · one RTX Pro 6000 · synthetic trace replay · repeat 0"
    )
    fig.savefig(root / "trace_overview.png", dpi=160)
    plt.close(fig)
    (root / "trace_audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (root / "timeline-comparison.json").write_text(
        json.dumps({"traceEvents": combined_timeline}) + "\n"
    )
    print(json.dumps({"comparison": comparison, "trace_audit": evidence}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    inspect(parser.parse_args().root)
