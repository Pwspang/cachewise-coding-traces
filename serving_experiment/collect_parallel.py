"""Preserve repeat 0, stop duplicate work, and audit the three parallel blocks."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from analyze_oracle import CONSOLIDATED_POLICIES, analyze


def repeat_finished(root, repeat):
    for policy in CONSOLIDATED_POLICIES:
        path = root / f"{policy}-r{repeat}"
        try:
            manifest = json.loads((path / "manifest.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        if manifest["status"] != "complete" or summary["status"] != "complete":
            return False
        if summary["errors"] or summary["completed_sessions"] != 10:
            raise ValueError(f"Invalid completed run: {path}")
        if sum(s["requests"] for s in summary["sessions"]) != 83:
            raise ValueError(f"Wrong request count: {path}")
    return True


def state(job):
    result = subprocess.check_output(
        ["sacct", "-X", "-j", str(job), "--format=State", "--noheader", "--parsable2"],
        text=True,
    ).strip()
    return result.splitlines()[0].split("|")[0].split()[0] if result else "UNKNOWN"


def link(destination, source):
    if destination.is_symlink():
        assert destination.resolve() == source.resolve()
    else:
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def stage(root, sources):
    """Build a scoped view without editing the original job's completion marker."""
    root.mkdir(parents=True, exist_ok=True)
    for repeat, source in sources.items():
        link(root / f"source-r{repeat}", source)
        for policy in CONSOLIDATED_POLICIES:
            name = f"{policy}-r{repeat}"
            link(root / name, source / name)
            for path in source.glob(f"{name}-*"):
                link(root / path.name, path)
    first = sources[min(sources)]
    for name in (
        "prepared-inputs.json",
        "workload.json",
        "bootstrap.jsonl",
        "vllm-source.patch",
        "packages.txt",
        "model_snapshot.txt",
        "environment.json",
    ):
        link(root / name, first / name)
    (root / "COMPLETE").write_text("All runs in this scoped view completed.\n")


def collect(args):
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "repeat_0_original_job": args.original_job,
        "repeat_1_job": args.repeat_one_job,
        "repeat_2_job": args.repeat_two_job,
        "repeat_sources": [
            str(args.original),
            str(args.repeat_one),
            str(args.repeat_two),
        ],
        "design": "Three GPU blocks; all six policies within each repeat share one GPU.",
    }
    (args.output / "parallel-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    deadline = time.monotonic() + 6 * 3600
    staged_zero = False
    last_progress = None
    terminal = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
    }
    while time.monotonic() < deadline:
        zero_done = repeat_finished(args.original, 0)
        if zero_done and not staged_zero:
            # The next command file is written only after the last r0 server
            # and GPU monitor have stopped. This keeps all r0 files immutable.
            next_started = (args.original / "eviction_only-r1-command.json").exists()
            original_state = state(args.original_job)
            if next_started or original_state in terminal:
                if original_state not in terminal:
                    subprocess.run(["scancel", str(args.original_job)], check=True)
                    print(
                        f"Stopped original job {args.original_job} after repeat 0.",
                        flush=True,
                    )
                stage(args.output / "repeat-0", {0: args.original})
                analyze(args.output / "repeat-0", consolidated=True, repeat_index=0)
                staged_zero = True
                (args.output / "REPEAT0_AUDITED").write_text(
                    "Repeat 0 preserved and audited.\n"
                )
        if not zero_done and state(args.original_job) in terminal:
            raise RuntimeError("Original job ended before repeat 0 completed")
        ready = []
        for repeat, job, root in (
            (1, args.repeat_one_job, args.repeat_one),
            (2, args.repeat_two_job, args.repeat_two),
        ):
            complete = repeat_finished(root, repeat)
            audited = (root / "consolidated-comparison.json").exists()
            job_state = state(job)
            if job_state in terminal and not (
                complete and audited and job_state == "COMPLETED"
            ):
                raise RuntimeError(
                    f"Repeat {repeat} job {job} ended as {job_state} without successful audit"
                )
            ready.append(complete and audited and job_state == "COMPLETED")
        progress = (staged_zero, *ready)
        if progress != last_progress:
            print(f"Completed/audited repeat blocks: {progress}", flush=True)
            last_progress = progress
        if all(progress):
            merged = args.output / "comparison"
            stage(merged, {0: args.original, 1: args.repeat_one, 2: args.repeat_two})
            analyze(merged, consolidated=True, parallel=True)
            (args.output / "COMPLETE").write_text(
                "All 18 runs passed the parallel audit.\n"
            )
            return
        time.sleep(15)
    raise TimeoutError("Parallel collection exceeded six hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-job", type=int, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--repeat-one-job", type=int, required=True)
    parser.add_argument("--repeat-one", type=Path, required=True)
    parser.add_argument("--repeat-two-job", type=int, required=True)
    parser.add_argument("--repeat-two", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        collect(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILED").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
