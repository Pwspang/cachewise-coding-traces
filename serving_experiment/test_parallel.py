import json

import collect_parallel
import pytest
from analyze_oracle import CONSOLIDATED_POLICIES, normalize_parallel_entrypoint
from collect_parallel import repeat_finished, stage


def write_completed_repeat(root, repeat):
    root.mkdir()
    for name in (
        "prepared-inputs.json",
        "workload.json",
        "bootstrap.jsonl",
        "vllm-source.patch",
        "packages.txt",
        "model_snapshot.txt",
        "environment.json",
    ):
        (root / name).write_text("fixture")
    for policy in CONSOLIDATED_POLICIES:
        path = root / f"{policy}-r{repeat}"
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"status": "complete"}))
        (path / "summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "errors": [],
                    "completed_sessions": 10,
                    "sessions": [{"requests": 8}] * 9 + [{"requests": 11}],
                }
            )
        )
        (root / f"{path.name}-command.json").write_text("[]")


def test_repeat_gate_waits_for_all_policies_and_rejects_bad_counts(tmp_path):
    source = tmp_path / "original"
    assert not repeat_finished(source, 0)
    write_completed_repeat(source, 0)
    assert repeat_finished(source, 0)
    path = source / "cachewise-r0/summary.json"
    saved = path.read_text()
    path.write_text('{"status":')
    assert not repeat_finished(source, 0)
    bad = json.loads(saved)
    bad["sessions"] = [{"requests": 1}]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="request count"):
        repeat_finished(source, 0)


def test_staged_repeat_preserves_original_and_excludes_duplicate_work(tmp_path):
    source = tmp_path / "original"
    write_completed_repeat(source, 0)
    duplicate = source / "eviction_only-r1"
    duplicate.mkdir()
    (duplicate / "manifest.json").write_text('{"status":"running"}')
    output = tmp_path / "repeat0"
    stage(output, {0: source})
    assert (output / "COMPLETE").exists()
    assert not (source / "COMPLETE").exists()
    assert not (output / "eviction_only-r1").exists()
    assert repeat_finished(output, 0)
    stage(output, {0: source})  # Safe to rebuild the same view after interruption.


def test_completion_during_slurm_query_reads_fresh_files(tmp_path, monkeypatch):
    root = tmp_path / "repeat1"

    def finish_during_query(job):
        write_completed_repeat(root, 1)
        (root / "consolidated-comparison.json").write_text("{}")
        return "COMPLETED"

    monkeypatch.setattr(collect_parallel, "state", finish_during_query)
    assert collect_parallel.block_ready(root, 1, 123, {}, 0)


def test_completed_job_waits_briefly_for_audit_visibility(tmp_path, monkeypatch):
    root = tmp_path / "repeat1"
    write_completed_repeat(root, 1)
    monkeypatch.setattr(collect_parallel, "state", lambda job: "COMPLETED")
    missing = {}
    assert not collect_parallel.block_ready(root, 1, 123, missing, 0)
    assert not collect_parallel.block_ready(root, 1, 123, missing, 119)
    with pytest.raises(RuntimeError, match="remain missing"):
        collect_parallel.block_ready(root, 1, 123, missing, 120)
    (root / "consolidated-comparison.json").write_text("{}")
    assert collect_parallel.block_ready(root, 1, 123, missing, 121)
    assert missing == {}


def test_parallel_entrypoint_aliases_preserve_actual_server_options():
    first = [
        "/root/.venv/bin/python",
        "/root/server_queue_diagnostics.py",
        "serve",
        "model-A",
        "--port",
        "18080",
    ]
    second = [
        "/root/parallel/.venv/bin/python",
        "/root/parallel/server_queue_diagnostics.py",
        *first[2:],
    ]
    assert normalize_parallel_entrypoint(first) == normalize_parallel_entrypoint(second)
    second[3] = "model-B"
    assert normalize_parallel_entrypoint(first) != normalize_parallel_entrypoint(second)
    second[1] = "/root/different_wrapper.py"
    with pytest.raises(AssertionError):
        normalize_parallel_entrypoint(second)
