import json

import pytest
from analyze_oracle import CONSOLIDATED_POLICIES
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
