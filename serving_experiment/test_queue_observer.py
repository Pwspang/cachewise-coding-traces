import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from serving_experiment.probe_scheduler import load_queue_namespace
from serving_experiment.queue_observer import install


@pytest.mark.parametrize("limit", [0, 30])
def test_observer_preserves_actual_queue_order_and_counts_cached_choices(
    tmp_path, limit
):
    source = (
        Path(__file__).resolve().parents[1]
        / "external/vllm-cachewise/vllm/v1/core/sched/request_queue.py"
    )
    if not source.exists():
        pytest.skip("Requires the pinned external vLLM checkout")
    cls = load_queue_namespace(source)["PrefixAwareRequestQueue"]

    def make_queue():
        queue = cls(scorer=lambda r: r.required_blocks, max_wait_s=limit)
        queue.add_request(
            SimpleNamespace(request_id="old", arrival_time=960, required_blocks=6000)
        )
        queue.add_request(
            SimpleNamespace(request_id="new", arrival_time=995, required_blocks=1)
        )
        return queue

    original = make_queue()
    expected = [original.pop_request().request_id, original.pop_request().request_id]
    stats = install(cls, tmp_path / "queue", clock=lambda: 1000.0)
    observed = make_queue()
    observed.peek_request()
    observed.peek_request()
    assert stats["fresh_choices"] == 1
    assert [
        observed.pop_request().request_id,
        observed.pop_request().request_id,
    ] == expected
    assert stats["pops"] == stats["fresh_choices"] == 2
    assert stats["aging_pops"] == (1 if limit else 0)
    rows = [
        json.loads(line)
        for f in tmp_path.glob("queue-*.jsonl")
        for line in f.read_text().splitlines()
    ]
    assert [r["request_id"] for r in rows] == expected
    assert sum(r["aging_choice"] for r in rows) == (1 if limit else 0)
