"""Reproduce predictor coverage and idle-vs-tool eviction ordering on CPU."""

import argparse
import json
from collections import Counter
from pathlib import Path

from vllm.v1.core.cachewise.eviction import PredictiveFreeBlockQueue
from vllm.v1.core.cachewise.predictor import ToolReusePredictor
from vllm.v1.core.kv_cache_utils import KVCacheBlock


def probe(bootstrap, workload, output):
    predictor = ToolReusePredictor(120, bootstrap_path=str(bootstrap))
    forecasts = {}
    for names in [[], ["Bash"], ["Read"], ["Read", "Read"]]:
        key = predictor._key([(n, "<redacted>") for n in names])
        forecasts[key] = {
            str(t): predictor.predict_remaining([(n, "<redacted>") for n in names], t)
            for t in [0, 0.08, 1, 11, 58, 300]
        }
    coverage = Counter()
    missing = Counter()
    for s in json.loads(workload.read_text())["sessions"]:
        for t in s["turns"][:-1]:
            key = predictor._key([(x["name"], x["args"]) for x in t["next_tools"]])
            found = key in predictor.tool_dists
            coverage["trained_key" if found else "global_fallback"] += 1
            if not found:
                missing[key] += 1
    priority = {0: forecasts["__human__"]["0"], 1: forecasts["Bash"]["0"]}

    def victim(priorities):
        blocks = [KVCacheBlock(0), KVCacheBlock(1)]
        queue = PredictiveFreeBlockQueue(
            blocks, lambda bid: bid, priorities.__getitem__, lambda *_: None, 120
        )
        queue.popleft_n(2)
        for b in blocks:
            b._block_hash = bytes([b.block_id + 1]) * 8
        queue.append_n(blocks)
        queue.rebuild(priorities)
        return ["idle_human", "tool_Bash"][queue.popleft().block_id]

    result = {
        "bootstrap_forecasts_s": forecasts,
        "workload_initial_key_coverage": dict(coverage),
        "missing_keys": dict(missing),
        "fresh_idle_vs_tool_priorities_s": priority,
        "original_eviction_victim": victim(priority),
        "human_first_eviction_victim": victim({0: 1e12, 1: priority[1]}),
    }
    assert result["original_eviction_victim"] == "tool_Bash"
    assert result["human_first_eviction_victim"] == "idle_human"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.bootstrap, args.workload, args.output)
