"""Exercise the fork's actual queue selection code without importing vLLM.

Only the heavyweight Request import is replaced by SimpleNamespace test inputs.
All queue classes, selection logic, and base classes execute from the source.
This demonstrates the aging rule; it is not a performance experiment.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def load_queue_namespace(source):
    code = source.read_text()
    tree = ast.parse(code, filename=str(source))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "vllm.v1.request")
    ]
    namespace = {"Request": SimpleNamespace}
    # Deliberately execute the trusted local checkout, not downloaded/user data.
    exec(compile(tree, str(source), "exec"), namespace)  # noqa: S102
    # Substitute the module's clock only; do not change Python's global time module.
    namespace["time"] = SimpleNamespace(time=lambda: 1000.0)
    return namespace


def probe(source):
    namespace = load_queue_namespace(source)
    result = {}
    for wait_limit in (30.0, 0.0):
        queue = namespace["PrefixAwareRequestQueue"](
            scorer=lambda request: request.required_blocks,
            max_wait_s=wait_limit,
        )
        queue.add_request(
            SimpleNamespace(
                request_id="old_expensive",
                arrival_time=960.0,
                required_blocks=6000,
            )
        )
        queue.add_request(
            SimpleNamespace(
                request_id="new_reusable",
                arrival_time=995.0,
                required_blocks=1,
            )
        )
        result[str(wait_limit)] = queue.pop_request().request_id
    assert result == {"30.0": "old_expensive", "0.0": "new_reusable"}
    return {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "method": "Execute actual queue module with Request import replaced by SimpleNamespace and deterministic time=1000. Old request age=40s, cost=6000 blocks; new request age=5s, cost=1 block.",
        "selected_by_max_wait_s": result,
        "limitation": "Confirms selection semantics, not override frequency or latency impact in the recorded GPU runs.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.source)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
