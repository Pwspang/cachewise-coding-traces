"""Observe queue choices without extra scoring or changes to selection state."""

import json
import os
import time


def install(queue_class, output_prefix, clock=time.time):
    original_select = queue_class._select_request
    original_pop = queue_class.pop_request
    counters = {
        "fresh_choices": 0,
        "aging_choices": 0,
        "pops": 0,
        "aging_pops": 0,
        "observer_seconds": 0.0,
    }

    def select(self):
        started = time.perf_counter()
        fresh = self._selected is None and bool(self._requests)
        aged = False
        if fresh:
            head = next(iter(self._requests.values()))
            aged = (
                self._scorer is not None
                and self._max_wait_s > 0
                and clock() - head.arrival_time > self._max_wait_s
            )
        counters["observer_seconds"] += time.perf_counter() - started
        result = original_select(self)
        started = time.perf_counter()
        if fresh:
            counters["fresh_choices"] += 1
            counters["aging_choices"] += int(aged)
            self._queue_observed_aging = aged
        counters["observer_seconds"] += time.perf_counter() - started
        return result

    def pop(self):
        result = original_pop(self)
        started = time.perf_counter()
        counters["pops"] += 1
        counters["aging_pops"] += int(getattr(self, "_queue_observed_aging", False))
        with open(f"{output_prefix}-{os.getpid()}.jsonl", "a") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "queue_pop",
                        "request_id": result.request_id,
                        "arrival_time": result.arrival_time,
                        "time": clock(),
                        "max_wait_s": self._max_wait_s,
                        "aging_choice": getattr(self, "_queue_observed_aging", False),
                        **counters,
                    }
                )
                + "\n"
            )
        counters["observer_seconds"] += time.perf_counter() - started
        return result

    queue_class._select_request = select
    queue_class.pop_request = pop
    return counters
