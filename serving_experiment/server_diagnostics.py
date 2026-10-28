"""Opt-in observation hooks; human-first is an explicit diagnostic intervention.

No vLLM source files are changed. Hooks run in spawned engine processes too.
"""

import json
import os
import time
from collections import Counter

from vllm.v1.core.cachewise.manager import CacheWiseManager
from vllm.v1.core.cachewise.session_tracker import SessionTracker
from vllm.v1.core.sched.scheduler import Scheduler

_handle = None
_write_seconds = 0.0


def emit(kind, **data):
    global _handle, _write_seconds
    started = time.perf_counter()
    path = os.environ.get("CACHEWISE_DIAGNOSTICS_PATH")
    if not path:
        return
    if _handle is None:
        _handle = open(f"{path}-{os.getpid()}.jsonl", "a", buffering=1024 * 1024)
    _handle.write(
        json.dumps(
            {
                "event": kind,
                "time": time.time(),
                "logging_seconds": _write_seconds,
                **data,
            }
        )
        + "\n"
    )
    if kind != "eviction":
        _handle.flush()
    _write_seconds += time.perf_counter() - started


original_priorities = SessionTracker.priorities


def priorities(self, now):
    result = original_priorities(self, now)
    if os.environ.get("CACHEWISE_HUMAN_FIRST") == "1":
        # Paper section 6.1: idle/human sessions are evicted before sessions
        # executing tools. No knowledge of future requests is supplied.
        for sid, session in self.sessions.items():
            if not session.in_flight and session.pending_tools == []:
                result[sid] = 1e12
        self._last_priorities = result
    for sid in getattr(self, "_diagnostic_pending", {}).values():
        if sid in result:
            result[sid] = 0.0
    self._last_priorities = result
    return result


SessionTracker.priorities = priorities

original_add_request = Scheduler.add_request


def add_request(self, request):
    is_new = request.request_id not in self.requests
    original_add_request(self, request)
    if (
        os.environ.get("CACHEWISE_QUEUED_FIRST") != "1"
        or not is_new
        or self.cachewise is None
    ):
        return
    tracker = self.cachewise.tracker
    session = tracker._find_session(request.block_hashes)
    if session is None:
        return
    if not hasattr(tracker, "_diagnostic_pending"):
        tracker._diagnostic_pending = {}
    tracker._diagnostic_pending[request.request_id] = session.session_id
    self.cachewise.queue.rebuild(tracker.priorities(tracker.time_fn()))
    emit(
        "queued_protection",
        request_id=request.request_id,
        sid=session.session_id,
        prompt_tokens=request.num_prompt_tokens,
    )


Scheduler.add_request = add_request


def snapshot(manager):
    tracker = manager.tracker
    now = tracker.time_fn()
    return [
        {
            "sid": sid,
            "request_id": s.last_request_id,
            "in_flight": s.in_flight,
            "tools": s.pending_tools,
            "elapsed_s": max(0, now - s.last_finish_ts),
            "priority_s": tracker.session_priority(sid),
            "blocks": len(s.block_ids),
        }
        for sid, s in tracker.sessions.items()
    ]


original_create = CacheWiseManager.create_free_queue


def create_queue(self, blocks):
    queue = original_create(self, blocks)
    original_pop = queue.popleft_n
    original_evict = queue._on_evicted_fn
    victims = Counter()

    def evicted(block_id, sid):
        victims[sid] += 1
        original_evict(block_id, sid)

    def pop(n):
        victims.clear()
        result = original_pop(n)
        if victims:
            emit("eviction", blocks=n, victims=dict(victims), sessions=snapshot(self))
        return result

    queue._on_evicted_fn = evicted
    queue.popleft_n = pop
    emit(
        "init",
        human_first=os.environ.get("CACHEWISE_HUMAN_FIRST") == "1",
        bootstrap_stats=self.predictor.stats(),
    )
    return queue


CacheWiseManager.create_free_queue = create_queue
original_scheduled = CacheWiseManager.on_request_scheduled
original_finished = CacheWiseManager.on_request_finished


def scheduled(self, request):
    original_scheduled(self, request)
    getattr(self.tracker, "_diagnostic_pending", {}).pop(request.request_id, None)
    emit(
        "admitted",
        request_id=request.request_id,
        prompt_tokens=request.num_prompt_tokens,
        arrival_time=request.arrival_time,
        sid=self.tracker.by_request_id.get(request.request_id),
        sessions=snapshot(self),
    )


def finished(self, request, block_ids):
    getattr(self.tracker, "_diagnostic_pending", {}).pop(request.request_id, None)
    original_finished(self, request, block_ids)
    emit(
        "finished",
        request_id=request.request_id,
        prompt_tokens=request.num_prompt_tokens,
        sessions=snapshot(self),
    )


CacheWiseManager.on_request_scheduled = scheduled
CacheWiseManager.on_request_finished = finished


if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
