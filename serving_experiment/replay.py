"""Replay sanitized traces against a dedicated vLLM server and record artifacts."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import random
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import aiohttp


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select_sessions(workload, count, max_turns=0, max_tokens=0):
    selected = []
    for session in workload["sessions"]:
        turns = session["turns"][:max_turns] if max_turns else session["turns"]
        # Select whole segments, never silently truncate a prompt.
        if max_tokens and any(
            t["prompt_tokens"] + t["output_tokens"] > max_tokens for t in turns
        ):
            continue
        if len(turns) < 2:
            continue
        selected.append({**session, "turns": turns})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"Only {len(selected)} eligible segments for requested count {count}"
        )
    return selected


async def stream_completion(client, base_url, payload, on_event):
    started = time.perf_counter()
    tokens, usage, ttft, response_id, finish = [], None, None, None, None
    async with client.post(base_url + "/v1/completions", json=payload) as response:
        if response.status != 200:
            raise RuntimeError(
                f"HTTP {response.status}: {(await response.text())[:2000]}"
            )
        async for raw_line in response.content:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            on_event(event)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            response_id = event.get("id", response_id)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                new_tokens = choice.get("token_ids") or []
                if new_tokens and ttft is None:
                    ttft = time.perf_counter() - started
                tokens.extend(new_tokens)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    elapsed = time.perf_counter() - started
    if not usage or usage["completion_tokens"] != payload["max_tokens"]:
        raise RuntimeError(
            f"Incomplete generation: expected {payload['max_tokens']}, usage={usage}"
        )
    if usage["prompt_tokens"] != len(payload["prompt"]):
        raise RuntimeError(
            f"Prompt length changed: {usage['prompt_tokens']} != {len(payload['prompt'])}"
        )
    if len(tokens) != payload["max_tokens"] or any(t != 42 for t in tokens):
        raise RuntimeError("Returned token IDs violate deterministic replay constraint")
    if finish != "length":
        raise RuntimeError(f"Unexpected finish reason: {finish}")
    return {
        "response_id": response_id,
        "latency_s": elapsed,
        "ttft_s": ttft,
        "usage": usage,
        "output_token_ids": tokens,
    }


async def run(args):
    workload = json.loads(args.workload.read_text())
    sessions = select_sessions(workload, args.sessions, args.max_turns, args.max_tokens)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "policy": args.policy,
        "seed": workload["seed"],
        "selected_workload_sha256": digest(sessions),
        "source_workload_sha256": hashlib.sha256(
            args.workload.read_bytes()
        ).hexdigest(),
        "session_ids": [s["id"] for s in sessions],
        "source_sessions": [s["source_session"] for s in sessions],
        "max_turns": args.max_turns,
        "max_tokens_filter": args.max_tokens,
        "delay_scale": args.delay_scale,
        "sampling": {"temperature": 0, "ignore_eos": True, "allowed_token_ids": [42]},
        "methodology": workload["methodology"],
        "metrics": "Client wall latency/TTFT and server Prometheus snapshots; Chrome trace uses client clock, not CUDA kernel profiling.",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    timeline, results, errors = [], [], []
    files = ExitStack()
    trace_file = files.enter_context((args.output / "requests.jsonl").open("w"))
    payload_file = files.enter_context(
        gzip.open(args.output / "payloads.jsonl.gz", "wt")
    )
    stream_file = files.enter_context(
        gzip.open(args.output / "responses.jsonl.gz", "wt")
    )
    metrics_file = files.enter_context((args.output / "metrics.jsonl").open("w"))
    clock_start = time.perf_counter()
    stop_metrics = asyncio.Event()

    def write(handle, record):
        handle.write(json.dumps(record) + "\n")
        handle.flush()

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    # vLLM includes up to 131k prompt IDs in one SSE event. The default
    # StreamReader line limit is smaller than this legitimate JSON payload.
    async with aiohttp.ClientSession(
        timeout=timeout, read_bufsize=4 * 1024 * 1024
    ) as client:

        async def snapshot():
            try:
                async with client.get(
                    args.base_url + "/metrics", timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    write(
                        metrics_file,
                        {
                            "elapsed_s": time.perf_counter() - clock_start,
                            "status": response.status,
                            "text": await response.text(),
                        },
                    )
            except (aiohttp.ClientError, TimeoutError) as exc:
                write(
                    metrics_file,
                    {"elapsed_s": time.perf_counter() - clock_start, "error": str(exc)},
                )

        async def sample_metrics():
            while not stop_metrics.is_set():
                await snapshot()
                try:
                    await asyncio.wait_for(stop_metrics.wait(), 2)
                except TimeoutError:
                    pass

        async def replay_session(session, lane):
            sid = session["id"]
            rng = random.Random(f"{workload['seed']}:{sid}")
            prefix, session_results = [], []
            for index, turn in enumerate(session["turns"]):
                request_key = f"{sid}:{index}"
                if len(prefix) > turn["prompt_tokens"]:
                    raise RuntimeError(f"Context shrank unexpectedly in {request_key}")
                prefix.extend(
                    rng.randrange(1000, 20000)
                    for _ in range(turn["prompt_tokens"] - len(prefix))
                )
                payload = {
                    "model": args.model,
                    "prompt": list(prefix),
                    "max_tokens": turn["output_tokens"],
                    "temperature": 0,
                    "ignore_eos": True,
                    "allowed_token_ids": [42],
                    "return_token_ids": True,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "vllm_xargs": {
                        "cachewise_next_tools": json.dumps(turn["next_tools"])
                    },
                }
                write(payload_file, {"request_key": request_key, "body": payload})
                started = time.perf_counter()
                relative_start = started - clock_start
                write(
                    trace_file,
                    {
                        "event": "request_start",
                        "request_key": request_key,
                        "elapsed_s": relative_start,
                        "prompt_sha256": digest(prefix),
                        "prompt_tokens": len(prefix),
                        "output_tokens": turn["output_tokens"],
                        "source_message": turn["message_id"],
                    },
                )

                def on_event(event, request_key=request_key):
                    write(
                        stream_file,
                        {
                            "request_key": request_key,
                            "elapsed_s": time.perf_counter() - clock_start,
                            "response": event,
                        },
                    )

                try:
                    result = await stream_completion(
                        client, args.base_url, payload, on_event
                    )
                except Exception as exc:
                    write(
                        trace_file,
                        {
                            "event": "request_error",
                            "request_key": request_key,
                            "error": str(exc),
                            "elapsed_s": time.perf_counter() - clock_start,
                        },
                    )
                    raise RuntimeError(f"{request_key}: {exc}") from exc
                result.update(
                    {
                        "request_key": request_key,
                        "session": sid,
                        "turn": index,
                        "start_s": relative_start,
                    }
                )
                prefix.extend(result.pop("output_token_ids"))
                write(trace_file, {"event": "request_end", **result})
                session_results.append(result)
                timeline.append(
                    {
                        "name": "LLM request",
                        "cat": args.policy,
                        "ph": "X",
                        "ts": relative_start * 1e6,
                        "dur": result["latency_s"] * 1e6,
                        "pid": 1,
                        "tid": lane,
                        "args": result,
                    }
                )
                if index + 1 < len(session["turns"]) and turn["delay_after_s"] > 0:
                    delay = turn["delay_after_s"] * args.delay_scale
                    sleep_start = time.perf_counter() - clock_start
                    await asyncio.sleep(delay)
                    timeline.append(
                        {
                            "name": "Tool/human delay",
                            "cat": "external",
                            "ph": "X",
                            "ts": sleep_start * 1e6,
                            "dur": (time.perf_counter() - clock_start - sleep_start)
                            * 1e6,
                            "pid": 1,
                            "tid": lane,
                            "args": {
                                "source_duration_s": turn["delay_after_s"],
                                "tools": turn["next_tools"],
                            },
                        }
                    )
            results.append(
                {
                    "id": sid,
                    "requests": len(session_results),
                    "llm_time_s": sum(r["latency_s"] for r in session_results),
                    "output_tokens": sum(
                        r["usage"]["completion_tokens"] for r in session_results
                    ),
                    "requests_detail": session_results,
                }
            )
            print(f"Completed {sid}: {len(session_results)} requests", flush=True)

        monitor = asyncio.create_task(sample_metrics())
        try:
            # All N segments start together; each issues the next request only
            # after generation and its recorded external delay have completed.
            outcomes = await asyncio.gather(
                *(replay_session(s, i) for i, s in enumerate(sessions)),
                return_exceptions=True,
            )
            errors.extend(str(x) for x in outcomes if isinstance(x, BaseException))
        finally:
            stop_metrics.set()
            await monitor
            await snapshot()
            files.close()
    elapsed = time.perf_counter() - clock_start
    summary = {
        "status": "failed" if errors else "complete",
        "elapsed_s": elapsed,
        "errors": errors,
        "completed_sessions": len(results),
        "sessions": results,
        "output_tokens_per_s": sum(r["output_tokens"] for r in results) / elapsed,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "timeline.json").write_text(
        json.dumps({"traceEvents": timeline}) + "\n"
    )
    manifest.update(
        {
            "status": summary["status"],
            "finished_utc": datetime.now(UTC).isoformat(),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sessions"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:18080")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    p.add_argument(
        "--policy",
        choices=[
            "baseline",
            "cachewise",
            "prefix_only",
            "eviction_only",
            "human_first",
            "queued_first",
        ],
        required=True,
    )
    p.add_argument("--sessions", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--delay-scale", type=float, default=1.0)
    p.add_argument("--request-timeout", type=float, default=1800)
    args = p.parse_args()
    if (
        args.sessions < 1
        or args.delay_scale < 0
        or args.max_turns < 0
        or args.max_tokens < 0
    ):
        p.error("Invalid replay limits")
    asyncio.run(run(args))
