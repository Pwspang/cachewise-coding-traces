import asyncio
import json

import pytest
from aiohttp import web
from prepare import human_boundary_messages, prepare
from replay import run, select_sessions, stream_completion
from summarize import summarize


def test_paired_repeat_summary_and_mismatched_workload_rejection(tmp_path):
    for policy, times in {"baseline": [10, 12], "cachewise": [8, 9]}.items():
        for repeat, duration in enumerate(times):
            path = tmp_path / f"{policy}-r{repeat}"
            path.mkdir()
            manifest = {
                "status": "complete",
                "policy": policy,
                "model": "test",
                "selected_workload_sha256": "same",
                "sampling": {},
                "delay_scale": 1,
            }
            (path / "manifest.json").write_text(json.dumps(manifest))
            request = {
                "latency_s": duration,
                "ttft_s": 1,
                "usage": {
                    "prompt_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 50},
                },
            }
            summary = {
                "status": "complete",
                "elapsed_s": duration,
                "output_tokens_per_s": 1,
                "sessions": [{"llm_time_s": duration, "requests_detail": [request]}],
            }
            (path / "summary.json").write_text(json.dumps(summary))
    summarize(tmp_path)
    aggregate = json.loads((tmp_path / "comparison-summary.json").read_text())
    assert aggregate["baseline"]["mean_segment_llm_time_s"] == 11
    assert aggregate["cachewise"]["mean_segment_llm_time_s"] == 8.5
    assert aggregate["baseline_over_cachewise_llm_time_ratio"] == pytest.approx(
        11 / 8.5
    )
    path = tmp_path / "cachewise-r1/manifest.json"
    bad = json.loads(path.read_text())
    bad["selected_workload_sha256"] = "different"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="workloads differ"):
        summarize(tmp_path)


def test_human_boundary_uses_user_input_not_tool_results(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [
        {"type": "assistant", "message": {"id": "m1"}},
        {"type": "user", "message": {"content": [{"type": "tool_result"}]}},
        {"type": "assistant", "message": {"id": "m2"}},
        {"type": "user", "message": {"content": "<redacted>"}},
        {"type": "assistant", "message": {"id": "m3"}},
    ]
    raw.write_text("\n".join(json.dumps(r) for r in rows))
    assert human_boundary_messages(raw) == {"m2"}


def test_prepare_deduplicates_and_keeps_train_out(tmp_path):
    root = tmp_path / "input"
    split = root / "tool_duration_prediction/datasets/splits/train_sessions.json"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({"train_sessions": ["train"]}))
    for sid in ("train", "eval"):
        raw = root / f"raw_traces/project/{sid}.jsonl"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(
            json.dumps({"type": "assistant", "message": {"id": "m1"}}) + "\n"
        )
        path = root / f"parsed_traces/project/{sid}/events.json"
        path.parent.mkdir(parents=True)

        def llm(mid, seconds, count):
            return {
                "event_type": "llm_call",
                "message_id": mid,
                "timestamp": f"2020-01-01T00:00:{seconds:02d}+00:00",
                "model": "claude",
                "input_tokens": 10,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
                "output_tokens": count,
            }

        path.write_text(
            json.dumps(
                [
                    llm("m1", 0, 1),
                    llm("m1", 1, 4),
                    {
                        "event_type": "tool_call",
                        "associated_llm_call_id": "m1",
                        "tool_name": "Read",
                        "tool_input": {},
                        "timestamp": "2020-01-01T00:00:01+00:00",
                        "result_timestamp": "2020-01-01T00:00:03+00:00",
                        "execution_time_ms": 2000,
                    },
                ]
            )
        )
    output = tmp_path / "prepared"
    prepare(root, output)
    data = json.loads((output / "workload.json").read_text())
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["source_session"] == "eval"
    assert data["sessions"][0]["turns"][0]["prompt_tokens"] == 60
    assert data["sessions"][0]["turns"][0]["output_tokens"] == 4
    assert len((output / "bootstrap.jsonl").read_text().splitlines()) == 1


def test_session_selection_does_not_truncate_prompts():
    workload = {
        "sessions": [
            {"id": "long", "turns": [{"prompt_tokens": 9000, "output_tokens": 10}] * 2},
            {"id": "short", "turns": [{"prompt_tokens": 100, "output_tokens": 10}] * 2},
        ]
    }
    assert select_sessions(workload, 1, max_tokens=8192)[0]["id"] == "short"


def test_stream_recorder_checks_token_counts_and_collects_usage():
    async def scenario():
        import aiohttp

        async def handler(request):
            body = await request.json()
            assert body["prompt"] == [1001, 1002]
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for event in [
                {"id": "test", "choices": [{"token_ids": [42], "finish_reason": None}]},
                {
                    "id": "test",
                    "choices": [{"token_ids": [42], "finish_reason": "length"}],
                },
                {
                    "id": "test",
                    "choices": [],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                },
            ]:
                await response.write(("data: " + json.dumps(event) + "\n\n").encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as client:
                events = []
                result = await stream_completion(
                    client,
                    f"http://127.0.0.1:{port}",
                    {"prompt": [1001, 1002], "max_tokens": 2},
                    events.append,
                )
                assert result["output_token_ids"] == [42, 42]
                assert result["ttft_s"] is not None
                assert len(events) == 3
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


@pytest.mark.parametrize("prompt_lengths", [(100, 110), (130000, 130010)])
def test_end_to_end_recorder_preserves_prefixes_and_artifacts(tmp_path, prompt_lengths):
    from argparse import Namespace

    async def scenario():
        requests = []

        async def handler(request):
            body = await request.json()
            requests.append(body)
            count = body["max_tokens"]
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            event = {
                "id": f"completion-{len(requests)}",
                "prompt_token_ids": body["prompt"],
                "choices": [{"token_ids": [42] * count, "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": len(body["prompt"]),
                    "completion_tokens": count,
                },
            }
            await response.write(
                ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
            )
            return response

        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        app.router.add_get("/metrics", lambda _: web.Response(text="test_metric 1\n"))
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        workload = tmp_path / "workload.json"
        turns = [
            {
                "message_id": "m1",
                "prompt_tokens": prompt_lengths[0],
                "output_tokens": 2,
                "delay_after_s": 0,
                "next_tools": [{"name": "Read", "args": "<redacted>"}],
            },
            {
                "message_id": "m2",
                "prompt_tokens": prompt_lengths[1],
                "output_tokens": 3,
                "delay_after_s": 0,
                "next_tools": [],
            },
        ]
        workload.write_text(
            json.dumps(
                {
                    "seed": 42,
                    "methodology": "test",
                    "sessions": [{"id": "s1", "source_session": "s1", "turns": turns}],
                }
            )
        )
        args = Namespace(
            workload=workload,
            output=tmp_path / "run",
            sessions=1,
            max_turns=0,
            max_tokens=0,
            model="test",
            policy="cachewise",
            delay_scale=1,
            request_timeout=5,
            base_url=f"http://127.0.0.1:{port}",
        )
        try:
            await run(args)
        finally:
            await runner.cleanup()
        assert requests[1]["prompt"][: prompt_lengths[0] + 2] == requests[0][
            "prompt"
        ] + [42, 42]
        summary = json.loads((args.output / "summary.json").read_text())
        assert summary["status"] == "complete"
        assert summary["sessions"][0]["requests"] == 2
        assert (
            len(json.loads((args.output / "timeline.json").read_text())["traceEvents"])
            == 2
        )
        import gzip

        with gzip.open(args.output / "payloads.jsonl.gz", "rt") as handle:
            assert len(handle.readlines()) == 2

    asyncio.run(scenario())
