"""Run each policy in a fresh server, retaining server logs and replay traces."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def serve(output, name, command, env, base_url):
    """Run one fresh server with GPU logging; stop it on exit."""
    with (
        (output / f"{name}-server.log").open("w") as log,
        (output / f"{name}-gpu.csv").open("w") as gpu_log,
    ):
        gpu_monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw",
                "--format=csv",
                "--loop=2",
            ],
            stdout=gpu_log,
            stderr=subprocess.STDOUT,
        )
        server = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            deadline = time.monotonic() + 1200
            last_health_error = "Health endpoint did not return HTTP 200"
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"{name} server exited {server.returncode}; inspect its log"
                    )
                try:
                    with urllib.request.urlopen(
                        base_url + "/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError) as exc:
                    last_health_error = str(exc)
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Server startup exceeded 20 minutes: {last_health_error}"
                    )
                time.sleep(2)
            yield
        finally:
            gpu_monitor.terminate()
            gpu_monitor.wait(timeout=10)
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
    time.sleep(5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--sessions", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--workload", type=Path)
    p.add_argument(
        "--policies",
        nargs="+",
        choices=[
            "baseline",
            "cachewise",
            "infercept",
            "prefix_only",
            "eviction_only",
            "human_first",
            "queued_first",
        ],
    )
    p.add_argument(
        "--infercept-profile",
        type=Path,
        help="Reuse a recompute profile instead of profiling a fresh LRU server.",
    )
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--num-gpu-blocks", type=int)
    a = p.parse_args()
    if set(a.policies or ()) & {"human_first", "queued_first"} and not a.diagnostics:
        p.error("Policy interventions require --diagnostics")
    tracker_source = (
        a.root / "src/vllm-cachewise/vllm/v1/core/cachewise/session_tracker.py"
    )
    if a.diagnostics and "def on_request_queued(" in tracker_source.read_text():
        p.error(
            "Historical diagnostic hooks require the unpatched fork; use normal baseline/cachewise policies with the integrated fixes."
        )
    a.output.mkdir(parents=True, exist_ok=False)
    model = "Qwen/Qwen2.5-Coder-14B-Instruct"
    base_url = f"http://127.0.0.1:{a.port}"
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", a.port)) == 0:
            raise SystemExit(f"Port {a.port} is already in use")
    revision = subprocess.check_output(
        ["git", "-C", str(a.root / "src/vllm-cachewise"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv",
        ],
        text=True,
    )
    packages = subprocess.check_output(
        [
            str(Path.home() / ".local/bin/uv"),
            "pip",
            "freeze",
            "--python",
            sys.executable,
        ],
        text=True,
    )
    (a.output / "environment.json").write_text(
        json.dumps(
            {
                "vllm_revision": revision,
                "host": socket.gethostname(),
                "gpu": gpu,
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "python": sys.version,
                "model": model,
                "smoke": a.smoke,
            },
            indent=2,
        )
    )
    (a.output / "packages.txt").write_text(packages)
    shutil.copytree(
        a.root / "serving_experiment",
        a.output / "runner_source",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    (a.output / "vllm-source.patch").write_text(
        subprocess.check_output(
            ["git", "-C", str(a.root / "src/vllm-cachewise"), "diff", "HEAD"], text=True
        )
    )
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        model,
        revision="aedcc2d42b622764e023cf882b6652e646b95671",
        local_files_only=True,
    )
    (a.output / "model_snapshot.txt").write_text(model_path + "\n")

    def server_command(policy):
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            model_path,
            "--served-model-name",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(a.port),
            "--tensor-parallel-size",
            "1",
            "--dtype",
            "bfloat16",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens",
            "512",
            "--max-num-seqs",
            "32",
            "--max-model-len",
            "131072",
            "--hf-overrides",
            '{"rope_parameters":{"rope_type":"yarn","rope_theta":1000000.0,"factor":4.0,"original_max_position_embeddings":32768}}',
            "--gpu-memory-utilization",
            "0.85",
            "--generation-config",
            "vllm",
            "--enable-prompt-tokens-details",
            "--kv-cache-metrics",
            "--kv-cache-metrics-sample",
            "0.01",
        ]
        if a.diagnostics:
            command[1:3] = [str(a.root / "serving_experiment/server_diagnostics.py")]
        if a.num_gpu_blocks:
            command.extend(["--num-gpu-blocks-override", str(a.num_gpu_blocks)])
        predictive = policy in {
            "cachewise",
            "eviction_only",
            "human_first",
            "queued_first",
        }
        prefix_aware = policy in {
            "cachewise",
            "prefix_only",
            "human_first",
            "queued_first",
        }
        command.extend(
            ["--scheduling-policy", "prefix_aware" if prefix_aware else "fcfs"]
        )
        if predictive:
            command.extend(
                [
                    "--kv-cache-eviction-policy",
                    "predictive",
                    "--cachewise-predictor",
                    "tool_name",
                    "--cachewise-bootstrap-path",
                    str(a.root / "prepared/bootstrap.jsonl"),
                ]
            )
        elif policy == "infercept":
            # InferCept resumes requests FCFS and estimates interception time
            # online; it uses neither the tool predictor nor its bootstrap.
            command.extend(
                [
                    "--kv-cache-eviction-policy",
                    "infercept",
                    "--infercept-recompute-profile",
                    str(profile_path),
                ]
            )
        else:
            command.extend(["--kv-cache-eviction-policy", "lru"])
        return command

    profile_path = a.output / "recompute-profile.json"
    if "infercept" in (a.policies or ()):
        if a.infercept_profile:
            shutil.copyfile(a.infercept_profile, profile_path)
        else:
            # Offline T_rec(C) profile on an identically configured LRU server.
            command = server_command("baseline")
            (a.output / "profile-command.json").write_text(
                json.dumps(command, indent=2)
            )
            print("Profiling recompute latency", flush=True)
            with serve(a.output, "profile", command, os.environ.copy(), base_url):
                subprocess.run(
                    [
                        sys.executable,
                        str(a.root / "serving_experiment/profile_recompute.py"),
                        "--base-url",
                        base_url,
                        "--model",
                        model,
                        "--output",
                        str(profile_path),
                    ],
                    check=True,
                )
    for repeat in range(a.repeats):
        # Rotate order on repeats to expose warm filesystem / thermal effects.
        policies = a.policies or ["baseline", "cachewise"]
        offset = repeat % len(policies)
        for policy in policies[offset:] + policies[:offset]:
            name = f"{policy}-r{repeat}"
            command = server_command(policy)
            (a.output / f"{name}-command.json").write_text(
                json.dumps(command, indent=2)
            )
            print(f"Starting {name}", flush=True)
            server_env = os.environ.copy()
            if a.diagnostics:
                server_env["CACHEWISE_DIAGNOSTICS_PATH"] = str(
                    a.output / f"{name}-decisions"
                )
                server_env["CACHEWISE_HUMAN_FIRST"] = (
                    "1" if policy == "human_first" else "0"
                )
                server_env["CACHEWISE_QUEUED_FIRST"] = (
                    "1" if policy == "queued_first" else "0"
                )
                (a.output / f"{name}-diagnostics.json").write_text(
                    json.dumps(
                        {
                            "human_first": policy == "human_first",
                            "queued_first": policy == "queued_first",
                            "diagnostics": True,
                        },
                        indent=2,
                    )
                )
            elif policy in {"human_first", "queued_first"}:
                raise ValueError("Policy intervention requires --diagnostics")
            with serve(a.output, name, command, server_env, base_url):
                replay = [
                    sys.executable,
                    str(a.root / "serving_experiment/replay.py"),
                    "--workload",
                    str(a.workload or a.root / "prepared/workload.json"),
                    "--output",
                    str(a.output / name),
                    "--base-url",
                    base_url,
                    "--policy",
                    policy,
                    "--sessions",
                    str(2 if a.smoke else a.sessions),
                ]
                if a.smoke:
                    replay.extend(["--max-turns", "2", "--max-tokens", "32768"])
                subprocess.run(replay, check=True)
    (a.output / "COMPLETE").write_text(
        "All requested policy runs completed successfully.\n"
    )

if __name__ == "__main__":
    main()
