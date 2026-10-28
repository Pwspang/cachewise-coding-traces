# CacheWise serving replay

This rerun uses `chughtapan/vllm` branch `cachewise`, pinned to
`a4b3affa7f746db57c2f19de839f72094c8f3ce1`, with
`Qwen/Qwen2.5-Coder-14B-Instruct` on one RTX Pro 6000 Blackwell GPU.
The upstream kernel-wheel base is
`fe042382925000e5adfe530a1cc2b91d7a125fd5`.

## Applied fixes

`patches/cachewise-session-lifecycle.patch` applies to the pinned fork above.
`setup_ccds.sh` applies it idempotently and rejects an incompatible checkout.
Its SHA-256 is `8ab7dae12954b37ac64e408f8ea3cb137685a8725811514ada69099be0015ad2`.

- Confirmed idle human sessions receive the lowest retention priority, ahead
  of tool-running sessions for eviction. Priority refreshes happen at lifecycle
  transitions, including before newly freed blocks enter the eviction queue.
- Returning sessions are protected when queued, before admission allocates
  cache blocks. Cancellation releases that protection without deleting the
  previous turn's cached-prefix metadata. Multiple queued/active requests are
  tracked independently.
- Predictor observations use request arrival time, excluding server queueing
  from tool-duration measurements.
- Workload preparation splits at actual user-input boundaries by default.
  `--legacy-human-pauses` retains the historical 300-second heuristic only for
  reproducing older runs.

Validation: 58 CacheWise engine tests and seven replay/summary tests passed.
Applicable source pre-commit checks passed. These validate behavior, not a
performance claim. The integrated comparison is CCDS job 9494: ten complete
held-out segments, 83 requests and 15,235 output tokens per policy run, two
repeats in baseline/CacheWise then CacheWise/baseline order. Every run uses
17,981 KV blocks, a fresh server, and the same selected workload and seed.

## Fixed comparison: CCDS job 9494, 2026-10-28

Slurm completed successfully on node06 (exit 0), using one RTX Pro 6000 and
Qwen2.5-Coder-14B-Instruct. Fixed CacheWise beat the LRU/FCFS baseline in both
paired repeats on this corrected replay.

| Metric | Baseline | Fixed CacheWise |
| --- | ---: | ---: |
| Mean segment LLM time, repeat 0 | 575.64 s | 542.33 s |
| Mean segment LLM time, repeat 1 | 579.54 s | 548.22 s |
| Mean across repeats | 577.59 s | 545.27 s |
| Mean prompt-token cache-hit fraction | 64.69% | 68.72% |
| Mean uncached prompt tokens per run | 1,728,277 | 1,531,021 |
| Mean replay wall time, including external waits | 874.76 s | 814.69 s |
| Median request TTFT, repeat 0 | 4.666 s | 3.799 s |
| Median request TTFT, repeat 1 | 4.823 s | 6.479 s |

Mean segment LLM time fell **5.6%** overall (1.059x baseline/CacheWise ratio),
with reductions of 5.8% and 5.4% in the individual pairs. Mean replay wall
time fell 6.9%, and uncached prompt tokens fell 11.4%. TTFT did not improve
consistently: the second CacheWise run had a higher median TTFT. The result
supports a modest aggregate latency improvement, not an improvement in every
metric or every session.

All 332 requests completed successfully. Every run used identical prompt
hashes and requested lengths: 4,894,853 prompt tokens and 15,235 output tokens
per run, with actual usage matching those counts. All gzip recordings passed
the integrity audit, all 1,689 metric snapshots succeeded, and all four runs
reported zero preemptions. The recorded source patch matches the SHA-256 above.

This is a two-repeat result on one selected synthetic workload, not a
statistical or full-paper replication. The 14B/single-RTX setup, redacted
inputs, tool-name predictor, and published training split still differ from
the paper. The corrected ten-segment workload differs from job 9465's original
five segments, so absolute timings between those jobs are not comparable.
This benchmark also does not isolate each fix's individual contribution.

Local artifacts: `runs/ccds-9494/fixed-comparison-9494/`, including
`comparison-summary.json`, `comparison.json`, `repeat-trace-audit.json`,
`trace_overview.png` (repeat 0), `timeline-comparison.json` (repeat 0), and all
four runs' raw request/response/metrics/timeline traces. Remote artifacts are
under `~/cachewise-rerun/runs/fixed-comparison-9494` on `ccds-ai`. The test log,
smoke traces, and Slurm log are also saved locally under `runs/ccds-9494/`.

## Recorded run: CCDS job 9465, 2026-10-28

Slurm completed successfully on node06 using one RTX Pro 6000 Blackwell
Max-Q Workstation Edition (96 GB class). Both policies completed 118 requests
across five held-out segments, generating 7,741 tokens each. The paired audit
verified identical prompt hashes and requested token lengths for every request.
All 624 Prometheus snapshots succeeded, and neither policy had preemptions.

| Metric | Baseline: LRU / FCFS | CacheWise: predictive / prefix-aware |
| --- | ---: | ---: |
| Mean segment LLM time, excluding external waits | 171.88 s | 189.69 s |
| Median request latency | 1.395 s | 1.611 s |
| Median time to first token | 0.463 s | 0.514 s |
| Prompt-token cache-hit fraction (request usage) | 94.56% | 93.44% |
| Uncached prompt tokens | 466,047 | 561,567 |
| Total replay wall time, including external waits | 621.90 s | 626.85 s |

CacheWise's mean segment LLM time was 10.4% higher in this single comparison.
It processed 95,520 more uncached prompt tokens. The largest segment latency
increase was `session_0067_006` (+41.94 s), followed by `session_0067_001`
(+25.71 s). These traces show no improvement for this configuration; one pair
of runs does not establish a general performance conclusion or isolate the
cause of the difference.

Local artifacts are under `runs/ccds-9465/comparison-9465/`:
`comparison.json`, `trace_audit.json`, `trace_overview.png`, and the combined
`timeline-comparison.json`, plus each policy's raw recordings and provenance.
The remote copy is `~/cachewise-rerun/runs/comparison-9465` on `ccds-ai`.
Runs under job 9463 include a failed long-stream attempt and are excluded.

## Regression investigation

Job 9473 replayed a 74-request prefix of the same five segments, retaining
the three requests responsible for 87,376 of the original 95,520 extra
uncached prompt tokens. Every variant used the same inputs and 17,981 KV
blocks (287,696 tokens). This is a targeted diagnostic subset, not a new
representative benchmark. The vLLM checkout was unchanged; opt-in observation
and intervention hooks are in `server_diagnostics.py`.

| Diagnostic configuration | Mean segment LLM time | Uncached prompt tokens |
| --- | ---: | ---: |
| LRU / FCFS baseline | 150.12 s | 418,209 |
| LRU / prefix-aware scheduling | 156.44 s | 429,953 |
| Current CacheWise | 169.83 s | 512,481 |
| CacheWise, evict idle no-tool sessions first | 151.41 s | 371,329 |
| CacheWise, protect returning sessions at arrival | 163.18 s | 453,425 |

The idle-session intervention removed most of the diagnostic slowdown,
without changing the workload, model, cache capacity, or training samples.
The unpatched fork predicted reuse for an idle no-tool session through a learned
human/global duration distribution. In a minimal reproduction with the actual
bootstrap, it predicts 16.68 seconds for an idle human session versus 28.67
seconds for Bash and evicts Bash first. The paper explicitly states that idle
human sessions are evicted before tool-running sessions
([section 6.1](https://arxiv.org/html/2606.16824v1)).

Decision traces also expose an admission-timing problem: session protection
is applied after allocation/admission, not at arrival. In the current-CacheWise
diagnostic run, 4,205 session-tagged block eviction events happened while the
next request was already waiting. Another 2,152 happened less than three
seconds before the next arrival. These are eviction events, not unique block
identities. The observer's cumulative serialization/write time was 0.040 s.
The separate arrival-protection intervention (job 9477) reduced these
already-waiting eviction events from 4,205 to 242 and mean segment LLM time
from 169.83 to 163.18 seconds. It ran on the same physical GPU UUID as job
9473, with identical request hashes and cache capacity. It was tested
independently of the idle-session intervention; no combined result is claimed.

There were reproduction limitations as well:

- The initial replay split only no-tool gaps of at least 300 seconds. Raw
  records confirm that `session_0076_003` contains actual user input after
  pauses of 79.507 and 49.153 seconds that were retained in one segment.
  The paper splits at human pauses. `prepare.py --split-human-pauses` now
  detects raw user-input boundaries while excluding tool results; prepared
  output is in `runs/prepared-paper-boundaries`. It is the default for the
  integrated fixed comparison; older recorded experiments are unchanged.
- Argument redaction prevents the paper's argument-sensitive predictor from
  being reproduced. The rerun used the weaker tool-name predictor. The
  bootstrap includes real timestamp gaps above 25 hours. For example, its
  Bash distribution predicts 353 additional seconds after 58 seconds have
  elapsed, while an affected replay call finished after 58.411 seconds.
  This is a bootstrap-only forecast, not a claim that every online prediction
  has the same value. No outliers were silently removed.
- The bootstrap contains individual-tool rather than tool-batch samples.
  This is a general preparation limitation, but all 67 tool-bearing
  transitions in the diagnostic subset had trained keys; its two missing
  keys were human pauses. Batch-key coverage therefore does not explain
  this observed regression.

All 44 focused upstream predictor/tracker/eviction tests passed. The replay
tests and intervention lifecycle checks passed too. Passing tests do not
establish fidelity to the paper: the human-idle policy difference and the
late protection hook are behavior gaps not covered by those tests.

Artifacts: `runs/diagnosis/diagnostic-9473/diagnostic_comparison.json`,
`diagnostic_comparison.png`, `predictor-probe.json`, and
`cachewise-eviction-audit.json`, plus the raw engine-decision streams.
The fifth variant's raw traces and eviction audit are in
`runs/diagnosis/arrival-9477/`. All 370 diagnostic requests completed with no
errors. Both Slurm jobs exited successfully. To regenerate the combined plot:

```bash
.venv/bin/python serving_experiment/analyze_diagnostics.py \
  runs/diagnosis/diagnostic-9473 --additional-root runs/diagnosis/arrival-9477
```

These historical interventions were opt-in diagnostic hooks. The integrated
source patch above now applies the lifecycle fixes without those hooks.
The five comparisons use one run per variant; they support a diagnosis of
this regression, not a general claim about CacheWise performance.

## Methodology

- This is an adapted serving experiment, not a reproduction of the paper's
  32B model / two-H200 results. The released prompts and tool arguments are
  redacted; no original text can be replayed.
- Group parsed LLM records by session and message ID. Prompt size includes
  input, cache-creation, and cache-read tokens. Decode size is the maximum
  output-token usage reported for that message.
- Generate seeded synthetic prompt token IDs and retain the entire prior
  prompt and generated output as the next turn's prefix. Generation is
  constrained to token 42, with EOS ignored, so both policies see identical
  token sequences and decode lengths. This is a performance workload, not
  an assessment of model answer quality.
- Replay sessions concurrently and turns sequentially. Tool delay runs from
  the first tool launch to the last result (overlapping tools are not summed).
  Actual human-input boundaries split segments. Context shrink and gaps
  >=300 seconds also split segments. The legacy preparation used only the
  gap threshold and context shrink.
- Exclude synthetic-model records, zero-token requests, and requests beyond
  131,072 total tokens, splitting segments at exclusions. Record these counts.
  No prompt is silently shortened to fit the model.
- Retain the release's 34 training sessions, excluding them from evaluation.
  This differs from the paper's stated 80/20 split. Predictor bootstrap samples
  come exclusively from training sessions. The `tool_name` predictor is used
  because all released tool arguments are redacted. No ground-truth duration
  hint is supplied to the server. Tool-name hints accompany both policies.
- The server uses BF16, TP=1, prefix caching, 512-token chunked prefill,
  85% GPU memory utilization, and YaRN factor 4 for a 131,072-token context.
  KV-block lifetime/reuse metrics sample 1% of blocks for both policies.
  Each policy starts in a fresh server. The initial comparison used five
  segments and one run per policy; the fixed comparison uses ten segments and
  two repeats. Both preserve external delays. Two repeats do not establish
  statistical significance.

## Local preparation and checks

```bash
uv venv .venv
uv pip install --python .venv/bin/python aiohttp pytest
.venv/bin/python -m pytest serving_experiment -q
.venv/bin/python serving_experiment/prepare.py --output runs/prepared-paper-boundaries
rsync -a serving_experiment/ ccds-ai:cachewise-rerun/serving_experiment/
scp runs/prepared-paper-boundaries/workload.json ccds-ai:cachewise-rerun/prepared/paper-workload.json
scp runs/prepared-paper-boundaries/bootstrap.jsonl ccds-ai:cachewise-rerun/prepared/bootstrap.jsonl
```

Preparation refuses to overwrite an existing output directory. The output
contains the workload, input hashes, exclusions, and predictor bootstrap.

## CCDS run

The dedicated remote root is `~/cachewise-rerun` on SSH host `ccds-ai`.
The login host has P100 GPUs; the experiment must run through Slurm on `4RTXq`.

```bash
source /etc/profile.d/modules.sh
module load slurm/slurm/24.11
cd ~/cachewise-rerun
sbatch serving_experiment/ccds.slurm
squeue -u "$USER"
```

The batch script allocates exactly one RTX Pro 6000. It runs a two-session,
two-turn smoke comparison before the ten-segment comparison with two repeats. A failed
smoke test prevents the full run. Model weights must already be downloaded.

## Artifacts

Every policy run has:

- `manifest.json`: policy, seed, workload hashes, limits, and completion status.
- `requests.jsonl`: request starts/ends, token counts, timing, and usage.
- `payloads.jsonl.gz`: exact synthetic request payloads.
- `responses.jsonl.gz`: streamed responses and token IDs with arrival times.
- `metrics.jsonl`: server Prometheus snapshots approximately every two seconds.
- `timeline.json`: Chrome/Perfetto-compatible client request and external-delay
  timeline. This is not a CUDA kernel profile.
- `summary.json`: per-session sum of request latency excluding external delays,
  TTFTs, token usage, failures, and output-token throughput.

The parent run directory holds GPU utilization/memory/power CSVs, server logs, exact server commands, package
versions, model snapshot path, GPU information, and source revision. A `COMPLETE`
file is written only after every requested policy run succeeds. Runs marked
`failed` must not be reported as valid performance comparisons.

After copying a completed comparison directory locally:

```bash
uv pip install --python .venv/bin/python matplotlib
.venv/bin/python serving_experiment/inspect_traces.py runs/ccds-9465/comparison-9465
```

This checks prompt hashes, token lengths, actual token usage, and gzip archive
integrity across every repeat, writing `repeat-trace-audit.json`. It also writes
`trace_audit.json`, `trace_overview.png`, and `timeline-comparison.json` for
repeat 0. `summarize.py` writes individual results to `comparison.json` and
aggregates with observed repeat ranges to `comparison-summary.json`.
Load the combined timeline in Perfetto to inspect requests and external waits
for both policies, with each run aligned to elapsed time zero. The plot's KV
usage is active block usage; zero active usage does not imply prefix-cache
contents have been discarded. Startup and model loading are outside replay
timings. The smoke test also warms shared kernel compilation caches.

The setup pins the CUDA 13.0 compiler, CRT, NVVM, and CCCL packages to match
PyTorch's runtime. The batch script supplies the pip toolkit path and a local
unversioned CUDA-runtime symlink required by FlashInfer's JIT linker.
The recorder uses an enlarged SSE line buffer because long-context responses
include the full prompt token-ID array in a single event; a 130,000-token
regression test covers this case.
