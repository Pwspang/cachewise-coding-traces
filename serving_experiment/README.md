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

## Aging-override ablation: CCDS job 10772, 2026-11-02

This controlled follow-up compared fixed CacheWise with the default
`--prefix-aware-max-wait-s 30` against the same engine with the override disabled
(`--prefix-aware-max-wait-s 0`). The experiment used one full run per setting,
30 seconds first, on the same physical RTX Pro 6000 as job 9494. Model,
workload, 17,981-block cache capacity, predictor, and all other server arguments
were identical. Both fresh servers used the same lightweight queue observer.

| Metric | 30-second override | Override disabled |
| --- | ---: | ---: |
| Mean segment LLM time | 544.88 s | 366.07 s |
| Maximum segment LLM time | 788.98 s | 643.90 s |
| Total replay wall time | 812.94 s | 654.04 s |
| Prompt-token cache-hit fraction | 68.41% | 78.98% |
| Uncached prompt tokens | 1,546,501 | 1,028,677 |
| Mean summed queue time per segment | 320.35 s | 138.66 s |
| Median request TTFT | 3.816 s | 6.524 s |
| Maximum request latency | 330.90 s | 475.09 s |
| Aging-selected queue pops | 31 / 83 | 0 / 83 |

Disabling the override reduced mean segment LLM time **32.8% (1.488x)**,
replay wall time 19.5%, and uncached prompt tokens 33.5%. Nine of ten segments
improved; `session_0072_004` worsened from 441.88 to 529.27 seconds. Median
TTFT and maximum individual-request latency also worsened, demonstrating the
fairness trade-off. The largest measured phase reduction was queue waiting;
phase metrics remain overlapping residence times, not exclusive resource costs.

The control reproduced the previous CacheWise mean (545.27 s) closely. This
paired intervention supports the aging rule being a substantial performance
limiter on this workload, rather than the difference being caused by a newer
vLLM version. It is one ordered pair, not a repeated statistical evaluation.
There was no fresh LRU or InferCept baseline in this job, so it does not establish
a new matched speedup against either baseline or replicate the paper's results.
The ordinary `cachewise` policy retains the 30-second default; the experiment
adds the explicit `cachewise_no_aging` variant.

The observer records eligibility for the actual selected queue choice without
extra prefix scoring or modifying queue order. It observed 135 aging selections
across 187 fresh selection calls in the control, of which 31 popped requests;
the disabled run recorded zero across 626 fresh selection calls. A counted
aging choice need not differ from the counterfactual prefix-aware winner.
Accumulated observer work before the last trace write was 0.128 s and 0.160 s.
Tests exercising the actual checkout's queue verified preserved order and
correct accounting of cached choices under both settings. All nine replay and
observer tests passed. The engine patch was unchanged from job 9494.

Both runs completed all 83 requests with identical prompt hashes and token
lengths; each processed 4,894,853 prompt tokens and generated 15,235 tokens.
All 733 metric snapshots succeeded, all compressed recordings passed the audit,
and neither run had preemptions. Slurm completed with exit 0 on node06 in
28 minutes 25 seconds, including model startup.

Run using `sbatch serving_experiment/aging.slurm` from the configured remote
root. After retrieving the output locally:

```bash
.venv/bin/python serving_experiment/analyze_aging.py runs/ccds-10772/aging-10772
.venv/bin/python serving_experiment/inspect_traces.py runs/ccds-10772/aging-10772 \
  --policies cachewise cachewise_no_aging
```

Artifacts: `runs/ccds-10772/aging-10772/aging-comparison.json`,
`repeat-trace-audit.json`, `trace_overview.png`, `timeline-comparison.json`,
the `*-queue-*.jsonl` decision observations, and each run's full recordings.
The remote copy is `~/cachewise-rerun/runs/aging-10772` on `ccds-ai`.

## Scheduling, eviction, and capacity ablation: CCDS job 10779

Publication checkpoint: the normal-capacity cases below are complete. The two
half-capacity cases are still running and are not included in these results.
Committed numerical summaries and trace audits are under
[`results/2026-11-02/`](results/2026-11-02/). This bundle also includes the
completed aging ablation, earlier baseline analysis, and paper-fidelity audit.
Full request/response recordings remain in the documented `runs/` directories
locally and on CCDS; they are excluded from Git by the repository's ignore rules.

This follow-up prespecifies five fresh-server cases on the same GPU, model,
engine patch, bootstrap, and complete ten-segment workload. At 17,981 KV blocks
the order is FCFS/LRU, prefix-aware/LRU, then prefix-aware/predictive eviction.
At 8,990 blocks the order is prefix-aware/predictive then FCFS/LRU. Every
prefix-aware case disables the aging override. Reducing capacity changes cache
pressure without filtering tool delays or choosing another session subset; it
is not a reproduction of the paper's concurrency sweep.

Completed normal-capacity comparison (287,696 KV tokens):

| Metric | FCFS + LRU | Prefix-aware + LRU | CacheWise, no aging |
| --- | ---: | ---: | ---: |
| Mean segment LLM time | 577.45 s | 324.58 s | 337.28 s |
| Replay wall time | 875.17 s | 622.64 s | 625.04 s |
| Uncached prompt tokens | 1,728,277 | 1,026,469 | 1,028,677 |
| Mean summed queue time per segment | 337.86 s | 128.17 s | 129.79 s |

Prefix scheduling alone gives a 1.779x baseline/time ratio (43.8% reduction).
CacheWise gives 1.712x (41.6% reduction). Adding predictive eviction to the same
prefix scheduler is 3.9% slower in this pair, with 0.2% more computed prompt
tokens. Thus this experiment reproduces a substantial scheduling benefit but
does not reproduce the paper's additional predictive-eviction benefit.
The 3.9% difference is a single-pair observation, not a statistically established
regression. The previous no-aging CacheWise run was 366.07 seconds despite the
same aggregate uncached-token count, showing timing variability.

An illustrative retention difference is `session_0065_003:1`: its preceding
Bash call has a recorded delay of 0.143 seconds. Prefix-aware/LRU retains
89,152 cached tokens for the returning request; CacheWise retains zero. Its
latency rises from 6.57 to 77.74 seconds. Other requests show the reverse
trade-off, and aggregate uncached-token totals are close. Cache hit counts
alone therefore do not capture when recomputation occurs or how it changes
queue order. This observation does not independently isolate bootstrap key
mismatch, argument redaction, or other predictive-policy details.

All three normal-capacity runs completed 83 requests each, with identical input
hashes and actual token counts, valid compressed recordings, 1,061 successful
metric snapshots, and zero preemptions. The fresh baseline closely matches the
earlier 577.59-second mean and exactly matches its uncached-token count.

Run the experiment with `sbatch serving_experiment/pressure.slurm`. The analysis
command requires both capacity groups to be complete and validates equal inputs
across all five cases, allowed server-argument differences, source patches,
server counters, queue observations, and recordings:

```bash
.venv/bin/python serving_experiment/analyze_pressure.py \
  runs/ccds-10779/pressure-10779
```

## Comparison with the `infercept-comparison` branch

The committed results at
[`06521ca`](https://github.com/Pwspang/cachewise-coding-traces/tree/06521ca86baa8c1f67df98e7d8c933268938a48e/serving_experiment/results/ccds-9763)
report three rotating-order repeats of each policy in job 9763:

| Policy | Mean segment LLM time | Prompt-token hit rate |
| --- | ---: | ---: |
| FCFS + LRU | 584.89 s | 64.69% |
| CacheWise, default aging | 575.81 s | 66.12% |
| InferCept adaptation | 518.69 s | 70.17% |

InferCept wins that matched comparison: 11.3% lower mean time than its baseline
and 9.9% lower than its CacheWise configuration. Its three run means range from
518.40 to 519.04 seconds, with the same 70.17% hit rate in every repeat. It
computes 1,460,085 uncached prompt tokens versus the baseline's 1,728,277, a
15.5% reduction. For example, in repeat 0 `session_0084_006:2` retains 81,232
more cached tokens with InferCept and takes 82.10 seconds versus 143.95 seconds
with the baseline. The absolute 70% hit rate is therefore consistent with real
retention gains on this workload, not evidence of a reporting anomaly.

That branch's runner does not override the pinned fork's default 30-second
aging threshold. Our newer no-aging cases in job 10779 report 337.28 seconds /
78.98% hits for CacheWise and 324.58 seconds / 79.03% for prefix-aware LRU.
The new CacheWise time is numerically 35.0% below job 9763's InferCept mean,
but this is a **cross-job comparison**, not a fresh matched three-way result.
The GPU model, model weights, cache capacity, and reported workload match;
the physical GPU UUID, engine policy patch, repeat count, and scheduling
configuration differ. Request identities and per-request prompt/output lengths
match across the committed repeat-0 timelines and our replay. The committed
9763 audit verifies hashes within its job, but does not expose per-request
hashes for independent cross-job verification.

This branch's InferCept policy uses elapsed pause time and a profiled
recomputation-cost curve to choose preservation versus discard. It does not
use tool-name/argument predictions, does not swap to CPU, and represents
preservation through eviction priority rather than hard pinning. Consequently
it avoids our predictor's redacted-argument and bootstrap-key issues. It is
also different from the moving-average-based InferCept eviction baseline
described in the CacheWise paper's sections 6.1 and 6.3. These adaptations
prevent an exact numerical comparison to that paper.

All hit rates in these tables use completed-request cached prompt tokens over
total prompt tokens. Server cache-lookup counters can count repeated lookups
during admission and are not interchangeable with this metric, even below
the console's 1,000-request rolling-window limit. The committed InferCept
trace audit's lookup-hit ratio is about 25.45%, while its completed-request
token hit rate is 70.17%; these measure different events.

The numerical comparison and qualifications are saved in
[`infercept-branch-comparison.json`](results/2026-11-02/infercept-branch-comparison.json).
No InferCept raw-log search or rerun was needed for this branch comparison.

## Paper-fidelity audit (2026-11-02)

The paper's headline is conditional on its experimental setup, rather than a
guaranteed speedup at every load. It reports similar performance at N <= 10 and
2.7–3.5x lower session completion time at higher load
([sections 6.1–6.2](https://arxiv.org/html/2606.16824v1#S6.SS1)). Our ten selected
segments already exceed this GPU's KV capacity, so equal session counts do not
imply equal memory pressure across the two testbeds.

The local input audit identifies these additional reproducibility gaps:

- Our serving bootstrap inherits 34 training source sessions out of the 94
  parsed source sessions in this repository (36.2%). The paper describes an 80%
  random training split. Its exact serving split and selected session IDs are
  unavailable here; the published tool-duration split is not evidence that the
  same split was used for the serving plots.
- All 8,215 bootstrap argument strings are identical redacted placeholders.
  Argument-sensitive prediction cannot be recovered from these data. The paper
  clusters argument embeddings; this run explicitly uses the weaker tool-name
  predictor. The fork also hardcodes eight clusters and 512 TF-IDF features,
  versus the paper's evaluation of up to 100 clusters and 5,000 features. Those
  clustering settings are inactive in the current tool-name runs.
- Bootstrap samples represent individual tool calls, whereas runtime keys
  represent complete parallel batches, including repeated names. Sixteen of
  73 between-turn transitions have no matching initial bootstrap key (seven
  are `Bash+Bash`); they fall back to the global distribution until online
  observations create a matching distribution. This is a concrete preparation
  mismatch. Its independent latency impact has not been measured. A correction
  should train on complete batches using the same key and elapsed-duration
  definition, with the evaluation sessions excluded from training.
- The ten selected segments contain 83 requests from eight original sessions.
  Fifty-seven of the 73 between-turn delays are at most one second. This leaves
  short windows for predictive eviction to add value over prefix scheduling.
  It does not prove the predictor has no value at other loads or tool mixes.
- Increasing the deterministic selection to 20 segments introduces a
  178,164.206-second (49.49-hour) tool interval. Raw tool-use/result timestamps
  confirm that the interval is present in the source, not introduced by the
  replay parser. A quick unmodified 20–50-segment replay cannot finish within
  a normal short job. We have not clipped waits or selected replacement
  sessions based on performance results.

The hardware/model difference remains one RTX Pro 6000 with a 14B model versus
two H200s with a 32B model and TP=2. Its quantitative contribution is unmeasured.
Prefix caching and chunked prefill are enabled in both our policies; chunk size
512 matches the paper. The paper does not specify the full-input admission
check setting, so there is no basis to claim that it disabled that check.

Reproduce the input audit without modifying training or replay data:

```bash
.venv/bin/python serving_experiment/audit_paper_fidelity.py \
  --prepared runs/prepared-paper-boundaries \
  --output runs/paper-fidelity/audit.json
```

## Bottleneck analysis of job 9494 (2026-11-02)

This is an offline analysis of the existing four runs, not a new GPU experiment.
Both policies used the same fork revision, `a4b3affa7` (commit date June 12,
2026), the same patch, and the same compiled kernel environment. This was not
a comparison of a newer upstream vLLM release against an older CacheWise engine.
The baseline used FCFS/LRU with automatic prefix caching and chunked prefill;
CacheWise used prefix-aware/predictive policies in that same engine.

The final Prometheus histograms account for all 83 requests in each run.
Mean summed server request residence time per segment, averaged over repeats:

| Component | Baseline | Fixed CacheWise |
| --- | ---: | ---: |
| Waiting in the engine queue | 336.86 s | 309.88 s |
| Prefill phase | 73.08 s | 66.53 s |
| Decode phase | 164.84 s | 158.59 s |
| Unattributed residual | 2.29 s | 9.78 s |

Queueing is 58.4% of baseline server request time, decode 28.6%, and prefill
12.7%. These are overlapping request residence times, not exclusive GPU or
CPU execution measurements. The residual is retained rather than assigned
to a particular resource. Client latency adds roughly 0.5 seconds per segment
to the server totals. Prefill's fraction is not a bound on CacheWise's possible
gain: reducing prefill work can also shorten other requests' queues.

There is real cache pressure. The ten initial prompts total 487,696 tokens,
1.70 times the 287,696-token KV capacity. The server samples show about 2.7
running requests on average (maximum observed 5), despite `max_num_seqs=32`.
The baseline computes 1,728,277 prompt KV tokens per run versus 1,531,021 on
average for CacheWise. Fresh initial and appended prompt tokens total only
605,956; the excess includes repeated computation and block alignment/last-token
overhead. Zero request preemptions does not mean zero cached-prefix eviction.
The full-input-length admission gate is enabled by default in both policies;
its separate performance contribution was not measured. The selected workload
also has 57 of 73 between-turn delays at most one second, leaving less opportunity
for long-horizon tool-duration ranking than workloads with longer tool waits.

An additional policy difference was uncovered: the fork defaults to
`prefix_aware_max_wait_s=30`. When the oldest waiting request exceeds that age,
the prefix-aware queue selects it in FCFS order regardless of cached overlap.
Each CacheWise run had 31/83 requests with queue time above 30 seconds. This
shows substantial exposure to the threshold, not a measured count of override
decisions. `probe_scheduler.py` executes the checkout's actual queue code and
confirms that a 40-second-old request needing 6,000 blocks wins over a newer
request needing one block at the default threshold; setting the threshold to
zero reverses the choice. The earlier lifecycle fixes did not change this rule.
Its impact was unmeasured in this offline analysis; the subsequent ablation
above measured it. The paper's described
prefix-aware policy does not specify this 30-second override
([section 5.1](https://arxiv.org/html/2606.16824v1#S5.SS1)).

The targeted follow-up was the paired 30-versus-0-second aging ablation above,
recording override selections and queue delays. A further experiment could vary the 512-token
prefill budget identically for both policies with fixed KV capacity. The 512
budget matches the paper, but vLLM documents the throughput/latency trade-off
and recommends larger budgets for throughput in some settings
([vLLM tuning guide](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)).
Neither a prefill-budget tuning improvement nor a newer-version advantage is established here.
A separate controlled version comparison would be required for the latter.

Reproduce the analysis and queue-semantics check:

```bash
.venv/bin/python serving_experiment/analyze_bottlenecks.py \
  runs/ccds-9494/fixed-comparison-9494 \
  --workload runs/prepared-paper-boundaries/workload.json
.venv/bin/python serving_experiment/probe_scheduler.py \
  external/vllm-cachewise/vllm/v1/core/sched/request_queue.py \
  --output runs/ccds-9494/fixed-comparison-9494/scheduler-aging-probe.json
```

Outputs: `bottleneck-analysis.json`, `bottleneck-breakdown.png`, and
`scheduler-aging-probe.json` in the comparison directory. No serving policy,
historical measurement, or GPU job was changed for this analysis.

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
