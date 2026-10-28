# CacheWise Coding Traces

Sanitized Claude Code trace dataset collected from real world claude code users. 
Raw, sanitized traces are located in `release/raw_traces/`. 

## CacheWise/vLLM serving experiment

See [serving_experiment/README.md](serving_experiment/README.md) for the
single-RTX-Pro-6000 replay using Qwen2.5-Coder-14B-Instruct, the CCDS Slurm
runner, and request/response, cache-metric, and timeline trace recordings.
The replay uses synthetic tokens because the released prompt text is redacted.

## Install Requirements

```bash
pip install -r release/requirements.txt
```

## Extracting events (LLM calls, Tool calls) from raw traces

```bash
python release/extract_events.py \
  --input release/raw_traces \
  --output release/parsed_traces
```

## Workload analysis plots

 Generate all plots:

```bash
python release/workload_analysis/generate_all.py
```

Or run individual scripts:

```bash
python release/workload_analysis/plot_session_duration.py
python release/workload_analysis/plot_prefill_decode_cdf.py
python release/workload_analysis/plot_context_evolution.py
python release/workload_analysis/plot_turns_distribution.py
python release/workload_analysis/plot_tool_durations.py
python release/workload_analysis/plot_user_vs_tool_cdf.py
```

## Tool duration prediction

```bash
python release/tool_duration_prediction/train_all.py
python release/tool_duration_prediction/infer.py --demo
```

