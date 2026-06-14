# CacheWise Coding Traces

Sanitized Claude Code trace dataset collected from real world claude code users. 
Raw, sanitized traces are located in `release/raw_traces/`. 

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

