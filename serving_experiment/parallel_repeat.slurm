#!/bin/bash
#SBATCH --job-name=cachewise-repeat
#SBATCH --partition=4RTXq
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
TASK_ROOT="${CACHEWISE_PARALLEL_ROOT:?Set the frozen experiment root}"
TASK_REPEAT="${CACHEWISE_REPEAT_INDEX:?Set repeat 1 or 2}"
TASK_SCRATCH="/local/cachewise-$USER-$SLURM_JOB_ID"
mkdir -p "$TASK_SCRATCH/tmp" "$TASK_SCRATCH/vllm" "$TASK_SCRATCH/triton"
export TMPDIR="$TASK_SCRATCH/tmp"
export VLLM_CACHE_ROOT="$TASK_SCRATCH/vllm"
export TRITON_CACHE_DIR="$TASK_SCRATCH/triton"
export CUDA_HOME="$TASK_ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$TASK_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
mkdir -p "$TASK_SCRATCH/lib"
ln -s "$CUDA_HOME/lib/libcudart.so.13" "$TASK_SCRATCH/lib/libcudart.so"
export FLASHINFER_EXTRA_LDFLAGS="-L$TASK_SCRATCH/lib -L$CUDA_HOME/lib"
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
cd "$TASK_ROOT"
RUN_DIR="$TASK_ROOT/runs/repeat-$TASK_REPEAT-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
cd "$TASK_ROOT/src/vllm-cachewise"
"$TASK_ROOT/.venv/bin/python" -m pytest --confcutdir=tests/v1/core/cachewise tests/v1/core/cachewise -q > "$RUN_DIR/engine-tests.txt" 2>&1
cat "$RUN_DIR/engine-tests.txt"
cd "$TASK_ROOT"
# Same implementation, physical GPU, input lengths, delays, and KV budget.
# Each repeat runs all policies on one GPU; its index retains the planned rotation.
# All cases mark completed segments idle; only the oracle sees tool durations.
"$TASK_ROOT/.venv/bin/python" serving_experiment/run_job.py \
    --root "$TASK_ROOT" --output "$RUN_DIR/smoke" --smoke \
    --workload "$TASK_ROOT/prepared/paper-workload.json" \
    --policies baseline eviction_only prefix_only_no_aging cachewise_no_aging cachewise_oracle_no_aging cachewise \
    --num-gpu-blocks 17981 --queue-diagnostics --mark-terminal-idle
"$TASK_ROOT/.venv/bin/python" serving_experiment/analyze_oracle.py "$RUN_DIR/smoke" --smoke --consolidated
"$TASK_ROOT/.venv/bin/python" serving_experiment/run_job.py \
    --root "$TASK_ROOT" --output "$RUN_DIR/comparison" \
    --workload "$TASK_ROOT/prepared/paper-workload.json" --sessions 10 --repeats 1 --repeat-start "$TASK_REPEAT" \
    --policies baseline eviction_only prefix_only_no_aging cachewise_no_aging cachewise_oracle_no_aging cachewise \
    --num-gpu-blocks 17981 --queue-diagnostics --mark-terminal-idle
"$TASK_ROOT/.venv/bin/python" serving_experiment/analyze_oracle.py "$RUN_DIR/comparison" --consolidated --repeat-index "$TASK_REPEAT"
