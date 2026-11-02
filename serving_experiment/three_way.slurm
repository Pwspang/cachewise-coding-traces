#!/bin/bash
#SBATCH --job-name=cachewise-infercept
#SBATCH --partition=4RTXq
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
TASK_ROOT="$HOME/cachewise-rerun"
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
SMOKE="$TASK_ROOT/runs/three-way-smoke-$SLURM_JOB_ID"
# The smoke run profiles recompute latency once on an LRU server with the
# same settings; the full comparison reuses that profile for every repeat.
"$TASK_ROOT/.venv/bin/python" serving_experiment/run_job.py \
    --root "$TASK_ROOT" --output "$SMOKE" --smoke \
    --policies baseline cachewise infercept --num-gpu-blocks 17981
"$TASK_ROOT/.venv/bin/python" serving_experiment/run_job.py \
    --root "$TASK_ROOT" --output "$TASK_ROOT/runs/three-way-$SLURM_JOB_ID" \
    --workload "$TASK_ROOT/prepared/paper-workload.json" --sessions 10 \
    --policies baseline cachewise infercept --repeats 3 \
    --infercept-profile "$SMOKE/recompute-profile.json" --num-gpu-blocks 17981
"$TASK_ROOT/.venv/bin/python" serving_experiment/summarize.py \
    "$TASK_ROOT/runs/three-way-$SLURM_JOB_ID"
