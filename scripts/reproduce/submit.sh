#!/bin/bash
#SBATCH --job-name=fm-reproduce
#SBATCH --account=rrg-hsn
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/sabrina7/scratch/instanovofm_runs/%x_%j.out
#SBATCH --error=/home/sabrina7/scratch/instanovofm_runs/%x_%j.err
# One result script on one H100:
#   mkdir -p ~/scratch/instanovofm_runs   # SLURM opens the log there before the job body runs
#   sbatch --job-name=result1_40M scripts/reproduce/submit.sh scripts/reproduce/result1_40M_probes_retrieval.py
# The environment is the uv venv built into scratch (home is nearly full); the ESM2 weights are read from
# $TORCH_HOME, pre-fetched on the login node, so the job needs no download.
set -euo pipefail
cd "$HOME/experiments/InstaNovo-FM"
source .envrc
unset PYTHONPATH
export TORCH_HOME="$HOME/scratch/torch-hub"
export NUMBA_THREADING_LAYER=tbb   # Numba's default workqueue layer aborted under EVoC (SIGABRT, jobs 22695360/1)
export INSTANOVO_FM_DATA_DIR="$PWD/data"
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  commit $(git rev-parse --short HEAD)  script $1"
exec "$HOME/scratch/venvs/instanovo-fm/bin/python" "$1"
