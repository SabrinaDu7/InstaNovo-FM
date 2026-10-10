#!/bin/bash
#SBATCH --job-name=fm-evals
#SBATCH --account=rrg-hsn
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/sabrina7/scratch/instanovofm_runs/%x_%j.out
#SBATCH --error=/home/sabrina7/scratch/instanovofm_runs/%x_%j.err
# One protocol of the evaluation suite on one external dataset with one checkpoint, on one H100:
#   sbatch [--mem=250G --time=..] --job-name=ev_<dataset>_<model>_<protocol> scripts/evals/submit.sh \
#          --dataset <dataset> --model <model> --protocol <protocol>
# `scripts/evals/submit_all.py` issues these with the memory and time each protocol needs. The environment is
# the one of `scripts/reproduce/submit.sh` (uv venv in scratch, ESM2 weights in $TORCH_HOME, Numba on TBB).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/experiments/InstaNovo-FM}"  # the checkout sbatch was run from (SLURM spools a copy of this script)
source .envrc
unset PYTHONPATH
export TORCH_HOME="$HOME/scratch/torch-hub"
export NUMBA_THREADING_LAYER=tbb
export LD_LIBRARY_PATH="${FM_VENV:-$HOME/scratch/venvs/instanovo-fm}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export INSTANOVO_FM_DATA_DIR="$PWD/data"
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  commit $(git rev-parse --short HEAD)  args $*"
exec "${FM_VENV:-$HOME/scratch/venvs/instanovo-fm}/bin/python" scripts/evals/run.py "$@"
