#!/bin/bash
#SBATCH --job-name=fm-train
#SBATCH --account=rrg-hsn
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/sabrina7/scratch/instanovofm_runs/%x_%j.out
#SBATCH --error=/home/sabrina7/scratch/instanovofm_runs/%x_%j.err
# One training (or validation-only) run of the foundation model on the MCFM tier:
#
#   sbatch [--gpus-per-node=h100:4 --time=...] --job-name=<run-name> scripts/train/submit.sh <run-name> [hydra overrides...]
#
# The run lives in $RUNS/<run-name> (checkpoints/, logs/, mlflow.db, train.log); the job's working directory is
# that folder, so every relative path the trainer writes lands there. MLflow logs to sqlite:///<run>/mlflow.db
# (no server); read it with `mlflow ui --backend-store-uri sqlite:///<run>/mlflow.db` on the login node behind an
# SSH tunnel. STAGE=1 copies the MCFM shards to $SLURM_TMPDIR first and trains from node-local disk.
set -euo pipefail
cd "$HOME/experiments/InstaNovo-FM"
source .envrc
unset PYTHONPATH
export TORCH_HOME="$HOME/scratch/torch-hub"
export NUMBA_THREADING_LAYER=tbb
export LD_LIBRARY_PATH="$HOME/scratch/venvs/instanovo-fm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export INSTANOVO_FM_DATA_DIR="$PWD/data"
PY="$HOME/scratch/venvs/instanovo-fm/bin/python"
REPO="$PWD"
NAME="$1"; shift
RUN="$RUNS/$NAME"
mkdir -p "$RUN"
echo "host $(hostname)  gpus $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,)  commit $(git rev-parse --short HEAD)  run $RUN"
echo "cpus $SLURM_CPUS_PER_TASK  mem $SLURM_MEM_PER_NODE MB  shm $(df -h /dev/shm | awk 'NR==2{print $2" total, "$4" free"}')  nofile $(ulimit -n)"

SRC="$DATA/splits/mcfm"
if [ "${STAGE:-0}" = "1" ]; then
  t0=$(date +%s)
  mkdir -p "$SLURM_TMPDIR/mcfm"
  cp "$SRC"/mcfm-*.parquet "$SLURM_TMPDIR/mcfm/"
  SRC="$SLURM_TMPDIR/mcfm"
  echo "staged $(du -sh "$SRC" | cut -f1) to $SRC in $(( $(date +%s) - t0 )) s"
fi

cd "$RUN"
# nvidia-smi sampler: one line every 30 s (utilisation, memory) into gpu.log, killed with the job
( while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader >> gpu.log; sleep 30; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

"$PY" -m instanovo_fm.trainer.train --config-name foundational \
  dataset=mcfm \
  "dataset.train_path=$SRC/mcfm-*train*.parquet" \
  "dataset.valid_path=$SRC/mcfm-*valid*.parquet" \
  "dataset.test_path=$SRC/mcfm-*test*.parquet" \
  "model_save_folder_path=$RUN/checkpoints" \
  "tb_summarywriter=$RUN/logs" \
  mlflow_enabled=True \
  "mlflow_tracking_uri=sqlite:///$RUN/mlflow.db" \
  mlflow_experiment_name=instanovo-fm-reproduction \
  "evaluation.output_dir=$RUN/evaluation" \
  "$@" 2>&1 | tee -a "$RUN/train.log"
