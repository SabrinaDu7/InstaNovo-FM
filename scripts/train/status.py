"""Status of one training run under $RUNS, in one screen: the SLURM jobs carrying its name, the last training and
validation lines of its log, the newest value of each tracked metric read directly from the run's MLflow SQLite
store (no server), the best checkpoint metric so far, the checkpoints on disk, and the GPU sampler's last line.

    source .envrc && python scripts/train/status.py <run-name> [--tail N]
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

METRICS = (
    "train/loss_smooth",
    "train/loss_raw",
    "train/group_ce_loss",
    "train/offset_ce_loss",
    "train/intensity_loss",
    "optim/lr",
    "eval/loss",
    "eval/median_ae_ppm",
    "eval/bin_accuracy",
    "eval/pct_within_20ppm",
    "eval/intensity_r2",
    "eval/embed/embeddingstatisticstask/anisotropy_ratio",
    "eval/embed/embeddingstatisticstask/effective_rank",
    "eval/embed/duplicateretrievaltask/recall@1",
    "eval/embed/duplicateretrievaltask/map@20",
)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False).stdout.strip()


def main() -> None:
    name = sys.argv[1]
    tail = int(sys.argv[sys.argv.index("--tail") + 1]) if "--tail" in sys.argv else 4
    run = Path(os.path.expandvars(os.path.expanduser(os.environ["RUNS"]))) / name
    print(f"== {name}  ({run})")
    print("jobs:", sh(f'squeue -u $USER -h -o "%.10i %.18j %.9T %.10M %R" | grep -i "{name[:8]}" || echo none queued/running'))
    if not run.exists():
        print("no run directory yet")
        return
    log = run / "train.log"
    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        keep = [ln for ln in lines if re.search(r"\[TRAIN\]|\[VALIDATION\].*Loss|Saved|staged|sharing strategy|ERROR|Traceback|Error|complete", ln)]
        for ln in keep[-tail:]:
            print("  " + re.sub(r"^\S+ \S+ +INFO +", "", ln.strip())[:170])
    db = run / "mlflow.db"
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        runs = con.execute("select run_uuid, name, status from runs order by start_time desc limit 1").fetchall()
        if runs:
            uuid, rname, status = runs[0]
            print(f"mlflow run: {rname} [{status}]")
            for key in METRICS:
                row = con.execute(
                    "select step, value from metrics where run_uuid=? and key=? order by step desc, timestamp desc limit 1", (uuid, key)
                ).fetchone()
                if row:
                    print(f"  {key:60s} step {row[0]:>7d}  {row[1]:.4f}")
            best = con.execute("select min(value), step from metrics where run_uuid=? and key='eval/median_ae_ppm'", (uuid,)).fetchone()
            if best and best[0] is not None:
                print(f"  best eval/median_ae_ppm {best[0]:.2f} (step {best[1]})")
        con.close()
    ck = run / "checkpoints"
    if ck.exists():
        files = sorted(p for p in ck.rglob("*") if p.is_file() and p.suffix in (".ckpt", ".bin", ".safetensors", ".pt"))
        print("checkpoints:", ", ".join(f"{p.relative_to(ck)} ({p.stat().st_size / 2**20:.0f} MB)" for p in files[-6:]) or "none")
    gpu = run / "gpu.log"
    if gpu.exists():
        last = gpu.read_text().strip().splitlines()[-1:]
        print("gpu:", last[0] if last else "no samples")


if __name__ == "__main__":
    main()
