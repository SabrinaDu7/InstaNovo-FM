"""Export a training run's validation metrics from its MLflow SQLite store into the repository.

Writes `docs/references/rerun/train_<run-name>/metrics.json` in the shape the earlier runs keep
(`train_40M_mcfm_90k_b2048/metrics.json`): `final_eval_step_<N>` holds every `eval/*` metric at the last validation
step, `trajectory` the per-step series of the metrics `fill_results.py` and the notes read, `_run` the job, run
directory, MLflow run name and global batch; `--also-step N` adds an `eval_step_N` block. Row 23 of the results tables
(`fill_results._trainer_validation`) reads the first `eval/*` keys of the file, so `final_eval_step_<N>` comes first.

    source .envrc && python scripts/train/export_metrics.py arch-seed102 [--also-step 80001 --also-step 10001] [--job 24017759] [--dest train_arch-seed102]

The trajectory holds the validation steps only (every `validation_interval` of the run's parameters, logged at
step + 1), not the sanity-check batches the trainer logs at steps 0 to 15. `_run` also carries the median seconds per step over the run's `[Step N/M] [..., x s/step]` log lines (`train.log`) and the
peak GPU memory of the 30-second sampler (`gpu.log`), when those files exist.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAJECTORY = (
    "median_ae_ppm",
    "bin_accuracy",
    "pct_within_20ppm",
    "intensity_r2",
    "loss",
    "effective_rank",
    "anisotropy_ratio",
)
STEP_TIME = re.compile(r"\[Step (\d+)/\d+\] \[[^,]+, ([0-9.]+) s/step\]")


def step_time(log: Path) -> float | None:
    """Median seconds per step over the trainer's progress lines; None without a log."""
    if not log.exists():
        return None
    times = sorted(float(m.group(2)) for m in STEP_TIME.finditer(log.read_text(errors="replace")))
    return times[len(times) // 2] if times else None


def gpu_peak(log: Path) -> int | None:
    """Largest memory.used (MiB) sampled by the job's nvidia-smi loop; None without a log."""
    if not log.exists():
        return None
    used = [int(m.group(1)) for m in re.finditer(r", (\d+) MiB", log.read_text(errors="replace"))]
    return max(used) if used else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", help="run name under $RUNS")
    ap.add_argument(
        "--also-step",
        type=int,
        action="append",
        default=[],
        help="add an eval_step_<N> block for this validation step (repeatable)",
    )
    ap.add_argument(
        "--job",
        type=int,
        default=None,
        help="SLURM job id; default: from the newest log named after the run in ~/scratch/instanovofm_runs",
    )
    ap.add_argument(
        "--dest",
        default=None,
        help="directory name under docs/references/rerun; default train_<run>",
    )
    args = ap.parse_args()
    run_dir = Path(os.path.expandvars(os.path.expanduser(os.environ["RUNS"]))) / args.run
    con = sqlite3.connect(f"file:{run_dir / 'mlflow.db'}?mode=ro", uri=True)
    uuid, name = con.execute(
        "select run_uuid, name from runs order by start_time desc limit 1"
    ).fetchone()
    rows = con.execute(
        "select key, step, value from metrics where run_uuid=? and key like 'eval/%' order by key, step, timestamp",
        (uuid,),
    ).fetchall()
    params = dict(con.execute("select key, value from params where run_uuid=?", (uuid,)).fetchall())
    con.close()
    by_step: dict[int, dict[str, float]] = {}
    for key, step, value in rows:
        by_step.setdefault(step, {})[key] = value
    interval = int(params.get("validation_interval", 0))
    validation_steps = sorted(
        s for s in by_step if not interval or (s > 1 and (s - 1) % interval == 0)
    )
    last = max(validation_steps)
    out: dict[str, object] = {f"final_eval_step_{last}": dict(sorted(by_step[last].items()))}
    out["trajectory"] = {
        m: [[s, by_step[s][f"eval/{m}"]] for s in validation_steps if f"eval/{m}" in by_step[s]]
        for m in TRAJECTORY
    }
    job = args.job
    if job is None:
        logs = sorted(
            Path.home().glob(f"scratch/instanovofm_runs/{args.run}_*.out"),
            key=lambda p: p.stat().st_mtime,
        )
        job = int(re.search(r"_(\d+)\.out$", logs[-1].name).group(1)) if logs else None
    accumulation = int(params.get("grad_accumulation", 1))
    out["_run"] = {
        "job": job,
        "run": str(run_dir),
        "mlflow_run": name,
        "global_batch": int(params.get("train_batch_size", 0)) * accumulation,
        "grad_accumulation": accumulation,
        "seed": params.get("seed"),
        "seconds_per_step": step_time(run_dir / "train.log"),
        "gpu_mem_peak_mib": gpu_peak(run_dir / "gpu.log"),
    }
    for step in args.also_step:
        out[f"eval_step_{step}"] = dict(sorted(by_step[step].items()))
    dest = (
        REPO / "docs" / "references" / "rerun" / (args.dest or f"train_{args.run}") / "metrics.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"{dest.relative_to(REPO)}: validation steps {validation_steps}, last {last}, job {job}")  # noqa: T201


if __name__ == "__main__":
    main()
