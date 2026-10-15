"""Export a training run's validation metrics from its MLflow SQLite store into the repository.

Writes `docs/references/rerun/train_<run-name>/metrics.json` in the shape the earlier runs keep
(`train_40M_mcfm_90k_b2048/metrics.json`): `final_eval_step_<N>` holds every `eval/*` metric at the last validation
step, `trajectory` the per-step series of the metrics `fill_results.py` and the notes read, `_run` the job, run
directory, MLflow run name and global batch; `--also-step N` adds an `eval_step_N` block. Row 23 of the results tables
(`fill_results._trainer_validation`) reads the first `eval/*` keys of the file, so `final_eval_step_<N>` comes first.

    source .envrc && python scripts/train/export_metrics.py arch-seed102 [--also-step 80001 --also-step 10001] [--job 24017759] [--dest train_arch-seed102]

The trajectory holds the validation steps only (every `validation_interval` of the run's parameters, logged at
step + 1), not the sanity-check batches the trainer logs at steps 0 to 15. `_run` also carries the median seconds per step over the run's `[Step N/M] [..., x s/step]` log lines (`train.log`) and the
peak GPU memory of the 30-second sampler (`gpu.log`), when those files exist; `seconds_per_step` is per optimizer step (the
elapsed clock over the steps), `seconds_per_forward_pass` the trainer's own `s/step`, which with gradient accumulation is
the smaller figure (and makes the trainer's ETA short by that factor).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAJECTORY = {  # series name in the export -> MLflow key under `eval/`
    "median_ae_ppm": "median_ae_ppm",
    "bin_accuracy": "bin_accuracy",
    "pct_within_20ppm": "pct_within_20ppm",
    "intensity_r2": "intensity_r2",
    "loss": "loss",
    "effective_rank": "embed/embeddingstatisticstask/effective_rank",
    "anisotropy_ratio": "embed/embeddingstatisticstask/anisotropy_ratio",
}
STEP_LINE = re.compile(r"\[Step (\d+)/\d+\] \[(\d+):(\d+):(\d+)/[^,]+, ([0-9.]+) s/step\]")


def step_times(log: Path) -> tuple[float | None, float | None]:
    """(seconds per optimizer step, seconds per forward pass) from the trainer's progress lines: the first is the
    elapsed clock of the last line divided by its step (validation passes included), the second the median of the
    logged `s/step`, which the trainer measures per forward pass, so with gradient accumulation it is the smaller."""
    if not log.exists():
        return None, None
    lines = STEP_LINE.findall(log.read_text(errors="replace"))
    if not lines:
        return None, None
    step, h, m, sec, _ = lines[-1]
    per_step = (int(h) * 3600 + int(m) * 60 + int(sec)) / int(step)
    passes = sorted(float(x[-1]) for x in lines)
    return round(per_step, 3), passes[len(passes) // 2]


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
        name: [
            [s, by_step[s][f"eval/{key}"]] for s in validation_steps if f"eval/{key}" in by_step[s]
        ]
        for name, key in TRAJECTORY.items()
    }
    job = args.job
    if job is None:
        logs = sorted(
            Path.home().glob(f"scratch/instanovofm_runs/{args.run}_*.out"),
            key=lambda p: p.stat().st_mtime,
        )
        job = int(re.search(r"_(\d+)\.out$", logs[-1].name).group(1)) if logs else None
    accumulation = int(params.get("grad_accumulation", 1))
    per_step, per_pass = step_times(run_dir / "train.log")
    out["_run"] = {
        "job": job,
        "run": str(run_dir),
        "mlflow_run": name,
        "global_batch": int(params.get("train_batch_size", 0)) * accumulation,
        "grad_accumulation": accumulation,
        "seed": params.get("seed"),
        "seconds_per_step": per_step,
        "seconds_per_forward_pass": per_pass,
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
