"""Export a training run's validation metrics from its MLflow SQLite store into the repository.

Writes `docs/references/rerun/train_<run-name>/metrics.json` in the shape the earlier runs keep
(`train_40M_mcfm_90k_b2048/metrics.json`): `final_eval_step_<N>` holds every `eval/*` metric at the last validation
step, `trajectory` the per-step series of the metrics `fill_results.py` and the notes read, `_run` the job, run
directory, MLflow run name and global batch; `--also-step N` adds an `eval_step_N` block. Row 23 of the results tables
(`fill_results._trainer_validation`) reads the first `eval/*` keys of the file, so `final_eval_step_<N>` comes first.

    source .envrc && python scripts/train/export_metrics.py arch-seed102 [--also-step 80001] [--job 24017759]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAJECTORY = ("median_ae_ppm", "bin_accuracy", "pct_within_20ppm", "intensity_r2", "loss")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", help="run name under $RUNS")
    ap.add_argument(
        "--also-step",
        type=int,
        default=None,
        help="add an eval_step_<N> block for this validation step",
    )
    ap.add_argument(
        "--job",
        type=int,
        default=None,
        help="SLURM job id; default: from the newest log named after the run in ~/scratch/instanovofm_runs",
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
    last = max(by_step)
    out: dict[str, object] = {f"final_eval_step_{last}": dict(sorted(by_step[last].items()))}
    out["trajectory"] = {
        m: [[s, by_step[s][f"eval/{m}"]] for s in sorted(by_step) if f"eval/{m}" in by_step[s]]
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
    }
    if args.also_step is not None:
        out[f"eval_step_{args.also_step}"] = dict(sorted(by_step[args.also_step].items()))
    dest = REPO / "docs" / "references" / "rerun" / f"train_{args.run}" / "metrics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"{dest.relative_to(REPO)}: steps {sorted(by_step)}, last {last}, job {job}")  # noqa: T201


if __name__ == "__main__":
    main()
