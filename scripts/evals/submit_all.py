"""Submit the evaluation suite as one SLURM job per (dataset, model, protocol), with the memory and time each
protocol needs, and record the job ids.

    python scripts/evals/submit_all.py --datasets ms2bac --models 40M 40M-ours 89M [--protocols retrieval probes] [--dry-run]

Job names are `ev_<dataset>_<model>_<protocol>`; ids and arguments are appended to `$RESULTS/evals/jobs.csv`.
Memory and time per protocol come from the reproduction's measurements (`docs/agent_logs/session-2026-09-25-*.md`):
the per-peak tasks held more than 30 GB at 20,000 spectra, retrieval on a 200,000 pool took 47 min on CPU FAISS
without AVX2, a probe job 1 h 27 min, geometry 30 min, validation of 256,000 spectra 16 min.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, PROTOCOLS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproduce"))
from _common import MODELS, env_path  # noqa: E402

RESOURCES: dict[str, tuple[str, str]] = {  # protocol -> (memory, time)
    "probes": ("96G", "06:00:00"),
    "retrieval": ("128G", "24:00:00"),  # CPU FAISS without AVX2 scales with the square of the pool: 466,891 spectra took 9-10 h; `retrieval_time` sizes it per dataset
    "geometry": ("128G", "08:00:00"),
    "peak_level": ("250G", "08:00:00"),
    "unlabelled": ("128G", "08:00:00"),
    "validation": ("128G", "06:00:00"),
}
SUBMIT = Path(__file__).resolve().parent / "submit.sh"


def retrieval_time(dataset: str) -> str:
    """Wall time for the retrieval pool of `dataset`: the ProteomeTools pool (466,891) took 10 h, and the search is
    quadratic in the pool, so 12 h per (pool / 466,891)^2 with a 6 h floor, rounded up to whole hours."""
    import math

    import pyarrow.parquet as pq

    from run import files

    n = pq.ParquetFile(files(dataset)["identified"]).metadata.num_rows
    return f"{max(6, math.ceil(12 * (n / 466_891) ** 2)):02d}:00:00"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, required=True)
    ap.add_argument("--models", nargs="+", choices=sorted(MODELS), default=["40M", "40M-ours", "89M"])
    ap.add_argument("--protocols", nargs="+", choices=sorted(PROTOCOLS), default=sorted(PROTOCOLS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    log = env_path("RESULTS") / "evals" / "jobs.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["submitted", "job_id", "dataset", "model", "protocol", "mem", "time"])
        for dataset in args.datasets:
            for model in args.models:
                for protocol in args.protocols:
                    mem, hours = RESOURCES[protocol]
                    if protocol == "retrieval":
                        hours = retrieval_time(dataset)
                    cmd = [
                        "sbatch", "--parsable", f"--job-name=ev_{dataset}_{model}_{protocol}", f"--mem={mem}", f"--time={hours}",
                        str(SUBMIT), "--dataset", dataset, "--model", model, "--protocol", protocol,
                    ]
                    if args.dry_run:
                        print(" ".join(cmd))
                        continue
                    job = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), job, dataset, model, protocol, mem, hours])
                    print(f"{job}  {dataset} {model} {protocol}  ({mem}, {hours})")


if __name__ == "__main__":
    main()
