"""Shared plumbing for `scripts/reproduce/result*.py`.

Every result script names one model, one corpus tier and split, and the evaluation overrides that define its
protocol; this module turns that into a run: it resolves `$DATA`, `$CHECKPOINTS` and `$RESULTS` (from
`.envrc`), composes the package's `foundational` config with the overrides, runs `run_evaluation`, and copies
each task's `task_summary.json` (and its `task_results.json` when it is small) plus a `run.json` (overrides, commit, duration) into
`docs/references/rerun/<script>/`, so the numbers quoted in `docs/references/results_paper_or_rerun.md`
travel with the repository and name the command that made them.

    source .envrc && python scripts/reproduce/result1_40M_probes_retrieval.py      # on a GPU node
    sbatch --job-name=result1_40M scripts/reproduce/submit.sh scripts/reproduce/result1_40M_probes_retrieval.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from instanovo_fm.eval.embed_evaluation import run_evaluation
from instanovo_fm.utils.hydra_config import compose_fm_config

REPO = Path(__file__).resolve().parents[2]
MODELS: dict[str, str] = {  # tag used in script names -> release checkpoint id
    "40M": "instanovo-fm-mcfm-90k-v0.1.0",
    "89M": "instanovo-fm-v0.1.0",
    "89M-ts-pa": "instanovo-fm-lcfm-ts-pa-v0.1.0",
    "89M-sa-nopa": "instanovo-fm-lcfm-sa-nopa-v0.1.0",
    "89M-sa-pa": "instanovo-fm-lcfm-sa-pa-v0.1.0",
}
SPLIT_GLOB = {"train": "*train*", "valid": "*valid*", "test": "*test*"}
RESULTS_MAX_BYTES = 20 * 2**20  # task_results.json above this stays under $RESULTS only


def env_path(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise SystemExit(f"{name} is not set: `source .envrc` first (see .envrc.example)")
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def dataset_overrides(tier: str) -> list[str]:
    """The tier's three split globs under `$DATA/splits/<tier>/`, on top of the package's `dataset=<tier>`."""
    root = env_path("DATA") / "splits" / tier
    return [f"dataset={tier}"] + [
        f"dataset.{split}_path={root}/{tier}-{glob}.parquet" for split, glob in SPLIT_GLOB.items()
    ]


def run(*, script: str, model: str, tier: str, split: str, overrides: list[str]) -> Path:
    """Evaluate `model` on `split` of `tier` with `overrides`; returns the directory the summaries were copied to."""
    checkpoint = env_path("CHECKPOINTS") / f"{MODELS[model]}.ckpt"
    out = env_path("RESULTS") / script
    os.environ.setdefault("INSTANOVO_FM_DATA_DIR", str(REPO / "data"))
    all_overrides = [
        "evaluation.enabled=True",
        f"evaluation.checkpoint_path={checkpoint}",
        f"evaluation.split={split}",
        f"evaluation.output_dir={out}",
        *dataset_overrides(tier),
        *overrides,
    ]
    t0 = time.time()
    run_evaluation(compose_fm_config("foundational", all_overrides))
    dest = REPO / "docs" / "references" / "rerun" / script
    dest.mkdir(parents=True, exist_ok=True)
    summaries = sorted(p for p in out.rglob("task_summary.json") if p.stat().st_mtime >= t0)
    for path in summaries:
        shutil.copy(path, dest / f"{path.parent.name}.json")
        results = path.with_name("task_results.json")  # per-class and per-ion detail; kept when it is small
        if results.exists() and results.stat().st_size <= RESULTS_MAX_BYTES:
            shutil.copy(results, dest / f"{path.parent.name}.results.json")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()
    (dest / "run.json").write_text(
        json.dumps(
            {
                "script": script,
                "model": MODELS[model],
                "checkpoint": str(checkpoint),
                "tier": tier,
                "split": split,
                "overrides": all_overrides,
                "tasks_written": [p.parent.name for p in summaries],
                "seconds": round(time.time() - t0),
                "commit": commit,
                "host": os.uname().nodename,
            },
            indent=1,
        )
    )
    print(f"{script}: {len(summaries)} task summaries -> {dest} ({time.time() - t0:.0f} s)", flush=True)
    return dest


# The protocols, each used by one script per model.

PROBES_RETRIEVAL = [  # docs/reproducing_paper_results.md, "Reproducing the probe results", verbatim
    "evaluation.max_samples=200000",
    "evaluation.batch_size=256",
    "evaluation.random_state=42",
    "evaluation.embedding_pooling=[mean_pool]",
    "evaluation.tasks_to_run=[linearprobetask,duplicateretrievaltask]",
    "evaluation.task_configs.duplicateretrievaltask.max_samples=20000",
    "evaluation.task_configs.linearprobetask.use_project_split=false",
    "evaluation.task_configs.linearprobetask.max_iter=5000",
]
PEAK_LEVEL = [  # the paper's peak-level battery: Table S9 says 10,000 test spectra
    "evaluation.max_samples=10000",
    "evaluation.batch_size=128",
    "evaluation.random_state=42",
    "evaluation.tasks_to_run=[peaktypeclassificationtask,confidencesignalanalysistask,headanalysistask,igattributiontask]",
]
GEOMETRY = [  # embedding-space tasks with no per-peak storage; the three opt-in tasks included
    "evaluation.max_samples=20000",
    "evaluation.batch_size=256",
    "evaluation.random_state=42",
    "evaluation.tasks_to_run=[embeddingstatisticstask,umapvisualisationtask,evocclusteringtask,cosinehyperscorecorrelationtask,esm2crossmodalalignmenttask,glassboxattributiontask]",
]
