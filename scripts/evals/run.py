"""Run one protocol of InstaNovo-FM's evaluation suite on one of our exported datasets with one checkpoint.

    source .envrc && python scripts/evals/run.py --dataset ms2bac --model 40M --protocol retrieval
    sbatch --job-name=ev_ms2bac_40M_retrieval scripts/evals/submit.sh --dataset ms2bac --model 40M --protocol retrieval

Datasets are the exports of `prepare_dataset.py` under `$EXTERNAL/<dataset>/`; models are the tags of
`scripts/reproduce/_common.py::MODELS` (`40M` released, `40M-ours` trained here, `89M` released). The protocols
are the paper's, as pinned in `scripts/reproduce/_common.py`, with the sample caps lifted (`evaluation.max_samples=null`
evaluates every spectrum of the file) except where a task's memory or time grows faster than linearly, where the
paper's own cap applies above a size and the run.json says so:

    probes       linear probes; train / valid / test are the dataset's own peptide-disjoint 80/10/10 files
                 (100,000 / 10,000 / 10,000 caps of the package apply, as in the paper)
    retrieval    duplicate-spectrum retrieval over every identified spectrum (every duplicate group, no group cap)
    geometry     embedding statistics, UMAP, EVoC, ESM2 alignment, Glass Box, cosine-hyperscore on the identified
                 spectra; capped at the paper's 20,000 above 100,000 spectra (UMAP and EVoC are superlinear)
    peak_level   peak-type classification, confidence signal, head analysis, IG attribution on the identified
                 spectra; capped at the paper's 10,000 above 25,000 spectra (per-peak embeddings are held in RAM)
    unlabelled   embedding statistics and UMAP on every MS2 spectrum, identified or not (`dataset.is_annotated=false`,
                 this branch's evaluator change); UMAP capped at 100,000 above that size
    validation   the trainer's own validation pass (median |ppm| error over masked peaks, bin accuracy, intensity
                 R2) on every identified spectrum, the way the released checkpoint was validated in Step 2 of
                 `docs/agent_logs/session-2026-09-25-train40M.md`

Outputs: run directory `$RESULTS/evals/<dataset>/<model>/<protocol>/`; task summaries (or the validation metrics)
copied to `docs/results/rerun/<dataset>/<model>/<protocol>/` with a `run.json` naming the overrides, commit and
duration, which `render_results.py` reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproduce"))
from _common import MODELS, REPO, env_path, run  # noqa: E402  (the reproduction's runner, shared)

DATASETS = ("ms2bac", "proteometools", "ups1", "campi")


@dataclass(frozen=True)
class Protocol:
    view: str  # the evaluated file: identified | all | probe
    tasks: tuple[str, ...]
    overrides: tuple[str, ...] = ()
    cap_above: int | None = None  # evaluate `cap` samples when the evaluated file holds more rows than this
    cap: int = 20_000
    batch_size: int = 256


PROTOCOLS: dict[str, Protocol] = {
    "probes": Protocol(
        view="probe",
        tasks=("linearprobetask",),
        overrides=(
            "evaluation.task_configs.linearprobetask.use_project_split=false",
            "evaluation.task_configs.linearprobetask.max_iter=5000",
        ),
    ),
    "retrieval": Protocol(
        view="identified",
        tasks=("duplicateretrievaltask",),
        overrides=("evaluation.task_configs.duplicateretrievaltask.max_samples=1000000000",),  # every duplicate group
    ),
    "geometry": Protocol(
        view="identified",
        tasks=(
            "embeddingstatisticstask",
            "umapvisualisationtask",
            "evocclusteringtask",
            "esm2crossmodalalignmenttask",
            "glassboxattributiontask",
            "cosinehyperscorecorrelationtask",
        ),
        overrides=("evaluation.task_configs.cosinehyperscorecorrelationtask.peptide_key=peptides",),
        cap_above=100_000,
        cap=20_000,
    ),
    "peak_level": Protocol(
        view="identified",
        tasks=("peaktypeclassificationtask", "confidencesignalanalysistask", "headanalysistask", "igattributiontask"),
        cap_above=25_000,
        cap=10_000,
        batch_size=128,
    ),
    "unlabelled": Protocol(
        view="all",
        tasks=("embeddingstatisticstask", "umapvisualisationtask"),
        overrides=("+dataset.is_annotated=false",),
        cap_above=100_000,
        cap=100_000,
    ),
    "validation": Protocol(view="identified", tasks=()),
}


def files(dataset: str) -> dict[str, Path]:
    root = env_path("EXTERNAL") / dataset
    return {name: root / f"{dataset}-{name}.parquet" for name in ("all", "identified", "probe-train", "probe-valid", "probe-test")}


def dataset_paths(dataset: str, view: str) -> list[str]:
    f = files(dataset)
    test = {"identified": f["identified"], "all": f["all"], "probe": f["probe-test"]}[view]
    return [
        "dataset=mcfm",  # the corpus schema and metadata columns; only the paths and the metadata table differ
        f"dataset.train_path={f['probe-train']}",
        f"dataset.valid_path={f['probe-valid']}",
        f"dataset.test_path={test}",
        f"dataset.search_data_path={env_path('EXTERNAL') / 'search_data.xlsx'}",  # scripts/evals/search_data.py
    ]


def rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def evaluate(*, dataset: str, model: str, protocol: str) -> Path:
    p = PROTOCOLS[protocol]
    evaluated = {"identified": "identified", "all": "all", "probe": "probe-test"}[p.view]
    n = rows(files(dataset)[evaluated])
    max_samples = p.cap if p.cap_above is not None and n > p.cap_above else None
    overrides = [
        f"evaluation.max_samples={'null' if max_samples is None else max_samples}",
        f"evaluation.batch_size={p.batch_size}",
        "evaluation.random_state=42",
        "evaluation.embedding_pooling=[mean_pool]",
        f"evaluation.tasks_to_run=[{','.join(p.tasks)}]",
        *p.overrides,
    ]
    name = f"{dataset}/{model}/{protocol}"
    dest = run(
        script=name,
        model=model,
        tier="mcfm",
        split="test",
        overrides=overrides,
        dataset_paths=dataset_paths(dataset, p.view),
        output_dir=env_path("RESULTS") / "evals" / name,
        dest=REPO / "docs" / "results" / "rerun" / name,
    )
    meta = json.loads((dest / "run.json").read_text())
    meta.update(dataset=dataset, protocol=protocol, evaluated_file=evaluated, rows_in_file=n, max_samples=max_samples)
    (dest / "run.json").write_text(json.dumps(meta, indent=1))
    return dest


ARCHITECTURE_KEYS = ("dim_model", "n_heads", "dim_feedforward", "n_layers")


def architecture_overrides(checkpoint: Path) -> list[str]:
    """`model.<key>=<value>` for the architecture stored in the checkpoint: the trainer builds the model from the Hydra
    config (the 40M of `foundation_base.yaml`), the evaluator from the checkpoint; the released 89M has 12 layers and a
    3,072-wide feed-forward, so its weights cannot load into the default model."""
    import torch

    config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
    return [f"model.{k}={config[k]}" for k in ARCHITECTURE_KEYS if k in config]


def validation_metrics(db: Path) -> dict[str, float]:
    """The `eval/*` metrics of the run's MLflow SQLite store (the trainer logs validation there)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    uuid = con.execute("select run_uuid from runs order by start_time desc limit 1").fetchone()[0]
    out = {
        key: value
        for key, value in con.execute(
            "select key, value from metrics where run_uuid=? and key like 'eval/%' order by step, timestamp", (uuid,)
        )
    }
    con.close()
    return out


def validate(*, dataset: str, model: str) -> Path:
    """The trainer's validation loop on every identified spectrum: weights from `resume_checkpoint_path`,
    `validate_before_training=True`, one training step so the run ends (the Step 2 recipe of the training log)."""
    f = files(dataset)
    name = f"{dataset}/{model}/validation"
    out = env_path("RESULTS") / "evals" / name
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = env_path("CHECKPOINTS") / f"{MODELS[model]}.ckpt"
    overrides = [
        *architecture_overrides(checkpoint),
        "dataset=mcfm",
        f"dataset.train_path={f['probe-train']}",
        f"dataset.valid_path={f['identified']}",
        f"dataset.test_path={f['identified']}",
        f"dataset.search_data_path={env_path('EXTERNAL') / 'search_data.xlsx'}",
        f"resume_checkpoint_path={checkpoint}",
        "training_steps=1",
        "validate_before_training=True",
        "num_sanity_val_steps=0",
        "max_valid_steps=null",
        "num_workers=8",
        "+mp_sharing_strategy=file_system",  # the loader setting the 90k training run used on nibi
        "post_training_evaluation.enabled=false",
        "compile_model=False",  # a compiled model prefixes its keys with _orig_mod.; load_model_state then matches nothing and only warns
        "embedding_evaluation.tasks_to_run=[embeddingstatisticstask]",
        f"model_save_folder_path={out / 'checkpoints'}",
        f"tb_summarywriter={out / 'logs'}",
        "mlflow_enabled=True",
        f"mlflow_tracking_uri=sqlite:///{out / 'mlflow.db'}",
        "mlflow_experiment_name=instanovo-fm-external-validation",
        f"evaluation.output_dir={out / 'evaluation'}",
    ]
    t0 = time.time()
    with (out / "train.log").open("a") as log:
        subprocess.run(
            [sys.executable, "-m", "instanovo_fm.trainer.train", "--config-name", "foundational", *overrides],
            cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    metrics = validation_metrics(out / "mlflow.db")
    dest = REPO / "docs" / "results" / "rerun" / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=1))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()
    (dest / "run.json").write_text(
        json.dumps(
            {
                "script": name, "dataset": dataset, "protocol": "validation", "model": MODELS[model],
                "checkpoint": str(checkpoint), "evaluated_file": "identified", "rows_in_file": rows(f["identified"]),
                "max_samples": None, "overrides": overrides, "seconds": round(time.time() - t0), "commit": commit,
                "host": os.uname().nodename,
            },
            indent=1,
        )
    )
    print(f"{name}: {len(metrics)} validation metrics -> {dest} ({time.time() - t0:.0f} s)", flush=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    args = ap.parse_args()
    os.environ.setdefault("INSTANOVO_FM_DATA_DIR", str(REPO / "data"))
    if args.protocol == "validation":
        validate(dataset=args.dataset, model=args.model)
    else:
        evaluate(dataset=args.dataset, model=args.model, protocol=args.protocol)


if __name__ == "__main__":
    main()
