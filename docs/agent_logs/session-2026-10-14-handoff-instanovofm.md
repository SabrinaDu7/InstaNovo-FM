# Handoff: using this fork of InstaNovo-FM, and what differs from the original
Date: 14/10/2026

## Purpose

For teammates picking up `SabrinaDu7/InstaNovo-FM` (`main`). It says what this fork is, how it differs from
InstaDeep's repository, how to install and run it (workstation and nibi), where the data, checkpoints and results
are, and what is known to be imperfect. Companion documents: `14-10-2026-handoff-data.md` in nibi's
`~/projects/rrg-hsn/proteomies/data/` (the datasets), the suite's own README (`SabrinaDu7/instanovofm-evals`), and
the session logs in this directory for the history behind each number.

## What this fork is

The fork point is upstream's merge commit 5953cc4 (`instadeepai/InstaNovo-FM`, PR #40). `main` carries the work of
every session log in this directory on top of it. By area (`git diff --name-status 5953cc4..main`):

| Area | Change | Where |
| :---- | :---- | :---- |
| Evaluation | The evaluation code is no longer in this repository. It is the package `instanovofm-evals`, a dependency pinned to a commit under `[tool.uv.sources]` in `pyproject.toml`. `src/instanovo_fm/eval/` is deleted; `instanovo-fm evaluate` calls the suite; the trainer's in-loop embedding evaluation is gone and a `post_training_evaluation` block runs the suite's paper protocols on `model_best.ckpt` at the end of training; `configs/evaluation/` and the `foundational_eval_*.yaml` configs are gone; the extras `interpret`, `clustering`, `esm` pass through to the suite's. | `pyproject.toml`, `src/instanovo_fm/cli.py`, `src/instanovo_fm/trainer/train.py`, `src/instanovo_fm/configs/foundational.yaml` |
| Baselines | The de novo and embedding baseline tooling (Casanovo, XuanjiNovo, InstaNovo) that lived under `eval/` moved to `src/instanovo_fm/baselines/`, unchanged in substance. | `src/instanovo_fm/baselines/`, `docker/` |
| Not ported | Cross-set annotation transfer, spectral rescue (all variants) and the embedding similarity study were deleted with the evaluator and have no protocol in the suite. The downstream scripts that post-process their outputs remain under `scripts/downstream/` and import the suite's modules. | `scripts/downstream/` |
| Paper reproduction | One script per published table on nibi, the paper's numbers beside the reruns for the released 40M and 89M, the 40M retrained here, and the batch-2,048 retrain. | `scripts/reproduce/`, `docs/references/results_paper_or_rerun.md`, `docs/references/rerun/` |
| Package check | The nine reproduction scripts rerun through the suite, compared metric by metric with the embedded evaluator's runs: 220 of 227 metrics identical to 1e-9, the seven others explained in the session log of 10/10. | `scripts/reproduce/compare_package.py`, `docs/references/package_vs_embedded.md`, `docs/references/rerun_embedded/` |
| Training on nibi | A SLURM wrapper and a status reader for training runs; the 40M retrained at the paper's budget and at a global batch of 2,048. | `scripts/train/`, `docs/agent_logs/session-2026-09-25-train40M.md`, `session-2026-10-02-batch2048.md` |
| Our datasets | Export of our labelled datasets (CAMPI, MS2Bac, ProteomeTools, UPS1) into the corpus schema, the suite run on them, and the rendered result pages. | `scripts/evals/`, `docs/results/eval_<dataset>.md` |
| Analysis | Why fragment-ion reconstruction did not reproduce for the retrained 40M (the model, not the metric). | `scripts/analysis/fragment_reconstruction.py`, `docs/analysis/` |

Everything in `docs/` that names a number points at the file and commit it came from; `scripts/reproduce/fill_results.py`
regenerates the results table from those files and `compare_package.py` regenerates the package comparison, on any clone.

## Install

From a clone, with uv. `pip install instanovo-fm` installs InstaDeep's release from PyPI, not this fork, and pip does
not read `[tool.uv.sources]`, so it cannot find the suite.

```bash
git clone git@github.com:SabrinaDu7/InstaNovo-FM.git && cd InstaNovo-FM
uv sync --extra interpret --extra clustering --extra esm
cp .envrc.example .envrc   # then edit the paths; see below
```

`uv sync` fetches `instanovofm-evals` from `ssh://git@github.com/SabrinaDu7/instanovofm-evals.git` at the pinned
commit, so the machine needs an ssh key registered with GitHub and read access to that repository (it is private).

On nibi the clone is `~/experiments/InstaNovo-FM` (on `main`) with its environment at `~/scratch/venvs/instanovo-fm`,
built with `UV_PROJECT_ENVIRONMENT=$HOME/scratch/venvs/instanovo-fm uv sync --frozen --inexact --extra clustering
--extra interpret --extra esm`. Every `scripts/*/submit.sh` uses that environment unless `FM_VENV` in `.envrc` names
another.

## `.envrc`

Copied from `.envrc.example`, git-ignored, sourced by every script and SLURM wrapper. On nibi:

| Variable | Value on nibi | Used for |
| :---- | :---- | :---- |
| `DATA` | `~/projects/rrg-hsn/proteomies/data/instanovofm` | the corpus (`splits/<tier>/`, `peptide_registry.parquet`) |
| `CORPUS` | `$DATA/splits` | the `paper-*` protocols |
| `EXTERNAL` | `~/projects/rrg-hsn/proteomies/data` | our exported datasets (`<dataset>/`) and `search_data.xlsx` |
| `CHECKPOINTS` | `~/projects/rrg-hsn/proteomies/checkpoints/instanovofm` | `<tag>.ckpt`; the tags are `scripts/reproduce/_common.py::MODELS` |
| `RESULTS` | `~/projects/rrg-hsn/proteomies/results/instanovofm` | full task outputs of the suite (large) |
| `EVAL_WORKDIR` | `$RESULTS/package-workdir` | the suite's scratch |
| `RUNS` | `~/projects/rrg-hsn/proteomies/runs/instanovofm` | training runs |

The workstation copy points at the local mirrors instead (two MCFM train shards, the MCFM validation and test shards,
the exported datasets).

## Evaluate a checkpoint

The suite decides what a number means: a *protocol* fixes which spectra, which caps, which seed and which tasks. Its
README lists the tasks and protocols; `src/instanovofm_evals/protocols.py` is the definition.

```bash
# the paper's protocols, on a corpus split
instanovo-fm evaluate --checkpoint $CHECKPOINTS/instanovo-fm-mcfm-90k-v0.1.0.ckpt --dataset lcfm-test --protocol paper-probes-retrieval --name 40M
# our datasets (protocols: probes, retrieval, geometry, peak_level, unlabelled, validation)
instanovo-fm evaluate --checkpoint <ckpt> --dataset ms2bac --protocol retrieval --name my-model
# the same through the fork's scripts, results committed under docs/results/rerun/<dataset>/<model>/<protocol>/
python scripts/evals/run.py --dataset ms2bac --model 40M --protocol retrieval
sbatch --job-name=ev_ms2bac_40M_retrieval scripts/evals/submit.sh --dataset ms2bac --model 40M --protocol retrieval
python scripts/evals/submit_all.py --datasets ms2bac --models 40M --protocols retrieval probes   # memory and time per protocol
python scripts/evals/render_results.py --dataset ms2bac                                         # -> docs/results/eval_ms2bac.md
```

`evaluate` writes `<results>/<dataset>/<name>/<protocol>/<task>.json` (the headline metrics), `<task>.results.json`
(everything the task computed) and `run.json` (checkpoint, protocol, library versions, commits, seconds).

## Reproduce a paper table

```bash
sbatch --time=02:59:00 --job-name=result1_40M scripts/reproduce/submit.sh scripts/reproduce/result1_40M_probes_retrieval.py
python scripts/reproduce/fill_results.py       # docs/references/results_paper_or_rerun.md from docs/references/rerun/
python scripts/reproduce/compare_package.py    # docs/references/package_vs_embedded.md, suite runs against the embedded evaluator's
```

Each `result*.py` names a model tag, a corpus split and a protocol (`_common.run`), copies the suite's run into
`docs/references/rerun/<script>/`, and `fill_results.py` reads the rows from there. Every script took under 1 h 50 min
on one H100 with 128 GB (`run.json` of each run); keep `--time` under three hours so the job lands in the 3-hour GPU
partition, which backfills far sooner than the 12-hour one.

## Train

```bash
sbatch --time=14:00:00 --job-name=train-40M scripts/train/submit.sh <run-name> [hydra overrides]
python scripts/train/status.py <run-name>     # SLURM state, train.log tail, MLflow metrics, checkpoints
```

The run lives in `$RUNS/<run-name>`; `STAGE=1` copies the MCFM shards to node-local disk first. The 40M at the paper's
budget (90,000 steps, batch 1,024) took 9 h 54 min on one H100. Two things to decide before a long run:
`post_training_evaluation.enabled` (default True) runs the three paper protocols on LCFM test when training ends,
which needs `$CORPUS` with LCFM present and more than 128 GB (`session-2026-09-25-train40M.md`); set it to `false`
and run `scripts/reproduce/` afterwards instead. And `compile_model` (default True) prefixes the saved keys with
`_orig_mod.`; the suite's validation adapter disables it when it reloads a checkpoint.

## nibi

- Account `rrg-hsn` (`rrg-hsn_gpu` for GPU jobs). Priority is fairshare-bound: the account has consumed several
  times its share, so jobs queue for hours regardless of what they request; `def-hsn_gpu` is nearly unused.
- GPU partitions by time limit: under 3 h, 12 h, 24 h. Our jobs ask for one H100, 12 CPUs, 128 GB, the queue's most
  common request.
- DataLoader workers failed in the shared-memory handoff on compute nodes (`session-2026-09-25-reproduceResults1.md`);
  the evaluation runs use `num_workers=0`. Numba needs `NUMBA_THREADING_LAYER=tbb` under EVoC; the wrappers set it.
- `faiss-cpu` loads without AVX2 there and on the workstation; the retrieval task's per-query search loop takes
  about 50 min on a 200,000-spectrum pool on nibi and about 9 h on the workstation.

## Known imperfections, inherited or found

- Two tasks are unseeded in the original code and in the suite: the peak-type probe's classifier (torch's global
  RNG; macro-F1 moves by about 0.005 between identical runs) and the cosine-hyperscore subsample above 10,000
  spectra. Both are documented in the task headers; fixing them changes the numbers against the paper.
- The original evaluator's peak-level annotation differs by one peak in about 77,000 between two processes on the
  same spectra; the matcher itself is deterministic, the variation enters in the processed spectra (suite
  `docs/regression.md`). Peak-level AUROCs agree to 1e-5, position counts to ±2.
- The paper's probe macro-F1 values were computed with cuML on project-disjoint splits; ours use scikit-learn on the
  tier's splits, and the macro-F1 rows fall short of the paper by a few points while every other probe metric and
  retrieval reproduce (`results_paper_or_rerun.md`).
- The retrained 40M matches the released one on everything the mean-pooled embedding carries and is 2.7 to 4.4
  points under on fragment-ion reconstruction; the batch-2,048 retrain closes the gap (`docs/analysis/`).
- Updating the suite: change `rev` in `pyproject.toml`, `uv lock --upgrade-package instanovofm-evals`, re-sync every
  environment that runs it. The validation adapter passes Hydra overrides into this trainer, so a key removed from
  `configs/foundational.yaml` breaks it; the suite's `tests/test_contract.py` composes the overrides against this
  repository's config whenever `instanovo_fm` is importable, so run the suite's tests from this environment after
  touching the trainer config.

## Next steps

- Decide whether the cross-set and spectral-rescue evaluations are wanted as suite protocols.
- Remove the redundant second clone on nibi (`~/experiments/InstaNovo-FM-evalpkg`, venv
  `~/scratch/venvs/instanovo-fm-evalpkg`), kept from the branch's confirmation runs.
- Make the suite repository public if friends outside the GitHub organisation need `uv sync` to work.
