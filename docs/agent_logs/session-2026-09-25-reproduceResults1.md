# Session log: reproduce the InstaNovo-FM paper's results from the released checkpoints
Date: 25/09/2026

## Initial purpose of the session

Make the paper's numbers ours before any training run: read them out of the paper, rerun every evaluation we
can on the released checkpoints (40M MCFM baseline and 89M published model) on the corpus validation and test
splits on nibi, and put paper and rerun side by side in `docs/references/results_paper_or_rerun.md`. A
training run of the 40M model is launched separately.

The steps as agreed:

1. Mine the paper (bioRxiv 10.64898/2026.09.03.747733v2) for every numerical result on the 40M and 89M
   models and record them in `docs/references/results_paper_or_rerun.md` with their source table or figure.
2. Download the training and evaluation data into `$DATA` (`~/projects/rrg-hsn/proteomies/data/proteometoolsI`,
   from the repo's `.envrc`) and the checkpoints into `$CHECKPOINTS`
   (`~/projects/rrg-hsn/proteomies/checkpoints/instanovofm`): `scripts/reproduce/fetch_data.sh`.
3. Rerun every result found, one script per result or group of results, in
   `scripts/reproduce/result<N>_<40M|89M>_<short_description>.py`, each naming the checkpoint, the data and
   the evaluation settings it used; the rerun should match the paper within the tolerance the paper's own
   protocol allows (sampling seeds, cuML versus scikit-learn probes).
4. This log.

## What was accomplished during the session

Filled in as the session goes; every row names the artifact it produced.

| Task | What we learned |
| :---- | :---- |
| Local dry run of the trainer (workstation, 500 steps, `foundational_local`) before touching nibi | Hydra's override grammar rejects `[..]` globs; the validation split is materialised in RAM (`to_dataset(in_memory=True)`) while training streams lazily; 790k validation rows crossed a 24 GB ceiling; 0.12 s/step at batch 16 for the 40M model on an RTX 4060. Recorded in `~/Documents/experiments/Arc/instanovo-fm-runs/NOTES.md` on the workstation. |
| Step 2 started (19:04 EDT): `scripts/reproduce/fetch_data.sh` detached on the nibi login node | Checkpoints arrive from GitHub at about 90 MB/s; Hugging Face shards at about 28 MB/s (MCFM 52 GiB, LCFM validation and test 162 GiB, so about 2.2 h in all). The Hugging Face tree listings with sizes and sha256 are saved next to the data (`hf_tree_*.json`) for a later checksum pass. |
| Repo housekeeping on the branch | `docs/references/results_paper_or_rerun].md` renamed to the intended `results_paper_or_rerun.md`; `.envrc` quoted `~`, which bash does not expand inside quotes, changed to `$HOME`; the Python environment for nibi is built with `uv` into `~/scratch/venvs/instanovo-fm` (`UV_PROJECT_ENVIRONMENT`) with the cache in scratch, because home is at 37 of 50 GiB. |
| Step 1 done: `docs/references/results_paper_or_rerun.md` (commit 4eaee93) | `pdftotext -layout` extracts the supplementary tables cleanly, so the mining was not hard. The 40M model's only author numbers are Table S5's ten, measured on **LCFM test**, not on MCFM. The 89M model has Tables S5/S10 (test), S4 (validation, the factorial ablation), and main-text peak-level and reconstruction numbers; the reconstruction rows (70.1 %, y/b, ppm) are the IG task's `prediction_quality` on LCFM test (Methods lines 2295 and 3730-3739). Hela qc, the de novo recall tables and the Fig. 6 panels need external data. |
| Step 3 code (commit 2a31484): `scripts/reproduce/_common.py`, `result1..5_*.py`, `submit.sh` | One protocol per result group in `_common.py` (the documented probe-and-retrieval command; the 10,000-spectrum peak-level battery; the geometry tasks), thin per-model scripts, summaries copied into `docs/references/rerun/<script>/` with a `run.json` naming overrides, commit and host. All four protocols compose against the package configs (checked on the workstation). Task settings live under `evaluation.task_configs.<task>.<key>`, not at the top level as the guide's command writes them. |
| Environment on nibi | `uv sync` into `~/scratch/venvs/instanovo-fm` (torch 2.8.0+cu128), extras `clustering` and `interpret` added; `fair-esm` is not a dependency of the fork, so an `esm` extra was added to `pyproject.toml` and the ESM2 650M weights are pre-fetched into `~/scratch/torch-hub` for the compute nodes. cuML is not installable from the lock, so probes run on scikit-learn: the guide says those numbers are not comparable with the paper's cuML ones. |
| Local reference run of the 40M checkpoint (workstation, MCFM validation sample, seven tasks) | Pipeline check, not a reproduction (different split, backend, probe sample). Retrieval recall@1 0.566 on a 20,000-spectrum MCFM pool against the paper's 0.307 on a 200,000-spectrum LCFM pool; probes within 0.03 to 0.10 of Table S5 in both directions; anisotropy ratio 23.9. Full table in the workstation's `instanovo-fm-runs/NOTES.md`. |
| First nibi job: `result5_40M_mcfm_test` (job 22690634, `rrg-hsn`, one H100, 128 GB, 8 h) | Submitted before the LCFM shards finished so the wrapper, the environment and the results path get exercised on the split that is already there. |

## Problems encountered

- The trainer's Hydra override grammar rejects bracket globs (`[01]`); `*` globs work (workstation dry run).
- The evaluation's peak-type and IG tasks store per-peak embeddings for every sample; at 20,000 samples that
  exceeded 30 GB of RAM on the workstation. On nibi the jobs request 128 GB.
- The fork's `.gitignore` covers none of `checkpoints/`, `logs/`, `mlruns` or Hydra's `outputs/`; results are
  written under `$RESULTS`, outside the checkout, and only summaries are copied into `docs/references/rerun/`.
- `.envrc` quoted `~`, which bash does not expand inside quotes; changed to `$HOME`.

## Next steps

- When `fetch.log` says `ALL_DONE` (LCFM validation and test present): submit `result1..3` for both models and
  `result4_89M_factorial_validation`; then fill the rerun columns of `results_paper_or_rerun.md` from
  `docs/references/rerun/*/`.
- Decide whether to install cuML (RAPIDS) on nibi so the probe backend matches the paper; the guide says
  scikit-learn scores are not comparable.
- IG attribution: the Methods give 413 quality-gated spectra and 1,239 masked groups, which the task's default
  `max_spectra` 500 reproduces; the 10,000 in Table S9 belongs to the peak-type and cross-spectrum analyses.
- The ablation checkpoints (`lcfm-ts-pa`, `lcfm-sa-nopa`, `lcfm-sa-pa`) against the other columns of Table S4.
