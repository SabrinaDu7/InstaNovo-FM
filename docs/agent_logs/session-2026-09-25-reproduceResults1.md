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
| First nibi job: `result5_40M_mcfm_test` (job 22690634, then 22691003 after the fix below; `rrg-hsn`, one H100, 128 GB, 8 h) | Submitted before the LCFM shards finished so the wrapper, the environment and the results path get exercised on the split that is already there. The model loads and samples 200,000 of the 5,761,808 MCFM test spectra in 25 s; theoretical-spectrum generation for them takes about 2.5 min. |
| Smoke test passed: job 22691003, 1 h 37 min on g10 (`docs/references/rerun/result5_40M_mcfm_test/`, rows 16-17 of the results document, commit 53ba3e1) | 40M on MCFM test, 200,000-spectrum pool: recall@1 0.629, mAP@20 0.329; probes fragment type 0.776, instrument 0.680, PTM 0.799, hydrophobicity 0.622, mass 0.755, m/z 0.928, charge 0.608, confidence 0.981 (scikit-learn, shared projects). Where the time went: embedding 200,000 spectra with `num_workers=0` about 30 min (the GPU mostly idle), linear probe 34 min on CPU, duplicate retrieval 47 min because the installed `faiss-cpu` loads without its AVX2/AVX512 modules. Every 200,000-sample job will cost about this until workers and a faster FAISS build are restored. |
| Step 2 closed at 20:04 (`ALL_DONE`, 215 files, 53 GiB MCFM + 163 GiB LCFM validation/test + 3.5 GiB checkpoints); `verify_data.py` running detached (sha256 of every shard against the saved listing, report in `$DATA/verify.csv`) | |
| The watcher submitted the seven LCFM jobs at 21:20: 22694948 result1_40M, 22694949 result1_89M, 22694950 result2_40M, 22694951 result2_89M, 22694952 result3_40M, 22694953 result3_89M, 22694954 result4_89M | Each one H100, 128 GB, 8 h; results land in `docs/references/rerun/<script>/` and `fill_results.py` renders them into the document. |
| Job 22694948 (`result1_40M`) failed at 10 min: `No files matching .../splits/lcfm/lcfm-*train*.parquet` | The probe task with `use_project_split=false` still draws its 100,000 / 10,000 / 10,000 samples from the corpus's own train / validation / test files (confirmed in the smoke-test log: model-train 100,000, model-valid 10,000, model-test 10,000), so every probe job needs LCFM train, which step 2 had left out. LCFM train (293 shards, 305 GiB) is downloading since 21:33 (`$DATA/fetch_lcfm_train.sh`, about 3.3 h); a watcher verifies it and submits `result1_40M`, `result1_89M`, `result4_89M` when it lands. The four pending jobs were cancelled; the three without a probe task were resubmitted at once (22695359 `result2_89M`, 22695360 `result3_40M`, 22695361 `result3_89M`); `result2_40M` (22694950) kept running past the failure point. |

## Problems encountered

- The trainer's Hydra override grammar rejects bracket globs (`[01]`); `*` globs work (workstation dry run).
- The evaluation's peak-type and IG tasks store per-peak embeddings for every sample; at 20,000 samples that
  exceeded 30 GB of RAM on the workstation. On nibi the jobs request 128 GB.
- The fork's `.gitignore` covers none of `checkpoints/`, `logs/`, `mlruns` or Hydra's `outputs/`; results are
  written under `$RESULTS`, outside the checkout, and only summaries are copied into `docs/references/rerun/`.
- `.envrc` quoted `~`, which bash does not expand inside quotes; changed to `$HOME`.
- Job 22690634 (`result5_40M_mcfm_test`) failed 50 s into embedding generation with
  `rebuild_storage_fd: unable to mmap ... Cannot allocate memory (12)`: the evaluation DataLoader's eight
  workers (top-level `num_workers`) could not hand tensors to the main process through shared memory on
  the compute node; RSS was 14.7 GB of the 128 GB requested, so not a RAM shortage. `_common.run` now
  passes `num_workers=0` (commit 2561c4e); resubmitted as job 22691003.
- The nibi venv runs on the cluster's Python 3.11.4 (uv picked the system interpreter, not a managed one).

## Next steps

- As each of the seven LCFM jobs finishes: `python scripts/reproduce/fill_results.py`, commit, and compare with the
  author column; the retrieval rows are the ones that must match, the probe rows carry the backend caveat.
- Restore DataLoader workers for evaluation with `torch.multiprocessing.set_sharing_strategy("file_system")` (or
  a larger `/dev/shm` request) and install a FAISS build with AVX2, so a 200,000-sample job takes under an hour.
- Decide whether to install cuML (RAPIDS) on nibi so the probe backend matches the paper; the guide says
  scikit-learn scores are not comparable.
- IG attribution: the Methods give 413 quality-gated spectra and 1,239 masked groups, which the task's default
  `max_spectra` 500 reproduces; the 10,000 in Table S9 belongs to the peak-type and cross-spectrum analyses.
- The ablation checkpoints (`lcfm-ts-pa`, `lcfm-sa-nopa`, `lcfm-sa-pa`) against the other columns of Table S4.
