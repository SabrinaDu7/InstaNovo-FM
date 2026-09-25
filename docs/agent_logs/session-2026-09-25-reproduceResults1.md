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

## Problems encountered

- Filled in as they come.

## Next steps

- Step 1: paper tables into `docs/references/results_paper_or_rerun.md`.
- Step 3: the rerun scripts, first on the 40M checkpoint on the MCFM validation and test splits.
