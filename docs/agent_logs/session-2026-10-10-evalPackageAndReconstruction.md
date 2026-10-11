# Session log: the fragment-reconstruction gap, and the evaluation suite as a dependency
Date: 10/10/2026

## Initial purpose of the session

Two tasks, in parallel. (1) Why did fragment-ion reconstruction not reproduce when the 40M was retrained here
(`session-2026-09-25-train40M.md`: 49.2 % against the released 55.0 % fragment-group bin accuracy, while retrieval,
probes and geometry matched)? The embeddings being alike, the user suspects the metric. (2) A branch
`sdu/switch-to-eval-package` of the fork that takes the evaluation suite from its own repository
(`SabrinaDu7/instanovofm-evals`, the suite extracted from this package) as a dependency, reruns the paper's
evaluations on the corpus test splits with it, checks them against `docs/references/results_paper_or_rerun.md`, and
then removes the embedded evaluation code; the user reviews before anything is merged.

## What was accomplished during the session

| Task | What we learned |
| :---- | :---- |
| What the IG "fragment-group bin accuracy" is (code, `eval/embed_eval_tasks/ig_attribution.py`, `ig_attribution_helper.py:527-670`, `configs/evaluation/default.yaml:293-303`): the first `max_spectra` = 500 spectra of the draw, those with at least two b/y fragment groups (408 on LCFM test), up to 3 groups each chosen by a per-spectrum RNG, multi-peak groups first; a group = the base ion plus its isotopes and neutral losses at one charge; the whole group is masked; one forward at batch 1 in fp32 with the masked m/z blurred by a 10 Da Gaussian drawn from the global RNG; greedy argmax of group and offset, the offset head conditioned on the predicted group; `bin_accuracy` = base ion's group and offset both exact; `median_error_ppm` measured from the bin centre | The headline number rests on 1,224 masked groups (±1.4 points of sampling error at 50 %) and includes blur-noise randomness replayed in a fixed order. The trainer's `eval/bin_accuracy` is a different quantity (every masked peak of the span masking, annotated or not, top-1, over 9.7 M peaks), and on it the two models are equal (27.37 % vs 27.16 %), as are group accuracy (62.4 vs 62.3 %) and offset accuracy (37.5 vs 37.2 %); the only validation metrics where the retrained model differs are the error magnitudes (median 3,897 vs 4,493 ppm, MAE 5.25 vs 5.99 Da) and the intensity head (R² 0.990 vs 0.980). |
| `median_error_ppm` on y / b ions (137 / 277 vs 203 / 827 ppm) | An artefact of a bimodal distribution: a correct bin gives about 100 ppm (half a 0.2 Da bin), a wrong bin thousands; the median jumps from one mode to the other as accuracy crosses 50 %. The 827 ppm of the retrained model's b ions is "b-ion accuracy fell below one half" (44.6 % against 50.0 %), not a three-fold loss of precision. The error distribution's quantiles or the accuracy itself are the honest statistics. |
| Attribution profile of the same task (`igattributiontask.json`, LCFM test): top-1 attributions on ladder neighbours 31.6 % (released) vs 24.8 % (retrained), complementary pairs 1.6 vs 0.3 %, unannotated peaks 33.8 vs 40.7 %, charge-variant leakage 6.6 vs 7.4 % | The retrained model predicts a masked fragment from noise peaks and charge variants more, from the fragment ladder less: a real difference in what the two models use, consistent with a lower fragment accuracy when the whole group is hidden. |
| `scripts/analysis/fragment_reconstruction.py`: the same checkpoints measured outside the framework on a seeded draw of 3,000 MCFM test spectra, every b/y group of every spectrum, two masking regimes (whole group; base ion only), three decoders (greedy; the trainer's top-3 joint; teacher-forced offsets), three blur seeds, plus the trainer's own span-masking regime with masked peaks split into annotated and unannotated; paired cluster-bootstrap intervals by spectrum | `docs/analysis/fragment_reconstruction.md`: the gap is in the model, not the metric. On 42,441 masked groups the retrained 40M is 4.4 ± 0.4 points under the released 40M under every variant of the metric (whole group or base ion masked; greedy, top-3 or teacher-forced; any blur seed, a ±0.1 effect), and in the trainer's own regime restricted to annotated peaks (53.3 against 58.0 %) and even over all masked peaks on test spectra (30.1 against 32.5 %); the equality seen on the trainer's criterion held only on the validation split the checkpoint was selected on. At the IG task's own sample size the difference is negative in every one of 2,000 emulated draws (mean −4.6, sd 1.0). The deficit concentrates on b ions (−6.2) and single-peak groups (−6.0). The batch-2,048 retrain (twice the spectra per step) scores 56.6 % on the IG task, within noise of the released 55.0 %: the released checkpoint's fragment accuracy needs about twice the training signal of the paper's stated budget in our hands. Teacher forcing cannot change bin accuracy (the conditioning differs only when the group is already wrong), so the trainer's bin accuracy is not inflated by it. |
| Branch `sdu/switch-to-eval-package` (fork) and `sdu/paper-protocols` (suite): the suite gains the corpus splits as datasets (`lcfm-test`, `lcfm-valid`, `mcfm-test`, `mcfm-valid` under `$CORPUS`) and the paper's protocols (`paper-probes-retrieval` 200,000 spectra and 20,000 duplicate groups, `paper-peak-level` 10,000, `paper-geometry` 20,000, `paper-validation` 250 batches), with the probe's splits always the tier's three and the per-file metadata from the merged table or InstaNovo-FM's own (suite commit d326878); the fork depends on it (`pyproject.toml` `[tool.uv.sources]`, ssh git source pinned to that commit, `uv.lock`), its result scripts name the suite's protocols, `scripts/evals/run.py` delegates to the suite, `scripts/reproduce/compare_package.py` compares the suite's runs with the embedded evaluator's kept under `docs/references/rerun_embedded/`; job scripts take their checkout and venv from the submission (`SLURM_SUBMIT_DIR`, `FM_VENV`), so a second clone on nibi (`~/experiments/InstaNovo-FM-evalpkg`, venv `~/scratch/venvs/instanovo-fm-evalpkg`) runs the branch without touching the clone the retraining sessions use | Nine jobs (`pkg_result*`, `pkg_validate_40M_mcfm_valid`) rerun every row of the 40M and 89M tables through the suite; outcome below when they finish. The repository is private, hence an ssh source rather than the https form of the example. |
| What removing the embedded evaluation code would take (inventory, for the review): `src/instanovo_fm/eval/` is 53 modules; the suite covers `evaluator`, `embedding_io`, `embed_eval_tasks/*`, `probe_splitting`, `spectrum_metrics/*` and the chemistry helpers. It does not cover the de novo baseline tooling that lives in the same folder (`run_xuanjinovo`, `score_xuanjinovo`, `_xuanjinovo_encoder`, `predict_casanovo_de_novo`, `run_baseline_de_novo`, `_predict_de_novo_common`, `extract_*_embeddings`, `convert_parquet_to_mgf`, `cross_set_dataset`, `embedding_sequence_similarity_study`, `explainability_analysis`, `runs_classificatoin_eval`, `run_tasks_from_embeddings`, `split_data_path`, the three READMEs). Users of the folder outside it: `cli.py` (`evaluate` command), `trainer/train.py` (in-loop embedding evaluation at 1318 and post-training evaluation at 1504), eleven `scripts/downstream/*` scripts (spectral rescue, cross-set transfer: the suite has those tasks and metrics), `scripts/run_baseline_*.sh`, `configs/evaluation/`, four `configs/foundational_eval_*.yaml`, fifteen test modules, and the docs | Done on the branch (one commit, reversible): `src/instanovo_fm/eval/` is gone; the de novo baseline tooling moved to `src/instanovo_fm/baselines/` (its one dependency on the evaluator, `embedding_io.save`, now comes from the suite; `tests/test_baseline_isolation.py` guards the new package); `cli.py evaluate` is the suite's `run` with `--dataset`/`--protocol`; the trainer's in-loop embedding evaluation is removed and `post_training_evaluation` runs the suite's protocols on `model_best.ckpt` (`foundational.yaml`: `dataset`, `protocols`, `results_dir`), logging each task's summary to the tracker; `configs/evaluation/` and the eight `foundational_eval_*.yaml` are deleted, `scripts/downstream` imports the suite's tasks and metrics, the extras `clustering`, `interpret`, `esm` pass through to the suite's; twelve tests of the removed modules are deleted, the baseline tests follow their modules (88 passed); README, docs and the Docker notes describe the suite. Two things wait on the suite: the cross-set and spectral-rescue runs have no suite protocol yet (their `foundational_eval_*` configurations went with the evaluator), and the baselines' `run_baseline_eval.sh` chain (extract embeddings, then `run_tasks_from_embeddings`) is gone, since the second step was the evaluator. Historical logs and the reference tables keep their mentions of the old paths. |

## Problems encountered
- The diagnostic's annotation raised on a peptide with an iTRAQ label (`UNIMOD:214`), which pyOpenMS rejects in
  that notation; the eval silently falls back to the unmodified sequence (`eval/embedding_io.py:433-438`), and the
  diagnostic now does the same. The fallback is itself a flaw worth knowing: labelled peptides are annotated as if
  unlabelled, so their b/y masses are off by the label and they mostly count as unannotated.

## Handoff for the next session (written 10/10 evening; nibi's GPU partition drained all day)

**State of the branches.**
- Fork `sdu/switch-to-eval-package` @ aefd0f3, pushed; the workstation clone and nibi's second clone
  `~/experiments/InstaNovo-FM-evalpkg` (venv `~/scratch/venvs/instanovo-fm-evalpkg`, `.envrc` with `EXTERNAL`, `CORPUS`,
  `EVAL_WORKDIR`, `FM_VENV`) are on it. It depends on the suite (`pyproject.toml` `[tool.uv.sources]`, ssh git source,
  `rev = "d326878"`), runs the paper's protocols through it, and no longer contains `src/instanovo_fm/eval/`.
- Suite `instanovofm-evals`, branch `sdu/paper-protocols` @ 3bae95d, pushed: corpus datasets, `paper-*` protocols
  (d326878), the A/B harness for corpus splits and the regression note (two later commits; no package code change, so the
  fork's pin can stay or move to the branch's merge commit, then `uv lock`).
- Fork `sdu/reproduce-results-40M` untouched @ d56b9af. nibi's main clone `~/experiments/InstaNovo-FM` carries the
  uncommitted batch-2,048 peak-level runs (`docs/references/rerun/result2_40Mb2048*_peak_level/`) and the old
  `.gitignore` edit.
- Merge order for the review: the suite branch first, then the fork branch (the fork pins the suite's commit).

**Task 1, answered:** `docs/analysis/fragment_reconstruction.md` (the gap is in the model; the batch-2,048 retrain
closes it). One addendum pending: job `fragrecon_lcfm` (23703865) runs the same diagnostic on LCFM test with the nine
kept checkpoints of the batch-1,024 run; its `summary.md` lands in `$RESULTS/analysis/fragrecon-lcfmtest-n3000/` and
belongs under `docs/analysis/fragment_reconstruction/lcfm-test-n3000/` with a paragraph on the trajectory.

**Task 2, one confirmation pending.** Jobs 23703686-23703694 (`pkg_result1_40M_probes_retrieval`,
`pkg_result1_89M_probes_retrieval`, `pkg_result2_{40M,89M}_peak_level`, `pkg_result3_{40M,89M}_geometry`,
`pkg_result4_89M_factorial_validation`, `pkg_result5_40M_mcfm_test`, `pkg_validate_40M_mcfm_valid`) rerun every row of
the 40M and 89M tables through the suite; submitted 11:3x from the evalpkg clone, pending all day
(`ReqNodeNotAvail`; 30 of the GPU nodes drained). When they have run:

```
cd ~/experiments/InstaNovo-FM-evalpkg && source .envrc
sacct -X -n -o JobName%40,State,Elapsed -S 2026-10-10T11:00 | grep -E "pkg_|fragrecon"
$FM_VENV/bin/python scripts/reproduce/compare_package.py        # -> docs/references/package_vs_embedded.md
$FM_VENV/bin/python scripts/reproduce/fill_results.py           # results_paper_or_rerun.md from the package runs
git add docs/references && git commit && git push origin sdu/switch-to-eval-package   # from the login node
```

Expected from the local A/B (suite `docs/regression.md`): every metric identical to 1e-9 except the unseeded peak-type
macro-F1 (±0.005) and the confidence AUROCs within 1e-5. Anything else is a finding about the corpus-split wiring
(`protocols.py::corpus_paths`, `runner.py::search_data_path`), not about the tasks. The package's runs also overwrite
`docs/references/rerun/<script>/`; the embedded evaluator's originals stay under `docs/references/rerun_embedded/`.

**Open items for the review** (also in the removal row above): the cross-set and spectral-rescue runs have no suite
protocol yet; the baselines' `run_baseline_eval.sh` chain is gone with the evaluator; `instanovo-fm evaluate` now takes
`--dataset`/`--protocol`; historical logs and reference tables keep the old paths on purpose; the batch-2,048 peak-level
rows (row 11 of their section) are not yet filled or committed.
