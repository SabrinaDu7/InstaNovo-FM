# Why fragment-ion reconstruction did not reproduce: the metric, or the model?

Date: 10/10/2026. Script: `scripts/analysis/fragment_reconstruction.py`. Runs: `docs/analysis/fragment_reconstruction/<run>/`
(`summary.md`, `run.json`; the per-prediction parquet files stay outside the repository, path in `run.json`).

## Question

Retraining the 40M on MCFM (`docs/agent_logs/session-2026-09-25-train40M.md`) reproduced retrieval, the probes and
the embedding geometry of the released `instanovo-fm-mcfm-90k-v0.1.0`, but not the IG task's fragment-group bin
accuracy: 49.2 % against 55.0 % on LCFM test, with the b-ion median error at 827 ppm against 277
(`docs/references/results_paper_or_rerun.md`, rows 11). The trainer's own validation found the two models equal on bin
accuracy over every masked peak (27.2 % against 27.4 %, MCFM validation). Since the embeddings agree, is the gap a
property of the metric?

## What the metric is

Read from `src/instanovo_fm/eval/embed_eval_tasks/ig_attribution.py` and `ig_attribution_helper.py:527-670`, with the
suite's `configs/evaluation/default.yaml:293-303`:

- **Sample.** The first 500 spectra of the 10,000-spectrum draw (`max_spectra`), of which those with at least two
  annotated b/y fragment groups qualify (408 on LCFM test); up to 3 groups per spectrum, multi-peak groups first,
  chosen by a per-spectrum RNG: **1,224 masked groups** behind every number in rows 11.
- **Masking.** A fragment group is the base ion with its isotopes and neutral losses at one charge; the whole group is
  masked; the masked m/z is blurred by a 10 Da Gaussian drawn from the global RNG at each forward (`model/encoder.py:503`),
  seeded once before the loop, so the draw depends on the order of the 50 integrated-gradient forwards that precede it.
- **Decoding.** Greedy argmax of the group and of the offset, the offset head conditioned on the predicted group;
  `bin_accuracy` = both exact for the base ion (a 0.2 Da bin); `median_error_ppm` from the bin centre.
- **The trainer's `eval/bin_accuracy`** is a different quantity: every masked peak of the training-time span masking
  (annotated or not, 9.7 M peaks), top-1 of each head, the offset head conditioned on the true group.

Two of these are flaws of the metric regardless of the answer below. The sample is small: at n = 1,224 a single
model's accuracy moves by ±1.9 points between draws of that size (measured below), so a 5.8-point difference between
two models evaluated on the same draw is near the resolution limit of the number as published. And the median error
is bimodal: a correct bin gives about 75 ppm (half a bin), a wrong bin tens of thousands, so the median jumps between
the two modes as accuracy crosses 50 %. The b-ion "827 ppm against 277" is "b-ion accuracy fell from 50.0 % to
44.6 %", not a three-fold loss of precision (Table C below: the correct-bin medians of the three models are 73, 73
and 77 ppm).

## Method

3,000 identified spectra of `mcfm-test-00000-of-00014.parquet` (seeded draw) go through each checkpoint's own
processor with no masking; every peak is annotated with the eval's theoretical b/y ions and grouped into fragment
groups by the IG task's own functions; **every** b/y group with a base peak is masked (42,441 groups in 2,715 spectra,
35 times the IG sample), twice (the whole group; the base ion alone), with three blur seeds, and decoded three ways
(greedy as the IG task; the trainer's top-3 joint decoding; greedy with the offset head teacher-forced on the true
group as the trainer's bin accuracy). The trainer's regime is run on the same spectra: the processor's span masking
with isotope co-masking (seeded), every masked peak scored and split into annotated and unannotated. Differences are
paired by spectrum with a cluster bootstrap. Checkpoints: the released 40M, the 40M trained here, the released 89M.
Workstation RTX 4060, 1 h 50 min in all (`run.json`). A second run, `lcfm-test-n3000`, does the same on 3,000 identified
spectra of the first LCFM test shard (34,658 groups in 2,721 spectra) and adds the nine checkpoints the batch-1,024
training run kept, steps 10,001 to 90,001 (one H100, 1 h 02 min).

## Results (`fragment_reconstruction/mcfm-test-n3000/summary.md`)

Bin accuracy of the masked base ion, by regime and decoder (mean over three blur seeds; ± the spread across seeds):

| model | group, greedy (the IG task) | group, top-3 | group, teacher-forced | base only, greedy | base only, top-3 |
|---|---|---|---|---|---|
| released 40M | 42.1 % ± 0.1 | 42.6 % | 42.1 % | 43.6 % | 44.2 % |
| 40M trained here | 37.5 % ± 0.1 | 38.1 % | 37.5 % | 39.1 % | 39.7 % |
| released 89M | 52.1 % ± 0.1 | 52.6 % | 52.1 % | 51.6 % | 52.2 % |

Difference against the released 40M, points of bin accuracy with the 95 % interval from resampling spectra:

| model | group, greedy | group, top-3 | base only, greedy |
|---|---|---|---|
| 40M trained here | −4.4 [−4.8, −4.0] | −4.4 [−4.8, −4.0] | −4.3 [−4.8, −3.9] |
| released 89M | +9.7 [+9.1, +10.3] | +9.7 [+9.1, +10.3] | +7.8 [+7.3, +8.4] |

Where the deficit sits (group regime, greedy):

| model | b ions (n = 15,185) | y ions (n = 27,256) | single-peak groups (n = 12,871) | multi-peak groups (n = 29,570) | group (10 Da) accuracy | median ppm, correct bin / wrong bin |
|---|---|---|---|---|---|---|
| released 40M | 36.6 % | 45.2 % | 32.4 % | 46.3 % | 48.0 % | 73 / 200,130 |
| 40M trained here | 30.4 % | 41.5 % | 26.4 % | 42.4 % | 44.7 % | 73 / 143,354 |
| released 89M | 48.2 % | 54.3 % | 47.6 % | 54.0 % | 55.4 % | 77 / 349,035 |

The trainer's regime (span masking, every masked peak; greedy decoding):

| model | annotated peaks: bin / group | unannotated peaks: bin / group | all masked peaks: bin / group |
|---|---|---|---|
| released 40M | 58.0 / 78.2 % | 23.2 / 59.6 % | 32.5 / 64.6 % |
| 40M trained here | 53.3 / 75.6 % | 21.6 / 58.8 % | 30.1 / 63.2 % |
| released 89M | 73.8 / 86.7 % | 32.9 / 65.7 % | 43.8 / 71.3 % |

The IG task's sample size, emulated on these groups (2,000 draws of 408 spectra with up to 3 groups each, one blur
seed): a single model's accuracy has a standard deviation of 1.9 points across draws; the paired difference "trained
here minus released" averages −4.6 points with a standard deviation of 1.0 and never reaches zero in 2,000 draws; 13 %
of the groups flip between correct and wrong across blur seeds, moving a one-seed accuracy by about a point.

## Answer

**The gap is in the model, not in the metric.** On 42,441 masked groups the 40M trained here is 4.4 ± 0.4 points under
the released 40M, under every choice the metric could have made differently: the whole group masked or the base ion
alone (isotopes visible), greedy or top-3 joint or teacher-forced decoding, and any blur seed (a ±0.1 effect at this
size). The same gap appears in the trainer's own masking regime once it is restricted to annotated peaks (53.3 against
58.0 %), and even over all masked peaks on these held-out spectra (30.1 against 32.5 %): the "equality on the trainer's
criterion" of the training log was measured on the MCFM validation split, the split the checkpoint was selected on;
on test spectra, and on every external dataset (`docs/results/eval_*.md`, trainer validation rows), the retrained model
is behind there too. The metric's real flaws (1,224 groups, a median that flips at 50 %, blur noise) make the published
number imprecise and the b-ion ppm row misleading, but they do not create the gap: at the IG task's own sample size the
difference is still negative in every one of 2,000 draws.

**What differs.** The retrained model loses most on b ions (−6.2 points) and on groups with a single peak (−6.0), least
on y ions and multi-peak groups; its 10 Da group accuracy is 3.3 points lower, so it mislocates fragments at both the
coarse and the fine scale. The IG task's attribution profile on LCFM test says the same thing from the other side: the
retrained model's top-1 attributions fall on ladder neighbours less often (24.8 against 31.6 %) and on unannotated
peaks more (40.7 against 33.8 %). It predicts a hidden fragment from the fragment ladder less, from noise more.

**What closes it.** The batch-2,048 retrain of 03/10 (twice the spectra per step, 80,000 and 90,000 steps; its runs are
on nibi under `~/experiments/InstaNovo-FM/docs/references/rerun/result2_40Mb2048*_peak_level/`, not yet committed)
scores 56.6 % on the IG task at both steps, within the metric's noise of the released 55.0 % (b ions 54.5 / 53.3 %
against 50.0 %), and 3,215 ppm on the trainer's criterion against the released 4,493. So the released checkpoint's
fragment accuracy is reached with about twice the training signal our batch-1,024, 90,000-step run received, under the
same architecture and hyperparameters; the single run at the paper's stated budget lands 4 to 6 points short on
fragment reconstruction while matching on everything the mean-pooled embedding carries. Whether the released
checkpoint saw more steps, a larger batch, or simply a luckier seed and data order, the paper does not say; the
trajectory below says the single run was still gaining at step 90,000, but slowly.

## Trajectory of the batch-1,024 run (`fragment_reconstruction/lcfm-test-n3000/summary.md`)

Group regime, greedy decoding, LCFM test, 34,658 masked groups; the difference is against the released 40M, paired by
spectrum with its 95 % interval:

| checkpoint | bin accuracy | difference | gain over the previous 10,000 steps |
|---|---|---|---|
| step 10,001 | 10.0 % | −24.0 [−24.9, −23.1] | |
| step 20,001 | 18.9 % | −15.3 [−16.0, −14.6] | +8.9 |
| step 30,001 | 24.1 % | −10.2 [−10.7, −9.7] | +5.2 |
| step 40,001 | 27.5 % | −6.7 [−7.2, −6.2] | +3.4 |
| step 50,001 | 29.0 % | −5.5 [−5.9, −5.1] | +1.5 |
| step 60,001 | 30.2 % | −4.3 [−4.7, −3.9] | +1.2 |
| step 70,001 | 31.2 % | −3.4 [−3.8, −3.0] | +1.0 |
| step 80,001 | 31.7 % | −2.8 [−3.2, −2.4] | +0.5 |
| step 90,001 (the 40M trained here) | 31.9 % | −2.7 [−3.1, −2.4] | +0.2 |
| released 40M | 34.7 % | | |
| released 89M | 45.1 % | +10.5 [+10.0, +11.0] | |

Fragment accuracy was still rising at step 90,000, but the gain per 10,000 steps had fallen from 8.9 to 0.2 points,
so the run ended 2.7 points short with no sign of reaching the released checkpoint under its own schedule; whether the
flattening is the learning-rate decay or the data order is not separated here (the batch-2,048 run, which does reach
parity, changes both the signal per step and the schedule's position). The LCFM-test gap is smaller than the MCFM-test
one (2.7 against 4.4 points) on spectra where every model scores lower (released 40M 34.7 % against 42.1 %), and it has
the same shape: b ions 25.8 against 29.1 %, y ions 34.5 against 37.1 %, single-peak groups 23.0 against 26.3 %, and in
the trainer's own regime 47.7 against 50.8 % on annotated peaks, 25.9 against 27.4 % on all masked peaks. The final
checkpoint appears twice in that run (`40M-ours` and `ours-step90001`, the same file) and its rows agree to the last
digit, a check that the pipeline is deterministic under the seeds it sets.

## Flaws found in passing

- `clean_peptide_for_pyopenms` raises on a modification pyOpenMS does not know in that notation (iTRAQ, `UNIMOD:214`);
  the eval falls back to the unmodified sequence (`eval/embedding_io.py:433-438`), so labelled peptides are annotated
  with wrong b/y masses and mostly count as unannotated. The diagnostic does the same, to measure what the eval measures.
- The trainer's `eval/offset_accuracy` is teacher-forced (the offset head sees the true group); its `eval/bin_accuracy`
  is not inflated by this, because the conditioning differs only when the group is already wrong.
