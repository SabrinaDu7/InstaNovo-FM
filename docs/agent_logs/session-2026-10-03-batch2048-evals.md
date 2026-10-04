# Session log: does the 40M retrained at a global batch of 2,048 reproduce the released 40M?
Date: 03/10/2026 (continues on 04/10)

## Initial purpose of the session

Evaluate the batch-2,048 retrain (job 23151784, `$RUNS/train-40M-mcfm-90k-b2048`,
`session-2026-10-02-batch2048.md`) with the same corpus reruns and external-dataset protocols as the released 40M
and the first retrain (`40M-ours`, batch 1,024), and decide whether it reproduces the released checkpoint. Criterion,
fixed before any result was read: every row within sampling noise of the released 40M rerun, with fragment-group bin
accuracy on LCFM test (IG task; released 55.0 %, batch 1,024 49.2 %) the deciding row. Both clones (workstation,
nibi) on `sdu/reproduce-results-40M`.

## What was accomplished during the session

| Task | What we learned |
| :---- | :---- |
| Checkpoints of the run | `model_best.ckpt` is byte-identical to `model_epoch_15_step_90001.ckpt` (epoch 15, global_step 90001); `model_epoch_14_step_80001.ckpt` carries epoch 14 at step 80,001, the released checkpoint's own counters. The trainer logged 5,713 actual steps per epoch (batch 1,024: 11,428). Both evaluated: step 80,001 as `40M-b2048`, step 90,001 as `40M-b2048-90k`, copied unchanged into `$CHECKPOINTS` as `instanovo-fm-mcfm-90k-b2048-step{80k,90k}-2026-10-03.ckpt` (commit 1fde366). |
| Trainer validation (MCFM validation, 256,000 spectra), `docs/references/rerun/train_40M_mcfm_90k_b2048/` (commit 8672a51) | Step 80,001: median 3,306 ppm, MAE 5.07 Da, bin accuracy 29.4 %, within 20 ppm 4.5 %, intensity R² 0.990; step 90,001: 3,215 ppm, 5.05 Da, 29.6 %, 4.6 %, 0.990. Batch 1,024 at step 90,001: 3,897 ppm, 27.2 %; released 40M: 4,493 ppm, 27.4 %. On the trainer's own criterion the batch-2,048 run is ahead of both from step 60,001 on (bin accuracy 28.2 % there). Effective rank of the in-loop embedding statistics 83 (batch 1,024: 95; released on the reproduction protocol: 89). |
| Scaffolding: result scripts `result{1,2,3,5,6}_40Mb2048[s90k]_*.py`; `fill_results.py` builds every retrain's rows from one function (`_retrain_rows`), the document unchanged for the existing sections (`--check` up to date); a new section in `results_paper_or_rerun.md` with both steps beside the batch-1,024 run and the released rerun; `render_results.py` / `plot_results.py` gain the two checkpoints (commit 2318284) | The plot's five-series palette passes the dataviz validator only in slot order (released 40M, 40M trained here, 89M, then the two new checkpoints); yellow beside orange fails the normal-vision floor (ΔE 13.7). |
| `submit_all.py` retrieval wall time 18 h per unit pool (was 12 h; commit before the external submissions) | ProteomeTools retrieval took 6 h 19 min to 10 h 04 min on the same pool depending on the node, CAMPI 23 h 14 min to 28 h 03 min; the 12 h rule left a 20 % margin on the slowest. |
| Jobs submitted 03/10 23:10 from nibi's clone: corpus 23216892 / 23216893 (result2, steps 80k / 90k), 23216894-97 (result1, 3, 5, 6 at 80k), 23216898-901 (the same at 90k); external 48 jobs 23216918-23216985 (4 datasets x 2 checkpoints x 6 protocols, `$RESULTS/evals/jobs.csv`) | None had started by 04/10 13:00: every one PENDING (`ReqNodeNotAvail`) through the outage below. No result of the batch-2,048 checkpoints exists yet beyond the trainer's own validation, so the reproduction question is open; the document's cells stay "-". |

## Problems encountered
- nibi lost almost every node from about 13:15 on 03/10 ("Node unexpectedly rebooted"; all 37 GPU nodes down,
  drained or invalid, 6 of about 700 CPU nodes up). Every job submitted here stayed PENDING with `ReqNodeNotAvail`.
- 48 `ife_*` jobs appeared under the same account at 00:02 on 04/10 from `~/experiments/instanovofm-evals` (another
  session's package); separate checkout and outputs, so no collision; not touched.
- `validate_palette.js` of the dataviz skill is an ES module in a `.js` file; under Node 18 it needs a
  `package.json` with `"type": "module"` beside it.

## Next steps
- When the corpus jobs end (`sacct -j 23216892,23216893,...`): `git pull` on the workstation after committing the
  `docs/references/rerun/result*_40Mb2048*` folders on nibi, `python scripts/reproduce/fill_results.py`, and read row 11
  of the new section against the released 55.0 % by the criterion stated there.
- When the external non-retrieval jobs end: `render_results.py --dataset <d> --registry $DATA/peptide_registry.parquet`
  for the four datasets on nibi, commit, then `plot_results.py` (PNGs stay ignored). ProteomeTools / CAMPI retrieval
  (23216961, 23216968, 23216975, 23216982; 18 h and 54 h limits) lands later; re-render then.
- If the batch closes the gap, the half-size training split is no longer needed as an explanation; if not, it is the
  next run (`session-2026-10-02-batch2048.md`).
