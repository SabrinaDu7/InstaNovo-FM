# Session log: train the 40M InstaNovo-FM model from scratch on nibi
Date: 25/09/2026 (started 23:30 EDT, continues on 26/09)

## Initial purpose of the session

Reproduce `instanovo-fm-mcfm-90k-v0.1.0` (9 layers, d 768, ffn 1024, about 40 M parameters, about 90,000 steps
on the MCFM tier) by training it ourselves on nibi, and document the training pipeline's design choices while
doing so: what is good engineering, what looks ad hoc, what could be improved, and every problem met. The
evaluation reproduction of the released checkpoints (same branch, `session-2026-09-25-reproduceResults1.md`)
supplies the numbers a trained model has to reach.

Steps: (1) read the training path end to end; (2) the trainer's own validation metrics on the released 40M
checkpoint as the target in the units the training log prints; (3) a smoke job on one H100 that settles data
loading, throughput, memory, checkpointing, in-loop evaluation and local MLflow; (4) the 90,000-step run;
(5) the result scripts on the best checkpoint beside the released model and the paper; (6) this log.

## What was accomplished during the session

| Task | What we learned |
| :---- | :---- |
| Step 1: the training path read end to end (`trainer/train.py`, `common/trainer.py`, `data/data.py`, `data/masking.py`, `model/encoder.py`, `model/heads.py`, `trainer/losses.py`, `trainer/metrics.py`, `configs/foundational.yaml`, `configs/model/foundation_base.yaml`, Table S8) | The entries under "Training choices" below. The run itself is one command: `instanovo-fm train dataset=mcfm training_steps=90000` plus the three split paths, because `foundation_base.yaml` already is the 40M architecture and `foundational.yaml` already carries Table S8's schedule (LR 1e-4, 5 % warmup, 30 % hold, cosine to 0.1×, batch 1,024, clip 1.0, fp16, checkpoints every 10,000 steps on median ppm error, top 3). The paper never states the MCFM run's batch size or GPU count; "differs only in depth, parameter count, training budget and corpus" is the whole specification. |
| Step 2 launched: `scripts/train/submit.sh` (run directory under `$RUNS`, local MLflow in `sqlite:///<run>/mlflow.db`, optional staging to node disk, a GPU sampler) and `scripts/train/status.py` (SLURM state, log, MLflow metrics from the SQLite store, checkpoints in one screen); jobs 22699796 (failed) and 22700328 | The trainer already has a validation-only path: `resume_checkpoint_path` loads weights (`common/trainer.py:912`), `validate_before_training=True` validates before any step, and `training_steps=1` ends the run after one step, so no code was added for Step 2. A config-driven `mp_sharing_strategy` hook was added to `trainer/train.py::main` (commit f93cece) for the loader-worker test. |
| Step 2 done: job 22700328, 16 min; the released 40M checkpoint through the trainer's validation loop on 256,000 MCFM validation spectra (9.7 M masked peaks); metrics in `docs/references/rerun/validate_released_40M/metrics.json` and row 23 of the 40M table | The targets in the trainer's own units: `eval/median_ae_ppm` 4,493 ppm (the checkpoint criterion), `eval/mae_daltons` 5.99, bin accuracy 27.4 % (±1 bin 31.5 %; group 62.4 %, top-5 99.5 %; offset 37.5 %, top-5 70.4 %), within 0.1 Da 27.4 %, within 1 Da 39.8 %, within 20 ppm 4.1 %, intensity R² 0.980, loss 1.517. Eight loader workers with the `file_system` sharing strategy ran the 250 validation batches in 2 min (the single-process evaluation jobs took 30 min for 200,000 spectra); the in-memory validation load took 12 min before the first batch. |
| Step 3 done: smoke job 22700918 (`smoke-40M`), one H100 on g7, 500 steps at batch 1,024, 25 min wall | Staging the 53 GB MCFM tier to node-local disk took 154 s. Node: 12 CPUs, 128 GB, /dev/shm 377 GB (so the earlier worker failure was the descriptor-passing strategy, not shared-memory size), open-file limit 131,072. Timeline: data handles open at +1 min, validation split in RAM and model set up by +15 min (torch.compile itself 25 s), then 500 steps in 3 min 53 s: the first 50 at 1.59 s/step while compiling, the remaining 450 at 0.34 s/step, about 3,000 spectra/s, GPU at 98-100 % throughout with eight `file_system` workers. Validation (250 batches) 80 s; in-loop embedding evaluation 34 s; `model_best.ckpt`, `model_latest.ckpt` (152 MB each) and `accelerator_state/latest` (optimizer 304 MB) written; every training and validation metric in the run's SQLite MLflow store. Validation at step 500: loss 2.32, median 10,927 ppm, bin accuracy 3.9 %, anisotropy ratio 24.4, effective rank 6.3. So the 90,000-step run is about 8.5 h of training plus about 40 min of validation, evaluation and checkpoints on one H100, inside the 12-hour partition bucket; one GPU also keeps `torch.compile` and avoids the multi-GPU caveats in the code. |
| Step 4 launched at 01:20 EDT on 26/09: job 22701776 (`train-40M-mcfm-90k`), one H100, `--time=14:00:00` (the 24-hour bucket), 128 GB, staging on; overrides `training_steps=90000 num_workers=8 +mp_sharing_strategy=file_system keep_model_every_interval=True embedding_evaluation.tasks_to_run=[embeddingstatisticstask]`, everything else `foundational.yaml`, so batch 1,024, LR 1e-4 with 5 % warm-up, 30 % hold and cosine to 1e-5, clip 1.0, fp16, gradient checkpointing, checkpoint and validation every 10,000 steps, post-training evaluation battery at the end | Expected from the smoke job: about 15 min to the first step, 8.5 h of steps, about 40 min of validation, evaluation and checkpoints, about 1 h of post-training evaluation; first validation and checkpoint at step 10,000 about 1 h 15 min after the first step. Every 10,000-step checkpoint is kept (`model_epoch_*_step_*.ckpt`) so the retrieval-versus-training trajectory can be computed afterwards with the result scripts. Run directory `$RUNS/train-40M-mcfm-90k`; status with `python scripts/train/status.py train-40M-mcfm-90k`. |
| Babysitting, first milestone at 02:29: step 10,000 validation, checkpoint and in-loop statistics all worked; first epoch closed at 1 h 13 min (11,428 steps of 1,024 over 11,703,040 spectra, so 90,000 steps is 7.9 epochs) | Validation at step 10,000 on the 256,000-spectrum protocol: loss 1.945, median 6,679 ppm, bin accuracy 10.7 %, within 20 ppm 1.5 %, intensity R² 0.974; anisotropy ratio 21.7, effective rank 50.3 (the released checkpoint: 4,493 ppm, 27.4 %, 25.0, 89.4). Throughput steady at 0.37-0.39 s/step, GPU 100 %, 47 GB of GPU memory in use; staging took 84 s on g19. Projected end of training about 11:05 EDT, post-training battery after. Checkpoints kept per interval as intended. |

## Training choices

Good, ad hoc, improvable, and problems, in the order the pipeline runs. File references are to
`src/instanovo_fm/` at commit 5953cc4 (plus this branch's changes).

**Data path**

- Training data streams: `common/trainer.py:615` loads shards lazily and `to_dataset(force_unified_schema=True)`
  yields an iterable dataset; validation is `to_dataset(in_memory=True)` (`common/trainer.py:661`), so the whole
  validation split sits in RAM before `valid_subset` or `max_valid_steps` apply. Good for training, a fixed
  memory cost per job for validation (the 790k-row MCFM validation shard needs well over 24 GB; measured on the
  workstation).
- Masking runs in the collate on the CPU (`data/data.py:436-520`), per batch, with Python loops per spectrum for
  span trimming and the isotope hard cap (`data/masking.py:565-583, 604-618`). This is why loader workers matter
  and why the GPU idles when they are absent; a vectorised trim, or masking on the GPU after transfer, would
  remove a data-loading ceiling.
- `dispatch_batches=True` with `split_batches=True` (`common/trainer.py:316-325`): one process loads batches of
  `train_batch_size × num_processes` and broadcasts. Required by the iterable dataset (the docs say rank shards
  can end at different lengths and hang NCCL), but it makes multi-GPU throughput a function of one node's loader
  workers. The comment above it describes the opposite setting from the one used.
- `persistent_workers: True` and `prefetch_factor` are in `foundational.yaml`, but `build_dataloaders`
  (`common/trainer.py:377-397`) never passes `persistent_workers`; the key is dead configuration.
- `perform_data_checks` is off in the paper config because the residue/charge check is "27 min on 12.5 M rows
  via Python lambdas" (`foundational.yaml`); the local config leaves it on and it drops rows with charge out of
  range, so the two configs train on slightly different data.
- The masking function's own defaults (`masking.py:464-486`: spans 4-7, kappa 4, gamma 0.7, cap 0.40) are not the
  config's (spans 3-4, kappa 8, gamma 1.2, cap 1.0); the config wins, but a reader of the function is misled.

**Objective**

- Thompson-span masking with isotope co-masking (`masking.py:464`): anchors sampled by a Beta(0.5, 0.5) prior
  tempered by intensity, spans of 3-4 peaks in m/z order, isotope neighbours added, 26 % of peaks masked on
  average (Table S3). Well motivated (Table S3 shows uniform masking leaks 47 % of masked fragments through a
  sibling), and the paper's ablation chose it.
- The masked peak keeps its intensity and gets its m/z blurred by a Gaussian of 10 Da (`encoder.py:500-511`,
  `blur_sigma_da: 10.0`, `mask_intensity: false`), plus a learned mask bias. So the model never predicts a
  missing peak; it refines a coarse location. Since a group of the classification head is 50 bins × 0.2 Da =
  10 Da, the blur hands the group to the model up to one standard deviation, and the offset head does the
  real work. The paper's 70.1 % "correct bin" is therefore a 0.2 Da resolution task from a 10 Da hint, which
  is worth stating whenever that number is quoted.
- The head is group (245 classes) plus offset (50 classes), each a hard cross-entropy at weight 0.5
  (`losses.py:93-172`); the offset head is conditioned on the true group at training time and the predicted
  (or top-3 joint) group at evaluation (`offset_conditioning: teacher_force`, `eval_topk_groups: 3`,
  `train.py:552-560`, `train.py:660-700`). A soft ordinal target for the offset (`sord_sigma`) is implemented
  and off. Group and offset being independent softmaxes means the loss does not know that bin 49 of group g
  and bin 0 of group g+1 are 0.2 Da apart; the top-3 decoding at evaluation papers over that seam.
- An intensity head is trained alongside (`auxiliary.enabled: True`, `lambda_intensity: 0.2`, Huber with
  delta 0.05, cap 0.7 chosen from data percentiles quoted in the config), the other auxiliary heads are
  weighted 0.0 but their code paths stay in the forward (`train.py:596-620`). Table S8's "λ intensity = 0.2"
  matches the config.
- `losses.py:120`: masked positions with m/z below `min_mz` are silently dropped from the loss because "some
  rare LCFM spectra have padding/zero-mass peaks that can end up in the mask". A data-quality patch in the
  loss function.

**Model and optimisation**

- Gradient checkpointing is on by default (`gradient_checkpointing: True`; applied per encoder chunk in
  `encoder_layers/unified_encoder.py:227` with `use_reentrant=False`). Justified at 1,024 spectra × 200 peaks
  per GPU; at 256 per GPU on four H100s it costs about 30 % compute for memory that is not needed. It is a
  fixed setting, not a function of the per-device batch.
- Mixed precision is hard-coded to fp16 when CUDA is present (`common/trainer.py:315`); the `fp16: True` key
  in the config is decorative and bf16, the natural choice on H100, is not selectable.
- `torch.compile` is applied only on a single GPU without the pairwise bias (`train.py:236-257`); the code
  documents why (backward partitioner crashes with checkpointing, NCCL hangs without). A multi-GPU run gives
  up the compile speed-up.
- DDP is built with `find_unused_parameters=True` (`common/trainer.py:312`) "when using frozen parameters";
  nothing is frozen here, so every step pays an unused-parameter search.
- AdamW with `weight_decay: 0.0` and `fused=True` (`train.py:265-273`): Adam in effect, fused (good). The
  schedule is a cosine with warm-up and hold expressed as fractions of the total steps
  (`common/trainer.py:420-448`), so `training_steps` changes the schedule shape with it; a 90,000-step run
  and a 230,000-step run share fractions, not step counts.
- In the foundational `train_epoch` (`train.py:1121-1140`) the LR scheduler is stepped on every micro-step
  inside `accelerator.accumulate`, not only on optimizer steps; harmless at `grad_accumulation: 1`, wrong
  otherwise. The metric-logging step differs by one between the base loop (`global_step + 1`) and the
  override (`global_step`).
- The checkpoint criterion is the median absolute ppm error over masked peaks of 250 validation batches
  (`metrics.py:641`, `foundational.yaml`), with `<=` so a tie moves the best checkpoint forward; the median
  is taken over an accumulator that keeps every per-peak error in memory.
- That criterion is measured over every masked peak, and about two thirds of the peaks in an HCD spectrum are
  unannotated (the paper's own figure). For the released 40M model it is 4,493 ppm, about 3 Da at m/z 700,
  against a 0.2 Da bin: the median sits in the unpredictable-peak regime, so the checkpoint choice is
  barely sensitive to how well the fragment ions are reconstructed. A criterion restricted to annotated
  peaks (the validation processor can carry sequences) would select on the quantity the downstream
  evaluations reward.
- `main()` reads `model_save_folder_path` from the model config (`train.py:1697`) where it does not live, so
  `mlflow_run_id.txt` always lands in `./checkpoints` relative to the working directory.

**Evaluation during training**

- Every 10,000 steps: validation over 256k spectra (`max_valid_steps: 250`), then the embedding evaluation
  with only `embeddingstatisticstask` and `duplicateretrievaltask` on the validation loader
  (`foundational.yaml`, `train.py:1293-1461`), the full battery after training. Good design: the anisotropy
  and retrieval trajectory over training comes free.
- Validation reseeds the RNGs to a fixed `validation_seed` and restores them afterwards
  (`train.py:836-845, 1050-1054`), so the masked positions are the same at every validation. Good.

## Problems encountered

- With MLflow on, `setup_tracking` resolves the whole config to YAML (`common/trainer.py:464`) and the `tags`
  block of `foundation_base.yaml` interpolated `${architecture...}`, `${dim_model}` and eleven more relative to
  the model file, not the composed root: `InterpolationKeyError`. The paper config has `mlflow_enabled: False`,
  so the authors never hit it; `setup_model` (`train.py:203`) even avoids resolving for the same reason. Fixed by
  writing the thirteen interpolations as `${model....}` (commit bf802db), checked by resolving the composed
  config. Job 22699796 died 14 min in on it.
- The in-loop `duplicateretrievaltask` fails at every interval: the training-time validation batches carry no
  peptide field (the metadata keys are the search columns and masks; no `peptides`, `peptide` or `sequence`),
  while the standalone evaluation path builds `peptides` itself. The paper's in-loop configuration
  (`embedding_evaluation.tasks_to_run: [embeddingstatisticstask, duplicateretrievaltask]`) therefore never
  produced a retrieval curve; the run keeps only the statistics task in the loop and every checkpoint on disk.
- `max_checkpoints: 3` in `foundational.yaml` (and Table S8's "top-3 retained") is read nowhere in the
  package; without `keep_model_every_interval` the trainer keeps `model_latest` and `model_best` only.
- Those 14 minutes were the validation split loading into RAM (`to_dataset(in_memory=True)`): a fixed
  per-job cost before the first step, larger than the paper's whole validation pass (250 batches).

## Next steps

- Babysit job 22701776 to its first checkpoint (step 10,000), then every 45 minutes through the status script.
- Step 5 when it ends: `result1..3_*` with a `40M-ours` tag on `model_best.ckpt`, plus the trainer validation
  metrics against row 23 (median 4,493 ppm, bin accuracy 27.4 %); the retrieval trajectory over the nine
  kept checkpoints.
