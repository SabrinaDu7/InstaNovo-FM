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
| Step 4 done: training completed at 10:58:35 EDT (job 22701776), 9 h 33 min wall from the first step for 90,000 steps at batch 1,024 (0.38 s/step), nine interval checkpoints plus `model_best.ckpt` at step 90,000; metrics and trajectories in `docs/references/rerun/train_40M_mcfm_90k/metrics.json` | Final validation (256,000 MCFM validation spectra, 9.7 M masked peaks): loss 1.520, median 3,897 ppm, MAE 5.25 Da, bin accuracy 27.2 % (group 62.3 %, offset 37.2 %), within 1 Da 40.7 %, within 20 ppm 4.2 %, intensity R² 0.990; embedding anisotropy ratio 22.5, effective rank 95.3. The released checkpoint on the same protocol: 4,493 ppm, 5.99 Da, 27.4 %, 39.8 %, 4.1 %, 0.980; 25.0, 89.4 (row 23). Median ppm improved monotonically at every checkpoint (6,679 → 5,653 → 5,048 → 4,600 → 4,375 → 4,175 → 4,055 → 3,952 → 3,897), bin accuracy 10.7 → 27.2 %, effective rank 50 → 95 while the anisotropy ratio stayed at 21-23 from the first checkpoint on. GPU idle: 98 of 1,139 thirty-second samples in the training window, about half of them the nine validation passes, the rest one-minute dips at shard boundaries, about 4-5 % of training time. Cost: one H100 for 9 h 55 min including set-up, about 10 GPU-hours. |
| The post-training evaluation battery was OOM-killed (system RAM, MaxRSS 128 GB of 128 GB, `slurmstepd: Detected 1 oom_kill event`) while generating 100,000 model-train embeddings for the linear probe with the peak-type and IG tasks in the same process, which store per-peak embeddings for every sample | The checkpoints were on disk before it started, so nothing was lost. The battery runs as separate jobs through `scripts/reproduce/result*_40Mours_*.py` instead (Step 5), the way the released checkpoints were evaluated. |
| Step 5: the trained checkpoint through the same result scripts as the released one (jobs 22721434-38; `docs/references/rerun/result*_40Mours_*/`; section "40M trained here" of `results_paper_or_rerun.md`, every row beside the released checkpoint's own rerun) | Defined before reading: reproduced if retrieval and peak-level rows sit within sampling noise of the released rerun, probes carrying the backend caveat. LCFM test, probes and retrieval (job 22721437, 1 h 27 min): duplicate retrieval recall@1 0.314 vs 0.305 and mAP@20 0.130 vs 0.125, so our model retrieves marginally better; probes fragment type 0.706 vs 0.733, instrument 0.721 vs 0.729, PTM 0.733 vs 0.756, hydrophobicity 0.516 vs 0.528, mass 0.693 vs 0.702, m/z 0.894 vs 0.897, charge 0.494 vs 0.515, confidence 0.977 vs 0.978, within 0.03 everywhere and mostly a point under. MCFM test, the training tier's own held-out split (job 22721438, 1 h 50 min): retrieval recall@1 0.638 vs 0.629 and mAP@20 0.339 vs 0.329, again marginally above; probes fragment type 0.746 vs 0.776, instrument 0.669 vs 0.680, PTM 0.779 vs 0.799, hydrophobicity 0.604 vs 0.622, mass 0.741 vs 0.755, m/z 0.919 vs 0.928, charge 0.599 vs 0.608, confidence 0.977 vs 0.981, one to three points under throughout (rows 16-17). Peak level (job 22721434, 17 min): peak-type accuracy 72.6 % vs 73.6 % and macro-F1 0.510 vs 0.519, cross-spectrum AUROC 0.869 vs 0.885, confidence AUROC 0.726 vs 0.718, but the IG task's fragment-group reconstruction is clearly lower, bin accuracy 49.2 % vs 55.0 % and median errors 203 / 827 ppm vs 137 / 277 on y / b ions, while the trainer's own validation criterion is better (3,897 vs 4,493 ppm, row 23). Geometry (job 22721435, 30 min): anisotropy 24.9 vs 25.0, effective rank 88.6 vs 89.4, ESM2 RSA 0.047 vs 0.048, Glass Box identical, UMAP kNN 0.086 vs 0.098, EVoC 20 clusters at 0.72 purity vs 17 at 0.81; cosine-hyperscore Spearman 0.071 vs 0.148 (both weak). Verdict: the representation is reproduced (retrieval, probes, geometry within a few percent of the released checkpoint, retrieval slightly above it); what is not reproduced is fragment-ion reconstruction accuracy, which the trainer's checkpoint criterion does not measure. Whether the released checkpoint was selected on something else, or trained with a different seed or data order, the paper does not say. |

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

- Data integrity: the trainer validates nothing about the shards beyond the unified-schema cast (`to_dataset(
  force_unified_schema=True)`); `perform_data_checks` (off, above) is the only content check and it is row-wise
  Python. The sha256 check of every shard against the Hugging Face manifest is ours (`scripts/reproduce/verify_data.py`),
  run once after the download, and the shard shuffle is `shuffle=True` with no seed tied to the run
  (`common/trainer.py:615-625`), so the sample order of a run is not reconstructible from its config.

**Architecture**

- Peak token: `MultiScalePeakEmbedding` (`model/embeddings.py:15-54`). The m/z, normalised by `max_mz` 2,500 to
  [0, 1], meets 384 sinusoid frequencies (sin and cos, 768 values), then a two-layer MLP; the intensity is
  concatenated as one scalar and a second two-layer MLP makes the 768-d token. The frequencies are an
  `nn.Parameter` with initial periods of 2.5 to 25 Da (`logspace(-3, -2)` in normalised units). Confirmed on both
  checkpoints: after 90,000 steps they sit at most 0.01 % from their initialisation (largest absolute move 0.10 on
  values of 628 to 6,283), so the "learnable" frequency bank is the initialisation; Adam moves a parameter by about
  the learning rate per step whatever its scale, and 90,000 × 1e-4 is 9 against magnitudes of hundreds (inferred;
  a per-parameter-group learning rate would test it). The finest period, 2.5 Da, is above the isotope spacing and
  twelve times the 0.2 Da bin, so sub-Dalton resolution is the MLP's to build from phase; the `fourier` alternative
  has the same floor (`x_min: 0.001`). Improvable: frequencies below 1 Da in the bank, or their own learning rate.
- No positional encoding and no relative bias (`foundation_base.yaml`, `architecture.positional_encoding.type:
  none`, `relative_bias.type: none`): the encoder is a set transformer over peak tokens; m/z order and Δm/z
  between peaks reach attention only through the token values. The pairwise-attention bias (`pa`, Fourier features
  of Δm/z projected to per-head biases, `model/pairwise_bias.py`) is the ablation the `*-pa` LCFM checkpoints
  carry; the released 40M and 89M models do without it.
- Layers: `UnifiedEncoderLayer` subclasses `nn.TransformerEncoderLayer` with its defaults
  (`encoder_layers/unified_encoder.py:41-43`): post-norm, ReLU, dropout 0.1, plus one final LayerNorm copied from
  `norm1` (`unified_encoder.py:305`). Nine layers, 12 heads, d 768 and a feed-forward width of 1,024, one third
  over d rather than the customary four times (the 89M model's is 3,072). Attention is `FlashMHA` over PyTorch
  SDPA (`encoder_layers/factories.py:120-122`); the layer builds it as `custom_attention` and then assigns it to
  `self_attn` after the parent has already registered its own `nn.MultiheadAttention`
  (`unified_encoder.py:72-80`), which is where the two copies of every attention weight in the state dict come
  from (below).
- Latent token (`model/encoder.py:196`, prepended at `encoder.py:622-623`, read back at `encoder.py:910-911`): its
  output feeds only the charge and retention-time auxiliary heads (`model/heads.py:535-548`), both at weight 0.0
  in the paper config, so no loss trains it as a summary vector; the embedding evaluations mean-pool the peak
  tokens instead (`evaluation.embedding_pooling=[mean_pool]` in every reproduction protocol). A CLS token that
  nothing reads: ad hoc.
- No precursor information: `meta_token.enabled: False` in `foundation_base.yaml` (the code's default is True,
  `encoder.py:1429`), so precursor m/z, charge, instrument and collision energy never enter the encoder, and the
  `precursors` argument of `forward` is marked deprecated. Every probe result (charge, instrument, mass) therefore
  measures what the peaks alone carry, which is the interesting reading of those rows; the configuration that
  would add the metadata tokens exists and is untested in the paper.
- Options carried in the config and off in every released model: ion-ladder encoder, noise injection, RBF,
  Fourier and linear peak encoders, regression m/z head with heteroscedastic sigma, sinusoidal and rotary
  positional encodings, ALiBi and RPE biases, meta token, `signal_aware_fragment` masking. The model config
  is a research surface with one point used; reading it does not tell a newcomer which paths are live.

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
- Mixed precision is hard-coded to fp16 when CUDA is present (`common/trainer.py:319`); the `fp16: True` key
  in the config is decorative and bf16, the natural choice on H100, is not selectable. Loss scaling is
  Accelerate's dynamic `GradScaler` (its state is `scaler.pt` in the accelerator state): at the end of our run
  the scale was 2,097,152 (2^21) with growth interval 2,000 and growth tracker 813, so the last overflow, which
  halves the scale and skips the optimizer step, happened about 10,800 steps before the end (inferred from the
  scaler arithmetic; init 65,536 would have reached 2^61 without overflows). Overflow-skipped steps are logged
  nowhere; `train.log` has no NaN or inf line. Logging `scaler.get_scale()` with the metrics would make the
  overflow history visible. The Fourier peak encoder forces fp32 for its sinusoids "to prevent fp16 NaN"
  (`model/embeddings.py:95-100`) while the multiscale encoder's equivalent guard is commented out
  (`embeddings.py:44`).
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
- Every checkpoint carries two copies of each layer's attention weights, `self_attn.*` and `custom_attention.*`
  (158 tensors, 61.1 M values for a 39.9 M-parameter model): both names point at the same `FlashMHA` module
  (`unified_encoder.py:72-80`), so `state_dict()` writes its tensors twice. Half the file, and a source of
  confusion when counting parameters from a state dict.
- Evidence for the checkpoint-criterion point above, from this run: our model beats the released one on the
  trainer's median ppm over all masked peaks (3,897 vs 4,493) and loses to it on the IG task's fragment-group
  bin accuracy (49.2 % vs 55.0 %) and median errors. The criterion the trainer optimises for checkpoint choice
  and the reconstruction the paper reports move in opposite directions between these two models.
- `main()` reads `model_save_folder_path` from the model config (`train.py:1697`) where it does not live, so
  `mlflow_run_id.txt` always lands in `./checkpoints` relative to the working directory.

**Checkpoints, resume and tracking**

- A `.ckpt` is `state_dict`, the model config as a plain dict, the residue masses, epoch and step
  (`trainer/train.py:469-478`): self-describing, loadable without the Hydra tree. Good. With
  `keep_model_every_interval` the file is named `step_{global_step + 1}` (`train.py:459`), so the files read
  `step_10001` … `step_90001` and the in-loop evaluation directories and MLflow eval points carry the same +1
  while the training metrics are logged at the round step.
- The accelerator state (`common/trainer.py:796-845`) is `model.safetensors`, `optimizer.bin`, `scheduler.bin`,
  `scaler.pt`, the RNG states and a `TrainingState` of epoch and step (`common/utils.py:14`), 457 MB, written to
  `accelerator_state/latest` at every checkpoint interval by `rmtree` then `save_state` (`trainer.py:808-814`):
  not atomic, a crash inside the save leaves no resumable state. `resume_accelerator_state` restores all of it
  (`trainer.py:186-188, 909`), but nothing restores the data position: no `skip_first_batches` anywhere in the
  trainer and no run-dependent shuffle seed, so a resumed run starts the stream over from the first shard while
  the step counter continues (inferred from the absence; the grep is the evidence). For a 90,000-step run over
  11.7 M spectra that is 7.9 epochs, so a resume repeats data rather than losing it.
- Tracking: MLflow to a SQLite file (`mlflow_tracking_uri=sqlite:///…/mlflow.db`, 54 metric keys, 12,470 rows
  for this run) plus TensorBoard (`tb_summarywriter`), and Hydra's own `outputs/<date>/<time>/.hydra/
  {config,overrides}.yaml` with the resolved config. Good: the run is reconstructible from its directory. Less
  good: `mlflow_log_checkpoints` defaults to True and `_log_checkpoint_artifact` (`trainer.py:790-794`, called
  at 825 and 845) copies every accelerator state into `mlruns/…/artifacts/checkpoints`, 913 MB of duplicates next
  to `checkpoints/`; and the SQLite backend exhausted its connection pool once during the run (`train.log`
  lines 1187-1188, "QueuePool limit of size 5 overflow 10 reached", the async logger reporting one failed
  batch of run data at 06:44). Checked: the 1,000-step training series and the nine validation points have no
  gap, so what was lost, if anything, was a system-metrics sample.

**Hyperparameters and sweeps**

- Every optimisation hyperparameter sits in `foundational.yaml:19-31`: seed 101, learning rate 1e-4, weight decay
  0, batch 1,024, gradient accumulation 1, clip 1.0, cosine schedule with 5 % warm-up, 30 % hold and a 0.1 floor.
  The global batch of 1,024 is the paper's on four GPUs (256 per device); one GPU here holds all 1,024, so the
  optimisation is identical and only the gradient-checkpointing trade-off differs (above). `training_steps:
  30_000` is the shipped default, which is neither released model's (90,000 and 230,000): a run that omits the
  override reproduces nothing.
- Data-derived constants are justified in comments, `intensity_head.max_intensity: 0.7` from an observed
  maximum of 0.603 and p99 of 0.197, `huber_delta: 0.05` from a standard deviation of 0.034 and kurtosis of 11.3
  (`foundation_base.yaml`). Good that the numbers are there; ad hoc that they live only in comments, with no
  script in the repository that recomputes them from the shards.
- There is no sweep infrastructure: no Hydra sweeper, Optuna or Weights & Biases configuration anywhere under
  `configs/` or in `pyproject.toml`, and one seed. The paper's ablations (Tables S3 and S4) survive as the three
  released LCFM checkpoints (`*-ts-pa`, `*-sa-nopa`, `*-sa-pa`, all in `$CHECKPOINTS`) and as "ablation optimum"
  comments on four masking keys in `foundation_base.yaml` (spans 3-4, isotopes on, no hard cap, intensity visible).
  Nothing in the repository re-runs the sweep or states its seeds and budgets; the provenance of the chosen
  hyperparameters is a comment.

**Evaluation during training**

- Every 10,000 steps: validation over 256k spectra (`max_valid_steps: 250`), then the embedding evaluation
  with only `embeddingstatisticstask` and `duplicateretrievaltask` on the validation loader
  (`foundational.yaml`, `train.py:1293-1461`), the full battery after training. Good design: the anisotropy
  and retrieval trajectory over training comes free.
- Validation reseeds the RNGs to a fixed `validation_seed` and restores them afterwards
  (`train.py:836-845, 1050-1054`), so the masked positions are the same at every validation. Good.

**Evaluation harness**

- The harness is the package's own (`eval/embed_eval_tasks/`); the paper's
  protocols are pinned in `scripts/reproduce/_common.py` and its bugs fixed here are listed in
  `session-2026-09-25-reproduceResults1.md`. Component-level observations:
- Duplicate retrieval's positive is an identical peptide string after `str().strip()`
  (`duplicate_retrieval.py:199-208`): charge is ignored, so the same peptide at 2+ and 3+ is a duplicate pair, and
  nothing normalises modification notation or I/L. The definition has to travel with every recall number.
- The linear probe uses cuML when it imports and scikit-learn otherwise, silently (`linear_probe.py:30-39`); the
  backend moves the macro-F1 rows by up to 5 points on the same embeddings (released 40M fragment type 0.733
  here against 0.781 in the paper), so a probe number without its backend is not comparable.
- In-loop and post-training evaluation share one code path with the standalone one but not one data path: the
  training validation batches carry no peptide strings (retrieval cannot run in the loop, Problems below) and the
  post-training battery runs every task in one process while the peak-type and IG tasks keep per-peak embeddings
  for every sample (OOM at 128 GB). The standalone result scripts, one task family per job, are the working
  configuration.
- The checkpoint criterion and the harness disagree about what "better" means (median ppm over all masked peaks
  against fragment-group accuracy over annotated peaks, above); the harness is the one the paper reports, so the
  trainer selects on a quantity nobody publishes.
- `faiss-cpu` loads without AVX2 on nibi, so the 200,000-spectrum retrieval pools take 36 to 47 min; a wheel built
  for the node or the GPU index would make retrieval the cheapest task rather than the longest.
- The methodological review of the tasks themselves (population choice, the m/z-distance confound in retrieval,
  anisotropy) is the workstation's `instanovo-fm-evals` repository, `docs/claude_logs/review-2026-09-25.md`, not
  this fork.

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
- The default `post_training_evaluation` runs eight tasks in one process on the training node's RAM budget; with
  the probe's 100,000 training embeddings and the two per-peak tasks it needs far more than 128 GB (job
  22701776 died there after training had finished). Splitting the battery by task family, as the result
  scripts do, or requesting the node's full 2 TB, are the two ways out.
- `max_checkpoints: 3` in `foundational.yaml` (and Table S8's "top-3 retained") is read nowhere in the
  package; without `keep_model_every_interval` the trainer keeps `model_latest` and `model_best` only.
- Those 14 minutes were the validation split loading into RAM (`to_dataset(in_memory=True)`): a fixed
  per-job cost before the first step, larger than the paper's whole validation pass (250 batches).

- MLflow's SQLite backend hit its connection-pool limit once (06:44, `train.log` lines 1187-1188): a warning from
  the system-metrics monitor and one failed asynchronous log batch. No metric series shows a gap at 1,000-step
  resolution, so the loss was a system-metrics sample; a run that logs more often than every 1,000 steps into
  SQLite would hit it harder, and the fix is a file-based `mlruns` store or a larger pool.

## Methodology, timings and cost

- One H100 (nibi `gpubase_bygpu_b3`, account rrg-hsn), 12 CPUs, 128 GB RAM, MCFM shards staged to node-local
  disk (53 GB in 84-154 s). Command: `scripts/train/submit.sh train-40M-mcfm-90k training_steps=90000 num_workers=8
  +mp_sharing_strategy=file_system keep_model_every_interval=True embedding_evaluation.tasks_to_run=[embeddingstatisticstask]`
  on `foundational.yaml` with `dataset=mcfm` (batch 1,024, LR 1e-4, 5 % warm-up, 30 % hold, cosine to 1e-5, clip 1.0,
  fp16, gradient checkpointing, `torch.compile`, validation and checkpoint every 10,000 steps).
- Time: 15 min from job start to the first step (staging 1.5 min, validation split into RAM 12 min, compile 25 s),
  9 h 33 min for 90,000 steps at 0.38 s/step (about 2,700 spectra/s), nine validation passes of about 80 s plus
  the statistics task, checkpoint writes of a few seconds. Total job 9 h 54 min; about 10 GPU-hours. GPU utilisation
  98-100 % between the one-minute shard-boundary dips (about 4-5 % of training time).
- Evaluation of the trained checkpoint: five jobs, 4 h 7 min of H100 time in all (probes and retrieval on LCFM
  test 1 h 27 min, on MCFM test 1 h 50 min, geometry 30 min, peak level 17 min, cosine-hyperscore 3 min;
  `sacct -j 22721434,22721435,22721436,22721437,22721438 -X -o JobName,Elapsed`).

## Where things are

- Run directory `$RUNS/train-40M-mcfm-90k`: `train.log`, `gpu.log`, `checkpoints/` (nine `model_epoch_*_step_*.ckpt`,
  `model_best.ckpt`, `accelerator_state/latest`), `evaluation/step_*/` (in-loop statistics), `mlflow.db`,
  `mlruns/` (913 MB of checkpoint copies logged as MLflow artifacts) and `outputs/2026-09-26/01-13-54/.hydra/`
  (the resolved config and the overrides).
- The checkpoint as a release-style file: `$CHECKPOINTS/instanovo-fm-mcfm-90k-ours-2026-09-26.ckpt`, model tag
  `40M-ours` in `scripts/reproduce/_common.py`.
- MLflow: `mlflow ui --backend-store-uri sqlite:///$RUNS/train-40M-mcfm-90k/mlflow.db --port 5000` on the nibi login
  node, then `ssh -L 5000:localhost:5000 nibi` and open http://localhost:5000; experiment `instanovo-fm-reproduction`,
  run `instanovo_foundational_26_09_26_01_13` (left in state RUNNING by the OOM kill after training). The same
  numbers without a UI: `python scripts/train/status.py train-40M-mcfm-90k`, and the final metrics and per-checkpoint
  trajectories in `docs/references/rerun/train_40M_mcfm_90k/metrics.json`.

## Next steps

- The retrieval-versus-training trajectory: `result1_*` on each of the nine kept checkpoints (about 1.5 h of H100
  each), which the in-loop retrieval never produced for the authors either.
- A checkpoint criterion restricted to annotated fragment peaks, and a run selected on it, to test whether the
  fragment-reconstruction gap to the released checkpoint is selection or training.
- cuML in the venv for the probe rows; the ablation checkpoints against Table S4.
- The three dead or misleading configuration keys (`persistent_workers`, `max_checkpoints`, `fp16`) and the
  duplicated attention weights, as a small upstream pull request with the tag-interpolation and metadata fixes.
