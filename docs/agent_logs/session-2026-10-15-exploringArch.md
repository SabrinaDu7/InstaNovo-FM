# Running notes: the architecture of InstaNovo-FM, and where it could improve
Started 15/10/2026 on branch `sdu/exploring-arch` (from `main` at 5508e71). Running file: each session appends to
the log at the end and revises the sections above it. Status: **reading and inventory; no experiment started.**

## Purpose

Understand the architectural choices behind the released InstaNovo-FM checkpoints (the 40M MCFM baseline and the
89M LCFM model) well enough to say which are principled, which are ad hoc, and which could be improved, then test
the most promising changes against the paper's own evaluations. Every claim below is marked **confirmed** (read in
the code or measured, with the file and the run) or **inferred** (a reading that an experiment would settle). File
references are to `src/instanovo_fm/` at 5508e71 unless another root is given.

Sources read for this inventory: `docs/foundation_model.md`, `docs/foundation_model_training.md`,
`configs/model/foundation_base.yaml`, `configs/foundational.yaml`, `model/encoder.py`, `model/embeddings.py`,
`model/heads.py`, `model/meta_token.py`, `model/ion_ladder.py`, `model/pairwise_bias.py`, `model/pad_token_mixin.py`,
`model/attention/flash.py`, `model/encoder_layers/{unified_encoder,factories}.py`, `data/data.py`,
`data/masking.py`, `data/metadata_builder.py`, `trainer/{train,losses,binning,utils,metrics,confidence_weighting}.py`,
`common/{trainer,scheduler}.py`; the preprint's architecture, objective, recipe and ablation passages (bioRxiv
10.64898/2026.09.03.747733v2, fetched 15/10); and the earlier logs in this directory, above all the "Training
choices" section of `session-2026-09-25-train40M.md`, which already lists what looked ad hoc while reproducing the
40M checkpoint. Those earlier findings are folded in below rather than repeated; where a point is theirs, the log
is named.

## 1. The model as built

### 1.1 Input: what a spectrum becomes

`FoundationalDataProcessor._process_spectrum` (`data/data.py:277-315`, the torch path; `use_spectrum_utils: False`):
keep peaks in 50 to 2,500 Da, drop peaks under 1 % of the base peak (`min_intensity: 0.01` on base-peak-normalised
intensities), keep the 200 most intense, take the square root of the intensities and L2-normalise them per
spectrum, divide m/z by 2,500. The precursor peak is kept (`remove_precursor_tol: 0.0`). Peaks are sorted by m/z in
the collate (`apply_peak_ordering`, `ordering_strategy: sorted`), zero-padded to 200, and the batch carries
`spectra (B, 200, 2)`, `spectra_mask`, the precursor triple and `peak_mask` (`data.py:713-798`). Nothing else
enters the model: no precursor m/z, charge, instrument or collision energy (`meta_token.enabled: False`; the
`precursors` argument of `forward` is marked deprecated, `encoder.py:834`). **Confirmed.**

Measured on 2,000 MCFM test spectra with that processor (`scripts/analysis/masking_stats.py`, seed 0): 148 peaks
per spectrum on average, median 161, and 37.8 % of spectra sit at the 200 cap, so for a third of the corpus the
input is a top-200 selection; the intensity channel has median 0.053 and per-spectrum maximum 0.30 (median), with
an absolute maximum of 0.91. **Confirmed.**

### 1.2 Peak token

`MultiScalePeakEmbedding` (`model/embeddings.py:15-58`): the normalised m/z multiplies 384 learnable frequencies
initialised at periods of 2.5 to 25 Da (`2π / logspace(-3, -2)` in normalised units, `embeddings.py:42`); sin and
cos give 768 features; a two-layer ReLU MLP with dropout 0.1; the intensity is concatenated as one scalar and a
second two-layer MLP produces the 768-d token. The frequencies did not move during training: on both the released
40M and our retrain they sit within 0.01 % of their initialisation (`session-2026-09-25-train40M.md`, "Architecture").
Alternatives exist in the code and are off: the DreaMS-style `FourierPeakEmbedding` with periods from 2.5 to 3,750
Da, an `RBFPeakEmbedding` of 1,024 Gaussians, a `DualPeakEmbedding` gating both, and a `LinearPeakEmbedding`
control. The preprint reports no peak-encoder ablation (fetched text, section 6). **Confirmed.**

### 1.3 Tokens around the peaks

- A **latent token** (`[CLS]`, `encoder.py:196`, `randn` init) is prepended. Its output feeds only the charge and
  retention-time heads (`heads.py:535-548`), both at weight 0 in the paper configuration, so no loss ever reaches
  it; the published embedding is the mean of the peak tokens excluding it (`encode_mean_pooled`,
  `encoder.py:1260-1294`; preprint: "the mean of the final-layer hidden states over the non-padding peak tokens,
  excluding the [CLS] token"). It still takes part in every attention layer as a key and value. **Confirmed.**
- A **mask token** (`randn * 0.02`) is an additive bias on masked positions, not a replacement (below).
- A **pad token** exists for the flash backend but is frozen and documented as inert, because padded keys are
  masked to −inf in every softmax (`pad_token_mixin.py:46-57`). **Confirmed.**
- **Meta tokens** (one per metadata field: fragmentation, instrument, charge bin, precursor mass, collision energy,
  ...; `model/meta_token.py`) are implemented with hard-coded vocabularies (`data/metadata_builder.py:15-46`: 20
  instrument names, 7 detectors, 7 enzymes) and switched off. **Confirmed.**

### 1.4 Encoder

`UnifiedEncoderLayer` subclasses `nn.TransformerEncoderLayer` with its defaults (`unified_encoder.py:43`): post-norm,
ReLU, dropout 0.1 on attention, residuals and the feed-forward; attention is `FlashMHA` over PyTorch SDPA with 12
heads of 64 (`attention/flash.py`), `qkv` without bias, attention dropout 0.1 in training; a final LayerNorm copied
from `norm1` closes the stack (`unified_encoder.py:305`). No positional encoding and no relative bias
(`architecture.positional_encoding.type: none`, `relative_bias.type: none`): the encoder is a set transformer over
peak tokens. Depth and width: 9 layers with feed-forward 1,024 for the 40M (a ratio of 1.33 to the model width),
12 layers with 3,072 for the 89M. Gradient checkpointing wraps every layer in training (`unified_encoder.py:359`).
The layer registers its attention module twice (`self_attn` and `custom_attention`, `unified_encoder.py:72-78`),
which is why every checkpoint holds two copies of the attention weights (`session-2026-09-25-train40M.md`).
**Confirmed.**

The pairwise attention bias (`model/pairwise_bias.py`, Lapin et al.) Fourier-encodes signed Δm/z over 64
log-spaced wavelengths from 0.001 to 10,000 Da, projects through a small MLP, and a per-layer linear map turns the
features into one additive bias per head. The released 40M and 89M do without it; the `*-pa` ablation checkpoints
carry it. In the default mode the bias is **zeroed for every pair that involves a masked or padded position**
(`encoder.py:714-723`, mode 3); the two alternatives that keep Δm/z information at masked positions
(`blur_masked_pa`, `pa_mask_to_mask`) are implemented and off in every released configuration
(`foundation_base.yaml:228-229`). **Confirmed.**

### 1.5 The objective

**Which peaks are hidden.** `thompson_sampling_span_mask` (`data/masking.py:464-655`): anchors are drawn per spectrum
by Thompson sampling, a Beta(0.5 + 8·I^1.2, 0.5 + 8·(1 − I^1.2)) score per peak on the sqrt-L2 intensity I; the
anchor budget is the target fraction divided by the mean span length, times 1.5 (`anchor_portion ≈ 0.111` for the
paper settings); each anchor opens a span of 3 or 4 peaks **to higher m/z only** (`bidirectional: False`); spans are
then trimmed back to the 26 % target by removing whole contiguous runs, **shortest and faintest first**
(`masking.py:598-607`); isotope co-masking then adds, round by round, the immediate m/z neighbours of a run whose
distance to the adjacent masked peak matches n·1.003355/z within max(20 ppm, 0.015 Da), for z up to the batch's
largest precursor charge and n up to 3, with no cap (`max_total_mask_ratio: 1.0`, `_vectorized_isotope_extension`).
Measured on 2,000 MCFM test spectra (`scripts/analysis/masking_stats.py`): **27.0 % of peaks masked** (spans alone
24.5 %, isotopes 2.5 points, so 9.7 % of masked peaks are co-masked isotopes); 7.1 contiguous runs per spectrum with
**mean length 5.2, not 3 to 4**: because adjacent spans merge into one run and the trimming removes the shortest
runs first, 59 % of span-masked peaks lie in runs longer than `span_max`, with runs up to 22 peaks (histogram in the
script's output); the leftmost peak of a run is its most intense in 38.8 % of runs against about 29 % by chance (the
one-sided expansion from an intensity-weighted anchor); after co-masking only 0.8 % of masked peaks still have a
visible +1 isotope, so the co-masking does what it claims. **Confirmed (measured).**

**What the model sees at a masked position.** Not a `[MASK]` token: the true m/z plus Gaussian noise of σ = 10 Da,
clamped to the range, goes through the same peak encoder **together with the true intensity**
(`mask_intensity: false`), and the learned mask token is added (`encoder.py:500-515`). The noise is drawn from the
global RNG at every forward, in evaluation too. The group of the classification head is 50 bins of 0.2 Da = 10 Da
wide, so the hint equals one group width: the released 40M gets the group right for 62 % of masked peaks and the
exact bin for 27 % on MCFM validation (row 23 of `docs/references/results_paper_or_rerun.md`), so the hint does not
give the group away but narrows it to a handful of candidates. **Confirmed.**

**What is predicted and how it is scored.** `MzClassificationHead` (`heads.py:167-250`): on each masked token a
two-layer MLP predicts one of 245 groups and a second two-layer MLP, fed the token plus a 64-d embedding of the
**true** group in training (`offset_conditioning: teacher_force`), predicts one of 50 offsets; two hard
cross-entropies at weight 0.5 each (`losses.py:146-169`); the fixed 0.2 Da grid over 50 to 2,500 Da gives 12,250
bins (`trainer/binning.py:192-197`; the preprint names the same numbers and a binning study, Figure 2C, choosing
0.2 Da). At evaluation the trainer decodes with the top-3 groups jointly (`train.py:688-746`); the IG task of the
suite decodes greedily. A soft ordinal target for the offset (`sord_sigma`) is implemented and off. Masked peaks
below `min_mz` are dropped from the loss as a data patch (`losses.py:127-131`). **Confirmed.**

**The auxiliary intensity task does not do what it says.** `IntensityRegressionHead` regresses the intensity of
every token with a Huber loss (δ 0.05, cap 0.7) at λ = 0.2 (`losses.py:370-388`). The loss restricts itself to
masked peaks only if the batch has a key named `mlm_mask` (`losses.py:376`), and the same guard sits in the metric
(`metrics.py:430`); but the collate writes the mask under `peak_mask` (`data.py:797`), the trainer reads it under
that name (`train.py:544`), and nothing in the package ever writes `mlm_mask` into a batch (grep; the batch keys of
a collated batch are printed by `scripts/analysis/masking_stats.py`). So the intensity loss runs over **every
non-padded peak**, and the intensity is in the input for all of them: visible peaks carry it directly, masked peaks
keep it because `mask_intensity` is false. One fifth of the loss weight trains the model to copy an input channel,
and the reported intensity R² of 0.98 to 0.99 (`eval/intensity_r2`, row 23) measures that copy. **Confirmed by
reading; the one-line experiment that settles it is a run with `lambda_intensity: 0`.** The other auxiliary heads
(charge, retention time, Δm/z chemical-gap buckets, PTM) exist and are at weight 0; the Δm/z head is the one
self-supervised, chemistry-shaped target in the file and it was never used.

### 1.6 Output embedding

Mean of the final-layer peak tokens over non-padded positions, L2-normalised (`encoder.py:1290-1296`), computed in
`eval()` mode without masking. Our reruns put its anisotropy ratio at about 25 and its effective rank at about 90 of
768 for both released models (`results_paper_or_rerun.md`, rows 15 and 41; `session-2026-09-25-reproduceResults1.md`).
**Confirmed.**

### 1.7 Training recipe

`configs/foundational.yaml` with the trainer: AdamW with weight decay 0 (Adam in effect), fused; learning rate 1e-4
with 5 % linear warm-up, 30 % hold, cosine to 1e-5, all as fractions of `training_steps` (`common/scheduler.py:182`,
`common/trainer.py:422-447`); global batch 1,024, gradient clipping at 1.0; mixed precision hard-coded to fp16
whenever CUDA is present (`common/trainer.py:319`; the `fp16` config key is decorative); `torch.compile` on one GPU
only; 90,000 steps for the 40M (7.9 epochs of MCFM at batch 1,024), about 230,000 for the 89M; validation every
10,000 steps on 250 batches with a fixed seed, checkpoint selection on `median_ae_ppm` over **all** masked peaks of
those batches (`metrics.py:641`, `check_if_best_checkpoint`), no weight averaging. Seed 101, one seed per
configuration, no sweep tooling in the repository. **Confirmed.**

Two facts from our own runs frame every comparison below (`session-2026-09-25-train40M.md`,
`session-2026-10-03-batch2048-evals.md`, `docs/analysis/fragment_reconstruction.md`): the 40M retrained at the
paper's stated budget reproduces the released checkpoint on retrieval, probes and geometry but lands 4.4 ± 0.4
points under it on fragment-ion reconstruction, while beating it on the trainer's own criterion (3,897 against 4,493
ppm); a global batch of 2,048 closes the gap (56.6 % against 55.0 %), and the released 40M's own counters (epoch 14
at step 80,001) say it saw twice the spectra per step that the stated recipe gives. So the recipe in the repository
is not the recipe that produced the released 40M, and the selection criterion does not track the metric the paper
reports.

### 1.8 What the preprint's ablations cover, and what they do not

From the fetched text and `docs/foundation_model.md`: masking strategy (uniform, Thompson-span, signal-aware; the
leakage argument of Figure 2B), the 2 × 2 of masking × pairwise bias on LCFM validation (Table S4; TS·noPA wins
most probes and has 8 of 12 "structural" heads, TS·PA is weakest on 9 of 12 metrics), the binning grid (Figure 2C,
12 strategies, 0.2 Da chosen), and positional encodings (none beat omitting them, per the docs; no table in the
fetched text). Not ablated anywhere we can find: the peak encoder, the blur σ (chosen by argument: an order of
magnitude above the isotope spacing), the visible-intensity choice (a comment in the config says "ablation optimum",
no numbers), the auxiliary intensity loss, the depth-versus-width split, the pooling (CLS against mean), batch size,
training length, dropout, normalisation placement, and the selection criterion. The preprint's own limitations
(fetched section 5): training on the identification-defined LCFM tier inherits the corpus-selection bias; the
objective does not model chimeric DIA spectra; no scaling study.

## 2. Points that look ad hoc, off, or improvable

Ordered within each group by how much a change could matter. "Cost" is H100 time for one 40M training at the
paper's budget on MCFM (about 10 GPU-hours measured, `session-2026-09-25-train40M.md`) plus the suite's paper
protocols on the checkpoint (about 4 GPU-hours, `scripts/reproduce/`).

### A. Objective and targets

| # | Observation | Status | Why it matters | What would settle it |
| :-- | :---- | :---- | :---- | :---- |
| A1 | The intensity auxiliary loss (λ 0.2) scores every non-padded peak, whose intensity is in the input (section 1.5). | confirmed by reading | A fifth of the gradient trains an identity map; the intensity head is read by nothing downstream; the "intensity R²" row is not a reconstruction metric. Removing it, or restricting it to masked peaks with `mask_intensity: true`, changes what the encoder is asked to learn. | One run with `lambda_intensity: 0`, one with the mask key fixed and intensity hidden at masked positions; compare on the nine reproduction protocols. 2 runs. |
| A2 | Realised masks are long merged runs (mean 5.2, 59 % of masked peaks in runs over 4, tail to 22), not spans of 3 to 4, because trimming drops the shortest runs first (section 1.5). | confirmed (measured) | The published description and the leakage analysis (Figure 2B) are about short spans; long runs remove more local context than intended and make the task harder in a way nobody chose. | Trim by splitting merged runs or by capping run length, keep 26 %, retrain once; judge on fragment-group accuracy (IG task), probes, retrieval. 1 run. |
| A3 | The checkpoint criterion is the median ppm error over all masked peaks, two thirds of which are unannotated; it moved opposite to fragment accuracy between the released 40M and our retrain (section 1.7). | confirmed | Selection chooses a quantity nobody publishes; the released checkpoints may have been selected differently. | Re-score the nine kept checkpoints of the batch-1,024 and batch-2,048 runs on an annotated-peak criterion (the validation processor can carry sequences) and on the IG task; no training needed. 0 runs, about 20 GPU-hours of evaluation. |
| A4 | Group and offset are independent softmaxes over a 0.2 Da grid: bin 49 of group g and bin 0 of group g+1 are 0.2 Da apart and unrelated to the loss; soft ordinal offsets (`sord_sigma`) exist and are off; the grid is fixed in Da whatever the instrument, and the ion-trap CID dataset (UPS1) is the one where the model is weakest (`docs/results/eval_ups1.md`: 38.4 % fragment-group accuracy). `fixed_ppm` and `adaptive` binnings exist and are unused. | confirmed (code and results) | The task's resolution is an instrument property; a Da grid over-asks of ion traps and under-asks of Orbitraps above 1,000 Da. | `sord_sigma` on; adaptive binning; evaluate by fragmentation type (the suite reports HCD / CID subsets). 2 runs. |
| A5 | The 10 Da hint and the visible intensity turn "predict a missing peak" into "refine a coarse location of a peak whose intensity you know". | confirmed (design); effect inferred | The representation the probes and retrieval reward may prefer a harder task (no hint) or a different one; the preprint fixes σ by argument. | σ in {0 (pure mask token), 5, 10, 20} at one setting of intensity visibility. 3 runs. |
| A6 | The Δm/z chemical-gap head is the only chemistry-shaped self-supervised target in the code and it is off; the retrained model's fragment reconstruction improved with more signal per step (batch 2,048), which says the objective is signal-starved rather than capacity-starved. | confirmed (code); inferred (reading of the batch result) | A target that asks "which residue mass separates these two peaks" is the ladder relationship the attention heads are said to discover on their own. | `lambda_dmz > 0` (the labels need a builder; check what `dmz_labels` expects) against the A1-fixed baseline. 1 run. |

### B. Input representation

| # | Observation | Status | Why it matters | What would settle it |
| :-- | :---- | :---- | :---- | :---- |
| B1 | The peak encoder's 384 frequencies span one decade of periods (2.5 to 25 Da) and did not learn; sub-Dalton structure and absolute position are both left to the MLP to build from phases (section 1.2). | confirmed | A 0.2 Da target from a 2.5 Da finest period is feasible but tight; the alternatives (wider Fourier bank, RBF, dual) are already in the file and have no number. | Encoder ablation: multiscale (baseline), fourier 2.5 to 3,750 Da, dual, linear control. 3 runs. |
| B2 | No precursor information reaches the encoder. | confirmed | For retrieval by peptide identity the precursor m/z and charge are the cheapest strong features; their absence is also what makes the charge and mass probes meaningful. A deliberate trade the preprint does not discuss. | Meta token on with precursor mass and charge only (code exists); judge retrieval and IG with the probes reinterpreted. 1 run. |
| B3 | Intensities are sqrt + L2 per spectrum, so their scale depends on the peak count (median 0.05, maximum 0.3 to 0.9), while the Thompson prior and the config comments assume [0, 1]; anchor selectivity therefore varies with spectrum density. | confirmed (measured) | Masking pressure differs between sparse and dense spectra for a reason nobody chose. | Normalise the Thompson score to each spectrum's base peak; measure the anchor-intensity distribution before and after; retrain if it changes the mask. 0 to 1 runs. |
| B4 | Top-200 by intensity with the precursor kept; 37.8 % of MCFM test spectra hit the cap. | confirmed (measured) | Dense spectra lose their faint fragments before the model sees them; no `n_peaks` ablation. | `n_peaks` 300 with the same batch, memory permitting (attention is quadratic in 200 → 300). 1 run. |

### C. Encoder

| # | Observation | Status | Why it matters | What would settle it |
| :-- | :---- | :---- | :---- | :---- |
| C1 | The pairwise bias is zeroed on every pair that touches a masked token, so under span masking the bias carries no Δm/z information exactly where reconstruction happens; the preprint's reading of "TS·PA is the weakest" is that the bias hands the ladder over and the heads stop specialising. | confirmed (code); the explanation is inferred | If the bias lost under TS because it was switched off around masks, the published conclusion about pairwise attention is an artefact of the masking mode. `pa_mask_to_mask` and `blur_masked_pa` exist to test it. | TS·PA with `pa_mask_to_mask: true` and with `blur_masked_pa: true` at 40M scale; the PA path is memory-chunked and skips `torch.compile`. 2 runs. |
| C2 | 2017-style layer: post-norm, ReLU, dropout 0.1 everywhere, feed-forward ratio 1.33 for the 40M; gradient checkpointing always on. | confirmed | Pre-norm with RMSNorm, SwiGLU and little or no dropout at this data scale is the default elsewhere and usually trains faster at equal parameters; the 40M's narrow feed-forward is an untested cost cut. | One modernised layer at matched parameters (pre-norm, SwiGLU, dropout 0), one 40M with feed-forward 3,072 at matched steps. 2 runs. |
| C3 | The latent token reads nothing (no loss) and is excluded from the embedding, yet attends and is attended at every layer. | confirmed | Either a free register token (possibly useful, never measured) or dead weight; the doc calls it the model's "latent", which it is not. | Remove it, or give it the pooling job (D1). Covered by D1. |
| C4 | Attention weights are stored twice in every checkpoint (`self_attn.*` and `custom_attention.*`). | confirmed | Half the file size; parameter counts from state dicts are wrong by a factor near two for attention. | Register the module once; a loader shim for old files. No run. |

### D. Pooling and the embedding

| # | Observation | Status | Why it matters | What would settle it |
| :-- | :---- | :---- | :---- | :---- |
| D1 | The spectrum embedding is an untrained mean over up to 200 tokens; anisotropy ratio about 25, effective rank about 90 of 768; retrieval recall@1 0.305 on LCFM test for the 40M. | confirmed (measured in the reruns) | Nothing in the objective asks the spectrum-level vector to be anything; the probes and retrieval are measured on a by-product. A spectrum-level term is the most direct way to change what the evaluations measure. | A contrastive term on the pooled (or CLS) vector with two maskings of the same spectrum as the positive pair and the batch as negatives (1,023 at batch 1,024), or an attention-pooling head trained by it; judge on retrieval, probes and IG together, since the two objectives can trade. 2 runs. |
| D2 | A label-free whitening transform fitted on one ProteomeTools run lifted the released 40M's same-peptide retrieval precision across analyzers from 0.196 to 0.952 and across runs from 0.494 to 0.664, against a binned-spectrum baseline of 0.828 and 0.845 (workstation evaluation repository `Arc/instanovo-fm-evals`, `docs/claude_logs/review-2026-09-25.md`, its table; not re-run here). | confirmed there, not re-verified here | Most of the retrieval deficit across acquisition conditions is geometry, not content: the mean-pooled space carries a few dominant acquisition directions. | Run the suite's probes and retrieval on whitened embeddings of the released 40M on the corpus splits. 0 runs. |

### E. Training recipe and selection

| # | Observation | Status | Why it matters | What would settle it |
| :-- | :---- | :---- | :---- | :---- |
| E1 | The stated recipe (batch 1,024, 90,000 steps) does not reproduce the released 40M's fragment accuracy; batch 2,048 does, and the released checkpoint's epoch counter says 2× spectra per step. | confirmed | Any architectural comparison must be made at the budget that reproduces, or both arms lose the same 4 points for a reason unrelated to architecture. | Use the batch-2,048 run (`40M-b2048`, steps 80,001 and 90,001) as the baseline of every experiment here. |
| E2 | fp16 hard-coded, overflow history unlogged; bf16 unavailable (`session-2026-09-25-train40M.md`). | confirmed | A free stability gain on H100 and one fewer silent variable. | Expose `mixed_precision`; log the scaler. No run on its own. |
| E3 | No weight decay, one seed, no sweep tooling; schedule shape tied to `training_steps`. | confirmed | Seed noise is unmeasured, so a one-run difference of a point cannot be read. | Two seeds of the baseline before any comparison; everything else is judged against that spread. 1 extra run. |
| E4 | Masking runs on the CPU in the collate with per-spectrum Python loops (trimming, isotope cap); GPU dips at shard boundaries. | confirmed (logs) | Not a modelling point, but it sets the cost of every run above. | Vectorise the trimming or move masking after the transfer. No run. |

### F. Data couplings worth keeping in view

- The corpus tiers are defined by database-search confidence (the preprint's own limitation), and 84 % of our
  ProteomeTools peptides were pretraining targets (`Arc/instanovo-fm-evals/docs/claude_logs/session-2026-09-23.md`),
  so "held out" is a property of peptides, not of chemistry or instruments. Not an architecture point, but it bounds
  what the evaluations can show. **Confirmed there, not re-verified here.**
- 7.6 % of the MCFM test sample has precursor charge 0 (unknown); the isotope co-masking then uses the batch's
  largest charge for the shift table, which is harmless, but any precursor-conditioned design (B2) must handle it.
  **Confirmed (measured).**

## 3. Decisions (15/10) and the plan that follows from them

The owner decided: (1) the batch-2,048 configuration is the baseline; (2) **the training objective is not changed**:
masking, the hint, the heads, the losses and the selection criterion stay as shipped, and the work is about the
architectural choices; an architectural change that cannot run without touching the objective is allowed only if
that dependency is stated in so many words; (3) every result is at 40M on MCFM, because the 89M cannot be trained
to completion now.

Consequences for section 2: A1 to A6 and B3 are **parked** (they are objective changes); D1 is **excluded** in its
contrastive form, because a pooled vector has no loss in the shipped objective and any pooling head would need a
new term (stated here as decision 2 requires); A3 survives only as part of how runs are read (section 4), not as a
change. C4, E2 and E4 are engineering and go in whenever a run touches that code.

### 3.1 Baseline

`40M-b2048`: `configs/foundational.yaml` with `dataset=mcfm`, 1,024 spectra per forward pass and `grad_accumulation=2`,
90,000 steps, one H100; the existing run took 18 h 57 min of training (job 23151784, `sacct`), about 10 GPU-hours
less than two single-GPU passes because `torch.compile` stays on. Its checkpoints are `$CHECKPOINTS/
instanovo-fm-mcfm-90k-b2048-step{80k,90k}-2026-10-03.ckpt`. A **second seed** of it is the first job of the plan:
without the seed spread no one-run difference can be read. Overrides of every run, besides the arm's own:
`training_steps=90000 grad_accumulation=2 num_workers=8 +mp_sharing_strategy=file_system keep_model_every_interval=True
post_training_evaluation.enabled=False` (the evaluation runs as separate jobs, as for every checkpoint so far).

### 3.2 Arms: one architectural change each on the baseline

The arm table moved on 15/10 (late) to `docs/arch_study/README.md`, the study's one home for the arms, their
overrides and parameter counts, the judging protocol, the runs and the problems met; the results pages beside it
(`smoke.md`, `results.md`) are generated from `docs/references/rerun/`. Phase 2 arms that need code (`prenorm-gelu`:
pre-norm layers with GELU through `UnifiedEncoderLayer`; `no-latent`: a flag that skips the latent token; `swiglu`: a
SwiGLU feed-forward at matched parameters) stay listed here and are not part of the current goal. Every arm keeps the
shipped objective; the one architectural idea that cannot (a pooled or CLS embedding trained to be an embedding) is
excluded by decision 2 and stays in section 2 as D1 for the record.

### 3.3 Cost and order

One arm is about 19 h of training on one H100 in the 24-hour partition plus about 4 h of evaluation in the 3-hour
partition (the result scripts). Phase 0 and the eight Phase 1 arms: 9 runs, about 210 GPU-hours; Phase 2: 3 runs,
about 70. Order inside Phase 1 by expected information: `meta-precursor`, `pa-mask2mask`, `enc-fourier`, `ionladder`,
then `ffn3072`, `prenorm-gelu` (Phase 2, cheap code), `dropout0`, `enc-dual`, `peaks300`, `no-latent`, `swiglu`.

Ready for the first job: `scripts/train/submit.sh` now takes its checkout from `SLURM_SUBMIT_DIR` and its venv from
`FM_VENV`, like the other wrappers; nibi's spare clone `~/experiments/InstaNovo-FM-evalpkg` (venv
`~/scratch/venvs/instanovo-fm-evalpkg`, `FM_VENV` in its `.envrc`) is on this branch, so the arms run from there and
`main`'s clone is untouched. The tags of the arms go into `scripts/reproduce/_common.py::MODELS` and
`fill_results.py::_retrain_rows` as the retrains did, one section per arm in `results_paper_or_rerun.md`.

## 4. How a change will be judged

The suite's paper protocols on the corpus splits, run by `scripts/reproduce/` exactly as for the released
checkpoints (`paper-probes-retrieval`, `paper-peak-level`, `paper-geometry` on LCFM test and MCFM test, the trainer
validation on MCFM validation), plus the IG fragment-group accuracy over the 42,441-group diagnostic
(`scripts/analysis/fragment_reconstruction.py`) because the paper's 1,224-group number resolves ±1.9 points at best.
Each arm is read at step 90,001 and at its `model_best` (the shipped criterion), so the selection question (A3)
is visible without changing anything. A change counts when it clears the two-seed spread of the baseline on
retrieval and fragment accuracy without losing a probe, with every number landing in
`docs/references/results_paper_or_rerun.md` through `fill_results.py` under a new section per arm.

## 5. Open points

- `ffn3072` is kept as a width test although it is not parameter-matched; `dropout0` was dropped (decision of 15/10).
- Launched 15/10 evening (owner's go through the goal): see the log.

## Log

| Date | What happened |
| :---- | :---- |
| 15/10/2026 | Branch `sdu/exploring-arch` created from `main` (5508e71) and pushed. Read the model, data, trainer and config code end to end, the preprint's architecture and ablation passages, and the earlier session logs. Wrote `scripts/analysis/masking_stats.py` and measured the deployed masking on 2,000 MCFM test spectra: 27.0 % masked, runs of mean length 5.2 with 59 % of masked peaks in runs longer than the configured 4, 0.8 % residual isotope leakage. Found that the intensity auxiliary loss never restricts itself to masked peaks (`mlm_mask` key never written) and so regresses a visible input channel. Sections 1 to 5 above are the state at the end of the day; no experiment started. |
| 15/10/2026, later | Owner's decisions recorded (batch-2,048 baseline; objective unchanged, architecture only, with any objective dependency stated; 40M on MCFM). Sections 3 to 5 rewritten into the arm table above: eight configuration-only arms and three small-code arms, each one change on the baseline, objective untouched; the contrastive pooling idea excluded under decision 2; A-group items parked. Not launched. |
| 15/10/2026, evening | Preflight without GPU time: all nine arms compose and build on the workstation (parameter counts in the table); `scripts/train/submit.sh` made checkout- and venv-relative; the nibi spare clone switched to this branch with its venv re-synced. Launch of Phase 0 and Phase 1 awaits the owner's go (about 210 GPU-hours). |
| 15/10/2026, night | Owner's go for Phase 0 and Phase 1; `dropout0` dropped (dropout stays 0.1 everywhere). Eight jobs submitted from nibi's branch checkout (`~/experiments/InstaNovo-FM-evalpkg`, run directories `$RUNS/arch-<arm>`), each one H100 with MCFM staged to node disk, overrides of section 3.1 plus the arm's: `arch-seed102` 24017759, `arch-meta-precursor` 24017760, `arch-pa-mask2mask` 24017761 (36 h limit, no `torch.compile` on the PA path), `arch-enc-fourier` 24017762, `arch-ionladder` 24017763, `arch-ffn3072` 24017764, `arch-enc-dual` 24017765, `arch-peaks300` 24017766 (micro-batch 512 × 4 accumulation for memory, same global batch); 30 h limits, so the 3-day partition (`gpubase_bygpu_b4`, 524 pending at submission against 1,805 in the 24-hour one). Speed observations go to `notes-speed.md` from now on. |
| 15/10/2026, late night | All eight arm jobs still pending (priority 1,216,426 on `rrg-hsn_gpu`, 57 to 69 jobs ahead in the 3-day partition; fairshare is the whole priority: the same jobs on `def-hsn_gpu` would score about 1,717,000, above every pending job there; the account choice is the owner's). Used the wait for speed idea S1 of `notes-speed.md`: the suite's retrieval task now searches every query in one FAISS call (suite commit 6ee1c29, 12.4 s against 1,595 s on 50,000 MCFM test spectra; equal to the old loop except at near-ties, where FAISS's kernels round differently, traced in the suite's `docs/regression.md`); the pin in `pyproject.toml` moved from 1c6153b to 6ee1c29 and the lock, the local venv and nibi's branch venv re-synced, so the arms' `result_arm.py` jobs run the batched task. No arm result yet. |
| 16/10/2026, early | Arms still pending (67 ahead). Second speed change while waiting: the suite's linear probe fits its ten targets in parallel single-threaded processes (suite commit a4eff7c; 520 of 520 numbers identical to the sequential task on real splits, 923 s against 2,087 s on 8 cores; the suite's `docs/regression.md`). Pin moved to a4eff7c, lock, local venv and nibi's branch venv re-synced; the arms' `result_arm.py` jobs now carry both the batched retrieval and the parallel probe, so a probes-retrieval job should take about 30 min instead of 90. The eight training jobs read the checkout at start: both changes touch evaluation code only, nothing in the trainer. |
| 15/10/2026, midday | Owner: smoke the top four arms on `def-hsn_gpu`, keep the eight full runs on `rrg-hsn_gpu`, document the study in `docs/`. Submitted `smoke10k-seed102` 24048089, `smoke10k-meta-precursor` 24048090, `smoke10k-ionladder` 24048092 (3-hour partition) and `smoke10k-pa-mask2mask` 24048091 (12-hour partition, 5 h): 10,000 steps with the baseline's exact learning-rate trajectory (`training_steps=10000 warmup_frac=null lr_hold_frac=null +warmup_iters=4500 +lr_hold_steps=27000`, composed and checked locally: warmup 4,500, hold through the run, validation and checkpoint at 10,000, 250 validation batches), priority 1,707,980 on def against 1.21 M on rrg. Checked before launching the PA arm that `pa_mask_to_mask` zeroes every mask-visible pair and keeps exact mass differences only between masked peaks (`encoder.py` mode 2); the objective is untouched, but the task is easier (hints can be pooled), recorded as a caveat. New directory `docs/arch_study/` (README: rules, baseline, arms with motivation and overrides, judging, runs, problems; `smoke.md` and `results.md` generated); `arch_results.py` writes there now, `export_metrics.py` records seconds per step and peak GPU memory, `smoke_results.py` renders the smoke table against the baseline's step-10,001 metrics. The arm table of section 3.2 moved to the README. |
| 15/10/2026, afternoon | Smoke runs started 11:36 on one node (four concurrent stagings 429 s each; first step at 28 min). `seed102`, `meta-precursor`, `ionladder`: 0.74 s per optimizer step, 30 to 33 GB, on course for step 10,000 at about 14:05. `pa-mask2mask`: step 1,000 only at 50 min 31 s of training (3.03 s a step, 59.7 GB; training loss 1.95 against 2.13 to 2.37), so 10,000 steps would take 8.4 h against the 5-hour limit and the full run's 90,000 would take 76 h against 36: cancelled at 1 h 19 min; the finding and the options are in `docs/arch_study/README.md` (problems). The trainer's `s/step` turned out to be per forward pass (ETA short by the accumulation factor); `export_metrics.py` now reports seconds per optimizer step from the elapsed clock and the forward-pass figure separately. Baseline export regenerated with its step-10,001 and step-80,001 blocks and the full trajectory (every previous value identical). |
| 15/10/2026, 14:00 | Smoke runs finished: `seed102`, `meta-precursor`, `ionladder` completed (2 h 26 to 2 h 30 min, 0.74 to 0.76 s a step, 34 to 42 GB); exports under `docs/references/rerun/train_smoke10k-*/`, table in `docs/arch_study/smoke.md`. At step 10,001 none of the three beats the baseline on the masked-m/z metrics (bin accuracy 12.03 baseline, 11.77 seed 102, 11.75 meta-precursor, 11.56 ionladder; median ppm 6,342 / 6,409 / 6,448 / 6,369): the second seed alone moves the metric by 0.26 points and 67 ppm, so at this step the arms are indistinguishable from seed noise, ionladder a little beyond it on the wrong side. `meta-precursor`'s precursor tokens have not made the masked-m/z task easier by step 10,000, contrary to the caveat's expectation; whether that changes later is what the full run will show. 8.7 GPU-hours of def-hsn_gpu used. Baseline export regenerated with the fixed step-time fields (0.75 s per optimizer step, 0.37 per forward pass). |
| 15/10/2026, evening | Owner: profile the PA path before any optimisation, on the workstation once its other experiments finish (def-hsn_gpu to be spared, rrg too slow). Wrote `scripts/analysis/profile_pa.py`: one training step as the trainer runs it (real MCFM batches through the training processor and collate, fp16 autocast, GradScaler, AdamW, dropout and checkpointing on), variants baseline / baseline-compiled / pa / pa-mask2mask / pa-compiled, profiler ranges around `PairwiseAttentionBias` (Fourier features and f_pw over every peak pair), `BatchedPairwiseProjection` (LayerNorm and the projection to one bias per head and layer) and `FlashMHA` (SDPA memory-efficient kernel with a bias, flash without), the backward and the optimizer; `--pa-chunk` reproduces the H100's chunking at a small batch. CPU dry run at batch 2 passes for baseline and pa-mask2mask. What the code already says: the PA path builds (B, L, L, 128) Fourier features and a 128-128-64 MLP over them in chunks of 64 with gradient checkpointing (recomputed in backward), then (B, L, L, 108) per-layer biases (12 heads x 9 layers) that all stay resident while the layers run (at batch 1,024 in fp16: 983 MB per layer, 8.8 GB for the nine), attention takes the bias through SDPA's memory-efficient kernel instead of flash, and `torch.compile` is off for the whole model. |
| 16/10/2026, morning | Owner's go to profile on the workstation (GPU idle). `profile_pa.py` run twice (the second with wall-clock phases after the first showed the backward range empty: autograd's kernels run on another thread), plus the 16-chunk variant and `sdpa_backend_check.py`. Findings in `docs/arch_study/README.md` (section "Profile of the pairwise-attention path"): PA step 0.429 s against 0.178 at batch 32; attention 4.5 times the baseline's because the per-layer bias reaches SDPA with strides (484812, 1, 2412, 12) and every fused kernel refuses a non-unit last stride, so attention runs the math path; features and projection a third of the extra; chunking +10 %; compile fails on the path with a Triton compilation error. Remedy (a contiguous bias) stated, not applied: owner's call. Full runs still pending on rrg (53 ahead). |
| 16/10/2026, midday | Owner: apply the fix, re-profile, then a local run instead of a full run. One `.contiguous()` in `BatchedPairwiseProjection._forward_chunk`; `sdpa_backend_check.py` confirms the memory-efficient kernel now runs and that math and fused outputs agree to 7e-4 on logits of scale 0.16 once the hint noise is reseeded (the first comparison without reseeding showed 7 % and was the noise, not the kernel). Re-profile: PA 0.326 s against baseline 0.178 at batch 32 (was 0.429); attention range 0.043 (was 0.130). Local speed runs launched in a capped scope: baseline then pa-mask2mask, 6,000 steps each at batch 64 on the two local MCFM train shards. Owner also asked for a walkthrough of the dataset evaluations (`docs/results/eval_*.md`) and of the arms tried, after this. |
| 16/10/2026, 12:00 | Owner spotted two things in `docs/results/eval_metrics2.png`: a "40M batch 2,048, step 80k" group with the same value on every dataset, and missing instrument bars. Cause of the first: `scripts/evals/plot_results.py` read the pages' table cells by position for five checkpoints while the pages hold three plus the reproduction column, so the reference column was drawn as that checkpoint and the dashed reference line disappeared (bug since commit 2318284). Fixed: columns read by header label, the three checkpoints only, MS2Bac named with its species, `eval_metrics2` datasets ordered by training overlap in red, purple, blue, light blue, grey for LCFM test. The missing bars are correct: MS2Bac, UPS1 and ProteomeTools each have one instrument, a one-class target the probe cannot score. The PNGs are ignored by git and regenerated in the working tree. |
| 16/10/2026, 12:30 | Local speed runs (owner's instruction: a local run instead of a full one): baseline at micro-batch 64 (0.311 s a step, 34 min end to end), then PA at 64 out of memory on the 8 GB card, then the pair at micro-batch 32: baseline 0.164 s a step (19 min), PA 0.337 (37 min), ratio 2.05 after the fix against 4.1 on the H100 before it; projected H100 PA step 1.53 s, 38 h for 90,000 steps, a few hours over job 24017761's 36-hour limit (owner to decide: 48-hour resubmission or drop). The PA arm's validation at step 6,001 is in another regime (bin accuracy 47 % against 5 %, median 422 ppm against 8,658), consistent with hint pooling for the group accuracy and beyond it for the offset accuracy; not comparable, recorded as such. Exports under `docs/references/rerun/train_local-speed-*/`. Also today: the figure script's column bug (reference drawn as a fifth checkpoint) fixed, the figures reordered and relabelled on the owner's request, and two owner questions answered in the conversation (why collision energy is negative on LCFM test: mixed units in the target; why the 89M has no corpus trainer-validation bar: no LCFM-validation pass exists, MCFM validation would do). |
| 20/10/2026, 12:50 | Five days queued on rrg-hsn_gpu without a start (fairshare: the group's rrg usage is above its target, def's far below). Owner: launch four arms on def-hsn_gpu and watch the trainer's validation metrics. Done from the single clone `~/experiments/InstaNovo-FM` (the second clone and its venv removed, see the handoff): the eight jobs of 15/10 cancelled, `arch-seed102` 24679078, `arch-meta-precursor` 24679079, `arch-ionladder` 24679080, `arch-enc-fourier` 24679081 on def-hsn_gpu (seed102 running within two minutes), `arch-pa-mask2mask` 24679082 (48 h), `arch-ffn3072` 24679083, `arch-enc-dual` 24679084, `arch-peaks300` 24679085 on rrg-hsn_gpu. Stopping rule agreed: clear losers only, at steps 30,000 and 50,000 against the baseline's trajectory (`docs/references/rerun/train_40M_mcfm_90k_b2048/metrics.json`) with the seed-102 gap as the noise; leaders run to 90,000 and get an equal-step suite comparison while running. Table and rule: `docs/arch_study/README.md`, Runs. |
| 20/10/2026, 14:35 | Owner: cancel the four rrg-hsn_gpu jobs (24679082 to 24679085: `pa-mask2mask`, `ffn3072`, `enc-dual`, `peaks300`). Done; the study runs the four def arms first. All four def runs at step 4,000 to 5,000, 0.37 s a forward pass. Two logs carry one MLflow async-logging error each (SQLite write timed out, 14:06); absent from every single-run log so far, so four concurrent runs on the project filesystem is the likely cause (inferred); training unaffected, at most one batch of training-loss points missing from the store. |
| 20/10/2026, 20:00 | First decision point, step 30,001 (held learning rate; the decay starts at 31,500). Trainer validation on MCFM validation, read from each run's `mlflow.db` (`eval/median_ae_ppm`, `eval/bin_accuracy`), baseline from `docs/references/rerun/train_40M_mcfm_90k_b2048/metrics.json`: baseline 4,712 ppm / 22.79 %; seed102 4,748 / 22.65; meta-precursor 4,701 / 22.92; ionladder 4,669 / 22.67; enc-fourier 4,741 / 22.70. Seed gap 36 ppm and 0.14 points; the whole field spans 79 ppm and 0.27 points, inside the run-to-run spread of identical configurations measured at step 10,001 (seed102 and ionladder each run twice: up to about 100 ppm and 0.35 points). Nobody is near the futility threshold (twice that unit), so no run is cancelled. meta-precursor's lead at 20,001 (116 ppm) shrank to 11 ppm. Steps 10,001 and 20,001 showed the same picture (20,001: baseline 5,414 / 19.25; seed102 5,437 / 18.99; meta-precursor 5,298 / 19.36; ionladder 5,363 / 18.65; enc-fourier 5,422 / 19.08). Validations arrive every 2 h 10 min; step 50,001 about 00:20, end about 09:00 on 21/10. |
| 21/10/2026, 00:15 | Step 50,001 (decaying learning rate since 31,500): baseline 3,897 ppm / 26.81 %; seed102 4,045 / 26.21; meta-precursor 4,017 / 26.42; ionladder 3,828 / 26.72; enc-fourier 4,019 / 26.38. The seed gap has widened to 148 ppm and 0.60 points, so that is the noise unit now; nobody approaches twice it, no cancellation. ionladder is the only arm ahead of both seeds on median ppm (69 ppm beyond seed 101, 217 beyond seed 102) and has been ahead at every validation (80, 51, 43, 165, 69 ppm); level on bin accuracy. Step 40,001 showed the same ordering (baseline 4,326 / 24.95; seed102 4,344 / 24.65; meta-precursor 4,303 / 25.00; ionladder 4,161 / 25.02; enc-fourier 4,308 / 24.93). Per the plan, the equal-step suite comparison runs now on the step-50,001 checkpoints: `scripts/reproduce/equal_step.py` (`paper-probes-retrieval` on MCFM test) for ionladder, the baseline and seed102, three jobs on def-hsn_gpu's 3-hour partition (about 2 GPU-hours); the seed side gives the band on retrieval and the probes, which the judging rule needs. Summaries to `docs/arch_study/equal_step/`. |
| 21/10/2026, 00:35 | The three equal-step jobs did not start on def-hsn_gpu: fairshare usage accrues while jobs run, and twelve hours of the four arm runs took the account's level fairshare from 4.13 to 0.60 (rrg-hsn_gpu: 0.97), my def factor from 0.307 to 0.197; the jobs had 1,999 ahead on the 3-hour partition against 205 under rrg. Cancelled (24721317 to 24721319) and resubmitted under rrg-hsn_gpu: 24723244 (ionladder), 24723245 (baseline), 24723246 (seed102). Consequence for the plan: def bought the four runs an immediate start and is now spent for about a week; the two smokes proposed for 21/10 (checkpointing off, Muon) go on rrg's 3-hour partition. The four training runs are unaffected. |
| 21/10/2026, 02:15 | Owner (01:20): stop `meta-precursor` and `enc-fourier` at their step-60,001 checkpoint to spare def's usage; `seed102` and `ionladder` run to 90,000. A detached waiter (`~/scratch/stop_at_60k.sh` on nibi, log `stop_at_60k.log`) cancelled 24679079 at 02:04:32 and 24679081 at 02:06:32, each after its `Saved accelerator state` line, so both keep a complete step-60,001 checkpoint (`model_epoch_10_step_60001.ckpt`). Step 60,001: baseline 3,609 ppm / 28.20 %; seed102 3,800 / 27.38; meta-precursor 3,757 / 27.37; enc-fourier 3,828 / 27.46; ionladder 3,433 / 28.14. The seed gap is now 191 ppm and 0.82 points; the two stopped arms sit beside seed102 on every metric; ionladder is 176 ppm beyond the better seed and 367 beyond the other, level with the better seed on bin accuracy, and has been ahead of seed102 at all six validations (114, 74, 79, 183, 217, 367 ppm). Checked that the growing seed gap is not code drift: `git diff 08b7979 sdu/exploring-arch` on the training path shows only the in-loop evaluator's removal and the pairwise-attention contiguity fix (no running arm uses PA); the recipe is identical. Reading: one seed per arm cannot resolve effects smaller than about 200 ppm at this stage; the equal-step suite comparison at 50,001 (jobs 24723244 to 24723246, queued on rrg) and the final checkpoints at 90,001 decide ionladder. |
| 21/10/2026, 08:50 | `seed102` (08:35) and `ionladder` (08:45) completed. Trainer validation at step 90,001 (MCFM validation, 256,000 spectra): baseline 3,215 ppm / 29.64 % / 4.56 %; seed102 3,455 / 29.11 / 4.46; ionladder 3,041 / 29.80 / 4.56 (median ppm / bin accuracy / within 20 ppm). Final seed gap 240 ppm and 0.53 points. ionladder beat seed102 at all nine validations and ends beyond both seeds on every metric; its lead over the better seed on median ppm (174) is about three quarters of the seed gap, so the trainer metric alone does not settle the judging rule. Steps 70,001 and 80,001: baseline 3,372 / 28.94 and 3,306 / 29.42; seed102 3,594 / 28.3 and 3,546 / 28.87; ionladder 3,256 / 29.02 and 3,099 / 29.54. Per-arm procedure run for both (checkpoints copied to their `.envrc` names, exports committed as 477819e from nibi, `results.md` rendered): result jobs seed102 24748530 to 24748533, ionladder 24748628, 24748629, 24748630, 24748652, on rrg-hsn_gpu's 3-hour partition with the three equal-step jobs. A finisher-script bug (two assignments in one `local`, `${!var}` expanded before `var` existed, fatal under `set -u`) killed the seed102 branch silently; that arm was done by hand, the script fixed for ionladder. The exports carry holes in the trajectories (S16 of notes-speed.md); the final records are complete. |
| 21/10/2026, 09:20 | The three equal-step jobs started at 09:11 and failed in six seconds: `scripts/reproduce/submit.sh` passed only its first argument to Python (`"$1"`), so `equal_step.py` saw no options; `result_arm.py` takes options too, so the eight result jobs queued at 08:41 would have failed the same way. Wrapper fixed to forward every argument (`"$@"`, 1165d13; the per-model result scripts take none and are unaffected; verified with `--help` through the wrapper on the login node). The eight cancelled and all eleven resubmitted on rrg-hsn_gpu's 3-hour partition: equal-step 24749871 (ionladder), 24749872 (baseline), 24749873 (seed102); results seed102 24749874 to 24749877, ionladder 24749878, 24749891, 24749892, 24749893. About 30 minutes of queue age lost. |
| 21/10/2026, 14:15 | Owner's wish of 20/10 ("should we try these speed-ups and potentially muon?"): Muon implemented while the evaluation jobs queue. `common/muon.py`: one optimizer, `MuonAdamW`, groups tagged `algo`; `muon_param_groups` puts the 36 encoder matrices under Muon (the fused qkv as three 768x768 blocks), everything else under an AdamW branch that matches torch's single-tensor AdamW to 1e-6 (tested); Newton-Schulz bit-identical to modded-nanogpt's on every shape tried (tested); Moonlight's 0.2*sqrt(max(rows, cols)) update scaling so `learning_rate`, `weight_decay` and the schedule are shared. Switch: `optimizer: muon` (configs default `adamw`; no existing run changes; `muon_momentum: 0.95`). Tests `tests/unit_test/foundational/test_muon.py` (6; two of my first assertions were wrong, not the code: five quintic steps leave singular directions below about 3 % of the largest small, as the reference does, so the tests check the captured range and reference equality). The two existing optimizer-adjacent test files pass (34). Local comparison started 14:15 on the workstation (GPU idle, scope `local-muon-b32`, 20 GB ceiling): the `baseline-b32` recipe of 16/10 with `optimizer=muon`, 6,000 steps at micro-batch 32, validation every 1,000; the AdamW run's trajectory is `docs/references/rerun/train_local-speed-baseline-b32/metrics.json` (6,001: 8,658 ppm, 5.09 %). Cluster smokes (checkpointing off, Muon) wait for the owner's word on the GPU-hours; def is spent, so they would go on rrg's 3-hour partition. |
| 21/10/2026, 14:40 | Local Muon comparison finished (exit 0, 22 min, scope peak 20.0 GB = the ceiling, reached during loading; no kill). Against the AdamW run of 16/10 at every validation step (median ppm / bin accuracy): 1,001: 11,657 / 3.51 vs 11,259 / 3.70; 2,001: 10,370 / 4.17 vs 10,324 / 4.05; 3,001: 9,819 / 4.48 vs 9,843 / 4.46; 4,001: 9,307 / 4.79 vs 9,119 / 4.81; 5,001: 9,135 / 4.96 vs 8,780 / 4.94; 6,001: 9,025 / 5.00 vs 8,658 / 5.09. Step time 0.196 vs 0.164 s. Reading: no gain at micro-batch 32 over 6,000 steps, bin accuracy level throughout, median ppm 2 to 4 % worse in the decay phase; one seed each and the local spread unmeasured, and this is the small-batch corner where Muon is not expected to help; the batch-2,048 question needs the cluster smoke (owner's call). Export committed: `docs/references/rerun/train_local-speed-muon-b32/`. |
| 21/10/2026, 16:02 | Owner: run what can be run on the workstation instead of waiting on rrg (the ten evaluation jobs had 56 ahead all day). The MCFM-test half can: all 14 MCFM test shards are mirrored locally (`$CORPUS/mcfm/`), and the suite's `docs/regression.md` puts retrieval and the regression probes within 1e-4 of nibi's on this GPU, the classification probes within 0.013. The LCFM-test half (results 1, 2, 3 of both arms) cannot: LCFM is not mirrored (162 GB). Four checkpoints copied from nibi into `instanovo-fm-runs/nibi-mirror/<run>/checkpoints/` (md5 verified). Chain `instanovo-fm-runs/local-evals.sh` started 16:02, one run after another in 32 GB scopes, through the repo's own scripts so the outputs land where nibi's would: equal-step baseline and seed102 at 50,001 (`docs/arch_study/equal_step/`), then result 5 of ionladder and seed102 (`docs/references/rerun/`). The matching nibi jobs (24749872, 24749873, 24749877, 24749893) stay queued until the local results land, then get cancelled. |
