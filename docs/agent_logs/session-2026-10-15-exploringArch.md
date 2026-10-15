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

## 3. A shortlist, ordered by information per GPU-hour

Nothing below has been started. Baseline for everything: the batch-2,048 40M run (`40M-b2048`), which is the
configuration that reproduces the released checkpoint; two seeds of it first (E3).

1. **A3, selection criterion:** re-score the kept checkpoints of the two existing runs; no training.
2. **A1, intensity auxiliary:** `lambda_intensity: 0`, then the masked-only variant with hidden intensity.
3. **A2, run length:** fix the trimming so realised runs are 3 to 4 peaks.
4. **D1, a spectrum-level term:** contrastive on the pooled vector.
5. **C1, pairwise bias around masks:** `pa_mask_to_mask` under Thompson-span.
6. **B1, peak encoder bank;** then **C2, layer modernisation;** then **B2, precursor meta token;** then **A5, σ;**
   then **A4, ordinal offsets and adaptive bins.**

About 150 H100-hours for the whole list at one seed each, within what the account has spent on evaluations alone
since September (`session-2026-10-10-evalPackageAndReconstruction.md`, queue analysis).

## 4. How a change will be judged

The suite's paper protocols on the corpus splits, run by `scripts/reproduce/` exactly as for the released
checkpoints (`paper-probes-retrieval`, `paper-peak-level`, `paper-geometry` on LCFM test and MCFM test, the trainer
validation on MCFM validation), plus the IG fragment-group accuracy over the 42,441-group diagnostic
(`scripts/analysis/fragment_reconstruction.py`) because the paper's 1,224-group number resolves ±1.9 points at best.
A change counts when it clears the two-seed spread of the baseline on retrieval and fragment accuracy without losing
a probe, with every number landing in `docs/references/results_paper_or_rerun.md` through `fill_results.py` under a
new section per experiment.

## 5. Questions for the owner before any run

- Is the batch-2,048 configuration accepted as the baseline, given it is the one that reproduces?
- A1 changes the objective in a way that makes the released checkpoints not comparable on the intensity row; is
  that acceptable (the row measures a copy anyway)?
- Should the first runs be on MCFM at 40M (10 GPU-hours each) or does any result need the 89M on LCFM to count?

## Log

| Date | What happened |
| :---- | :---- |
| 15/10/2026 | Branch `sdu/exploring-arch` created from `main` (5508e71) and pushed. Read the model, data, trainer and config code end to end, the preprint's architecture and ablation passages, and the earlier session logs. Wrote `scripts/analysis/masking_stats.py` and measured the deployed masking on 2,000 MCFM test spectra: 27.0 % masked, runs of mean length 5.2 with 59 % of masked peaks in runs longer than the configured 4, 0.8 % residual isotope leakage. Found that the intensity auxiliary loss never restricts itself to masked peaks (`mlm_mask` key never written) and so regresses a visible input channel. Sections 1 to 5 above are the state at the end of the day; no experiment started. |
