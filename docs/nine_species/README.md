# InstaNovo-FM on the nine-species benchmark, against Casanovo

Branch `sdu/eval-nine-species`, started 22/10/2026. Owner's brief: an in-depth evaluation of the model on the nine-species
benchmark compared with Casanovo's numbers; every assumption and uncertainty stated; two review passes for mistakes and
scientific dishonesty; a catalogue of what breaks the model and whether each weakness is data dependent. The session log
is `docs/agent_logs/session-2026-10-22-nineSpecies.md`. Values: truth, checkability, consistency.

## What is being evaluated, and what is not

InstaNovo-FM is an encoder. The benchmark measures de novo peptide sequencing, so the object under test is the released
sequencer `instanovo-fm-denovo-v0.1.0` (the FM encoder fine-tuned with an InstaNovo decoder, `src/instanovo_fm/downstream/
de_novo_sequencing/`), and, as controls for the encoder's contribution, `-denovo-frozen` (encoder kept frozen) and
`-denovo-scratch` (no pretraining). Casanovo is run as the external reference. The embedding suite's own protocols
(probes, retrieval) are a second view of the same spectra and are reported separately when run.

## The benchmark

The revised nine-species benchmark of Yilmaz et al. 2024 (MassIVE MSV000090982): the RAW files of nine PRIDE projects
re-searched with Tide/Crux 4.1 at 1 % FDR, one MGF set per species, every spectrum carrying its peptide in `SEQ=`.
Local copy: `~/Documents/experiments/datasets/nine_species/`.

| Species | PRIDE | Spectra | Files | Instrument, fragmentation (corpus metadata) |
| :---- | :---- | ---: | ---: | :---- |
| Apis mellifera (honeybee) | PXD004467 | 194,281 | 17 | Q Exactive, HCD |
| Bacillus subtilis | PXD004565 | 1,358,337 | 93 | Q Exactive, HCD |
| Candidatus endoloripes (clam bacteria) | PXD004536 | 82,290 | 11 | Q Exactive, HCD |
| Homo sapiens | PXD004424 | 44,604 | 26 | Q Exactive, HCD |
| Methanosarcina mazei | PXD004948 | 267,332 | 72 | Orbitrap Velos, HCD |
| Mus musculus (SILAC) | PXD005025 | 25,541 | 13 | Q Exactive, HCD |
| Saccharomyces cerevisiae | PXD003868 | 585,593 | 27 | Q Exactive, HCD |
| Solanum lycopersicum (tomato) | PXD004947 | 178,413 | 60 | Q Exactive, HCD |
| Vigna mungo (rice bean) | PXD004325 | 108,514 | 24 | Q Exactive, HCD |

Spectrum counts: `BEGIN IONS` lines, counted 22/10 (`docs/agent_logs/session-2026-10-22-nineSpecies.md`). Instrument and
fragmentation: `data/search_data.xlsx` of the corpus, by project; its `organism` column is wrong for three of these
projects (see flaw F1 below), so species come from the benchmark's own folders.

Modifications in the labels (DeepNovo notation): `C+57.021` on every cysteine, `M+15.995`, `N+0.984`, `Q+0.984`,
N-terminal `+42.011`, `+43.006`, `-17.027`, and carbamylation with ammonia loss written as two stacked deltas,
`+43.006-17.027` (541 of the 180,000 sampled labels; never the combined `+25.980` the fork's residue table names). Every
label matches residue-with-optional-delta after that prefix (checked over the sample, session log 02:25).

### Overlap with InstaNovo-FM's training data (confirmed)

All nine projects are among the 92 PRIDE projects of the pretraining corpus (`assets/table_s1_accessions.txt`; the
preprint says so in its own words: "the 9 projects comprising the standard nine-species benchmark" were included). The
corpus's peptide registry (`peptide_registry.parquet`, 5,613,657 peptides with a split each) shows how that was handled:
of the benchmark's labelled peptides, 99.3 to 99.8 % per species sit in the corpus's **test** split and 99.7 to 100 % of the
spectra, against the 11 % a random assignment would give (`scripts/analysis/nine_species_overlap.py`). The training
configs read the train shards only. So:

- the benchmark's peptides were held out of pretraining and of the decoder's training, deliberately (inferred from the
  numbers; the preprint does not describe the mechanism);
- the benchmark is a subset of the paper's own test split, so its spectra were never used for model selection either;
- what remains shared with training is everything but the peptides: the same runs' other peptides, the same instruments,
  labs and acquisition settings. That is the usual position of any test split of this corpus. Casanovo's training set is described by its paper as
  "31 TB of human data from 227 public proteomics datasets" (MassIVE-KB v.2018-06-15), so it shares no instrument run with
  the eight non-human projects; for the human project the paper's text does not say (U5).

## Assumptions (A) and uncertainties (U)

- **A1. Sample, not census.** Each species is evaluated on a seeded uniform sample of 20,000 spectra
  (`scripts/evals/nine_species_sample.py`, seed 20261022; 180,000 spectra in all; `summary.json` and `manifest.parquet`
  under the run directory). Published numbers are on every spectrum. Confidence intervals come from a bootstrap that
  resamples peptides, because spectra of one peptide are not independent.
- **A2. The metric is Casanovo's.** Amino-acid and peptide matches follow `aa_match_prefix_suffix` of Casanovo v5.0.0,
  copied into `scripts/analysis/nine_species_score.py` with only docstrings and type annotations removed (0.5 Da
  cumulative, 0.1 Da per residue, prefix then suffix), with Casanovo v5's residue masses. Both sequencers' outputs and the labels are mapped to one token set by one
  tokeniser (the fork's ProForma, Casanovo's named ProForma and the benchmark's DeepNovo spellings, numeric deltas with any
  number of decimals). Cross-check: the fork's own peptide precision on 200 mouse spectra equals the scorer's to three
  decimals (0.585). The same cross-check against Casanovo's own evaluate mode failed (U4).
- **A3. Every sampled spectrum counts.** A spectrum without a prediction (Casanovo skips spectra with fewer than 20 peaks;
  the fork drops nothing in this mode) is a miss for peptide recall, and coverage is reported beside precision.
- **A4. Decoding.** Both sequencers run greedy and with 5 beams (the fork's own baseline README runs Casanovo with 5
  beams; Casanovo's paper used beam search for its benchmark figure). On this GPU the fork's beam search runs at 15 s per
  128 spectra, forty minutes per species, so its 5-beam run is made on the first 5,000 sampled spectra of each species
  (`samples-5k`, a prefix of the uniform sample and so itself one) and compared with Casanovo's 5-beam output on those
  same 45,000 spectra (`--first-per-species 5000` in the scorer); the full-sample 5-beam run is queued last and added
  if it completes. The full-sample comparison is the fork's greedy run against Casanovo's 5 beams, stated as such.
- **A5. Casanovo's numbers.** The primary comparison is Casanovo v5.1.2 with the v5.0.0 checkpoint run here on the same
  spectra. Published per-species numbers (Yilmaz et al. 2024, Supplementary Figure S1) are in a figure, not a table, and
  are reported as read from the paper's text only where the text states them.
- **U1.** Whether the preprint's "held out" claim for the nine projects applies at the peptide level is inferred from the
  registry, not stated by the authors.
- **U2.** The benchmark MGFs use `SEQ=` with the DeepNovo spellings; the fork's reader remaps them only through the
  configuration added here (`configs/inference/denovo_ninespecies.yaml`); the stacked `+43.006-17.027` label is parsed by the fork
  as two N-terminal tokens, its sequencer cannot emit the combined modification, and the scorer merges stacked N-terminal
  deltas by mass into one token, so those 541 spectra count as misses for the fork and are reachable for Casanovo.
- **U3.** The scorer's precision-coverage area is the trapezoid integral of precision over coverage, coverage counted
  over all sampled spectra, from the first ranked spectrum to the last one with a prediction; a sequencer that skips
  spectra loses that area. That follows the paper's definition ("Average precision (AP) corresponds to the area under the
  precision-coverage curve", Figure 2 caption) but not necessarily its integration routine, which was not available, so
  published AP values are context, not a numerical comparison.
- **U4.** Casanovo v5.1.2's own evaluate mode, run on the 200 mouse spectra relabelled into its named alphabet, reported
  amino-acid precision 80 %, amino-acid recall 28 % and peptide precision 0 %, which its own predictions do not support
  (scored with the shared rule they reach peptide recall 0.40). Its target side seems to be tokenised per character. Not
  debugged; no number from that mode is used, so Casanovo's side of the comparison rests on the shared scorer alone.
- **U5.** Whether Casanovo's training set contains spectra or peptides of the benchmark's human project (PXD004424) is
  not addressed in the paper's text fetched 22/10 (see the overlap section).

## Flaws found on the way (named, not accommodated)

- **F1. Wrong organisms in the corpus metadata.** `data/search_data.xlsx` labels PXD004325 (Vigna mungo) "Methanosarcina
  mazei", PXD004948 (M. mazei) "Mus musculus" and PXD005025 (Mus musculus) "Vigna radiata"; it also lists PXD004565 as DIA.
  Any organism analysis from this table is wrong for these projects. Not fixed here (the table is the authors' artefact);
  species in this study come from the benchmark.
- **F2. The sequencer drops labelled spectra silently in evaluation mode.** Labels in a spelling outside its remapping
  table are "unsupported residues" and the rows are dropped from its metrics (57 of 200 mouse spectra, 28 %), with a
  warning only. Its built-in precision on such a file is a number on a biased subset.
- **F3. Two label classes are one isotope away from an unmodified reading.** Deamidation (+0.984 Da) and a one-isotope
  precursor error (+1.003 Da) differ by 19 mDa; N-terminal carbamyl (+43.006) and acetyl plus one isotope (+43.014) by
  8 mDa. The benchmark's search allowed one isotope error and chose the modified form; a mass-constrained decoder that
  also allows one isotope error may legitimately choose the plain or acetylated one, and the match rule then counts it
  wrong. Deamidation-only labels are 8.3 % of the sample and carbamyl-only 1.9 %, so this bounds how much of the
  "modified" penalty is the model's (session log 03:20 for what the fork predicts on them). Both sequencers face it.

## Method

1. Sample: `scripts/evals/nine_species_sample.py`.
2. Fork: `scripts/evals/nine_species_run_fork.sh` (greedy, then 5 beams) with `configs/inference/denovo_ninespecies.yaml`.
3. Casanovo: `scripts/evals/nine_species_run_casanovo.sh` (5 beams, and greedy), environment `envs/casanovo`.
4. Collect: `scripts/evals/nine_species_collect.py` joins each output to the manifest by position in the sample file and
   checks the fork's parsed target against the manifest label. Casanovo's peptide is read from its mzTab's
   `opt_ms_run[1]_proforma` column: the `sequence` column is the bare peptide (mzTab moves modifications to their own
   column), and scoring it would strip carbamidomethyl from every cysteine (the mistake behind the smoke figures, log 03:30).
5. Score: `scripts/analysis/nine_species_score.py` (per species, strata, bootstrap).
6. Overlap: `scripts/analysis/nine_species_overlap.py`.

## Results

Every number here is read from files under `docs/nine_species/results/`: `results.md` (rendered by
`scripts/analysis/nine_species_table.py` from `results/scores/*.json`, one file per run, per species with bootstrap
intervals and strata), `standardised.md`, `pairwise-*.md`, and the session log for the run times. Runs scored so far,
each on all 180,000 sampled spectra: the fork greedy, Casanovo greedy, Casanovo with 5 beams; the frozen and scratch
controls on the 45,000-spectrum prefix. Pending: the fork with 5 beams on that prefix (A4).

Pooled over the nine species (`results.md`):

| run | coverage | peptide recall | amino-acid precision | precision-coverage area | predictions off the precursor mass |
| :---- | ---: | ---: | ---: | ---: | ---: |
| fork, greedy | 1.000 | 0.642 | 0.820 | 0.883 | 13.7 % |
| Casanovo v5.0.0, greedy | 0.994 | 0.619 | 0.811 | 0.891 | 16.2 % |
| Casanovo v5.0.0, 5 beams | 0.994 | 0.649 | 0.835 | 0.898 | 7.7 % |

Under matched greedy decoding the fork leads Casanovo by 2.3 points of peptide recall pooled and in every species but
tomato (where they tie), with fewer off-mass predictions; Casanovo's 5 beams add 3 points of recall and halve its
off-mass share, which is what the fork's pending 5-beam run is for.

Per species the sequencers rank the species identically (rice bean, yeast and tomato at the top near 0.70 to 0.73,
C. endoloripes at the bottom near 0.51, the SILAC mouse set next at 0.55), and their peptide recalls lie within each
other's bootstrap interval except B. subtilis and yeast (Casanovo ahead by about two points) and rice bean (the fork
ahead by about 1.6). Casanovo's published figure for this benchmark, "the average precision of Casanovo increases from
0.83 to 0.95" when trained on MassIVE-KB (Yilmaz et al. 2024, Figure 2B, beam search, every spectrum, their model
version and integration routine), is context only: the measured area here is 0.898 for the v5.0.0 checkpoint on the
sample, under the area definition of U3, and the two are not compared numerically (A5, U3).

## What breaks the model, and whether it is the data

### The model's input contract (data independent, from the configuration)

Read from `configs/model/foundation_base.yaml`, `configs/inference/denovo.yaml` and `configs/residues/default.yaml`.

| Assumption | Value | What happens outside it |
| :---- | :---- | :---- |
| Peaks per spectrum | the 200 most intense after dropping intensity below 1 % of the base peak | the rest of the spectrum is never seen |
| m/z range | 50 to 2,500 Da, binned at 0.2 Da for the masked-peak objective | peaks outside are dropped |
| Precursor charge | 1 to 10 | higher charges are not representable |
| Peptide length (sequencer) | at most 40 residues | longer peptides cannot be produced |
| Vocabulary (sequencer) | 20 residues, C carbamidomethyl, M oxidation, N and Q deamidation, R citrullination, P hydroxylation, S/T/Y phosphorylation, N-terminal acetyl, carbamyl and ammonia loss | any other modification is unsayable; the combined carbamyl plus ammonia loss (`+25.980`) is explicitly suppressed |
| Label spelling (evaluation mode) | the fork's remapping table | other spellings drop the row from its own metrics silently (F2) |
| Precursor mass | the decoder extends a sequence only with residues that fit the remaining mass; when none fits, the one unmasked token (padding) is chosen, the sequence ends, and the score adds that token's unmasked log-probability (InstaNovo's `inference/greedy_search.py`, mass-constrained step; read 22/10) | an off-mass prediction is a dead end of the search, scored hundreds of nats below any complete one, so it ranks last; de novo mode also drops predictions beyond `filter_precursor_ppm` 20, evaluation mode keeps them with the deviation as a column |
| Bare cysteine | `C` and `C[UNIMOD:4]` are both tokens | a bare-C prediction against a carbamidomethylated label is wrong by 57 Da |

Casanovo v5, for contrast: 20 to 150 peaks (fewer than 20 skips the spectrum), m/z 50 to 2,500, charge at most 4,
peptide length 6 to 100, every benchmark modification in its alphabet, a 50 ppm precursor check folded into the score's sign.

### How much of the benchmark the contract touches (data dependent, measured)

From `scripts/analysis/nine_species_constraints.py` on the 180,000 sampled spectra
(`results/constraints.json`; shares of spectra):

| Species | peaks over 200 (fork keeps 200) | peaks over 150 (Casanovo keeps 150) | m/z over 2,500 | charge over 4 | length over 40 | label modified | median peaks |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apis mellifera | 59.7 % | 72.0 % | 1.5 % | 0.2 % | 0.2 % | 30.6 % | 244 |
| Bacillus subtilis | 78.5 % | 87.7 % | 1.8 % | 0.2 % | 0.4 % | 28.4 % | 334 |
| Candidatus endoloripes | 9.2 % | 17.5 % | 5.3 % | 0.6 % | 0.4 % | 36.2 % | 79 |
| Homo sapiens | 32.4 % | 45.7 % | 3.1 % | 0.6 % | 0.2 % | 36.6 % | 137 |
| Methanosarcina mazei | 70.5 % | 82.1 % | 0.5 % | 0.1 % | 0.0 % | 26.8 % | 293 |
| Mus musculus | 1.5 % | 6.0 % | 0.6 % | 0.0 % | 0.0 % | 36.7 % | 64 |
| Saccharomyces cerevisiae | 96.3 % | 98.3 % | 4.9 % | 0.3 % | 0.4 % | 29.6 % | 520 |
| Solanum lycopersicum | 21.9 % | 36.2 % | 3.0 % | 0.7 % | 0.2 % | 32.2 % | 117 |
| Vigna mungo | 79.8 % | 90.2 % | 1.1 % | 0.2 % | 0.4 % | 23.2 % | 348 |
| all | 50.0 % | 59.5 % | 2.4 % | 0.3 % | 0.2 % | 31.2 % | 200 |

No sampled spectrum has fewer than 20 peaks in the MGF, no charge exceeds 10, and 0.3 % of labels carry the
carbamylation-with-ammonia-loss state the fork cannot emit.
Casanovo's 20-peak floor applies after its own filtering (intensity under 1 % of the base peak, m/z outside 50 to 2,500
and peaks within 2 Da of the precursor removed), which is why it skipped 4 of the 200 smoke spectra; its coverage in the
results is the measured share. The one constraint that bites is the peak cap: the benchmark's MGFs are raw peak lists, and for yeast,
rice bean, B. subtilis and M. mazei the sequencer sees the 200 most intense of 300 to 520 peaks. Whether that costs
accuracy is a data-dependent question the stratified results below answer (recall by peak count).

### What the authors claim, and do not (preprint v2, fetched 22/10)

The preprint claims no organism, species or tissue signal in the embeddings. What it does claim about acquisition
context: "Embeddings separate by fragmentation type into three well-defined lobes corresponding to HCD/HCID, CID, and
ETD", and "Instrument identity is linearly decodable at macro-F1 0.804 when projects are shared between the probe's
training and test splits, but falls to 0.110 when projects are held disjoint", which it reads as "project-level
acquisition context rather than instrument physics". A species probe on this benchmark would inherit that confound
exactly (one project per species), so none is run here.

### Stratified results: what is in the data and what is in the model

Read from the `by` tables of `results/scores/*.json` (pooled in `results.md`), `results/standardised.md` and
`results/pairwise-fork-greedy-vs-casanovo-beam5.md`; the fork is its greedy run until the 5-beam run is added.

**Shared by both sequencers and by every species (a property of the spectra, not of InstaNovo-FM):**

- Peptide recall falls monotonically with peptide length (pooled, fork: 86 % at 9 residues and under to 12 % at 30 and
  over; Casanovo 92 % to 8 %), with precursor charge (2 then 3 then 4 and above) and with precursor m/z, in every one of
  the nine species.
- Spectra with fewer than 50 peaks are the hardest group everywhere: 29.3 % pooled for both sequencers, against 62 to
  75 % for richer spectra. Above that, peak count does not matter: spectra with more than 200 peaks, where the fork sees
  only the 200 most intense and Casanovo the 150 most intense, score 68.7 % and 68.4 %. The peak cap of the contract
  table costs nothing visible.
- Labels with a modification are harder for both (fork 40.5 % against 75.0 % unmodified; Casanovo 48.7 % against 72.2 %).
- The species ordering is the same for both, and standardising each species to the pooled peptide composition (length,
  charge, modification) widens rather than closes the spread for both (`standardised.md`: fork 0.204 crude to 0.237
  standardised; Casanovo 0.221 to 0.270). The species differences are therefore in what each project's spectra look like
  (instrument settings, fragmentation quality, sample preparation), not in which peptides it contains, and they hit both
  sequencers alike.
- Paired spectrum by spectrum, 87 % of what either sequencer gets right the other gets right too (`pairwise-*.md`, 0.85 to
  0.92 in every species); 27.2 % of spectra defeat both.

**Where the fork and Casanovo differ on the same spectra (a property of the model):**

- The fork is better on unmodified labels (75.0 % against 72.2 %; it alone gets 10,679 of them, Casanovo alone 7,237), on
  oxidised methionine (62.4 % against 57.8 %), at charge 3 and above and at length 20 and above.
- The fork fails N-terminal carbamylation: 1.3 % against Casanovo's 56.9 % on 3,500 spectra. Its vocabulary has the
  token; on these spectra it emits acetyl (one isotope away in mass) in 89 % of its residue-correct predictions and
  dead-ends in 32 %. Casanovo resolves the same spectra, so this is not the benchmark's ambiguity (F3) at work but the
  fork's decoder or its training distribution: carbamylation is a sample-preparation artefact concentrated in particular
  projects, so the weakness is data dependent in the training sense and will reappear on any urea-treated sample.
- Deamidation defeats both (fork 14.2 %, Casanovo 34.0 % on 14,855 spectra), the fork more: it emits the plain peptide one
  isotope below the precursor in a third of cases and dead-ends in 43 %. F3 bounds how much of this is the benchmark.
- The combined carbamyl plus ammonia loss is never recovered by either (0 of 275; the fork cannot say it, Casanovo has
  the token and still misses).
- The fork's greedy search dead-ends on 13.7 % of spectra against Casanovo's 7.7 % off-mass predictions, and none of the
  fork's dead ends is a correct peptide. How much of that is greedy decoding rather than the model is answered by the
  fork's 5-beam run.

### Controls: what the pretrained encoder contributes

The fork releases three sequencers trained with the same budget (2.5M steps at batch 128, its README): the published one
(encoder fine-tuned from step 100K), one with the encoder frozen, and one trained from scratch with no pretraining. All
three were run greedy on the first 5,000 sampled spectra of each species (45,000 in all, `samples-5k/`), and the
full-sample runs were re-scored on those same spectra (`results-prefix5k.md`, `results/scores-prefix5k/`,
`results/pairwise-prefix5k-*.md`, `results/standardised-prefix5k.md`):

| sequencer, greedy, 45,000 spectra | peptide recall | amino-acid precision | precision-coverage area | off-mass |
| :---- | ---: | ---: | ---: | ---: |
| published (encoder fine-tuned) | 0.650 | 0.825 | 0.887 | 13.0 % |
| from scratch (no pretraining) | 0.648 | 0.823 | 0.887 | 13.1 % |
| encoder frozen | 0.573 | 0.777 | 0.846 | 17.9 % |
| Casanovo greedy, same spectra | 0.626 | 0.814 | 0.894 | 15.6 % |
| Casanovo 5 beams, same spectra | 0.655 | 0.839 | 0.901 | 7.5 % |

- **Pretraining adds nothing measurable to de novo sequencing on this benchmark once the encoder is fine-tuned.** The
  from-scratch sequencer matches the published one pooled and in every species within the bootstrap intervals, and the
  two differ spectrum by spectrum only symmetrically (the published one alone right on 2,172 spectra, the scratch one
  alone on 2,081, of 45,000). The ranking of species, the strata gradients and the modification failures (carbamyl 1.0 %
  against 1.1 %, deamidation 16.1 % against 16.8 %) are the same for both, so those weaknesses belong to the InstaNovo
  decoder and the de novo training data, not to the foundation encoder.
- **The frozen encoder is a worse sequencer than no pretraining at all**: 7.7 points of peptide recall below the
  published one (88 % of it, consistent with the fork README's "~85 %") and 7.5 below scratch, with the largest losses on
  modified labels (pyro-glu 22.5 % against 38.1 %) and peak-poor spectra. A representation learned by masked-peak
  prediction is not, by itself, the representation a sequencer wants.
- The preprint (v2, fetched 22/10) reports the frozen figure ("with the encoder held frozen it retains on average 85% of
  the fine-tuned peptide recall") and no number for the from-scratch sequencer it released; its de novo comparison with
  Casanovo and InstaNovo v1.2 is on six other held-out datasets (GluC, snake venoms, TPL antibodies, HeLa QC, S. brodae,
  wound fluids), not on this benchmark.
- For the user's purpose (building on InstaNovo-FM): on de novo sequencing of these nine projects the encoder is a
  starting point no better than random initialisation. Whether it helps on smaller fine-tuning sets, or on the probing
  and retrieval tasks the paper is about, is not tested here.

## Review passes

Two passes after the runs, recorded in the session log: each statement above re-read against its source, each number
re-derived from the committed outputs.
