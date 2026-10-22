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
  copied verbatim into `scripts/analysis/nine_species_score.py` (0.5 Da cumulative, 0.1 Da per residue, prefix then
  suffix), with Casanovo v5's residue masses. Both sequencers' outputs and the labels are mapped to one token set by one
  tokeniser (the fork's ProForma, Casanovo's named ProForma and the benchmark's DeepNovo spellings, numeric deltas with any
  number of decimals). Cross-check: the fork's own peptide precision on 200 mouse spectra equals the scorer's to three
  decimals (0.585). The same cross-check against Casanovo's own evaluate mode failed (U4).
- **A3. Every sampled spectrum counts.** A spectrum without a prediction (Casanovo skips spectra with fewer than 20 peaks;
  the fork drops nothing in this mode) is a miss for peptide recall, and coverage is reported beside precision.
- **A4. Decoding.** Both sequencers run greedy and with 5 beams; the 5-beam figures are the comparison (the fork's own
  baseline README runs Casanovo with 5 beams; Casanovo's paper used beam search for its benchmark figure).
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

## Method

1. Sample: `scripts/evals/nine_species_sample.py`.
2. Fork: `scripts/evals/nine_species_run_fork.sh` (greedy, then 5 beams) with `configs/inference/denovo_ninespecies.yaml`.
3. Casanovo: `scripts/evals/nine_species_run_casanovo.sh` (5 beams, and greedy), environment `envs/casanovo`.
4. Collect: `scripts/evals/nine_species_collect.py` joins each output to the manifest by position in the sample file and
   checks the fork's parsed target against the manifest label.
5. Score: `scripts/analysis/nine_species_score.py` (per species, strata, bootstrap).
6. Overlap: `scripts/analysis/nine_species_overlap.py`.

## Results

To be filled from the run directory `~/Documents/experiments/Arc/instanovo-fm-runs/nine-species/` (this section states
only numbers that the scripts above reproduce from committed outputs under `docs/nine_species/results/`).

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

### Stratified results and controls

To be filled from the runs: peptide recall by charge, length, modification presence, precursor m/z and peak count,
and the frozen-encoder and from-scratch sequencers on the same spectra.

## Review passes

Two passes after the runs, recorded in the session log: each statement above re-read against its source, each number
re-derived from the committed outputs.
