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
N-terminal `+42.011`, `+43.006`, `-17.027`, and `+25.980` (carbamylation with ammonia loss).

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
  labs and acquisition settings. That is the usual position of any test split of this corpus, and it is more than Casanovo
  shares with the benchmark (Casanovo trained on MassIVE-KB, human-only, peptide-disjoint from the benchmark).

## Assumptions (A) and uncertainties (U)

- **A1. Sample, not census.** Each species is evaluated on a seeded uniform sample of 20,000 spectra
  (`scripts/evals/nine_species_sample.py`, seed 20261022; 180,000 spectra in all; `summary.json` and `manifest.parquet`
  under the run directory). Published numbers are on every spectrum. Confidence intervals come from a bootstrap that
  resamples peptides, because spectra of one peptide are not independent.
- **A2. The metric is Casanovo's.** Amino-acid and peptide matches follow `aa_match_prefix_suffix` of Casanovo v5.0.0,
  copied verbatim into `scripts/analysis/nine_species_score.py` (0.5 Da cumulative, 0.1 Da per residue, prefix then
  suffix), with Casanovo v5's residue masses. Both sequencers' outputs and the labels are mapped to one token set by one
  tokeniser (the fork's ProForma, Casanovo's named ProForma and the benchmark's DeepNovo spellings, numeric deltas with any
  number of decimals). Cross-checks: the fork's own peptide precision on 200 mouse spectra equals the scorer's to three
  decimals (0.585); Casanovo's own evaluate mode is run on a relabelled file for the same purpose.
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
  configuration added here (`configs/inference/denovo_ninespecies.yaml`); the label `+25.980` is unsupported by the fork's
  sequencer and counts as a miss.
- **U3.** The scorer's precision-coverage area integrates the curve over coverage with the trapezoid rule; Casanovo's
  "average precision" may be computed differently and is not compared numerically.

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

To be filled: constraints audit (peaks kept, m/z range, charge, peptide length, vocabulary), stratified results, and the
frozen and from-scratch controls.

## Review passes

Two passes after the runs, recorded in the session log: each statement above re-read against its source, each number
re-derived from the committed outputs.
