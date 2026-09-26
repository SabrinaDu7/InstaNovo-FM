# InstaNovo-FM evaluation suite on ups1

Rendered by `scripts/evals/render_results.py` from `docs/results/rerun/ups1/<model>/<protocol>/` (jobs of `scripts/evals/run.py`, data from `scripts/evals/prepare_dataset.py`). Reference column: the released 40M on the corpus split named in the header, from the reproduction (`docs/references/results_paper_or_rerun.md`). "-" = not run yet or the task reported nothing.

## The data

- 3 runs, 115,758 MS2 spectra, 27,023 identified with an expressible sequence, 5,941 distinct peptides (I/L collapsed, modifications dropped); probe split 21,293 / 3,141 / 2,589 spectra (train / valid / test, peptide-disjoint, seed 42).
- Instrument: LTQ Orbitrap Velos; fragmentation: CID; detector: IonTrap; organism: Saccharomyces cerevisiae; Homo sapiens. Search engine: mascot_dat. Carbamidomethyl treated as fixed: True.

| run | MS2 | identified | unsupported modifications | peptides | median peaks | charges of identified |
|---|---:|---:|---:|---:|---:|---|
| UPS1_50amol_R1 | 36,443 | 8,846 | 0 | 4,434 | 655 | 1+ 1, 2+ 5,758, 3+ 2,848, 4+ 225, 5+ 14 |
| UPS1_2500amol_R1 | 37,482 | 8,891 | 0 | 3,402 | 669 | 2+ 6,131, 3+ 2,627, 4+ 121, 5+ 12 |
| UPS1_50000amol_R1 | 41,833 | 9,286 | 0 | 3,766 | 651 | 2+ 6,532, 3+ 2,588, 4+ 157, 5+ 9 |

## How much of it the model trained on

Identified spectra by the corpus split of their peptide (`peptide_registry.parquet` of `InstaDeepAI/InstaNovo`, peptide-disjoint 80/10/10; `absent` = not in the corpus at all). The released models trained on the train split; a spectrum here is never itself in the corpus unless the run is.

| split | spectra | share |
|---|---:|---:|
| train | 2,598 | 9.6 % |
| validation | 309 | 1.1 % |
| test | 23,710 | 87.7 % |
| absent | 406 | 1.5 % |

## Read against the corpus test splits

From `docs/results/profiles/*.json` (`scripts/evals/corpus_profile.py`, metadata columns plus the package's per-file table). Percentages within each set; top values only.

| | ups1 | MCFM test | LCFM test |
|---|---|---|---|
| spectra | 115,758 | 5,761,808 | 56,831,096 |
| identified | 27,023 | 5,761,808 | 56,831,096 |
| runs / projects | 3 / 1 | 15013 / 31 | 15219 / 31 |
| fragmentation (`frag_type`) | CID 100 % | HCID 48 %; HCD 30 %; CID 13 %; (missing) 9 % | HCID 47 %; HCD 34 %; (missing) 10 %; CID 9 % |
| instrument (search data) | LTQ Orbitrap Velos 100 % | Orbitrap Fusion 20 %; Q Exactive 19 %; Q Exactive HF 18 %; Orbitrap Fusion Lumos 14 % | Q Exactive 19 %; Orbitrap Fusion 18 %; Q Exactive HF 18 %; Orbitrap Fusion Lumos 14 % |
| detector (search data) | IonTrap 100 % | Orbitrap 88 %; Orbitrap|IonTrap 6 %; Astral 6 %; TOF 0 % | Orbitrap 90 %; Orbitrap|IonTrap 5 %; Astral 5 %; TOF 0 % |
| organism (search data) | Saccharomyces cerevisiae; Homo sapiens 100 % | Homo sapiens 70 %; Homo sapiens; Arabidopsis thaliana 11 %; Homo Sapiens 5 %; Saccharomyces cerevisiae 4 %; Mus musculus 3 % | Homo sapiens 69 %; Homo sapiens; Arabidopsis thaliana 11 %; Saccharomyces cerevisiae 5 %; Homo Sapiens 5 %; Mus musculus 3 % |
| acquisition | DDA 100 % | DDA 93 %; DIA 7 % | DDA 95 %; DIA 5 % |
| collision energy | 30.0 100 % | 27.0 22 %; 25.0 16 %; 35.0 14 %; 30.0 14 %; (missing) 9 % | 27.0 22 %; 25.0 17 %; 30.0 13 %; 35.0 11 %; 28.0 11 % |
| precursor charge | 2 59 %; 3 34 %; 4 6 %; 5 1 %; 6 0 % | 2 73 %; 3 17 %; 0 7 %; 4 2 %; 1 0 % | 2 57 %; 3 31 %; 4 6 %; 0 5 %; 5 1 % |
| modified sequences | 11.3 % | 28.7 % | 30.6 % |
| modification tokens | C[UNIMOD:4] 80 %; M[UNIMOD:35] 12 %; [UNIMOD:1]- 8 % | C[UNIMOD:4] 45 %; M[UNIMOD:35] 27 %; K[UNIMOD:259] 9 %; [UNIMOD:2016]- 4 % | C[UNIMOD:4] 39 %; M[UNIMOD:35] 34 %; K[UNIMOD:259] 9 %; [UNIMOD:737]- 3 % |
| peaks per spectrum (median, p5-p95) | 658 (368-958) | 294 (92-800) | 236 (58-800) |
| precursor m/z (median, p5-p95) | 609 (382-1049) | 685 (442-1094) | 640 (417-1053) |
| peptide length (median, p5-p95) | 14 (8-27) | 13 (8-24) | 13 (8-25) |

## Results

### Linear probes

| metric | released 40M (all 2,589) | 40M trained here (all 2,589) | released 89M (all 2,589) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Fragment type (macro-F1) | - | - | - | 0.733 |
| Instrument (macro-F1) | - | - | - | 0.729 |
| PTM presence (balanced accuracy) | 0.688 | 0.651 | 0.706 | 0.756 |
| Modification class (macro-F1) | 0.298 | 0.273 | 0.456 | 0.466 |
| Hydrophobicity (R²) | 0.506 | 0.440 | 0.565 | 0.528 |
| Precursor mass (R²) | 0.920 | 0.912 | 0.936 | 0.702 |
| Precursor m/z (R²) | 0.982 | 0.980 | 0.982 | 0.897 |
| Charge (macro-F1) | 0.789 | 0.793 | 0.912 | 0.515 |
| Collision energy (R²) | - | - | - | -1.678 |
| Spectrum confidence (R²) | 0.952 | 0.940 | 0.920 | 0.978 |

Probe train / valid / test are this dataset's own peptide-disjoint files (package caps 100,000 / 10,000 / 10,000). A one-class target (one instrument, one fragmentation) cannot be probed and shows "-". A cell names its metric when the probe chose another than the row label's (balanced accuracy for a two-class target, macro-F1 for a collision energy with few settings); the probe scores only the classes present in its test split.

### Duplicate-spectrum retrieval

| metric | released 40M (all 27,023) | 40M trained here (all 27,023) | released 89M (all 27,023) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Recall@1 | 0.933 | 0.930 | 0.862 | 0.305 |
| mAP@20 | 0.745 | 0.742 | 0.597 | 0.125 |
| Proportional recall@1 | 0.214 | 0.213 | 0.183 | 0.068 |
| Recall@1, HCD Orbitrap subset | - | - | - | 0.328 |
| Recall@1, CID subset | 0.933 | 0.930 | 0.862 | 0.365 |

Every identified spectrum is in the pool and every duplicate group is queried; a positive is an identical peptide string (charge ignored).

### Peak level

| metric | released 40M (10,000 of 27,023) | 40M trained here (10,000 of 27,023) | released 89M (10,000 of 27,023) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Fragment-group bin accuracy (IG task) | 38.4 % | 28.6 % | 46.9 % | 55.0 % |
| Bin accuracy y / b ions | 43.1 % / 33.1 % | 34.3 % / 22.3 % | 51.7 % / 41.6 % | 56.9 % / 50.0 % |
| Median error y / b ions | 251.6 / 1051.3 ppm | 1195.5 / 3600.9 ppm | 130.4 / 258.7 ppm | 136.8 / 276.7 ppm |
| Peak-type accuracy | 74.6 % | 73.4 % | 76.6 % | 73.6 % |
| Peak-type macro-F1 | 0.515 | 0.504 | 0.535 | 0.519 |
| Cross-spectrum AUROC | 0.955 | 0.944 | 0.912 | 0.885 |
| Confidence AUROC, per spectrum / pooled | 0.782 / 0.786 | 0.768 / 0.775 | 0.686 / 0.679 | 0.718 / 0.705 |
| Structural attention heads | 11 | 11 | 8 | 11 |
| Isotope-spacing enrichment (mean) | 3.989 | 3.462 | 2.213 | 3.114 |

Theoretical b/y ions from the sequence with the checkpoint's residue masses; the column header says how many spectra were used.

### Embedding geometry

| metric | released 40M (all 27,023) | 40M trained here (all 27,023) | released 89M (all 27,023) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Anisotropy ratio | 26.4 | 26.5 | 27.4 | 25.0 |
| Effective rank | 57.2 | 58.6 | 32.0 | 89.4 |
| Top-component energy | 0.218 | 0.203 | 0.298 | 0.132 |
| Mean cosine | 0.908 | 0.914 | 0.976 | 0.811 |
| UMAP kNN preservation (k=15) | 0.205 | 0.212 | 0.179 | 0.098 |
| EVoC clusters / noise / purity | 7 / 0.462 / 1.000 | 10 / 0.571 / 1.000 | 7 / 0.518 / 1.000 | 17 / 0.502 / 0.811 |
| ESM2 alignment RSA / CKA | 0.104 / 0.165 | 0.081 / 0.155 | 0.091 / 0.150 | 0.048 / 0.060 |
| ESM2 metadata-baseline RSA | 0.236 | 0.236 | 0.236 | 0.198 |
| Glass Box residual / top importance | 4.58e-05 / 5.158 | 4.58e-05 / 5.158 | 4.58e-05 / 6.415 | 2.29e-05 / 3.704 |
| Cosine-hyperscore Spearman (pairs) | - | - | - | - |

Header says how many spectra were used (the paper's 20,000 above 100,000). EVoC purity is by fragmentation type and is 1 by construction when a dataset has one. The cosine-hyperscore correlation needs a hyperscore (X!Tandem or MSFragger); a dataset searched with another engine has none and shows "-". Glass Box attribution returned the same values for every checkpoint here and in the reproduction (released 40M against the one trained here); its output does not separate models.

### Every MS2 spectrum, identified or not

| metric | released 40M (100,000 of 115,758) | 40M trained here (100,000 of 115,758) | released 89M (100,000 of 115,758) |
|---|---|---|---|
| Anisotropy ratio | 26.3 | 26.4 | 27.3 |
| Effective rank | 62.1 | 63.0 | 33.9 |
| Top-component energy | 0.192 | 0.180 | 0.253 |
| Mean cosine | 0.902 | 0.907 | 0.971 |
| UMAP kNN preservation (k=15) | 0.152 | 0.161 | 0.153 |

Identified and unidentified spectra together (`dataset.is_annotated=false`); no reference column because the corpus holds identified spectra only.

### Trainer validation (masked-peak reconstruction)

| metric | released 40M (all 27,023) | 40M trained here (all 27,023) | released 89M (all 27,023) | released 40M, MCFM validation (reproduction) |
|---|---|---|---|---|
| Median |error| over masked peaks | 5506 ppm | 5995 ppm | 4075 ppm | 4493 ppm |
| MAE | 6.86 Da | 7.02 Da | 6.18 Da | 5.99 Da |
| Bin accuracy (0.2 Da) | 13.8 % | 12.4 % | 18.9 % | 27.4 % |
| Within 20 ppm | 2.7 % | 2.4 % | 3.6 % | 4.1 % |
| Intensity R² | 0.978 | 0.980 | 0.985 | 0.980 |

The trainer's validation loop on every identified spectrum (masking in the collate, seed fixed), the criterion the trainer selects checkpoints on.

## Sources

- Data: `$EXTERNAL/ups1/` (`prepare_dataset.py`, manifest `proteomies-eval-data/outputs/fm_manifest.csv`, summary `ups1-summary.json`).
- Runs: `docs/results/rerun/ups1/<model>/<protocol>/run.json` (overrides, commit, host, seconds, rows in file, cap).
