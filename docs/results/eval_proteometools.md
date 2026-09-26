# InstaNovo-FM evaluation suite on proteometools

Rendered by `scripts/evals/render_results.py` from `docs/results/rerun/proteometools/<model>/<protocol>/` (jobs of `scripts/evals/run.py`, data from `scripts/evals/prepare_dataset.py`). Reference column: the released 40M on the corpus split named in the header, from the reproduction (`docs/references/results_paper_or_rerun.md`). "-" = not run yet or the task reported nothing.

## The data

- 11 runs, 639,632 MS2 spectra, 466,891 identified with an expressible sequence, 4,873 distinct peptides (I/L collapsed, modifications dropped); probe split 372,140 / 49,878 / 44,873 spectra (train / valid / test, peptide-disjoint, seed 42).
- Instrument: Orbitrap Fusion Lumos; fragmentation: HCD; HCD|CID; detector: Orbitrap; Orbitrap|IonTrap; organism: Homo sapiens. Search engine: maxquant. Carbamidomethyl treated as fixed: True.

| run | MS2 | identified | unsupported modifications | peptides | median peaks | charges of identified |
|---|---:|---:|---:|---:|---:|---|
| 01625b_GA1-TUM_first_pool_1_01_01-3xHCD-1h-R1 | 48,417 | 35,913 | 0 | 938 | 156 | 2+ 34,258, 3+ 1,655 |
| 01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1 | 57,760 | 44,236 | 0 | 946 | 353 | 2+ 41,899, 3+ 2,336, 7+ 1 |
| 01625b_GB1-TUM_first_pool_2_01_01-3xHCD-1h-R1 | 45,687 | 33,696 | 0 | 907 | 155 | 2+ 33,455, 3+ 241 |
| 01625b_GC1-TUM_first_pool_3_01_01-3xHCD-1h-R1 | 45,756 | 34,751 | 0 | 914 | 159 | 2+ 34,305, 3+ 443, 4+ 3 |
| 01625b_GC1-TUM_first_pool_3_01_01-2xIT_2xHCD-1h-R1 | 56,336 | 42,216 | 0 | 909 | 348 | 2+ 41,573, 3+ 641, 4+ 1, 6+ 1 |
| 02208a_GA1-TUM_second_addon_1_01_01-3xHCD-1h-R1 | 55,954 | 40,190 | 0 | 854 | 87 | 1+ 3, 2+ 39,627, 3+ 560 |
| 02208a_GA1-TUM_second_addon_1_01_01-2xIT_2xHCD-1h-R1 | 66,716 | 49,195 | 0 | 864 | 215 | 1+ 4, 2+ 48,376, 3+ 815 |
| 02208a_GA2-TUM_second_addon_2_01_01-3xHCD-1h-R1 | 56,815 | 41,382 | 0 | 850 | 96 | 2+ 40,851, 3+ 531 |
| 02208a_GA2-TUM_second_addon_2_01_01-2xIT_2xHCD-1h-R1 | 75,382 | 53,673 | 0 | 861 | 212 | 1+ 2, 2+ 52,887, 3+ 764, 4+ 16, 7+ 4 |
| 02208a_GA3-TUM_second_addon_3_01_01-3xHCD-1h-R1 | 57,568 | 40,421 | 0 | 837 | 95 | 2+ 40,400, 3+ 21 |
| 02208a_GA3-TUM_second_addon_3_01_01-2xIT_2xHCD-1h-R1 | 73,241 | 51,218 | 0 | 873 | 214 | 2+ 51,172, 3+ 38, 4+ 7, 7+ 1 |

## How much of it the model trained on

Identified spectra by the corpus split of their peptide (`peptide_registry.parquet` of `InstaDeepAI/InstaNovo`, peptide-disjoint 80/10/10; `absent` = not in the corpus at all). The released models trained on the train split; a spectrum here is never itself in the corpus unless the run is.

| split | spectra | share |
|---|---:|---:|
| train | 390,015 | 83.5 % |
| validation | 30,317 | 6.5 % |
| test | 46,020 | 9.9 % |
| absent | 539 | 0.1 % |

## Read against the corpus test splits

From `docs/results/profiles/*.json` (`scripts/evals/corpus_profile.py`, metadata columns plus the package's per-file table). Percentages within each set; top values only.

| | proteometools | MCFM test | LCFM test |
|---|---|---|---|
| spectra | 639,632 | 5,761,808 | 56,831,096 |
| identified | 466,891 | 5,761,808 | 56,831,096 |
| runs / projects | 11 / 2 | 15013 / 31 | 15219 / 31 |
| fragmentation (`frag_type`) | HCD 87 %; CID 13 % | HCID 48 %; HCD 30 %; CID 13 %; (missing) 9 % | HCID 47 %; HCD 34 %; (missing) 10 %; CID 9 % |
| instrument (search data) | Orbitrap Fusion 100 % | Orbitrap Fusion 20 %; Q Exactive 19 %; Q Exactive HF 18 %; Orbitrap Fusion Lumos 14 % | Q Exactive 19 %; Orbitrap Fusion 18 %; Q Exactive HF 18 %; Orbitrap Fusion Lumos 14 % |
| detector (search data) | Orbitrap|IonTrap 52 %; Orbitrap 48 % | Orbitrap 88 %; Orbitrap|IonTrap 6 %; Astral 6 %; TOF 0 % | Orbitrap 90 %; Orbitrap|IonTrap 5 %; Astral 5 %; TOF 0 % |
| organism (search data) | Homo sapiens 100 % | Homo sapiens 70 %; Homo sapiens; Arabidopsis thaliana 11 %; Homo Sapiens 5 %; Saccharomyces cerevisiae 4 %; Mus musculus 3 % | Homo sapiens 69 %; Homo sapiens; Arabidopsis thaliana 11 %; Saccharomyces cerevisiae 5 %; Homo Sapiens 5 %; Mus musculus 3 % |
| acquisition | DDA 100 % | DDA 93 %; DIA 7 % | DDA 95 %; DIA 5 % |
| collision energy | 35.0 29 %; 25.0 16 %; 30.0 16 %; 28.0 13 %; 20.0 13 % | 27.0 22 %; 25.0 16 %; 35.0 14 %; 30.0 14 %; (missing) 9 % | 27.0 22 %; 25.0 17 %; 30.0 13 %; 35.0 11 %; 28.0 11 % |
| precursor charge | 2 98 %; 3 2 %; 4 0 %; 1 0 %; 7 0 % | 2 73 %; 3 17 %; 0 7 %; 4 2 %; 1 0 % | 2 57 %; 3 31 %; 4 6 %; 0 5 %; 5 1 % |
| modified sequences | 15.3 % | 28.7 % | 30.6 % |
| modification tokens | M[UNIMOD:35] 57 %; C[UNIMOD:4] 43 % | C[UNIMOD:4] 45 %; M[UNIMOD:35] 27 %; K[UNIMOD:259] 9 %; [UNIMOD:2016]- 4 % | C[UNIMOD:4] 39 %; M[UNIMOD:35] 34 %; K[UNIMOD:259] 9 %; [UNIMOD:737]- 3 % |
| peaks per spectrum (median, p5-p95) | 142 (29-734) | 294 (92-800) | 236 (58-800) |
| precursor m/z (median, p5-p95) | 440 (379-633) | 685 (442-1094) | 640 (417-1053) |
| peptide length (median, p5-p95) | 7 (7-11) | 13 (8-24) | 13 (8-25) |

## Results

### Linear probes

| metric | released 40M (all 44,873) | 40M trained here (all 44,873) | released 89M (all 44,873) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Fragment type (macro-F1) | 0.998 (balanced accuracy) | 0.998 (balanced accuracy) | 0.998 (balanced accuracy) | 0.733 |
| Instrument (macro-F1) | - | - | - | 0.729 |
| PTM presence (balanced accuracy) | 0.853 | 0.839 | 0.897 | 0.756 |
| Modification class (macro-F1) | 0.765 | 0.625 | 0.775 | 0.466 |
| Hydrophobicity (R²) | 0.651 | 0.639 | 0.719 | 0.528 |
| Precursor mass (R²) | 0.872 | 0.875 | 0.853 | 0.702 |
| Precursor m/z (R²) | 0.889 | 0.890 | 0.891 | 0.897 |
| Charge (macro-F1) | 0.929 (balanced accuracy) | 0.975 (balanced accuracy) | 0.940 (balanced accuracy) | 0.515 |
| Collision energy (R²) | 0.840 (macro-F1) | 0.840 (macro-F1) | 0.888 (macro-F1) | -1.678 |
| Spectrum confidence (R²) | 0.955 | 0.951 | 0.938 | 0.978 |

Probe train / valid / test are this dataset's own peptide-disjoint files (package caps 100,000 / 10,000 / 10,000). A one-class target (one instrument, one fragmentation) cannot be probed and shows "-". A cell names its metric when the probe chose another than the row label's (balanced accuracy for a two-class target, macro-F1 for a collision energy with few settings); the probe scores only the classes present in its test split.

### Duplicate-spectrum retrieval

| metric | released 40M (-) | 40M trained here (-) | released 89M (-) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Recall@1 | - | - | - | 0.305 |
| mAP@20 | - | - | - | 0.125 |
| Proportional recall@1 | - | - | - | 0.068 |
| Recall@1, HCD Orbitrap subset | - | - | - | 0.328 |
| Recall@1, CID subset | - | - | - | 0.365 |

Every identified spectrum is in the pool and every duplicate group is queried; a positive is an identical peptide string (charge ignored).

### Peak level

| metric | released 40M (10,000 of 466,891) | 40M trained here (10,000 of 466,891) | released 89M (10,000 of 466,891) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Fragment-group bin accuracy (IG task) | 70.6 % | 67.4 % | 84.0 % | 55.0 % |
| Bin accuracy y / b ions | 73.3 % / 62.7 % | 70.6 % / 58.3 % | 87.0 % / 75.4 % | 56.9 % / 50.0 % |
| Median error y / b ions | 109.1 / 236.4 ppm | 113.1 / 279.2 ppm | 101.5 / 200.4 ppm | 136.8 / 276.7 ppm |
| Peak-type accuracy | 80.0 % | 76.2 % | 80.1 % | 73.6 % |
| Peak-type macro-F1 | 0.616 | 0.588 | 0.627 | 0.519 |
| Cross-spectrum AUROC | 0.895 | 0.882 | 0.835 | 0.885 |
| Confidence AUROC, per spectrum / pooled | 0.753 / 0.753 | 0.748 / 0.750 | 0.700 / 0.691 | 0.718 / 0.705 |
| Structural attention heads | 9 | 8 | 4 | 11 |
| Isotope-spacing enrichment (mean) | 2.082 | 1.813 | 1.533 | 3.114 |

Theoretical b/y ions from the sequence with the checkpoint's residue masses; the column header says how many spectra were used.

### Embedding geometry

| metric | released 40M (20,000 of 466,891) | 40M trained here (20,000 of 466,891) | released 89M (20,000 of 466,891) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Anisotropy ratio | 23.6 | 23.3 | 26.6 | 25.0 |
| Effective rank | 72.2 | 76.6 | 36.0 | 89.4 |
| Top-component energy | 0.147 | 0.158 | 0.258 | 0.132 |
| Mean cosine | 0.725 | 0.707 | 0.919 | 0.811 |
| UMAP kNN preservation (k=15) | 0.143 | 0.137 | 0.152 | 0.098 |
| EVoC clusters / noise / purity | 6 / 0.520 / 0.996 | 8 / 0.523 / 0.986 | 8 / 0.374 / 0.996 | 17 / 0.502 / 0.811 |
| ESM2 alignment RSA / CKA | 0.042 / 0.110 | 0.032 / 0.105 | 0.041 / 0.060 | 0.048 / 0.060 |
| ESM2 metadata-baseline RSA | 0.156 | 0.156 | 0.156 | 0.198 |
| Glass Box residual / top importance | 4.20e-05 / 5.387 | 4.20e-05 / 5.387 | 4.20e-05 / 5.387 | 2.29e-05 / 3.704 |
| Cosine-hyperscore Spearman (pairs) | - | - | - | - |

Header says how many spectra were used (the paper's 20,000 above 100,000). EVoC purity is by fragmentation type and is 1 by construction when a dataset has one. The cosine-hyperscore correlation needs a hyperscore (X!Tandem or MSFragger); a dataset searched with another engine has none and shows "-". Glass Box attribution returned the same values for every checkpoint here and in the reproduction (released 40M against the one trained here); its output does not separate models.

### Every MS2 spectrum, identified or not

| metric | released 40M (100,000 of 639,632) | 40M trained here (100,000 of 639,632) | released 89M (100,000 of 639,632) |
|---|---|---|---|
| Anisotropy ratio | 23.8 | 23.6 | 26.6 |
| Effective rank | 68.3 | 72.7 | 33.7 |
| Top-component energy | 0.148 | 0.159 | 0.241 |
| Mean cosine | 0.738 | 0.723 | 0.918 |
| UMAP kNN preservation (k=15) | 0.122 | 0.122 | 0.146 |

Identified and unidentified spectra together (`dataset.is_annotated=false`); no reference column because the corpus holds identified spectra only.

### Trainer validation (masked-peak reconstruction)

| metric | released 40M (all 466,891) | 40M trained here (all 466,891) | released 89M (all 466,891) | released 40M, MCFM validation (reproduction) |
|---|---|---|---|---|
| Median |error| over masked peaks | 5718 ppm | 6859 ppm | 1201 ppm | 4493 ppm |
| MAE | 5.75 Da | 5.92 Da | 4.61 Da | 5.99 Da |
| Bin accuracy (0.2 Da) | 32.1 % | 30.2 % | 44.4 % | 27.4 % |
| Within 20 ppm | 3.2 % | 3.0 % | 4.3 % | 4.1 % |
| Intensity R² | 0.979 | 0.978 | 0.986 | 0.980 |

The trainer's validation loop on every identified spectrum (masking in the collate, seed fixed), the criterion the trainer selects checkpoints on.

## Sources

- Data: `$EXTERNAL/proteometools/` (`prepare_dataset.py`, manifest `proteomies-eval-data/outputs/fm_manifest.csv`, summary `proteometools-summary.json`).
- Runs: `docs/results/rerun/proteometools/<model>/<protocol>/run.json` (overrides, commit, host, seconds, rows in file, cap).
