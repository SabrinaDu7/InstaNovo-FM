# InstaNovo-FM evaluation suite on campi

Rendered by `scripts/evals/render_results.py` from `docs/results/rerun/campi/<model>/<protocol>/` (jobs of `scripts/evals/run.py`, data from `scripts/evals/prepare_dataset.py`). Reference column: the released 40M on the corpus split named in the header, from the reproduction (`docs/references/results_paper_or_rerun.md`). "-" = not run yet or the task reported nothing.

## The data

- 21 runs, 2,424,154 MS2 spectra, 802,128 identified with an expressible sequence, 66,982 distinct peptides (I/L collapsed, modifications dropped); probe split 644,495 / 80,636 / 76,997 spectra (train / valid / test, peptide-disjoint, seed 42).
- Instrument: Q Exactive HF; Q Exactive HF-X; Q Exactive Plus; timsTOF Pro; fragmentation: CID; HCD; detector: Orbitrap; TOF; organism: Anaerostipes caccae; Bacteroides thetaiotaomicron; Bifidobacterium longum; Blautia producta; Clostridium butyricum; Escherichia coli; Lactiplantibacillus plantarum; Thomasclavelia ramosa. Search engine: tandem_xml. Carbamidomethyl treated as fixed: False.

| run | MS2 | identified | unsupported modifications | peptides | median peaks | charges of identified |
|---|---:|---:|---:|---:|---:|---|
| S01 | 108,486 | 24,006 | 4 | 15,664 | 39 | 2+ 11,911, 3+ 9,976, 4+ 2,119 |
| S02 | 109,720 | 21,532 | 2 | 14,044 | 33 | 2+ 10,335, 3+ 9,296, 4+ 1,901 |
| S03 | 157,489 | 65,437 | 7 | 22,930 | 69 | 2+ 31,154, 3+ 27,656, 4+ 6,627 |
| S04 | 158,884 | 65,877 | 9 | 22,482 | 71 | 2+ 31,483, 3+ 27,645, 4+ 6,749 |
| S05 | 240,740 | 87,775 | 13 | 25,739 | 78 | 2+ 42,479, 3+ 36,189, 4+ 9,107 |
| S06 | 242,948 | 89,137 | 13 | 25,912 | 79 | 2+ 43,482, 3+ 36,666, 4+ 8,989 |
| S07 | 46,847 | 15,165 | 11 | 11,707 | 257 | 2+ 8,925, 3+ 5,376, 4+ 864 |
| S08 | 81,654 | 40,984 | 19 | 16,411 | 121 | 2+ 27,509, 3+ 12,141, 4+ 1,334 |
| S09 | 48,312 | 23,887 | 4 | 15,154 | 219 | 2+ 14,520, 3+ 9,367 |
| S10 | 89,607 | 42,728 | 11 | 20,568 | 211 | 2+ 24,374, 3+ 18,354 |
| S11_Fraction1 | 44,317 | 21,141 | 5 | 12,993 | 180 | 2+ 12,523, 3+ 8,618 |
| S11_Fraction2 | 46,124 | 21,716 | 4 | 13,216 | 218 | 2+ 12,206, 3+ 9,510 |
| S11_Fraction3 | 47,520 | 22,862 | 4 | 13,681 | 229 | 2+ 13,494, 3+ 9,368 |
| S11_Fraction4 | 38,254 | 15,536 | 11 | 8,069 | 126 | 2+ 8,223, 3+ 7,313 |
| S12 | 67,751 | 16,596 | 1 | 12,695 | 52 | 2+ 10,785, 3+ 5,202, 4+ 609 |
| S13_Rep1 | 227,367 | 54,098 | 2 | 20,455 | 383 | 1+ 187, 2+ 38,004, 3+ 13,495, 4+ 2,412 |
| S13_Rep2 | 226,986 | 53,794 | 7 | 20,399 | 376 | 1+ 181, 2+ 37,884, 3+ 13,374, 4+ 2,355 |
| S13_Rep3 | 226,716 | 53,913 | 4 | 20,479 | 372 | 1+ 169, 2+ 37,991, 3+ 13,428, 4+ 2,325 |
| S14_Rep1 | 71,265 | 21,810 | 3 | 11,655 | 200 | 1+ 83, 2+ 14,552, 3+ 6,451, 4+ 724 |
| S14_Rep2 | 71,230 | 21,796 | 7 | 11,686 | 201 | 1+ 82, 2+ 14,359, 3+ 6,607, 4+ 748 |
| S14_Rep3 | 71,937 | 22,338 | 6 | 11,995 | 203 | 1+ 70, 2+ 14,677, 3+ 6,819, 4+ 772 |

## How much of it the model trained on

Identified spectra by the corpus split of their peptide (`peptide_registry.parquet` of `InstaDeepAI/InstaNovo`, peptide-disjoint 80/10/10; `absent` = not in the corpus at all). The released models trained on the train split; a spectrum here is never itself in the corpus unless the run is.

| split | spectra | share |
|---|---:|---:|
| train | 53,976 | 6.7 % |
| validation | 8,125 | 1.0 % |
| test | 29,102 | 3.6 % |
| absent | 710,925 | 88.6 % |

## Read against the corpus test splits

From `docs/results/profiles/*.json` (`scripts/evals/corpus_profile.py`, metadata columns plus the package's per-file table). Percentages within each set; top values only.

| | campi | MCFM test | LCFM test |
|---|---|---|---|
| spectra | 2,424,154 | 5,761,808 | 56,831,096 |
| identified | 802,128 | 5,761,808 | 56,831,096 |
| runs / projects | 21 / 1 | 15013 / 31 | 15219 / 31 |
| fragmentation (`frag_type`) | HCD 63 %; CID 37 % | HCID 48 %; HCD 30 %; CID 13 %; (missing) 9 % | HCID 47 %; HCD 34 %; (missing) 10 %; CID 9 % |
| instrument (search data) | Q Exactive HF-X 42 %; timsTOF Pro 37 %; Q Exactive HF 18 %; Q Exactive Plus 3 % | Orbitrap Fusion 20 %; Q Exactive 19 %; Q Exactive HF 18 %; Orbitrap Fusion Lumos 14 % | Q Exactive 19 %; Orbitrap Fusion 18 %; Q Exactive HF 18 %; Orbitrap Fusion Lumos 14 % |
| detector (search data) | Orbitrap 63 %; TOF 37 % | Orbitrap 88 %; Orbitrap|IonTrap 6 %; Astral 6 %; TOF 0 % | Orbitrap 90 %; Orbitrap|IonTrap 5 %; Astral 5 %; TOF 0 % |
| organism (search data) | Anaerostipes caccae; Bacteroides thetaiotaomicron; Bifidobacterium longum; Blautia producta; Clostridium butyricum; Escherichia coli; Lactiplantibacillus plantarum; Thomasclavelia ramosa 100 % | Homo sapiens 70 %; Homo sapiens; Arabidopsis thaliana 11 %; Homo Sapiens 5 %; Saccharomyces cerevisiae 4 %; Mus musculus 3 % | Homo sapiens 69 %; Homo sapiens; Arabidopsis thaliana 11 %; Saccharomyces cerevisiae 5 %; Homo Sapiens 5 %; Mus musculus 3 % |
| acquisition | DDA 100 % | DDA 93 %; DIA 7 % | DDA 95 %; DIA 5 % |
| collision energy | 25.0 42 %; (missing) 37 %; 27.0 16 %; 28.0 3 %; 30.0 2 % | 27.0 22 %; 25.0 16 %; 35.0 14 %; 30.0 14 %; (missing) 9 % | 27.0 22 %; 25.0 17 %; 30.0 13 %; 35.0 11 %; 28.0 11 % |
| precursor charge | 2 52 %; 3 35 %; 4 8 %; 1 2 %; 5 2 % | 2 73 %; 3 17 %; 0 7 %; 4 2 %; 1 0 % | 2 57 %; 3 31 %; 4 6 %; 0 5 %; 5 1 % |
| modified sequences | 28.5 % | 28.7 % | 30.6 % |
| modification tokens | C[UNIMOD:4] 60 %; M[UNIMOD:35] 36 %; Q[UNIMOD:28] 3 %; [UNIMOD:385]- 2 % | C[UNIMOD:4] 45 %; M[UNIMOD:35] 27 %; K[UNIMOD:259] 9 %; [UNIMOD:2016]- 4 % | C[UNIMOD:4] 39 %; M[UNIMOD:35] 34 %; K[UNIMOD:259] 9 %; [UNIMOD:737]- 3 % |
| peaks per spectrum (median, p5-p95) | 143 (19-581) | 294 (92-800) | 236 (58-800) |
| precursor m/z (median, p5-p95) | 747 (424-1339) | 685 (442-1094) | 640 (417-1053) |
| peptide length (median, p5-p95) | 16 (9-31) | 13 (8-24) | 13 (8-25) |

## Results

### Linear probes

| metric | released 40M (all 76,997) | 40M trained here (all 76,997) | released 89M (all 76,997) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Fragment type (macro-F1) | 0.999 (balanced accuracy) | 0.999 (balanced accuracy) | 0.999 (balanced accuracy) | 0.733 |
| Instrument (macro-F1) | 0.943 | 0.921 | 0.957 | 0.729 |
| PTM presence (balanced accuracy) | 0.741 | 0.729 | 0.761 | 0.756 |
| Modification class (macro-F1) | 0.383 | 0.371 | 0.436 | 0.466 |
| Hydrophobicity (R²) | 0.553 | 0.549 | 0.579 | 0.528 |
| Precursor mass (R²) | 0.849 | 0.848 | 0.867 | 0.702 |
| Precursor m/z (R²) | 0.932 | 0.931 | 0.944 | 0.897 |
| Charge (macro-F1) | 0.792 | 0.713 | 0.779 | 0.515 |
| Collision energy (R²) | 0.909 (macro-F1) | 0.870 (macro-F1) | 0.936 (macro-F1) | -1.678 |
| Spectrum confidence (R²) | 0.951 | 0.950 | 0.959 | 0.978 |

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

| metric | released 40M (10,000 of 802,128) | 40M trained here (10,000 of 802,128) | released 89M (10,000 of 802,128) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Fragment-group bin accuracy (IG task) | 48.1 % | 44.5 % | 61.9 % | 55.0 % |
| Bin accuracy y / b ions | 50.4 % / 43.0 % | 47.7 % / 37.5 % | 65.6 % / 53.5 % | 56.9 % / 50.0 % |
| Median error y / b ions | 220.3 / 1568.7 ppm | 283.1 / 3259.2 ppm | 101.9 / 210.8 ppm | 136.8 / 276.7 ppm |
| Peak-type accuracy | 71.1 % | 69.7 % | 70.9 % | 73.6 % |
| Peak-type macro-F1 | 0.516 | 0.505 | 0.524 | 0.519 |
| Cross-spectrum AUROC | 0.884 | 0.867 | 0.817 | 0.885 |
| Confidence AUROC, per spectrum / pooled | 0.704 / 0.700 | 0.713 / 0.709 | 0.642 / 0.648 | 0.718 / 0.705 |
| Structural attention heads | 11 | 11 | 10 | 11 |
| Isotope-spacing enrichment (mean) | 2.913 | 2.393 | 2.020 | 3.114 |

Theoretical b/y ions from the sequence with the checkpoint's residue masses; the column header says how many spectra were used.

### Embedding geometry

| metric | released 40M (20,000 of 802,128) | 40M trained here (20,000 of 802,128) | released 89M (20,000 of 802,128) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Anisotropy ratio | 25.4 | 25.4 | 27.0 | 25.0 |
| Effective rank | 89.8 | 90.9 | 31.9 | 89.4 |
| Top-component energy | 0.128 | 0.122 | 0.295 | 0.132 |
| Mean cosine | 0.842 | 0.840 | 0.947 | 0.811 |
| UMAP kNN preservation (k=15) | 0.086 | 0.082 | 0.116 | 0.098 |
| EVoC clusters / noise / purity | 4 / 0.338 / 0.997 | 15 / 0.512 / 0.999 | 15 / 0.488 / 0.999 | 17 / 0.502 / 0.811 |
| ESM2 alignment RSA / CKA | 0.092 / 0.103 | 0.082 / 0.101 | 0.061 / 0.071 | 0.048 / 0.060 |
| ESM2 metadata-baseline RSA | 0.218 | 0.218 | 0.218 | 0.198 |
| Glass Box residual / top importance | 2.67e-05 / 9.012 | 2.29e-05 / 9.060 | 2.29e-05 / 9.060 | 2.29e-05 / 3.704 |
| Cosine-hyperscore Spearman (pairs) | 0.008 (3694) | -0.022 (3694) | 0.055 (3694) | - |

Header says how many spectra were used (the paper's 20,000 above 100,000). EVoC purity is by fragmentation type and is 1 by construction when a dataset has one. The cosine-hyperscore correlation needs a hyperscore (X!Tandem or MSFragger); a dataset searched with another engine has none and shows "-". Glass Box attribution returned the same values for every checkpoint here and in the reproduction (released 40M against the one trained here); its output does not separate models.

### Every MS2 spectrum, identified or not

| metric | released 40M (100,000 of 2,424,154) | 40M trained here (100,000 of 2,424,154) | released 89M (100,000 of 2,424,154) |
|---|---|---|---|
| Anisotropy ratio | 25.1 | 25.1 | 26.8 |
| Effective rank | 77.6 | 73.3 | 30.6 |
| Top-component energy | 0.134 | 0.151 | 0.263 |
| Mean cosine | 0.821 | 0.818 | 0.932 |
| UMAP kNN preservation (k=15) | 0.116 | 0.113 | 0.143 |

Identified and unidentified spectra together (`dataset.is_annotated=false`); no reference column because the corpus holds identified spectra only.

### Trainer validation (masked-peak reconstruction)

| metric | released 40M (all 802,128) | 40M trained here (all 802,128) | released 89M (all 802,128) | released 40M, MCFM validation (reproduction) |
|---|---|---|---|---|
| Median |error| over masked peaks | 7007 ppm | 7364 ppm | 4147 ppm | 4493 ppm |
| MAE | 6.96 Da | 7.07 Da | 6.15 Da | 5.99 Da |
| Bin accuracy (0.2 Da) | 22.0 % | 20.8 % | 30.8 % | 27.4 % |
| Within 20 ppm | 3.5 % | 3.3 % | 4.8 % | 4.1 % |
| Intensity R² | 0.974 | 0.975 | 0.979 | 0.980 |

The trainer's validation loop on every identified spectrum (masking in the collate, seed fixed), the criterion the trainer selects checkpoints on.

## Sources

- Data: `$EXTERNAL/campi/` (`prepare_dataset.py`, manifest `proteomies-eval-data/outputs/fm_manifest.csv`, summary `campi-summary.json`).
- Runs: `docs/results/rerun/campi/<model>/<protocol>/run.json` (overrides, commit, host, seconds, rows in file, cap).
