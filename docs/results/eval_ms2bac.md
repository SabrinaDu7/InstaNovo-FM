# InstaNovo-FM evaluation suite on ms2bac

Rendered by `scripts/evals/render_results.py` from `docs/results/rerun/ms2bac/<model>/<protocol>/` (jobs of `scripts/evals/run.py`, data from `scripts/evals/prepare_dataset.py`). Reference column: the released 40M on the corpus split named in the header, from the reproduction (`docs/references/results_paper_or_rerun.md`). "-" = not run yet or the task reported nothing.

## The data

- 3 runs, 42,410 MS2 spectra, 20,269 identified with an expressible sequence, 16,197 distinct peptides (I/L collapsed, modifications dropped); probe split 16,176 / 2,039 / 2,054 spectra (train / valid / test, peptide-disjoint, seed 42).
- Instrument: Orbitrap Exploris 480; fragmentation: HCD; detector: Orbitrap; organism: Bacillus cereus DSM 31; Escherichia coli DSM 30083; Staphylococcus aureus DSM 20231. Search engine: maxquant. Carbamidomethyl treated as fixed: True.

| run | MS2 | identified | unsupported modifications | peptides | median peaks | charges of identified |
|---|---:|---:|---:|---:|---:|---|
| bacillus_cereus_BBM_749_P110_38_MIA_001 | 14,156 | 6,945 | 0 | 5,789 | 62 | 1+ 15, 2+ 4,891, 3+ 1,720, 4+ 273, 5+ 42, 6+ 4 |
| e_coli_BBM_749_P110_38_MIA_049 | 14,392 | 9,504 | 0 | 8,062 | 105 | 1+ 16, 2+ 6,460, 3+ 2,500, 4+ 444, 5+ 72, 6+ 12 |
| staph_a_BBM_750_P110_38_MIA_049 | 13,862 | 3,820 | 0 | 3,236 | 30 | 1+ 6, 2+ 2,990, 3+ 747, 4+ 75, 5+ 2 |

## How much of it the model trained on

Identified spectra by the corpus split of their peptide (`peptide_registry.parquet` of `InstaDeepAI/InstaNovo`, peptide-disjoint 80/10/10; `absent` = not in the corpus at all). The released models trained on the train split; a spectrum here is never itself in the corpus unless the run is.

| split | spectra | share |
|---|---:|---:|
| train | 8,077 | 39.8 % |
| validation | 1,477 | 7.3 % |
| test | 3,488 | 17.2 % |
| absent | 7,227 | 35.7 % |

## Read against the corpus test splits

From `docs/results/profiles/*.json` (`scripts/evals/corpus_profile.py`, metadata columns plus the package's per-file table). Percentages within each set; top values only.

| | ms2bac | MCFM test | LCFM test |
|---|---|---|---|
| spectra | 42,410 | 5,761,808 | 56,831,096 |
| identified | 20,269 | 5,761,808 | 56,831,096 |
| runs / projects | 3 / 1 | 15013 / 31 | 15219 / 31 |
| fragmentation (`frag_type`) | HCD 100 % | HCID 48 %; HCD 30 %; CID 13 %; (missing) 9 % | HCID 47 %; HCD 34 %; (missing) 10 %; CID 9 % |
| instrument (search data) | Orbitrap Exploris 480 100 % | Orbitrap Fusion 20 %; Q Exactive 19 %; Q Exactive HF 18 %; Orbitrap Fusion Lumos 14 % | Q Exactive 19 %; Orbitrap Fusion 18 %; Q Exactive HF 18 %; Orbitrap Fusion Lumos 14 % |
| detector (search data) | Orbitrap 100 % | Orbitrap 88 %; Orbitrap|IonTrap 6 %; Astral 6 %; TOF 0 % | Orbitrap 90 %; Orbitrap|IonTrap 5 %; Astral 5 %; TOF 0 % |
| organism (search data) | Escherichia coli DSM 30083 34 %; Bacillus cereus DSM 31 33 %; Staphylococcus aureus DSM 20231 33 % | Homo sapiens 70 %; Homo sapiens; Arabidopsis thaliana 11 %; Homo Sapiens 5 %; Saccharomyces cerevisiae 4 %; Mus musculus 3 % | Homo sapiens 69 %; Homo sapiens; Arabidopsis thaliana 11 %; Saccharomyces cerevisiae 5 %; Homo Sapiens 5 %; Mus musculus 3 % |
| acquisition | DDA 100 % | DDA 93 %; DIA 7 % | DDA 95 %; DIA 5 % |
| collision energy | 28.0 100 % | 27.0 22 %; 25.0 16 %; 35.0 14 %; 30.0 14 %; (missing) 9 % | 27.0 22 %; 25.0 17 %; 30.0 13 %; 35.0 11 %; 28.0 11 % |
| precursor charge | 2 69 %; 3 26 %; 4 4 %; 5 1 %; 6 0 % | 2 73 %; 3 17 %; 0 7 %; 4 2 %; 1 0 % | 2 57 %; 3 31 %; 4 6 %; 0 5 %; 5 1 % |
| modified sequences | 11.4 % | 28.7 % | 30.6 % |
| modification tokens | C[UNIMOD:4] 65 %; M[UNIMOD:35] 34 %; [UNIMOD:1]- 0 % | C[UNIMOD:4] 45 %; M[UNIMOD:35] 27 %; K[UNIMOD:259] 9 %; [UNIMOD:2016]- 4 % | C[UNIMOD:4] 39 %; M[UNIMOD:35] 34 %; K[UNIMOD:259] 9 %; [UNIMOD:737]- 3 % |
| peaks per spectrum (median, p5-p95) | 63 (8-231) | 294 (92-800) | 236 (58-800) |
| precursor m/z (median, p5-p95) | 595 (380-1073) | 685 (442-1094) | 640 (417-1053) |
| peptide length (median, p5-p95) | 12 (7-25) | 13 (8-24) | 13 (8-25) |

## Results

### Linear probes

| metric | released 40M (all 2,054) | 40M trained here (all 2,054) | released 89M (all 2,054) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Fragment type (macro-F1) | - | - | - | 0.733 |
| Instrument (macro-F1) | - | - | - | 0.729 |
| PTM presence (balanced accuracy) | 0.824 | 0.789 | 0.848 | 0.756 |
| Modification class (macro-F1) | 0.672 | 0.614 | 0.664 | 0.466 |
| Hydrophobicity (R²) | 0.608 | 0.593 | 0.646 | 0.528 |
| Precursor mass (R²) | 0.843 | 0.850 | 0.878 | 0.702 |
| Precursor m/z (R²) | 0.943 | 0.952 | 0.963 | 0.897 |
| Charge (macro-F1) | 0.440 | 0.492 | 0.510 | 0.515 |
| Collision energy (R²) | - | - | - | -1.678 |
| Spectrum confidence (R²) | 0.917 | 0.910 | 0.899 | 0.978 |

Probe train / valid / test are this dataset's own peptide-disjoint files (package caps 100,000 / 10,000 / 10,000). A one-class target (one instrument, one fragmentation) cannot be probed and shows "-". A cell names its metric when the probe chose another than the row label's (balanced accuracy for a two-class target, macro-F1 for a collision energy with few settings); the probe scores only the classes present in its test split.

### Duplicate-spectrum retrieval

| metric | released 40M (all 20,269) | 40M trained here (all 20,269) | released 89M (all 20,269) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Recall@1 | 0.421 | 0.431 | 0.290 | 0.305 |
| mAP@20 | 0.419 | 0.431 | 0.276 | 0.125 |
| Proportional recall@1 | 0.348 | 0.358 | 0.229 | 0.068 |
| Recall@1, HCD Orbitrap subset | 0.421 | 0.431 | 0.290 | 0.328 |
| Recall@1, CID subset | - | - | - | 0.365 |

Every identified spectrum is in the pool and every duplicate group is queried; a positive is an identical peptide string (charge ignored).

### Peak level

| metric | released 40M (all 20,269) | 40M trained here (all 20,269) | released 89M (all 20,269) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Fragment-group bin accuracy (IG task) | 66.6 % | 63.2 % | 83.1 % | 55.0 % |
| Bin accuracy y / b ions | 69.7 % / 56.2 % | 67.7 % / 47.8 % | 86.6 % / 71.2 % | 56.9 % / 50.0 % |
| Median error y / b ions | 117.0 / 213.4 ppm | 121.2 / 355.3 ppm | 106.7 / 148.9 ppm | 136.8 / 276.7 ppm |
| Peak-type accuracy | 78.4 % | 77.0 % | 82.0 % | 73.6 % |
| Peak-type macro-F1 | 0.611 | 0.593 | 0.654 | 0.519 |
| Cross-spectrum AUROC | 0.937 | 0.932 | 0.889 | 0.885 |
| Confidence AUROC, per spectrum / pooled | 0.735 / 0.716 | 0.750 / 0.731 | 0.660 / 0.638 | 0.718 / 0.705 |
| Structural attention heads | 9 | 10 | 6 | 11 |
| Isotope-spacing enrichment (mean) | 2.442 | 2.196 | 1.776 | 3.114 |

Theoretical b/y ions from the sequence with the checkpoint's residue masses; the column header says how many spectra were used.

### Embedding geometry

| metric | released 40M (all 20,269) | 40M trained here (all 20,269) | released 89M (all 20,269) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Anisotropy ratio | 24.9 | 24.7 | 27.0 | 25.0 |
| Effective rank | 104.4 | 100.2 | 59.5 | 89.4 |
| Top-component energy | 0.121 | 0.127 | 0.195 | 0.132 |
| Mean cosine | 0.808 | 0.795 | 0.952 | 0.811 |
| UMAP kNN preservation (k=15) | 0.042 | 0.040 | 0.052 | 0.098 |
| EVoC clusters / noise / purity | 8 / 0.595 / 1.000 | 5 / 0.629 / 1.000 | 5 / 0.581 / 1.000 | 17 / 0.502 / 0.811 |
| ESM2 alignment RSA / CKA | 0.088 / 0.115 | 0.094 / 0.117 | 0.077 / 0.105 | 0.048 / 0.060 |
| ESM2 metadata-baseline RSA | 0.245 | 0.245 | 0.245 | 0.198 |
| Glass Box residual / top importance | 2.67e-05 / 9.412 | 2.67e-05 / 9.412 | 2.67e-05 / 9.412 | 2.29e-05 / 3.704 |
| Cosine-hyperscore Spearman (pairs) | - | - | - | - |

Header says how many spectra were used (the paper's 20,000 above 100,000). EVoC purity is by fragmentation type and is 1 by construction when a dataset has one. The cosine-hyperscore correlation needs a hyperscore (X!Tandem or MSFragger); a dataset searched with another engine has none and shows "-". Glass Box attribution returned the same values for every checkpoint here and in the reproduction (released 40M against the one trained here); its output does not separate models.

### Every MS2 spectrum, identified or not

| metric | released 40M (all 42,410) | 40M trained here (all 42,410) | released 89M (all 42,410) |
|---|---|---|---|
| Anisotropy ratio | 24.5 | 24.2 | 26.7 |
| Effective rank | 72.8 | 63.4 | 34.4 |
| Top-component energy | 0.215 | 0.213 | 0.364 |
| Mean cosine | 0.781 | 0.761 | 0.930 |
| UMAP kNN preservation (k=15) | 0.059 | 0.061 | 0.072 |

Identified and unidentified spectra together (`dataset.is_annotated=false`); no reference column because the corpus holds identified spectra only.

### Trainer validation (masked-peak reconstruction)

| metric | released 40M (all 20,269) | 40M trained here (all 20,269) | released 89M (all 20,269) | released 40M, MCFM validation (reproduction) |
|---|---|---|---|---|
| Median |error| over masked peaks | 4185 ppm | 4989 ppm | 383 ppm | 4493 ppm |
| MAE | 5.79 Da | 5.94 Da | 4.54 Da | 5.99 Da |
| Bin accuracy (0.2 Da) | 34.7 % | 32.6 % | 48.9 % | 27.4 % |
| Within 20 ppm | 5.0 % | 4.8 % | 6.8 % | 4.1 % |
| Intensity R² | 0.980 | 0.980 | 0.985 | 0.980 |

The trainer's validation loop on every identified spectrum (masking in the collate, seed fixed), the criterion the trainer selects checkpoints on.

## Sources

- Data: `$EXTERNAL/ms2bac/` (`prepare_dataset.py`, manifest `proteomies-eval-data/outputs/fm_manifest.csv`, summary `ms2bac-summary.json`).
- Runs: `docs/results/rerun/ms2bac/<model>/<protocol>/run.json` (overrides, commit, host, seconds, rows in file, cap).
