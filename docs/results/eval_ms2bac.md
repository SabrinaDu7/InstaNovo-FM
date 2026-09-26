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
| spectra | 42,410 | - | - |
| identified | 20,269 | - | - |
| runs / projects | 3 / 1 | - | - |
| fragmentation (`frag_type`) | HCD 100 % | - | - |
| instrument (search data) | Orbitrap Exploris 480 100 % | - | - |
| detector (search data) | Orbitrap 100 % | - | - |
| organism (search data) | Escherichia coli DSM 30083 34 %; Bacillus cereus DSM 31 33 %; Staphylococcus aureus DSM 20231 33 % | - | - |
| acquisition | DDA 100 % | - | - |
| collision energy | 28.0 100 % | - | - |
| precursor charge | 2 69 %; 3 26 %; 4 4 %; 5 1 %; 6 0 % | - | - |
| modified sequences | 11.4 % | - | - |
| modification tokens | C[UNIMOD:4] 65 %; M[UNIMOD:35] 34 %; [UNIMOD:1]- 0 % | - | - |
| peaks per spectrum (median, p5-p95) | 63 (8-231) | - | - |
| precursor m/z (median, p5-p95) | 595 (380-1073) | - | - |
| peptide length (median, p5-p95) | 12 (7-25) | - | - |

## Results

### Linear probes

| metric | released 40M (-) | 40M trained here (-) | released 89M (-) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Fragment type (macro-F1) | - | - | - | 0.733 |
| Instrument (macro-F1) | - | - | - | 0.729 |
| PTM presence (balanced accuracy) | - | - | - | 0.756 |
| Modification class (macro-F1) | - | - | - | 0.466 |
| Hydrophobicity (R²) | - | - | - | 0.528 |
| Precursor mass (R²) | - | - | - | 0.702 |
| Precursor m/z (R²) | - | - | - | 0.897 |
| Charge (macro-F1) | - | - | - | 0.515 |
| Collision energy (R²) | - | - | - | -1.678 |
| Spectrum confidence (R²) | - | - | - | 0.978 |

Probe train / valid / test are this dataset's own peptide-disjoint files (package caps 100,000 / 10,000 / 10,000). A one-class target (one instrument, one fragmentation) cannot be probed and shows "-".

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

| metric | released 40M (-) | 40M trained here (-) | released 89M (-) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Fragment-group bin accuracy (IG task) | - | - | - | 55.0 % |
| Bin accuracy y / b ions | - | - | - | 56.9 % / 50.0 % |
| Median error y / b ions | - | - | - | 136.8 / 276.7 ppm |
| Peak-type accuracy | - | - | - | 73.6 % |
| Peak-type macro-F1 | - | - | - | 0.519 |
| Cross-spectrum AUROC | - | - | - | 0.885 |
| Confidence AUROC, per spectrum / pooled | - | - | - | 0.718 / 0.705 |
| Structural attention heads | - | - | - | 11 |
| Isotope-spacing enrichment (mean) | - | - | - | 3.114 |

Theoretical b/y ions from the sequence with the checkpoint's residue masses; the column header says how many spectra were used.

### Embedding geometry

| metric | released 40M (-) | 40M trained here (-) | released 89M (-) | released 40M, LCFM test (reproduction) |
|---|---|---|---|---|
| Anisotropy ratio | - | - | - | 25.0 |
| Effective rank | - | - | - | 89.4 |
| Top-component energy | - | - | - | 0.132 |
| Mean cosine | - | - | - | 0.811 |
| UMAP kNN preservation (k=15) | - | - | - | 0.098 |
| EVoC clusters / noise / purity | - | - | - | 17 / 0.502 / 0.811 |
| ESM2 alignment RSA / CKA | - | - | - | 0.048 / 0.060 |
| ESM2 metadata-baseline RSA | - | - | - | 0.198 |
| Glass Box residual / top importance | - | - | - | 2.29e-05 / 3.704 |
| Cosine-hyperscore Spearman (pairs) | - | - | - | - |

Header says how many spectra were used (the paper's 20,000 above 100,000).

### Every MS2 spectrum, identified or not

| metric | released 40M (-) | 40M trained here (-) | released 89M (-) |
|---|---|---|---|
| Anisotropy ratio | - | - | - |
| Effective rank | - | - | - |
| Top-component energy | - | - | - |
| Mean cosine | - | - | - |
| UMAP kNN preservation (k=15) | - | - | - |

Identified and unidentified spectra together (`dataset.is_annotated=false`); no reference column because the corpus holds identified spectra only.

### Trainer validation (masked-peak reconstruction)

| metric | released 40M (-) | 40M trained here (-) | released 89M (-) | released 40M, MCFM validation (reproduction) |
|---|---|---|---|---|
| Median |error| over masked peaks | - | - | - | 4493 ppm |
| MAE | - | - | - | 5.99 Da |
| Bin accuracy (0.2 Da) | - | - | - | 27.4 % |
| Within 20 ppm | - | - | - | 4.1 % |
| Intensity R² | - | - | - | 0.980 |

The trainer's validation loop on every identified spectrum (masking in the collate, seed fixed), the criterion the trainer selects checkpoints on.

## Sources

- Data: `$EXTERNAL/ms2bac/` (`prepare_dataset.py`, manifest `proteomies-eval-data/outputs/fm_manifest.csv`, summary `ms2bac-summary.json`).
- Runs: `docs/results/rerun/ms2bac/<model>/<protocol>/run.json` (overrides, commit, host, seconds, rows in file, cap).
