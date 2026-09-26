# List of Results

This document compiles the results obtained from the paper or from inference on the 40M or 89M checkpoint. Results' origin:
1. Us using the 40M or 89M checkpoint and rerunning evals on the val/test data.
2. Listed in the paper or on one of their websites.

The sources are either links to the original paper or to scripts we ran ourselves to produce the rerun results (lives in `scripts/reproduce/` and includes path to data we used). In all the results presented here, no training was involved.

**Paper.** bioRxiv 10.64898/2026.09.03.747733v2 (posted 8 September 2026); table and line references are to
the full-text PDF (`pdftotext -layout`, 88 pages). The 40M model is `instanovo-fm-mcfm-90k-v0.1.0` (9 layers,
d 768, ffn 1024, about 90,000 steps on MCFM); the 89M model is `instanovo-fm-v0.1.0` (12 layers, ffn 3072,
about 230,000 steps on LCFM), the paper's "TS·noPA" / "deployed" model. Both are release assets
(`FoundationModel.describe_pretrained()`).

**Protocol behind the author numbers** (Table S9 and `docs/reproducing_paper_results.md`). Linear probes:
LCFM train / validation / test samples of 100,000 / 10,000 / 10,000, project-disjoint, cuML solvers on GPU,
L2 tuned on validation, one report on test. Duplicate retrieval: LCFM test, 20,000 query groups against a
200,000-spectrum pool, exact peptide-string match, k in 1, 5, 10, 20. Peak-type, cross-spectrum ion identity
and IG attribution: 10,000 LCFM test spectra (about 411,000 peaks). UMAP: 100,000 of a 140,000-spectrum
LCFM test pool. Signal composition and theoretical annotation: 200 HCFM validation spectra. The factorial
ablation (Table S4) is on the LCFM **validation** split; the deployed-model tables are on the **test** split.

**Comparability rules for the rerun column.** The repo's own reproduction command differs from the paper's
protocol in two ways that change what a number means, and each rerun records both:
`config.projects_shared_across_splits` (the documented command uses `use_project_split=false`: the probe still
draws 100,000 / 10,000 / 10,000 samples from the corpus's own train / validation / test files, the paper's
sizes, but without the paper's project assignment; `evaluator.py` line 1340) and `config.backend` (cuML on GPU
in the paper; scikit-learn on CPU when cuML is absent, and the two are stated to be non-comparable). Because
the probe reads the train split, every probe job needs the tier's training shards present. Sampling seeds and
the exact 20,000 / 200,000 subsets are not published, so agreement is expected within sampling noise, not to
the third decimal.

## 40M Results

Checkpoint `instanovo-fm-mcfm-90k-v0.1.0`. Every author number for this model is in Table S5, measured on
held-out **LCFM** spectra (the same pool as the 89M model), not on MCFM.

| No. | Metric | Result from our rerun | Source | Author-shared result | Source | Eval data used |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | Fragment type macro-F1 (4 classes) | 0.733 | `rerun/result1_40M_probes_retrieval/linearprobetask.json` @ 90c274e | 0.781 | Table S5 | LCFM test, probe protocol |
| 2 | Instrument macro-F1 (13 classes) | 0.729 | `rerun/result1_40M_probes_retrieval/linearprobetask.json` @ 90c274e | 0.697 | Table S5 | LCFM test, probe protocol |
| 3 | PTM presence balanced accuracy | 0.756 | `rerun/result1_40M_probes_retrieval/linearprobetask.json` @ 90c274e | 0.751 | Table S5 | LCFM test, probe protocol |
| 4 | Hydrophobicity R² | 0.528 | `rerun/result1_40M_probes_retrieval/linearprobetask.json` @ 90c274e | 0.518 | Table S5 | LCFM test, probe protocol |
| 5 | Precursor mass R² | 0.702 | `rerun/result1_40M_probes_retrieval/linearprobetask.json` @ 90c274e | 0.698 | Table S5 | LCFM test, probe protocol |
| 6 | Precursor m/z R² | 0.897 | `rerun/result1_40M_probes_retrieval/linearprobetask.json` @ 90c274e | 0.896 | Table S5 | LCFM test, probe protocol |
| 7 | Precursor charge macro-F1 (7 classes) | 0.515 | `rerun/result1_40M_probes_retrieval/linearprobetask.json` @ 90c274e | 0.602 | Table S5 | LCFM test, probe protocol |
| 8 | Spectrum confidence R² | 0.978 | `rerun/result1_40M_probes_retrieval/linearprobetask.json` @ 90c274e | 0.978 | Table S5 | LCFM test, probe protocol |
| 9 | Duplicate retrieval Recall@1 | 0.305 | `rerun/result1_40M_probes_retrieval/duplicateretrievaltask.json` @ 90c274e | 0.307 | Table S5 | LCFM test, 20,000 groups in a 200,000 pool |
| 10 | Duplicate retrieval mAP@20 | 0.125 | `rerun/result1_40M_probes_retrieval/duplicateretrievaltask.json` @ 90c274e | 0.126 | Table S5 | LCFM test, same pool |

No 40M number exists in the paper for reconstruction accuracy, peak-type classification, cross-spectrum ion
identity, attention-head structure, IG attribution, UMAP quality or clustering; the rerun will produce those
for the first time and they go in the rows below as they arrive, with no author column.

| No. | Metric | Result from our rerun | Source | Author-shared result | Source | Eval data used |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 11 | Masked-group bin accuracy, median ppm error | 55.0 % overall; y 56.9 % / b 50.0 %; 136.8 / 276.7 ppm | `rerun/result2_40M_peak_level/igattributiontask.results.json` @ ccf02d3 | none | | |
| 12 | Peak-type 4-way accuracy and macro-F1 | accuracy 73.6 %; macro-F1 0.519 | `rerun/result2_40M_peak_level/peaktypeclassificationtask.json` @ ccf02d3 | none | | |
| 13 | Cross-spectrum same-ion AUROC | 0.885 | `rerun/result2_40M_peak_level/peaktypeclassificationtask.json` @ ccf02d3 | none | | |
| 14 | Per-spectrum confidence AUROC (annotated vs not) | per-spectrum mean 0.718; pooled 0.705 | `rerun/result2_40M_peak_level/confidencesignalanalysistask.json` @ ccf02d3 | none | | |
| 15 | Embedding statistics (anisotropy ratio, effective rank, top-component energy) | anisotropy ratio 25.0; effective rank 89.4; top-component energy 0.132; mean cosine 0.811 | `rerun/result3_40M_geometry/embeddingstatisticstask.json` @ a772f6d | none | | |
| 16 | MCFM test, probes: fragment type / instrument / PTM presence / hydrophobicity / mass / m/z / charge / confidence | 0.776; 0.680; 0.799; 0.622; 0.755; 0.928; 0.608; 0.981 | `rerun/result5_40M_mcfm_test/linearprobetask.json` @ 17c0043 | none (our own number on the training tier) | | MCFM test, probe protocol |
| 17 | MCFM test, duplicate retrieval Recall@1 / mAP@20 | 0.629; 0.329 | `rerun/result5_40M_mcfm_test/duplicateretrievaltask.json` @ 17c0043 | none (our own number on the training tier) | | MCFM test, 20,000 groups in a 200,000 pool |
| 18 | UMAP kNN preservation at k 15: all / HCD-Orbitrap / CID | 0.098; HCD-Orbitrap 0.101; CID 0.264 | `rerun/result3_40M_geometry/umapvisualisationtask.json` @ a772f6d | none | | LCFM test, 20,000 spectra |
| 19 | EVoC clustering: clusters, noise fraction, silhouette, fragmentation-type purity | clusters 17; noise 0.502; silhouette 0.006; purity 0.811 | `rerun/result3_40M_geometry/evocclusteringtask.json` @ a772f6d | none | | LCFM test, 20,000 spectra |
| 20 | ESM2 cross-modal alignment: RSA rho (all pairs), CKA; shuffled / metadata RSA baselines | RSA 0.048; CKA 0.060; shuffled RSA 0.005; metadata RSA 0.198 | `rerun/result3_40M_geometry/esm2crossmodalalignmenttask.json` @ a772f6d | reported with shuffled and metadata baselines, values not given in the text | text, section "Cross-modal alignment" | LCFM test, 20,000 spectra |
| 21 | Glass Box attribution: features, reconstruction residual, top feature importance | features 14; residual 0.000; top importance 3.704 | `rerun/result3_40M_geometry/glassboxattributiontask.json` @ a772f6d | none | | LCFM test, 20,000 spectra |
| 22 | Cosine vs hyperscore: Spearman / Pearson | Spearman 0.148; Pearson 0.100; pairs 2252 | `rerun/result6_40M_cosine_hyperscore/cosinehyperscorecorrelationtask.results.json` @ c58e624 | none | | LCFM test, 20,000 spectra |
| 23 | Trainer validation metrics: median ppm error, MAE (Da), bin accuracy, within 20 ppm, intensity R² | median 4493 ppm; MAE 5.99 Da; bin accuracy 27.4 %; within 20 ppm 4.1 %; intensity R² 0.980 | `rerun/validate_released_40M/metrics.json` | none (the training run's target, in the trainer's units) | | MCFM validation, 256,000 spectra, Thompson-span masking as in training |

## 40M trained here

Checkpoint `instanovo-fm-mcfm-90k-ours-2026-09-26` (`$CHECKPOINTS`), `model_best.ckpt` of run `train-40M-mcfm-90k`
(job 22701776, `docs/agent_logs/session-2026-09-25-train40M.md`): the 40M architecture trained by us for 90,000 steps
on MCFM with the paper's schedule. Row numbers match the 40M table above; the comparison column holds the
**released 40M checkpoint's own rerun** from that table, not a paper number, so the two runs are compared under one
protocol, one backend and one sampling seed. Filled by `scripts/reproduce/fill_results.py`.

| No. | Metric | Result from our rerun | Source | Released 40M rerun | Source | Eval data used |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | Fragment type macro-F1 (4 classes) | | | | | LCFM test, probe protocol |
| 2 | Instrument macro-F1 (13 classes) | | | | | LCFM test, probe protocol |
| 3 | PTM presence balanced accuracy | | | | | LCFM test, probe protocol |
| 4 | Hydrophobicity R² | | | | | LCFM test, probe protocol |
| 5 | Precursor mass R² | | | | | LCFM test, probe protocol |
| 6 | Precursor m/z R² | | | | | LCFM test, probe protocol |
| 7 | Precursor charge macro-F1 (7 classes) | | | | | LCFM test, probe protocol |
| 8 | Spectrum confidence R² | | | | | LCFM test, probe protocol |
| 9 | Duplicate retrieval Recall@1 | | | | | LCFM test, 20,000 groups in a 200,000 pool |
| 10 | Duplicate retrieval mAP@20 | | | | | LCFM test, same pool |
| 11 | Masked-group bin accuracy, median ppm error | | | | | |
| 12 | Peak-type 4-way accuracy and macro-F1 | | | | | |
| 13 | Cross-spectrum same-ion AUROC | | | | | |
| 14 | Per-spectrum confidence AUROC (annotated vs not) | | | | | |
| 15 | Embedding statistics (anisotropy ratio, effective rank, top-component energy) | | | | | |
| 16 | MCFM test, probes: fragment type / instrument / PTM presence / hydrophobicity / mass / m/z / charge / confidence | | | | | MCFM test, probe protocol |
| 17 | MCFM test, duplicate retrieval Recall@1 / mAP@20 | | | | | MCFM test, 20,000 groups in a 200,000 pool |
| 18 | UMAP kNN preservation at k 15: all / HCD-Orbitrap / CID | | | | | LCFM test, 20,000 spectra |
| 19 | EVoC clustering: clusters, noise fraction, silhouette, fragmentation-type purity | | | | | LCFM test, 20,000 spectra |
| 20 | ESM2 cross-modal alignment: RSA rho (all pairs), CKA; shuffled / metadata RSA baselines | | | | | LCFM test, 20,000 spectra |
| 21 | Glass Box attribution: features, reconstruction residual, top feature importance | | | | | LCFM test, 20,000 spectra |
| 22 | Cosine vs hyperscore: Spearman / Pearson | | | | | LCFM test, 20,000 spectra |
| 23 | Trainer validation metrics: median ppm error, MAE (Da), bin accuracy, within 20 ppm, intensity R² | median 3897 ppm; MAE 5.25 Da; bin accuracy 27.2 %; within 20 ppm 4.2 %; intensity R² 0.990 | `rerun/train_40M_mcfm_90k/metrics.json` | median 4493 ppm; MAE 5.99 Da; bin accuracy 27.4 %; within 20 ppm 4.1 %; intensity R² 0.980 | `rerun/validate_released_40M/metrics.json` | MCFM validation, 256,000 spectra, Thompson-span masking as in training |

## 89M Results

Checkpoint `instanovo-fm-v0.1.0`.

### Frozen probes and duplicate retrieval, LCFM test (Tables S5 and S10)

| No. | Metric | Result from our rerun | Source | Author-shared result | Source | Eval data used |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | Fragment type macro-F1 (4 classes) | 0.817 | `rerun/result1_89M_probes_retrieval/linearprobetask.json` @ 769aaf6 | 0.855 | Tables S5, S10 | LCFM test, probe protocol |
| 2 | Instrument macro-F1 (13 classes) | 0.817 | `rerun/result1_89M_probes_retrieval/linearprobetask.json` @ 769aaf6 | 0.804 | Tables S5, S10 | LCFM test, probe protocol |
| 3 | PTM presence balanced accuracy | 0.806 | `rerun/result1_89M_probes_retrieval/linearprobetask.json` @ 769aaf6 | 0.802 | Tables S5, S10 | LCFM test, probe protocol |
| 4 | Modification class macro-F1 | 0.568 | `rerun/result1_89M_probes_retrieval/linearprobetask.json` @ 769aaf6 | 0.622 | Table S10 | LCFM test, probe protocol |
| 5 | Hydrophobicity R² | 0.615 | `rerun/result1_89M_probes_retrieval/linearprobetask.json` @ 769aaf6 | 0.605 | Tables S5, S10 | LCFM test, probe protocol |
| 6 | Precursor mass R² | 0.742 | `rerun/result1_89M_probes_retrieval/linearprobetask.json` @ 769aaf6 | 0.732 | Tables S5, S10 | LCFM test, probe protocol |
| 7 | Precursor m/z R² | 0.932 | `rerun/result1_89M_probes_retrieval/linearprobetask.json` @ 769aaf6 | 0.929 | Tables S5, S10 | LCFM test, probe protocol |
| 8 | Precursor charge macro-F1 (7 classes) | 0.609 | `rerun/result1_89M_probes_retrieval/linearprobetask.json` @ 769aaf6 | 0.650 | Tables S5, S10 | LCFM test, probe protocol |
| 9 | Spectrum confidence R² | 0.973 | `rerun/result1_89M_probes_retrieval/linearprobetask.json` @ 769aaf6 | 0.973 | Table S5 | LCFM test, probe protocol |
| 10 | Duplicate retrieval Recall@1 | 0.209 | `rerun/result1_89M_probes_retrieval/duplicateretrievaltask.json` @ 769aaf6 | 0.215 | Tables S5, S10 | LCFM test, 20,000 groups in a 200,000 pool |
| 11 | Duplicate retrieval mAP@20 | 0.074 | `rerun/result1_89M_probes_retrieval/duplicateretrievaltask.json` @ 769aaf6 | 0.076 | Tables S5, S10 | LCFM test, same pool |
| 12 | Precursor charge probe macro-AUROC | 0.970 | `rerun/result1_89M_probes_retrieval/linearprobetask.results.json` @ 769aaf6 | 0.956 | Fig. 5 caption (line 4450) | LCFM test |
| 13 | Fragmentation method probe macro-AUROC | 0.966 | `rerun/result1_89M_probes_retrieval/linearprobetask.results.json` @ 769aaf6 | 0.962 | Fig. 5 caption | LCFM test |

### Peak level and reconstruction (main text, sections "Peak-level representations" and "Reconstruction")

| No. | Metric | Result from our rerun | Source | Author-shared result | Source | Eval data used |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 14 | Masked-group bin accuracy, overall | 68.7 % | `rerun/result2_89M_peak_level/igattributiontask.results.json` @ 8d0befe | 70.1 % | text, line 479; Methods, lines 3730-3739 | LCFM test; IG attribution over 413 quality-gated spectra, 1,239 masked groups |
| 15 | Bin accuracy y-ions / b-ions | 72.1 % / 59.8 % | `rerun/result2_89M_peak_level/igattributiontask.results.json` @ 8d0befe | 74.3 % / 59.2 % | text, line 481 | as 14 (`prediction_quality.by_ion_type` of the IG task) |
| 16 | Median reconstruction error y / b | 102.1 / 194.3 ppm | `rerun/result2_89M_peak_level/igattributiontask.results.json` @ 8d0befe | 89.8 / 196.4 ppm | text, line 492 | as 14 |
| 17 | Per-spectrum confidence AUROC, annotated vs unannotated | 0.659 | `rerun/result2_89M_peak_level/confidencesignalanalysistask.json` @ 8d0befe | 0.658 | text, line 500 | LCFM test (general rule, Methods line 2295) |
| 18 | Peak-type 4-way accuracy after the transformer / pre-transformer baseline | after transformer 77.7 %; pre-transformer 50.2 % | `rerun/result2_89M_peak_level/peaktypeclassificationtask.results.json` @ 8d0befe | 77.5 % / 49.1 % | text, lines 1145-1146 | LCFM test, 10,000 spectra |
| 19 | Peak-type macro-F1 | 0.564 | `rerun/result2_89M_peak_level/peaktypeclassificationtask.json` @ 8d0befe | 0.575 | text, line 1146; Table S4 (validation) | LCFM test / validation |
| 20 | Unannotated-vs-annotated precision / F1 | precision 0.949; F1 0.844 | `rerun/result2_89M_peak_level/peaktypeclassificationtask.results.json` @ 8d0befe | 0.950 / 0.840 | text, line 1149 | LCFM test, 10,000 spectra |
| 21 | y-ion F1 / b-ion F1 | y 0.725; b 0.514 | `rerun/result2_89M_peak_level/peaktypeclassificationtask.results.json` @ 8d0befe | 0.734 / 0.531 | text, line 1151 | LCFM test, 10,000 spectra |
| 22 | Cross-spectrum same-ion AUROC | 0.832 | `rerun/result2_89M_peak_level/peaktypeclassificationtask.json` @ 8d0befe | 0.837 (test); 0.857 (validation) | text, line 1165; Table S4 | LCFM test 10,000 spectra; validation |
| 23 | Same-ion cosine / m/z-matched / unrelated | same ion 0.920; m/z-matched 0.833; random 0.700 | `rerun/result2_89M_peak_level/peaktypeclassificationtask.results.json` @ 8d0befe | 0.921 / 0.830 / 0.697 | text, lines 1167-1169 | LCFM test |
| 24 | Joint confidence of matched vs unmatched unannotated peaks, AUROC | | | median 0.26 vs 0.13, AUROC 0.684 | text, line 1231 | HCFM validation, 200 spectra (Table S9) |
| 25 | IG attribution: ion-ladder neighbour is top-1 / in top-5 | top-1 29.4 %; top-5 55.3 % | `rerun/result2_89M_peak_level/igattributiontask.json` @ 8d0befe | 34 % / 55.0 % | text, lines 472-473 | LCFM test, 10,000 spectra |

### Factorial ablation, LCFM validation (Table S4, the TS·noPA column is this checkpoint)

| No. | Metric | Result from our rerun | Source | Author-shared result | Source | Eval data used |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 26 | Fragment type F1 (4 classes) | 0.817 | `rerun/result4_89M_factorial_validation/linearprobetask.json` @ 769aaf6 | 0.865 | Table S4 | LCFM validation |
| 27 | Instrument macro-F1 | 0.817 | `rerun/result4_89M_factorial_validation/linearprobetask.json` @ 769aaf6 | 0.807 | Table S4 | LCFM validation |
| 28 | Spectrum confidence R² | 0.973 | `rerun/result4_89M_factorial_validation/linearprobetask.json` @ 769aaf6 | 0.973 | Table S4 | LCFM validation |
| 29 | PTM balanced accuracy | 0.806 | `rerun/result4_89M_factorial_validation/linearprobetask.json` @ 769aaf6 | 0.802 | Table S4 | LCFM validation |
| 30 | Modification class macro-F1 | 0.568 | `rerun/result4_89M_factorial_validation/linearprobetask.json` @ 769aaf6 | 0.620 | Table S4 | LCFM validation |
| 31 | Precursor m/z R² | 0.932 | `rerun/result4_89M_factorial_validation/linearprobetask.json` @ 769aaf6 | 0.929 | Table S4 | LCFM validation |
| 32 | Cross-spectrum AUROC | | | 0.857 | Table S4 | LCFM validation |
| 33 | Recall@1 overall / CID | 0.398; CID 0.420 | `rerun/result4_89M_factorial_validation/duplicateretrievaltask.json` @ 769aaf6 | 0.400 / 0.426 | Table S4 | LCFM validation |
| 34 | Structural attention heads of 12 | | | 8 | Table S4 | LCFM validation |
| 35 | Mean isotope-spacing enrichment | | | 1.99 | Table S4 | LCFM validation |
| 36 | Peak-type macro-F1 | | | 0.575 | Table S4 | LCFM validation |

The other three Table S4 columns belong to the ablation checkpoints `instanovo-fm-lcfm-sa-nopa-v0.1.0`,
`instanovo-fm-lcfm-sa-pa-v0.1.0` and `instanovo-fm-lcfm-ts-pa-v0.1.0`, all downloaded; they are a second
pass once the two main checkpoints match.

### Embedding geometry, LCFM test (no author numbers; Fig. 3A is the UMAP)

| No. | Metric | Result from our rerun | Source | Author-shared result | Source | Eval data used |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 41 | Embedding statistics: anisotropy ratio, effective rank, top-component energy, mean pairwise cosine | anisotropy ratio 26.8; effective rank 46.6; top-component energy 0.192; mean cosine 0.937 | `rerun/result3_89M_geometry/embeddingstatisticstask.json` @ a772f6d | none | | LCFM test, 20,000 spectra |
| 42 | UMAP kNN preservation at k 15: all / HCD-Orbitrap / CID | 0.157; HCD-Orbitrap 0.161; CID 0.328 | `rerun/result3_89M_geometry/umapvisualisationtask.json` @ a772f6d | none | | LCFM test, 20,000 spectra |
| 43 | EVoC clustering: clusters, noise fraction, silhouette, fragmentation-type purity | clusters 15; noise 0.405; silhouette 0.060; purity 0.789 | `rerun/result3_89M_geometry/evocclusteringtask.json` @ a772f6d | none | | LCFM test, 20,000 spectra |
| 44 | ESM2 cross-modal alignment: RSA rho (all pairs), CKA; shuffled / metadata RSA baselines | RSA 0.039; CKA 0.046; shuffled RSA 0.006; metadata RSA 0.198 | `rerun/result3_89M_geometry/esm2crossmodalalignmenttask.json` @ a772f6d | reported with shuffled and metadata baselines, values not given in the text | text, section "Cross-modal alignment" | LCFM test, 20,000 spectra |
| 45 | Glass Box attribution: features, reconstruction residual, top feature importance | features 14; residual 0.000; top importance 3.704 | `rerun/result3_89M_geometry/glassboxattributiontask.json` @ a772f6d | none | | LCFM test, 20,000 spectra |
| 46 | Cosine vs hyperscore: Spearman / Pearson | Spearman 0.189; Pearson 0.168; pairs 2350 | `rerun/result6_89M_cosine_hyperscore/cosinehyperscorecorrelationtask.results.json` @ c58e624 | none | | LCFM test, 20,000 spectra |

### Results that need data we do not have

| No. | Metric | Author-shared result | Source | Eval data used |
| :---- | :---- | :---- | :---- | :---- |
| 37 | Hela qc probes: charge macro-F1, m/z R², mass R², hydrophobicity R², PTM bal. acc., mod. class macro-F1 | 0.577, 0.951, 0.887, 0.568, 0.761, 0.545 | Table S11 | Hela qc, 80/10/10, n 17,683 (external) |
| 38 | De novo peptide recall, fine-tuned FM (GluC, S. brodae, snake venoms, Hela QC, TPL antibodies, wound fluids) | 0.821, 0.740, 0.230, 0.662, 0.528, 0.366 | Table S12 | six biological validation sets (external); checkpoint `instanovo-fm-denovo-v0.1.0` |
| 39 | De novo recall, from-scratch / frozen-encoder variants | S. brodae 0.747 / 0.726; Hela QC 0.657 / 0.577 (full rows in Table S13) | Table S13 | as 38; checkpoints `-denovo-scratch`, `-denovo-frozen` |
| 40 | Database-free retrieval, rescue and run classification panels | controlled panel (5 peptides × 50 replicates among 200 others, 250 queries): recall@1 0.91, about 1.0 by k = 20; rescue counts in Fig. 6 | text line 1525, Fig. 6, Supplementary Fig. S16D, Table S9 (PXD074343 run 477-1, 30,404 spectra) | built by `scripts/create_*_dataset.py` from an external project; evidence blocks need unpublished code (`docs/sanitisation.md`) |

**Note on rows 14 to 17.** The Methods (line 2295) state that evaluations of the deployed checkpoint use the
held-out LCFM test split unless stated otherwise, and the IG attribution protocol (lines 3730-3739) masks up
to three fragment groups in each spectrum that passes the quality gates, 413 spectra and 1,239 masked-group
predictions in the paper. The evaluation task's default `max_spectra` of 500 with the same gates is the
matching setting, so `result2_*_peak_level.py` keeps it.

## Data and checkpoints

`$DATA` (`~/projects/rrg-hsn/proteomies/data/proteometoolsI`): `splits/mcfm/` (29 train, 1 validation, 14 test
shards), `splits/lcfm/` (validation and test shards; the 305 GiB training split is not downloaded),
`peptide_registry.parquet`, `manifests/`, and `hf_tree_*.json` (the Hugging Face listing with sizes and
sha256 for a checksum pass). `$CHECKPOINTS` (`~/projects/rrg-hsn/proteomies/checkpoints/instanovofm`): the
eight v0.1.0 release checkpoints. Fetched by `scripts/reproduce/fetch_data.sh` on 25/09/2026.
