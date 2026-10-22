# Session log: InstaNovo-FM on the nine-species benchmark

Date: 22/10/2026. Branch `sdu/eval-nine-species`. The study's one home is `docs/nine_species/README.md`; this file is
the chronological record, with the assumptions and the uncertainties as they arose.

## Initial purpose of the session

Owner's goal (set 22/10 00:45): a separate branch; an in-depth evaluation on the nine-species benchmark compared with
Casanovo's numbers; log the work, the core assumptions and what is unsure; after the runs, two review passes for
mistakes and scientific dishonesty; the goal is truth, correctness and consistency; find what breaks the model, its
key assumptions, and whether each strength or weakness is data dependent.

## Log

| When | What |
| :---- | :---- |
| 00:50 | Exploration before any code. The benchmark is local (`datasets/nine_species/`, 9 species, 2,844,905 spectra in MGFs with `SEQ=` on every spectrum, 29 GB; counted). Casanovo compares de novo sequencing, so the object under test is the released sequencer `instanovo-fm-denovo-v0.1.0` plus the frozen and from-scratch controls; copied from nibi (md5 verified). |
| 00:55 | **All nine PRIDE projects are in the pretraining corpus** (`assets/table_s1_accessions.txt`; the preprint: "the 9 projects comprising the standard nine-species benchmark" were included). The September session had excluded the benchmark for that reason. |
| 01:05 | The corpus peptide registry (41 MB, copied from nibi) resolves the overlap at the peptide level: 99.3 to 99.8 % of each species' labelled peptides, and 99.7 to 100 % of its spectra, are in the corpus's **test** split (`scripts/analysis/nine_species_overlap.py`; a random split would give 11 %). The training configs read train shards only. Assumption recorded: the peptides were held out deliberately; what remains shared is run, instrument and lab context. |
| 01:10 | Casanovo's metric fetched live: `casanovo/denovo/evaluate.py` v4.2.1 and v5.0.0 (sha256 9f68d042191b, f2b477db1b5a), the prefix-suffix match with 0.5 Da cumulative and 0.1 Da per-residue thresholds; v5 `config.yaml` for residue masses and defaults (50 ppm precursor tolerance, 20 to 150 peaks, m/z 50 to 2,500, max charge 4). The authors' own baseline ran Casanovo v5.0.0 with 5 beams (`src/instanovo_fm/baselines/README_casanovo.md`). |
| 01:15 | Casanovo's published nine-species numbers: the paper (Nature Communications 2024; preprint text fetched) gives averages across species and refers per-species values to Supplementary Figure S1, a figure. Decision: run Casanovo here on the same spectra as the primary comparison; cite the paper's averages as context only. |
| 01:22 | First smoke of the fork's sequencer, 200 mouse spectra on the CPU (93 s): the benchmark's DeepNovo spellings (`C+57.021`, `M+15.995`, `N+0.984`, `Q+0.984`, `+43.006`, `-17.027`) are not in the sequencer's remapping table, so evaluation mode **dropped 57 of 200 spectra silently** and reported peptide precision 0.629 on the rest. With the spellings added to the remapping (`denovo_ninespecies.yaml`) 200 of 200 are scored: peptide precision 0.585, amino-acid precision 0.744 by the fork's own metric. Flaw F2 in the README. |
| 01:28 | Corpus metadata flaw F1: `search_data.xlsx` gives the wrong organism for three of the nine projects (Vigna mungo as M. mazei, M. mazei as Mus musculus, Mus musculus as Vigna radiata) and marks the B. subtilis project DIA. Species in this study come from the benchmark's folders. All nine are Q Exactive HCD except M. mazei (Orbitrap Velos). |
| 01:30 | Sampler written and run: 20,000 spectra per species, seed 20261022, 180,000 in all, copied verbatim with manifest (`instanovo-fm-runs/nine-species/samples/`). |
| 01:35 | Scorer written (`nine_species_score.py`): Casanovo's match copied verbatim, one tokeniser for the three spellings, Casanovo v5 masses plus bare C and the fork's phospho tokens, peptide-clustered bootstrap, strata. Cross-check on the 200-spectrum smoke: peptide precision 0.585 = the fork's own 0.58500. The fork's amino-acid precision (0.744) differs from the scorer's (0.804): its own metric is not Casanovo's rule; not used further. |
| 01:40 | Collector written (`nine_species_collect.py`): the fork numbers spectra from 0 in `scan_number` (not the MGF's `SCANS=`); the join is by position and the parsed target is checked against the manifest label. Casanovo v5.1.2 installed in `envs/casanovo` with the v5.0.0 checkpoint (575 MB); its evaluate mode refuses the benchmark's spellings (and numeric ProForma): its checkpoint alphabet is the named form (`C[Carbamidomethyl]`, `[Acetyl]-`, `[+25.980265]-`, bare `C` also present). Plain sequence mode plus the shared scorer is the path; evaluate mode on a renamed file serves as the cross-check. Casanovo skips spectra with fewer than 20 peaks (4 of the 200). |
| 01:50 | Run scripts written for both sequencers. The GPU is busy with the Muon learning-rate sweep of the previous goal until about 02:10; the branch is created then (switching the working tree earlier would change the files a running process was started from). |
