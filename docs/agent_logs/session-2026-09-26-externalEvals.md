# Session log: InstaNovo-FM's evaluation suite on our own datasets
Date: 26/09/2026

## Initial purpose of the session

Run the evaluations of `docs/references/results_paper_or_rerun.md` on data that is ours, MS2Bac first and every
other dataset with verified labels on the workstation, on the whole of each dataset rather than a sample, for the
released 40M checkpoint, the 40M trained here (`session-2026-09-25-train40M.md`) and the released 89M; put the
scripts in `scripts/evals/` and one document per dataset in `docs/results/eval_<dataset>.md`; then read each
dataset's metadata (instrument, species, fragmentation, chromatography) against the corpus split the model is
evaluated on. Both clones (workstation, nibi) on `sdu/reproduce-results-40M`.

## What was accomplished during the session

| Task | What we learned |
| :---- | :---- |
| Sync: the workstation clone was on `main`; a `git pull` of the branch fast-forwarded local `main` to the branch head by mistake. Restored `main` to the upstream commit (5953cc4, unchanged upstream) and checked the branch out; nothing was pushed to `main`. | Pull into the branch you mean to be on; the fork's `main` is never touched. |
| Input contract of the evaluator, read from the code (a subagent traced it; the confirmed points were re-checked): five hard-required columns (`mz_array`, `intensity_array`, `precursor_mz`, `precursor_charge`, `sequence`), every name in `dataset.metadata_columns` must exist (`set_format` raises), `sequence` must be non-empty in every row because `is_annotated=True` is hard-coded (`eval/evaluator.py:305`) | The corpus schema (30 columns, read from `mcfm-test-00000-of-00014.parquet`) is the contract: write our data in it and the suite runs unchanged. Per-file metadata (instrument, detector, fragmentation, organism, enzyme) comes from `data/search_data.xlsx`, keyed by the project and file name of the `usi`; the probe target `search_instrument` and the "HCD Orbitrap" retrieval subset depend on it. `evaluation.max_samples=null` is the only "use everything"; `0` and `-1` yield an empty dataset. The probe embeds `train_path` / `valid_path` / `test_path` separately (caps 100,000 / 10,000 / 10,000); with `use_project_split=false` nothing else is filtered. |
| Sequence notation: the corpus writes `C[UNIMOD:4]`, `M[UNIMOD:35]`, `[UNIMOD:1]-` for N-terminal acetyl, `Q[UNIMOD:28]`, `E[UNIMOD:27]`; our label tables write `Name@pos` (1-based, 0 = N-terminus) and omit fixed carbamidomethyl for MaxQuant and Mascot searches | `prepare_dataset.py::proforma` maps one to the other and applies `C[UNIMOD:4]` to every C when the table carries no explicit carbamidomethyl (the rule of `proteomies-eval-data/src/ids.py::fixed_carbamidomethyl`). The evaluator's own fragment code adds carbamidomethyl to every bare C anyway (`common/dataset.py::clean_peptide_for_pyopenms`, `add_carbamidomethyl=True`), so an un-alkylated search would be mis-annotated by 57 Da with no config switch; none of ours is. The released checkpoints' residue set (32 tokens) lacks `Q[UNIMOD:28]` / `E[UNIMOD:27]`, which the corpus nevertheless contains; CAMPI has 12,301 and 1,265 such spectra (1.7 %), written the corpus way. |
| Export: `scripts/evals/prepare_dataset.py` reads the deposited mzML / MGF again with pyteomics using the index and scan conventions of `proteomies-eval-data/src/spectra.py`, joins the label table on the spectrum index and refuses any row whose scan number disagrees, and writes per dataset `-all` (every MS2), `-identified`, `-probe-{train,valid,test}` (peptide-disjoint 80/10/10, seed 42), a search-data row per run and a summary. The run facts (accession, instrument, lab, peak file, organism) come from `proteomies-eval-data/src/fm_manifest.py` (new, commit 01fe310 there) so the registry stays the one home. | MS2Bac: 42,410 MS2, 20,269 identified, 16,197 peptides, 46 s. UPS1 (PXD001819, LTQ Orbitrap Velos, ion-trap CID): 115,758 MS2, 27,023 identified, 5,941 peptides. ProteomeTools (11 runs): 639,632 MS2, 466,891 identified, 4,873 peptides. The first version held the whole dataset in memory (ProteomeTools peaked at the 16 GB ceiling of its scope); rewritten to stream one run per row group before CAMPI. |
| Evaluator change: `dataset.is_annotated` (default True) at the two places the flag was hard-coded (`eval/evaluator.py:255, 305`), so label-free tasks (embedding statistics, UMAP) run on the file that includes unidentified spectra | Smoke-tested on the workstation GPU with 600 MS2Bac spectra: statistics and UMAP run with `+dataset.is_annotated=false`; statistics and retrieval run on the identified file with the search-data lookup hitting our rows. |
| Protocols (`scripts/evals/run.py`): probes, retrieval, geometry, peak_level, unlabelled, validation (the trainer's own validation pass, the Step 2 recipe), one SLURM job each per (dataset, model), `submit_all.py` with per-protocol memory and time | "Entire dataset" is honoured wherever the cost is linear (retrieval pool and every duplicate group, statistics, validation, probes up to the package's 100,000-spectrum training cap). UMAP and EVoC (superlinear) fall back to the paper's 20,000 above 100,000 spectra, the per-peak tasks (embeddings of every peak in RAM, more than 30 GB at 20,000 spectra in the reproduction) to the paper's 10,000 above 25,000; each run.json records the cap. Probe train and test are peptide-disjoint here, as the corpus splits are. |
| Corpus overlap and metadata comparison: `scripts/evals/corpus_profile.py` profiles any corpus-schema file set (fragmentation, instrument, detector, organism, charge, collision energy, modifications, peaks per spectrum); `render_results.py` puts a dataset beside the MCFM and LCFM test splits and joins its peptides to the corpus `peptide_registry.parquet` | MS2Bac's identified spectra by corpus split of their peptide: train 39.8 %, validation 7.3 %, test 17.2 %, absent 35.7 % (the same as `instanovo-fm-evals` found on its solid view). The package's own table calls the ProteomeTools instrument "Orbitrap Fusion" where the deposit says Fusion Lumos; the evaluator uses the package's label for those files (their keys are in the table), ours for the other datasets. |
| Results: see `docs/results/eval_<dataset>.md` (filled as jobs finish; this row is updated at the end of the session) | pending |

## Problems encountered
- `is_annotated=True` hard-coded in the evaluator: a file with unidentified spectra is refused ("some or all
  sequence annotations are missing"). Fixed by the config key above; the alternative, a placeholder sequence,
  would have been fabricated data.
- The workstation clone was on `main` when the branch was pulled; local `main` was fast-forwarded and then
  restored (never pushed).
- The exporter's first `main` held the whole dataset (ProteomeTools reached its 16 GB scope ceiling, measured with
  `systemctl --user show`); rewritten to stream per run.
- The trainer's streaming schema (`to_dataset(force_unified_schema=True)`) has no Boolean type: it dropped the two
  flag columns from the schema and then refused the file ("Couldn't cast ... because column names don't match").
  The flags are now 0/1 integers; every dataset was re-exported.
- The first validation runs scored a random model: `foundational.yaml` compiles the model (`compile_model: True`),
  a compiled model's keys carry the `_orig_mod.` prefix, and `common/trainer.py::load_model_state` only warns
  ("Model keys do not match") when nothing matches, then continues (median error 1.6 million ppm, bin accuracy
  0.02 %). The reproduction's Step 2 had passed `compile_model=false`; the validation protocol now does too. A
  loader that proceeds on a total key mismatch is a flaw in the package: a validation-only run should refuse.
- Intensity scale: every corpus spectrum has a base peak of exactly 1 and `scale_factor` holds the raw magnitude
  (2,000 rows of `mcfm-test-00000`, checked); the processor's intensity floor (`min_intensity` 0.01, `data/data.py`
  step 3) is applied on that scale before the 200 most intense peaks are kept, so raw intensities kept sub-1 %
  peaks the corpus preprocessing removes. Found after 18 jobs had completed on raw intensities; those outputs were
  set aside (`$RESULTS/evals_superseded_raw_intensities`), every dataset was re-exported with base peak 1 and every
  job resubmitted. On raw intensities the released 40M had scored 66.0 % fragment-group bin accuracy on MS2Bac
  against 55.0 % on LCFM test; the rescaled numbers are the ones in `docs/results/`.
- Nine-species (MSV000090982) was not exported: every one of its PRIDE accessions is in the corpus's Table S1
  list (`assets/table_s1_accessions.txt`), so it is training data, not an external test. D-PSM was not exported:
  its labels are not verified (`proteomies-eval-data`, `docs/claude_logs/session-solid-eval-labels.md`).

## Next steps
- Filled in at the end of the session.
