"""Render `docs/results/eval_<dataset>.md` from the evaluation runs of `scripts/evals/run.py`.

    source .envrc && python scripts/evals/render_results.py --dataset ms2bac [--registry $DATA/peptide_registry.parquet]

One document per dataset: what the data is (runs, instrument, fragmentation, species, counts, from the export's
summary and `docs/results/profiles/<dataset>.json`), how much of it the model trained on (peptides joined to the
corpus's `peptide_registry.parquet`, I/L collapsed, modifications dropped), the dataset read against the corpus
splits the model is evaluated on (`docs/results/profiles/{mcfm,lcfm}-test.json`), then one table per protocol with
one column per checkpoint evaluated here and one reference column, the released 40M on the corpus test split
(`docs/references/rerun/result*_40M_*`, the reproduction), extracted by the same getters as
`scripts/reproduce/fill_results.py`. Every number names the JSON it came from; a cell is "-" until its job has run.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
RERUN = REPO / "docs" / "results" / "rerun"
REFERENCE = REPO / "docs" / "references" / "rerun"
PROFILES = REPO / "docs" / "results" / "profiles"
OUT = REPO / "docs" / "results"
MODELS = [("40M", "released 40M"), ("40M-ours", "40M trained here"), ("89M", "released 89M")]
Getter = Callable[[Path], str | None]


def load(folder: Path, name: str) -> Any:
    p = folder / name
    return json.load(p.open()) if p.exists() else None


def find_key(node: Any, key: str) -> Any:
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            hit = find_key(v, key)
            if hit is not None:
                return hit
    return None


def f3(x: Any) -> str:
    return f"{float(x):.3f}"


def f1(x: Any) -> str:
    return f"{float(x):.1f}"


def pct(x: Any) -> str:
    return f"{100 * float(x):.1f} %"


def key(name: str, k: str, fmt: Callable[[Any], str] = f3) -> Getter:
    """`k` at the top level of `<name>.json` (the task summary), or anywhere in `<name>.results.json`."""

    def get(folder: Path) -> str | None:
        d = load(folder, name)
        v = d.get(k) if isinstance(d, dict) and k in d else find_key(d, k)
        return None if v is None or isinstance(v, (dict, list)) or (isinstance(v, float) and v != v) else fmt(v)

    return get


def at(name: str, fmt: Callable[[Any], str], *keys: str | int) -> Getter:
    def get(folder: Path) -> str | None:
        node = load(folder, name)
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            elif isinstance(node, list) and isinstance(k, int) and k < len(node):
                node = node[k]
            else:
                return None
        return None if node is None or isinstance(node, (dict, list)) else fmt(node)

    return get


def ig(which: str) -> Getter:
    def get(folder: Path) -> str | None:
        q = find_key(load(folder, "igattributiontask.results.json"), "prediction_quality")
        if not isinstance(q, dict):
            return None
        ion = q.get("by_ion_type", {})
        if which == "overall":
            return pct(q["bin_accuracy"])
        if {"y", "b"} <= ion.keys():
            return (f"{pct(ion['y']['bin_accuracy'])} / {pct(ion['b']['bin_accuracy'])}" if which == "yb_acc"
                    else f"{f1(ion['y']['median_error_ppm'])} / {f1(ion['b']['median_error_ppm'])} ppm")
        return None

    return get


# (protocol, reference script of the released 40M on the corpus test split) -> rows (label, getter)
# The probe names its metric by the target's kind: macro-F1 for a multi-class target, balanced accuracy when it has two
# classes, R² for a regression, macro-F1 again when a numeric target has few distinct values (collision energy on a dataset
# with a handful of settings). A cell says which metric it holds when it is not the one in the row label.
PROBE_METRICS = {"macro_f1": "macro-F1", "balanced_accuracy": "balanced accuracy", "r2": "R²"}
PROBE_TARGETS = [
    ("Fragment type (macro-F1)", "frag_type", ("macro_f1", "balanced_accuracy")),
    ("Instrument (macro-F1)", "search_instrument", ("macro_f1", "balanced_accuracy")),
    ("PTM presence (balanced accuracy)", "ptm_present", ("balanced_accuracy", "macro_f1")),
    ("Modification class (macro-F1)", "modification_class", ("macro_f1", "balanced_accuracy")),
    ("Hydrophobicity (R²)", "hydrophobicity", ("r2",)), ("Precursor mass (R²)", "precursor_mass", ("r2",)), ("Precursor m/z (R²)", "precursor_mz", ("r2",)),
    ("Charge (macro-F1)", "precursor_charge", ("macro_f1", "balanced_accuracy")),
    ("Collision energy (R²)", "collision_energy", ("r2", "macro_f1", "balanced_accuracy")),
    ("Spectrum confidence (R²)", "spectrum_confidence", ("r2",)),
]


def probe(target: str, metrics: tuple[str, ...]) -> Getter:
    def get(folder: Path) -> str | None:
        d = load(folder, "linearprobetask.json") or {}
        for i, m in enumerate(metrics):
            v = d.get(f"{target}/{m}")
            if v is not None and not (isinstance(v, float) and v != v):
                return f3(v) if i == 0 else f"{f3(v)} ({PROBE_METRICS[m]})"
        return None

    return get
TABLES: list[tuple[str, str, str, list[tuple[str, Getter]]]] = [
    ("probes", "Linear probes", "result1_40M_probes_retrieval",
     [(label, probe(target, metrics)) for label, target, metrics in PROBE_TARGETS]),
    ("retrieval", "Duplicate-spectrum retrieval", "result1_40M_probes_retrieval",
     [("Recall@1", key("duplicateretrievaltask.json", "recall@1")), ("mAP@20", key("duplicateretrievaltask.json", "map@20")),
      ("Proportional recall@1", key("duplicateretrievaltask.json", "prop_recall@1")),
      ("Recall@1, HCD Orbitrap subset", key("duplicateretrievaltask.json", "cond_hcd_orbitrap_recall@1")),
      ("Recall@1, CID subset", key("duplicateretrievaltask.json", "cond_cid_recall@1"))]),
    ("peak_level", "Peak level", "result2_40M_peak_level",
     [("Fragment-group bin accuracy (IG task)", ig("overall")), ("Bin accuracy y / b ions", ig("yb_acc")), ("Median error y / b ions", ig("yb_ppm")),
      ("Peak-type accuracy", at("peaktypeclassificationtask.results.json", pct, "multiclass_classification", "accuracy")),
      ("Peak-type macro-F1", key("peaktypeclassificationtask.json", "multiclass_macro_f1")),
      ("Cross-spectrum AUROC", key("peaktypeclassificationtask.json", "cross_spectrum_auroc")),
      ("Confidence AUROC, per spectrum / pooled", lambda f: " / ".join(v for v in (key("confidencesignalanalysistask.json", "per_spectrum_auroc_mean")(f), key("confidencesignalanalysistask.json", "auroc")(f)) if v) or None),
      ("Structural attention heads", key("headanalysistask.json", "n_structural_heads", lambda x: str(int(float(x))))),
      ("Isotope-spacing enrichment (mean)", key("headanalysistask.json", "mean_isotope_spacing_enrichment"))]),
    ("geometry", "Embedding geometry", "result3_40M_geometry",
     [("Anisotropy ratio", key("embeddingstatisticstask.json", "anisotropy_ratio", f1)), ("Effective rank", key("embeddingstatisticstask.json", "effective_rank", f1)),
      ("Top-component energy", key("embeddingstatisticstask.json", "pca_energy_top1")), ("Mean cosine", key("embeddingstatisticstask.json", "mean_similarity")),
      ("UMAP kNN preservation (k=15)", key("umapvisualisationtask.json", "knn_preservation_k15")),
      ("EVoC clusters / noise / purity", lambda f: " / ".join(v for v in (key("evocclusteringtask.json", "n_clusters", lambda x: str(int(float(x))))(f), key("evocclusteringtask.json", "noise_fraction")(f), key("evocclusteringtask.json", "mean_purity_frag_type")(f)) if v) or None),
      ("ESM2 alignment RSA / CKA", lambda f: " / ".join(v for v in (key("esm2crossmodalalignmenttask.json", "rsa_all_pairs_rho")(f), key("esm2crossmodalalignmenttask.json", "cka_score")(f)) if v) or None),
      ("ESM2 metadata-baseline RSA", key("esm2crossmodalalignmenttask.json", "baseline_metadata_rsa_diff")),
      ("Glass Box residual / top importance", lambda f: " / ".join(v for v in (key("glassboxattributiontask.json", "reconstruction_residual", lambda x: f"{float(x):.2e}")(f), key("glassboxattributiontask.json", "top_feature_importance")(f)) if v) or None),
      ("Cosine-hyperscore Spearman (pairs)", lambda f: (lambda s, n: f"{s} ({n})" if s and n else s)(key("cosinehyperscorecorrelationtask.results.json", "spearman_correlation")(f), key("cosinehyperscorecorrelationtask.results.json", "num_correlation_pairs", lambda x: str(int(x)))(f)))]),
    ("unlabelled", "Every MS2 spectrum, identified or not", "",
     [("Anisotropy ratio", key("embeddingstatisticstask.json", "anisotropy_ratio", f1)), ("Effective rank", key("embeddingstatisticstask.json", "effective_rank", f1)),
      ("Top-component energy", key("embeddingstatisticstask.json", "pca_energy_top1")), ("Mean cosine", key("embeddingstatisticstask.json", "mean_similarity")),
      ("UMAP kNN preservation (k=15)", key("umapvisualisationtask.json", "knn_preservation_k15"))]),
    ("validation", "Trainer validation (masked-peak reconstruction)", "validate_released_40M",
     [("Median |error| over masked peaks", key("metrics.json", "eval/median_ae_ppm", lambda x: f"{float(x):.0f} ppm")),
      ("MAE", key("metrics.json", "eval/mae_daltons", lambda x: f"{float(x):.2f} Da")),
      ("Bin accuracy (0.2 Da)", key("metrics.json", "eval/bin_accuracy", lambda x: f"{float(x):.1f} %")),
      ("Within 20 ppm", key("metrics.json", "eval/pct_within_20ppm", lambda x: f"{float(x):.1f} %")),
      ("Intensity R²", key("metrics.json", "eval/intensity_r2"))]),
]
REFERENCE_SPLIT = {"result1_40M_probes_retrieval": "LCFM test", "result2_40M_peak_level": "LCFM test", "result3_40M_geometry": "LCFM test", "validate_released_40M": "MCFM validation"}


def peptide_key(sequence: str) -> str:
    return re.sub(r"\[UNIMOD:\d+\]-?", "", sequence).replace("I", "L")


def overlap(identified: Path, registry: Path | None) -> pd.Series | None:
    """Spectra by corpus split of their peptide: train / valid / test / absent."""
    if registry is None or not registry.exists():
        return None
    keys = pd.Series([peptide_key(s) for s in pq.read_table(identified, columns=["sequence"])["sequence"].to_pylist()])
    reg = pd.read_parquet(registry, columns=["peptide", "split"]).drop_duplicates("peptide").set_index("peptide")["split"]
    return keys.map(reg).fillna("absent").value_counts()


def top(d: dict[str, int] | None, n: int = 4) -> str:
    if not d:
        return "-"
    total = sum(d.values())
    return "; ".join(f"{k} {100 * v / total:.0f} %" for k, v in list(d.items())[:n])


def run_meta(folder: Path) -> str:
    m = load(folder, "run.json")
    if not m:
        return "-"
    n = m.get("rows_in_file")
    cap = m.get("max_samples")
    return f"{cap:,} of {n:,}" if cap else f"all {n:,}"


def render(dataset: str, registry: Path | None, external: Path) -> str:
    summary = json.load((external / dataset / f"{dataset}-summary.json").open())
    profile = load(PROFILES, f"{dataset}.json") or {}
    search = pd.read_csv(external / dataset / f"{dataset}-search_data.csv")
    engine = ", ".join(sorted(set(pq.read_table(external / dataset / f"{dataset}-identified.parquet", columns=["engine"])["engine"].to_pylist())))
    lines = [f"# InstaNovo-FM evaluation suite on {dataset}", ""]
    lines += [f"Rendered by `scripts/evals/render_results.py` from `docs/results/rerun/{dataset}/<model>/<protocol>/` (jobs of "
              f"`scripts/evals/run.py`, data from `scripts/evals/prepare_dataset.py`). Reference column: the released 40M on the corpus "
              f"split named in the header, from the reproduction (`docs/references/results_paper_or_rerun.md`). \"-\" = not run yet or the task reported nothing.", ""]
    t = summary["totals"]
    lines += ["## The data", "",
              f"- {len(summary['runs'])} runs, {t['ms2']:,} MS2 spectra, {t['identified']:,} identified with an expressible sequence, {t['peptides']:,} distinct peptides "
              f"(I/L collapsed, modifications dropped); probe split {t['probe_split_spectra']['train']:,} / {t['probe_split_spectra']['valid']:,} / {t['probe_split_spectra']['test']:,} spectra "
              f"(train / valid / test, peptide-disjoint, seed 42).",
              f"- Instrument: {'; '.join(sorted(set(search.instrument)))}; fragmentation: {'; '.join(sorted(set(search.fragmentation)))}; detector: {'; '.join(sorted(set(search.detector)))}; "
              f"organism: {'; '.join(sorted(set(search.organism)))}. Search engine: {engine}. "
              f"Carbamidomethyl treated as fixed: {summary['fixed_carbamidomethyl']}.", ""]
    lines += ["| run | MS2 | identified | unsupported modifications | peptides | median peaks | charges of identified |", "|---|---:|---:|---:|---:|---:|---|"]
    for run, c in summary["runs"].items():
        charges = ", ".join(f"{k}+ {v:,}" for k, v in sorted(c["charges_identified"].items(), key=lambda kv: int(kv[0])))
        lines.append(f"| {run} | {c['ms2']:,} | {c['identified']:,} | {c.get('unsupported_modification', 0):,} | {c['peptides']:,} | {c['median_peaks']:.0f} | {charges} |")
    lines.append("")
    ov = overlap(external / dataset / f"{dataset}-identified.parquet", registry)
    if ov is not None:
        n = int(ov.sum())
        lines += ["## How much of it the model trained on", "",
                  "Identified spectra by the corpus split of their peptide (`peptide_registry.parquet` of `InstaDeepAI/InstaNovo`, peptide-disjoint 80/10/10; "
                  "`absent` = not in the corpus at all). The released models trained on the train split; a spectrum here is never itself in the corpus unless the run is.", "",
                  "| split | spectra | share |", "|---|---:|---:|"]
        order = [s for s in ("train", "validation", "valid", "test", "absent") if s in ov.index] + [s for s in ov.index if s not in ("train", "validation", "valid", "test", "absent")]
        lines += [f"| {s} | {int(ov[s]):,} | {100 * ov[s] / n:.1f} % |" for s in order]
        lines.append("")
    refs = {name: load(PROFILES, f"{name}.json") for name in ("mcfm-test", "lcfm-test")}
    if profile:
        lines += ["## Read against the corpus test splits", "",
                  "From `docs/results/profiles/*.json` (`scripts/evals/corpus_profile.py`, metadata columns plus the package's per-file table). Percentages within each set; top values only.", "",
                  "| | " + " | ".join([dataset, "MCFM test", "LCFM test"]) + " |", "|---|---|---|---|"]
        sets = [profile, refs["mcfm-test"] or {}, refs["lcfm-test"] or {}]
        rows = [("spectra", lambda p: f"{p.get('spectra', 0):,}" if p else "-"), ("identified", lambda p: f"{p.get('identified', 0):,}" if p else "-"),
                ("runs / projects", lambda p: f"{p.get('runs')} / {len(p.get('projects', {}))}" if p else "-"),
                ("fragmentation (`frag_type`)", lambda p: top(p.get("frag_type"))), ("instrument (search data)", lambda p: top(p.get("search_instrument"))),
                ("detector (search data)", lambda p: top(p.get("search_detector"))), ("organism (search data)", lambda p: top(p.get("search_organism"), 5)),
                ("acquisition", lambda p: top(p.get("acquisition"))), ("collision energy", lambda p: top(p.get("collision_energy"), 5)),
                ("precursor charge", lambda p: top(p.get("precursor_charge"), 5)), ("modified sequences", lambda p: pct(p["modified_fraction"]) if p.get("modified_fraction") is not None else "-"),
                ("modification tokens", lambda p: top(p.get("modification_tokens"), 4)),
                ("peaks per spectrum (median, p5-p95)", lambda p: f"{p['peaks_per_spectrum']['median']:.0f} ({p['peaks_per_spectrum']['p05']:.0f}-{p['peaks_per_spectrum']['p95']:.0f})" if p.get("peaks_per_spectrum") else "-"),
                ("precursor m/z (median, p5-p95)", lambda p: f"{p['precursor_mz']['median']:.0f} ({p['precursor_mz']['p05']:.0f}-{p['precursor_mz']['p95']:.0f})" if p.get("precursor_mz") else "-"),
                ("peptide length (median, p5-p95)", lambda p: f"{p['peptide_length']['median']:.0f} ({p['peptide_length']['p05']:.0f}-{p['peptide_length']['p95']:.0f})" if p.get("peptide_length") else "-")]
        lines += [f"| {label} | " + " | ".join(get(p) for p in sets) + " |" for label, get in rows]
        lines.append("")
    lines += ["## Results", ""]
    for protocol, title, reference, rows_ in TABLES:
        folders = [(label, RERUN / dataset / tag / protocol) for tag, label in MODELS]
        ref_folder = REFERENCE / reference if reference else None
        header = [f"{label} ({run_meta(f)})" for label, f in folders] + ([f"released 40M, {REFERENCE_SPLIT[reference]} (reproduction)"] if ref_folder else [])
        lines += [f"### {title}", "", "| metric | " + " | ".join(header) + " |", "|---|" + "---|" * len(header)]
        for label, get in rows_:
            cells = [get(f) or "-" for _, f in folders] + ([get(ref_folder) or "-"] if ref_folder else [])
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        note = {"probes": "Probe train / valid / test are this dataset's own peptide-disjoint files (package caps 100,000 / 10,000 / 10,000). A one-class target (one instrument, one fragmentation) cannot be probed and shows \"-\". A cell names its metric when the probe chose another than the row label's (balanced accuracy for a two-class target, macro-F1 for a collision energy with few settings); the probe scores only the classes present in its test split.",
                "retrieval": "Every identified spectrum is in the pool and every duplicate group is queried; a positive is an identical peptide string (charge ignored).",
                "peak_level": "Theoretical b/y ions from the sequence with the checkpoint's residue masses; the column header says how many spectra were used.",
                "geometry": "Header says how many spectra were used (the paper's 20,000 above 100,000). EVoC purity is by fragmentation type and is 1 by construction when a dataset has one. "
                            "The cosine-hyperscore correlation needs a hyperscore (X!Tandem or MSFragger); a dataset searched with another engine has none and shows \"-\". "
                            "Glass Box attribution returned the same values for every checkpoint here and in the reproduction (released 40M against the one trained here); its output does not separate models.",
                "unlabelled": "Identified and unidentified spectra together (`dataset.is_annotated=false`); no reference column because the corpus holds identified spectra only.",
                "validation": "The trainer's validation loop on every identified spectrum (masking in the collate, seed fixed), the criterion the trainer selects checkpoints on."}[protocol]
        lines += ["", note, ""]
    lines += ["## Sources", "", f"- Data: `$EXTERNAL/{dataset}/` (`prepare_dataset.py`, manifest `proteomies-eval-data/outputs/fm_manifest.csv`, summary `{dataset}-summary.json`).",
              f"- Runs: `docs/results/rerun/{dataset}/<model>/<protocol>/run.json` (overrides, commit, host, seconds, rows in file, cap).", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--external", type=Path, default=None, help="defaults to $EXTERNAL")
    ap.add_argument("--registry", type=Path, default=None, help="the corpus peptide_registry.parquet, for the overlap table")
    args = ap.parse_args()
    import os

    external = args.external or Path(os.path.expandvars(os.path.expanduser(os.environ["EXTERNAL"])))
    text = render(args.dataset, args.registry, external)
    out = OUT / f"eval_{args.dataset}.md"
    out.write_text(text)
    print(f"wrote {out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
