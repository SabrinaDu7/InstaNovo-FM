"""Fill the "Result from our rerun" and its "Source" columns of `docs/references/results_paper_or_rerun.md`
from the task summaries the result scripts copied into `docs/references/rerun/<script>/`.

Each numbered row of the document is mapped here to one script, one task file and one metric key (or a small
formatter over several keys). Rows whose files or keys are absent are left untouched and listed, so the
document never shows a number that has no JSON behind it. The source cell names the file and the commit
recorded in the run's `run.json`.

    python scripts/reproduce/fill_results.py            # rewrites the document in place
    python scripts/reproduce/fill_results.py --check    # exit 1 if the document is not already up to date
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "references" / "results_paper_or_rerun.md"
RERUN = REPO / "docs" / "references" / "rerun"


def load(script: str, name: str) -> dict[str, Any] | None:
    path = RERUN / script / name
    return json.load(path.open()) if path.exists() else None


def find(node: Any, *tokens: str) -> Any:
    """First leaf under `node` whose key path contains `tokens` in order (a subsequence), depth first."""

    def walk(n: Any, path: tuple[str, ...]) -> Any:
        if isinstance(n, dict):
            for k, v in n.items():
                hit = walk(v, (*path, str(k)))
                if hit is not None:
                    return hit
            return None
        it = iter(path)
        return n if all(any(t in p for p in it) for t in tokens) else None

    return walk(node, ())


def find_key(node: Any, key: str) -> Any:
    """Value of the first dict entry named exactly `key`, depth first (for dict-valued blocks)."""
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


def pct(x: Any) -> str:
    return f"{100 * float(x):.1f} %"


def probe(script: str, key: str) -> Callable[[], str | None]:
    return lambda: (lambda d: None if d is None or key not in d else f3(d[key]))(load(script, "linearprobetask.json"))


def summary(script: str, task: str, key: str, fmt: Callable[[Any], str] = f3) -> Callable[[], str | None]:
    return lambda: (lambda d: None if d is None or key not in d else fmt(d[key]))(load(script, f"{task}.json"))


def ig_quality(script: str, which: str) -> Callable[[], str | None]:
    def get() -> str | None:
        d = load(script, "igattributiontask.results.json")
        q = find_key(d, "prediction_quality") if d else None
        if not isinstance(q, dict):
            return None
        ion = q.get("by_ion_type", {})
        if which == "overall":
            return pct(q["bin_accuracy"])
        if which == "yb_acc" and {"y", "b"} <= ion.keys():
            return f"{pct(ion['y']['bin_accuracy'])} / {pct(ion['b']['bin_accuracy'])}"
        if which == "yb_ppm" and {"y", "b"} <= ion.keys():
            return f"{float(ion['y']['median_error_ppm']):.1f} / {float(ion['b']['median_error_ppm']):.1f} ppm"
        if which == "all":
            parts = [f"{pct(q['bin_accuracy'])} overall"]
            if {"y", "b"} <= ion.keys():
                parts.append(f"y {pct(ion['y']['bin_accuracy'])} / b {pct(ion['b']['bin_accuracy'])}")
                parts.append(f"{float(ion['y']['median_error_ppm']):.1f} / {float(ion['b']['median_error_ppm']):.1f} ppm")
            return "; ".join(parts)
        return None

    return get


def nested(script: str, name: str, fmt: Callable[[Any], str], *tokens: str) -> Callable[[], str | None]:
    def get() -> str | None:
        d = load(script, name)
        v = find(d, *tokens) if d else None
        return None if v is None or isinstance(v, (dict, list)) else fmt(v)

    return get


def at(script: str, name: str, fmt: Callable[[Any], str], *keys: str | int) -> Callable[[], str | None]:
    """Exact path lookup: `keys` are dict keys or list indices, walked in order."""

    def get() -> str | None:
        node: Any = load(script, name)
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            elif isinstance(node, list) and isinstance(k, int) and k < len(node):
                node = node[k]
            else:
                return None
        return None if isinstance(node, (dict, list)) or node is None else fmt(node)

    return get


def per_class(script: str, metric: str, class_name: str, fmt: Callable[[Any], str] = f3) -> Callable[[], str | None]:
    """A per-class value of the peak-type multiclass probe, addressed by class name (`class_names` gives the order)."""

    def get() -> str | None:
        d = load(script, "peaktypeclassificationtask.results.json")
        mc = (d or {}).get("multiclass_classification") or {}
        names, values = mc.get("class_names") or [], mc.get(metric) or []
        return fmt(values[names.index(class_name)]) if class_name in names and len(values) == len(names) else None

    return get


def keyed(script: str, name: str, key: str, fmt: Callable[[Any], str] = f3) -> Callable[[], str | None]:
    """The first value stored under exactly `key`, at any depth of the file."""

    def get() -> str | None:
        v = find_key(load(script, name), key)
        return None if v is None or isinstance(v, (dict, list)) else fmt(v)

    return get


def joined(*parts: tuple[str, Callable[[], str | None]]) -> Callable[[], str | None]:
    def get() -> str | None:
        got = [(label, p()) for label, p in parts]
        got = [f"{label} {v}" if label else v for label, v in got if v is not None]
        return "; ".join(got) or None

    return get


R1_40, R2_40, R3_40 = "result1_40M_probes_retrieval", "result2_40M_peak_level", "result3_40M_geometry"
R1_89, R2_89, R4_89 = "result1_89M_probes_retrieval", "result2_89M_peak_level", "result4_89M_factorial_validation"
R3_89 = "result3_89M_geometry"
R5_40 = "result5_40M_mcfm_test"
R6_40, R6_89 = "result6_40M_cosine_hyperscore", "result6_89M_cosine_hyperscore"

ROWS: dict[tuple[str, int], tuple[str, str, Callable[[], str | None]]] = {  # (section, No.) -> (script, file, getter)
    ("40M", 1): (R1_40, "linearprobetask.json", probe(R1_40, "frag_type/macro_f1")),
    ("40M", 2): (R1_40, "linearprobetask.json", probe(R1_40, "search_instrument/macro_f1")),
    ("40M", 3): (R1_40, "linearprobetask.json", probe(R1_40, "ptm_present/balanced_accuracy")),
    ("40M", 4): (R1_40, "linearprobetask.json", probe(R1_40, "hydrophobicity/r2")),
    ("40M", 5): (R1_40, "linearprobetask.json", probe(R1_40, "precursor_mass/r2")),
    ("40M", 6): (R1_40, "linearprobetask.json", probe(R1_40, "precursor_mz/r2")),
    ("40M", 7): (R1_40, "linearprobetask.json", probe(R1_40, "precursor_charge/macro_f1")),
    ("40M", 8): (R1_40, "linearprobetask.json", probe(R1_40, "spectrum_confidence/r2")),
    ("40M", 9): (R1_40, "duplicateretrievaltask.json", summary(R1_40, "duplicateretrievaltask", "recall@1")),
    ("40M", 10): (R1_40, "duplicateretrievaltask.json", summary(R1_40, "duplicateretrievaltask", "map@20")),
    ("40M", 11): (R2_40, "igattributiontask.results.json", ig_quality(R2_40, "all")),
    ("40M", 12): (R2_40, "peaktypeclassificationtask.json", joined(("accuracy", at(R2_40, "peaktypeclassificationtask.results.json", pct, "multiclass_classification", "accuracy")), ("macro-F1", summary(R2_40, "peaktypeclassificationtask", "multiclass_macro_f1")))),
    ("40M", 13): (R2_40, "peaktypeclassificationtask.json", summary(R2_40, "peaktypeclassificationtask", "cross_spectrum_auroc")),
    ("40M", 14): (R2_40, "confidencesignalanalysistask.json", joined(("per-spectrum mean", summary(R2_40, "confidencesignalanalysistask", "per_spectrum_auroc_mean")), ("pooled", summary(R2_40, "confidencesignalanalysistask", "auroc")))),
    ("40M", 15): (R3_40, "embeddingstatisticstask.json", joined(("anisotropy ratio", summary(R3_40, "embeddingstatisticstask", "anisotropy_ratio", lambda x: f"{float(x):.1f}")), ("effective rank", summary(R3_40, "embeddingstatisticstask", "effective_rank", lambda x: f"{float(x):.1f}")), ("top-component energy", summary(R3_40, "embeddingstatisticstask", "pca_energy_top1")), ("mean cosine", summary(R3_40, "embeddingstatisticstask", "mean_similarity")))),
    ("40M", 16): (R5_40, "linearprobetask.json", joined(*[("", probe(R5_40, k)) for k in ("frag_type/macro_f1", "search_instrument/macro_f1", "ptm_present/balanced_accuracy", "hydrophobicity/r2", "precursor_mass/r2", "precursor_mz/r2", "precursor_charge/macro_f1", "spectrum_confidence/r2")])),
    ("40M", 17): (R5_40, "duplicateretrievaltask.json", joined(("", summary(R5_40, "duplicateretrievaltask", "recall@1")), ("", summary(R5_40, "duplicateretrievaltask", "map@20")))),
    ("40M", 18): (R3_40, "umapvisualisationtask.json", joined(("", summary(R3_40, "umapvisualisationtask", "knn_preservation_k15")), ("HCD-Orbitrap", summary(R3_40, "umapvisualisationtask", "cond_hcd_orbitrap_knn_preservation_k15")), ("CID", summary(R3_40, "umapvisualisationtask", "cond_cid_knn_preservation_k15")))),
    ("40M", 19): (R3_40, "evocclusteringtask.json", joined(("clusters", summary(R3_40, "evocclusteringtask", "n_clusters", lambda x: str(int(float(x))))), ("noise", summary(R3_40, "evocclusteringtask", "noise_fraction")), ("silhouette", summary(R3_40, "evocclusteringtask", "silhouette")), ("purity", summary(R3_40, "evocclusteringtask", "mean_purity_frag_type")))),
    ("40M", 20): (R3_40, "esm2crossmodalalignmenttask.json", joined(("RSA", summary(R3_40, "esm2crossmodalalignmenttask", "rsa_all_pairs_rho")), ("CKA", summary(R3_40, "esm2crossmodalalignmenttask", "cka_score")), ("shuffled RSA", summary(R3_40, "esm2crossmodalalignmenttask", "baseline_shuffled_rsa_diff")), ("metadata RSA", summary(R3_40, "esm2crossmodalalignmenttask", "baseline_metadata_rsa_diff")))),
    ("40M", 21): (R3_40, "glassboxattributiontask.json", joined(("features", summary(R3_40, "glassboxattributiontask", "n_features", lambda x: str(int(float(x))))), ("residual", summary(R3_40, "glassboxattributiontask", "reconstruction_residual")), ("top importance", summary(R3_40, "glassboxattributiontask", "top_feature_importance")))),
    ("40M", 22): (R6_40, "cosinehyperscorecorrelationtask.results.json", joined(("Spearman", keyed(R6_40, "cosinehyperscorecorrelationtask.results.json", "spearman_correlation")), ("Pearson", keyed(R6_40, "cosinehyperscorecorrelationtask.results.json", "pearson_correlation")), ("pairs", keyed(R6_40, "cosinehyperscorecorrelationtask.results.json", "num_correlation_pairs", lambda x: str(int(x)))))),
    ("89M", 41): (R3_89, "embeddingstatisticstask.json", joined(("anisotropy ratio", summary(R3_89, "embeddingstatisticstask", "anisotropy_ratio", lambda x: f"{float(x):.1f}")), ("effective rank", summary(R3_89, "embeddingstatisticstask", "effective_rank", lambda x: f"{float(x):.1f}")), ("top-component energy", summary(R3_89, "embeddingstatisticstask", "pca_energy_top1")), ("mean cosine", summary(R3_89, "embeddingstatisticstask", "mean_similarity")))),
    ("89M", 42): (R3_89, "umapvisualisationtask.json", joined(("", summary(R3_89, "umapvisualisationtask", "knn_preservation_k15")), ("HCD-Orbitrap", summary(R3_89, "umapvisualisationtask", "cond_hcd_orbitrap_knn_preservation_k15")), ("CID", summary(R3_89, "umapvisualisationtask", "cond_cid_knn_preservation_k15")))),
    ("89M", 43): (R3_89, "evocclusteringtask.json", joined(("clusters", summary(R3_89, "evocclusteringtask", "n_clusters", lambda x: str(int(float(x))))), ("noise", summary(R3_89, "evocclusteringtask", "noise_fraction")), ("silhouette", summary(R3_89, "evocclusteringtask", "silhouette")), ("purity", summary(R3_89, "evocclusteringtask", "mean_purity_frag_type")))),
    ("89M", 44): (R3_89, "esm2crossmodalalignmenttask.json", joined(("RSA", summary(R3_89, "esm2crossmodalalignmenttask", "rsa_all_pairs_rho")), ("CKA", summary(R3_89, "esm2crossmodalalignmenttask", "cka_score")), ("shuffled RSA", summary(R3_89, "esm2crossmodalalignmenttask", "baseline_shuffled_rsa_diff")), ("metadata RSA", summary(R3_89, "esm2crossmodalalignmenttask", "baseline_metadata_rsa_diff")))),
    ("89M", 45): (R3_89, "glassboxattributiontask.json", joined(("features", summary(R3_89, "glassboxattributiontask", "n_features", lambda x: str(int(float(x))))), ("residual", summary(R3_89, "glassboxattributiontask", "reconstruction_residual")), ("top importance", summary(R3_89, "glassboxattributiontask", "top_feature_importance")))),
    ("89M", 46): (R6_89, "cosinehyperscorecorrelationtask.results.json", joined(("Spearman", keyed(R6_89, "cosinehyperscorecorrelationtask.results.json", "spearman_correlation")), ("Pearson", keyed(R6_89, "cosinehyperscorecorrelationtask.results.json", "pearson_correlation")), ("pairs", keyed(R6_89, "cosinehyperscorecorrelationtask.results.json", "num_correlation_pairs", lambda x: str(int(x)))))),
    ("89M", 1): (R1_89, "linearprobetask.json", probe(R1_89, "frag_type/macro_f1")),
    ("89M", 2): (R1_89, "linearprobetask.json", probe(R1_89, "search_instrument/macro_f1")),
    ("89M", 3): (R1_89, "linearprobetask.json", probe(R1_89, "ptm_present/balanced_accuracy")),
    ("89M", 4): (R1_89, "linearprobetask.json", probe(R1_89, "modification_class/macro_f1")),
    ("89M", 5): (R1_89, "linearprobetask.json", probe(R1_89, "hydrophobicity/r2")),
    ("89M", 6): (R1_89, "linearprobetask.json", probe(R1_89, "precursor_mass/r2")),
    ("89M", 7): (R1_89, "linearprobetask.json", probe(R1_89, "precursor_mz/r2")),
    ("89M", 8): (R1_89, "linearprobetask.json", probe(R1_89, "precursor_charge/macro_f1")),
    ("89M", 9): (R1_89, "linearprobetask.json", probe(R1_89, "spectrum_confidence/r2")),
    ("89M", 10): (R1_89, "duplicateretrievaltask.json", summary(R1_89, "duplicateretrievaltask", "recall@1")),
    ("89M", 11): (R1_89, "duplicateretrievaltask.json", summary(R1_89, "duplicateretrievaltask", "map@20")),
    ("89M", 12): (R1_89, "linearprobetask.results.json", nested(R1_89, "linearprobetask.results.json", f3, "precursor_charge", "macro_auroc")),
    ("89M", 13): (R1_89, "linearprobetask.results.json", nested(R1_89, "linearprobetask.results.json", f3, "frag_type", "macro_auroc")),
    ("89M", 14): (R2_89, "igattributiontask.results.json", ig_quality(R2_89, "overall")),
    ("89M", 15): (R2_89, "igattributiontask.results.json", ig_quality(R2_89, "yb_acc")),
    ("89M", 16): (R2_89, "igattributiontask.results.json", ig_quality(R2_89, "yb_ppm")),
    ("89M", 17): (R2_89, "confidencesignalanalysistask.json", summary(R2_89, "confidencesignalanalysistask", "per_spectrum_auroc_mean")),
    ("89M", 18): (R2_89, "peaktypeclassificationtask.results.json", joined(("after transformer", at(R2_89, "peaktypeclassificationtask.results.json", pct, "multiclass_classification", "accuracy")), ("pre-transformer", at(R2_89, "peaktypeclassificationtask.results.json", pct, "pretransformer_probe", "pretransformer_multiclass", "accuracy")))),
    ("89M", 19): (R2_89, "peaktypeclassificationtask.json", summary(R2_89, "peaktypeclassificationtask", "multiclass_macro_f1")),
    ("89M", 20): (R2_89, "peaktypeclassificationtask.results.json", joined(("precision", per_class(R2_89, "per_class_precision", "unannotated")), ("F1", per_class(R2_89, "per_class_f1", "unannotated")))),
    ("89M", 21): (R2_89, "peaktypeclassificationtask.results.json", joined(("y", per_class(R2_89, "per_class_f1", "y-ion")), ("b", per_class(R2_89, "per_class_f1", "b-ion")))),
    ("89M", 22): (R2_89, "peaktypeclassificationtask.json", summary(R2_89, "peaktypeclassificationtask", "cross_spectrum_auroc")),
    ("89M", 23): (R2_89, "peaktypeclassificationtask.results.json", joined(("same ion", at(R2_89, "peaktypeclassificationtask.results.json", f3, "cross_spectrum_identity", "same_ion_cross_spectrum", "mean")), ("m/z-matched", at(R2_89, "peaktypeclassificationtask.results.json", f3, "cross_spectrum_identity", "different_ion_similar_mz", "mean")), ("random", at(R2_89, "peaktypeclassificationtask.results.json", f3, "cross_spectrum_identity", "random_cross_spectrum", "mean")))),
    ("89M", 25): (R2_89, "igattributiontask.json", joined(("top-1", summary(R2_89, "igattributiontask", "topk_top1_ladder", pct)), ("top-5", summary(R2_89, "igattributiontask", "topk_ladder_top5_hit_rate", pct)))),
    ("89M", 26): (R4_89, "linearprobetask.json", probe(R4_89, "frag_type/macro_f1")),
    ("89M", 27): (R4_89, "linearprobetask.json", probe(R4_89, "search_instrument/macro_f1")),
    ("89M", 28): (R4_89, "linearprobetask.json", probe(R4_89, "spectrum_confidence/r2")),
    ("89M", 29): (R4_89, "linearprobetask.json", probe(R4_89, "ptm_present/balanced_accuracy")),
    ("89M", 30): (R4_89, "linearprobetask.json", probe(R4_89, "modification_class/macro_f1")),
    ("89M", 31): (R4_89, "linearprobetask.json", probe(R4_89, "precursor_mz/r2")),
    ("89M", 33): (R4_89, "duplicateretrievaltask.json", joined(("", summary(R4_89, "duplicateretrievaltask", "recall@1")), ("CID", summary(R4_89, "duplicateretrievaltask", "cond_cid_recall@1")))),
}


def source_cell(script: str, name: str) -> str:
    run = load(script, "run.json") or {}
    return f"`rerun/{script}/{name}`" + (f" @ {run['commit']}" if run.get("commit") else "")


def fill(text: str) -> tuple[str, list[str]]:
    out, missing, section = [], [], ""
    for line in text.split("\n"):
        if line.startswith("## "):
            section = "40M" if "40M" in line else "89M" if "89M" in line else ""
        cells = line.split("|")
        if section and len(cells) >= 8 and cells[1].strip().isdigit():
            key = (section, int(cells[1]))
            if key in ROWS:
                script, name, getter = ROWS[key]
                value = getter()
                if value is None:
                    missing.append(f"{section} row {key[1]}: {script}/{name}")
                else:
                    cells[3], cells[4] = f" {value} ", f" {source_cell(script, name)} "
                    line = "|".join(cells)
        out.append(line)
    return "\n".join(out), missing


def main() -> int:
    text = DOC.read_text()
    new, missing = fill(text)
    if "--check" in sys.argv:
        print("up to date" if new == text else "stale")
        return 0 if new == text else 1
    DOC.write_text(new)
    print(f"filled {DOC.relative_to(REPO)}; {len(missing)} rows without a rerun file or key:")
    print("\n".join(f"  {m}" for m in missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
