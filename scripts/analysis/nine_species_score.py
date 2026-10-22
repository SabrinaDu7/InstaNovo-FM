"""Score de novo predictions on the nine-species benchmark with Casanovo's own matching rule.

Analysis only. Input: a predictions table (parquet or CSV) with the columns `species`, `peptide_id` (the label, used
to cluster the bootstrap), `target`, `prediction`, `score`, `charge`, `length`, `n_peaks`, `precursor_mz`; an empty
`prediction` means the sequencer produced nothing for the spectrum. `target` and `prediction` may be in any of three
notations: the benchmark's DeepNovo form (`C+57.021`, N-terminal `+42.011`, stacked `+43.006-17.027`), the fork's ProForma form
(`C[UNIMOD:4]`, `[UNIMOD:1]-`) or Casanovo v5's (`C[Carbamidomethyl]`, `[Acetyl]-`). Every peptide is tokenised into
one token per residue plus one for the N-terminal modification (stacked N-terminal deltas merge by mass), and masses come from one table (Casanovo v5's
`config.yaml`, with bare cysteine and the phosphorylations the fork's sequencer can emit added).

The match is `aa_match_prefix_suffix` of Casanovo v5.0.0 `casanovo/denovo/evaluate.py` (sha256 f2b477db1b5a of the
file fetched 22/10/2026), copied verbatim below under its Apache-2.0 licence: an amino acid matches when its mass is
within 0.1 Da of the aligned one and the cumulative mass up to it, from the N- or the C-terminus, is within 0.5 Da;
a peptide matches when every amino acid matches.

Per species (and overall) the output holds: spectra, coverage (share with a prediction), amino-acid precision
(matched / predicted amino acids) and recall (matched / target amino acids), peptide precision among predicted
spectra and peptide recall over all spectra (Casanovo's "precision at full coverage" when coverage is 1), the area
under the precision-coverage curve obtained by sorting spectra by `score`, coverage counted over all spectra and the curve
ending at the achieved coverage (Casanovo's "average precision"; a sequencer that skips spectra loses that area),
peptide-clustered bootstrap 95 % intervals (resampling peptides, `--bootstrap` draws, seed 0) for peptide recall and
amino-acid precision, the share of predictions whose mass is more than 50 ppm from the precursor (isotope errors 0 and 1
allowed; computed from the shared mass table for both sequencers) with the peptide precision among them, and the same
peptide recall by charge, length, modification presence and kind, precursor m/z and peak count. Written as `<out>.json` and `<out>.md`;
`--rows` adds `<out>.rows.parquet`, one row per spectrum (species, peptide_id, predicted, n_target, n_pred, n_match,
pep_match, score, ppm and the stratum labels) for analyses that need the spectrum level.

    python scripts/analysis/nine_species_score.py --predictions <table> --out <path-without-extension> [--bootstrap 300]
"""
# ruff: noqa: T201  (command-line script: the result is printed)

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Casanovo v5 config.yaml residue masses (monoisotopic), plus the tokens the fork's sequencer can also emit.
MASS: dict[str, float] = {
    "G": 57.021464, "A": 71.037114, "S": 87.032028, "P": 97.052764, "V": 99.068414, "T": 101.047670, "L": 113.084064,
    "I": 113.084064, "N": 114.042927, "D": 115.026943, "Q": 128.058578, "K": 128.094963, "E": 129.042593, "M": 131.040485,
    "H": 137.058912, "F": 147.068414, "R": 156.101111, "Y": 163.063329, "W": 186.079313,
    "C": 103.009185, "C[+57.021]": 160.030649, "M[+15.995]": 147.035400, "N[+0.984]": 115.026943, "Q[+0.984]": 129.042594,
    "S[+79.966]": 166.998359, "T[+79.966]": 181.014001, "Y[+79.966]": 243.029660, "R[+0.984]": 157.085127, "P[+15.995]": 113.047679,
    "[+42.011]-": 42.010565, "[+43.006]-": 43.005814, "[-17.027]-": -17.026549, "[+25.980]-": 25.980265,
}
# Every spelling of a modification seen in the three notations -> the canonical delta used in MASS keys.
DELTA: dict[str, str] = {
    "UNIMOD:4": "+57.021", "Carbamidomethyl": "+57.021", "+57.021": "+57.021", "+57.02": "+57.021",
    "UNIMOD:35": "+15.995", "Oxidation": "+15.995", "+15.995": "+15.995", "+15.99": "+15.995",
    "UNIMOD:7": "+0.984", "Deamidated": "+0.984", "+0.984": "+0.984", "+0.98": "+0.984", "+.98": "+0.984",
    "UNIMOD:21": "+79.966", "Phospho": "+79.966", "+79.966": "+79.966", "+79.97": "+79.966",
    "UNIMOD:1": "+42.011", "Acetyl": "+42.011", "+42.011": "+42.011", "+42.01": "+42.011",
    "UNIMOD:5": "+43.006", "Carbamyl": "+43.006", "+43.006": "+43.006", "+43.01": "+43.006",
    "UNIMOD:385": "-17.027", "Ammonia-loss": "-17.027", "-17.027": "-17.027", "-17.03": "-17.027",
    "+25.980265": "+25.980", "+25.980": "+25.980", "+25.98": "+25.980",
}
NTERM = re.compile(r"\[([^\]]+)\]-?|([+-][0-9.]+)")  # before the first residue only
RESIDUE = re.compile(r"([A-Z])(\[[^\]]+\]|\([^)]+\)|[+-][0-9.]+)?")
LEGACY = {"ox": "+15.995", "p": "+79.966"}  # the fork's legacy spellings M(ox), S(p)
BY_ROUNDED_DELTA: dict[float, str] = {round(float(v), 2): v for v in set(DELTA.values())}  # any numeric spelling -> canonical
PROTON, WATER, NEUTRON = 1.007276, 18.010565, 1.003355  # monoisotopic; NEUTRON is the 13C-12C difference used for isotope errors
STRATA = ("charge", "length", "modified", "mods", "precursor_mz", "n_peaks")  # per-spectrum groupings reported with peptide recall
OFF_MASS_PPM = 50.0  # Casanovo's precursor_mass_tol; the share of predictions beyond it is reported for both sequencers


def canonical_delta(spelling: str) -> str:
    """The canonical delta for a modification spelled by name, UNIMOD id or a mass with any number of decimals."""
    if spelling in DELTA:
        return DELTA[spelling]
    try:
        return BY_ROUNDED_DELTA[round(float(spelling), 2)]
    except (ValueError, KeyError):
        raise ValueError(f"unknown modification {spelling!r}") from None


def nterm_token(deltas: list[str]) -> str:
    """Stacked N-terminal modifications (the benchmark writes carbamylation with ammonia loss as `+43.006-17.027`) are one
    chemical state: merged by mass into the canonical token."""
    if len(deltas) == 1:
        return f"[{deltas[0]}]-"
    return f"[{canonical_delta(f'{sum(MASS[f"[{d}]-"] for d in deltas):+.3f}')}]-"


def tokenize(peptide: str) -> list[str]:
    """One token per residue (with its modification), preceded by one for the N-terminal modification if any."""
    s = peptide.strip()
    pos, deltas = 0, []
    while m := NTERM.match(s, pos):
        deltas.append(canonical_delta(m.group(1) or m.group(2)))
        pos = m.end()
    tokens = [nterm_token(deltas)] if deltas else []
    while pos < len(s):
        m = RESIDUE.match(s, pos)
        if not m:
            raise ValueError(f"cannot tokenise {peptide!r} at {pos}")
        pos = m.end()
        aa, mod = m.group(1), m.group(2)
        if mod is None:
            tokens.append(aa)
        else:
            key = mod.strip("[]()")
            tokens.append(f"{aa}[{canonical_delta(LEGACY.get(key, key))}]")
    return tokens


def precursor_ppm(tokens: list[str], precursor_mz: float, charge: int) -> float:
    """Smallest |ppm| between the predicted peptide's mass and the observed precursor mass over isotope errors 0 and 1
    (Casanovo's isotope_error_range), from the shared mass table; nan when the charge is unknown."""
    if charge < 1:
        return float("nan")
    observed = (precursor_mz - PROTON) * charge
    predicted = sum(MASS[t] for t in tokens) + WATER
    return min(abs(predicted + k * NEUTRON - observed) for k in (0, 1)) / observed * 1e6


def mass_diff(mz1: float, mz2: float, mode_is_da: bool) -> float:
    """spectrum_utils.utils.mass_diff in Dalton mode: the signed difference."""
    return mz1 - mz2


# ---- verbatim from Casanovo v5.0.0, casanovo/denovo/evaluate.py (Apache-2.0), tokens as strings ----
def aa_match_prefix(peptide1, peptide2, aa_dict, cum_mass_threshold: float = 0.5, ind_mass_threshold: float = 0.1):
    aa_matches = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True)) < cum_mass_threshold:
            aa_matches[max(i1, i2)] = abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            i1, i2 = i1 + 1, i2 + 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 + 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 + 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()


def aa_match_prefix_suffix(peptide1, peptide2, aa_dict, cum_mass_threshold: float = 0.5, ind_mass_threshold: float = 0.1):
    aa_matches, pep_match = aa_match_prefix(peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold)
    if pep_match:
        return aa_matches, pep_match
    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    i_stop = np.argwhere(~aa_matches)[0]
    cum_mass1, cum_mass2 = 0.0, 0.0
    while i1 >= i_stop and i2 >= i_stop:
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True)) < cum_mass_threshold:
            aa_matches[max(i1, i2)] = abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            i1, i2 = i1 - 1, i2 - 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 - 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 - 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()
# ---- end of the verbatim block ----


@dataclass
class Scored:
    """Per-spectrum outcome."""
    species: str
    peptide_id: str
    predicted: bool
    n_target: int
    n_pred: int
    n_match: int
    pep_match: bool
    score: float
    ppm: float  # predicted-to-observed precursor mass deviation, nan without a prediction
    strata: dict[str, str] = field(default_factory=dict)


def modification_kind(target: list[str]) -> str:
    """Which modifications the label carries: `none`, `C+57.021 only`, or the sorted distinct other deltas (N-terminal
    ones prefixed `nterm`), so the modification penalty can be attributed to a kind."""
    kinds = sorted({("nterm" + t[1:-2]) if t.endswith("]-") else t[t.index("[") + 1:-1] for t in target if "[" in t and t != "C[+57.021]"})
    if kinds:
        return ",".join(kinds)
    return "C+57.021 only" if any(t == "C[+57.021]" for t in target) else "none"


def score_row(r: pd.Series) -> Scored:
    target = tokenize(str(r["target"]))
    pred_s = "" if pd.isna(r["prediction"]) else str(r["prediction"]).strip()
    strata = {"charge": f"{int(r['charge'])}" if int(r["charge"]) < 4 else "4+",
              "length": ("<=9" if r["length"] <= 9 else "10-14" if r["length"] <= 14 else "15-19" if r["length"] <= 19 else "20-29" if r["length"] <= 29 else "30+"),
              "modified": "yes" if any("[" in t for t in target) else "no",
              "mods": modification_kind(target),
              "precursor_mz": ("<500" if r["precursor_mz"] < 500 else "500-700" if r["precursor_mz"] < 700 else "700-900" if r["precursor_mz"] < 900 else "900+"),
              "n_peaks": ("<50" if r["n_peaks"] < 50 else "50-99" if r["n_peaks"] < 100 else "100-199" if r["n_peaks"] < 200 else "200+")}
    if not pred_s:
        return Scored(r["species"], r["peptide_id"], False, len(target), 0, 0, False, float("-inf"), float("nan"), strata)
    pred = tokenize(pred_s)
    matches, pep = aa_match_prefix_suffix(pred, target, MASS)
    return Scored(r["species"], r["peptide_id"], True, len(target), len(pred), int(matches.sum()), bool(pep), float(r["score"]),
                  precursor_ppm(pred, float(r["precursor_mz"]), int(r["charge"])), strata)


def summarise(rows: list[Scored]) -> dict[str, Any]:
    n = len(rows)
    predicted = [r for r in rows if r.predicted]
    n_pred_aa = sum(r.n_pred for r in predicted)
    n_tgt_aa = sum(r.n_target for r in rows)
    n_match = sum(r.n_match for r in predicted)
    correct = sum(r.pep_match for r in predicted)
    off = [r for r in predicted if r.ppm > OFF_MASS_PPM]
    out: dict[str, Any] = {"spectra": n, "coverage": len(predicted) / n if n else float("nan"),
           "off_mass_share": len(off) / len(predicted) if predicted else float("nan"),  # of predictions, beyond OFF_MASS_PPM
           "peptide_precision_off_mass": sum(r.pep_match for r in off) / len(off) if off else float("nan"),
           "aa_precision": n_match / n_pred_aa if n_pred_aa else float("nan"), "aa_recall": n_match / n_tgt_aa if n_tgt_aa else float("nan"),
           "peptide_precision_predicted": correct / len(predicted) if predicted else float("nan"), "peptide_recall": correct / n if n else float("nan")}
    if predicted:  # precision-coverage curve: spectra sorted by score, coverage = share of all spectra, area = Casanovo's "average precision"
        order = sorted(predicted, key=lambda r: -r.score)
        hits = np.cumsum([r.pep_match for r in order])
        k = np.arange(1, len(order) + 1)
        precision = hits / k
        coverage = k / n
        out["average_precision"] = float(np.trapezoid(precision, coverage))  # ends at the achieved coverage, as Casanovo's curves do
        out["peptide_precision_at_coverage"] = {f"{c:.0%}": float(precision[int(np.ceil(c * n)) - 1]) if np.ceil(c * n) <= len(order) else float("nan")
                                                for c in (0.25, 0.5, 0.75, 1.0)}
    return out


def bootstrap(rows: list[Scored], draws: int, rng: np.random.Generator) -> dict[str, list[float]]:
    """Peptide-clustered bootstrap: resample peptides with replacement, keep every spectrum of a drawn peptide."""
    by_pep: dict[str, list[Scored]] = defaultdict(list)
    for r in rows:
        by_pep[r.peptide_id].append(r)
    peps = list(by_pep)
    stats: dict[str, list[float]] = {"peptide_recall": [], "aa_precision": []}
    for _ in range(draws):
        sample = [r for p in rng.choice(peps, size=len(peps), replace=True) for r in by_pep[p]]
        s = summarise(sample)
        stats["peptide_recall"].append(s["peptide_recall"])
        stats["aa_precision"].append(s["aa_precision"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--rows", action="store_true", help="also write every spectrum's outcome to <out>.rows.parquet (one row per input row, same order)")
    ap.add_argument("--demote-ppm", type=float, default=None,
                    help="sensitivity check: rank predictions whose precursor deviation (computed here) exceeds this last, as Casanovo's score sign does")
    args = ap.parse_args()
    df = pd.read_parquet(args.predictions) if args.predictions.suffix == ".parquet" else pd.read_csv(args.predictions)
    rows = [score_row(r) for _, r in df.iterrows()]
    if args.rows:
        pd.DataFrame([{**{k: v for k, v in vars(r).items() if k != "strata"}, **r.strata} for r in rows]).to_parquet(f"{args.out}.rows.parquet")
    if args.demote_ppm is not None:
        for r in rows:
            if r.ppm > args.demote_ppm:
                r.score -= 1e6  # below every in-tolerance prediction, order among themselves kept
    rng = np.random.default_rng(0)
    result: dict = {"predictions": str(args.predictions), "spectra": len(rows), "rule": "Casanovo v5.0.0 aa_match_prefix_suffix, 0.5 Da cumulative, 0.1 Da per residue",
                    "demote_ppm": args.demote_ppm, "species": {}}
    for sp in sorted({r.species for r in rows}):
        sub = [r for r in rows if r.species == sp]
        entry = summarise(sub)
        entry["ci95"] = bootstrap(sub, args.bootstrap, rng) if args.bootstrap else {}
        entry["by"] = {}
        for key in STRATA:
            groups: dict[str, list[Scored]] = defaultdict(list)
            for r in sub:
                groups[r.strata[key]].append(r)
            entry["by"][key] = {g: {"spectra": len(v), "peptide_recall": summarise(v)["peptide_recall"]} for g, v in sorted(groups.items())}
        result["species"][sp] = entry
    result["all"] = summarise(rows)
    Path(f"{args.out}.json").write_text(json.dumps(result, indent=1))  # appended, not with_suffix: a dot in the name must not swallow part of it
    lines = ["| species | spectra | coverage | AA precision | AA recall | peptide precision (predicted) | peptide recall (all) | AP | off-mass share | 95 % CI peptide recall |",
             "| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---- |"]
    for sp, e in list(result["species"].items()) + [("all", result["all"])]:
        ci = e.get("ci95", {}).get("peptide_recall")
        ci_s = f"{ci[0]:.3f} to {ci[1]:.3f}" if ci else "-"
        lines.append(f"| {sp} | {e['spectra']:,} | {e['coverage']:.3f} | {e['aa_precision']:.3f} | {e['aa_recall']:.3f} | {e['peptide_precision_predicted']:.3f} | {e['peptide_recall']:.3f} | {e.get('average_precision', float('nan')):.3f} | {e['off_mass_share']:.3f} | {ci_s} |")
    Path(f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
