"""Do two sequencers fail on the same spectra? Paired outcomes, spectrum by spectrum.

Analysis only. Input: two `<run>.rows.parquet` files written by `nine_species_score.py --rows` on the same predictions
order (same manifest, same spectra), named A and B. Output, per species and pooled: the counts of spectra both get
right, only A, only B, neither; the share of B's correct spectra that A also gets right (and the converse); and the
same four counts within each stratum of `mods` and `n_peaks`. If A's failures are mostly B's failures too, the
difficulty is in the spectra; spectra one gets and the other misses are where the models differ. Written as `<out>.json`
and `<out>.md`.

    python scripts/analysis/nine_species_pairwise.py --a <A.rows.parquet> --b <B.rows.parquet> --out <path-without-extension>
"""
# ruff: noqa: T201  (command-line script: the result is printed)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def table(a: pd.Series, b: pd.Series) -> dict[str, float | int]:
    both, only_a, only_b, neither = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
    return {"spectra": len(a), "both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
            "share_of_b_correct_also_a": both / (both + only_b) if both + only_b else float("nan"),
            "share_of_a_correct_also_b": both / (both + only_a) if both + only_a else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a, b = pd.read_parquet(args.a), pd.read_parquet(args.b)
    if len(a) != len(b) or (a["peptide_id"].values != b["peptide_id"].values).any():
        raise SystemExit("the two rows files do not describe the same spectra in the same order")
    name_a, name_b = args.a.name.removesuffix(".rows.parquet"), args.b.name.removesuffix(".rows.parquet")
    ok_a, ok_b = a["pep_match"].astype(bool), b["pep_match"].astype(bool)
    result = {"a": name_a, "b": name_b, "species": {}, "all": table(ok_a, ok_b), "by": {}}
    for sp in sorted(a["species"].unique()):
        m = a["species"] == sp
        result["species"][sp] = table(ok_a[m], ok_b[m])
    for key in ("mods", "n_peaks", "length"):
        result["by"][key] = {g: table(ok_a[a[key] == g], ok_b[a[key] == g]) for g in sorted(a[key].unique(), key=lambda s: (len(s), s))}
    head = f"| group | spectra | both right | only {name_a} | only {name_b} | neither | of {name_b}'s correct, {name_a} also right | of {name_a}'s correct, {name_b} also right |"
    sep = "| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    row = lambda g, t: f"| {g} | {t['spectra']:,} | {t['both']:,} | {t['only_a']:,} | {t['only_b']:,} | {t['neither']:,} | {t['share_of_b_correct_also_a']:.3f} | {t['share_of_a_correct_also_b']:.3f} |"  # noqa: E731
    lines = [f"# Paired outcomes: {name_a} against {name_b}\n", "## By species\n", head, sep]
    lines += [row(sp, t) for sp, t in result["species"].items()] + [row("all", result["all"]), ""]
    for key, groups in result["by"].items():
        lines += [f"## By {key}\n", head, sep] + [row(g, t) for g, t in groups.items() if t["spectra"] >= 100] + [""]
    Path(f"{args.out}.json").write_text(json.dumps(result, indent=1))
    Path(f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
