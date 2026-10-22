"""Is a species' peptide recall explained by what its spectra look like? Direct standardisation of the per-spectrum outcomes.

Analysis only. Input: one or more `<run>.rows.parquet` files written by `nine_species_score.py --rows` (one row per
spectrum: `species`, `pep_match`, and the stratum labels `length`, `charge`, `modified`, `n_peaks`). For each run and
species the output gives the crude peptide recall, the recall standardised to the pooled composition over the cells
`length x charge x modified` (each species' cell recalls weighted by the pooled cell shares; cells the species lacks are
dropped and the weights renormalised, the dropped share is reported), and the same over `length x charge x modified x
n_peaks`. If standardisation pulls the species together, their differences are composition (what peptides and spectra
the benchmark holds for that species); the spread that remains is what the spectra of that project do to the model
beyond those labels. Written as `<out>.json` and `<out>.md`.

    python scripts/analysis/nine_species_standardise.py --rows <run.rows.parquet> [...] --out <path-without-extension>
"""
# ruff: noqa: T201  (command-line script: the result is printed)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CELLS = {"peptide": ["length", "charge", "modified"], "peptide+spectrum": ["length", "charge", "modified", "n_peaks"]}


def standardised(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per species: crude recall, recall at the pooled cell composition, and the pooled share of cells the species lacks."""
    pooled = rows.groupby(keys, observed=True).size() / len(rows)
    out = []
    for sp, g in rows.groupby("species", sort=True):
        cell = g.groupby(keys, observed=True)["pep_match"].mean()
        w = pooled.reindex(cell.index)
        out.append({"species": sp, "spectra": len(g), "crude": float(g["pep_match"].mean()),
                    "standardised": float((cell * w).sum() / w.sum()), "missing_share": float(1 - w.sum())})
    return pd.DataFrame(out).set_index("species")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result: dict[str, dict] = {}
    lines: list[str] = []
    for path in args.rows:
        run = path.name.removesuffix(".rows.parquet")
        rows = pd.read_parquet(path)
        tables = {name: standardised(rows, keys) for name, keys in CELLS.items()}
        result[run] = {name: t.to_dict(orient="index") for name, t in tables.items()}
        lines += [f"## {run}\n", "| species | spectra | crude recall | standardised (length x charge x modified) | missing share | standardised (+ peak count) | missing share |",
                  "| :---- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        a, b = tables["peptide"], tables["peptide+spectrum"]
        for sp in a.index:
            lines.append(f"| {sp} | {a.loc[sp, 'spectra']:,} | {a.loc[sp, 'crude']:.3f} | {a.loc[sp, 'standardised']:.3f} | {a.loc[sp, 'missing_share']:.3f} | {b.loc[sp, 'standardised']:.3f} | {b.loc[sp, 'missing_share']:.3f} |")
        spread = lambda t: t["standardised"].max() - t["standardised"].min()  # noqa: E731
        lines.append(f"\nSpread across species (max - min): crude {a['crude'].max() - a['crude'].min():.3f}, standardised {spread(a):.3f}, with peak count {spread(b):.3f}\n")
        result[run]["spread"] = {"crude": float(a["crude"].max() - a["crude"].min()), "peptide": float(spread(a)), "peptide+spectrum": float(spread(b))}
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
