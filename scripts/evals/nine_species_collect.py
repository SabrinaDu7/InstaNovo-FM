"""Join a sequencer's output on a nine-species sample to the sample manifest, in the scorer's input schema.

Inputs: the sample `manifest.parquet` of `nine_species_sample.py` (one row per sampled spectrum, in the order the
spectra were written to `<species>.mgf`), and one output file per species from one sequencer:
  --fork      the fork's `instanovo-fm denovo predict --evaluation` CSV: `scan_number` is the 0-based position of the
              spectrum in the MGF it read (the fork's reader numbers spectra from 0, it does not read `SCANS=`), `predictions` ProForma,
              `log_probs` the sequence log-probability, `targets` the label as the fork parsed it.
  --casanovo  Casanovo's mzTab: PSM rows keyed by `spectra_ref` (`ms_run[1]:index=<0-based position>`), the peptide
              from `opt_ms_run[1]_proforma` (Casanovo 5's full ProForma string; its `sequence` column is the bare
              peptide with the modifications moved to `modifications`, mzTab style), `search_engine_score[1]` the score.
Output: `<out>.parquet` with `species`, `peptide_id` (label without modifications, I as L), `target` (the manifest's
label), `prediction`, `score`, `charge`, `length`, `n_peaks`, `precursor_mz`, `source_file`, `scan`; every sampled
spectrum appears once, with an empty `prediction` when the sequencer gave none. The join is positional (position in
the sample MGF), and the target the sequencer parsed is checked against the manifest's label on the fork side so a
misaligned row cannot pass silently.

    python scripts/evals/nine_species_collect.py --manifest <dir>/manifest.parquet --fork '<dir>/fork-greedy/*.csv' --out <path>
    python scripts/evals/nine_species_collect.py --manifest <dir>/manifest.parquet --casanovo '<dir>/casanovo/*.mztab' --out <path>
"""
# ruff: noqa: T201  (command-line script: the result is printed)

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

MOD = re.compile(r"\[[^\]]*\]-?|\([^)]*\)|[+-]\d+(\.\d+)?")
PROFORMA = "opt_ms_run[1]_proforma"  # Casanovo 5's modified-sequence column


def peptide_id(label: str) -> str:
    return MOD.sub("", label).replace("I", "L")


def bare_equal(a: str, b: str) -> bool:
    """The two labels name the same residues once modifications and I/L are dropped (notation may differ)."""
    return peptide_id(a) == peptide_id(b)


def species_rows(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """The manifest rows of the species a sequencer output file is named after (`<species>.csv` or `.mztab`; species
    names contain dots, as in `H.-sapiens`), in sample order."""
    sub = manifest[manifest["species"] == path.stem].reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"{path}: no manifest rows for a species named {path.stem!r}")
    return sub


def fork_table(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    sub = species_rows(path, manifest)
    d = pd.read_csv(path)
    d = d.set_index(d["scan_number"].astype(int)).sort_index()
    if not d.index.is_unique:
        raise SystemExit(f"{path}: repeated scan_number values")
    if not d.index.isin(range(len(sub))).all():
        raise SystemExit(f"{path}: positions outside the sample ({d.index.max()} >= {len(sub)})")
    out = sub.copy()
    out["prediction"] = ""
    out["score"] = float("-inf")
    out.loc[d.index, "prediction"] = d["predictions"].fillna("").astype(str).values
    out.loc[d.index, "score"] = d["log_probs"].astype(float).values
    out["delta_mass_ppm"] = float("nan")
    out.loc[d.index, "delta_mass_ppm"] = d["delta_mass_ppm"].astype(float).values
    mism = [(i, t, m) for i, t, m in zip(d.index, d["targets"].astype(str), sub.loc[d.index, "label"], strict=False) if not bare_equal(t, m)]
    if mism:
        raise SystemExit(f"{path}: {len(mism)} rows whose parsed target differs from the manifest label, first {mism[:3]}")
    return out


def casanovo_table(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    sub = species_rows(path, manifest)
    rows = []
    with open(path) as fh:
        header = None
        for line in fh:
            if line.startswith("PSH"):
                header = line.rstrip("\n").split("\t")
            elif line.startswith("PSM") and header:
                rows.append(dict(zip(header, line.rstrip("\n").split("\t"), strict=False)))
    d = pd.DataFrame(rows)
    if PROFORMA not in d:
        raise SystemExit(f"{path}: no {PROFORMA} column; the `sequence` column is bare and must not be scored")
    pos = d["spectra_ref"].str.extract(r"index=(\d+)")[0].astype(int)
    d = d.assign(pos=pos, **{"search_engine_score[1]": d["search_engine_score[1]"].astype(float)})
    d = d.sort_values("search_engine_score[1]", ascending=False).drop_duplicates("pos").set_index("pos").sort_index()
    out = sub.copy()
    out["prediction"] = ""
    out["score"] = float("-inf")
    out.loc[d.index, "prediction"] = d[PROFORMA].fillna("").astype(str).values
    out.loc[d.index, "score"] = d["search_engine_score[1]"].astype(float).values
    out["delta_mass_ppm"] = float("nan")  # Casanovo folds the precursor check into the score's sign
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--fork", help="glob of the fork's per-species CSVs")
    ap.add_argument("--casanovo", help="glob of Casanovo's per-species mzTab files")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if bool(args.fork) == bool(args.casanovo):
        raise SystemExit("give exactly one of --fork or --casanovo")
    manifest = pd.read_parquet(args.manifest)
    files = sorted(Path(p) for p in glob.glob(args.fork or args.casanovo))
    if not files:
        raise SystemExit("no files match")
    parts = [(fork_table if args.fork else casanovo_table)(f, manifest) for f in files]
    table = pd.concat(parts, ignore_index=True)
    table["peptide_id"] = table["label"].map(peptide_id)
    table = table.rename(columns={"label": "target"})[["species", "peptide_id", "target", "prediction", "score", "charge", "length", "n_peaks", "precursor_mz", "source_file", "scan", "unsupported_label", "delta_mass_ppm"]]
    out = Path(f"{args.out}.parquet")  # appended, not with_suffix: a dot in the name must not swallow part of it
    table.to_parquet(out)
    covered = (table["prediction"] != "").mean()
    print(f"wrote {out}: {len(table):,} spectra from {len(files)} files; {covered:.1%} with a prediction")


if __name__ == "__main__":
    main()
