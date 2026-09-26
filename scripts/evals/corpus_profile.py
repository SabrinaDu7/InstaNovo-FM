"""Profile a set of corpus-schema parquet files (a corpus split or one of our exports): what the spectra are.

    python scripts/evals/corpus_profile.py --name mcfm-test --files "$DATA/splits/mcfm/mcfm-*test*.parquet" \
        --search-data data/search_data.xlsx --out docs/results/profiles
    python scripts/evals/corpus_profile.py --name ms2bac --files "$EXTERNAL/ms2bac/ms2bac-all.parquet" \
        --search-data "$EXTERNAL/search_data.xlsx" --out docs/results/profiles

Reads only the metadata columns (never the peak arrays, except their lengths), joins each spectrum to the
package's per-file metadata table the way the evaluator does (project and file name of the `usi`,
`instanovo_fm.data.search_data_manager`), and writes `<out>/<name>.json` with: spectra, identified spectra
(non-empty `sequence`), files, projects, distinct peptides; and the distribution of `frag_type`, `acquisition`,
`collision_energy`, `precursor_charge`, `search_instrument`, `search_detector`, `search_organism`, `search_enzyme`,
modification tokens in `sequence`, peaks per spectrum, precursor m/z and peptide length. `render_results.py` puts
these side by side so a dataset can be read against the split the model is evaluated on.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

META = ["usi", "frag_type", "acquisition", "collision_energy", "precursor_charge", "precursor_mz", "experiment_name", "sequence"]
MS_EXTENSIONS = (".mzML.gz", ".mzML", ".mzXML", ".raw", ".wiff", ".d", ".mgf", ".gz")
SEARCH_COLUMNS = ["instrument", "detector", "organism", "enzyme", "fragmentation", "quant", "modifications"]


def usi_key(usi: str) -> str:
    """`project/filename` as `search_data_manager._extract_lookup_key_from_usi` forms it."""
    parts = usi.split(":")
    if len(parts) < 3:
        return ""
    name = parts[2]
    for ext in MS_EXTENSIONS:
        if name.lower().endswith(ext.lower()):
            name = name[: -len(ext)]
    return f"{parts[1]}/{name.split('.')[0]}"


def file_key(path: str) -> str:
    parent, name = path.split("/")[0], path.split("/")[-1]
    for ext in MS_EXTENSIONS:
        if name.lower().endswith(ext.lower()):
            name = name[: -len(ext)]
    return f"{parent}/{name.split('.')[0]}"


def distribution(values: pd.Series, *, top: int = 30) -> dict[str, int]:
    vc = values.fillna("(missing)").astype(str).replace("", "(empty)").value_counts()
    return {str(k): int(v) for k, v in vc.head(top).items()} | ({"(other)": int(vc.iloc[top:].sum())} if len(vc) > top else {})


def quantiles(values: pd.Series) -> dict[str, float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return {} if v.empty else {q: round(float(v.quantile(p)), 3) for q, p in (("p05", 0.05), ("median", 0.5), ("p95", 0.95))} | {"mean": round(float(v.mean()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--files", required=True, help="glob of corpus-schema parquet files")
    ap.add_argument("--search-data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.files).expanduser())))
    if not files:
        raise SystemExit(f"no files match {args.files}")
    parts, n_peaks = [], []
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in META if c in pf.schema_arrow.names]
        t = pf.read(columns=cols + ["mz_array"])
        n_peaks.append(pc.list_value_length(t["mz_array"]).to_numpy(zero_copy_only=False))
        parts.append(t.select(cols).to_pandas())
    df = pd.concat(parts, ignore_index=True)
    peaks = np.concatenate(n_peaks)
    search = pd.read_excel(args.search_data)
    search["key"] = search["file path"].astype(str).map(file_key)
    lookup = search.drop_duplicates("key").set_index("key")[SEARCH_COLUMNS]
    keys = df["usi"].fillna("").map(usi_key) if "usi" in df else pd.Series([""] * len(df))
    joined = lookup.reindex(keys.to_numpy())
    seq = df["sequence"].fillna("") if "sequence" in df else pd.Series([""] * len(df))
    identified = seq != ""
    tokens = Counter(m for s in seq[identified] for m in re.findall(r"(?:^\[UNIMOD:\d+\]-|[A-Z]\[UNIMOD:\d+\])", s))
    profile = {
        "name": args.name,
        "files": len(files),
        "spectra": int(len(df)),
        "identified": int(identified.sum()),
        "runs": int(df["experiment_name"].nunique()) if "experiment_name" in df else None,
        "projects": distribution(keys.str.split("/").str[0]),
        "matched_in_search_data": int(joined["instrument"].notna().sum()),
        "peptides": int(seq[identified].str.replace(r"\[UNIMOD:\d+\]-?", "", regex=True).str.replace("I", "L").nunique()),
        "frag_type": distribution(df.get("frag_type", pd.Series(dtype=str))),
        "acquisition": distribution(df.get("acquisition", pd.Series(dtype=str))),
        "collision_energy": distribution(df.get("collision_energy", pd.Series(dtype=str)), top=15),
        "precursor_charge": distribution(df.get("precursor_charge", pd.Series(dtype=int))),
        "search_instrument": distribution(joined["instrument"]),
        "search_detector": distribution(joined["detector"]),
        "search_fragmentation": distribution(joined["fragmentation"]),
        "search_organism": distribution(joined["organism"], top=15),
        "search_enzyme": distribution(joined["enzyme"]),
        "search_quant": distribution(joined["quant"]),
        "modification_tokens": {k: int(v) for k, v in tokens.most_common(15)},
        "modified_fraction": round(float(seq[identified].str.contains(r"\[UNIMOD", regex=True).mean()), 4) if identified.any() else None,
        "peaks_per_spectrum": {"p05": float(np.percentile(peaks, 5)), "median": float(np.median(peaks)), "p95": float(np.percentile(peaks, 95)), "mean": round(float(peaks.mean()), 2)},
        "precursor_mz": quantiles(df.get("precursor_mz", pd.Series(dtype=float))),
        "peptide_length": quantiles(seq[identified].str.replace(r"\[UNIMOD:\d+\]-?", "", regex=True).str.len()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}.json").write_text(json.dumps(profile, indent=1))
    print(f"{args.name}: {profile['spectra']} spectra, {profile['identified']} identified, {profile['peptides']} peptides, {profile['matched_in_search_data']} matched in search data -> {args.out / (args.name + '.json')}")


if __name__ == "__main__":
    main()
