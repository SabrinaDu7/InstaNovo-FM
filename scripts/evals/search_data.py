"""Merge the package's per-file metadata table with the rows of our exported datasets.

The evaluator looks every spectrum up in `data/search_data.xlsx` by the project and file name of its `usi`
(`instanovo_fm.data.search_data_manager`) to get `search_instrument`, `search_detector`, `search_fragmentation`,
`search_organism`, ...; the instrument probe and the "HCD Orbitrap" retrieval subset read those keys. Our runs are not
in that table (ProteomeTools's are), so `prepare_dataset.py` writes one row per run in the table's columns and this
script appends them to a copy the jobs point at with `dataset.search_data_path`:

    python scripts/evals/search_data.py --external $EXTERNAL [--package-table data/search_data.xlsx]

writes `$EXTERNAL/search_data.xlsx`. A row whose file key (project/filename without extension) is already in the
package table is not added, so the package's own description of a shared file wins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MS_EXTENSIONS = (".mzML.gz", ".mzML", ".mzXML", ".raw", ".wiff", ".d", ".mgf", ".gz")  # search_data_manager._MS_EXTENSIONS


def file_key(path: str) -> str:
    """`project/filename` as the manager keys it: parent folder plus the file name stripped of MS extensions."""
    parent, name = path.split("/")[0], path.split("/")[-1]
    for ext in MS_EXTENSIONS:
        if name.lower().endswith(ext.lower()):
            name = name[: -len(ext)]
    return f"{parent}/{name.split('.')[0]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external", type=Path, required=True, help="root holding <dataset>/<dataset>-search_data.csv")
    ap.add_argument("--package-table", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "search_data.xlsx")
    args = ap.parse_args()
    package = pd.read_excel(args.package_table)
    known = set(package["file path"].astype(str).map(file_key))
    ours = pd.concat([pd.read_csv(p) for p in sorted(args.external.glob("*/*-search_data.csv"))], ignore_index=True)
    new = ours[~ours["file path"].map(file_key).isin(known)]
    merged = pd.concat([package, new[package.columns]], ignore_index=True)
    out = args.external / "search_data.xlsx"
    merged.to_excel(out, index=False)
    print(f"{len(package)} package rows + {len(new)} of our {len(ours)} run rows ({len(ours) - len(new)} already described) -> {out}")


if __name__ == "__main__":
    main()
