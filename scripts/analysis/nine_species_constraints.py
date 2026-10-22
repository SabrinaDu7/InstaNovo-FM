"""Which sampled spectra fall outside each sequencer's input assumptions, per species, from the sample manifest alone.

Input: `manifest.parquet` of `nine_species_sample.py`. The limits are read from the two sequencers' configurations:
InstaNovo-FM's `configs/inference/denovo.yaml` and model config (200 peaks kept, m/z 50 to 2,500 Da bins, `max_length` 40
residues, `max_charge` 10, the residue table without `+25.980`), Casanovo v5's `config.yaml` (`min_peaks` 20, `max_peaks`
150, `min_mz` 50, `max_mz` 2500, `max_charge` 4, `min_peptide_len` 6, every benchmark modification in the alphabet).
Output: one table (markdown to stdout, JSON to `--out`) with, per species, the share of spectra that each limit touches:
`peaks_over_200` (the fork keeps the 200 most intense: information lost, not a failure), `peaks_over_150` (Casanovo
likewise), `peaks_under_20` (Casanovo skips the spectrum), `mz_over_2500` (peaks above the binning range), `charge_over_4`
(Casanovo clamps or skips), `length_over_40` (the fork cannot emit the label), `length_under_6` (Casanovo's minimum),
`label_unsupported_fork` (`+25.980`), `label_modified` (any modification), and the charge and length distributions.

    python scripts/analysis/nine_species_constraints.py --manifest <dir>/manifest.parquet --out <path>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

LIMITS = {
    "peaks_over_200": lambda d: d["n_peaks"] > 200,
    "peaks_over_150": lambda d: d["n_peaks"] > 150,
    "peaks_under_20": lambda d: d["n_peaks"] < 20,
    "mz_over_2500": lambda d: d["mz_max"] > 2500,
    "charge_over_4": lambda d: d["charge"] > 4,
    "charge_over_10": lambda d: d["charge"] > 10,
    "length_over_40": lambda d: d["length"] > 40,
    "length_under_6": lambda d: d["length"] < 6,
    "label_unsupported_fork": lambda d: d["unsupported_label"],
    "label_modified": lambda d: d["label"].str.contains(r"[+-]\d", regex=True),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = pd.read_parquet(args.manifest)
    out: dict[str, dict] = {}
    for sp, g in list(d.groupby("species")) + [("all", d)]:
        out[sp] = {k: float(f(g).mean()) for k, f in LIMITS.items()}
        out[sp]["spectra"] = int(len(g))
        out[sp]["charge"] = {str(k): int(v) for k, v in g["charge"].value_counts().sort_index().items()}
        out[sp]["length_median"] = float(g["length"].median())
        out[sp]["n_peaks_median"] = float(g["n_peaks"].median())
    args.out.write_text(json.dumps(out, indent=1))
    cols = list(LIMITS)
    print("| species | spectra | " + " | ".join(c.replace("_", " ") for c in cols) + " | median length | median peaks |")
    print("| :---- | ---: |" + " ---: |" * (len(cols) + 2))
    for sp, e in out.items():
        print(f"| {sp} | {e['spectra']:,} | " + " | ".join(f"{100*e[c]:.1f} %" for c in cols) + f" | {e['length_median']:.0f} | {e['n_peaks_median']:.0f} |")
    print("\ncharge distribution (all):", out["all"]["charge"])


if __name__ == "__main__":
    main()
