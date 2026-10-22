"""Draw the evaluation subsets of the nine-species benchmark: a seeded uniform sample of spectra per species.

Input: the benchmark root (one folder per species holding the revised benchmark's MGFs, Yilmaz et al. 2024,
MassIVE MSV000090982, with `SEQ=` on every spectrum). Output, under `--out`: `<species>.mgf` with the sampled
spectra copied verbatim (so both sequencers read exactly the same peaks and the same labels), `manifest.parquet`
with one row per sampled spectrum (species, source file, 0-based index in that file, scan, charge, precursor m/z,
label, peak count, m/z range, whether the label carries a modification the fork's sequencer cannot emit) and
`summary.json` (totals, the draw, the seed). The draw is `numpy.random.default_rng(seed + species rank)` over the
species' spectra in file order, `n` without replacement, or every spectrum when the species has fewer than `n`.

    python scripts/evals/nine_species_sample.py --root <benchmark> --out <dir> --n 20000 --seed 20261022
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SKIP = {"ccms_parameters", "sequence"}
UNSUPPORTED_MODS = ("+25.980",)  # carbamylation plus ammonia loss: in the benchmark's labels, suppressed by the fork's sequencer
MOD = re.compile(r"[+-]\d+(\.\d+)?")


@dataclass(frozen=True)
class Spectrum:
    species: str
    source_file: str
    index_in_file: int
    scan: int
    charge: int
    precursor_mz: float
    label: str
    n_peaks: int
    mz_min: float
    mz_max: float
    block: str

    @property
    def unsupported_label(self) -> bool:
        return any(m in self.label for m in UNSUPPORTED_MODS)

    @property
    def length(self) -> int:
        return len(MOD.sub("", self.label))


def read_blocks(mgf: Path, species: str):
    """Every spectrum of `mgf` as a Spectrum, in file order."""
    lines: list[str] = []
    k = 0
    with open(mgf, errors="ignore") as fh:
        for line in fh:
            if line.startswith("BEGIN IONS"):
                lines = [line]
            elif line.startswith("END IONS"):
                lines.append(line)
                yield parse(lines, mgf, species, k)
                k += 1
            elif lines:
                lines.append(line)


def parse(lines: list[str], mgf: Path, species: str, k: int) -> Spectrum:
    head = {}
    peaks = []
    for line in lines[1:-1]:
        if "=" in line and not line[0].isdigit():
            key, _, val = line.partition("=")
            head[key.strip()] = val.strip()
        elif line.strip():
            peaks.append(float(line.split()[0]))
    charge = int(head.get("CHARGE", "0").rstrip("+-") or 0)
    scan = int(head.get("SCANS", re.search(r"scan=(\d+)", head.get("TITLE", "")).group(1) if re.search(r"scan=(\d+)", head.get("TITLE", "")) else -1))
    return Spectrum(species, mgf.name, k, scan, charge, float(head.get("PEPMASS", "nan").split()[0]), head.get("SEQ", ""),
                    len(peaks), min(peaks) if peaks else float("nan"), max(peaks) if peaks else float("nan"), "".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    species_dirs = sorted(p for p in args.root.iterdir() if p.is_dir() and p.name not in SKIP)
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for rank, sdir in enumerate(species_dirs):
        files = sorted(sdir.glob("*.mgf"))
        counts = [sum(1 for line in open(f, errors="ignore") if line.startswith("BEGIN IONS")) for f in files]
        total = sum(counts)
        rng = np.random.default_rng(args.seed + rank)
        chosen = np.sort(rng.choice(total, size=min(args.n, total), replace=False))
        chosen_set = set(chosen.tolist())
        offset = 0
        kept = 0
        with open(args.out / f"{sdir.name}.mgf", "w") as out:
            for f, c in zip(files, counts, strict=True):
                for sp in read_blocks(f, sdir.name):
                    if offset + sp.index_in_file in chosen_set:
                        out.write(sp.block)
                        rows.append(dict(species=sp.species, source_file=sp.source_file, index_in_file=sp.index_in_file, scan=sp.scan,
                                         charge=sp.charge, precursor_mz=sp.precursor_mz, label=sp.label, length=sp.length, n_peaks=sp.n_peaks,
                                         mz_min=sp.mz_min, mz_max=sp.mz_max, unsupported_label=sp.unsupported_label))
                        kept += 1
                offset += c
        summary[sdir.name] = dict(files=len(files), spectra_total=total, sampled=kept)
        print(f"{sdir.name:26s} {total:9,d} spectra in {len(files):3d} files -> {kept:6,d} sampled", flush=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out / "manifest.parquet")
    (args.out / "summary.json").write_text(json.dumps(dict(seed=args.seed, n=args.n, root=str(args.root), species=summary), indent=1))
    print(f"wrote {args.out / 'manifest.parquet'} ({len(rows):,} rows) and summary.json")


if __name__ == "__main__":
    main()
