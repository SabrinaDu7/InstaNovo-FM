"""Peptide-level overlap between the nine-species benchmark's labels and InstaNovo-FM's corpus splits.

Inputs: the benchmark's MGFs (one folder per species, `SEQ=` on every spectrum, DeepNovo notation such as
`C+57.021`, `M+15.995`, `N+0.984`, N-terminal `+42.011`) and the corpus `peptide_registry.parquet` (columns
`peptide`, `split`). Peptides are compared after stripping modifications and collapsing I to L, which is how the
corpus keys its peptide-disjoint splits (checked below: the registry holds no `I`). Output: per species, the number
of spectra and distinct peptides, and the share of spectra and of peptides whose peptide is in the corpus's train,
validation or test split, or absent.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(sys.argv[1])
REGISTRY = Path(sys.argv[2])
MOD = re.compile(r"[+-]\d+(\.\d+)?")


def bare(seq: str) -> str:
    """The peptide without modification masses, I collapsed to L."""
    return MOD.sub("", seq).replace("I", "L")


registry = pq.read_table(REGISTRY).to_pydict()
split_of = dict(zip(registry["peptide"], registry["split"], strict=True))
assert not any("I" in p for p in list(split_of)[:200000]), "registry keeps I: collapse rule wrong"
print(f"registry: {len(split_of):,} peptides")
print(f"{'species':26s} {'spectra':>9s} {'peptides':>9s} | spectra in train/valid/test/absent (%) | peptides in train/valid/test/absent (%)")
for species in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name not in ("ccms_parameters", "sequence")):
    spectra = Counter()
    for mgf in species.glob("*.mgf"):
        with open(mgf, errors="ignore") as fh:
            for line in fh:
                if line.startswith("SEQ="):
                    spectra[bare(line[4:].strip())] += 1
    n_spec = sum(spectra.values())
    by_spec = Counter(); by_pep = Counter()
    for pep, n in spectra.items():
        s = split_of.get(pep, "absent"); by_spec[s] += n; by_pep[s] += 1
    f = lambda c, tot: "/".join(f"{100*c[k]/tot:4.1f}" for k in ("train", "validation", "test", "absent"))
    print(f"{species.name:26s} {n_spec:9,d} {len(spectra):9,d} | {f(by_spec, n_spec):>36s} | {f(by_pep, len(spectra)):>36s}", flush=True)
