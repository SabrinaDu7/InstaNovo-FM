"""Export one of our labelled datasets as parquet in InstaNovo-FM's corpus schema, so the package's evaluation
suite runs on it unchanged.

Inputs (all produced by `proteomies-eval-data`): its run manifest `outputs/fm_manifest.csv` (`src.fm_manifest`
there: accession, instrument, lab, peak file, organism and label table per run) and the label table it names
(`src.eval_labels` there: one row per MS2 spectrum that the deposited search identified, or for MS2Bac every MS2
spectrum, keyed by 0-based spectrum `index` and native `scan`). The peaks are read again from the deposited mzML
or MGF with the same index and scan conventions as `proteomies-eval-data/src/spectra.py`, and every joined row is
gated on the scan number agreeing, so a misaligned label cannot pass silently.

    python scripts/evals/prepare_dataset.py --dataset ms2bac --manifest <fm_manifest.csv> --out $EXTERNAL

writes under `<out>/<dataset>/`:

    <dataset>-all.parquet              every MS2 spectrum of every run (identified or not; `sequence` is "" when not)
    <dataset>-identified.parquet       the identified spectra whose modifications the corpus notation can express
    <dataset>-probe-{train,valid,test}.parquet   the identified spectra split 80/10/10 by peptide (I/L collapsed,
                                       modifications dropped, seed 42): the corpus splits are peptide-disjoint, and
                                       the linear probe trains on the train file and tests on the test file
    <dataset>-search_data.csv          one row per run in the columns of `data/search_data.xlsx` (instrument,
                                       fragmentation, detector, organism...), for the package's metadata lookup
    <dataset>-summary.json             counts per run: spectra, identified, unsupported modifications, peptides, charges

Schema: the 30 columns of the corpus shards (read from `mcfm-test-00000-of-00014.parquet`, 2026-09-26), filled from
the deposit where the value exists and null otherwise, plus our own columns (`run`, `species`, `category`,
`passes_floor`, `explained`, `engine`, `engine_score`, `analyzer`, `identified`), which the loader ignores.
Sequences use the corpus's ProForma notation (`C[UNIMOD:4]`, `M[UNIMOD:35]`, `[UNIMOD:1]-` N-terminal acetyl,
`Q[UNIMOD:28]`, `E[UNIMOD:27]`); a modification outside that table blanks the sequence and is counted.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROTON = 1.007276466812
WATER = 18.010564684
# Residue masses of the package (`src/instanovo_fm/configs/residues/default.yaml`) plus the two pyro-glutamate
# tokens the corpus writes (`assets/mod_dicts/residue_masses.yaml`), used only for `peptide_calc_mz`.
RESIDUE_MASS = {
    "G": 57.021464, "A": 71.037114, "S": 87.032028, "P": 97.052764, "V": 99.068414, "T": 101.047670,
    "C": 103.009185, "L": 113.084064, "I": 113.084064, "N": 114.042927, "D": 115.026943, "Q": 128.058578,
    "K": 128.094963, "E": 129.042593, "M": 131.040485, "H": 137.058912, "F": 147.068414, "R": 156.101111,
    "Y": 163.063329, "W": 186.079313,
    "M[UNIMOD:35]": 147.035400, "C[UNIMOD:4]": 160.030649, "N[UNIMOD:7]": 115.026943, "Q[UNIMOD:7]": 129.042594,
    "R[UNIMOD:7]": 157.085127, "P[UNIMOD:35]": 113.047679, "S[UNIMOD:21]": 166.998028, "T[UNIMOD:21]": 181.01367,
    "Y[UNIMOD:21]": 243.029329, "Q[UNIMOD:28]": 111.032029, "E[UNIMOD:27]": 111.032028,
}
N_TERMINAL_MASS = {"[UNIMOD:1]": 42.010565}
# Our label tables write modifications as `Name@pos` (1-based; 0 = N-terminus); the corpus attaches UNIMOD ids.
RESIDUE_MODS = {  # (name, residue) -> corpus token
    ("Oxidation", "M"): "M[UNIMOD:35]",
    ("Oxidation", "P"): "P[UNIMOD:35]",
    ("Carbamidomethyl", "C"): "C[UNIMOD:4]",
    ("Deamidated", "N"): "N[UNIMOD:7]",
    ("Deamidated", "Q"): "Q[UNIMOD:7]",
    ("Gln->pyro-Glu", "Q"): "Q[UNIMOD:28]",
    ("Glu->pyro-Glu", "E"): "E[UNIMOD:27]",
}
TERMINAL_MODS = {"Acetyl": "[UNIMOD:1]"}
TOKEN = re.compile(r"(\[UNIMOD:\d+\]-)?([A-Z](?:\[UNIMOD:\d+\])?)")

MS_EXTENSIONS = (".mzML.gz", ".mzML", ".mgf")
CORPUS_COLUMNS = [  # order of the corpus shards
    "usi", "index", "scan", "header", "retention_time", "frag_type", "acquisition", "collision_energy",
    "isolation_target", "precursor_mz", "precursor_charge", "precursor_intensity", "lower_offset", "upper_offset",
    "mz_array", "intensity_array", "scale_factor", "peptide_observed_mz", "peptide_calc_mz", "delta_mass",
    "retention", "expectation", "hyperscore", "nextscore", "probability", "auc_intensity", "protein",
    "experiment_name", "unmodified_peptide", "sequence",
]
OURS = ["run", "species", "category", "passes_floor", "explained", "engine", "engine_score", "analyzer", "identified"]
SCHEMA = pa.schema(
    [
        ("usi", pa.string()), ("index", pa.int64()), ("scan", pa.string()), ("header", pa.string()),
        ("retention_time", pa.float64()), ("frag_type", pa.string()), ("acquisition", pa.string()),
        ("collision_energy", pa.string()), ("isolation_target", pa.float64()), ("precursor_mz", pa.float64()),
        ("precursor_charge", pa.int64()), ("precursor_intensity", pa.float64()), ("lower_offset", pa.float64()),
        ("upper_offset", pa.float64()), ("mz_array", pa.large_list(pa.float64())),
        ("intensity_array", pa.large_list(pa.float64())), ("scale_factor", pa.float32()),
        ("peptide_observed_mz", pa.float64()), ("peptide_calc_mz", pa.float64()), ("delta_mass", pa.float64()),
        ("retention", pa.float64()), ("expectation", pa.float64()), ("hyperscore", pa.float64()),
        ("nextscore", pa.float64()), ("probability", pa.float64()), ("auc_intensity", pa.float64()),
        ("protein", pa.string()), ("experiment_name", pa.string()), ("unmodified_peptide", pa.string()),
        ("sequence", pa.string()),
        ("run", pa.string()), ("species", pa.string()), ("category", pa.string()), ("passes_floor", pa.bool_()),
        ("explained", pa.float64()), ("engine", pa.string()), ("engine_score", pa.float64()),
        ("analyzer", pa.string()), ("identified", pa.bool_()),
    ]
)
SEARCH_DATA_COLUMNS = ["project", "file path", "workflow", "acquisition", "detector", "fragmentation", "instrument",
                       "fasta", "enzyme", "quant", "modifications", "organism"]


@dataclass(frozen=True)
class Peaks:
    """One MS2 spectrum as read from the deposit, in the units the corpus uses (seconds, Da, strings)."""

    index: int
    scan: int
    header: str | None
    retention_time: float
    analyzer: str  # FT | IT | TOF | ""
    activation: str  # HCD | CID | ETD | ""
    collision_energy: float
    precursor_mz: float
    charge: int
    precursor_intensity: float
    isolation_target: float
    lower_offset: float
    upper_offset: float
    mz: np.ndarray
    intensity: np.ndarray


def _activation(act: dict) -> str:
    names = " ".join(k for k in act if k != "collision energy").lower()
    if "beam-type" in names or "higher-energy" in names or "supplemental" in names:
        return "HCD"
    if "electron transfer" in names:
        return "ETD"
    if "collision-induced" in names:
        return "CID"
    return ""


def _analyzers(path: Path) -> dict[str, str]:
    """instrumentConfiguration id -> FT | IT | TOF, from the mzML header."""
    from pyteomics import mzml

    out: dict[str, str] = {}
    with mzml.MzML(str(path), use_index=False) as reader:
        for conf in reader.iterfind("instrumentConfiguration", recursive=True):
            names = " ".join(str(k) for c in conf.get("componentList", {}).get("analyzer", []) for k in c).lower()
            out[conf["id"]] = "FT" if "orbitrap" in names or "fourier" in names else "IT" if "ion trap" in names or "iontrap" in names else "TOF" if "time-of-flight" in names or "tof" in names else ""
    return out


def iter_mzml(path: Path) -> Iterator[Peaks]:
    """Every MS2 spectrum of an mzML file; `index` counts every spectrum (MS1 included) from 0, as the label tables do."""
    from pyteomics import mzml

    configs = _analyzers(path)
    only = next(iter(configs.values())) if len(configs) == 1 else ""
    with mzml.MzML(str(path), use_index=False) as reader:
        for i, s in enumerate(reader):
            if int(s.get("ms level") or 0) != 2:
                continue
            m = re.search(r"scan=(\d+)", s["id"])
            first = s["scanList"]["scan"][0]
            rt = first["scan start time"]
            seconds = float(rt) if getattr(rt, "unit_info", "minute") == "second" else float(rt) * 60
            pre = s["precursorList"]["precursor"][0]
            ion = pre["selectedIonList"]["selectedIon"][0]
            act = pre.get("activation", {})
            win = pre.get("isolationWindow", {})
            mz = np.asarray(s["m/z array"], dtype=np.float64)
            it = np.asarray(s["intensity array"], dtype=np.float64)
            order = np.argsort(mz, kind="stable")
            yield Peaks(
                index=i,
                scan=int(m.group(1)) if m else i + 1,
                header=first.get("filter string"),
                retention_time=seconds,
                analyzer=configs.get(first.get("instrumentConfigurationRef", ""), only),
                activation=_activation(act),
                collision_energy=float(act.get("collision energy", np.nan)),
                precursor_mz=float(ion["selected ion m/z"]),
                charge=int(ion.get("charge state") or 0),
                precursor_intensity=float(ion.get("peak intensity", np.nan)),
                isolation_target=float(win.get("isolation window target m/z", np.nan)),
                lower_offset=float(win.get("isolation window lower offset", np.nan)),
                upper_offset=float(win.get("isolation window upper offset", np.nan)),
                mz=mz[order],
                intensity=it[order],
            )


def iter_mgf(path: Path) -> Iterator[Peaks]:
    """Every spectrum of an MGF file (all MS2); `index` is the file position, `scan` the SCANS field when numeric."""
    i, hdr, mz, it = 0, {}, [], []
    with path.open() as f:
        for line in f:
            if line[0] in "0123456789":
                a, b = line.split()[:2]
                mz.append(float(a))
                it.append(float(b))
            elif line.startswith("END IONS"):
                scans = hdr.get("SCANS", "")
                charge = hdr.get("CHARGE", "").rstrip("+")
                m, inten = np.asarray(mz), np.asarray(it)
                order = np.argsort(m, kind="stable")
                yield Peaks(
                    index=i,
                    scan=int(scans) if scans.isdigit() else i,
                    header=None,
                    retention_time=float(hdr["RTINSECONDS"]) if "RTINSECONDS" in hdr else np.nan,
                    analyzer="TOF",
                    activation="CID",
                    collision_energy=np.nan,
                    precursor_mz=float(hdr["PEPMASS"].split()[0]) if "PEPMASS" in hdr else np.nan,
                    charge=int(charge) if charge.isdigit() else 0,
                    precursor_intensity=float(hdr["PEPMASS"].split()[1]) if "PEPMASS" in hdr and len(hdr["PEPMASS"].split()) > 1 else np.nan,
                    isolation_target=np.nan,
                    lower_offset=np.nan,
                    upper_offset=np.nan,
                    mz=m[order],
                    intensity=inten[order],
                )
                i, hdr, mz, it = i + 1, {}, [], []
            elif "=" in line:
                k, v = line.split("=", 1)
                hdr[k] = v.strip()


def read_peaks(path: Path) -> Iterator[Peaks]:
    return (iter_mgf if path.suffix.lower() == ".mgf" else iter_mzml)(path)


def proforma(sequence: str, mods: str, *, fixed_carbamidomethyl: bool) -> str | None:
    """`Name@pos` modifications applied to a plain sequence in the corpus notation; None when one is not expressible."""
    residues = list(sequence)
    prefix = ""
    for site in filter(None, mods.split(";")):
        name, pos = site.rsplit("@", 1)
        p = int(pos)
        if p == 0:
            if name not in TERMINAL_MODS:
                return None
            prefix = TERMINAL_MODS[name] + "-"
            continue
        token = RESIDUE_MODS.get((name, sequence[p - 1]))
        if token is None:
            return None
        residues[p - 1] = token
    if fixed_carbamidomethyl:
        residues = ["C[UNIMOD:4]" if r == "C" else r for r in residues]
    return prefix + "".join(residues)


def calc_mz(proforma_sequence: str, charge: int) -> float:
    prefix, body = (proforma_sequence.split("-", 1) + [""])[:2] if proforma_sequence.startswith("[") else ("", proforma_sequence)
    mass = WATER + N_TERMINAL_MASS.get(prefix, 0.0) + sum(RESIDUE_MASS[t] for t in re.findall(r"[A-Z](?:\[UNIMOD:\d+\])?", body))
    return (mass + charge * PROTON) / charge if charge > 0 else np.nan


def engine_scores(engine: str, label: pd.Series) -> dict[str, float]:
    """The corpus score columns an engine can fill: X!Tandem's hyperscore and expectation are the corpus's own
    columns (its searches were MSFragger, whose hyperscore is X!Tandem's); MaxQuant's PEP gives a probability;
    Mascot's ion score fills none of them."""
    score, pep = float(label["score"]), float(label["pep"]) if pd.notna(label["pep"]) else np.nan
    out = {"hyperscore": np.nan, "expectation": np.nan, "probability": np.nan}
    if engine == "tandem_xml":
        out.update(hyperscore=score, expectation=pep)  # `pep` holds X!Tandem's expectation value in the label table
    elif engine == "maxquant" and not np.isnan(pep):
        out.update(probability=1.0 - pep)
    return out


def run_rows(*, manifest_row: pd.Series, labels: pd.DataFrame, fixed_carbamidomethyl: bool) -> tuple[list[dict], dict]:
    """Every MS2 spectrum of one run as corpus-schema rows, plus the run's counts."""
    run, accession, engine = manifest_row.run, manifest_row.accession, manifest_row.engine
    by_index = labels.set_index("index")
    if not by_index.index.is_unique:
        raise ValueError(f"{run}: label table has several rows per spectrum index")
    rows: list[dict] = []
    counts: Counter = Counter()
    peak_file = Path(manifest_row.peak)
    for p in read_peaks(peak_file):
        counts["ms2"] += 1
        lab = by_index.loc[p.index] if p.index in by_index.index else None
        if lab is not None and int(lab["scan"]) != p.scan:
            raise ValueError(f"{run}: label index {p.index} has scan {lab['scan']}, file has scan {p.scan}")
        identified = lab is not None and isinstance(lab["sequence"], str) and lab["sequence"] != "" and lab["category"] not in ("unidentified", "decoy")
        seq = proforma(lab["sequence"], lab["mods"] if isinstance(lab["mods"], str) else "", fixed_carbamidomethyl=fixed_carbamidomethyl) if identified else ""
        if identified and seq is None:
            counts["unsupported_modification"] += 1
            seq, identified = "", False
        charge = int(lab["charge"]) if identified else p.charge
        counts["identified"] += identified
        if identified:
            counts["passes_floor"] += bool(lab["passes_floor"])
        scores = engine_scores(engine, lab) if identified else {"hyperscore": np.nan, "expectation": np.nan, "probability": np.nan}
        calc = calc_mz(seq, charge) if identified else np.nan
        rows.append(
            {
                "usi": f"mzspec:{accession}:{run}:scan:{p.scan}" + (f":{seq}/{charge}" if identified else ""),
                "index": p.index,
                "scan": f"controllerType=0 controllerNumber=1 scan={p.scan}" if peak_file.suffix.lower() != ".mgf" else str(p.scan),
                "header": p.header,
                "retention_time": p.retention_time,
                "frag_type": p.activation or None,
                "acquisition": "DDA",
                "collision_energy": f"{p.collision_energy:.1f}" if not np.isnan(p.collision_energy) else None,
                "isolation_target": p.isolation_target if not np.isnan(p.isolation_target) else p.precursor_mz,
                "precursor_mz": p.precursor_mz,
                "precursor_charge": charge,
                "precursor_intensity": p.precursor_intensity,
                "lower_offset": p.lower_offset,
                "upper_offset": p.upper_offset,
                "mz_array": p.mz,
                "intensity_array": p.intensity,
                "scale_factor": np.nan,  # the corpus's meaning is not documented in the package; left empty
                "peptide_observed_mz": p.precursor_mz if identified else np.nan,
                "peptide_calc_mz": calc,
                "delta_mass": (p.precursor_mz - calc) * charge if identified else np.nan,
                "retention": np.nan,
                "expectation": scores["expectation"],
                "hyperscore": scores["hyperscore"],
                "nextscore": np.nan,
                "probability": scores["probability"],
                "auc_intensity": np.nan,
                "protein": str(lab["proteins"]) if identified else None,
                "experiment_name": peak_file.name,
                "unmodified_peptide": lab["sequence"] if identified else None,
                "sequence": seq,
                "run": run,
                "species": str(lab["label"]) if lab is not None and lab["label_kind"] == "species" else None,
                "category": str(lab["category"]) if lab is not None else "unlabelled",
                "passes_floor": bool(lab["passes_floor"]) if identified else False,
                "explained": float(lab["explained"]) if lab is not None and pd.notna(lab["explained"]) else np.nan,
                "engine": engine,
                "engine_score": float(lab["score"]) if identified and pd.notna(lab["score"]) else np.nan,
                "analyzer": p.analyzer,
                "identified": identified,
            }
        )
    counts["labelled_rows"] = len(labels)
    counts["frag_types"] = dict(Counter(r["frag_type"] for r in rows))
    counts["analyzers"] = dict(Counter(r["analyzer"] for r in rows))
    counts["charges_identified"] = dict(Counter(r["precursor_charge"] for r in rows if r["identified"]))
    counts["peptides"] = len({r["sequence"] for r in rows if r["identified"]})
    counts["median_peaks"] = float(np.median([len(r["mz_array"]) for r in rows])) if rows else 0.0
    return rows, dict(counts)


def search_data_row(manifest_row: pd.Series, counts: dict) -> dict:
    """One row of `data/search_data.xlsx` describing the run, in that table's vocabulary."""
    analyzers = {a for a in counts["analyzers"] if a}
    frags = {f for f in counts["frag_types"] if f}
    detector = "|".join(n for n, ok in (("Orbitrap", "FT" in analyzers), ("IonTrap", "IT" in analyzers), ("TOF", "TOF" in analyzers)) if ok)
    return {
        "project": manifest_row.accession,
        "file path": f"{manifest_row.accession}/{Path(manifest_row.peak).name}",
        "workflow": "",
        "acquisition": "DDA",
        "detector": detector,
        "fragmentation": "|".join(sorted(frags, key=lambda f: ("HCD", "CID", "ETD").index(f))),
        "instrument": manifest_row.instrument,
        "fasta": "",
        "enzyme": "trypsin",
        "quant": "precursor",
        "modifications": "default (N-term acetylation, Met oxidation)",
        "organism": manifest_row.organism,
    }


def peptide_key(sequence: str) -> str:
    """The corpus registry's peptide identity: modifications dropped, I and L collapsed."""
    return re.sub(r"\[UNIMOD:\d+\]-?", "", sequence).replace("I", "L")


class Writer:
    """A parquet file written one run at a time (one row group per run), so memory holds one run, not the dataset."""

    def __init__(self, path: Path) -> None:
        self.path, self.rows, self._w = path, 0, None

    def append(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        self._w = self._w or pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self._w.write_table(table)
        self.rows += table.num_rows

    def close(self) -> None:
        if self._w is None:  # nothing appended: still leave a valid, empty file
            pq.write_table(SCHEMA.empty_table(), self.path, compression="zstd")
        else:
            self._w.close()
        print(f"  wrote {self.path.name}: {self.rows} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest).query("dataset == @args.dataset")
    if manifest.empty:
        raise SystemExit(f"no runs for {args.dataset} in {args.manifest}")
    labels = pd.read_parquet(manifest.labels.iloc[0])
    fixed_cam = not labels.mods.fillna("").str.contains("Carbamidomethyl").any()  # the rule of proteomies-eval-data's ids.fixed_carbamidomethyl
    out = args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    summary, search_rows = {"dataset": args.dataset, "fixed_carbamidomethyl": fixed_cam, "runs": {}}, []
    all_w, ident_w = Writer(out / f"{args.dataset}-all.parquet"), Writer(out / f"{args.dataset}-identified.parquet")
    keys: list[str] = []  # peptide key of every identified row, in file order of the identified file
    for _, m in manifest.iterrows():
        rows, counts = run_rows(manifest_row=m, labels=labels[labels.run == m.run], fixed_carbamidomethyl=fixed_cam)
        print(f"{m.run}: {counts['ms2']} MS2, {counts['identified']} identified, {counts.get('unsupported_modification', 0)} unsupported mods", flush=True)
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        del rows
        identified = table.filter(pa.compute.equal(table["identified"], True))
        all_w.append(table)
        ident_w.append(identified)
        keys.extend(peptide_key(s) for s in identified["sequence"].to_pylist())
        summary["runs"][m.run] = counts
        search_rows.append(search_data_row(m, counts))
    all_w.close()
    ident_w.close()
    peptides = np.array(sorted(set(keys)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(peptides)
    n = len(peptides)
    split_of = {p: "train" for p in peptides[: int(0.8 * n)]} | {p: "valid" for p in peptides[int(0.8 * n) : int(0.9 * n)]} | {p: "test" for p in peptides[int(0.9 * n) :]}
    assignment = np.array([split_of[k] for k in keys])
    writers = {split: Writer(out / f"{args.dataset}-probe-{split}.parquet") for split in ("train", "valid", "test")}
    reader, offset = pq.ParquetFile(ident_w.path), 0  # one row group per run, re-read so memory stays at one run
    for i in range(reader.num_row_groups):
        group = reader.read_row_group(i)
        mask = assignment[offset : offset + group.num_rows]
        offset += group.num_rows
        for split, w in writers.items():
            w.append(group.filter(pa.array(mask == split)))
    for w in writers.values():
        w.close()
    pd.DataFrame(search_rows, columns=SEARCH_DATA_COLUMNS).to_csv(out / f"{args.dataset}-search_data.csv", index=False)
    summary["totals"] = {
        "ms2": all_w.rows,
        "identified": ident_w.rows,
        "peptides": int(n),
        "probe_split_spectra": {s: w.rows for s, w in writers.items()},
    }
    (out / f"{args.dataset}-summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary["totals"]), flush=True)


if __name__ == "__main__":
    main()
