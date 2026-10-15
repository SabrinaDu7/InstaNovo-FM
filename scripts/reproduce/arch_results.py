"""Render `docs/arch_study/results.md`: the architecture study's checkpoints side by side on the 40M rows of
`results_paper_or_rerun.md`, from the files under `docs/references/rerun/`.

Columns: the released 40M, the batch-2,048 baseline (seed 101, step 90,001), its second seed, then one column per arm
of `docs/arch_study/README.md`. Rows: the 23 rows of the 40M table, read with the very
getters `fill_results.py` uses for that table, re-pointed at each column's `result<N>_<tag>_*` directories; row 23
from `train_<run>/metrics.json` (`scripts/train/export_metrics.py`). A cell is "-" until its files exist, so the
document never shows a number without a JSON behind it. Re-run after every evaluation job.

    python scripts/reproduce/arch_results.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_results as fr  # noqa: E402  (the row getters are its; this module only re-points them)

DOC = REPO / "docs" / "arch_study" / "results.md"
NOTES = "docs/arch_study/README.md"


@dataclass(frozen=True)
class Column:
    label: str  # column header
    tag: str  # `result<N>_<tag>_*` directories under docs/references/rerun/
    validation: str  # directory holding metrics.json for row 23
    note: str  # one line for the sources table


COLUMNS = (
    Column(
        "released 40M",
        "40M",
        "validate_released_40M",
        "`instanovo-fm-mcfm-90k-v0.1.0`, the reference",
    ),
    Column(
        "baseline b2048 (seed 101)",
        "40Mb2048s90k",
        "train_40M_mcfm_90k_b2048",
        "run `train-40M-mcfm-90k-b2048`, step 90,001",
    ),
    Column("seed 102", "arch-seed102", "train_arch-seed102", "the baseline again with `seed=102`"),
    Column(
        "meta-precursor",
        "arch-meta-precursor",
        "train_arch-meta-precursor",
        "metadata tokens: precursor mass, charge bin, fragmentation, collision energy",
    ),
    Column(
        "pa-mask2mask",
        "arch-pa-mask2mask",
        "train_arch-pa-mask2mask",
        "pairwise attention bias kept between masked tokens",
    ),
    Column(
        "enc-fourier",
        "arch-enc-fourier",
        "train_arch-enc-fourier",
        "Fourier peak encoder, periods 2.5 to 3,750 Da",
    ),
    Column(
        "ionladder",
        "arch-ionladder",
        "train_arch-ionladder",
        "ion-ladder residue-mass features on each peak token",
    ),
    Column("ffn3072", "arch-ffn3072", "train_arch-ffn3072", "feed-forward 3,072"),
    Column("enc-dual", "arch-enc-dual", "train_arch-enc-dual", "multiscale plus RBF peak encoder"),
    Column("peaks300", "arch-peaks300", "train_arch-peaks300", "300 peaks kept"),
)


def row_labels() -> dict[int, str]:
    """Row number -> metric label, from the 40M tables of results_paper_or_rerun.md (their one home)."""
    labels: dict[int, str] = {}
    section = False
    for line in fr.DOC.read_text().split("\n"):
        if line.startswith("## "):
            section = line.startswith("## 40M Results")
        m = re.match(r"\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|", line) if section else None
        if m:
            labels[int(m.group(1))] = m.group(2)
    return labels


def cell(col: Column, n: int) -> str:
    script, name, getter = fr.ROWS[("40M", n)]
    if script == "validate_released_40M":
        value = fr._trainer_validation(col.validation)()
    else:
        value = (
            fr._rebind(getter, script, script.replace("_40M_", f"_{col.tag}_"))()
            if col.tag != "40M"
            else getter()
        )
        if value is None and name.startswith("cosinehyperscorecorrelationtask"):
            # the embedded evaluator ran the cosine task as a script of its own (result6_*); the suite runs it inside result3
            value = fr._rebind(getter, script, f"result6_{col.tag}_cosine_hyperscore")()
    return value if value is not None else "-"


def main() -> None:
    labels = row_labels()
    rows = sorted(n for (sec, n) in fr.ROWS if sec == "40M")
    head = "| No. | Metric | " + " | ".join(c.label for c in COLUMNS) + " |"
    sep = "| :-- | :---- | " + " | ".join(":----" for _ in COLUMNS) + " |"
    body = [
        f"| {n} | {labels.get(n, '')} | " + " | ".join(cell(c, n) for c in COLUMNS) + " |"
        for n in rows
    ]
    filled = {c.label: sum(cell(c, n) != "-" for n in rows) for c in COLUMNS}
    sources = [
        "| Column | What | Directories under `docs/references/rerun/` |",
        "| :---- | :---- | :---- |",
    ]
    for c in COLUMNS:
        dirs = sorted(p.name for p in fr.RERUN.glob(f"result*_{c.tag}_*")) + (
            [c.validation] if (fr.RERUN / c.validation / "metrics.json").exists() else []
        )
        sources.append(
            f"| {c.label} | {c.note} | "
            + (", ".join(f"`{d}`" for d in dirs) if dirs else "not yet run")
            + " |"
        )
    text = "\n".join(
        [
            "# Architecture study: the arms beside the baseline",
            "",
            f"Generated by `scripts/reproduce/arch_results.py` from `docs/references/rerun/`; the arms and their overrides are defined in `{NOTES}`.",
            "Every arm is the batch-2,048 40M configuration on MCFM with one change; the rows are those of the 40M table of",
            "`results_paper_or_rerun.md`, read with the same getters (LCFM test unless the row says otherwise; row 23 is the",
            "trainer's own validation on MCFM validation, from the run's MLflow store). A \"-\" means the files do not exist yet.",
            "The probe rows 5 to 7 (mass, m/z, charge) are trivial by construction for `meta-precursor`, which is given them.",
            "",
            head,
            sep,
            *body,
            "",
            "## Sources",
            "",
            *sources,
        ]
    )
    DOC.write_text(text + "\n")
    print(f"wrote {DOC.relative_to(REPO)}; rows filled per column: {filled}")  # noqa: T201


if __name__ == "__main__":
    main()
