"""Plot every metric of `docs/results/eval_<dataset>.md` as one bar chart per metric.

    uv run --group figures python scripts/evals/plot_results.py

Reads the tables `render_results.py` wrote, so the figure shows exactly the numbers in the documents. One panel per
metric (a cell holding "a / b" becomes two panels), datasets on the x axis, one bar per checkpoint, and the released
40M on the corpus split as a dashed line where the document has a reference column. A missing cell ("-") is marked
"n/a"; a cell scored with another metric than its row label (e.g. "0.999 (balanced accuracy)") is marked "*".
Writes `docs/results/eval_metrics.png` (groups = datasets, bars = checkpoints) and `eval_metrics2.png` (groups =
checkpoints, bars = datasets, plus the corpus test split the paper evaluates on: our rerun of each checkpoint as a
bar and the paper's reported number as a diamond).
"""

from __future__ import annotations

import re
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # render_results, for the corpus getters
RESULTS = REPO / "docs" / "results"
DATASETS = [("ms2bac", "MS2Bac"), ("ups1", "UPS1"), ("proteometools", "ProteomeTools"), ("campi", "CAMPI")]
MODELS = ["released 40M", "40M trained here", "released 89M", "40M batch 2,048, step 80k", "40M batch 2,048, step 90k"]
SHORT_MODELS = ["released\n40M", "40M trained\nhere", "released\n89M", "40M b2048\nstep 80k", "40M b2048\nstep 90k"]  # tick labels of eval_metrics2
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5 of the dataviz reference palette, in slot order
# (validated: an order with yellow beside orange fails the normal-vision floor, so the batch-2,048 checkpoints come last)
DATASET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4, for eval_metrics2 (bars per dataset)
CORPUS_COLOR = "#a3a29c"  # neutral grey: the corpus split is the reference, not another dataset
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*(?:e-?\d+)?")
# Rows whose parts the document joins with " / ", and the name of each part.
SPLITS = {
    "Bin accuracy y / b ions": ["Bin accuracy, y ions", "Bin accuracy, b ions"],
    "Median error y / b ions": ["Median error, y ions (ppm)", "Median error, b ions (ppm)"],
    "Confidence AUROC, per spectrum / pooled": ["Confidence AUROC, per spectrum", "Confidence AUROC, pooled"],
    "EVoC clusters / noise / purity": ["EVoC clusters", "EVoC noise fraction", "EVoC purity"],
    "ESM2 alignment RSA / CKA": ["ESM2 alignment RSA", "ESM2 alignment CKA"],
    "Glass Box residual / top importance": ["Glass Box residual", "Glass Box top importance"],
}
# The corpus split the paper evaluates on, added as a fifth group to eval_metrics2: our reruns of each checkpoint
# (`docs/references/rerun/`, read with the getters of `render_results.py`) and, as a marker, the paper's own number
# (`docs/references/results_paper_or_rerun.md`: Table S5 for the 40M; Tables S5, S10 and the main text for the 89M).
CORPUS = "LCFM test"
REFERENCE = REPO / "docs" / "references" / "rerun"
MODEL_TAGS = ["40M", "40Mours", "89M", "40Mb2048", "40Mb2048s90k"]
VALIDATION = {"40M": "validate_released_40M", "40Mours": "train_40M_mcfm_90k", "40Mb2048": "train_40M_mcfm_90k_b2048/step_80001", "40Mb2048s90k": "train_40M_mcfm_90k_b2048"}  # MCFM validation; none for the 89M
PAPER = {
    "Fragment type (macro-F1)": [0.781, None, 0.855],
    "Instrument (macro-F1)": [0.697, None, 0.804],
    "PTM presence (balanced accuracy)": [0.751, None, 0.802],
    "Modification class (macro-F1)": [None, None, 0.622],
    "Hydrophobicity (R²)": [0.518, None, 0.605],
    "Precursor mass (R²)": [0.698, None, 0.732],
    "Precursor m/z (R²)": [0.896, None, 0.929],
    "Charge (macro-F1)": [0.602, None, 0.650],
    "Spectrum confidence (R²)": [0.978, None, 0.973],
    "Recall@1": [0.307, None, 0.215],
    "mAP@20": [0.126, None, 0.076],
    "Fragment-group bin accuracy (IG task)": [None, None, 70.1],
    "Bin accuracy, y ions": [None, None, 74.3],
    "Bin accuracy, b ions": [None, None, 59.2],
    "Median error, y ions (ppm)": [None, None, 89.8],
    "Median error, b ions (ppm)": [None, None, 196.4],
    "Peak-type accuracy": [None, None, 77.5],
    "Peak-type macro-F1": [None, None, 0.575],
    "Cross-spectrum AUROC": [None, None, 0.837],
    "Confidence AUROC, per spectrum": [None, None, 0.658],
}
LOWER_IS_BETTER = ("error", "MAE", "noise", "Mean cosine", "Top-component", "Anisotropy", "residual")


def parse(cell: str) -> tuple[list[float | None], bool]:
    """Numbers of a cell (one per " / " part) and whether it names a substitute metric in parentheses."""
    cell = cell.strip()
    if cell in ("-", ""):
        return [None], False
    substitute = bool(re.search(r"\((?:balanced accuracy|macro-F1)\)", cell))
    cell = re.sub(r"\s*\([^)]*\)", "", cell)  # "(balanced accuracy)", "(3694)" pair counts
    values = []
    for part in cell.split(" / "):
        m = NUMBER.search(part)
        values.append(float(m.group().replace(",", "")) if m else None)
    return values, substitute


def read_tables(path: Path) -> dict[str, dict[str, list[tuple[list[float | None], bool]]]]:
    """{section: {metric: [cell per column]}} for the tables under "## Results"."""
    sections: dict[str, dict] = {}
    section = None
    in_results = False
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            in_results = line.strip() == "## Results"
        elif in_results and line.startswith("### "):
            section = line[4:].strip()
            sections[section] = {}
        elif in_results and section and line.startswith("|") and not line.startswith(("| metric", "|---")):
            cells = line.replace("|error|", "error").strip().strip("|").split("|")
            sections[section][cells[0].strip()] = [parse(c) for c in cells[1:]]
    return sections


def corpus_cells() -> dict[tuple[str, str], list[tuple[list[float | None], bool]]]:
    """{(section, row): [cell per checkpoint]} on the corpus split, from the reruns in `docs/references/rerun/`."""
    import render_results as rr

    cells = {}
    for _, section, ref_folder, rows in rr.TABLES:
        for label, getter in rows:
            label = label.replace("|error|", "error")
            per_model = []
            for tag in MODEL_TAGS:
                if section.startswith("Trainer validation"):
                    folders = [VALIDATION.get(tag)]
                elif ref_folder:
                    folders = [ref_folder.replace("_40M_", f"_{tag}_"), f"result6_{tag}_cosine_hyperscore"]
                else:
                    folders = []  # the corpus holds identified spectra only
                text = next((t for f in folders if f and (t := getter(REFERENCE / f))), None)
                per_model.append(parse(text) if text else ([None], False))
            cells[(section, label)] = per_model
    return cells


def collect(corpus: bool = False):
    """[(section, metric, values[group][model], substitute[group][model], reference or None, paper[model])].

    Groups are the datasets, plus the corpus split last when `corpus` is set.
    """
    docs = {key: read_tables(RESULTS / f"eval_{key}.md") for key, _ in DATASETS}
    extra = corpus_cells() if corpus else {}
    n_groups = len(DATASETS) + bool(corpus)
    panels = []
    for section, rows in docs[DATASETS[0][0]].items():
        for row in rows:
            names = SPLITS.get(row, [row])
            for part, name in enumerate(names):
                values = np.full((n_groups, len(MODELS)), np.nan)
                substitute = np.zeros_like(values, dtype=bool)
                reference = None
                for d, (key, _) in enumerate(DATASETS):
                    cells = docs[key].get(section, {}).get(row, [])
                    for m, (nums, sub) in enumerate(cells[: len(MODELS)]):
                        v = nums[part] if part < len(nums) else None
                        values[d, m] = np.nan if v is None else v
                        substitute[d, m] = sub
                    if len(cells) > len(MODELS):
                        nums = cells[len(MODELS)][0]
                        reference = nums[part] if part < len(nums) else None
                if corpus:
                    for m, (nums, sub) in enumerate(extra.get((section, row), [([None], False)] * len(MODELS))):
                        v = nums[part] if part < len(nums) else None
                        values[-1, m] = np.nan if v is None else v
                        substitute[-1, m] = sub
                if np.isnan(values).all():
                    continue
                paper = [np.nan if v is None else v for v in PAPER.get(name, [None] * len(MODELS))]
                panels.append((section, name, values, substitute, reference, paper))
    return panels


def plot(panels, by: str, out: Path, corpus: bool = False) -> None:
    """by="dataset": one group of bars per dataset, one bar per checkpoint; by="model": the transpose.

    With `corpus`, the panels carry the corpus split as a last dataset: its bars are our reruns, a black diamond marks
    the paper's number, and the dashed line (the released 40M's rerun of the same thing) is left out as redundant.
    """
    datasets = [label for _, label in DATASETS] + ([CORPUS] if corpus else [])
    groups, series = (datasets, MODELS) if by == "dataset" else (SHORT_MODELS, datasets)
    colors = COLORS if by == "dataset" else DATASET_COLORS + ([CORPUS_COLOR] if corpus else [])
    sections = list(dict.fromkeys(p[0] for p in panels))
    ncols = 5
    rows_per_section = [int(np.ceil(sum(p[0] == s for p in panels) / ncols)) for s in sections]
    fig = plt.figure(figsize=(ncols * 3.3, sum(rows_per_section) * 2.55 + len(sections) * 0.45 + 1.2), facecolor="white")
    grid = fig.add_gridspec(sum(rows_per_section) + len(sections), ncols,
                            height_ratios=[h for n in rows_per_section for h in [0.18] + [1] * n],
                            hspace=0.75, wspace=0.32, top=0.95, bottom=0.02, left=0.04, right=0.99)
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    row = 0
    for section, n_rows in zip(sections, rows_per_section):
        title_ax = fig.add_subplot(grid[row, :])
        title_ax.axis("off")
        title_ax.text(0, 0, section, fontsize=13, fontweight="bold", color=INK, va="bottom")
        row += 1
        for i, (_, name, values, substitute, reference, paper) in enumerate(p for p in panels if p[0] == section):
            paper = np.array(paper)
            if by == "model":
                values, substitute = values.T, substitute.T
            ax = fig.add_subplot(grid[row + i // ncols, i % ncols])
            for s_ in range(len(series)):
                xs = x + (s_ - (len(series) - 1) / 2) * width
                ax.bar(xs, np.nan_to_num(values[:, s_]), width * 0.9, color=colors[s_], zorder=2)
                for g in range(len(groups)):
                    if np.isnan(values[g, s_]):
                        ax.text(xs[g], 0, "n/a", ha="center", va="bottom", fontsize=6, color=MUTED, rotation=90)
                    elif substitute[g, s_]:
                        ax.text(xs[g], values[g, s_], "*", ha="center", va="bottom", fontsize=9, color=INK)
            if corpus:  # the paper's number, on the corpus bar of each checkpoint that has one
                for m in np.flatnonzero(~np.isnan(paper)):
                    g, s_ = (len(groups) - 1, m) if by == "dataset" else (m, len(series) - 1)
                    ax.plot(x[g] + (s_ - (len(series) - 1) / 2) * width, paper[m], marker="D", ms=5, color=INK,
                            markeredgecolor="white", markeredgewidth=0.8, ls="none", zorder=4)
            # The reference is scored with the row label's metric; drop it where these bars use the substitute.
            elif reference is not None and not substitute[~np.isnan(values)].all():
                if by == "dataset":
                    ax.axhline(reference, color=MUTED, ls="--", lw=1.2, zorder=3)
                else:  # the reference is the released 40M's, so it spans that group only
                    ax.hlines(reference, x[0] - 0.45, x[0] + 0.45, color=MUTED, ls="--", lw=1.2, zorder=3)
            arrow = " \u2193" if any(k in name for k in LOWER_IS_BETTER) else ""
            ax.set_title(name + arrow, fontsize=9, color=INK, loc="left")
            ax.set_xticks(x, groups, fontsize=7.5, color=MUTED)
            ax.tick_params(axis="y", labelsize=7.5, colors=MUTED, length=0)
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
            ax.axhline(0, color=MUTED, lw=0.8, zorder=3)
            for spine in ax.spines.values():
                spine.set_visible(False)
        row += n_rows
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    labels = [label.replace("\n", " ") for label in series]
    if corpus:
        handles.append(plt.Line2D([], [], marker="D", ms=6, color=INK, ls="none"))
        labels.append("paper's reported number (LCFM test)")
        note = (f"{CORPUS} = the corpus test split the paper evaluates on, rerun here with each checkpoint "
                "(trainer validation: MCFM validation; no 89M rerun).\n")
    else:
        handles.append(plt.Line2D([], [], color=MUTED, ls="--", lw=1.2))
        labels.append("released 40M on the corpus split (LCFM test / MCFM validation)")
        note = ""
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.972, "InstaNovo-FM evaluation suite on external datasets. " + note + "n/a = not run or one-class "
             "target; * = scored with the probe's substitute metric (balanced accuracy / macro-F1); \u2193 = lower is better.",
             ha="center", va="top", fontsize=9, color=MUTED, linespacing=1.5)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out} ({len(panels)} panels)")


def main() -> None:
    plot(collect(), "dataset", RESULTS / "eval_metrics.png")
    plot(collect(corpus=True), "model", RESULTS / "eval_metrics2.png", corpus=True)


if __name__ == "__main__":
    main()
