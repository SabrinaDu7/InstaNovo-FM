"""Fragment-ion reconstruction measured outside the evaluation framework, for several checkpoints on the same spectra.

Question. The IG task reports the released 40M at 55.0 % fragment-group bin accuracy on LCFM test and the 40M trained
here at 49.2 % (1,224 masked groups of the first 408 qualifying spectra, `docs/references/results_paper_or_rerun.md`
rows 11), while the trainer's validation finds the two equal on bin accuracy over every masked peak (27.4 % against
27.2 %, `docs/references/rerun/{validate_released_40M,train_40M_mcfm_90k}/metrics.json`). The IG metric and the
trainer's differ in four choices (the suite's `instanovofm_evals/tasks/ig_attribution_helper.py:527-670`, once `src/instanovo_fm/eval/embed_eval_tasks/` here,
`src/instanovo_fm/trainer/train.py:550-767`): what is masked (a whole b/y fragment group against the span masking of
training), how the bins are decoded (greedy argmax against the trainer's top-3 joint decoding, with the offset head
teacher-forced on the true group in the trainer's bin accuracy), the Gaussian blur of a masked m/z (10 Da, a fresh draw
of the global RNG at every forward), and the sample size. This script measures the same checkpoints under every
combination, so the gap can be attributed to a choice of the metric or to the models.

Method. `--n` identified spectra of a corpus-schema parquet (seeded draw) go through the checkpoint's own processor
with no masking; their peaks are annotated with the eval's theoretical b/y ions (`match_with_conditional_features`,
the eval's parameters) and grouped into fragment groups (`build_fragment_groups`, the IG task's); every b/y group
with a base peak is masked twice (the whole group; the base peak only) and predicted with `--seeds` blur draws,
decoded three ways (greedy; top-3 joint as the trainer's ppm metrics; greedy with the offset head teacher-forced on the
true group as the trainer's bin accuracy). The trainer's own regime is run too: the processor's span masking with a
fixed seed, every masked peak scored and split into annotated (a theoretical b/y base ion, isotope or loss) and
unannotated. Per prediction: bin correct (group and offset exact), group correct, |error| in ppm from the bin centre.

Outputs under `--out`: `groups.parquet` (one row per masked group, regime, decoder, seed, model), `span.parquet` (one
row per masked peak of the span regime), `summary.json` and `summary.md` (accuracies, blur-seed spread, paired
cluster-bootstrap confidence intervals of each model against the first, by ion type and group size, annotated against
unannotated), each naming the checkpoints, the parquet, the draw and the commit.

    python scripts/analysis/fragment_reconstruction.py --parquet "$DATA/splits/mcfm/mcfm-test-00000-of-00014.parquet" \\
        --checkpoint 40M=$CHECKPOINTS/instanovo-fm-mcfm-90k-v0.1.0.ckpt --checkpoint 40M-ours=... --n 2000 --seeds 3 --out <dir>
    python scripts/analysis/fragment_reconstruction.py --summarize <dir>      # tables only, from the parquet files
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ANNOTATION = {"ppm_tol": 10.0, "ion_types": ("b", "y"), "add_losses": True, "loss_types": ("H2O", "NH3"), "add_isotopes": True, "max_isotope": 3,
              "isotope_intensity_threshold": 0.02, "add_precursor": True}  # the eval's theoretical-spectrum settings (embedding_io.py)
DECODERS = ("greedy", "top3", "teacher")
REGIMES = ("group", "base")


def load_rows(parquet: str, n: int, seed: int) -> pd.DataFrame:
    """`n` identified spectra (non-empty sequence, charge > 0) of the first file of `parquet`, a seeded draw."""
    import glob

    files = sorted(glob.glob(os.path.expandvars(os.path.expanduser(parquet))))
    if not files:
        raise SystemExit(f"no parquet matches {parquet}")
    cols = ["mz_array", "intensity_array", "precursor_mz", "precursor_charge", "sequence", "usi"]
    df = pq.read_table(files[0], columns=cols).to_pandas()
    df = df[(df.sequence.fillna("") != "") & (df.precursor_charge > 0)]
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def processor_for(cfg: Any, *, masking: bool) -> Any:
    """The checkpoint's own processor (the trainer's validation processor, `trainer/train.py:398-445`), with or without masking."""
    from instanovo_fm.data.data import FoundationalDataProcessor

    m = cfg.get("masking", {})
    return FoundationalDataProcessor(
        n_peaks=cfg.get("n_peaks", 200), min_mz=cfg.get("min_mz", 50.0), max_mz=cfg.get("max_mz", 2500.0),
        min_intensity=cfg.get("min_intensity", 0.01), remove_precursor_tol=cfg.get("remove_precursor_tol", 0.0),
        use_spectrum_utils=cfg.get("use_spectrum_utils", False), normalize_mz=cfg.get("normalize_mz", True),
        peak_ordering=m.get("ordering_strategy", "sorted"), residue_set=None, annotated=True, return_str=True, metadata_columns=[],
        masking_strategy=m.get("strategy", "thompson_span") if masking else "none",
        mask_portion=m.get("mask_portion", 0.30), thompson_alpha=m.get("alpha", 0.5), thompson_beta=m.get("beta", 0.5),
        thompson_kappa=m.get("kappa", 4.0), thompson_gamma=m.get("gamma", 0.7), span_min=m.get("span_min", 4), span_max=m.get("span_max", 7),
        span_bidirectional=m.get("bidirectional", True), include_isotopes=m.get("include_isotopes", True), isotope_ppm=m.get("isotope_ppm", 25.0),
        isotope_da_floor=m.get("isotope_da_floor", 0.02), isotope_max_charge=m.get("isotope_max_charge", 3),
        isotope_max_order=m.get("isotope_max_order", 2), max_total_mask_ratio=m.get("max_total_mask_ratio", 0.40),
    )


def annotate(mz: np.ndarray, intensity: np.ndarray, sequence: str, charge: int) -> tuple[list, list, list]:
    """matched_annotation, feature_type, parent_annotation per real peak, as the eval computes them."""
    from instanovo_fm.common.dataset import DataProcessor
    from instanovo_fm.utils.theoretical_spectra import match_with_conditional_features

    for keep in (True, False):  # the eval's fallback: a modification pyOpenMS rejects (e.g. UNIMOD:214) -> the unmodified sequence
        clean = DataProcessor.clean_peptide_for_pyopenms(sequence, keep_modifications=keep, add_carbamidomethyl=True) or sequence
        try:
            res = match_with_conditional_features(mz, intensity, clean, precursor_charge=int(charge), **ANNOTATION)
        except (ValueError, RuntimeError):
            continue
        return res["matched_annotation"], res["feature_type"], res["parent_annotation"]
    n = len(mz)
    return [None] * n, [None] * n, [None] * n


def decode(model: Any, group_logits: torch.Tensor, offset_logits: torch.Tensor, aux: dict, decoder: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted (group, offset) per position: greedy argmax (the IG task, and the trainer's bin accuracy when the forward was
    teacher-forced) or the trainer's top-3 joint decoding (`trainer/train.py:688-745`)."""
    if decoder != "top3":
        return group_logits.argmax(-1), offset_logits.argmax(-1)
    head = model.prediction_heads.mz_head
    probs, groups = torch.softmax(group_logits.float(), -1).topk(3, dim=-1)
    last_group, last_size = int(model.n_bin_groups_tensor.item()) - 1, int(model.last_group_size_tensor.item())
    best_joint = torch.full(groups.shape[:-1], -1.0, device=groups.device)
    best_g, best_o = groups[..., 0].clone(), torch.zeros_like(groups[..., 0])
    for k in range(groups.shape[-1]):
        g_k = groups[..., k]
        olog = head.offset_head(torch.cat([aux["x_tokens"], head.group_embedding(g_k)], dim=-1))
        if last_size < olog.shape[-1]:
            olog[g_k == last_group, last_size:] = float("-inf")
        o_prob, o_k = torch.softmax(olog.float(), -1).max(-1)
        joint = probs[..., k] * o_prob
        better = joint > best_joint
        best_joint, best_g, best_o = torch.where(better, joint, best_joint), torch.where(better, g_k, best_g), torch.where(better, o_k, best_o)
    return best_g, best_o


@torch.no_grad()
def forward(model: Any, spectra: torch.Tensor, pad: torch.Tensor, mlm: torch.Tensor, *, seed: int, teacher: bool) -> tuple:
    """One forward with the blur noise seeded; `teacher` conditions the offset head on the true group of every position."""
    torch.manual_seed(seed)
    tg = to = None
    if teacher:
        tg, to = model.binning_strategy.mz_to_bin_groups((spectra[..., 0] * model.max_mz).reshape(-1))
        tg, to = tg.reshape(spectra.shape[:2]), to.reshape(spectra.shape[:2])
    preds, aux = model(spectra, spectra_mask=pad, mlm_mask=mlm, target_groups=tg, target_offsets=to, bin_edges=model.bin_edges)
    return preds[0], preds[1], aux


def batches(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def evaluate_checkpoint(*, tag: str, path: str, rows: pd.DataFrame, seeds: int, batch_size: int, device: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from instanovofm_evals.tasks.ig_attribution_helper import build_fragment_groups  # the IG task's own grouping, from the suite
    from instanovo_fm.model.encoder import FoundationModel

    model, cfg = FoundationModel.load(path)
    model = model.to(device).eval()
    plain, span = processor_for(cfg, masking=False), processor_for(cfg, masking=True)
    processed = [plain.process_row({k: (np.asarray(v) if k.endswith("_array") else v) for k, v in r.items()}) for r in rows.to_dict("records")]
    binning, max_mz = model.binning_strategy, float(model.max_mz)

    # Fragment groups per spectrum, from the processed peaks (what the model sees).
    jobs: list[dict] = []  # one per (spectrum, group, regime)
    info = {"spectra": len(processed), "spectra_with_groups": 0, "groups": 0}
    for s, item in enumerate(processed):
        spec = item["spectra"].numpy() if hasattr(item["spectra"], "numpy") else np.asarray(item["spectra"])
        real = (spec[:, 0] > 0) & (spec[:, 1] > 0)
        mz, it = spec[real, 0] * max_mz, spec[real, 1]
        ann, ftype, parent = annotate(mz, it, rows.sequence[s], int(rows.precursor_charge[s]))
        groups = [g for g in build_fragment_groups(ann, parent, ftype, mz, it) if g.ion_type in ("b", "y") and g.base_idx is not None]
        if not groups:
            continue
        info["spectra_with_groups"] += 1
        rank = (-it).argsort().argsort() + 1  # 1 = most intense
        for g in groups:
            info["groups"] += 1
            base = {"spectrum": s, "usi": rows.usi[s], "group_key": g.group_key, "ion_type": g.ion_type, "charge": g.group_key.count("+"),
                    "group_size": len(g.peak_indices), "n_isotopes": sum("[+" in (ann[i] or "") for i in g.peak_indices),
                    "base_rank": int(rank[g.base_idx]), "true_mz": float(mz[g.base_idx]), "base_idx": int(g.base_idx)}
            jobs.append({**base, "regime": "group", "mask_idx": list(g.peak_indices)})
            jobs.append({**base, "regime": "base", "mask_idx": [int(g.base_idx)]})

    true_g_all, true_o_all = binning.mz_to_bin_groups(torch.tensor([j["true_mz"] for j in jobs], dtype=torch.float32))
    records: list[dict] = []
    for chunk_start, chunk in zip(range(0, len(jobs), batch_size), batches(jobs, batch_size)):
        batch = plain.collate_fn([processed[j["spectrum"]] for j in chunk])
        spectra, pad = batch["spectra"].to(device), batch["spectra_mask"].to(device)
        mlm = torch.zeros_like(pad)
        for b, j in enumerate(chunk):
            mlm[b, j["mask_idx"]] = True
        base_pos = torch.tensor([j["base_idx"] for j in chunk], device=device)
        rows_idx = torch.arange(len(chunk), device=device)
        for seed in range(seeds):
            outs = {False: forward(model, spectra, pad, mlm, seed=seed, teacher=False), True: forward(model, spectra, pad, mlm, seed=seed, teacher=True)}
            for decoder in DECODERS:
                gl, ol, aux = outs[decoder == "teacher"]
                g, o = decode(model, gl, ol, aux, decoder)
                g, o = g[rows_idx, base_pos].cpu(), o[rows_idx, base_pos].cpu()
                pred_mz = binning.bin_groups_to_mz(g, o)
                for b, j in enumerate(chunk):
                    i = chunk_start + b
                    tg, to_ = int(true_g_all[i]), int(true_o_all[i])
                    records.append({"model": tag, **{k: v for k, v in j.items() if k != "mask_idx"}, "seed": seed, "decoder": decoder,
                                    "pred_group": int(g[b]), "pred_offset": int(o[b]), "true_group": tg, "true_offset": to_,
                                    "group_correct": int(g[b]) == tg, "bin_correct": int(g[b]) == tg and int(o[b]) == to_,
                                    "error_ppm": abs(float(pred_mz[b]) - j["true_mz"]) / j["true_mz"] * 1e6})
    groups_df = pd.DataFrame(records)

    # The trainer's regime: the processor's span masking (seeded), every masked peak, annotated or not.
    span_records: list[dict] = []
    annotations = {}
    for s, item in enumerate(processed):
        spec = item["spectra"].numpy() if hasattr(item["spectra"], "numpy") else np.asarray(item["spectra"])
        real = (spec[:, 0] > 0) & (spec[:, 1] > 0)
        ann, ftype, _ = annotate(spec[real, 0] * max_mz, spec[real, 1], rows.sequence[s], int(rows.precursor_charge[s]))
        annotations[s] = (ann, ftype)
    span_items = [span.process_row({k: (np.asarray(v) if k.endswith("_array") else v) for k, v in r.items()}) for r in rows.to_dict("records")]
    for seed in range(seeds):
        for chunk_start, chunk in zip(range(0, len(span_items), batch_size), batches(span_items, batch_size)):
            np.random.seed(seed + chunk_start)
            torch.manual_seed(seed + chunk_start)
            batch = span.collate_fn(chunk)
            spectra, pad, mlm = batch["spectra"].to(device), batch["spectra_mask"].to(device), batch["peak_mask"].to(device)
            valid = mlm & ~pad
            tg_all, to_all = binning.mz_to_bin_groups((spectra[..., 0] * max_mz).reshape(-1))
            tg_all, to_all = tg_all.reshape(spectra.shape[:2]), to_all.reshape(spectra.shape[:2])
            for teacher in (False, True):
                gl, ol, aux = forward(model, spectra, pad, mlm, seed=seed, teacher=teacher)
                for decoder in (("greedy", "top3") if not teacher else ("teacher",)):
                    g, o = decode(model, gl, ol, aux, decoder)
                    pred_mz = binning.bin_groups_to_mz(g.reshape(-1).cpu(), o.reshape(-1).cpu()).reshape(spectra.shape[:2])
                    true_mz = (spectra[..., 0] * max_mz).cpu()
                    for b in range(spectra.shape[0]):
                        s = chunk_start + b
                        ann, ftype = annotations[s]
                        for pos in torch.nonzero(valid[b]).flatten().tolist():
                            a = ann[pos] if pos < len(ann) else None
                            span_records.append({"model": tag, "spectrum": s, "seed": seed, "decoder": decoder, "position": pos,
                                                 "annotated": a is not None, "feature_type": (ftype[pos] if pos < len(ftype) else None) or "none",
                                                 "group_correct": bool(g[b, pos] == tg_all[b, pos]),
                                                 "bin_correct": bool(g[b, pos] == tg_all[b, pos] and o[b, pos] == to_all[b, pos]),
                                                 "error_ppm": abs(float(pred_mz[b, pos]) - float(true_mz[b, pos])) / float(true_mz[b, pos]) * 1e6})
    return groups_df, pd.DataFrame(span_records), info


def bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, *, reps: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Difference of bin accuracy (b minus a) with a 95 % interval from resampling spectra (the clusters)."""
    pa = a.groupby("spectrum").bin_correct.mean()
    pb = b.groupby("spectrum").bin_correct.mean()
    common = pa.index.intersection(pb.index)
    d = (pb[common] - pa[common]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)])
    return float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarize(out: Path) -> str:
    g = pd.read_parquet(out / "groups.parquet")
    sp = pd.read_parquet(out / "span.parquet")
    meta = json.loads((out / "run.json").read_text())
    models = list(dict.fromkeys(g.model))
    lines = [f"# Fragment-ion reconstruction outside the framework ({meta['parquet']}, {meta['n']} spectra, {meta['seeds']} blur seeds)", ""]
    lines += ["Produced by `scripts/analysis/fragment_reconstruction.py` (commit " + meta["commit"] + "). Bin correct = predicted group and offset both exact "
              "(0.2 Da bin); ppm = |bin centre - true m/z| / true m/z. Greedy = the IG task's decoding; top-3 = the trainer's joint decoding for its ppm metrics; "
              "teacher = the offset head conditioned on the true group, as in the trainer's bin accuracy. Group = the whole fragment group masked (the IG task); "
              "base = the base ion alone masked.", ""]
    n_groups = g[(g.model == models[0]) & (g.regime == "group") & (g.decoder == "greedy") & (g.seed == 0)]
    lines += [f"Masked groups per model and regime: {len(n_groups):,} ({n_groups.spectrum.nunique():,} spectra with at least one b/y group).", ""]
    # Table A
    lines += ["## A. Bin accuracy by masking regime and decoder (mean over blur seeds; ± = spread of the per-seed accuracies)", ""]
    header = "| model | " + " | ".join(f"{r} / {d}" for r in REGIMES for d in DECODERS) + " |"
    lines += [header, "|---|" + "---|" * (len(REGIMES) * len(DECODERS))]
    for m in models:
        cells = []
        for r in REGIMES:
            for d in DECODERS:
                per_seed = g[(g.model == m) & (g.regime == r) & (g.decoder == d)].groupby("seed").bin_correct.mean()
                cells.append(f"{100 * per_seed.mean():.1f} % ± {100 * per_seed.std(ddof=0):.1f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    # Table B
    lines += [f"## B. Difference against {models[0]} (points of bin accuracy, 95 % interval from resampling spectra; seeds pooled)", "",
              "| model | " + " | ".join(f"{r} / {d}" for r in REGIMES for d in DECODERS) + " |", "|---|" + "---|" * (len(REGIMES) * len(DECODERS))]
    for m in models[1:]:
        cells = []
        for r in REGIMES:
            for d in DECODERS:
                a = g[(g.model == models[0]) & (g.regime == r) & (g.decoder == d)]
                b = g[(g.model == m) & (g.regime == r) & (g.decoder == d)]
                mean, lo, hi = bootstrap_diff(a, b)
                cells.append(f"{100 * mean:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    # Table C: ion type, group size, median ppm (group regime, greedy)
    lines += ["## C. Group regime, greedy (the IG task's setting): by ion type and group size; median ppm from the bin centre", "",
              "| model | b | y | single-peak groups | multi-peak groups | group accuracy | median ppm (all) | median ppm (correct bin) | median ppm (wrong bin) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        sub = g[(g.model == m) & (g.regime == "group") & (g.decoder == "greedy")]

        def acc(frame: pd.DataFrame) -> str:
            return f"{100 * frame.bin_correct.mean():.1f} % (n={len(frame[frame.seed == 0]):,})"

        lines.append(f"| {m} | {acc(sub[sub.ion_type == 'b'])} | {acc(sub[sub.ion_type == 'y'])} | {acc(sub[sub.group_size == 1])} | {acc(sub[sub.group_size > 1])} | "
                     f"{100 * sub.group_correct.mean():.1f} % | {sub.error_ppm.median():.0f} | {sub[sub.bin_correct].error_ppm.median():.0f} | {sub[~sub.bin_correct].error_ppm.median():.0f} |")
    lines.append("")
    # Table D: trainer regime
    lines += ["## D. The trainer's regime (span masking with isotope co-masking, every masked peak)", "",
              "| model | decoder | annotated: bin / group | unannotated: bin / group | all: bin / group | share annotated | median ppm all / annotated |",
              "|---|---|---|---|---|---|---|"]
    for m in models:
        for d in ("greedy", "top3", "teacher"):
            sub = sp[(sp.model == m) & (sp.decoder == d)]
            an, un = sub[sub.annotated], sub[~sub.annotated]
            lines.append(f"| {m} | {d} | {100 * an.bin_correct.mean():.1f} / {100 * an.group_correct.mean():.1f} % | "
                         f"{100 * un.bin_correct.mean():.1f} / {100 * un.group_correct.mean():.1f} % | {100 * sub.bin_correct.mean():.1f} / {100 * sub.group_correct.mean():.1f} % | "
                         f"{100 * sub.annotated.mean():.1f} % | {sub.error_ppm.median():.0f} / {an.error_ppm.median():.0f} |")
    lines.append("")
    text = "\n".join(lines)
    (out / "summary.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", help="corpus-schema parquet file or glob (the first file is read)")
    ap.add_argument("--checkpoint", action="append", default=[], help="tag=path, repeatable; the first is the reference")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summarize", action="store_true", help="only rebuild summary.md from the parquet files in --out")
    args = ap.parse_args()
    if args.summarize:
        print(summarize(args.out))
        return
    if not (args.parquet and args.checkpoint):
        raise SystemExit("--parquet and at least one --checkpoint are required")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.parquet, args.n, seed=42)
    groups, spans, infos, t0 = [], [], {}, time.time()
    for spec in args.checkpoint:
        tag, path = spec.split("=", 1)
        t1 = time.time()
        g, s, info = evaluate_checkpoint(tag=tag, path=os.path.expandvars(os.path.expanduser(path)), rows=rows, seeds=args.seeds,
                                         batch_size=args.batch_size, device=args.device)
        groups.append(g)
        spans.append(s)
        infos[tag] = {**info, "seconds": round(time.time() - t1)}
        print(f"{tag}: {info['groups']} groups in {info['spectra_with_groups']} spectra, {len(s):,} span-masked peaks, {time.time() - t1:.0f} s", flush=True)
    pd.concat(groups, ignore_index=True).to_parquet(args.out / "groups.parquet")
    pd.concat(spans, ignore_index=True).to_parquet(args.out / "span.parquet")
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True, check=False).stdout.strip()
    (args.out / "run.json").write_text(json.dumps({"parquet": args.parquet, "n": len(rows), "seeds": args.seeds, "batch_size": args.batch_size,
                                                   "checkpoints": dict(c.split("=", 1) for c in args.checkpoint), "annotation": ANNOTATION,
                                                   "per_model": infos, "commit": commit, "seconds": round(time.time() - t0),
                                                   "host": os.uname().nodename}, indent=1))
    print(summarize(args.out))


if __name__ == "__main__":
    main()
