"""Profile one training step of the 40M foundation model, baseline against the pairwise-attention (PA) path, on this
machine's GPU: where the PA arm's time goes (smoke run of 15/10/2026: 3.03 s per optimizer step against 0.74, 59.7 GB
against 30 GB on an H100 at batch 2,048), measured with `torch.profiler` ranges around the three PA stages
(`PairwiseAttentionBias`: Fourier features and the f_pw MLP over every peak pair; `BatchedPairwiseProjection`: LayerNorm and
the projection to one bias per head and layer; `FlashMHA`: attention, which runs SDPA's memory-efficient kernel when a
bias is present and the flash kernel otherwise), the backward pass and the optimizer step.

Analysis only: real batches from a corpus-schema parquet through the training processor and collate (masking
included), the model built as `trainer/train.py::setup_model` builds it from `configs/foundational.yaml` with
`dataset=mcfm`, fp16 autocast with a GradScaler and AdamW as the trainer runs, dropout and gradient checkpointing on.
Absolute times are this GPU's; the split between stages and the ratios between variants are what transfers.

    python scripts/analysis/profile_pa.py --parquet <mcfm-test shard> [--batch 32] [--steps 10] [--warmup 3]
        [--variants baseline,baseline-compiled,pa,pa-mask2mask] [--pa-chunk N] [--out docs/arch_study/profile_pa.md]

`--pa-chunk N` sets the batch-chunk size of the two chunked PA modules (64 in the code: 16 chunks at the trainer's
micro-batch 1,024), so the chunking and its gradient checkpointing can be exercised at a small batch.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import polars as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function

import instanovo_fm
from instanovo_fm.data import FoundationalDataProcessor
from instanovo_fm.model import FoundationModel
from instanovo_fm.trainer.losses import (
    compute_auxiliary_losses,
    compute_classification_loss,
    compute_total_loss,
)
from instanovo_fm.trainer.utils import mz_to_bin_groups

REPO = Path(__file__).resolve().parents[2]
CONFIGS = Path(instanovo_fm.__file__).parent / "configs"
VARIANTS: dict[
    str, tuple[str, ...]
] = {  # name -> Hydra overrides on the baseline (compile is a separate flag below)
    "baseline": (),
    "baseline-compiled": (),
    "pa": ("model.architecture.relative_bias.type=pa",),
    "pa-mask2mask": (
        "model.architecture.relative_bias.type=pa",
        "model.architecture.relative_bias.config.pa_mask_to_mask=true",
    ),
    "pa-compiled": ("model.architecture.relative_bias.type=pa",),
}
STAGES = ("PairwiseAttentionBias", "BatchedPairwiseProjection", "FlashMHA", "backward", "optimizer")


@dataclass
class Measured:
    variant: str
    step_seconds: list[float] = field(default_factory=list)
    peak_memory_gib: float | None = None
    stage_cuda_seconds: dict[str, float] = field(
        default_factory=dict
    )  # per step, from the profiler ranges
    top_ops: list[tuple[str, float]] = field(default_factory=list)  # (op, CUDA seconds per step)
    error: str | None = None

    @property
    def median_step(self) -> float | None:
        return statistics.median(self.step_seconds) if self.step_seconds else None


def config(overrides: tuple[str, ...]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name="foundational", overrides=["dataset=mcfm", *overrides])


def build_model(cfg: DictConfig) -> FoundationModel:
    """`trainer/train.py::setup_model` without the ion-ladder injection (no arm here needs residue masses)."""
    m = cfg.model
    peak_encoder = m.get("peak_encoder", "multiscale")
    return FoundationModel(
        dim_model=m.get("dim_model", 512),
        n_heads=m.get("n_heads", 8),
        dim_feedforward=m.get("dim_feedforward", 2048),
        n_layers=m.get("n_layers", 6),
        dropout=m.get("dropout", 0.1),
        n_peaks=m.get("n_peaks", 200),
        max_mz=m.get("max_mz", 2500.0),
        min_mz=m.get("min_mz", 0.0),
        max_charge=m.get("max_charge", 10),
        peak_encoder_type=peak_encoder.get("type", "multiscale")
        if isinstance(peak_encoder, DictConfig)
        else peak_encoder,
        mz_task=m.get("mz_head", {}).get("task", "regression"),
        use_meta_token=m.get("meta_token", {}).get("enabled", False),
        cfg=m,
    )


def real_batches(
    parquet: Path, cfg: DictConfig, batch_size: int, n_batches: int, seed: int
) -> list[dict[str, Any]]:
    """`n_batches` training batches from the first rows of the shard, through the processor and collate built as
    `trainer/train.py::setup_dataloaders` builds the training processor (same keys, same defaults)."""
    torch.manual_seed(seed)
    m = cfg.model
    masking_cfg = m.get("masking", {})
    proc = FoundationalDataProcessor(
        n_peaks=m.get("n_peaks", 200),
        min_mz=m.get("min_mz", 50.0),
        max_mz=m.get("max_mz", 2500.0),
        min_intensity=m.get("min_intensity", 0.01),
        remove_precursor_tol=m.get("remove_precursor_tol", 2.0),
        use_spectrum_utils=m.get("use_spectrum_utils", False),
        normalize_mz=m.get("normalize_mz", True),
        peak_ordering=masking_cfg.get("ordering_strategy", m.get("peak_ordering", "sorted")),
        return_str=True,
        masking_strategy=masking_cfg.get("strategy", m.get("masking_strategy", "thompson_span")),
        mask_portion=masking_cfg.get("mask_portion", m.get("mask_portion", 0.30)),
        thompson_alpha=masking_cfg.get("alpha", m.get("thompson_alpha", 0.5)),
        thompson_beta=masking_cfg.get("beta", m.get("thompson_beta", 0.5)),
        thompson_kappa=masking_cfg.get("kappa", m.get("thompson_kappa", 4.0)),
        thompson_gamma=masking_cfg.get("gamma", m.get("thompson_gamma", 0.7)),
        span_min=masking_cfg.get("span_min", m.get("span_min", 4)),
        span_max=masking_cfg.get("span_max", m.get("span_max", 7)),
        span_bidirectional=masking_cfg.get("bidirectional", m.get("span_bidirectional", True)),
        include_isotopes=masking_cfg.get("include_isotopes", m.get("include_isotopes", True)),
        isotope_ppm=masking_cfg.get("isotope_ppm", m.get("isotope_ppm", 25.0)),
        isotope_da_floor=masking_cfg.get("isotope_da_floor", m.get("isotope_da_floor", 0.02)),
        isotope_max_charge=masking_cfg.get("isotope_max_charge", m.get("isotope_max_charge", 3)),
        isotope_max_order=masking_cfg.get("isotope_max_order", m.get("isotope_max_order", 2)),
        max_total_mask_ratio=masking_cfg.get(
            "max_total_mask_ratio", m.get("max_total_mask_ratio", 0.40)
        ),
    )
    rows = pl.read_parquet(parquet, n_rows=batch_size * n_batches)
    processed = [proc.process_row(r) for r in rows.iter_rows(named=True)]
    return [
        proc.collate_fn(processed[i : i + batch_size]) for i in range(0, len(processed), batch_size)
    ]


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def instrument(model: torch.nn.Module) -> None:
    """Wrap the forward of every PA stage and attention module in a named profiler range."""
    for module in model.modules():
        name = type(module).__name__
        if name in STAGES[:3]:
            original: Callable[..., Any] = module.forward

            def ranged(
                *args: Any, _f: Callable[..., Any] = original, _n: str = name, **kwargs: Any
            ) -> Any:
                with record_function(_n):
                    return _f(*args, **kwargs)

            module.forward = ranged  # type: ignore[method-assign]


def set_pa_chunk(model: torch.nn.Module, chunk: int) -> None:
    for module in model.modules():
        if hasattr(module, "_CHUNK_SIZE"):
            module._CHUNK_SIZE = chunk


class Stepper:
    """One optimizer step as `trainer/train.py::_shared_step` performs it (classification head with teacher-forced
    offsets, auxiliary losses, fp16 autocast, GradScaler, AdamW), on an already-built model."""

    def __init__(
        self, model: torch.nn.Module, raw: FoundationModel, cfg: DictConfig, device: torch.device
    ) -> None:
        self.model, self.raw, self.cfg, self.device = model, raw, cfg, device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("learning_rate", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.0),
        )
        self.cuda = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cuda)
        self.group_size = int(raw.bin_group_size_tensor.item())
        self.n_groups = int(raw.n_bin_groups_tensor.item())
        self.last_group = int(raw.last_group_size_tensor.item())
        self.bin_edges = getattr(raw, "bin_edges", None)
        self.bin_size = (
            float(raw.bin_size_tensor.item()) if hasattr(raw, "bin_size_tensor") else None
        )
        self.conditioning = cfg.model.get("mz_head", {}).get("offset_conditioning", "none")

    def __call__(self, batch: dict[str, Any]) -> float:
        m = self.cfg.model
        mlm_mask = batch["peak_mask"]
        autocast = (
            torch.autocast("cuda", dtype=torch.float16) if self.cuda else contextlib.nullcontext()
        )
        with autocast:
            target_groups = target_offsets = None
            if self.conditioning != "none":
                target_groups, target_offsets = mz_to_bin_groups(
                    batch["spectra"][:, :, 0] * m.max_mz,
                    self.bin_size,
                    m.max_mz,
                    self.group_size,
                    m.get("min_mz", 0.0),
                    bin_edges=self.bin_edges,
                )
            preds, aux_out = self.model(
                spectra=batch["spectra"],
                spectra_mask=batch.get("spectra_mask"),
                mlm_mask=mlm_mask,
                meta=batch.get("meta"),
                target_groups=target_groups,
                target_offsets=target_offsets,
                bin_edges=self.bin_edges,
            )
            padding = batch.get("spectra_mask")
            valid = mlm_mask & ~padding if padding is not None else mlm_mask
            total, _, _, _ = compute_classification_loss(
                preds,
                batch["spectra"][:, :, 0],
                valid,
                m,
                self.group_size,
                self.n_groups,
                self.last_group,
                bin_edges=self.bin_edges,
            )
            aux_cfg = m.get("auxiliary", {})
            if aux_cfg.get("enabled", False):
                *_, aux = compute_auxiliary_losses(aux_out, batch, aux_cfg, m)
                total = compute_total_loss(total, aux)
        with record_function("backward"):
            self.scaler.scale(total).backward()
        with record_function("optimizer"):
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return float(total.detach())


def measure(
    variant: str, batches: list[dict[str, Any]], args: argparse.Namespace, device: torch.device
) -> Measured:
    out = Measured(variant)
    try:
        cfg = config(VARIANTS[variant])
        raw = build_model(cfg).to(device)
        raw.train()
        instrument(raw)
        if args.pa_chunk:
            set_pa_chunk(raw, args.pa_chunk)
        model: torch.nn.Module = (
            torch.compile(raw, mode="default") if variant.endswith("-compiled") else raw
        )
        stepper = Stepper(model, raw, cfg, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for i in range(args.warmup):
            stepper(batches[i % len(batches)])
        sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
        activities = [ProfilerActivity.CPU] + (
            [ProfilerActivity.CUDA] if device.type == "cuda" else []
        )
        with profile(activities=activities) as prof:
            for i in range(args.steps):
                sync()
                t0 = time.perf_counter()
                stepper(batches[(args.warmup + i) % len(batches)])
                sync()
                out.step_seconds.append(time.perf_counter() - t0)
        if device.type == "cuda":
            out.peak_memory_gib = torch.cuda.max_memory_allocated(device) / 2**30
        key = "device_time_total" if device.type == "cuda" else "cpu_time_total"
        averages = prof.key_averages()
        for row in averages:
            if row.key in STAGES:
                out.stage_cuda_seconds[row.key] = getattr(row, key) / 1e6 / args.steps
        ranked = sorted(
            (r for r in averages if r.key not in STAGES and not r.key.startswith("ProfilerStep")),
            key=lambda r: -getattr(r, key),
        )
        out.top_ops = [(r.key, getattr(r, key) / 1e6 / args.steps) for r in ranked[:12]]
    except (
        Exception
    ) as e:  # a variant that cannot run on this GPU is a result too (the compiled PA path)
        out.error = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def render(results: list[Measured], args: argparse.Namespace, device: torch.device) -> str:
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    head = ["# Profile of one training step: baseline against the pairwise-attention path", ""]
    head += [
        f"Generated by `scripts/analysis/profile_pa.py` on {name}: micro-batch {args.batch} spectra of `{Path(args.parquet).name}`, "
        f"{args.steps} profiled steps after {args.warmup} warm-up steps, fp16 autocast, AdamW, dropout and gradient checkpointing on"
        + (f", PA chunk size {args.pa_chunk}" if args.pa_chunk else "")
        + ". Times are this GPU's; the stage split and the ratios transfer, the absolute values do not (an H100 at batch 2,048: 0.74 s baseline, 3.03 s PA).",
        "",
        "| Variant | Median step (s) | Ratio to baseline | Peak memory (GiB) | "
        + " | ".join(f"`{s}` (s per step) | " for s in STAGES).rstrip(" |")
        + " |",
        "| :---- | :---- | :---- | :---- | " + " | ".join(":----" for _ in STAGES) + " |",
    ]
    base = next((r.median_step for r in results if r.variant == "baseline" and r.median_step), None)
    rows = []
    for r in results:
        if r.error:
            rows.append(
                f"| `{r.variant}` | failed: {r.error} | | | "
                + " | ".join("" for _ in STAGES)
                + " |"
            )
            continue
        med = r.median_step or float("nan")
        ratio = f"{med / base:.2f}" if base else "-"
        mem = f"{r.peak_memory_gib:.2f}" if r.peak_memory_gib is not None else "-"
        stages = " | ".join(
            f"{r.stage_cuda_seconds[s]:.3f}" if s in r.stage_cuda_seconds else "-" for s in STAGES
        )
        rows.append(f"| `{r.variant}` | {med:.3f} | {ratio} | {mem} | {stages} |")
    tops = ["", "## Heaviest operations per variant (device time per step, seconds)", ""]
    for r in results:
        if r.error:
            continue
        tops += [f"### `{r.variant}`", "", "| Operation | s per step |", "| :---- | :---- |"]
        tops += [f"| `{op}` | {sec:.3f} |" for op, sec in r.top_ops]
        tops.append("")
    return "\n".join(head + rows + tops)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--variants", default="baseline,baseline-compiled,pa,pa-mask2mask")
    ap.add_argument(
        "--pa-chunk",
        type=int,
        default=0,
        help="chunk size of the PA modules (0 keeps the code's 64)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "docs" / "arch_study" / "profile_pa.md"))
    args = ap.parse_args()
    device = torch.device(args.device)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    n_batches = min(args.warmup + args.steps, 8)
    batches = [
        to_device(b, device)
        for b in real_batches(Path(args.parquet), config(()), args.batch, n_batches, args.seed)
    ]
    results = []
    for v in variants:
        r = measure(v, batches, args, device)
        results.append(r)
        print(  # noqa: T201
            f"{v}: median step {r.median_step}, peak {r.peak_memory_gib}, stages {r.stage_cuda_seconds}, error {r.error}",
            flush=True,
        )  # noqa: T201
    text = render(results, args, device)
    Path(args.out).write_text(text + "\n")
    print(f"wrote {args.out}")  # noqa: T201


if __name__ == "__main__":
    main()
