"""What the deployed masking objective masks, measured on real spectra with the repository's own processor.

Analysis only: reads a corpus-schema parquet, runs `FoundationalDataProcessor` with the paper configuration's
masking settings (`configs/model/foundation_base.yaml`), and prints the realised mask fraction, the share added by
isotope co-masking, the distribution of contiguous masked runs, the position of the most intense peak in a run, and
the residual isotope leakage. Numbers feed `docs/agent_logs/session-2026-10-15-exploringArch.md`.

    python scripts/analysis/masking_stats.py --parquet <mcfm-test shard> [--n 2000] [--seed 0]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from jaxtyping import Bool, Float, Int

from instanovo_fm.data import FoundationalDataProcessor
from instanovo_fm.data.masking import thompson_sampling_span_mask

# The masking block of configs/model/foundation_base.yaml, as the trainer hands it to the processor.
PAPER_MASKING = {
    "masking_strategy": "thompson_span",
    "mask_portion": 0.26,
    "thompson_alpha": 0.5,
    "thompson_beta": 0.5,
    "thompson_kappa": 8.0,
    "thompson_gamma": 1.2,
    "span_min": 3,
    "span_max": 4,
    "span_bidirectional": False,
    "include_isotopes": True,
    "isotope_ppm": 20.0,
    "isotope_da_floor": 0.015,
    "isotope_max_charge": 3,
    "isotope_max_order": 3,
    "max_total_mask_ratio": 1.0,
}
PAPER_PROCESSING = {
    "n_peaks": 200,
    "min_mz": 50.0,
    "max_mz": 2500.0,
    "min_intensity": 0.01,
    "remove_precursor_tol": 0.0,
    "use_spectrum_utils": False,
    "normalize_mz": True,
}
ISOTOPE_DA = 1.003355


@dataclass(frozen=True)
class Runs:
    """Contiguous masked runs over m/z-sorted peaks."""

    lengths: Int[np.ndarray, " runs"]
    leftmost_is_most_intense: Bool[np.ndarray, " runs"]


def runs_of(
    mask: Bool[torch.Tensor, "batch peaks"],
    valid: Bool[torch.Tensor, "batch peaks"],
    intensity: Float[torch.Tensor, "batch peaks"],
) -> Runs:
    lengths: list[int] = []
    leftmost: list[bool] = []
    for m_row, v_row, i_row in zip(
        (mask & valid).numpy(), valid.numpy(), intensity.numpy(), strict=True
    ):
        edges = np.flatnonzero(
            np.diff(np.concatenate(([False], m_row & v_row, [False])).astype(np.int8))
        )
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            lengths.append(int(stop - start))
            leftmost.append(bool(i_row[start] >= i_row[start:stop].max()))
    return Runs(np.array(lengths), np.array(leftmost))


def visible_isotope_share(
    mz_da: Float[torch.Tensor, "batch peaks"],
    mask: Bool[torch.Tensor, "batch peaks"],
    valid: Bool[torch.Tensor, "batch peaks"],
    charges: Int[torch.Tensor, " batch"],
) -> float:
    """Masked peaks whose +1 isotope, at any fragment charge up to the precursor's (at least 1, at most 3), is still visible."""
    leaked = total = 0
    for x, m, v, z in zip(
        mz_da.numpy(), (mask & valid).numpy(), valid.numpy(), charges.numpy(), strict=True
    ):
        visible = x[v & ~m]
        for i in np.flatnonzero(m):
            total += 1
            tol = max(20e-6 * x[i], 0.015)
            if any(
                np.any(np.abs(visible - x[i] - ISOTOPE_DA / zz) <= tol)
                for zz in range(1, min(max(int(z), 1), 3) + 1)
            ):
                leaked += 1
    return leaked / total


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rows = pl.read_parquet(args.parquet, n_rows=args.n)
    proc = FoundationalDataProcessor(**PAPER_PROCESSING, **PAPER_MASKING)
    batch = proc.collate_fn([proc.process_row(r) for r in rows.iter_rows(named=True)])
    spectra, pad, mask, charges = (
        batch["spectra"],
        batch["spectra_mask"],
        batch["peak_mask"],
        batch["precursor_charge"],
    )
    valid = ~pad
    mz, intensity = spectra[..., 0], spectra[..., 1]
    n_valid = valid.sum(1).float()
    print(f"batch keys: {sorted(batch)}")  # noqa: T201
    print(  # noqa: T201
        f"peaks per spectrum after processing: mean {n_valid.mean():.1f}, median {n_valid.median():.0f}, at the {PAPER_PROCESSING['n_peaks']} cap {(n_valid == PAPER_PROCESSING['n_peaks']).float().mean() * 100:.1f} %"
    )
    kept = intensity[valid]
    print(  # noqa: T201
        f"intensity channel (sqrt, L2 per spectrum): median {kept.median():.3f}, p99 {kept.quantile(0.99):.3f}, max {kept.max():.3f}; per-spectrum maximum, median {intensity.max(1).values.median():.3f}"
    )
    frac = (mask & valid).sum(1).float() / n_valid
    print(  # noqa: T201
        f"masked fraction per spectrum: mean {frac.mean() * 100:.1f} %, p10 {frac.quantile(0.1) * 100:.1f} %, p90 {frac.quantile(0.9) * 100:.1f} %"
    )

    span_kwargs = {
        k: v
        for k, v in PAPER_MASKING.items()
        if k
        not in (
            "masking_strategy",
            "thompson_alpha",
            "thompson_beta",
            "thompson_kappa",
            "thompson_gamma",
            "span_bidirectional",
        )
    }
    span_kwargs.update(
        alpha=0.5,
        beta=0.5,
        kappa=8.0,
        gamma=1.2,
        bidirectional=False,
        max_mz=PAPER_PROCESSING["max_mz"],
        normalize_mz=True,
    )
    torch.manual_seed(args.seed)
    spans_only = thompson_sampling_span_mask(
        intensity, pad, mz=mz, charges=None, **{**span_kwargs, "include_isotopes": False}
    )
    torch.manual_seed(args.seed)
    with_isotopes = thompson_sampling_span_mask(
        intensity, pad, mz=mz, charges=charges, **span_kwargs
    )
    f_spans = ((spans_only & valid).sum(1).float() / n_valid).mean()
    f_iso = ((with_isotopes & valid).sum(1).float() / n_valid).mean()
    iso_share = (with_isotopes & ~spans_only & valid).sum().item() / (
        with_isotopes & valid
    ).sum().item()
    print(  # noqa: T201
        f"spans alone mask {f_spans * 100:.1f} % of peaks; isotope co-masking brings it to {f_iso * 100:.1f} % ({iso_share * 100:.1f} % of masked peaks are co-masked isotopes)"
    )

    runs = runs_of(spans_only, valid, intensity)
    hist = dict(zip(*np.unique(runs.lengths, return_counts=True), strict=True))
    long = runs.lengths[runs.lengths > PAPER_MASKING["span_max"]].sum() / runs.lengths.sum()
    print(  # noqa: T201
        f"contiguous masked runs before isotopes: {len(runs.lengths)} ({len(runs.lengths) / mask.shape[0]:.1f} per spectrum), mean length {runs.lengths.mean():.2f}, "
        f"{long * 100:.0f} % of span-masked peaks lie in runs longer than span_max={PAPER_MASKING['span_max']}; length histogram {hist}"
    )
    print(  # noqa: T201
        f"leftmost peak of a run is its most intense: {runs.leftmost_is_most_intense.mean() * 100:.1f} % (about 29 % by chance for runs of 3 to 4)"
    )
    print(  # noqa: T201
        f"masked peaks whose +1 isotope is still visible after co-masking: {visible_isotope_share(mz * PAPER_PROCESSING['max_mz'], with_isotopes, valid, charges) * 100:.1f} %"
    )
    print(  # noqa: T201
        f"precursor charges in the sample: {dict(zip(*np.unique(charges.numpy(), return_counts=True), strict=True))}"
    )


if __name__ == "__main__":
    main()
