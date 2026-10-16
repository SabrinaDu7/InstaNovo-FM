"""Which attention kernel does the pairwise-attention (PA) path get? Forces each fused SDPA backend in turn on one
forward of the PA model and records the reason PyTorch gives for refusing it, after logging the attn_mask the model
hands to SDPA (shape, dtype, strides, requires_grad, contiguity). Finding of 16/10/2026 on the RTX 4060 (PyTorch's
dispatcher, so the same on an H100): the bias arrives as (B, 12, 201, 201) fp16 with strides (484812, 1, 2412, 12),
heads innermost, and "All fused kernels require the last dimension of the input to have stride 1", so attention runs
the math implementation; the baseline's (B, 1, 1, L) padding mask is accepted by the memory-efficient kernel.

    python scripts/analysis/sdpa_backend_check.py <mcfm-test shard>
"""

import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_pa import VARIANTS, build_model, config, real_batches, to_device  # noqa: E402

device = torch.device("cuda")
cfg = config(VARIANTS["pa"])
model = build_model(cfg).to(device).train()
batch = to_device(real_batches(Path(sys.argv[1]), cfg, 4, 1, 0)[0], device)

seen = []
original = F.scaled_dot_product_attention


def logging_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):  # noqa: ANN001, ANN201
    if len(seen) < 2:
        m = attn_mask
        seen.append(
            {
                "q": (tuple(q.shape), str(q.dtype), q.stride()),
                "mask": None
                if m is None
                else (tuple(m.shape), str(m.dtype), m.stride(), m.requires_grad, m.is_contiguous()),
                "dropout_p": dropout_p,
            }
        )
    return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)


F.scaled_dot_product_attention = logging_sdpa
with torch.autocast("cuda", dtype=torch.float16):
    model(
        spectra=batch["spectra"],
        spectra_mask=batch.get("spectra_mask"),
        mlm_mask=batch["peak_mask"],
        meta=batch.get("meta"),
    )
F.scaled_dot_product_attention = original
for s in seen:
    print("SDPA call:", s)  # noqa: T201

# The same forward under the math implementation and under the memory-efficient kernel, eval mode: the two must agree
# to fp16 kernel precision, so a fix that changes which kernel runs changes no result beyond that.
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    outs = {}
    # the forward samples the 10 Da hint noise (and dropout in train mode) from torch's RNG: reseed before each
    # forward so the kernels see the same inputs; "math2" is the control (same kernel, same seed: must be identical)
    for name, backend in (
        ("math", SDPBackend.MATH),
        ("math2", SDPBackend.MATH),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
    ):
        try:
            torch.manual_seed(0)
            with sdpa_kernel([backend]):
                preds, _ = model(
                    spectra=batch["spectra"],
                    spectra_mask=batch.get("spectra_mask"),
                    mlm_mask=batch["peak_mask"],
                    meta=batch.get("meta"),
                )
            outs[name] = preds[0].float()
        except Exception as e:  # the kernel refused
            print(f"{name}: refused: {str(e).splitlines()[0][:120]}")  # noqa: T201
    if len(outs) == 3:
        for other in ("math2", "efficient"):
            d = (outs["math"] - outs[other]).abs()
            print(  # noqa: T201
                f"group logits, math against {other}: max |diff| {d.max().item():.3e}, mean |diff| {d.mean().item():.3e}, "
                f"logits scale {outs['math'].abs().mean().item():.3e}"
            )
model.train()

for backend in (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with sdpa_kernel([backend]), torch.autocast("cuda", dtype=torch.float16):
                model(
                    spectra=batch["spectra"],
                    spectra_mask=batch.get("spectra_mask"),
                    mlm_mask=batch["peak_mask"],
                    meta=batch.get("meta"),
                )
            outcome = "ran"
        except Exception as e:  # the refusal and its reason
            outcome = f"refused: {str(e).splitlines()[0][:160]}"
    reasons = sorted({str(w.message).splitlines()[0][:200] for w in caught})
    print(f"{backend.name}: {outcome}")  # noqa: T201
    for r in reasons:
        print(f"   warning: {r}")  # noqa: T201
