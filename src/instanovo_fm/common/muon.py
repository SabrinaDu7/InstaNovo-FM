"""Muon for the encoder's matrices, AdamW for everything else, as one optimizer.

One `torch.optim.Optimizer` rather than two, so accelerate, the GradScaler, the scheduler and the checkpoint see what
they always saw. Each parameter group carries `algo`: ``"muon"`` or ``"adamw"``.

Muon (Jordan et al. 2024): momentum on each 2-D weight's gradient, orthogonalised by five Newton-Schulz iterations,
so every singular direction of the update moves at the same speed. The update is scaled by ``0.2 * sqrt(max(rows,
cols))`` so its root-mean-square matches AdamW's (Liu et al. 2025, Moonlight), which is what lets the AdamW learning
rate, weight decay and schedule carry over unchanged. The AdamW branch is torch's single-tensor AdamW, written out
so the two algorithms share one state dict.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from jaxtyping import Float
from torch import Tensor, nn

NEWTON_SCHULZ_COEFFICIENTS = (3.4445, -4.7750, 2.0315)  # the quintic of modded-nanogpt: fast, lands singular values in about [0.7, 1.2]


def newton_schulz(g: Float[Tensor, "... m n"], *, steps: int = 5) -> Float[Tensor, "... m n"]:
    """The nearest orthogonal matrix to each matrix of `g`, to Newton-Schulz precision, in bfloat16."""
    a, b, c = NEWTON_SCHULZ_COEFFICIENTS
    x = g.to(torch.bfloat16)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.mT if transposed else x).to(g.dtype)


def muon_param_groups(model: nn.Module, *, lr: float, weight_decay: float, momentum: float, muon_lr: float | None = None) -> list[dict]:
    """Muon on the 2-D weights inside the encoder layers, the fused qkv weight as three square blocks; AdamW on the rest.

    `muon_lr` is the Muon groups' base learning rate (default `lr`, Moonlight's matched-RMS convention: the same rate serves
    both); the scheduler scales every group's base rate by one factor. On the 40M that is 36 matrices and 88.8 % of the
    parameters under Muon (notes-speed.md, S15).
    """
    matrices: list[nn.Parameter] = []
    qkv: list[nn.Parameter] = []
    rest: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "encoder.layers" in name:
            (qkv if name.endswith("qkv.weight") else matrices).append(p)
        else:
            rest.append(p)
    for p in qkv:
        if p.size(0) % 3:
            raise ValueError(f"a fused qkv weight must stack three blocks along dim 0, got {tuple(p.shape)}")
    muon = dict(algo="muon", lr=lr if muon_lr is None else muon_lr, weight_decay=weight_decay, momentum=momentum, nesterov=True, ns_steps=5)
    adamw = dict(algo="adamw", lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    return [
        dict(params=matrices, blocks=1, **muon),
        dict(params=qkv, blocks=3, **muon),
        dict(params=rest, **adamw),
    ]


class MuonAdamW(torch.optim.Optimizer):
    """Muon on the groups tagged ``algo="muon"``, AdamW on those tagged ``algo="adamw"``; see the module docstring."""

    def __init__(self, params: Iterable[dict]) -> None:
        groups = list(params)
        if any(g.get("algo") not in ("muon", "adamw") for g in groups):
            raise ValueError("every parameter group needs algo='muon' or algo='adamw'")
        super().__init__(groups, defaults={})

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        """One update of every group."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            (self._muon_step if group["algo"] == "muon" else self._adamw_step)(group)
        return loss

    def _muon_step(self, group: dict) -> None:
        lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(p))
            buf.lerp_(p.grad, 1 - mu)
            g = p.grad.lerp(buf, mu) if group["nesterov"] else buf
            blocks = group["blocks"]
            rows, cols = p.size(0) // blocks, p.size(1)
            update = newton_schulz(g.view(blocks, rows, cols), steps=group["ns_steps"]).view_as(p)
            p.mul_(1 - lr * wd)
            p.add_(update, alpha=-lr * 0.2 * math.sqrt(max(rows, cols)))

    def _adamw_step(self, group: dict) -> None:
        lr, wd, (b1, b2), eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            state["exp_avg"].lerp_(p.grad, 1 - b1)
            state["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            p.mul_(1 - lr * wd)
            denom = (state["exp_avg_sq"].sqrt() / math.sqrt(1 - b2**t)).add_(eps)
            p.addcdiv_(state["exp_avg"], denom, value=-lr / (1 - b1**t))
