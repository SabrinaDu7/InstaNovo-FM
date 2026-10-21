"""MuonAdamW: the orthogonalisation, the grouping rule, the AdamW branch against torch's, and the state dict."""

import copy

import pytest
import torch
from torch import nn

from instanovo_fm.common.muon import MuonAdamW, muon_param_groups, newton_schulz


def _reference_newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """modded-nanogpt's `zeropower_via_newtonschulz5`, one matrix at a time: the behaviour `newton_schulz` promises."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x /= x.norm() + eps
    if g.size(0) > g.size(1):
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.T if g.size(0) > g.size(1) else x).float()


def test_newton_schulz_equals_the_reference_and_orthogonalises_the_captured_directions() -> None:
    """Five quintic steps bring every singular value that starts above about 3 % of the largest into [0.6, 1.3];
    smaller ones are left small (the 0.003 floor of a 768-square Gaussian is the reference's too), so the test
    checks the captured range and bit-equality with the reference, batched blocks included."""
    torch.manual_seed(0)
    for shape in ((64, 32), (32, 64), (48, 48), (2304, 768)):
        g = torch.randn(*shape)
        out = newton_schulz(g)
        torch.testing.assert_close(out, _reference_newton_schulz(g), rtol=0, atol=0)
        s_in = torch.linalg.svdvals(g)
        s_out = torch.linalg.svdvals(out)
        captured = s_out[s_in > 0.03 * s_in.max()]
        assert captured.min() > 0.6 and s_out.max() < 1.3, (shape, captured.min().item(), s_out.max().item())
    g = torch.randn(2304, 768)
    blocks = g.view(3, 768, 768)
    torch.testing.assert_close(newton_schulz(blocks), torch.stack([_reference_newton_schulz(b) for b in blocks]), rtol=0, atol=0)


class _Toy(nn.Module):
    """Names shaped like the FoundationModel's: a fused qkv and three matrices inside `encoder.layers`, the rest outside."""

    def __init__(self) -> None:
        super().__init__()
        self.peak_encoder = nn.Linear(8, 16)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Module()])
        layer = self.encoder.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.qkv = nn.Linear(16, 48, bias=False)
        layer.self_attn.o = nn.Linear(16, 16)
        layer.linear1 = nn.Linear(16, 24)
        layer.norm1 = nn.LayerNorm(16)
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer = self.encoder.layers[0]
        h = self.peak_encoder(x)
        q, k, v = layer.self_attn.qkv(h).chunk(3, dim=-1)
        h = layer.norm1(h + layer.self_attn.o(q * k + v))
        return self.head(h) + layer.linear1(h).sum(-1, keepdim=True)


def test_grouping_rule() -> None:
    model = _Toy()
    matrices, qkv, rest = muon_param_groups(model, lr=1e-3, weight_decay=0.0, momentum=0.95)
    names = {id(p): n for n, p in model.named_parameters()}
    assert sorted(names[id(p)] for p in matrices["params"]) == ["encoder.layers.0.linear1.weight", "encoder.layers.0.self_attn.o.weight"]
    assert [names[id(p)] for p in qkv["params"]] == ["encoder.layers.0.self_attn.qkv.weight"] and qkv["blocks"] == 3
    assert all(p.ndim == 1 or "encoder.layers" not in names[id(p)] for p in rest["params"])
    assert sum(len(g["params"]) for g in (matrices, qkv, rest)) == len(list(model.parameters()))


def test_adamw_branch_matches_torch_adamw() -> None:
    torch.manual_seed(1)
    ours, ref = nn.Linear(6, 5), nn.Linear(6, 5)
    ref.load_state_dict(ours.state_dict())
    opt = MuonAdamW([dict(params=list(ours.parameters()), algo="adamw", lr=1e-2, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8)])
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-2, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    for _ in range(25):
        x = torch.randn(7, 6)
        for m, o in ((ours, opt), (ref, opt_ref)):
            o.zero_grad()
            m(x).square().mean().backward()
            o.step()
    for a, b in zip(ours.parameters(), ref.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


def test_muon_updates_only_through_orthogonal_directions_and_decays_loss() -> None:
    torch.manual_seed(2)
    model = _Toy()
    opt = MuonAdamW(muon_param_groups(model, lr=3e-3, weight_decay=0.0, momentum=0.95))
    x, y = torch.randn(64, 8), torch.randn(64, 4)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = (model(x) - y).square().mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], losses[::10]
    # the qkv weight moved as three blocks: within each block, every singular direction of the (Nesterov) gradient
    # that the Newton-Schulz iteration captures (above about 3 % of the largest) moves by the same amount; the
    # iteration keeps the singular order, so the captured directions are the first ones of the update's spectrum
    qkv = model.encoder.layers[0].self_attn.qkv.weight
    opt.zero_grad()
    (model(x) - y).square().mean().backward()
    mu = opt.param_groups[1]["momentum"]
    nesterov_grad = qkv.grad.lerp(opt.state[qkv]["momentum_buffer"].lerp(qkv.grad, 1 - mu), mu)
    before = qkv.detach().clone()
    opt.step()
    delta = (qkv.detach() - before).view(3, 16, 16)
    for s_in, s_out in zip(torch.linalg.svdvals(nesterov_grad.view(3, 16, 16)), torch.linalg.svdvals(delta), strict=True):
        captured = s_out[: int((s_in > 0.03 * s_in.max()).sum())]
        assert captured.max() / captured.min() < 2.5, (s_in, s_out)


def test_state_dict_roundtrip_resumes_identically() -> None:
    torch.manual_seed(3)
    model = _Toy()
    opt = MuonAdamW(muon_param_groups(model, lr=1e-3, weight_decay=0.01, momentum=0.9))
    x, y = torch.randn(16, 8), torch.randn(16, 4)

    def one_step(m: nn.Module, o: MuonAdamW) -> None:
        o.zero_grad()
        (m(x) - y).square().mean().backward()
        o.step()

    one_step(model, opt)
    model2 = copy.deepcopy(model)
    opt2 = MuonAdamW(muon_param_groups(model2, lr=1e-3, weight_decay=0.01, momentum=0.9))
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    one_step(model, opt)
    one_step(model2, opt2)
    for a, b in zip(model.parameters(), model2.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_rejects_untagged_group() -> None:
    with pytest.raises(ValueError, match="algo"):
        MuonAdamW([dict(params=[nn.Parameter(torch.zeros(2, 2))], lr=1e-3)])
