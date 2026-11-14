"""Tests for FLTrust (base paper arXiv 2601.06903 p. 9; Cao et al. NDSS 2021). CPU, tiny tensors.

Run with:  PYTHONPATH=src pytest tests/strategies
"""

import math

import pytest
import torch
from torch import nn

from fedrag.attacks.sign_flip import SignFlip
from fedrag.strategies.br_drag import BRDRAG, Downlink, Uplink
from fedrag.strategies.fltrust import FLTrust, modify_paper, rescale_cao, trust_score
from fedrag.strategies.root_trainer import get_params

D = 12
TOL = 1e-5


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _tiny_setup(seed=0):
    torch.manual_seed(seed)  # test-only: deterministic dummy model init
    model = nn.Linear(4, 3)
    g = _gen(seed)
    return model, (torch.randn(64, 4, generator=g), torch.randint(0, 3, (64,), generator=g))


def _strategy(variant, r):
    """FLTrust whose server side is driven by a fixed r (no root training needed)."""
    model, data = _tiny_setup()
    return FLTrust(model, data, _gen(0), variant=variant), Downlink(0, torch.zeros_like(r), r)


def _ups(vectors):
    return [Uplink(0, m, v, "raw", 10) for m, v in enumerate(vectors)]


def _round(strat, dl, vectors):
    vecs, diag = strat.server_transform(0, _ups(vectors), dl)
    return strat.aggregate(0, vecs), vecs, diag


def _orth(r, gen):
    x = torch.randn(D, generator=gen)
    return x - (x @ r) / (r @ r) * r


# 1. g parallel to r -------------------------------------------------------------------
@pytest.mark.parametrize("k", [1e-3, 0.7, 5.0])
def test_parallel_full_trust(k):
    r = torch.randn(D, generator=_gen(1))
    g = k * r
    assert trust_score(g, r).item() == pytest.approx(1.0, abs=1e-6)
    g_hat, _ = rescale_cao(g, r)  # Cao: g^ = r
    torch.testing.assert_close(g_hat, r, rtol=TOL, atol=1e-6)
    g_tilde, _ = modify_paper(g, r)  # paper: g~ = ||g|| r / ||r|| = k r
    torch.testing.assert_close(g_tilde, g, rtol=TOL, atol=1e-6)
    torch.testing.assert_close(modify_paper(r, r)[0], r, rtol=TOL, atol=1e-6)


# 2. anti-parallel / orthogonal -> trust 0, no contribution ------------------------------
@pytest.mark.parametrize("variant", ["paper", "cao"])
def test_untrusted_uploads_have_no_effect(variant):
    gen = _gen(2)
    r = torch.randn(D, generator=gen)
    honest = [0.5 * r + 0.1 * torch.randn(D, generator=gen) for _ in range(3)]
    bad = [-3.0 * r, _orth(r, gen) * 100.0]
    for b in bad:
        assert trust_score(b, r).item() == pytest.approx(0.0, abs=1e-6)

    strat, dl = _strategy(variant, r)
    d_honest, _, _ = _round(strat, dl, honest)
    d_all, vecs, diag = _round(strat, dl, honest + bad)
    assert diag["num_zero_trust"] == 2
    for v in vecs[3:]:
        assert v.abs().max().item() < 1e-5  # zero contribution vector
    if variant == "cao":
        torch.testing.assert_close(d_all, d_honest, rtol=TOL, atol=1e-6)  # delta unchanged
    else:
        # eq. (4) divides by S, so untrusted uploads add 0 to the sum but still count in S
        torch.testing.assert_close(d_all, d_honest * 3 / 5, rtol=TOL, atol=1e-6)


# 3. norms ------------------------------------------------------------------------------
def test_rescaled_norms():
    gen = _gen(3)
    r = torch.randn(D, generator=gen) * 0.2
    for _ in range(100):
        g = torch.randn(D, generator=gen) * 10 ** torch.empty(1).uniform_(-3, 3, generator=gen).item()
        g_hat, trust = rescale_cao(g, r)
        assert g_hat.norm().item() == pytest.approx(r.norm().item(), rel=TOL)  # Cao: ||g^|| = ||r||
        g_tilde, _ = modify_paper(g, r)
        assert g_tilde.norm().item() == pytest.approx(trust.item() * g.norm().item(), rel=TOL, abs=1e-9)
        if trust.item() > 0:  # paper: g~ always points along r
            assert torch.nn.functional.cosine_similarity(g_tilde, r, dim=0).item() == pytest.approx(1.0, abs=1e-5)


# 4. everything points away from r -> delta = 0 -------------------------------------------
@pytest.mark.parametrize("variant", ["paper", "cao"])
def test_all_untrusted_gives_zero_delta(variant):
    gen = _gen(4)
    r = torch.randn(D, generator=gen)
    away = [-(k + 1.0) * r + 0.01 * _orth(r, gen) for k in range(4)] + [torch.zeros(D)]
    strat, dl = _strategy(variant, r)
    delta, vecs, diag = _round(strat, dl, away)
    assert torch.isfinite(delta).all() and torch.all(delta == 0)
    assert all(torch.isfinite(v).all() for v in vecs)
    assert diag["all_zero_trust"] == 1.0 and diag["num_zero_trust"] == 5
    assert diag["trust_max"] == 0.0
    theta = torch.randn(D, generator=gen)
    assert torch.equal(strat.apply(0, theta, delta), theta)


def test_degenerate_inputs_finite():
    zero, tiny = torch.zeros(D), torch.zeros(D)
    tiny[0] = 1e-12
    r = torch.randn(D, generator=_gen(5))
    for g, ref in [(zero, r), (tiny, r), (r, zero), (zero, zero)]:
        for fn in (modify_paper, rescale_cao):
            v, trust = fn(g, ref)
            assert torch.isfinite(v).all() and torch.isfinite(trust)


# 5. shared root trainer --------------------------------------------------------------------
def test_same_r_as_br_drag():
    model, data = _tiny_setup()
    theta = get_params(model) + 0.05
    kw = dict(steps=5, batch_size=10, lr=0.1)
    r_br = BRDRAG(model, data, _gen(11), **kw).prepare_downlink(3, theta).reference
    r_ft = FLTrust(model, data, _gen(11), **kw).prepare_downlink(3, theta).reference
    assert torch.equal(r_br, r_ft) and r_ft.norm().item() > 0


# 6. sign flipping attacker gets trust 0 ----------------------------------------------------
@pytest.mark.parametrize("variant", ["paper", "cao"])
def test_sign_flip_attacker_untrusted(variant):
    gen = _gen(6)
    r = torch.randn(D, generator=gen)
    honest = [r + 0.3 * torch.randn(D, generator=gen) for _ in range(4)]
    uplinks = _ups(honest)
    own_g = {3: honest[3].clone()}  # client 3 turns malicious: uploads -g of its honest update
    attacked = SignFlip().corrupt(0, uplinks, {3}, own_g)
    assert trust_score(honest[3], r).item() > 0
    assert trust_score(attacked[3].vector, r).item() == 0.0

    strat, dl = _strategy(variant, r)
    vecs, diag = strat.server_transform(0, attacked, dl)
    assert diag["num_zero_trust"] == 1 and torch.all(vecs[3] == 0)


# 7. full round -----------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["paper", "cao"])
def test_full_round(variant):
    model, data = _tiny_setup()
    strat = FLTrust(model, data, _gen(7), variant=variant, steps=5, batch_size=10, lr=0.1)
    theta = get_params(model)
    d = theta.numel()
    gen = _gen(8)

    assert strat.name == "fltrust" and strat.needs_warmup(0) is False
    dl = strat.prepare_downlink(0, theta)
    r = dl.reference
    assert dl.round == 0 and r.shape == (d,)

    g_raw = torch.randn(d, generator=gen)
    out_g, kind, meta = strat.client_postprocess(0, g_raw, dl)
    assert out_g is g_raw and kind == "raw" and meta == {}

    vectors = [r * (m + 1) + 0.01 * torch.randn(d, generator=gen) for m in range(4)]
    vectors += [-r, torch.zeros(d)]  # one attacker, one degenerate client
    vecs, diag = strat.server_transform(0, _ups(vectors), dl)
    assert len(vecs) == len(vectors) and all(torch.isfinite(v).all() for v in vecs)

    for key in ["trust_mean", "trust_min", "trust_max", "num_zero_trust", "all_zero_trust",
                "r_norm", "g_norm_mean", "time_prepare_downlink_s", "time_server_transform_s"]:
        assert key in diag and math.isfinite(diag[key]), key
    assert diag["num_zero_trust"] == 2 and diag["all_zero_trust"] == 0.0
    assert 0.0 == diag["trust_min"] <= diag["trust_mean"] <= diag["trust_max"] <= 1.0

    delta = strat.aggregate(0, vecs)
    trusts = torch.stack([trust_score(v, r) for v in vectors])
    if variant == "cao":
        g_hats = torch.stack([rescale_cao(v, r)[0] for v in vectors])
        expected = (trusts[:, None] * g_hats).sum(0) / trusts.sum()
    else:
        expected = torch.stack([modify_paper(v, r)[0] for v in vectors]).mean(0)
    torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-7)

    theta1 = strat.apply(0, theta, delta)
    assert theta1.shape == theta.shape and theta1.dtype == torch.float32
    torch.testing.assert_close(theta1, theta + delta)

    assert strat.state_dict() == {}
    strat.load_state_dict({})
    with pytest.raises(ValueError):
        strat.load_state_dict({"x": 1})
    with pytest.raises(ValueError):
        FLTrust(model, data, _gen(0), variant="other")
