"""Tests for BR-DRAG (arXiv 2601.06903, Section IV, eqs. (12)-(16)). CPU only, tiny tensors.

Run with:  PYTHONPATH=src pytest tests/strategies
"""

import math

import pytest
import torch
from torch import nn

from fedrag.strategies.br_drag import (
    BRDRAG,
    Uplink,
    aggregate,
    apply,
    correct_update,
    divergence,
)
from fedrag.strategies.root_trainer import get_params, load_params, root_reference

D = 16
TOL = 1e-5


def _rand(n, gen, scale=1.0):
    return torch.randn(n, generator=gen) * scale


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(0)


# 1. lambda in [0, 2 c_t] -------------------------------------------------------
@pytest.mark.parametrize("c_t", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_lambda_in_range(c_t, gen):
    for _ in range(200):
        g = _rand(D, gen, scale=10 ** torch.empty(1).uniform_(-3, 3, generator=gen).item())
        r = _rand(D, gen)
        lam = divergence(g, r, c_t).item()
        assert 0.0 <= lam <= 2 * c_t + 1e-7
    r = _rand(D, gen)
    assert divergence(-r, r, c_t).item() == pytest.approx(2 * c_t)
    assert divergence(r, r, c_t).item() == pytest.approx(0.0, abs=1e-7)


# 2. g parallel to r -> lambda = 0, v = r -----------------------------------------
@pytest.mark.parametrize("k", [1e-3, 0.5, 3.0, 1e3])
def test_parallel_gives_r(k, gen):
    r = _rand(D, gen)
    g = k * r
    assert divergence(g, r, 0.5).item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(correct_update(g, r, 0.5), r, rtol=TOL, atol=1e-6)


# 3. g anti-parallel, c_t = 0.5 -> lambda = 1, v = r ------------------------------
@pytest.mark.parametrize("k", [1e-3, 0.5, 3.0, 1e3])
def test_antiparallel_gives_r(k, gen):
    r = _rand(D, gen)
    g = -k * r
    assert divergence(g, r, 0.5).item() == pytest.approx(1.0, abs=1e-6)
    torch.testing.assert_close(correct_update(g, r, 0.5), r, rtol=TOL, atol=1e-6)


# 4. c_t = 0 -> v = (||r|| / ||g||) g and ||v|| = ||r|| ----------------------------
def test_ct_zero_is_rescaled_g(gen):
    for _ in range(50):
        g, r = _rand(D, gen, 5.0), _rand(D, gen, 0.1)
        v = correct_update(g, r, 0.0)
        torch.testing.assert_close(v, (r.norm() / g.norm()) * g, rtol=TOL, atol=1e-7)
        assert v.norm().item() == pytest.approx(r.norm().item(), rel=TOL)


# 5. c_t <= 0.5 -> ||v|| <= ||r|| --------------------------------------------------
@pytest.mark.parametrize("c_t", [0.0, 0.1, 0.25, 0.5])
def test_norm_bound(c_t, gen):
    r = _rand(D, gen)
    cases = [_rand(D, gen, scale=10 ** (i % 7 - 3)) for i in range(300)]
    cases += [-r, -1e6 * r, r, -r + 1e-3 * _rand(D, gen)]  # adversarial directions/sizes
    for g in cases:
        v = correct_update(g, r, c_t)
        assert v.norm().item() <= r.norm().item() * (1 + TOL)


# 6. tiny / zero vectors -> no NaN or inf ------------------------------------------
@pytest.mark.parametrize("c_t", [0.1, 0.5])
def test_degenerate_inputs_finite(c_t, gen):
    r = _rand(D, gen)
    tiny = torch.zeros(D)
    tiny[0] = 1e-12  # ||g|| = 1e-12
    zero = torch.zeros(D)
    for g, ref in [(tiny, r), (zero, r), (r, zero), (zero, zero), (tiny, tiny)]:
        lam = divergence(g, ref, c_t)
        v = correct_update(g, ref, c_t)
        assert torch.isfinite(lam).all() and torch.isfinite(v).all()
        assert v.norm().item() <= ref.norm().item() * (1 + TOL) + 1e-6
    # g = 0 -> cos treated as 0 -> lambda = c_t, v = c_t * r
    assert divergence(zero, r, c_t).item() == pytest.approx(c_t)
    torch.testing.assert_close(correct_update(zero, r, c_t), c_t * r)


# 7. root trainer ------------------------------------------------------------------
def _tiny_setup(seed=0):
    torch.manual_seed(seed)  # test-only: deterministic dummy model init
    model = nn.Linear(4, 3)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(64, 4, generator=g)
    y = torch.randint(0, 3, (64,), generator=g)
    return model, (x, y)


def test_param_roundtrip():
    model, _ = _tiny_setup()
    theta = get_params(model)
    assert theta.dtype == torch.float32 and theta.dim() == 1 and theta.numel() == 4 * 3 + 3
    load_params(model, theta * 2)
    torch.testing.assert_close(get_params(model), theta * 2)


def test_root_reference_keeps_theta_and_is_deterministic():
    model, data = _tiny_setup()
    theta = get_params(model) + 0.1  # differs from the model's current params on purpose
    theta_before = theta.clone()

    r1 = root_reference(model, theta, data, steps=5, batch_size=10, lr=0.1,
                        generator=torch.Generator().manual_seed(123))
    assert torch.equal(theta, theta_before)  # caller's theta untouched
    assert r1.shape == theta.shape and r1.dtype == torch.float32
    assert r1.norm().item() > 0
    torch.testing.assert_close(get_params(model), theta + r1)  # model holds theta^(t,U)

    # same seed, model perturbed in between -> identical r (theta is loaded first)
    load_params(model, torch.randn_like(theta))
    r2 = root_reference(model, theta, data, steps=5, batch_size=10, lr=0.1,
                        generator=torch.Generator().manual_seed(123))
    assert torch.equal(r1, r2)

    r3 = root_reference(model, theta, data, steps=5, batch_size=10, lr=0.1,
                        generator=torch.Generator().manual_seed(999))
    assert not torch.equal(r1, r3)


def test_root_reference_is_plain_sgd():
    """One full-batch step must equal theta - lr * grad (no momentum, no weight decay)."""
    model, (x, y) = _tiny_setup()
    theta = get_params(model)
    lr = 0.05
    loss = nn.functional.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    expected = -lr * torch.cat([gr.reshape(-1) for gr in grads])
    r = root_reference(model, theta, (x, y), steps=1, batch_size=len(x), lr=lr,
                       generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(r, expected, rtol=1e-5, atol=1e-7)


# 8. full round ----------------------------------------------------------------------
def test_full_round(gen):
    model, data = _tiny_setup()
    strat = BRDRAG(model, data, torch.Generator().manual_seed(7), c_t=0.5, steps=5,
                   batch_size=10, lr=0.1)
    theta = get_params(model)
    d = theta.numel()

    assert strat.name == "br_drag" and strat.needs_warmup(0) is False
    dl = strat.prepare_downlink(0, theta)
    assert dl.round == 0 and dl.reference is not None and dl.reference.shape == (d,)

    g_raw = _rand(d, gen)
    out_g, kind, meta = strat.client_postprocess(0, g_raw, dl)
    assert out_g is g_raw and kind == "raw" and meta == {}

    uplinks = [Uplink(0, m, _rand(d, gen, scale=10 ** (m - 2)), "raw", 100) for m in range(5)]
    uplinks.append(Uplink(0, 5, torch.zeros(d), "raw", 100))  # degenerate client
    vecs, diag = strat.server_transform(0, uplinks, dl)
    assert len(vecs) == len(uplinks)
    r_norm = dl.reference.norm().item()
    for v in vecs:
        assert v.shape == (d,) and torch.isfinite(v).all()
        assert v.norm().item() <= r_norm * (1 + TOL)  # c_t = 0.5 bound

    for key in ["lambda_mean", "lambda_min", "lambda_max", "cos_g_r_mean", "r_norm",
                "g_norm_mean", "rho_min", "rho_max",
                "time_prepare_downlink_s", "time_server_transform_s"]:
        assert key in diag and math.isfinite(diag[key]), key
    assert 0.0 <= diag["lambda_min"] <= diag["lambda_mean"] <= diag["lambda_max"] <= 1.0
    assert diag["r_norm"] == pytest.approx(r_norm)
    assert diag["time_prepare_downlink_s"] >= 0 and diag["time_server_transform_s"] >= 0

    delta = strat.aggregate(0, vecs)
    torch.testing.assert_close(delta, torch.stack(vecs).mean(0))
    theta1 = strat.apply(0, theta, delta)
    assert theta1.shape == theta.shape and theta1.dtype == torch.float32
    assert torch.isfinite(theta1).all()
    torch.testing.assert_close(theta1, theta + delta)

    assert strat.state_dict() == {}
    strat.load_state_dict({})
    with pytest.raises(ValueError):
        strat.load_state_dict({"r": torch.zeros(1)})


def test_module_aggregate_apply():
    vs = [torch.ones(3), 3 * torch.ones(3)]
    torch.testing.assert_close(aggregate(vs), 2 * torch.ones(3))
    torch.testing.assert_close(apply(torch.ones(3), torch.ones(3)), 2 * torch.ones(3))
    with pytest.raises(ValueError):
        aggregate([])
