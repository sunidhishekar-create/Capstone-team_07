"""Tests for the five Byzantine attacks (arXiv 2601.06903, pp. 1-2 and 9-10). CPU, tiny tensors.

Run with:  PYTHONPATH=src pytest tests/attacks
"""

import math

import pytest
import torch

from fedrag.attacks.base import Attack
from fedrag.attacks.label_flip import LabelFlip, flip_labels
from fedrag.attacks.min_max import MinMax, bisect_max_gamma
from fedrag.attacks.min_sum import MinSum
from fedrag.attacks.noise import NoiseInjection
from fedrag.attacks.sign_flip import SignFlip
from fedrag.strategies.br_drag import Uplink

D = 8


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def make_round(n_benign=5, n_mal=3, d=D, seed=0, t=0):
    """Benign ids 0..n_benign-1, malicious ids after. Malicious uplinks carry their own raw g."""
    g = _gen(seed)
    uplinks, own_g = [], {}
    for cid in range(n_benign + n_mal):
        vec = torch.randn(d, generator=g) + 0.5  # shared bias so the benign mean is non-zero
        uplinks.append(Uplink(t, cid, vec, "raw", 100))
        if cid >= n_benign:
            own_g[cid] = vec.clone()
    mal = set(range(n_benign, n_benign + n_mal))
    return uplinks, mal, own_g


def snapshot(uplinks):
    return [(u.client_id, u.vector.clone()) for u in uplinks]


def assert_benign_untouched(out, inp, before, mal):
    assert [u.client_id for u in out] == [u.client_id for u in inp]
    for o, i, (cid, vec) in zip(out, inp, before):
        assert torch.equal(i.vector, vec)  # no input modified in place
        if cid not in mal:
            assert o is i and torch.equal(o.vector, vec)


# independent float64 reference computations for the Min-Max / Min-Sum constraints (p. 10)
def _B(uplinks, mal):
    return torch.stack([u.vector for u in uplinks if u.client_id not in mal]).double()


def max_dist(x, B):
    return (B - x.double()).norm(dim=1).max().item()


def max_pair(B):
    n = B.shape[0]
    return max(((B[i] - B[j]).norm().item() for i in range(n) for j in range(n)), default=0.0)


def sum_sq(x, B):
    return ((B - x.double()) ** 2).sum().item()


def max_sum_sq(B):
    return max(sum_sq(b, B) for b in B)


def direction(B32):
    g_b = B32.mean(0)
    return g_b.double(), (-g_b / g_b.norm()).double()


# ---------------------------------------------------------------------------- interface
ALL_ATTACKS = [
    lambda: NoiseInjection(),
    lambda: NoiseInjection(granularity="elementwise"),
    lambda: NoiseInjection(mode="replace"),
    lambda: SignFlip(),
    lambda: LabelFlip(num_classes=10),
    lambda: MinMax(),
    lambda: MinSum(),
    lambda: MinSum(squared=False),
]
IDS = ["noise", "noise_elem", "noise_replace", "sign_flip", "label_flip", "min_max",
       "min_sum", "min_sum_plain"]


@pytest.mark.parametrize("factory", ALL_ATTACKS, ids=IDS)
def test_every_attack_keeps_benign_and_is_finite(factory):
    atk = factory()
    assert isinstance(atk, Attack) and atk.level in ("data", "update")
    atk.setup({}, set(), _gen(1))
    for t in range(3):
        uplinks, mal, own_g = make_round(seed=t, t=t)
        before = snapshot(uplinks)
        out = atk.corrupt(t, uplinks, mal, own_g)
        assert_benign_untouched(out, uplinks, before, mal)
        for u in out:
            assert torch.isfinite(u.vector).all() and u.vector.shape == (D,)
            assert u.vector.dtype == torch.float32


@pytest.mark.parametrize("factory", ALL_ATTACKS, ids=IDS)
def test_zero_malicious_selected_is_noop(factory):
    atk = factory()
    atk.setup({}, set(), _gen(1))
    uplinks, _, _ = make_round(n_mal=0)
    out = atk.corrupt(0, uplinks, {99, 100}, {})
    assert len(out) == len(uplinks) and all(o is i for o, i in zip(out, uplinks))


# ---------------------------------------------------------------------------- noise (p. 9)
def test_noise_multiply_scalar_is_exact_multiple():
    atk = NoiseInjection()
    atk.setup({}, set(), _gen(3))
    uplinks, mal, own_g = make_round()
    out = atk.corrupt(0, uplinks, mal, own_g)
    p = atk.last_diagnostics["p"]
    assert set(p) == mal
    for u in out:
        if u.client_id in mal:
            assert torch.equal(u.vector, p[u.client_id] * own_g[u.client_id])
            ratio = u.vector / own_g[u.client_id]
            torch.testing.assert_close(ratio, torch.full_like(ratio, p[u.client_id]))


@pytest.mark.parametrize("kw", [{}, {"granularity": "elementwise"}, {"mode": "replace"}])
def test_noise_same_seed_same_upload(kw):
    outs = []
    for _ in range(2):
        atk = NoiseInjection(**kw)
        atk.setup({}, set(), _gen(42))
        uplinks, mal, own_g = make_round()
        outs.append([u.vector for u in atk.corrupt(0, uplinks, mal, own_g)])
    assert all(torch.equal(a, b) for a, b in zip(*outs))
    atk = NoiseInjection(**kw)
    atk.setup({}, set(), _gen(43))
    uplinks, mal, own_g = make_round()
    other = [u.vector for u in atk.corrupt(0, uplinks, mal, own_g)]
    assert not all(torch.equal(a, b) for a, b in zip(outs[0], other))


def test_noise_p_variance_is_3():
    atk = NoiseInjection(variance=3.0)
    atk.setup({}, set(), _gen(7))
    ps = []
    for t in range(300):
        uplinks, mal, own_g = make_round(n_benign=1, n_mal=20, d=2, seed=t, t=t)
        atk.corrupt(t, uplinks, mal, own_g)
        ps.extend(atk.last_diagnostics["p"].values())
    ps = torch.tensor(ps, dtype=torch.float64)
    assert len(ps) == 6000
    assert abs(ps.mean().item()) < 0.1
    assert abs(ps.var().item() - 3.0) < 0.25  # std error of the variance ~0.055


def test_noise_replace_ignores_g_and_elementwise_has_variance():
    atk = NoiseInjection(mode="replace")
    atk.setup({}, set(), _gen(0))
    uplinks, mal, own_g = make_round(n_benign=1, n_mal=1, d=20000)
    v = atk.corrupt(0, uplinks, mal, own_g)[1].vector
    assert abs(v.double().var().item() - 3.0) < 0.15
    assert abs(torch.corrcoef(torch.stack([v, own_g[1]]))[0, 1].item()) < 0.05


def test_noise_requires_setup():
    uplinks, mal, own_g = make_round()
    with pytest.raises(RuntimeError):
        NoiseInjection().corrupt(0, uplinks, mal, own_g)


# ---------------------------------------------------------------------------- sign flip (p. 9)
def test_sign_flip_is_minus_g():
    atk = SignFlip()
    uplinks, mal, own_g = make_round()
    before = snapshot(uplinks)
    out = atk.corrupt(0, uplinks, mal, own_g)
    assert_benign_untouched(out, uplinks, before, mal)
    for u in out:
        if u.client_id in mal:
            assert torch.equal(u.vector, -own_g[u.client_id])
            assert u.kind == "raw" and u.num_samples == 100 and u.round == 0


def test_missing_own_g_raises():
    uplinks, mal, _ = make_round()
    with pytest.raises(KeyError):
        SignFlip().corrupt(0, uplinks, mal, {})


# ---------------------------------------------------------------------------- label flip (pp. 2, 9)
def _client_labels(L=10, sizes=(7, 10, 1, 0, 13), seed=0):
    g = _gen(seed)
    return {cid: torch.randint(0, L, (n,), generator=g) for cid, n in enumerate(sizes)}


def test_label_flip_half_with_rule():
    L = 10  # even: l -> L - l - 1 never maps a label to itself
    clients = _client_labels(L)
    originals = {cid: y.clone() for cid, y in clients.items()}
    mal = {0, 2, 3, 4, 77}  # 77 is not a known client: ignored
    atk = LabelFlip(num_classes=L)
    atk.setup(clients, mal, _gen(5))

    assert set(atk.label_overrides) == {0, 2, 3, 4}
    for cid, y in clients.items():
        assert torch.equal(y, originals[cid])  # caller's labels are never modified in place
        if cid not in mal:
            assert cid not in atk.label_overrides  # honest client untouched
            continue
        new = atk.label_overrides[cid]
        changed = (new != y).nonzero().flatten()
        assert len(changed) == len(y) // 2
        assert torch.equal(changed, atk.flipped_positions[cid])
        assert torch.equal(new[changed], L - y[changed] - 1)
        unchanged = (new == y)
        assert int(unchanged.sum()) == len(y) - len(y) // 2


def test_label_flip_same_seed_same_flips():
    clients = _client_labels(sizes=(50, 60))
    a, b, c = LabelFlip(10), LabelFlip(10), LabelFlip(10)
    a.setup(clients, {0, 1}, _gen(9))
    b.setup(clients, {1, 0}, _gen(9))
    c.setup(clients, {0, 1}, _gen(10))
    for cid in (0, 1):
        assert torch.equal(a.label_overrides[cid], b.label_overrides[cid])
        assert torch.equal(a.flipped_positions[cid], b.flipped_positions[cid])
    assert not torch.equal(a.flipped_positions[0], c.flipped_positions[0])


def test_label_flip_odd_classes_fixed_point():
    """L odd: the middle class maps to itself, so it is 'flipped' without changing (p. 2 rule)."""
    y = torch.full((9,), 2)
    new, pos = flip_labels(y, num_classes=5, generator=_gen(0))
    assert len(pos) == 4 and torch.equal(new, y)


def test_label_flip_corrupt_is_identity_and_validates():
    atk = LabelFlip(10)
    uplinks, mal, own_g = make_round()
    out = atk.corrupt(0, uplinks, mal, own_g)
    assert all(o is i for o, i in zip(out, uplinks))
    with pytest.raises(ValueError):
        flip_labels(torch.tensor([0, 10]), 10, _gen(0))


# ---------------------------------------------------------------------------- Min-Max / Min-Sum (p. 10)
def _run(atk, **kw):
    atk.setup({}, set(), _gen(0))
    uplinks, mal, own_g = make_round(**kw)
    out = atk.corrupt(0, uplinks, mal, own_g)
    mal_vecs = [u.vector for u in out if u.client_id in mal]
    return uplinks, mal, own_g, out, mal_vecs


@pytest.mark.parametrize("n_benign", [2, 5, 9])
def test_min_max_constraint_is_tight(n_benign):
    atk = MinMax()
    uplinks, mal, _, _, mal_vecs = _run(atk, n_benign=n_benign, n_mal=3)
    assert all(torch.equal(v, mal_vecs[0]) for v in mal_vecs)  # all attackers identical
    B = _B(uplinks, mal)
    T = max_pair(B)
    gamma = atk.last_diagnostics["gamma"]
    assert gamma > 0
    assert atk.last_diagnostics["threshold"] == pytest.approx(T, rel=1e-9)
    assert max_dist(mal_vecs[0], B) <= T * (1 + 1e-5)  # constraint holds

    g_b, p = direction(B.float())
    torch.testing.assert_close(mal_vecs[0].double(), g_b + gamma * p, rtol=1e-5, atol=1e-6)
    assert max_dist(g_b + gamma * (1 + 1e-4) * p, B) > T  # slightly larger gamma breaks it

    # bisection agrees with the closed form gamma* = min_i(-c_i + sqrt(c_i^2 - a_i + T^2))
    diff = g_b - B
    a, c = (diff ** 2).sum(1), diff @ p
    closed = (-c + (c ** 2 - a + T ** 2).sqrt()).min().item()
    assert gamma == pytest.approx(closed, rel=1e-7)


@pytest.mark.parametrize("n_benign", [2, 5, 9])
def test_min_sum_constraint_is_tight(n_benign):
    atk = MinSum()
    uplinks, mal, _, _, mal_vecs = _run(atk, n_benign=n_benign, n_mal=3)
    assert all(torch.equal(v, mal_vecs[0]) for v in mal_vecs)
    B = _B(uplinks, mal)
    T = max_sum_sq(B)
    gamma = atk.last_diagnostics["gamma"]
    assert gamma > 0
    assert sum_sq(mal_vecs[0], B) <= T * (1 + 1e-5)

    g_b, p = direction(B.float())
    assert sum_sq(g_b + gamma * (1 + 1e-4) * p, B) > T

    # closed form: n g^2 + 2 C g + A = T
    diff = g_b - B
    n, A, C = B.shape[0], (diff ** 2).sum().item(), (diff @ p).sum().item()
    closed = (-C + math.sqrt(C * C - n * (A - T))) / n
    assert gamma == pytest.approx(closed, rel=1e-7)


def test_min_sum_is_stricter_than_min_max():
    """Paper p. 2: Min-Sum uses a stricter, total-distance constraint."""
    mm, ms = MinMax(), MinSum()
    _run(mm, n_benign=6)
    _run(ms, n_benign=6)
    assert 0 < ms.last_diagnostics["gamma"] < mm.last_diagnostics["gamma"]


def test_min_sum_plain_distance_constraint():
    atk = MinSum(squared=False)
    uplinks, mal, _, _, mal_vecs = _run(atk, n_benign=5)
    B = _B(uplinks, mal)
    T = max((B - b).norm(dim=1).sum().item() for b in B)
    assert (B - mal_vecs[0].double()).norm(dim=1).sum().item() <= T * (1 + 1e-5)
    g_b, p = direction(B.float())
    gamma = atk.last_diagnostics["gamma"]
    assert (B - (g_b + gamma * (1 + 1e-4) * p)).norm(dim=1).sum().item() > T


@pytest.mark.parametrize("cls", [MinMax, MinSum])
def test_crafted_one_benign_gives_gamma_zero(cls):
    """One benign update: no pairs, threshold taken as 0, so gamma = 0 and g_hat = that update."""
    atk = cls()
    uplinks, mal, _, out, mal_vecs = _run(atk, n_benign=1, n_mal=3)
    assert atk.last_diagnostics["gamma"] == 0.0
    assert atk.last_diagnostics["threshold"] == 0.0
    for v in mal_vecs:
        assert torch.equal(v, uplinks[0].vector)


@pytest.mark.parametrize("cls", [MinMax, MinSum])
def test_crafted_no_benign_falls_back_to_own_g(cls):
    atk = cls()
    uplinks, mal, own_g, out, mal_vecs = _run(atk, n_benign=0, n_mal=4)
    assert atk.last_diagnostics["reference"] == "own_g"
    assert atk.last_diagnostics["gamma"] > 0
    assert all(torch.equal(v, mal_vecs[0]) for v in mal_vecs) and torch.isfinite(mal_vecs[0]).all()
    B = torch.stack([own_g[m] for m in sorted(mal)]).double()
    if cls is MinMax:
        assert max_dist(mal_vecs[0], B) <= max_pair(B) * (1 + 1e-5)
    else:
        assert sum_sq(mal_vecs[0], B) <= max_sum_sq(B) * (1 + 1e-5)


@pytest.mark.parametrize("cls", [MinMax, MinSum])
def test_crafted_zero_benign_mean(cls):
    """||g_b|| = 0: p undefined, gamma = 0, attackers upload g_b = 0."""
    b = torch.randn(D, generator=_gen(0))
    uplinks = [Uplink(0, 0, b, "raw", 1), Uplink(0, 1, -b, "raw", 1),
               Uplink(0, 2, torch.ones(D), "raw", 1)]
    atk = cls()
    out = atk.corrupt(0, uplinks, {2}, {2: torch.ones(D)})
    assert atk.last_diagnostics["gamma"] == 0.0
    assert torch.isfinite(out[2].vector).all() and out[2].vector.abs().max().item() < 1e-6


def test_bisect_max_gamma():
    assert bisect_max_gamma(lambda g: g <= 3.7) == pytest.approx(3.7, rel=1e-12)
    assert bisect_max_gamma(lambda g: g <= 0.0) == 0.0
    assert bisect_max_gamma(lambda g: g * g <= 1e6) == pytest.approx(1e3, rel=1e-12)
