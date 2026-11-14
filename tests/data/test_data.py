"""Tests for the data pipeline, models and runner. CPU only, tiny synthetic data.

Run with:  PYTHONPATH=src pytest tests/data
"""

import csv
import importlib.util

import pytest
import torch

from fedrag.data import FedDataset, build_root_set, dirichlet_partition
from fedrag.models import build_model
from fedrag.runner.run import RunConfig, run


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _labels(num_classes=5, per_class=(30, 50, 80, 120, 200), seed=0):
    """Unbalanced labels in shuffled order."""
    y = torch.cat([torch.full((n,), c) for c, n in zip(range(num_classes), per_class)])
    return y[torch.randperm(y.numel(), generator=_gen(seed))]


# ----------------------------------------------------------------------------- root set
def test_root_set_class_balanced():
    y = _labels()
    root = build_root_set(y, 5, size=100, generator=_gen())
    assert root.numel() == 100
    assert torch.bincount(y[root], minlength=5).tolist() == [20] * 5
    assert root.unique().numel() == root.numel()


def test_root_set_size_floors_per_class():
    # paper default 3000 on EMNIST (47 classes) -> 63 per class = 2961
    y = torch.arange(47).repeat_interleave(100)
    root = build_root_set(y, 47, size=3000, generator=_gen())
    assert root.numel() == 63 * 47 == 2961
    assert set(torch.bincount(y[root]).tolist()) == {63}


def test_root_set_deterministic():
    y = _labels()
    a = build_root_set(y, 5, size=50, generator=_gen(3))
    b = build_root_set(y, 5, size=50, generator=_gen(3))
    c = build_root_set(y, 5, size=50, generator=_gen(4))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_root_set_too_few_samples_raises():
    with pytest.raises(ValueError):
        build_root_set(_labels(), 5, size=5 * 31, generator=_gen())  # class 0 has 30


@pytest.mark.skipif(importlib.util.find_spec("cleanlab") is None, reason="cleanlab not installed")
def test_root_set_vetting_drops_mislabelled():
    # 3 well-separated classes; corrupt some labels so cleanlab has something to flag
    g = _gen(1)
    k, n = 3, 200
    y = torch.arange(k).repeat_interleave(n)
    x = torch.zeros(k * n, 1, 4, 4, dtype=torch.uint8)
    for c in range(k):
        x[y == c, 0, c] = 200
    x = (x.float() + torch.rand(x.shape, generator=g) * 40).clamp(0, 255).to(torch.uint8)
    noisy = y.clone()
    bad = torch.randperm(k * n, generator=g)[:30]
    noisy[bad] = (noisy[bad] + 1) % k
    root = build_root_set(noisy, k, size=60, generator=_gen(), vet=True, images=x)
    assert root.numel() <= 60
    assert (torch.bincount(noisy[root], minlength=k) <= 20).all()
    assert (noisy[root] == y[root]).float().mean() > 0.95  # mislabelled samples mostly removed


# ----------------------------------------------------------------------------- partition
@pytest.mark.parametrize("beta", [0.1, 0.5, 5.0])
def test_partition_covers_non_root_exactly_once(beta):
    y = _labels(per_class=(300, 300, 300, 300, 300))
    root = build_root_set(y, 5, size=50, generator=_gen())
    parts = dirichlet_partition(y, num_clients=8, beta=beta, generator=_gen(), exclude=root, min_size=10)
    assert len(parts) == 8
    allidx = torch.cat(parts)
    assert allidx.unique().numel() == allidx.numel()  # no index twice
    expected = torch.ones(y.numel(), dtype=torch.bool)
    expected[root] = False
    assert torch.equal(allidx.sort().values, expected.nonzero().squeeze(1))  # every non-root once
    assert not torch.isin(allidx, root).any()  # no root index in any client
    assert min(p.numel() for p in parts) >= 10


def test_partition_deterministic():
    y = _labels(per_class=(300,) * 5)
    a = dirichlet_partition(y, 8, 0.5, _gen(7), min_size=10)
    b = dirichlet_partition(y, 8, 0.5, _gen(7), min_size=10)
    c = dirichlet_partition(y, 8, 0.5, _gen(8), min_size=10)
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert not all(torch.equal(p, q) for p, q in zip(a, c))


def test_partition_small_beta_is_skewed():
    y = _labels(per_class=(400,) * 5)
    parts = dirichlet_partition(y, 10, 0.1, _gen(), min_size=1)
    # with beta = 0.1 most clients hold few classes
    present = [int((torch.bincount(y[p], minlength=5) > 0).sum()) for p in parts]
    assert sum(present) / len(present) < 4


def test_partition_impossible_min_size_raises():
    y = _labels(per_class=(10,) * 5)
    with pytest.raises(ValueError):
        dirichlet_partition(y, 10, 0.5, _gen(), min_size=10)  # 50 samples < 10 * 10


# ----------------------------------------------------------------------------- models
@pytest.mark.parametrize("name,shape,k", [("emnist_balanced", (2, 1, 28, 28), 47),
                                          ("cifar10", (2, 3, 32, 32), 10),
                                          ("cifar100", (2, 3, 32, 32), 100)])
def test_model_shapes_and_seeded_init(name, shape, k):
    x = torch.randint(0, 256, shape, dtype=torch.uint8)
    m = build_model(name, seed=5)
    assert m(x).shape == (2, k)
    vec = lambda mod: torch.cat([p.detach().flatten() for p in mod.parameters()])  # noqa: E731
    assert torch.equal(vec(m), vec(build_model(name, seed=5)))
    assert not torch.equal(vec(m), vec(build_model(name, seed=6)))


# ----------------------------------------------------------------------------- runner
K = 4


def _tiny_data(seed=0, per_class=60, test_per_class=10):
    """28x28 grayscale, 4 classes, each class lights up its own image quadrant."""
    g = _gen(seed)

    def make(n):
        y = torch.arange(K).repeat_interleave(n)
        x = torch.rand(y.numel(), 1, 28, 28, generator=g) * 60
        for c in range(K):
            r, col = divmod(c, 2)
            x[y == c, 0, r * 14 : r * 14 + 14, col * 14 : col * 14 + 14] += 150
        perm = torch.randperm(y.numel(), generator=g)
        return x[perm].to(torch.uint8), y[perm]

    x_tr, y_tr = make(per_class)
    x_te, y_te = make(test_per_class)
    return FedDataset("emnist_balanced", x_tr, y_tr, x_te, y_te, num_classes=K)


def _cfg(tmp_path, name, **kw):
    base = dict(dataset="emnist_balanced", beta=1.0, num_clients=6, clients_per_round=3,
                malicious=0.34, rounds=3, eval_every=2, seed=0, root_size=20, min_client_size=10,
                checkpoint_every=50, device="cpu", out=str(tmp_path / name))
    base.update(kw)
    return RunConfig(**base)


@pytest.mark.parametrize("method,attack", [("fedavg", "none"), ("br_drag", "sign_flip"),
                                           ("fltrust_paper", "label_flip"), ("fltrust_cao", "min_max"),
                                           ("br_drag", "min_sum"), ("br_drag", "noise")])
def test_runner_three_rounds_writes_metrics(tmp_path, method, attack):
    cfg = _cfg(tmp_path, f"{method}_{attack}", method=method, attack=attack)
    res = run(cfg, data=_tiny_data(), verbose=False)
    assert torch.isfinite(res["theta"]).all()
    with open(cfg.out + "/metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["round"]) for r in rows] == [0, 1, 2, 3]
    evals = [r for r in rows if r["test_acc"] != ""]
    assert [int(r["round"]) for r in evals] == [0, 2, 3]  # round 0, every 2, and the last
    assert all(0.0 <= float(r["test_acc"]) <= 1.0 for r in evals)
    for name in ("config.json", "partition.csv", "checkpoint.pt"):
        assert (tmp_path / f"{method}_{attack}" / name).exists()
    if method == "br_drag":
        assert "lambda_mean" in rows[1]


def test_runner_resume_matches_uninterrupted(tmp_path):
    data = _tiny_data()
    kw = dict(method="br_drag", attack="noise", eval_every=1, checkpoint_every=2)
    full = run(_cfg(tmp_path, "full", rounds=5, **kw), data=data, verbose=False)
    # "interrupted": stop after round 3 (checkpoint saved at the end), then resume to round 5
    run(_cfg(tmp_path, "split", rounds=3, **kw), data=data, verbose=False)
    resumed = run(_cfg(tmp_path, "split", rounds=5, resume=True, **kw), data=data, verbose=False)
    assert torch.equal(full["theta"], resumed["theta"])
    key = lambda rows: [(r["round"], r.get("test_acc"), r.get("lambda_mean")) for r in rows]  # noqa: E731
    assert key(full["metrics"]) == key(resumed["metrics"])


def test_runner_refuses_to_overwrite_checkpoint(tmp_path):
    data = _tiny_data()
    run(_cfg(tmp_path, "x", rounds=1), data=data, verbose=False)
    with pytest.raises(FileExistsError):
        run(_cfg(tmp_path, "x", rounds=1), data=data, verbose=False)
    with pytest.raises(ValueError, match="config differs"):
        run(_cfg(tmp_path, "x", rounds=2, resume=True, lr=0.05), data=data, verbose=False)
