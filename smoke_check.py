"""SMOKE CHECKS (not formal tests, not paper experiments). Uses SYNTHETIC tensors because it must run
offline; it exercises the real code path. Run:  python smoke_check.py"""
import json, os, shutil, sys, tempfile, traceback

import numpy as np
import torch

results = []
def check(name):
    def deco(fn):
        try:
            fn(); results.append((name, "PASS", ""))
        except Exception as e:
            traceback.print_exc(); results.append((name, "FAIL", repr(e)))
    return deco

@check("imports")
def _():
    import src.algorithms.drag, src.fl.trainer, src.models, src.fl.partition, src.fl.data  # noqa

@check("models construct + forward (EMNIST CNN, CIFAR-10 CNN, CIFAR-100 CNN, WRN-28-4)")
def _():
    from src.models import build_model
    for name, ds, shape, k in [("cnn", "emnist", (2, 1, 28, 28), 47), ("cnn", "cifar10", (2, 3, 32, 32), 10),
                               ("cnn", "cifar100", (2, 3, 32, 32), 100), ("wrn28_4", "cifar100", (2, 3, 32, 32), 100)]:
        m = build_model(name, ds); m.eval()
        out = m(torch.randn(*shape)); assert out.shape == (2, k) and torch.isfinite(out).all()
    n_conv = sum(isinstance(x, torch.nn.Conv2d) for x in build_model("wrn28_4", "cifar100").modules())
    print("  WRN-28-4 conv layers (incl. 1x1 shortcuts):", n_conv)

@check("DRAG math: lambda range, Eq.(11) special cases, finite & shape-preserving")
def _():
    from src.algorithms.drag import drag_modify, divergence_of_directions, aggregate
    r = torch.tensor([2.0, 0.0]); c = 0.25
    g = torch.tensor([3.0, 0.0]); v, lam = drag_modify(g, r, c)          # cos=1 -> lambda=0 -> v=g
    assert lam.item() == 0 and torch.allclose(v, g)
    g = torch.tensor([-3.0, 0.0]); v, lam = drag_modify(g, r, 0.5)       # cos=-1, c=.5 -> lambda=1 -> v=(|g|/|r|) r
    assert abs(lam.item() - 1.0) < 1e-6 and torch.allclose(v, torch.tensor([3.0, 0.0]))
    g = torch.tensor([0.0, 1.0]); lam = divergence_of_directions(g, r, c)  # cos=0 -> lambda=c
    assert abs(lam.item() - c) < 1e-6
    for _ in range(50):
        g, r = torch.randn(1000), torch.randn(1000); v, lam = drag_modify(g, r, 0.25)
        assert v.shape == g.shape and torch.isfinite(v).all() and 0 <= lam.item() <= 0.5 + 1e-6
    v0, l0 = drag_modify(torch.randn(5), torch.zeros(5), 0.25); assert torch.isfinite(v0).all() and l0 == 0   # zero-norm r
    assert aggregate([torch.ones(3), 3 * torch.ones(3)]).tolist() == [2, 2, 2]

@check("reference direction: Eq.(5a) init, Eq.(5b) update, state persists")
def _():
    from src.algorithms.drag import ReferenceDirection
    ref = ReferenceDirection(0.25)
    r0 = ref.initialize([torch.tensor([1., 0.]), torch.tensor([3., 0.])]); assert r0.tolist() == [2., 0.]
    ref.record_delta(torch.tensor([0., 4.])); r1 = ref.advance()
    assert torch.allclose(r1, torch.tensor([1.5, 1.0]))   # .75*[2,0] + .25*[0,4]

@check("Dirichlet partition: exact cover, no duplicates, seed-reproducible, min size")
def _():
    from src.fl.partition import dirichlet_partition
    labels = np.repeat(np.arange(10), 5000)
    for beta in (0.1, 0.5):
        p = dirichlet_partition(labels, 40, beta, seed=1, min_size=10)
        allidx = np.concatenate(p); assert len(allidx) == len(labels) and len(np.unique(allidx)) == len(labels)
        assert min(len(x) for x in p) >= 10
        p2 = dirichlet_partition(labels, 40, beta, seed=1, min_size=10); assert all((a == b).all() for a, b in zip(p, p2))
        print(f"  beta={beta}: client sizes min/max = {min(map(len,p))}/{max(map(len,p))}")

@check("one local SGD step: update has param shape, finite, nonzero")
def _():
    from src.models import build_model
    from src.fl.client import local_update, param_vector
    m = build_model("cnn", "cifar10"); th = param_vector(m)
    x, y = torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,))
    g, loss, _ = local_update(m, th, {}, x, y, np.arange(100), 0.01, 1, 10, 0, torch.device("cpu"))
    assert g.shape == th.shape and torch.isfinite(g).all() and g.norm() > 0

@check("full DRAG rounds (synthetic data, all 3 datasets) + metrics + checkpoint + resume")
def _():
    from src.fl.trainer import run
    from src.utils.config import Config
    out = tempfile.mkdtemp()
    try:
        for ds, beta in [("emnist", 0.5), ("cifar10", 0.1), ("cifar100", 0.1)]:
            cfg = Config(dataset=ds, beta=beta, rounds=3, synthetic_data=True, output_dir=out, device="cpu",
                         checkpoint_every=1, min_partition_size=10)
            d = run(cfg, verbose=False)
            rows = open(os.path.join(d, "metrics.csv")).read().strip().splitlines(); assert len(rows) == 4
            assert os.path.exists(os.path.join(d, "checkpoints", "final.pt"))
            assert json.load(open(os.path.join(d, "status.json")))["status"] == "completed"
            print(f"  {ds}: ran 3 rounds; last row: {rows[-1][:90]}")
        cfg = Config(dataset="cifar10", beta=0.1, rounds=5, synthetic_data=True, output_dir=out, device="cpu", resume=True,
                     checkpoint_every=1)
        cfg.tag = ""; d = run(cfg, verbose=False)   # resumes the 3-round run? (different rounds -> same run name) 
        rows = open(os.path.join(d, "metrics.csv")).read().strip().splitlines(); assert len(rows) == 6, len(rows)
    finally:
        shutil.rmtree(out, ignore_errors=True)

@check("WRN-28-4 one DRAG round (BN buffers averaged), synthetic CIFAR-100")
def _():
    from src.fl.trainer import run
    from src.utils.config import Config
    out = tempfile.mkdtemp()
    try:
        run(Config(dataset="cifar100", model="wrn28_4", beta=0.5, rounds=1, clients_per_round=2, num_clients=4,
                   synthetic_data=True, output_dir=out, device="cpu", eval_batch_size=100), verbose=False)
    finally:
        shutil.rmtree(out, ignore_errors=True)

print("\n=== SMOKE CHECK SUMMARY (synthetic data; NOT full experiments) ===")
for n, s, e in results: print(f"[{s}] {n} {e}")
sys.exit(0 if all(s == "PASS" for _, s, _ in results) else 1)
