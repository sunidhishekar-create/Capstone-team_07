"""End-to-end smoke test for BR-DRAG, FLTrust and the attacks on a synthetic task (CPU only).

Not part of the package. Run from the repo root:
    PYTHONPATH=src python scripts/smoke_br_drag.py

Setup (paper-style, scaled down): M=40 clients, S=10 per round, U=5 plain-SGD steps, B=10,
lr=0.01, Dirichlet(0.5) label split, 30% fixed malicious clients, 300-sample class-balanced
root set held out from all clients. Methods {FedAvg, BR-DRAG, FLTrust-paper, FLTrust-cao} x
attacks {none, sign flip, noise injection, Min-Max}. Prints a final-accuracy table and saves
scripts/smoke_results.png.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedrag.attacks.min_max import MinMax  # noqa: E402
from fedrag.attacks.noise import NoiseInjection  # noqa: E402
from fedrag.attacks.sign_flip import SignFlip  # noqa: E402
from fedrag.strategies.br_drag import BRDRAG, Uplink, aggregate  # noqa: E402
from fedrag.strategies.fltrust import FLTrust  # noqa: E402
from fedrag.strategies.root_trainer import get_params, load_params, root_reference  # noqa: E402

SEED = 0
NUM_CLASSES, DIM, HIDDEN = 10, 20, 64
N_TRAIN_PER_CLASS, N_TEST_PER_CLASS, ROOT_PER_CLASS = 800, 200, 30
M, S, U, B, LR = 40, 10, 5, 10, 0.01
BETA, MAL_FRAC, ROUNDS, EVAL_EVERY = 0.5, 0.3, 100, 10
CLASS_SEP = 1.0  # std of the class means; samples are mean + N(0, I)


# ----------------------------------------------------------------------------- data
def make_data(rng: np.random.Generator):
    """Gaussian clusters: one random mean per class, unit-variance noise."""
    means = rng.normal(0.0, CLASS_SEP, size=(NUM_CLASSES, DIM))

    def sample(n_per_class):
        y = np.repeat(np.arange(NUM_CLASSES), n_per_class)
        x = means[y] + rng.normal(size=(len(y), DIM))
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    return sample(N_TRAIN_PER_CLASS), sample(N_TEST_PER_CLASS)


def split(y: torch.Tensor, rng: np.random.Generator):
    """Carve a class-balanced root set, then Dirichlet(BETA) split the rest across M clients."""
    yn = y.numpy()
    root_idx, rest = [], []
    for c in range(NUM_CLASSES):
        idx = rng.permutation(np.flatnonzero(yn == c))
        root_idx.extend(idx[:ROOT_PER_CLASS])
        rest.append(idx[ROOT_PER_CLASS:])
    for _ in range(100):  # redraw until every client has at least B samples
        shards = [[] for _ in range(M)]
        for idx in rest:
            cuts = (np.cumsum(rng.dirichlet(BETA * np.ones(M))) * len(idx)).astype(int)[:-1]
            for m, part in enumerate(np.split(idx, cuts)):
                shards[m].extend(part)
        if min(len(s) for s in shards) >= B:
            break
    else:
        raise RuntimeError("could not draw a Dirichlet split with >= B samples per client")
    assert not set(root_idx) & set(i for s in shards for i in s)  # root is disjoint from clients
    return torch.tensor(root_idx), [torch.tensor(s) for s in shards]


# ----------------------------------------------------------------------------- FL loop
def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, NUM_CLASSES))


@torch.no_grad()
def accuracy(model, theta, x, y) -> float:
    load_params(model, theta)
    model.eval()
    return (model(x).argmax(1) == y).float().mean().item()


class FedAvg:
    """Inline baseline: theta + unweighted mean of raw g (paper eq. (4))."""

    name = "fedavg"

    def prepare_downlink(self, t, theta):
        return None

    def server_transform(self, t, uplinks, dl):
        return [u.vector for u in uplinks], {}

    def aggregate(self, t, vectors):
        return aggregate(vectors)

    def apply(self, t, theta, delta):
        return theta + delta


def run(method: str, attack_name: str, ctx) -> list[tuple[int, float]]:
    model = make_model()
    theta = ctx["theta0"].clone()
    root_data = ctx["root"]
    root_gen = torch.Generator().manual_seed(SEED + 1)
    kw = dict(steps=U, batch_size=B, lr=LR)
    strategy = {
        "FedAvg": lambda: FedAvg(),
        "BR-DRAG": lambda: BRDRAG(model, root_data, root_gen, c_t=0.5, **kw),
        "FLTrust-paper": lambda: FLTrust(model, root_data, root_gen, variant="paper", **kw),
        "FLTrust-cao": lambda: FLTrust(model, root_data, root_gen, variant="cao", **kw),
    }[method]()
    attack = {"none": None, "sign flip": SignFlip(), "noise": NoiseInjection(),
              "Min-Max": MinMax()}[attack_name]
    if attack is not None:
        attack.setup({}, ctx["malicious"], torch.Generator().manual_seed(SEED + 2))

    # same selection and minibatch streams for every (method, attack) pair
    select_rng = np.random.default_rng(SEED + 3)
    batch_gen = torch.Generator().manual_seed(SEED + 4)
    history = [(0, accuracy(model, theta, *ctx["test"]))]
    for t in range(ROUNDS):
        selected = sorted(select_rng.choice(M, size=S, replace=False).tolist())
        dl = strategy.prepare_downlink(t, theta)
        uplinks, own_g = [], {}
        for m in selected:
            x, y = ctx["clients"][m]
            g = root_reference(model, theta, (x, y), steps=U, batch_size=B, lr=LR,
                               generator=batch_gen)  # plain local SGD, g = theta_U - theta
            uplinks.append(Uplink(t, m, g, "raw", len(y)))
            if m in ctx["malicious"]:
                own_g[m] = g
        if attack is not None:
            uplinks = attack.corrupt(t, uplinks, ctx["malicious"], own_g)
        vecs, _ = strategy.server_transform(t, uplinks, dl)
        theta = strategy.apply(t, theta, strategy.aggregate(t, vecs))
        if (t + 1) % EVAL_EVERY == 0:
            history.append((t + 1, accuracy(model, theta, *ctx["test"])))
    return history


# ----------------------------------------------------------------------------- main
def main() -> None:
    rng = np.random.default_rng(SEED)
    (x_tr, y_tr), test = make_data(rng)
    root_idx, shards = split(y_tr, rng)
    malicious = set(rng.choice(M, size=round(MAL_FRAC * M), replace=False).tolist())
    torch.manual_seed(SEED)  # same model init for every run
    ctx = {
        "theta0": get_params(make_model()),
        "root": (x_tr[root_idx], y_tr[root_idx]),
        "clients": [(x_tr[s], y_tr[s]) for s in shards],
        "test": test,
        "malicious": malicious,
    }
    sizes = [len(s) for s in shards]
    print(f"train={len(y_tr)} root={len(root_idx)} clients: min={min(sizes)} max={max(sizes)} "
          f"samples; malicious={sorted(malicious)}")

    methods = ["FedAvg", "BR-DRAG", "FLTrust-paper", "FLTrust-cao"]
    attacks = ["none", "sign flip", "noise", "Min-Max"]
    results = {}
    for a in attacks:
        for meth in methods:
            start = time.perf_counter()
            results[meth, a] = run(meth, a, ctx)
            print(f"  {meth:14s} | {a:9s} | acc@{ROUNDS}={results[meth, a][-1][1]:.3f} "
                  f"({time.perf_counter() - start:.1f}s)", flush=True)

    print(f"\nFinal test accuracy after {ROUNDS} rounds "
          f"(M={M}, S={S}, U={U}, B={B}, lr={LR}, beta={BETA}, {int(MAL_FRAC * 100)}% malicious)")
    print(f"{'method':14s}" + "".join(f"{a:>11s}" for a in attacks))
    for meth in methods:
        print(f"{meth:14s}" + "".join(f"{results[meth, a][-1][1]:>11.3f}" for a in attacks))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(attacks), figsize=(4 * len(attacks), 3.4), sharey=True)
    for ax, a in zip(axes, attacks):
        for meth in methods:
            rounds, accs = zip(*results[meth, a])
            ax.plot(rounds, accs, marker="o", ms=3, label=meth)
        ax.set_title(f"attack: {a}")
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "scripts" / "smoke_results.png"
    fig.savefig(out, dpi=120)
    print(f"\nplot saved to {out}")


if __name__ == "__main__":
    main()
