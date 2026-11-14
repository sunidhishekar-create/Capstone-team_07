"""Min-Max attack, plus the crafted-update machinery shared with Min-Sum.

Base paper: arXiv 2601.06903, p. 2 (Section I-A) and pp. 9-10 (Section VI-B). All malicious
workers upload the same g_hat = g_b + gamma * p, with g_b the mean benign update of round t.
Min-Max picks the largest gamma such that the maximum distance from g_hat to any benign update
does not exceed the maximum pairwise distance among benign updates (p. 10).

The paper does not give p; we use the inverse unit vector p = -g_b / ||g_b||
(architecture.md 15 #3). gamma is found by bisection (doubling, then halving).
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from fedrag.attacks.base import Attack, own_update, replace_vectors, split_uplinks
from fedrag.strategies.br_drag import Uplink, Vec

EPS = 1e-12


def inverse_unit_direction(g_b: Vec, eps: float = EPS) -> Vec:
    """Attack direction p = -g_b / ||g_b|| (paper p. 9 leaves p open). Zero if ||g_b|| < eps."""
    norm = torch.linalg.vector_norm(g_b)
    if norm < eps:
        return torch.zeros_like(g_b)
    return -g_b / norm


def bisect_max_gamma(feasible: Callable[[float], bool], iters: int = 200) -> float:
    """Largest gamma >= 0 with ``feasible(gamma)``, by bisection (paper p. 10, gamma "as large
    as possible").

    Assumes the feasible set on [0, inf) is an interval [0, gamma*] (true for Min-Max and
    Min-Sum: both constraints are convex in gamma and hold at gamma = 0). Doubles an upper
    bound until infeasible, then halves; returns the feasible end, so the result always
    satisfies the constraint.
    """
    if not feasible(0.0):
        return 0.0
    lo, hi = 0.0, 1.0
    while feasible(hi):
        lo, hi = hi, hi * 2.0
        if hi > 1e300:
            return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def reference_set(
    benign: list[Uplink], malicious: list[Uplink], own_g: dict[int, Vec]
) -> tuple[torch.Tensor, str]:
    """Stack the updates g_b and the constraints are built from (paper p. 9, B^t).

    Normally the benign uplinks of round t. If no benign client was selected, falls back to
    the malicious clients' own honest updates (the attacker's best estimate of benign
    behaviour). Returns (n x d tensor, source label).
    """
    if benign:
        return torch.stack([u.vector for u in benign]), "benign"
    return torch.stack([own_update(own_g, u.client_id) for u in malicious]), "own_g"


def _gamma_terms(refs: torch.Tensor, g_b: Vec, p: Vec) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Scalars so that ||g_b + gamma p - b_i||^2 = a_i + 2 gamma c_i + gamma^2 ||p||^2 (p. 10).

    Computed once in float64, so each bisection step costs O(n) instead of O(n d).
    """
    diff = g_b.double().unsqueeze(0) - refs.double()  # g_b - b_i
    a = (diff * diff).sum(dim=1)
    c = diff @ p.double()
    p2 = float(torch.dot(p.double(), p.double()))
    return a, c, p2


def pairwise_sq_dists(refs: torch.Tensor) -> torch.Tensor:
    """n x n matrix of ||b_i - b_j||^2 in float64 (thresholds on p. 10)."""
    r = refs.double()
    sq = (r * r).sum(dim=1)
    d2 = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (r @ r.T)
    d2 = d2.clamp_min(0.0)
    d2.fill_diagonal_(0.0)
    return d2


class _CraftedUpdateAttack(Attack):
    """Shared Min-Max / Min-Sum logic: identical g_hat = g_b + gamma p for every attacker (p. 9)."""

    level = "update"

    def _threshold(self, refs: torch.Tensor) -> float:
        """Bound the crafted update must respect, from the benign group alone (p. 10)."""
        raise NotImplementedError

    def _constraint(self, a: torch.Tensor, c: torch.Tensor, p2: float, gamma: float) -> float:
        """Value of the constrained quantity at gamma, comparable to ``_threshold`` (p. 10)."""
        raise NotImplementedError

    def corrupt(
        self,
        t: int,
        uplinks: list[Uplink],
        malicious_ids: set[int],
        own_g: dict[int, Vec],
    ) -> list[Uplink]:
        """Replace every malicious uplink with the same g_hat = g_b + gamma p (paper p. 9).

        Edge cases: no malicious client selected -> uplinks unchanged, gamma = nan.
        One reference update -> the pairwise threshold is the max over an empty set, taken as
        0, so gamma = 0 and attackers upload g_b (they cannot deviate without standing out).
        ||g_b|| ~ 0 -> p is undefined, gamma = 0, attackers upload g_b.
        """
        benign, malicious = split_uplinks(uplinks, malicious_ids)
        if not malicious:
            self.last_diagnostics = {"gamma": float("nan"), "threshold": float("nan"),
                                     "num_benign": len(benign), "num_malicious": 0,
                                     "reference": "none"}
            return list(uplinks)

        refs, source = reference_set(benign, malicious, own_g)
        g_b = refs.mean(dim=0)
        p = inverse_unit_direction(g_b)
        threshold = self._threshold(refs)  # 0 when there is a single reference update
        if bool(p.any()):
            a, c, p2 = _gamma_terms(refs, g_b, p)
            gamma = bisect_max_gamma(lambda gm: self._constraint(a, c, p2, gm) <= threshold)
        else:
            gamma = 0.0

        g_hat = g_b + gamma * p
        new = {u.client_id: g_hat.clone() for u in malicious}
        self.last_diagnostics = {
            "gamma": gamma,
            "threshold": threshold,
            "num_benign": len(benign),
            "num_malicious": len(malicious),
            "reference": source,
        }
        return replace_vectors(uplinks, new)


class MinMax(_CraftedUpdateAttack):
    """Min-Max (paper p. 2, p. 10).

    gamma = largest value with  max_i ||g_hat - b_i|| <= max_{i,j} ||b_i - b_j||,
    over the benign updates b_i of round t. ``last_diagnostics["gamma"]`` is logged.
    """

    name = "min_max"

    def _threshold(self, refs: torch.Tensor) -> float:
        """max_{i,j} ||b_i - b_j|| (p. 10); 0 for a single update (max over no pairs)."""
        return math.sqrt(float(pairwise_sq_dists(refs).max()))

    def _constraint(self, a: torch.Tensor, c: torch.Tensor, p2: float, gamma: float) -> float:
        """max_i ||g_b + gamma p - b_i|| (p. 10)."""
        return math.sqrt(max(float((a + 2.0 * gamma * c + gamma * gamma * p2).max()), 0.0))
