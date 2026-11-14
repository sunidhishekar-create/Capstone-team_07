"""Min-Sum attack.

Base paper: arXiv 2601.06903, p. 2 (Section I-A) and pp. 9-10 (Section VI-B). Same crafted
update as Min-Max, g_hat = g_b + gamma * p with p = -g_b / ||g_b||, but gamma is the largest
value keeping the total distance from g_hat to all benign updates within the largest such total
for any single benign update (p. 10).

"Total distance" (p. 10) is read as the sum of SQUARED L2 distances, as in the attack's source,
Shejwalkar & Houmansadr (NDSS 2021, paper ref. [26]). ``squared=False`` uses plain distances.
"""

from __future__ import annotations

import torch

from fedrag.attacks.min_max import _CraftedUpdateAttack, pairwise_sq_dists


class MinSum(_CraftedUpdateAttack):
    """Min-Sum (paper p. 2, p. 10).

    gamma = largest value with  sum_i ||g_hat - b_i||^2 <= max_k sum_i ||b_k - b_i||^2
    over the benign updates b_i of round t (``squared=True``, default). With a single benign
    update the right-hand side is 0, so gamma = 0. ``last_diagnostics["gamma"]`` is logged.
    """

    name = "min_sum"

    def __init__(self, squared: bool = True) -> None:
        """``squared`` selects squared (ref. [26], default) or plain L2 distances (p. 10)."""
        super().__init__()
        self.squared = bool(squared)

    def _threshold(self, refs: torch.Tensor) -> float:
        """max_k sum_i ||b_k - b_i||^2 (or plain distances), paper p. 10."""
        d2 = pairwise_sq_dists(refs)
        d = d2 if self.squared else d2.sqrt()
        return float(d.sum(dim=1).max())

    def _constraint(self, a: torch.Tensor, c: torch.Tensor, p2: float, gamma: float) -> float:
        """sum_i ||g_b + gamma p - b_i||^2 (or plain distances), paper p. 10."""
        sq = (a + 2.0 * gamma * c + gamma * gamma * p2).clamp_min(0.0)
        return float(sq.sum()) if self.squared else float(sq.sqrt().sum())

