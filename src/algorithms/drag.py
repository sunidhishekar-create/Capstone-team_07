"""Original DRAG operations (arXiv 2601.06903, Sec. III, Algorithm 1).

All vectors are flat 1-D tensors of the model parameters. A client update is
g_m^t = theta_m^{t,U} - theta^t (parameter difference after U local SGD steps).

ASSUMPTIONS (the paper is silent on degenerate norms):
  * ||r|| <= EPS  -> the DoD is undefined; we use lambda = 0 so v = g (no dragging).
  * ||g|| <= EPS  -> cosine undefined; lambda = 0 and v = g (= 0 up to EPS).
  * cos is clamped to [-1, 1] against float round-off so lambda stays in [0, 2c].
"""
from typing import List, Optional, Sequence, Tuple

import torch

EPS = 1e-12


def divergence_of_directions(g: torch.Tensor, r: torch.Tensor, c: float) -> torch.Tensor:
    """Eq. (10): lambda = c * (1 - <g, r> / (||g|| ||r||)) in [0, 2c]."""
    ng, nr = g.norm(), r.norm()
    if ng <= EPS or nr <= EPS:
        return g.new_zeros(())
    cos = torch.clamp(torch.dot(g, r) / (ng * nr), -1.0, 1.0)
    return c * (1.0 - cos)


def drag_modify(g: torch.Tensor, r: torch.Tensor, c: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. (11): v = (1 - lambda) g + lambda * (||g|| / ||r||) * r. Returns (v, lambda)."""
    lam = divergence_of_directions(g, r, c)
    nr = r.norm()
    if nr <= EPS:
        return g.clone(), lam
    v = (1.0 - lam) * g + lam * (g.norm() / nr) * r
    return v, lam


def aggregate(v_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Eq. (6): Delta^t = (1/S) sum_m v_m^t."""
    return torch.stack(list(v_list)).mean(dim=0)


class ReferenceDirection:
    """Maintains r^t across rounds (Eq. 5a / 5b). State = (r, last Delta); never reset per round."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.r: Optional[torch.Tensor] = None       # r^t used in the current/most recent round
        self.delta: Optional[torch.Tensor] = None   # Delta^{t-1} (aggregated modified update of the last round)

    def initialize(self, init_updates: List[torch.Tensor]) -> torch.Tensor:
        """Eq. (5a), t = 0: r^0 = (1/S) sum_m g_m^0 (from the initial U-step pass of Alg. 1, L5-8)."""
        self.r = aggregate(init_updates)
        return self.r

    def advance(self) -> torch.Tensor:
        """Eq. (5b), t >= 1: r^t = (1 - alpha) r^{t-1} + alpha Delta^{t-1}."""
        if self.r is None or self.delta is None:
            raise RuntimeError("ReferenceDirection.advance() called before round 0 completed.")
        self.r = (1.0 - self.alpha) * self.r + self.alpha * self.delta
        return self.r

    def record_delta(self, delta: torch.Tensor) -> None:
        """Retain Delta^t for the next round (Alg. 1, L18)."""
        self.delta = delta.clone()

    def state_dict(self) -> dict:
        return {"r": self.r, "delta": self.delta}

    def load_state_dict(self, s: dict) -> None:
        self.r, self.delta = s["r"], s["delta"]
