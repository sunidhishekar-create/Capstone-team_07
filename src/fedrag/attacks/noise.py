"""Noise injection attack.

Base paper: arXiv 2601.06903, p. 1 (Section I-A: "upload random vectors, e.g. sampled from a
Gaussian distribution") and p. 9 (Section VI-B: malicious gradient p_m^t g_m^t, p_m^t ~ N(0, 3)).
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from fedrag.attacks.base import Attack, own_update, replace_vectors, split_uplinks
from fedrag.strategies.br_drag import Uplink, Vec


class NoiseInjection(Attack):
    """Noise injection (paper p. 9): malicious upload = p * g, p ~ N(0, variance).

    - ``variance``: 3.0 by default; N(0, 3) is read as variance 3, std sqrt(3)
      (architecture.md 15 #2).
    - ``mode="multiply"`` (default, p. 9): upload p * g, g = the attacker's own honest update.
      ``mode="replace"`` (p. 1): upload a pure Gaussian vector N(0, variance * I) instead of g.
    - ``granularity="scalar"`` (default): one p per malicious client per round.
      ``"elementwise"``: an independent p for every coordinate. Ignored in replace mode, which is
      always elementwise.

    Draws use the ``setup`` generator on CPU, in uplink order, so a seed fixes every p.
    ``last_diagnostics["p"]`` maps client id -> scalar p (scalar multiply mode only).
    """

    name = "noise"
    level = "update"

    def __init__(
        self,
        variance: float = 3.0,
        mode: Literal["multiply", "replace"] = "multiply",
        granularity: Literal["scalar", "elementwise"] = "scalar",
    ) -> None:
        """Store N(0, variance) settings (paper p. 9 uses variance 3)."""
        super().__init__()
        if variance < 0:
            raise ValueError("variance must be >= 0")
        if mode not in ("multiply", "replace"):
            raise ValueError(f"unknown mode {mode!r}")
        if granularity not in ("scalar", "elementwise"):
            raise ValueError(f"unknown granularity {granularity!r}")
        self.variance = float(variance)
        self.std = math.sqrt(self.variance)
        self.mode = mode
        self.granularity = granularity

    def corrupt(
        self,
        t: int,
        uplinks: list[Uplink],
        malicious_ids: set[int],
        own_g: dict[int, Vec],
    ) -> list[Uplink]:
        """Replace each malicious uplink with p * g (p. 9) or a Gaussian vector (p. 1)."""
        gen = self._require_generator()
        _, malicious = split_uplinks(uplinks, malicious_ids)
        new: dict[int, Vec] = {}
        p_scalars: dict[int, float] = {}
        for u in malicious:
            g = own_update(own_g, u.client_id)
            if self.mode == "replace":
                noise = torch.randn(g.numel(), generator=gen, dtype=torch.float32) * self.std
                new[u.client_id] = noise.to(g.device).reshape(g.shape)
            elif self.granularity == "scalar":
                p = torch.randn(1, generator=gen, dtype=torch.float32).item() * self.std
                p_scalars[u.client_id] = p
                new[u.client_id] = p * g
            else:
                p_vec = torch.randn(g.numel(), generator=gen, dtype=torch.float32) * self.std
                new[u.client_id] = p_vec.to(g.device).reshape(g.shape) * g
        self.last_diagnostics = {"num_malicious": len(malicious), "p": p_scalars}
        return replace_vectors(uplinks, new)
