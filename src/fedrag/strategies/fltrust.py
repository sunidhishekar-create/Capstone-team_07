"""FLTrust: Byzantine-robust baseline with root-data trust scores.

Two variants, selected with ``variant``:

- ``"paper"`` (default): exactly as the base paper's baseline bullet, Xiao et al.,
  arXiv 2601.06903, Section VI-B, p. 9 ("FLTrust [30]"):

      g~_m = max(0, cos(g_m, r)) * ||g_m|| * r / ||r||

  i.e. each upload is replaced by the reference direction, scaled by trust * ||g_m||. The bullet
  only says the gradients are "modified"; the paper's baselines otherwise aggregate with eq. (4),
  theta^(t+1) = theta^t + (1/S) sum_m g~_m, so we use that unweighted mean.

- ``"cao"``: the original algorithm, Cao et al., "FLTrust: Byzantine-robust Federated Learning
  via Trust Bootstrapping", NDSS 2021:

      trust_m = ReLU(cos(g_m, r)),   g^_m = (||r|| / ||g_m||) g_m,
      delta   = sum_m trust_m g^_m / sum_m trust_m          (delta = 0 if sum_m trust_m = 0)

In both, r = theta^(t,U) - theta^t comes from the same root trainer as BR-DRAG ("designed in a
similar way to (13)", p. 9), and theta^(t+1) = theta^t + delta.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

import torch
from torch import nn

# TEMP: Downlink/Uplink are the stand-ins in br_drag.py until core/messages.py exists.
from fedrag.strategies.br_drag import (
    EPS,
    Downlink,
    Uplink,
    Vec,
    aggregate,
    apply,
    cosine,
    sync_device,
)
from fedrag.strategies.root_trainer import root_reference

Variant = Literal["paper", "cao"]


def trust_score(g: Vec, r: Vec, eps: float = EPS) -> torch.Tensor:
    """trust = ReLU(cos(g, r)) = max(0, cos(g, r)) (base paper p. 9; Cao et al. NDSS 2021).

    cos is taken as 0 when ||g|| < eps or ||r|| < eps (same guard as BR-DRAG), so trust is 0.
    """
    cos, _, _ = cosine(g, r, eps)
    return cos.clamp_min(0.0)


def modify_paper(g: Vec, r: Vec, eps: float = EPS) -> tuple[Vec, torch.Tensor]:
    """Base paper p. 9: g~ = max(0, cos(g, r)) * ||g|| * r / ||r||. Returns (g~, trust)."""
    cos, g_norm, r_norm = cosine(g, r, eps)
    trust = cos.clamp_min(0.0)
    return trust * g_norm * (r / r_norm.clamp_min(eps)), trust


def rescale_cao(g: Vec, r: Vec, eps: float = EPS) -> tuple[Vec, torch.Tensor]:
    """Cao et al. (NDSS 2021): g^ = (||r|| / ||g||) g, eps-guarded. Returns (g^, trust)."""
    cos, g_norm, r_norm = cosine(g, r, eps)
    return (r_norm / g_norm.clamp_min(eps)) * g, cos.clamp_min(0.0)


class FLTrust:
    """FLTrust baseline (base paper p. 9; Cao et al. NDSS 2021), same hooks as ``BRDRAG``.

    ``server_transform`` returns vectors whose unweighted mean (``aggregate``) is delta:
    - ``"paper"``: the g~_m themselves, so delta = (1/S) sum g~_m (eq. (4)).
    - ``"cao"``: S * trust_m * g^_m / sum trust, so the mean is the trust-weighted average; all
      zeros when every trust is 0.

    TEMP: like BRDRAG, takes the model workspace, root data and root RNG stream in ``__init__``
    until core/strategy.py provides ``setup(ctx)``.
    """

    name = "fltrust"

    def __init__(
        self,
        model: nn.Module,
        root_data: tuple[torch.Tensor, torch.Tensor],
        generator: torch.Generator,
        *,
        variant: Variant = "paper",
        steps: int = 5,
        batch_size: int = 10,
        lr: float = 0.01,
        eps: float = EPS,
    ) -> None:
        """Store root-trainer settings (same defaults as BR-DRAG: U = 5, B = 10, eta = 0.01)."""
        if variant not in ("paper", "cao"):
            raise ValueError(f"unknown FLTrust variant {variant!r}")
        self.model = model
        self.root_data = root_data
        self.generator = generator
        self.variant: Variant = variant
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.eps = float(eps)
        self._prepare_timing: Optional[tuple[int, float]] = None  # per-round scratch only

    # ---- server side -----------------------------------------------------
    def needs_warmup(self, t: int) -> bool:
        """No warm-up: r^t comes from the root dataset every round (p. 9, as in eq. (13))."""
        return False

    def prepare_downlink(self, t: int, theta: Vec) -> Downlink:
        """Compute r^t with the shared BR-DRAG root trainer (p. 9: "similar way to (13)")."""
        sync_device(theta.device)
        start = time.perf_counter()
        r = root_reference(
            self.model,
            theta,
            self.root_data,
            steps=self.steps,
            batch_size=self.batch_size,
            lr=self.lr,
            generator=self.generator,
        )
        sync_device(theta.device)
        self._prepare_timing = (t, time.perf_counter() - start)
        return Downlink(round=t, theta=theta, reference=r)

    def server_transform(
        self, t: int, uplinks: list[Uplink], dl: Downlink
    ) -> tuple[list[Vec], dict]:
        """Apply trust scores to every upload (base paper p. 9 / Cao et al.).

        Diagnostics: trust mean/min/max, number of zero-trust clients, whether all trust was
        zero (delta = 0, no update), ||r||, mean ||g||, and the timings of this round's
        prepare_downlink and server_transform.
        """
        if dl.reference is None:
            raise ValueError("FLTrust needs dl.reference (r^t); call prepare_downlink first")
        if len(uplinks) == 0:
            raise ValueError("server_transform got no uplinks")
        r = dl.reference
        sync_device(r.device)
        start = time.perf_counter()

        modify = modify_paper if self.variant == "paper" else rescale_cao
        mods, trusts = [], []
        for u in uplinks:
            m, trust = modify(u.vector, r, self.eps)
            mods.append(m)
            trusts.append(trust)
        trust_t = torch.stack(trusts)
        total = trust_t.sum()
        all_zero = bool(total <= 0)

        if self.variant == "paper":
            vecs = mods
        elif all_zero:
            vecs = [torch.zeros_like(r) for _ in uplinks]
        else:
            weight = len(uplinks) / total  # so that mean(vecs) = sum(trust * g^) / sum(trust)
            vecs = [(w * weight) * m for w, m in zip(trusts, mods)]

        g_norm_mean = torch.stack([torch.linalg.vector_norm(u.vector) for u in uplinks]).mean()
        diag = {
            "trust_mean": trust_t.mean().item(),
            "trust_min": trust_t.min().item(),
            "trust_max": trust_t.max().item(),
            "num_zero_trust": int((trust_t <= 0).sum().item()),
            "all_zero_trust": float(all_zero),
            "r_norm": torch.linalg.vector_norm(r).item(),
            "g_norm_mean": g_norm_mean.item(),
        }
        sync_device(r.device)
        diag["time_server_transform_s"] = time.perf_counter() - start
        prep = self._prepare_timing
        diag["time_prepare_downlink_s"] = prep[1] if prep is not None and prep[0] == t else float("nan")
        return vecs, diag

    def aggregate(self, t: int, vectors: list[Vec]) -> Vec:
        """Unweighted mean of the transformed vectors: eq. (4) / Cao's weighted average (p. 9)."""
        return aggregate(vectors)

    def apply(self, t: int, theta: Vec, delta: Vec) -> Vec:
        """theta^(t+1) = theta^t + delta (eq. (4)). No strategy state to update."""
        return apply(theta, delta)

    # ---- client side -----------------------------------------------------
    def client_postprocess(self, t: int, g: Vec, dl: Downlink) -> tuple[Vec, str, dict]:
        """Identity: FLTrust clients upload raw g; trust is applied on the server (p. 9)."""
        return g, "raw", {}

    # ---- checkpointing ---------------------------------------------------
    def state_dict(self) -> dict:
        """No cross-round state: r^t is recomputed every round; the root RNG stream is owned and
        checkpointed by RNGHub (architecture.md 5.10)."""
        return {}

    def load_state_dict(self, sd: dict) -> None:
        """Accept only the empty state produced by ``state_dict``."""
        if sd:
            raise ValueError(f"FLTrust has no state to load, got keys {sorted(sd)}")
