"""BR-DRAG: Byzantine-robust divergence-based adaptive aggregation.

Base paper: Xiao et al., "Divergence-Based Adaptive Aggregation for Byzantine Robust
Federated Learning", arXiv 2601.06903, Section IV, eqs. (12)-(16).

The server trains U SGD steps on its trusted root dataset to get a reference direction r^t
(eqs. (12)-(13), see ``root_trainer.py``), then corrects every raw client update g_m towards r^t
(eqs. (15)-(16)) and averages the corrected updates (eq. (14)).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from torch import nn

from fedrag.strategies.root_trainer import root_reference

Vec = torch.Tensor  # 1-D float32 tensor holding all model parameters

EPS = 1e-8


# ---------------------------------------------------------------------------
# TEMP: replace with core/messages.py once the Phase 0 contracts are merged.
# Minimal stand-ins matching architecture.md section 5.2.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Downlink:
    """TEMP: replace with core/messages.py. Server -> client message for round t."""

    round: int
    theta: Vec
    reference: Optional[Vec] = None
    extras: dict = field(default_factory=dict)


@dataclass
class Uplink:
    """TEMP: replace with core/messages.py. Client -> server message for round t."""

    round: int
    client_id: int
    vector: Vec
    kind: Literal["raw", "modified"]
    num_samples: int
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure functions (torch only)
# ---------------------------------------------------------------------------
def cosine(g: Vec, r: Vec, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """cos(g, r) as used in eq. (16), plus ||g|| and ||r||.

    Returns cos = 0 when ||g|| < eps or ||r|| < eps, and clamps cos to [-1, 1] so float
    round-off can never push lambda outside [0, 2 c_t].
    """
    g_norm = torch.linalg.vector_norm(g)
    r_norm = torch.linalg.vector_norm(r)
    ok = (g_norm >= eps) & (r_norm >= eps)
    denom = torch.where(ok, g_norm * r_norm, torch.ones_like(g_norm))
    cos = torch.where(ok, torch.dot(g, r) / denom, torch.zeros_like(g_norm))
    return cos.clamp(-1.0, 1.0), g_norm, r_norm


def divergence(g: Vec, r: Vec, c_t: float, eps: float = EPS) -> torch.Tensor:
    """Divergence weight lambda = c_t * (1 - cos(g, r)), eq. (16).

    Lies in [0, 2 c_t]. If ||g|| < eps or ||r|| < eps, cos is taken as 0 (lambda = c_t).
    Returns a 0-dim tensor on the same device as ``g``.
    """
    cos, _, _ = cosine(g, r, eps)
    return c_t * (1.0 - cos)


def _correct(g: Vec, r: Vec, c_t: float, eps: float) -> tuple[Vec, torch.Tensor, torch.Tensor]:
    """Eqs. (15)-(16) in one pass; returns (v, lambda, cos) so diagnostics need no recompute."""
    cos, g_norm, r_norm = cosine(g, r, eps)
    lam = c_t * (1.0 - cos)
    scale = r_norm / g_norm.clamp_min(eps)  # ||r|| / ||g||, eps-guarded
    v = (1.0 - lam) * scale * g + lam * r
    return v, lam, cos


def correct_update(g: Vec, r: Vec, c_t: float, eps: float = EPS) -> Vec:
    """BR-DRAG corrected update, eq. (15).

        v = (1 - lambda) * (||r|| / ||g||) * g + lambda * r,   lambda from eq. (16)

    The direction of g is rescaled to the norm of r, then mixed with r. For c_t <= 0.5
    (lambda <= 1), ||v|| <= ||r||. ``eps`` guards the division and the cosine, so the result
    is finite for g = 0 or r = 0 (g = 0 gives v = c_t * r).
    """
    v, _, _ = _correct(g, r, c_t, eps)
    return v


def aggregate(vs: list[Vec]) -> Vec:
    """Unweighted mean Delta^t = (1/S) * sum_m v_m, eq. (14)."""
    if len(vs) == 0:
        raise ValueError("cannot aggregate an empty list of vectors")
    return torch.stack(list(vs)).mean(dim=0)


def apply(theta: Vec, delta: Vec) -> Vec:
    """Global model update theta^(t+1) = theta^t + Delta^t, eq. (14). Returns a new tensor."""
    return theta + delta


def sync_device(device: torch.device) -> None:
    """Wait for queued CUDA work so wall-clock timers measure the real cost."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class BRDRAG:
    """BR-DRAG strategy (Section IV): server-side correction against a root-data reference.

    Hooks (architecture.md 5.3): the client hook is the identity (clients upload raw g),
    the correction happens in ``server_transform``.

    TEMP: until core/strategy.py exists this class does not subclass ``Strategy`` and takes its
    dependencies (model workspace, root data, root RNG stream) in ``__init__`` instead of
    ``setup(ctx)``.
    """

    name = "br_drag"

    def __init__(
        self,
        model: nn.Module,
        root_data: tuple[torch.Tensor, torch.Tensor],
        generator: torch.Generator,
        *,
        c_t: float = 0.5,
        steps: int = 5,
        batch_size: int = 10,
        lr: float = 0.01,
        eps: float = EPS,
    ) -> None:
        """Store BR-DRAG settings. Defaults: c_t = 0.5, U = 5, B = 10, eta = 0.01 (Section V)."""
        if c_t < 0:
            raise ValueError("c_t must be >= 0")
        self.model = model
        self.root_data = root_data
        self.generator = generator
        self.c_t = float(c_t)
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.eps = float(eps)
        # Per-round scratch only (not cross-round state): (round, seconds) of the last
        # prepare_downlink, reported in that round's server_transform diagnostics.
        self._prepare_timing: Optional[tuple[int, float]] = None

    # ---- server side -----------------------------------------------------
    def needs_warmup(self, t: int) -> bool:
        """BR-DRAG has no warm-up: r^t comes from the root dataset every round (eqs. (12)-(13))."""
        return False

    def prepare_downlink(self, t: int, theta: Vec) -> Downlink:
        """Train U root SGD steps from theta^t to get r^t (eqs. (12)-(13)) and send it with theta.

        The paper broadcasts r^t to clients (Section IV-A), so it is kept in ``reference``.
        """
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
        """Correct every uploaded g_m with eqs. (15)-(16) against r^t = ``dl.reference``.

        Returns the corrected vectors v_m and diagnostics: lambda mean/min/max, mean cos(g, r),
        ||r||, mean ||g||, rho = ||r|| / ||g|| min/max, and the wall-clock seconds of this
        round's prepare_downlink and server_transform.
        """
        if dl.reference is None:
            raise ValueError("BR-DRAG needs dl.reference (r^t); call prepare_downlink first")
        if len(uplinks) == 0:
            raise ValueError("server_transform got no uplinks")
        r = dl.reference
        sync_device(r.device)
        start = time.perf_counter()

        vs, lams, coss, g_norms = [], [], [], []
        for u in uplinks:
            v, lam, cos = _correct(u.vector, r, self.c_t, self.eps)
            vs.append(v)
            lams.append(lam)
            coss.append(cos)
            g_norms.append(torch.linalg.vector_norm(u.vector))

        lam_t = torch.stack(lams)
        g_norm_t = torch.stack(g_norms)
        r_norm = torch.linalg.vector_norm(r)
        rho = r_norm / g_norm_t.clamp_min(self.eps)
        diag = {
            "lambda_mean": lam_t.mean().item(),
            "lambda_min": lam_t.min().item(),
            "lambda_max": lam_t.max().item(),
            "cos_g_r_mean": torch.stack(coss).mean().item(),
            "r_norm": r_norm.item(),
            "g_norm_mean": g_norm_t.mean().item(),
            "rho_min": rho.min().item(),
            "rho_max": rho.max().item(),
        }
        sync_device(r.device)
        diag["time_server_transform_s"] = time.perf_counter() - start
        prep = self._prepare_timing
        diag["time_prepare_downlink_s"] = prep[1] if prep is not None and prep[0] == t else float("nan")
        return vs, diag

    def aggregate(self, t: int, vectors: list[Vec]) -> Vec:
        """Unweighted mean of the corrected updates, eq. (14)."""
        return aggregate(vectors)

    def apply(self, t: int, theta: Vec, delta: Vec) -> Vec:
        """theta^(t+1) = theta^t + Delta^t, eq. (14). No strategy state to update."""
        return apply(theta, delta)

    # ---- client side -----------------------------------------------------
    def client_postprocess(self, t: int, g: Vec, dl: Downlink) -> tuple[Vec, str, dict]:
        """Identity: BR-DRAG clients upload raw g; correction is server-side (Section IV)."""
        return g, "raw", {}

    # ---- checkpointing ---------------------------------------------------
    def state_dict(self) -> dict:
        """BR-DRAG keeps no cross-round state: r^t is recomputed from theta^t every round.

        The root minibatch RNG is the run's ``root`` stream, which RNGHub owns and checkpoints
        (architecture.md 5.10), so it is deliberately not saved here.
        """
        return {}

    def load_state_dict(self, sd: dict) -> None:
        """Accept only the empty state produced by ``state_dict`` (no cross-round state)."""
        if sd:
            raise ValueError(f"BR-DRAG has no state to load, got keys {sorted(sd)}")
