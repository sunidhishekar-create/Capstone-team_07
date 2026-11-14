"""BR-DRAG root trainer: the server's reference direction r^t from D_root.

Base paper: Xiao et al., arXiv 2601.06903, Section IV-A, eqs. (12)-(13).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

Vec = torch.Tensor  # 1-D float32 tensor holding all model parameters


# TEMP: replace with core/vec.py (get_params / load_params) once Phase 0 lands.
def get_params(model: nn.Module) -> Vec:
    """Flatten all model parameters into one detached float32 Vec (theta in eqs. (12)-(13)).

    Order is ``model.parameters()`` order. BatchNorm buffers are not included yet
    (architecture.md 5.1: only WRN-28-4 needs them, and that is a stretch goal).
    """
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).to(torch.float32)


# TEMP: replace with core/vec.py (get_params / load_params) once Phase 0 lands.
def load_params(model: nn.Module, vec: Vec) -> None:
    """Copy a flat Vec into the model's parameters in place (sets theta^(t,0) = theta^t, eq. (12)).

    Uses ``copy_`` so the model never aliases ``vec``. (``torch.nn.utils.vector_to_parameters``
    aliases its input, so later in-place SGD steps would silently overwrite the caller's theta.)
    """
    n = sum(p.numel() for p in model.parameters())
    if vec.dim() != 1 or vec.numel() != n:
        raise ValueError(f"expected a 1-D vector of {n} elements, got shape {tuple(vec.shape)}")
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            k = p.numel()
            p.copy_(vec[offset : offset + k].view_as(p))
            offset += k


def root_reference(
    model: nn.Module,
    theta: Vec,
    root_data: tuple[torch.Tensor, torch.Tensor],
    steps: int,
    batch_size: int,
    lr: float,
    generator: torch.Generator,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy,
) -> Vec:
    """Compute the BR-DRAG reference direction r^t = theta^(t,U) - theta^t (eqs. (12)-(13)).

    Loads ``theta`` into ``model``, runs ``steps`` (= U) plain-SGD steps (no momentum, no weight
    decay) on minibatches of size ``batch_size`` drawn independently for every step from
    ``root_data = (X, y)`` using ``generator``, and returns the parameter difference.

    Each minibatch is sampled without replacement within the batch; batches are independent of
    one another. ``generator`` must be a CPU generator (BR-DRAG's ``root`` RNG stream).
    The caller's ``theta`` is never modified. ``model`` is left holding theta^(t,U).
    """
    x, y = root_data
    n = x.shape[0]
    if n == 0:
        raise ValueError("root_data is empty")
    if steps < 0 or batch_size < 1:
        raise ValueError("steps must be >= 0 and batch_size >= 1")
    b = min(batch_size, n)

    theta0 = theta.detach().clone()  # private copy; caller's tensor is only read here
    load_params(model, theta0)
    model.train()
    params = list(model.parameters())

    for _ in range(steps):
        idx = torch.randperm(n, generator=generator)[:b].to(x.device)
        for p in params:
            p.grad = None
        loss = loss_fn(model(x[idx]), y[idx])
        loss.backward()
        with torch.no_grad():  # plain SGD, eq. (12): theta <- theta - eta * grad
            for p in params:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)

    for p in params:
        p.grad = None
    return get_params(model) - theta0  # eq. (13)
