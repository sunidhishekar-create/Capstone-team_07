"""Client side: U local SGD steps (Eq. 2) and the update g = theta^{t,U} - theta^t."""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def param_vector(model) -> torch.Tensor:
    return parameters_to_vector(model.parameters()).detach().clone()


def load_param_vector(model, vec: torch.Tensor) -> None:
    vector_to_parameters(vec.detach().clone(), model.parameters())


def buffer_dict(model) -> Dict[str, torch.Tensor]:
    """Non-parameter state (BatchNorm running stats; empty for the plain CNNs)."""
    pnames = {n for n, _ in model.named_parameters()}
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k not in pnames}


def local_update(model, global_vec, global_buffers, x, y, idx, lr, steps, batch_size,
                 seed, device) -> Tuple[torch.Tensor, float, Dict[str, torch.Tensor]]:
    """Run `steps` mini-batch SGD steps from the global model on worker data x[idx], y[idx].

    Returns (g, mean_train_loss, buffers) with g = theta^{t,U} - theta^t (flat, on `device`).
    Mini-batches are drawn without replacement within a step (size B) from the local dataset.
    """
    load_param_vector(model, global_vec)
    if global_buffers:
        model.load_state_dict(global_buffers, strict=False)
    model.train()
    gen = torch.Generator().manual_seed(int(seed) % (2 ** 63))
    n, b = len(idx), min(batch_size, len(idx))
    idx_t = torch.as_tensor(idx)
    losses = []
    for _ in range(steps):
        sel = idx_t[torch.randperm(n, generator=gen)[:b]]
        xb, yb = x[sel].to(device), y[sel].to(device)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)       # plain SGD, no momentum / weight decay
        losses.append(loss.item())
    g = param_vector(model) - global_vec
    return g, sum(losses) / len(losses), buffer_dict(model)
