"""Non-IID client split with a per-class Dirichlet distribution.

Base paper: Xiao et al., arXiv 2601.06903, Section VI: "we utilize the Dirichlet distribution
with parameter beta" to "allocate a proportion p_{k,j} of image class k instances to device j";
beta = 0.1 (strong) and 0.5 (moderate heterogeneity), M = 40 workers.

The root set is carved first (``fedrag.data.root``) and passed as ``exclude``, so no client ever
holds a root sample.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


def numpy_rng_from(generator: torch.Generator) -> np.random.Generator:
    """A numpy Generator seeded from (and advancing) a torch Generator.

    torch has no seeded Dirichlet sampler, so we draw one 63-bit seed from the torch stream.
    Deterministic for a given generator state.
    """
    seed = int(torch.randint(0, 2**62, (1,), generator=generator).item())
    return np.random.default_rng(seed)


def dirichlet_partition(
    labels: torch.Tensor,
    num_clients: int = 40,
    beta: float = 0.5,
    generator: torch.Generator | None = None,
    exclude: torch.Tensor | None = None,
    min_size: int = 10,
    max_retries: int = 1000,
) -> list[torch.Tensor]:
    """Split the non-excluded train indices across ``num_clients`` clients.

    For each class k: p ~ Dirichlet(beta * ones(num_clients)); the class's shuffled indices
    are cut at round(cumsum(p) * n_k). The whole draw is repeated (up to ``max_retries`` times)
    until every client holds >= ``min_size`` samples (the local batch size must fit).

    Returns one sorted int64 CPU index tensor per client. Every non-excluded index appears in
    exactly one client. Logs per-client sample counts and class counts.
    """
    if num_clients < 1 or beta <= 0:
        raise ValueError("num_clients must be >= 1 and beta > 0")
    generator = generator if generator is not None else torch.Generator().manual_seed(0)
    rng = numpy_rng_from(generator)

    y = labels.detach().cpu().numpy()
    keep = np.ones(y.shape[0], dtype=bool)
    if exclude is not None:
        keep[exclude.detach().cpu().numpy()] = False
    classes = np.unique(y[keep])
    by_class = [np.flatnonzero(keep & (y == c)) for c in classes]
    if keep.sum() < num_clients * min_size:
        raise ValueError(f"{keep.sum()} samples cannot give {num_clients} clients >= {min_size} each")

    for attempt in range(1, max_retries + 1):
        shards: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
        for idx in by_class:
            idx = rng.permutation(idx)
            p = rng.dirichlet(np.full(num_clients, beta))
            cuts = np.round(np.cumsum(p)[:-1] * idx.size).astype(int)
            for j, part in enumerate(np.split(idx, cuts)):
                shards[j].append(part)
        sizes = [sum(part.size for part in s) for s in shards]
        if min(sizes) >= min_size:
            break
    else:
        raise RuntimeError(
            f"no Dirichlet(beta={beta}) split gave every client >= {min_size} samples "
            f"after {max_retries} draws; lower min_size or raise beta"
        )

    parts = [torch.from_numpy(np.sort(np.concatenate(s))).long() for s in shards]
    _log_stats(labels, parts, int(classes.max()) + 1 if classes.size else 0, attempt)
    return parts


def class_counts(labels: torch.Tensor, parts: list[torch.Tensor], num_classes: int) -> torch.Tensor:
    """(num_clients x num_classes) int64 matrix of per-client label counts."""
    y = labels.detach().cpu()
    return torch.stack([torch.bincount(y[p], minlength=num_classes) for p in parts])


def _log_stats(labels: torch.Tensor, parts: list[torch.Tensor], num_classes: int, attempts: int) -> None:
    """Log per-client sample counts and class counts (INFO; full matrix at DEBUG)."""
    counts = class_counts(labels, parts, num_classes)
    sizes = counts.sum(dim=1)
    log.info("dirichlet partition: %d clients after %d draw(s); samples min=%d median=%d max=%d",
             len(parts), attempts, int(sizes.min()), int(sizes.median()), int(sizes.max()))
    for j, row in enumerate(counts.tolist()):
        present = {c: n for c, n in enumerate(row) if n}
        log.info("  client %2d: %5d samples, %3d classes", j, sum(row), len(present))
        log.debug("  client %2d class counts: %s", j, present)
