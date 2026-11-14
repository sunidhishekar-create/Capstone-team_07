"""Label flipping attack (data-level).

Base paper: arXiv 2601.06903, p. 2 (Section I-A: label l becomes L - l - 1, L = number of
classes) and p. 9 (Section VI-B: half of the malicious workers' local labels are reversed).
"""

from __future__ import annotations

from typing import Mapping

import torch

from fedrag.attacks.base import Attack
from fedrag.strategies.br_drag import Uplink, Vec


def flip_labels(
    labels: torch.Tensor, num_classes: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse half of ``labels`` with l -> L - l - 1 (paper p. 2 rule, p. 9 "half").

    Chooses floor(n/2) positions uniformly without replacement using ``generator`` and returns
    ``(new_labels, flipped_positions)``. Only the label tensor is copied, never image data.
    Note: if L is odd, the middle class l = (L-1)/2 maps to itself, so it is "flipped" but its
    value does not change (e.g. EMNIST balanced, L = 47, class 23).
    """
    if labels.dim() != 1:
        raise ValueError("labels must be a 1-D tensor")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError(f"labels must lie in [0, {num_classes})")
    n = labels.numel()
    pos = torch.randperm(n, generator=generator)[: n // 2].sort().values.to(labels.device)
    new = labels.clone()
    new[pos] = num_classes - 1 - labels[pos]
    return new, pos


class LabelFlip(Attack):
    """Label flipping (paper pp. 2, 9): data-level, poisons malicious clients' labels once.

    ``setup(clients, malicious_ids, generator)`` takes ``clients`` as a mapping
    client_id -> that client's 1-D label tensor (TEMP until core/client.py exists: the core will
    pass ``store_labels[client.indices]``). The chosen half is fixed at setup, so the same
    samples stay flipped every round. Results:
      - ``label_overrides[cid]``: the client's full new label tensor (for ``labels_override``)
      - ``flipped_positions[cid]``: local positions that were flipped
    ``corrupt`` is the identity: the poisoned client then trains and uploads normally.
    """

    name = "label_flip"
    level = "data"

    def __init__(self, num_classes: int) -> None:
        """Store L, the number of classes used in l -> L - l - 1 (paper p. 2)."""
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)
        self.label_overrides: dict[int, torch.Tensor] = {}
        self.flipped_positions: dict[int, torch.Tensor] = {}

    def setup(
        self,
        clients: Mapping[int, torch.Tensor],
        malicious_ids: set[int],
        generator: torch.Generator,
    ) -> None:
        """Flip half of every malicious client's labels once (paper p. 9).

        Clients are processed in sorted id order so the result depends only on the seed.
        Malicious ids absent from ``clients`` are ignored.
        """
        super().setup(clients, malicious_ids, generator)
        self.label_overrides, self.flipped_positions = {}, {}
        for cid in sorted(malicious_ids):
            if cid not in clients:
                continue
            new, pos = flip_labels(clients[cid], self.num_classes, generator)
            self.label_overrides[cid] = new
            self.flipped_positions[cid] = pos

    def corrupt(
        self,
        t: int,
        uplinks: list[Uplink],
        malicious_ids: set[int],
        own_g: dict[int, Vec],
    ) -> list[Uplink]:
        """Identity: label flipping acts on data at setup, not on uploads (paper p. 2)."""
        self.last_diagnostics = {}
        return list(uplinks)
