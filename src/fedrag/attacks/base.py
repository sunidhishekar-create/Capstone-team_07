"""Attack interface (architecture.md 5.5) and shared helpers.

Base paper: Xiao et al., arXiv 2601.06903. Attacks are listed in Section I-A (pp. 1-2) and
configured in Section VI-B (pp. 9-10).

Update-level attacks run in the simulator's attack stage: they see every uplink selected in
round t (needed by the omniscient Min-Max and Min-Sum) plus each malicious client's own honest
update. Data-level attacks (label flipping) poison client labels once in ``setup``.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

import torch

# TEMP: replace with core/messages.py once the Phase 0 contracts are merged.
from fedrag.strategies.br_drag import Uplink, Vec


class Attack(ABC):
    """Byzantine attack (paper Section I-A, pp. 1-2; settings Section VI-B, p. 9).

    ``last_diagnostics`` holds scalars from the most recent ``corrupt`` call (e.g. Min-Max's
    gamma) for the metrics logger. It is per-round scratch, not checkpointed state.
    """

    name: str = "attack"
    level: Literal["data", "update"] = "update"

    def __init__(self) -> None:
        """Initialise with no generator; ``setup`` provides it (paper p. 9 attacks are random)."""
        self.generator: Optional[torch.Generator] = None
        self.last_diagnostics: dict[str, Any] = {}

    def setup(self, clients: Any, malicious_ids: set[int], generator: torch.Generator) -> None:
        """Called once before training with the run's ``attack`` RNG stream (paper p. 9).

        Data-level attacks change client labels here. The default only stores the generator.
        """
        self.generator = generator

    @abstractmethod
    def corrupt(
        self,
        t: int,
        uplinks: list[Uplink],
        malicious_ids: set[int],
        own_g: dict[int, Vec],
    ) -> list[Uplink]:
        """Return round t's uplinks with malicious ones replaced (paper pp. 1-2, 9).

        Must not modify any input uplink or tensor in place; benign uplinks are returned as the
        same objects, unchanged.
        """

    def _require_generator(self) -> torch.Generator:
        """Return the generator from ``setup``; random attacks (p. 9) cannot run without it."""
        if self.generator is None:
            raise RuntimeError(f"{type(self).__name__}.setup() must be called before corrupt()")
        return self.generator


def split_uplinks(
    uplinks: list[Uplink], malicious_ids: set[int]
) -> tuple[list[Uplink], list[Uplink]]:
    """Split round t's uplinks into (benign B^t, malicious A^t), keeping order (paper p. 9)."""
    benign = [u for u in uplinks if u.client_id not in malicious_ids]
    malicious = [u for u in uplinks if u.client_id in malicious_ids]
    return benign, malicious


def own_update(own_g: dict[int, Vec], client_id: int) -> Vec:
    """Look up malicious client m's own honest update g_m^t (paper p. 9) with a clear error."""
    try:
        return own_g[client_id]
    except KeyError:
        raise KeyError(f"own_g has no honest update for malicious client {client_id}") from None


def replace_vectors(uplinks: list[Uplink], new_vectors: dict[int, Vec]) -> list[Uplink]:
    """Copy of ``uplinks`` where client ids in ``new_vectors`` get a new vector (paper p. 9).

    Replaced uplinks are new objects (round, kind, num_samples and meta kept, so the attacker
    looks protocol-compliant); all other uplinks are passed through as the same objects.
    """
    return [
        dataclasses.replace(u, vector=new_vectors[u.client_id]) if u.client_id in new_vectors else u
        for u in uplinks
    ]
