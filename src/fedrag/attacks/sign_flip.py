"""Sign flipping attack.

Base paper: arXiv 2601.06903, p. 1 (Section I-A) and p. 9 (Section VI-B: malicious workers
generate -g_m^t as the local gradient).
"""

from __future__ import annotations

from fedrag.attacks.base import Attack, own_update, replace_vectors, split_uplinks
from fedrag.strategies.br_drag import Uplink, Vec


class SignFlip(Attack):
    """Sign flipping (paper p. 9): each malicious client uploads -g, its own update negated."""

    name = "sign_flip"
    level = "update"

    def corrupt(
        self,
        t: int,
        uplinks: list[Uplink],
        malicious_ids: set[int],
        own_g: dict[int, Vec],
    ) -> list[Uplink]:
        """Replace each malicious uplink with -g_m^t (paper p. 9); deterministic."""
        _, malicious = split_uplinks(uplinks, malicious_ids)
        new = {u.client_id: -own_update(own_g, u.client_id) for u in malicious}
        self.last_diagnostics = {"num_malicious": len(malicious)}
        return replace_vectors(uplinks, new)
