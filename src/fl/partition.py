"""Dirichlet non-IID partitioning (paper Sec. VI): for each class k sample p_k ~ Dir(beta) over the
M workers and give worker j a fraction p_{k,j} of the class-k training samples.

Every training sample is assigned to exactly one worker (no loss, no duplication); the test set is
never touched. ASSUMPTION (not in paper): resample the whole partition until every worker has at
least `min_size` samples, so that each worker can form a mini-batch.
"""
from typing import Dict, List

import numpy as np


def dirichlet_partition(labels: np.ndarray, num_clients: int, beta: float, seed: int,
                        min_size: int = 10, max_tries: int = 10000) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_tries):
        buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for k in classes:
            idx = np.where(labels == k)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(np.full(num_clients, beta))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for j, part in enumerate(np.split(idx, cuts)):
                buckets[j].extend(part.tolist())
        if min(len(b) for b in buckets) >= min_size:
            return [np.array(sorted(b), dtype=np.int64) for b in buckets]
    raise RuntimeError(f"No Dirichlet(beta={beta}) partition with min client size {min_size} after {max_tries} tries.")


def partition_summary(parts: List[np.ndarray], labels: np.ndarray, num_classes: int) -> Dict:
    return {"sizes": [int(len(p)) for p in parts],
            "class_counts": [np.bincount(labels[p], minlength=num_classes).tolist() for p in parts]}
