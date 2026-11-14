"""Root dataset D_root for BR-DRAG / FLTrust, carved from the train split BEFORE partitioning.

Base paper: Xiao et al., arXiv 2601.06903.
- Section IV-B, p. 5: D_root is small (3,000 samples by default) and "drawn from a vetted pool
  with de-duplication and outlier filtering".
- Section VI-B, p. 10: "A vetted pool D_vet is first constructed by applying a verification tool
  based on confident learning [...]. Then, data samples are randomly and uniformly drawn from
  D_vet in equal proportions to form D_root."

``build_root_set`` draws floor(size / num_classes) samples per class, so the real size can be
below ``size`` (EMNIST balanced: 63 x 47 = 2,961). The caller records ``len(root_indices)``.

Vetting (``vet=True``) is OPTIONAL and off by default, so runs work without cleanlab installed.
Our train splits are clean (the label-flip attack only touches client data after the root set
is carved), so vetting mostly removes a few ambiguous images. What it does: draw a class-balanced
candidate pool twice the target size, get out-of-sample class probabilities from a 5-fold
cross-validated logistic regression on flattened pixels, drop the samples
``cleanlab.filter.find_label_issues`` flags, then draw the balanced root set from what is left.
De-duplication and outlier filtering from p. 5 are not implemented.
"""

from __future__ import annotations

import logging

import torch

from fedrag.data.partition import numpy_rng_from

log = logging.getLogger(__name__)


def _balanced_sample(
    labels: torch.Tensor, candidates: torch.Tensor, num_classes: int, per_class: int,
    generator: torch.Generator, strict: bool,
) -> torch.Tensor:
    """Draw ``per_class`` indices of every class uniformly from ``candidates`` (CPU tensors).

    ``strict``: raise if a class has too few candidates; otherwise take all it has and warn.
    """
    out = []
    cand_labels = labels[candidates]
    for c in range(num_classes):
        pool = candidates[cand_labels == c]
        if pool.numel() < per_class:
            if strict:
                raise ValueError(f"class {c} has {pool.numel()} samples, need {per_class}")
            log.warning("root set: class %d has only %d vetted samples (wanted %d)", c, pool.numel(), per_class)
        perm = torch.randperm(pool.numel(), generator=generator)[:per_class]
        out.append(pool[perm])
    return torch.cat(out).sort().values


def _find_label_issues(images: torch.Tensor, labels: torch.Tensor, seed: int) -> torch.Tensor:
    """Boolean mask of samples cleanlab flags, from out-of-sample logistic-regression probs."""
    try:
        from cleanlab.filter import find_label_issues
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("vet=True needs cleanlab and scikit-learn (pip install cleanlab scikit-learn)") from e

    x = images.reshape(images.shape[0], -1).float().div(255.0).numpy()
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)  # standardise pixels for a fast LR fit
    y = labels.numpy()
    clf = LogisticRegression(max_iter=200, C=0.1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probs = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")
    return torch.as_tensor(find_label_issues(labels=y, pred_probs=probs), dtype=torch.bool)


def build_root_set(
    labels: torch.Tensor,
    num_classes: int,
    size: int = 3000,
    generator: torch.Generator | None = None,
    vet: bool = False,
    images: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class-balanced root set: floor(size / num_classes) train indices per class (p. 10).

    Args:
        labels: 1-D train labels (any device).
        num_classes: L.
        size: target |D_root| (paper default 3,000, Section IV-B).
        generator: CPU torch.Generator; fixes the draw for a seed.
        vet: optional confident-learning vetting (see module docstring). Needs ``images``
            (the train images, uint8 N x ...) and cleanlab + scikit-learn.

    Returns sorted int64 indices into the train split, on CPU. The real size is
    ``num_classes * floor(size / num_classes)`` (smaller if vetting leaves a class short).
    """
    generator = generator if generator is not None else torch.Generator().manual_seed(0)
    per_class = size // num_classes
    if per_class < 1:
        raise ValueError(f"size={size} is smaller than num_classes={num_classes}")
    labels = labels.detach().cpu()
    all_idx = torch.arange(labels.numel())

    if not vet:
        root = _balanced_sample(labels, all_idx, num_classes, per_class, generator, strict=True)
    else:
        if images is None:
            raise ValueError("vet=True needs the train images")
        pool = _balanced_sample(labels, all_idx, num_classes, 2 * per_class, generator, strict=True)
        seed = int(numpy_rng_from(generator).integers(2**31 - 1))
        issues = _find_label_issues(images[pool.to(images.device)].cpu(), labels[pool], seed)
        log.info("root vetting: cleanlab flagged %d of %d candidates", int(issues.sum()), pool.numel())
        root = _balanced_sample(labels, pool[~issues], num_classes, per_class, generator, strict=False)

    log.info("root set: %d samples (%d per class x %d classes, target %d, vet=%s)",
             root.numel(), per_class, num_classes, size, vet)
    return root
