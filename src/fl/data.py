"""Dataset loading: EMNIST Balanced, CIFAR-10, CIFAR-100 (torchvision), held as normalised tensors.

Preprocessing is not specified in the paper: ASSUMPTION = scale to [0,1], per-channel mean/std
normalisation using training-set statistics, no augmentation.
EMNIST images are stored transposed in the raw files; they are transposed back to upright.
"""
from dataclasses import dataclass

import torch

SHAPES = {"emnist": (1, 28, 28, 47), "cifar10": (3, 32, 32, 10), "cifar100": (3, 32, 32, 100)}


@dataclass
class DataBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    num_classes: int
    synthetic: bool = False


def _normalise(x_tr, x_te):
    mean = x_tr.mean(dim=(0, 2, 3), keepdim=True)
    std = x_tr.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    return (x_tr - mean) / std, (x_te - mean) / std


def _synthetic(dataset: str, seed: int) -> DataBundle:
    """SMOKE-CHECK ONLY. Class-prototype + noise tensors with the real dataset's shapes."""
    ch, h, w, k = SHAPES[dataset]
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(k, ch, h, w, generator=g)

    def make(n):
        y = torch.arange(n) % k
        return protos[y] + 0.5 * torch.randn(n, ch, h, w, generator=g), y.long()

    xtr, ytr = make(max(2000, 20 * k))
    xte, yte = make(500)
    return DataBundle(xtr, ytr, xte, yte, k, synthetic=True)


def load_data(dataset: str, root: str, synthetic: bool = False, seed: int = 0) -> DataBundle:
    if synthetic:
        return _synthetic(dataset, seed)
    import torchvision
    try:
        if dataset == "emnist":
            tr = torchvision.datasets.EMNIST(root, split="balanced", train=True, download=True)
            te = torchvision.datasets.EMNIST(root, split="balanced", train=False, download=True)
            f = lambda d: d.data.transpose(1, 2).unsqueeze(1).float() / 255.0
            ytr, yte, k = tr.targets.long(), te.targets.long(), 47
        else:
            cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
            tr, te = cls(root, train=True, download=True), cls(root, train=False, download=True)
            f = lambda d: torch.from_numpy(d.data).permute(0, 3, 1, 2).float() / 255.0
            ytr, yte, k = torch.tensor(tr.targets).long(), torch.tensor(te.targets).long(), (10 if dataset == "cifar10" else 100)
    except Exception as e:  # network / disk problems
        raise RuntimeError(
            f"Could not load/download '{dataset}' into '{root}': {e}\n"
            f"Check internet access, or place the torchvision files under '{root}' and re-run.") from e
    xtr, xte = _normalise(f(tr), f(te))
    return DataBundle(xtr, ytr, xte, yte, k)
