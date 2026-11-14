"""Dataset loading: whole datasets as uint8 tensors on the training device.

Base paper: Xiao et al., arXiv 2601.06903, Section VI (EMNIST "balanced" with 47 classes,
CIFAR-10, CIFAR-100). The official train split feeds the root set and the clients; the official
test split is used only for evaluation.

Images stay uint8 (N, C, H, W) on the device and are normalised per batch by ``Normalize``,
the first module of every model built by ``fedrag.models.build_model``. So training, the root
trainer and evaluation all pass raw uint8 batches straight into the model, with no DataLoader.
The paper does not mention normalisation or augmentation; we use the standard per-dataset
mean/std below and no augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

# Standard per-channel statistics of each training split, in [0, 1] pixel units.
DATASET_STATS: dict[str, dict] = {
    "emnist_balanced": {"num_classes": 47, "mean": (0.1751,), "std": (0.3332,)},
    "cifar10": {
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "num_classes": 100,
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
    },
}
DATASETS = tuple(DATASET_STATS)


@dataclass
class FedDataset:
    """A dataset held as device tensors. Images uint8 (N, C, H, W); labels int64 (N,)."""

    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    num_classes: int

    def __post_init__(self) -> None:
        for split, x, y in (("train", self.x_train, self.y_train), ("test", self.x_test, self.y_test)):
            if x.dtype != torch.uint8 or x.dim() != 4:
                raise ValueError(f"{split} images must be uint8 (N, C, H, W), got {x.dtype} {tuple(x.shape)}")
            if y.dtype != torch.int64 or y.shape != (x.shape[0],):
                raise ValueError(f"{split} labels must be int64 of shape ({x.shape[0]},)")


class Normalize(nn.Module):
    """Per-batch normalisation: uint8 -> float in [0, 1] -> (x - mean) / std.

    mean/std are buffers, not parameters, so they are not part of the flat parameter vector
    theta used by the strategies (``root_trainer.get_params``). Float inputs are assumed to be
    already scaled to [0, 1].
    """

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        return (x - self.mean) / self.std


def _to_tensors(ds) -> tuple[torch.Tensor, torch.Tensor]:
    """torchvision dataset -> (uint8 N x C x H x W images, int64 labels), on CPU."""
    x = torch.as_tensor(ds.data)
    if x.dim() == 3:  # grayscale (N, H, W)
        x = x.unsqueeze(1)
    elif x.shape[-1] in (1, 3):  # CIFAR numpy arrays are (N, H, W, C)
        x = x.permute(0, 3, 1, 2)
    y = torch.as_tensor(ds.targets, dtype=torch.int64)
    return x.contiguous().to(torch.uint8), y


def load_dataset(name: str, data_dir: str | Path, device: torch.device | str) -> FedDataset:
    """Load ``name`` in {"emnist_balanced", "cifar10", "cifar100"} via torchvision.

    Downloads to ``data_dir`` on first use. Returns the official train and test splits as
    uint8 image / int64 label tensors on ``device``. EMNIST images are stored transposed in the
    raw files, so they are transposed back to be upright.
    """
    from torchvision import datasets as tvd  # imported lazily: tests never need torchvision

    if name not in DATASET_STATS:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    root = str(Path(data_dir))
    if name == "emnist_balanced":
        make = lambda train: tvd.EMNIST(root, split="balanced", train=train, download=True)  # noqa: E731
    elif name == "cifar10":
        make = lambda train: tvd.CIFAR10(root, train=train, download=True)  # noqa: E731
    else:
        make = lambda train: tvd.CIFAR100(root, train=train, download=True)  # noqa: E731

    x_tr, y_tr = _to_tensors(make(True))
    x_te, y_te = _to_tensors(make(False))
    if name == "emnist_balanced":
        x_tr, x_te = x_tr.transpose(2, 3).contiguous(), x_te.transpose(2, 3).contiguous()

    device = torch.device(device)
    return FedDataset(
        name=name,
        x_train=x_tr.to(device),
        y_train=y_tr.to(device),
        x_test=x_te.to(device),
        y_test=y_te.to(device),
        num_classes=DATASET_STATS[name]["num_classes"],
    )
