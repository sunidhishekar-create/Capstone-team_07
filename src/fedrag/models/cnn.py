"""Small CNNs from the base paper's experiment setup.

Base paper: Xiao et al., arXiv 2601.06903, Section VI, p. 7:
- EMNIST: "two 5x5 convolutional layers and two fully connected layers", 47-class output.
- CIFAR-10: "a CNN with two 5x5 padded convolutional layers", 10 classes.
- CIFAR-100: "three 3x3 padded convolutional layers followed by max pooling operations" and
  "two fully connected layers", 100-class output.

The paper gives no channel counts, hidden sizes, pooling (for EMNIST/CIFAR-10) or activations.
Our choices follow the classic FedAvg CNN (McMahan et al., AISTATS 2017: 32 / 64 channels,
2x2 max pooling, a 512-unit hidden layer, ReLU), with no dropout and no BatchNorm (BatchNorm
buffers are not in the flat parameter vector theta):

| model     | layers                                                                | params  |
|-----------|-----------------------------------------------------------------------|---------|
| EMNIST    | conv5x5(1->32) pool, conv5x5(32->64) pool, fc(1024->512), fc(512->47) | 601,007 |
| CIFAR-10  | conv5x5 pad2(3->32) pool, conv5x5 pad2(32->64) pool,                  |         |
|           | fc(4096->512), fc(512->10)                                            | 2,156,490 |
| CIFAR-100 | conv3x3 pad1(3->32) pool, (32->64) pool, (64->128) pool,              |         |
|           | fc(2048->512), fc(512->100)                                           | 1,193,636 |

``build_model`` prepends ``Normalize`` (no parameters) so models take raw uint8 batches, and
seeds the initialisation locally: the same seed gives the same initial weights, and the global
torch RNG is left untouched.
"""

from __future__ import annotations

import torch
from torch import nn

from fedrag.data.datasets import DATASET_STATS, Normalize


def _conv_block(c_in: int, c_out: int, k: int, pad: int) -> list[nn.Module]:
    return [nn.Conv2d(c_in, c_out, k, padding=pad), nn.ReLU(inplace=True), nn.MaxPool2d(2)]


class EMNISTCNN(nn.Module):
    """Two 5x5 convs (no padding) + two FC layers, 1 x 28 x 28 input (p. 7)."""

    def __init__(self, num_classes: int = 47) -> None:
        super().__init__()
        self.features = nn.Sequential(*_conv_block(1, 32, 5, 0), *_conv_block(32, 64, 5, 0))  # 28->12->4
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CIFAR10CNN(nn.Module):
    """Two padded 5x5 convs + two FC layers, 3 x 32 x 32 input (p. 7)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(*_conv_block(3, 32, 5, 2), *_conv_block(32, 64, 5, 2))  # 32->16->8
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 512), nn.ReLU(inplace=True), nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CIFAR100CNN(nn.Module):
    """Three padded 3x3 convs with max pooling + two FC layers, 3 x 32 x 32 input (p. 7)."""

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(3, 32, 3, 1), *_conv_block(32, 64, 3, 1), *_conv_block(64, 128, 3, 1)
        )  # 32->16->8->4
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


ARCHS = {"emnist_balanced": EMNISTCNN, "cifar10": CIFAR10CNN, "cifar100": CIFAR100CNN}


def build_model(dataset: str, seed: int, num_classes: int | None = None) -> nn.Sequential:
    """``Normalize`` + the paper's CNN for ``dataset``, initialised from ``seed``.

    ``num_classes`` defaults to the dataset's (47 / 10 / 100); tests pass a smaller value.
    The model is returned on CPU; move it with ``.to(device)``.
    """
    if dataset not in ARCHS:
        raise ValueError(f"no model for dataset {dataset!r}; choose from {tuple(ARCHS)}")
    stats = DATASET_STATS[dataset]
    k = stats["num_classes"] if num_classes is None else num_classes
    with torch.random.fork_rng(devices=[]):  # local seeding, global RNG untouched
        torch.manual_seed(seed)
        net = ARCHS[dataset](k)
    return nn.Sequential(Normalize(stats["mean"], stats["std"]), net)
