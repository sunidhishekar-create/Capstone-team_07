"""Data pipeline: dataset tensors, root set (carved first), Dirichlet client split."""

from fedrag.data.datasets import DATASET_STATS, DATASETS, FedDataset, Normalize, load_dataset
from fedrag.data.partition import class_counts, dirichlet_partition
from fedrag.data.root import build_root_set

__all__ = [
    "DATASET_STATS",
    "DATASETS",
    "FedDataset",
    "Normalize",
    "load_dataset",
    "build_root_set",
    "dirichlet_partition",
    "class_counts",
]
