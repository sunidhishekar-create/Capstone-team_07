from .cnn import cifar10_cnn, cifar100_cnn, emnist_cnn
from .wrn import wrn28_4


def build_model(name: str, dataset: str):
    """name: 'cnn' (per-dataset CNN from the paper) or 'wrn28_4' (CIFAR-100)."""
    if name == "wrn28_4":
        if dataset != "cifar100":
            raise ValueError("WRN-28-4 is used only on CIFAR-100 in the paper.")
        return wrn28_4(100)
    return {"emnist": emnist_cnn, "cifar10": cifar10_cnn, "cifar100": cifar100_cnn}[dataset]()
