"""DRAG training entry point.  Example:  python train.py --dataset cifar10 --beta 0.1"""
import argparse
from dataclasses import fields

from src.fl.trainer import run
from src.utils.config import Config, config_from_yaml


def main():
    ap = argparse.ArgumentParser(description="Original DRAG (arXiv 2601.06903) federated training")
    ap.add_argument("--config", help="optional YAML file; CLI flags override it")
    for f in fields(Config):
        if isinstance(f.default, bool):
            ap.add_argument(f"--{f.name}", action="store_true", default=None)
        else:
            t = float if f.name == "c" else type(f.default)   # c is Optional[float] (default None)
            ap.add_argument(f"--{f.name}", type=t, default=None)
    a = vars(ap.parse_args())
    cfg = config_from_yaml(a.pop("config")) if a.get("config") else Config()
    for k, v in a.items():
        if v is not None and k != "config":
            setattr(cfg, k, v)
    print("Run directory:", run(cfg))


if __name__ == "__main__":
    main()
