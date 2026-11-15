"""Centralised experiment configuration.

Defaults reproduce the DRAG setup of the base paper (Sec. VI):
M=40 workers, S=10 per round, U=5 local SGD steps, batch size 10, eta=0.01,
alpha=0.25, c=0.1 (beta=0.5) / c=0.25 (beta=0.1).
"""
from dataclasses import asdict, dataclass, fields
from typing import Optional

# Number of communication rounds: the paper text does not state T explicitly.
# These values are the x-axis extents of Figs. 3-6 (EMNIST 3000, CIFAR 4000).
PAPER_ROUNDS = {"emnist": 3000, "cifar10": 4000, "cifar100": 4000}
# Paper: "c = 0.1 for moderate (beta=0.5), c = 0.25 for strong (beta=0.1) heterogeneity".
PAPER_C = {0.1: 0.25, 0.5: 0.1}


@dataclass
class Config:
    # --- problem ---
    dataset: str = "cifar10"          # emnist | cifar10 | cifar100
    model: str = "cnn"                # cnn (per-dataset CNN) | wrn28_4 (CIFAR-100 only)
    data_root: str = "data"
    # --- federated setup (paper Sec. VI) ---
    num_clients: int = 40             # M
    clients_per_round: int = 10       # S
    local_steps: int = 5              # U
    batch_size: int = 10              # B
    lr: float = 0.01                  # eta
    rounds: int = 0                   # T; 0 -> paper-figure default for the dataset
    beta: float = 0.1                 # Dirichlet concentration
    min_partition_size: int = 10      # ASSUMPTION (not in paper): resample Dirichlet until every client has >= this many samples
    # --- DRAG ---
    alpha: float = 0.25
    c: Optional[float] = None         # None -> paper value for the chosen beta
    # --- run control ---
    seed: int = 0
    device: str = "auto"
    eval_every: int = 1
    eval_batch_size: int = 1000
    checkpoint_every: int = 100
    output_dir: str = "results"
    tag: str = ""
    resume: bool = False
    synthetic_data: bool = False      # SMOKE CHECKS ONLY: random tensors instead of the real dataset

    def resolved(self) -> "Config":
        cfg = Config(**asdict(self))
        if cfg.dataset not in PAPER_ROUNDS:
            raise ValueError(f"dataset must be one of {list(PAPER_ROUNDS)}")
        if cfg.model == "wrn28_4" and cfg.dataset != "cifar100":
            raise ValueError("The paper uses WRN-28-4 only on CIFAR-100.")
        if cfg.model not in ("cnn", "wrn28_4"):
            raise ValueError("model must be 'cnn' or 'wrn28_4'")
        if cfg.rounds <= 0:
            cfg.rounds = PAPER_ROUNDS[cfg.dataset]
        if cfg.c is None:
            if cfg.beta not in PAPER_C:
                raise ValueError(f"No paper value of c for beta={cfg.beta}; pass --c explicitly.")
            cfg.c = PAPER_C[cfg.beta]
        if not 0.0 < cfg.alpha < 1.0:
            raise ValueError("alpha must lie in (0,1)")
        if not 0.0 <= cfg.c <= 1.0:
            raise ValueError("c must lie in [0,1]")
        if cfg.clients_per_round > cfg.num_clients:
            raise ValueError("clients_per_round > num_clients")
        return cfg

    def run_name(self) -> str:
        name = (f"drag_{self.dataset}_{self.model}_beta{self.beta}_a{self.alpha}"
                f"_c{self.c}_S{self.clients_per_round}_seed{self.seed}")
        return name + (f"_{self.tag}" if self.tag else "")


def config_from_yaml(path: str) -> Config:
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    valid = {f.name for f in fields(Config)}
    bad = set(d) - valid
    if bad:
        raise ValueError(f"Unknown config keys: {sorted(bad)}")
    return Config(**d)
