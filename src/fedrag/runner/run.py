"""Single-process federated training runner for BR-DRAG and its baselines.

Base paper: Xiao et al., arXiv 2601.06903, Section VI. Defaults follow the paper: M = 40 clients,
S = 10 selected per round, U = 5 local plain-SGD steps, batch 10, lr 0.01, c_t = 0.5,
|D_root| = 3,000 (class-balanced), 30% malicious clients.

    python -m fedrag.runner.run --dataset emnist_balanced --beta 0.5 --method br_drag \
        --attack sign_flip --malicious 0.3 --rounds 200 --eval-every 10 --seed 0 \
        --out results/emnist_brdrag_signflip [--resume]

Design (kept deliberately small so it can be swapped for the full simulator later):
- One shared model on the device, no per-client copies. Each selected client loads theta^t,
  runs U SGD steps on random minibatches of its own data (``root_trainer.root_reference``, the
  same plain-SGD code the server uses for r^t) and returns g = theta^(t,U) - theta^t.
- Malicious clients are a fixed set chosen at start. ``label_flip`` poisons their labels once
  (``LabelFlip.label_overrides``); update-level attacks replace their uploads via ``corrupt()``.
  With ``--attack none`` there are no malicious clients.
- Per round: strategy.prepare_downlink -> local training -> attack -> server_transform ->
  aggregate -> apply.
- Every random draw has its own seeded torch.Generator stream (root set, partition, malicious
  set, attack, client selection, client minibatches, server root minibatches), so changing the
  method does not change which clients are selected or which data they see.
- Checkpoints (every ``--checkpoint-every`` rounds and at the end) hold theta, every RNG state,
  the round and the metrics. ``--resume`` continues from it; on CPU the result is bit-identical
  to an uninterrupted run. On CUDA, cuDNN autotuning makes runs close but not bit-identical.

Outputs in ``--out``:
- console.log: everything the run prints (stdout + stderr, tracebacks), appended live; a
  --resume appends a new session to it.
- run_info.json: command, resolved config, git branch / commit / uncommitted changes, Python /
  torch / CUDA versions and device name, start / end / total / setup / training / evaluation
  time, final and best test accuracy and the status (running / resumed / finished / crashed /
  interrupted). Written at the start, on every evaluation and at the end; one entry per session
  in "sessions".
- data_info.json: root set size and per-class counts, malicious client IDs, each client's
  sample count and number of classes.
- config.json, partition.csv (client x class counts), metrics.csv (one row per round:
  test_acc / test_loss on eval rounds, strategy and attack diagnostics, timings).
- checkpoint.pt (for --resume), final_model.pt (final theta + state_dict) and accuracy.png
  (test accuracy and loss vs round).
A run without --resume refuses to start in a folder that already holds any of these files.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fedrag.attacks.base import Attack
from fedrag.attacks.label_flip import LabelFlip
from fedrag.attacks.min_max import MinMax
from fedrag.attacks.min_sum import MinSum
from fedrag.attacks.noise import NoiseInjection
from fedrag.attacks.sign_flip import SignFlip
from fedrag.data import DATASETS, FedDataset, build_root_set, class_counts, dirichlet_partition, load_dataset
from fedrag.models import build_model
from fedrag.strategies.br_drag import BRDRAG, Downlink, Uplink, Vec, aggregate, apply, sync_device
from fedrag.strategies.fltrust import FLTrust
from fedrag.strategies.root_trainer import get_params, load_params, root_reference
from fedrag.runner.run_files import (CONSOLE_LOG, DATA_INFO, FINAL_MODEL, RUN_FILES, RUN_INFO, RunRecorder,
                                     eval_summary, now_iso, save_torch, write_json)

log = logging.getLogger("fedrag.runner")

METHODS = ("fedavg", "br_drag", "fltrust_paper", "fltrust_cao")
ATTACKS = ("none", "noise", "sign_flip", "label_flip", "min_max", "min_sum")
CKPT_NAME = "checkpoint.pt"
# Config fields that may change between a run and its --resume (they do not change the math).
RESUME_MUTABLE = {"rounds", "eval_every", "resume", "checkpoint_every", "eval_batch_size",
                  "device", "data_dir", "out"}
# RNG streams used inside the training loop, saved in checkpoints. Setup streams (root,
# partition, malicious) are re-derived from the seed on resume.
LOOP_STREAMS = ("attack", "select", "client", "root_train")


@dataclass
class RunConfig:
    """Everything that defines a run. Paper defaults (Section VI) unless noted."""

    dataset: str = "emnist_balanced"
    beta: float = 0.5
    method: str = "br_drag"
    attack: str = "none"
    malicious: float = 0.3  # A / M
    rounds: int = 200
    eval_every: int = 10
    seed: int = 0
    out: str = "results/run"
    resume: bool = False
    data_dir: str = "data"
    num_clients: int = 40  # M
    clients_per_round: int = 10  # S
    local_steps: int = 5  # U
    batch_size: int = 10  # B
    lr: float = 0.01  # eta
    c_t: float = 0.5  # BR-DRAG
    root_size: int = 3000
    vet_root: bool = False
    min_client_size: int = 10
    checkpoint_every: int = 50
    eval_batch_size: int = 2000
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; choose from {METHODS}")
        if self.attack not in ATTACKS:
            raise ValueError(f"unknown attack {self.attack!r}; choose from {ATTACKS}")
        if not 0.0 <= self.malicious <= 1.0:
            raise ValueError("malicious must be in [0, 1]")
        if not 1 <= self.clients_per_round <= self.num_clients:
            raise ValueError("need 1 <= clients_per_round <= num_clients")
        if self.rounds < 0 or self.eval_every < 1 or self.checkpoint_every < 1:
            raise ValueError("need rounds >= 0, eval_every >= 1, checkpoint_every >= 1")


# ----------------------------------------------------------------------------- helpers
def make_stream(seed: int, name: str) -> torch.Generator:
    """Independent CPU generator for stream ``name``, derived from ``seed`` by hashing."""
    h = hashlib.sha256(f"fedrag:{seed}:{name}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & (2**63 - 1))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class FedAvg:
    """Inline baseline (paper eq. (4)): theta^(t+1) = theta^t + unweighted mean of raw g."""

    name = "fedavg"

    def prepare_downlink(self, t: int, theta: Vec) -> Downlink:
        return Downlink(round=t, theta=theta)

    def server_transform(self, t: int, uplinks: list[Uplink], dl: Downlink) -> tuple[list[Vec], dict]:
        vs = [u.vector for u in uplinks]
        return vs, {"g_norm_mean": torch.stack([v.norm() for v in vs]).mean().item()}

    def aggregate(self, t: int, vectors: list[Vec]) -> Vec:
        return aggregate(vectors)

    def apply(self, t: int, theta: Vec, delta: Vec) -> Vec:
        return apply(theta, delta)

    def client_postprocess(self, t: int, g: Vec, dl: Downlink) -> tuple[Vec, str, dict]:
        return g, "raw", {}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, sd: dict) -> None:
        if sd:
            raise ValueError(f"FedAvg has no state to load, got keys {sorted(sd)}")


def make_strategy(cfg: RunConfig, model: nn.Module, root_data, generator: torch.Generator):
    kw = dict(steps=cfg.local_steps, batch_size=cfg.batch_size, lr=cfg.lr)
    if cfg.method == "fedavg":
        return FedAvg()
    if cfg.method == "br_drag":
        return BRDRAG(model, root_data, generator, c_t=cfg.c_t, **kw)
    variant = "paper" if cfg.method == "fltrust_paper" else "cao"
    return FLTrust(model, root_data, generator, variant=variant, **kw)


def make_attack(name: str, num_classes: int) -> Optional[Attack]:
    return {
        "none": lambda: None,
        "noise": NoiseInjection,
        "sign_flip": SignFlip,
        "label_flip": lambda: LabelFlip(num_classes),
        "min_max": MinMax,
        "min_sum": MinSum,
    }[name]()


@torch.no_grad()
def evaluate(model: nn.Module, theta: Vec, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> tuple[float, float]:
    """(accuracy, mean cross-entropy) of theta on the full (x, y), batched."""
    load_params(model, theta)
    model.eval()
    correct = torch.zeros((), device=x.device)
    loss = torch.zeros((), device=x.device)
    for i in range(0, x.shape[0], batch_size):
        out = model(x[i : i + batch_size])
        yb = y[i : i + batch_size]
        loss += F.cross_entropy(out.float(), yb, reduction="sum")
        correct += (out.argmax(dim=1) == yb).sum()
    n = x.shape[0]
    return correct.item() / n, loss.item() / n


def _scalars(d: dict, prefix: str = "") -> dict:
    """Keep only scalar (int/float/bool) entries of a diagnostics dict, with a key prefix."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            v = v.item()
        if isinstance(v, (bool, int, float)):
            out[prefix + k] = float(v) if isinstance(v, bool) else v
    return out


FIRST_COLUMNS = ["round", "test_acc", "test_loss", "num_malicious_selected", "delta_norm",
                 "time_round_s", "time_clients_s", "time_eval_s"]


def write_metrics(path: Path, rows: list[dict]) -> None:
    """Rewrite metrics.csv (union of all keys; blanks where a round has no value)."""
    keys = list(FIRST_COLUMNS)
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- setup
@dataclass
class Setup:
    """Everything derived once from (config, data): fixed for the whole run."""

    root_idx: torch.Tensor
    parts: list[torch.Tensor]
    malicious: set[int]
    client_data: list[tuple[torch.Tensor, torch.Tensor]]
    root_data: tuple[torch.Tensor, torch.Tensor]
    attack: Optional[Attack]


def build_setup(cfg: RunConfig, data: FedDataset, attack_gen: torch.Generator) -> Setup:
    """Root set (carved first) -> Dirichlet split -> malicious set -> attack setup -> client tensors."""
    dev = data.x_train.device
    root_idx = build_root_set(data.y_train, data.num_classes, cfg.root_size,
                              make_stream(cfg.seed, "root"), vet=cfg.vet_root, images=data.x_train)
    parts = dirichlet_partition(data.y_train, cfg.num_clients, cfg.beta, make_stream(cfg.seed, "partition"),
                                exclude=root_idx, min_size=cfg.min_client_size)

    n_mal = 0 if cfg.attack == "none" else int(round(cfg.malicious * cfg.num_clients))
    perm = torch.randperm(cfg.num_clients, generator=make_stream(cfg.seed, "malicious"))
    malicious = set(perm[:n_mal].tolist())

    attack = make_attack(cfg.attack, data.num_classes)
    labels = [data.y_train[p.to(dev)] for p in parts]
    if attack is not None:
        own_labels = {cid: labels[cid] for cid in sorted(malicious)}
        attack.setup(own_labels, malicious, attack_gen)
        if isinstance(attack, LabelFlip):
            for cid, new in attack.label_overrides.items():
                labels[cid] = new

    client_data = [(data.x_train[p.to(dev)], labels[cid]) for cid, p in enumerate(parts)]
    root_data = (data.x_train[root_idx.to(dev)], data.y_train[root_idx.to(dev)])
    return Setup(root_idx, parts, malicious, client_data, root_data, attack)


def _write_setup_files(out: Path, cfg: RunConfig, data: FedDataset, s: Setup, device, n_params: int) -> None:
    counts = class_counts(data.y_train, s.parts, data.num_classes)
    with open(out / "partition.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client", "num_samples", "malicious"] + [f"c{k}" for k in range(data.num_classes)])
        for cid, row in enumerate(counts.tolist()):
            w.writerow([cid, sum(row), int(cid in s.malicious)] + row)
    info = {
        "config": asdict(cfg),
        "root_size_actual": int(s.root_idx.numel()),
        "malicious_ids": sorted(s.malicious),
        "client_sizes": counts.sum(dim=1).tolist(),
        "num_train": int(data.y_train.numel()),
        "num_test": int(data.y_test.numel()),
        "num_params": n_params,
        "device": str(device),
        "torch": torch.__version__,
    }
    (out / "config.json").write_text(json.dumps(info, indent=2))

    sizes = counts.sum(dim=1).tolist()
    root_counts = torch.bincount(data.y_train[s.root_idx.to(data.y_train.device)].cpu(),
                                 minlength=data.num_classes).tolist()
    write_json(out / DATA_INFO, {
        "dataset": cfg.dataset,
        "num_classes": data.num_classes,
        "num_train": int(data.y_train.numel()),
        "num_test": int(data.y_test.numel()),
        "num_params": n_params,
        "root": {"size": int(s.root_idx.numel()), "requested_size": cfg.root_size,
                 "vetted": cfg.vet_root, "class_counts": root_counts},
        "malicious": {"attack": cfg.attack, "fraction": cfg.malicious, "num": len(s.malicious),
                      "ids": sorted(s.malicious)},
        "client_sizes": {"min": min(sizes), "median": int(np.median(sizes)), "max": max(sizes),
                         "total": sum(sizes)},
        # class counts use the true labels (as in partition.csv), also for label_flip clients
        "clients": [{"id": cid, "num_samples": int(sum(row)), "num_classes": sum(1 for c in row if c > 0),
                     "malicious": cid in s.malicious} for cid, row in enumerate(counts.tolist())],
    })


def save_final_model(path: Path, model: nn.Module, theta: Vec, t: int, cfg: RunConfig,
                     metrics: list[dict]) -> None:
    """final_model.pt: the final global model (flat theta and the matching state_dict)."""
    load_params(model, theta)
    save_torch(path, {
        "round": t,
        "theta": theta.detach().cpu(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg),
        "num_params": int(theta.numel()),
        **{k: v for k, v in eval_summary(metrics).items() if k in ("final_test_acc", "best_test_acc")},
    })


# ----------------------------------------------------------------------------- checkpoints
def _rng_snapshot(streams: dict[str, torch.Generator], device: torch.device) -> dict:
    return {
        "streams": {k: g.get_state() for k, g in streams.items()},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _rng_restore(snap: dict, streams: dict[str, torch.Generator], device: torch.device) -> None:
    for k, g in streams.items():
        g.set_state(snap["streams"][k])
    torch.set_rng_state(snap["torch"])
    if device.type == "cuda" and snap["cuda"] is not None:
        torch.cuda.set_rng_state_all(snap["cuda"])
    np.random.set_state(snap["numpy"])
    random.setstate(snap["python"])


def save_checkpoint(path: Path, t: int, theta: Vec, metrics: list[dict], cfg: RunConfig,
                    streams: dict, strategy, malicious: set[int], device: torch.device) -> None:
    ckpt = {
        "version": 1,
        "round": t,
        "theta": theta.detach().cpu(),
        "metrics": metrics,
        "config": asdict(cfg),
        "rng": _rng_snapshot(streams, device),
        "strategy": strategy.state_dict(),
        "malicious": sorted(malicious),
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never corrupts the last checkpoint


def load_checkpoint(path: Path, cfg: RunConfig) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # our own file: has RNG tuples
    old, new = ckpt["config"], asdict(cfg)
    diff = {k: (old.get(k), v) for k, v in new.items() if k not in RESUME_MUTABLE and old.get(k) != v}
    if diff:
        raise ValueError(f"cannot resume: config differs from the checkpoint (old, new): {diff}")
    return ckpt


# ----------------------------------------------------------------------------- main loop
def _prepare_out(cfg: RunConfig, out: Path) -> Optional[dict]:
    """Check that ``out`` may be written; return the checkpoint to resume from (or None).

    - Without --resume, any existing run file is an error (nothing is ever overwritten).
    - With --resume and no checkpoint: a finished run (final_model.pt, or run_info.json saying
      "finished") is an error, since starting over would replace it. Otherwise the run starts
      from round 0; an old metrics.csv is kept as metrics.before_restart_<time>.csv.
    """
    existing = [n for n in RUN_FILES if (out / n).exists()]
    if existing and not cfg.resume:
        raise FileExistsError(f"{out} already holds a run ({', '.join(existing)}); "
                              "pass --resume to continue it or use a new --out")
    ckpt_path = out / CKPT_NAME
    if cfg.resume and ckpt_path.exists():
        return load_checkpoint(ckpt_path, cfg)
    if cfg.resume:
        status = None
        if (out / RUN_INFO).exists():
            try:
                status = json.loads((out / RUN_INFO).read_text(encoding="utf-8")).get("status")
            except (OSError, ValueError):
                pass
        if (out / FINAL_MODEL).exists() or status == "finished":
            raise FileExistsError(f"{out} holds a finished run but no {CKPT_NAME} to resume from; "
                                  "starting over would overwrite it, use a new --out")
        if (out / "metrics.csv").exists():
            os.replace(out / "metrics.csv", out / f"metrics.before_restart_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    return None


def run(cfg: RunConfig, data: Optional[FedDataset] = None, verbose: bool = True,
        argv: Optional[list[str]] = None) -> dict:
    """Train according to ``cfg``. ``data`` overrides dataset loading (used by tests); ``argv``
    is the command line recorded in run_info.json (set by ``main``).

    Returns {"theta": final theta (CPU), "metrics": list of per-round dicts, "out": out dir}.
    """
    say = print if verbose else (lambda *a, **k: None)
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed batch shapes: let cuDNN pick fast kernels
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = _prepare_out(cfg, out)

    if ckpt is not None and ckpt["round"] >= cfg.rounds:
        # Nothing to train: leave every file of the run as it is, only note the call in console.log.
        msg = (f"[done] checkpoint already at round {ckpt['round']} >= --rounds {cfg.rounds}; "
               "nothing to do, run files left unchanged")
        with open(out / CONSOLE_LOG, "a", encoding="utf-8") as f:
            f.write(f"===== {now_iso()} --resume with nothing to do\n{msg}\n")
        say(msg)
        return {"theta": ckpt["theta"], "metrics": ckpt["metrics"], "out": out}

    rec = RunRecorder(out, asdict(cfg), device, argv, resume_requested=cfg.resume,
                      resumed_from_round=None if ckpt is None else ckpt["round"])
    with rec:
        return _train(cfg, data, device, out, ckpt, rec, say)


def _train(cfg: RunConfig, data: Optional[FedDataset], device: torch.device, out: Path,
           ckpt: Optional[dict], rec: RunRecorder, say) -> dict:
    ckpt_path = out / CKPT_NAME
    t_load = time.perf_counter()
    if data is None:
        data = load_dataset(cfg.dataset, cfg.data_dir, device)
    else:
        data = dataclasses.replace(data, x_train=data.x_train.to(device), y_train=data.y_train.to(device),
                                   x_test=data.x_test.to(device), y_test=data.y_test.to(device))
    say(f"[setup] {cfg.dataset}: {data.y_train.numel()} train / {data.y_test.numel()} test "
        f"on {device} ({time.perf_counter() - t_load:.1f}s)")

    streams = {k: make_stream(cfg.seed, k) for k in LOOP_STREAMS}
    s = build_setup(cfg, data, streams["attack"])
    model = build_model(cfg.dataset, cfg.seed, num_classes=data.num_classes).to(device)
    theta = get_params(model)
    strategy = make_strategy(cfg, model, s.root_data, streams["root_train"])
    _write_setup_files(out, cfg, data, s, device, theta.numel())
    sizes = [int(p.numel()) for p in s.parts]
    say(f"[setup] root={s.root_idx.numel()} | clients min/median/max = {min(sizes)}/"
        f"{int(np.median(sizes))}/{max(sizes)} | malicious={sorted(s.malicious)} | "
        f"params={theta.numel():,} | {cfg.method} vs {cfg.attack}")

    metrics: list[dict] = []
    start = 0
    if ckpt is not None:
        if ckpt["malicious"] != sorted(s.malicious):
            raise RuntimeError("malicious set differs from the checkpoint; setup is not reproducible")
        start, metrics = ckpt["round"], ckpt["metrics"]
        theta = ckpt["theta"].to(device)
        strategy.load_state_dict(ckpt["strategy"])
        _rng_restore(ckpt["rng"], streams, device)
        say(f"[resume] from round {start}")
    elif cfg.resume:
        say("[resume] no checkpoint found, starting fresh")
    rec.metrics = metrics
    rec.setup_s = time.perf_counter() - t_load

    M, S, U = cfg.num_clients, cfg.clients_per_round, cfg.local_steps
    total_train = total_eval = 0.0
    since_eval: list[float] = []

    def do_eval(t: int, row: dict) -> None:
        nonlocal total_eval
        sync_device(device)
        t0 = time.perf_counter()
        acc, loss = evaluate(model, theta, data.x_test, data.y_test, cfg.eval_batch_size)
        sec = time.perf_counter() - t0
        total_eval += sec
        row.update(test_acc=acc, test_loss=loss, time_eval_s=sec)
        per_round = f"{np.mean(since_eval):.3f} s/round" if since_eval else "-"
        say(f"[round {t:>5}/{cfg.rounds}] acc={acc:.4f} loss={loss:.4f} | train {per_round} | "
            f"eval {sec:.2f} s", flush=True)
        since_eval.clear()
        write_metrics(out / "metrics.csv", metrics)
        rec.update(metrics, total_train, total_eval)

    if start == 0:
        row = {"round": 0}
        metrics.append(row)
        do_eval(0, row)

    for t in range(start + 1, cfg.rounds + 1):
        sync_device(device)
        t0 = time.perf_counter()
        dl = strategy.prepare_downlink(t, theta)
        selected = torch.randperm(M, generator=streams["select"])[:S].sort().values.tolist()

        t_cl = time.perf_counter()
        uplinks, own_g = [], {}
        for cid in selected:
            x, y = s.client_data[cid]
            g = root_reference(model, theta, (x, y), steps=U, batch_size=cfg.batch_size, lr=cfg.lr,
                               generator=streams["client"])  # g = theta^(t,U) - theta^t
            vec, kind, meta = strategy.client_postprocess(t, g, dl)
            uplinks.append(Uplink(round=t, client_id=cid, vector=vec, kind=kind,
                                  num_samples=int(y.numel()), meta=meta))
            if cid in s.malicious:
                own_g[cid] = g
        sync_device(device)
        t_clients = time.perf_counter() - t_cl

        if s.attack is not None and s.attack.level == "update":
            uplinks = s.attack.corrupt(t, uplinks, s.malicious, own_g)
        vectors, sdiag = strategy.server_transform(t, uplinks, dl)
        delta = strategy.aggregate(t, vectors)
        theta = strategy.apply(t, theta, delta)
        delta_norm = delta.norm().item()  # also syncs the device
        t_round = time.perf_counter() - t0
        total_train += t_round
        rec.train_s = total_train  # kept current so a crash still reports the training time
        since_eval.append(t_round)

        row = {"round": t, "num_malicious_selected": len(own_g), "delta_norm": delta_norm,
               "time_round_s": t_round, "time_clients_s": t_clients, **_scalars(sdiag)}
        if s.attack is not None:
            row.update(_scalars(s.attack.last_diagnostics, "attack_"))
        metrics.append(row)

        if t % cfg.eval_every == 0 or t == cfg.rounds:
            do_eval(t, row)
        if t % cfg.checkpoint_every == 0 or t == cfg.rounds:
            save_checkpoint(ckpt_path, t, theta, metrics, cfg, streams, strategy, s.malicious, device)
            write_metrics(out / "metrics.csv", metrics)

    save_final_model(out / FINAL_MODEL, model, theta, cfg.rounds, cfg, metrics)
    rec.update(metrics, total_train, total_eval)
    rec.finished = True
    n_done = cfg.rounds - start
    say(f"[done] {n_done} rounds: training {total_train:.1f}s ({total_train / max(n_done, 1):.3f} s/round), "
        f"evaluation {total_eval:.1f}s total | results in {out}")
    return {"theta": theta.detach().cpu(), "metrics": metrics, "out": out}


# ----------------------------------------------------------------------------- CLI
def parse_args(argv: Optional[list[str]] = None) -> RunConfig:
    d = RunConfig()
    p = argparse.ArgumentParser(prog="python -m fedrag.runner.run", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default=d.dataset, choices=DATASETS)
    p.add_argument("--beta", type=float, default=d.beta, help="Dirichlet beta (paper: 0.1, 0.5)")
    p.add_argument("--method", default=d.method, choices=METHODS)
    p.add_argument("--attack", default=d.attack, choices=ATTACKS)
    p.add_argument("--malicious", type=float, default=d.malicious, help="fraction A/M of malicious clients")
    p.add_argument("--rounds", type=int, default=d.rounds)
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--out", default=None, help="output dir (default results/<dataset>_<method>_<attack>_b<beta>"
                                                "[_ct<c_t>]_s<seed>; _ct only when c_t != 0.5)")
    p.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.pt if present")
    p.add_argument("--data-dir", default=d.data_dir)
    p.add_argument("--num-clients", type=int, default=d.num_clients)
    p.add_argument("--clients-per-round", type=int, default=d.clients_per_round)
    p.add_argument("--local-steps", type=int, default=d.local_steps)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--c-t", "--ct", dest="c_t", type=float, default=d.c_t, help="BR-DRAG c_t (--ct is an alias)")
    p.add_argument("--root-size", type=int, default=d.root_size)
    p.add_argument("--vet-root", action="store_true", help="vet the root pool with cleanlab (optional)")
    p.add_argument("--min-client-size", type=int, default=d.min_client_size)
    p.add_argument("--checkpoint-every", type=int, default=d.checkpoint_every)
    p.add_argument("--eval-batch-size", type=int, default=d.eval_batch_size)
    p.add_argument("--device", default=d.device, help="auto (cuda if available), cpu, cuda, cuda:1, ...")
    a = vars(p.parse_args(argv))
    if a["out"] is None:
        ct = "" if a["c_t"] == d.c_t else f"_ct{a['c_t']}"
        a["out"] = f"results/{a['dataset']}_{a['method']}_{a['attack']}_b{a['beta']}{ct}_s{a['seed']}"
    return RunConfig(**a)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    argv = list(sys.argv[1:] if argv is None else argv)
    run(parse_args(argv), argv=argv)


if __name__ == "__main__":
    main()
