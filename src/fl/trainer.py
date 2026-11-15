"""End-to-end DRAG federated training loop (Algorithm 1 of the paper)."""
import csv
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from src.algorithms.drag import ReferenceDirection, aggregate, drag_modify
from src.fl.client import buffer_dict, load_param_vector, local_update, param_vector
from src.fl.data import load_data
from src.fl.partition import dirichlet_partition, partition_summary
from src.fl.server import average_buffers, evaluate
from src.models import build_model
from src.utils.common import atomic_torch_save, get_device, set_seed, write_json
from src.utils.config import Config

COLS = ["round", "train_loss_local", "test_loss", "test_acc", "lambda_mean", "lambda_min",
        "lambda_max", "delta_norm", "ref_norm", "round_time_s"]


def _local_seed(cfg, t, m, pass_id):
    return cfg.seed * 1_000_003 + t * 10_007 + m * 101 + pass_id


def run(cfg: Config, verbose: bool = True) -> str:
    cfg = cfg.resolved()
    run_dir = os.path.join(cfg.output_dir, cfg.run_name())
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(metrics_path) and not cfg.resume:
        raise FileExistsError(f"{run_dir} already has results. Use --resume, --tag, or another --output_dir "
                              f"(existing files are never overwritten).")
    os.makedirs(ckpt_dir, exist_ok=True)
    write_json(os.path.join(run_dir, "config.json"), asdict(cfg))

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    data = load_data(cfg.dataset, cfg.data_root, cfg.synthetic_data, cfg.seed)
    parts = dirichlet_partition(data.y_train.numpy(), cfg.num_clients, cfg.beta, cfg.seed, cfg.min_partition_size)
    write_json(os.path.join(run_dir, "partition.json"), partition_summary(parts, data.y_train.numpy(), data.num_classes))

    torch.manual_seed(cfg.seed)
    model = build_model(cfg.model, cfg.dataset).to(device)
    theta = param_vector(model)                       # global model theta^t (flat)
    buffers = buffer_dict(model)
    ref = ReferenceDirection(cfg.alpha)
    start = 0
    last_ckpt = os.path.join(ckpt_dir, "last.pt")
    if cfg.resume and os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device, weights_only=True)
        theta, buffers, start = ck["theta"].to(device), {k: v.to(device) for k, v in ck["buffers"].items()}, ck["round"]
        ref.load_state_dict({k: (v.to(device) if v is not None else None) for k, v in ck["ref"].items()})
        if verbose: print(f"Resumed from round {start}")
    write_json(os.path.join(run_dir, "status.json"), {"status": "running", "rounds_completed": start, "rounds_target": cfg.rounds})

    new_file = not os.path.exists(metrics_path)
    f = open(metrics_path, "a", newline="")
    wr = csv.writer(f, lineterminator="\n")
    if new_file: wr.writerow(COLS)

    def save(round_done, name):
        atomic_torch_save({"theta": theta.cpu(), "buffers": {k: v.cpu() for k, v in buffers.items()},
                           "round": round_done, "config": asdict(cfg),
                           "ref": {k: (v.cpu() if v is not None else None) for k, v in ref.state_dict().items()}},
                          os.path.join(ckpt_dir, name))

    try:
        for t in range(start, cfg.rounds):
            t0 = time.time()
            # Alg.1 L3: select S workers uniformly at random without replacement (seeded per round).
            sel = np.sort(np.random.default_rng([cfg.seed, t]).choice(cfg.num_clients, cfg.clients_per_round, replace=False))
            def train(m, pass_id):
                return local_update(model, theta, buffers, data.x_train, data.y_train, parts[m], cfg.lr,
                                    cfg.local_steps, cfg.batch_size, _local_seed(cfg, t, int(m), pass_id), device)
            # Alg.1 L4-11: reference direction.
            if t == 0:
                init_g = [train(m, 0)[0] for m in sel]       # initial U-step pass; theta^0 is NOT updated by it
                r = ref.initialize(init_g)                   # Eq. (5a)
            else:
                r = ref.advance()                            # Eq. (5b)
            # Alg.1 L13-17: local training from theta^t, DoD, modification.
            v_list, lams, losses, bufs = [], [], [], []
            for m in sel:
                g, loss, b = train(m, 1)
                v, lam = drag_modify(g, r, cfg.c)            # Eqs. (10),(11)
                v_list.append(v); lams.append(lam.item()); losses.append(loss); bufs.append(b)
            # Alg.1 L18: Delta^t (Eq. 6), keep r^t and Delta^t, theta^{t+1} = theta^t + Delta^t (Eq. 7).
            delta = aggregate(v_list)
            if not torch.isfinite(delta).all():
                raise RuntimeError(f"Non-finite aggregated update at round {t}; training diverged.")
            theta = theta + delta
            ref.record_delta(delta)
            if buffers: buffers = average_buffers(bufs)

            tl, ta = float("nan"), float("nan")
            if (t + 1) % cfg.eval_every == 0 or t + 1 == cfg.rounds:
                load_param_vector(model, theta)
                if buffers: model.load_state_dict(buffers, strict=False)
                tl, ta = evaluate(model, data.x_test, data.y_test, device, cfg.eval_batch_size)
            wr.writerow([t + 1, float(np.mean(losses)), tl, ta, float(np.mean(lams)), min(lams), max(lams),
                         delta.norm().item(), r.norm().item(), time.time() - t0]); f.flush()
            if verbose and ((t + 1) % max(1, cfg.eval_every) == 0 or t == 0):
                print(f"[round {t+1}/{cfg.rounds}] train_loss(local)={np.mean(losses):.4f} test_loss={tl:.4f} test_acc={ta:.2f}% lam_mean={np.mean(lams):.3f}")
            if (t + 1) % cfg.checkpoint_every == 0:
                save(t + 1, "last.pt")
                write_json(os.path.join(run_dir, "status.json"), {"status": "running", "rounds_completed": t + 1, "rounds_target": cfg.rounds})
        save(cfg.rounds, "last.pt"); save(cfg.rounds, "final.pt")
        write_json(os.path.join(run_dir, "status.json"), {"status": "completed", "rounds_completed": cfg.rounds, "rounds_target": cfg.rounds, "synthetic_data": data.synthetic})
    except BaseException as e:
        write_json(os.path.join(run_dir, "status.json"), {"status": "failed_or_interrupted", "error": repr(e)})
        raise
    finally:
        f.close()
    return run_dir
