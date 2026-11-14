"""Tests for the files every run writes (console.log, run_info.json, data_info.json,
final_model.pt, accuracy.png), resume / crash bookkeeping, the sweep's plan.json and the
git-ignore rules. CPU only, tiny synthetic data.

Run with:  PYTHONPATH=src pytest tests/runner
"""

import argparse
import csv
import importlib.util
import json
import logging
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
import torch

import fedrag.runner.run as runner
from fedrag.data import FedDataset
from fedrag.models import build_model
from fedrag.runner.run import RunConfig, run
from fedrag.strategies.root_trainer import get_params

REPO = Path(__file__).resolve().parents[2]
K = 4


def _tiny_data(seed=0, per_class=60, test_per_class=10):
    """28x28 grayscale, 4 classes, each class lights up its own image quadrant."""
    g = torch.Generator().manual_seed(seed)

    def make(n):
        y = torch.arange(K).repeat_interleave(n)
        x = torch.rand(y.numel(), 1, 28, 28, generator=g) * 60
        for c in range(K):
            r, col = divmod(c, 2)
            x[y == c, 0, r * 14 : r * 14 + 14, col * 14 : col * 14 + 14] += 150
        perm = torch.randperm(y.numel(), generator=g)
        return x[perm].to(torch.uint8), y[perm]

    x_tr, y_tr = make(per_class)
    x_te, y_te = make(test_per_class)
    return FedDataset("emnist_balanced", x_tr, y_tr, x_te, y_te, num_classes=K)


def _cfg(out, **kw):
    base = dict(dataset="emnist_balanced", beta=1.0, num_clients=6, clients_per_round=3,
                malicious=0.34, rounds=3, eval_every=2, seed=0, root_size=20, min_client_size=10,
                checkpoint_every=50, device="cpu", out=str(out), method="br_drag", attack="sign_flip")
    base.update(kw)
    return RunConfig(**base)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _snapshot(out, skip=()):
    return {p.name: p.read_bytes() for p in Path(out).iterdir() if p.is_file() and p.name not in skip}


def _metrics(out):
    with open(Path(out) / "metrics.csv", newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- a full run
def test_three_round_run_writes_everything(tmp_path, capsys):
    out = tmp_path / "run"
    cfg = _cfg(out)
    argv = ["--method", "br_drag", "--attack", "sign_flip", "--rounds", "3", "--out", str(out)]
    res = run(cfg, data=_tiny_data(), verbose=True, argv=argv)

    for name in ("console.log", "run_info.json", "data_info.json", "final_model.pt", "accuracy.png",
                 "checkpoint.pt", "metrics.csv", "config.json", "partition.csv"):
        assert (out / name).is_file(), name

    # still printed to the terminal, and the same lines are in console.log
    term = capsys.readouterr().out
    log = (out / "console.log").read_text(encoding="utf-8")
    for needle in ("[setup]", "[round     3/3]", "[done] 3 rounds"):
        assert needle in term and needle in log
    assert "session start: fresh start" in log and "session end: finished" in log
    assert "--attack sign_flip" in log  # the command line

    # run_info.json
    info = _json(out / "run_info.json")
    assert info["status"] == "finished" and info["resumed"] is False and info["error"] is None
    assert info["command"][-len(argv):] == argv
    for key in ("dataset", "beta", "method", "attack", "malicious", "rounds", "eval_every", "seed",
                "c_t", "lr", "batch_size", "local_steps", "num_clients", "clients_per_round"):
        assert info["config"][key] == getattr(cfg, key), key
    git = info["git"]
    assert set(git) >= {"branch", "commit", "uncommitted_changes"}
    if shutil.which("git") and git["commit"] is not None:
        assert len(git["commit"]) == 40 and isinstance(git["uncommitted_changes"], bool)
    env = info["environment"]
    assert env["python"] == platform.python_version() and env["torch"] == torch.__version__
    assert "cuda" in env and env["device"] == "cpu" and env["device_name"]
    start, end = datetime.fromisoformat(info["start_time"]), datetime.fromisoformat(info["end_time"])
    assert start <= end
    assert info["train_time_s"] > 0 and info["eval_time_s"] > 0
    assert info["total_time_s"] >= info["train_time_s"] + info["eval_time_s"]
    evals = [r for r in _metrics(out) if r["test_acc"] != ""]
    assert info["rounds_done"] == 3 and info["final_round"] == 3
    assert info["final_test_acc"] == pytest.approx(float(evals[-1]["test_acc"]))
    assert info["best_test_acc"] == pytest.approx(max(float(r["test_acc"]) for r in evals))
    assert len(info["sessions"]) == 1 and info["sessions"][0]["status"] == "finished"

    # data_info.json
    data = _json(out / "data_info.json")
    config = _json(out / "config.json")
    assert data["root"]["size"] == 20 and data["root"]["class_counts"] == [5] * K
    assert data["malicious"]["ids"] == config["malicious_ids"] and data["malicious"]["num"] == 2
    assert len(data["clients"]) == cfg.num_clients
    assert sum(c["num_samples"] for c in data["clients"]) + data["root"]["size"] == data["num_train"]
    with open(out / "partition.csv", newline="") as f:
        part = list(csv.DictReader(f))
    for c, row in zip(data["clients"], part):
        counts = [int(row[f"c{k}"]) for k in range(K)]
        assert c["num_samples"] == sum(counts)
        assert c["num_classes"] == sum(1 for n in counts if n > 0)
        assert c["malicious"] == bool(int(row["malicious"]))

    # final_model.pt: theta and a state_dict that loads into the model
    fm = torch.load(out / "final_model.pt", map_location="cpu", weights_only=False)
    assert fm["round"] == 3 and torch.equal(fm["theta"], res["theta"])
    model = build_model(cfg.dataset, seed=123, num_classes=K)
    model.load_state_dict(fm["state_dict"])
    assert torch.equal(get_params(model), res["theta"])

    # accuracy.png is a real PNG
    png = (out / "accuracy.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000


def test_logging_output_goes_to_console_log(tmp_path, capsys):
    """Log records (the partition / root-set stats) land in console.log, like in the CLI."""
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    old_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    try:
        run(_cfg(tmp_path / "run", rounds=1), data=_tiny_data(), verbose=False)
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)
    assert handler.stream is sys.stdout  # restored after the run
    log = (tmp_path / "run" / "console.log").read_text(encoding="utf-8")
    assert "dirichlet partition" in log and "root set:" in log


# ----------------------------------------------------------------------------- resume
def test_resume_appends_instead_of_overwriting(tmp_path):
    out = tmp_path / "run"
    data = _tiny_data()
    run(_cfg(out, rounds=2, eval_every=1), data=data, verbose=True)
    log1 = (out / "console.log").read_text(encoding="utf-8")
    info1 = _json(out / "run_info.json")

    res = run(_cfg(out, rounds=4, eval_every=1, resume=True), data=data, verbose=True)
    log2 = (out / "console.log").read_text(encoding="utf-8")
    assert log2.startswith(log1) and len(log2) > len(log1)
    assert "session start: resume from round 2" in log2[len(log1):]
    assert "[resume] from round 2" in log2 and "[round     4/4]" in log2

    info2 = _json(out / "run_info.json")
    assert info2["status"] == "finished" and info2["resumed"] is True and info2["num_sessions"] == 2
    assert info2["sessions"][0] == info1["sessions"][0]  # first session kept as it was
    s2 = info2["sessions"][1]
    assert s2["kind"] == "resume" and s2["resumed_from_round"] == 2 and s2["status"] == "finished"
    assert info2["start_time"] == info1["start_time"] and info2["rounds_done"] == 4
    assert info2["config"]["rounds"] == 4
    assert info2["train_time_s"] == pytest.approx(
        info2["sessions"][0]["train_time_s"] + s2["train_time_s"], abs=1e-3)
    assert [int(r["round"]) for r in _metrics(out)] == [0, 1, 2, 3, 4]
    fm = torch.load(out / "final_model.pt", map_location="cpu", weights_only=False)
    assert fm["round"] == 4 and torch.equal(fm["theta"], res["theta"])


def test_resume_of_finished_run_changes_nothing_but_console_log(tmp_path):
    out = tmp_path / "run"
    data = _tiny_data()
    run(_cfg(out, rounds=2), data=data, verbose=False)
    before = _snapshot(out)
    run(_cfg(out, rounds=2, resume=True), data=data, verbose=False)
    after = _snapshot(out)
    assert set(after) == set(before)
    assert {k: v for k, v in after.items() if k != "console.log"} == \
           {k: v for k, v in before.items() if k != "console.log"}
    assert after["console.log"].startswith(before["console.log"])
    assert b"nothing to do" in after["console.log"][len(before["console.log"]):]


def test_never_overwrites_an_existing_run(tmp_path):
    out = tmp_path / "run"
    data = _tiny_data()
    run(_cfg(out, rounds=2), data=data, verbose=False)
    before = _snapshot(out)
    with pytest.raises(FileExistsError, match="already holds a run"):
        run(_cfg(out, rounds=2), data=data, verbose=False)
    assert _snapshot(out) == before

    # finished run whose checkpoint is gone (e.g. results copied without *.pt): --resume
    # cannot continue it and must not start over on top of it
    (out / "checkpoint.pt").unlink()
    before = _snapshot(out)
    with pytest.raises(FileExistsError, match="finished run"):
        run(_cfg(out, rounds=4, resume=True), data=data, verbose=False)
    assert _snapshot(out) == before

    # any single runner file is enough to refuse a fresh start
    other = tmp_path / "other"
    other.mkdir()
    (other / "console.log").write_text("earlier output\n")
    with pytest.raises(FileExistsError):
        run(_cfg(other, rounds=1), data=data, verbose=False)
    assert (other / "console.log").read_text() == "earlier output\n"


# ----------------------------------------------------------------------------- crash
def test_crash_is_recorded_and_resume_continues(tmp_path, monkeypatch):
    out = tmp_path / "run"
    data = _tiny_data()
    real_evaluate = runner.evaluate
    calls = {"n": 0}

    def flaky_evaluate(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:  # evaluation of round 2 (after 0, 1), before its checkpoint
            print("boom on stderr", file=sys.stderr)
            raise RuntimeError("boom")
        return real_evaluate(*a, **k)

    monkeypatch.setattr(runner, "evaluate", flaky_evaluate)
    stdout, stderr = sys.stdout, sys.stderr
    kw = dict(rounds=4, eval_every=1, checkpoint_every=1)
    with pytest.raises(RuntimeError, match="boom"):
        run(_cfg(out, **kw), data=data, verbose=True)
    assert sys.stdout is stdout and sys.stderr is stderr  # streams restored

    log = (out / "console.log").read_text(encoding="utf-8")
    assert "boom on stderr" in log  # stderr is captured
    assert "Traceback" in log and "RuntimeError: boom" in log and "session end: crashed" in log
    assert "[round     1/4]" in log
    info = _json(out / "run_info.json")
    assert info["status"] == "crashed" and "boom" in info["error"] and info["end_time"]
    assert info["final_round"] == 1 and info["final_test_acc"] is not None
    assert not (out / "final_model.pt").exists()
    assert (out / "checkpoint.pt").exists() and (out / "accuracy.png").exists()

    monkeypatch.setattr(runner, "evaluate", real_evaluate)
    run(_cfg(out, resume=True, **kw), data=data, verbose=True)
    log2 = (out / "console.log").read_text(encoding="utf-8")
    assert log2.startswith(log) and "session start: resume from round 1" in log2  # last checkpoint
    info2 = _json(out / "run_info.json")
    assert info2["status"] == "finished" and info2["resumed"] is True and info2["error"] is None
    assert [s["status"] for s in info2["sessions"]] == ["crashed", "finished"]
    assert info2["sessions"][0]["error"] == "RuntimeError: boom"
    assert (out / "final_model.pt").exists() and info2["rounds_done"] == 4


def test_resume_without_checkpoint_keeps_old_metrics(tmp_path):
    """A run that crashed before its first checkpoint restarts at round 0 under --resume,
    keeping the old metrics.csv and appending to console.log / run_info.json."""
    out = tmp_path / "run"
    data = _tiny_data()
    run(_cfg(out, rounds=2), data=data, verbose=False)
    (out / "checkpoint.pt").unlink()
    (out / "final_model.pt").unlink()
    info = _json(out / "run_info.json")
    info["status"] = info["sessions"][0]["status"] = "crashed"
    (out / "run_info.json").write_text(json.dumps(info))
    old_metrics = (out / "metrics.csv").read_bytes()

    run(_cfg(out, rounds=2, resume=True), data=data, verbose=False)
    backups = list(out.glob("metrics.before_restart_*.csv"))
    assert len(backups) == 1 and backups[0].read_bytes() == old_metrics
    info2 = _json(out / "run_info.json")
    assert info2["status"] == "finished" and len(info2["sessions"]) == 2
    assert info2["sessions"][1]["kind"] == "fresh" and "no checkpoint" in info2["sessions"][1]["note"]


# ----------------------------------------------------------------------------- sweep
def _load_sweep():
    spec = importlib.util.spec_from_file_location("overnight_sweep", REPO / "scripts" / "overnight_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sweep_writes_plan_json(tmp_path):
    sweep = _load_sweep()
    runs = sweep.run_list("round2")
    args = argparse.Namespace(plan="round2", rounds=None, device="cpu", data_dir="/tmp/data", limit=2,
                              python="python")
    path = sweep.write_plan(tmp_path, runs, 2, args)
    plan = _json(path)
    assert plan["plan"] == "round2"
    assert plan["num_runs"] == len(runs) and plan["num_selected"] == 2
    assert [r["name"] for r in plan["runs"]] == [r["name"] for r in runs]
    assert [r["selected"] for r in plan["runs"]] == [True, True] + [False] * (len(runs) - 2)
    first = plan["runs"][0]
    assert first["command"][first["command"].index("--out") + 1] == str(tmp_path / first["name"])
    assert first["command"][first["command"].index("--c-t") + 1] == "0.25"
    assert plan["total_rounds"] == 2000  # two selected runs x 1000

    sweep.write_plan(tmp_path, runs, 2, args)  # same plan again: no backup
    assert list(tmp_path.glob("plan.*.json")) == []
    runs = sweep.run_list("round2", rounds=5)  # a different plan: the old one is kept
    args.rounds = 5
    sweep.write_plan(tmp_path, runs, 2, args)
    assert len(list(tmp_path.glob("plan.*.json"))) == 1
    assert _json(path)["runs"][0]["rounds"] == 5 and _json(path)["settings"]["rounds_override"] == 5


def test_sweep_out_alias_and_resume_detection(tmp_path, capsys):
    sweep = _load_sweep()
    kaggle_like = tmp_path / "kaggle" / "working" / "results" / "sweep"
    sweep.main(["--dry-run", "--limit", "1", "--out", str(kaggle_like)])
    assert str(kaggle_like) in capsys.readouterr().out
    assert not kaggle_like.exists()  # a dry run writes nothing

    d = tmp_path / "r"
    assert not sweep.has_run_files(d)
    d.mkdir()
    (d / "runner.log").write_text("x")
    assert not sweep.has_run_files(d)  # only the sweep's own log
    (d / "console.log").write_text("x")
    assert sweep.has_run_files(d)


# ----------------------------------------------------------------------------- git
@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
@pytest.mark.parametrize("path,ignored", [
    ("results/sweep/x/final_model.pt", True),
    ("results/sweep/x/checkpoint.pt", True),
    ("results/sweep/x/checkpoint.pt.tmp", True),
    ("results/sweep/x/console.log", False),
    ("results/sweep/x/runner.log", False),
    ("results/sweep/x/run_info.json", False),
    ("results/sweep/x/data_info.json", False),
    ("results/sweep/x/config.json", False),
    ("results/sweep/x/metrics.csv", False),
    ("results/sweep/x/partition.csv", False),
    ("results/sweep/x/accuracy.png", False),
    ("results/sweep/plan.json", False),
    ("results/sweep/summary.csv", False),
    ("results/sweep/sweep.log", False),
    ("results/sweep/errors.log", False),
    ("results/sweep/plots/sign_flip_0.3.png", False),
])
def test_gitignore_rules(path, ignored):
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=REPO)
    if r.returncode not in (0, 1):
        pytest.skip("not a git checkout")
    assert (r.returncode == 0) == ignored, path


# ----------------------------------------------------------------------------- --ct
def test_ct_is_an_alias_of_c_t():
    from fedrag.runner.run import parse_args
    assert parse_args(["--ct", "0.25"]).c_t == 0.25
    assert parse_args(["--c-t", "0.25"]).c_t == 0.25
    assert parse_args([]).c_t == 0.5
    # one option, not two: the last spelling given wins, and --help lists them together
    assert parse_args(["--c-t", "0.1", "--ct", "1.0"]).c_t == 1.0
    assert parse_args(["--ct", "1.0", "--c-t", "0.1"]).c_t == 0.1


def test_ct_in_default_out_name_only_when_not_default():
    from fedrag.runner.run import parse_args
    base = ["--dataset", "emnist_balanced", "--method", "br_drag", "--attack", "min_max", "--beta", "0.5"]
    assert parse_args(base).out == "results/emnist_balanced_br_drag_min_max_b0.5_s0"
    assert parse_args(base + ["--ct", "0.5"]).out == "results/emnist_balanced_br_drag_min_max_b0.5_s0"
    assert parse_args(base + ["--ct", "0.25"]).out == "results/emnist_balanced_br_drag_min_max_b0.5_ct0.25_s0"
    assert parse_args(base + ["--ct", "1.0", "--out", "x"]).out == "x"  # explicit --out wins


def test_ct_reaches_br_drag(tmp_path, monkeypatch):
    """--ct on the command line ends up as BRDRAG.c_t and changes the lambdas it computes."""
    from fedrag.runner.run import parse_args
    from fedrag.strategies.br_drag import BRDRAG

    seen = []
    real_init = BRDRAG.__init__

    def spy_init(self, *a, **k):
        real_init(self, *a, **k)
        seen.append(self.c_t)

    monkeypatch.setattr(BRDRAG, "__init__", spy_init)
    lam_max = {}
    for ct in ("0.25", "1.0"):
        out = tmp_path / ct
        cfg = parse_args(["--method", "br_drag", "--attack", "sign_flip", "--ct", ct, "--rounds", "2",
                          "--eval-every", "1", "--num-clients", "6", "--clients-per-round", "3",
                          "--malicious", "0.34", "--root-size", "20", "--beta", "1.0", "--device", "cpu",
                          "--out", str(out)])
        res = run(cfg, data=_tiny_data(), verbose=False)
        assert _json(out / "run_info.json")["config"]["c_t"] == float(ct)
        lam_max[ct] = max(r["lambda_max"] for r in res["metrics"][1:])
    assert seen == [0.25, 1.0]
    # lambda = c_t * (1 - cos) lies in [0, 2 c_t]
    assert lam_max["0.25"] <= 0.5 + 1e-6 and lam_max["1.0"] > 0.5
