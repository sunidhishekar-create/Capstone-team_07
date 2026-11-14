"""Tests for scripts/overnight_sweep.py plans and scripts/summarize_sweep.py seed aggregation.

Run with:  PYTHONPATH=src pytest tests/scripts
"""

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(REPO / "scripts"))
    return mod


@pytest.fixture(scope="module")
def sweep():
    return _load("overnight_sweep")


@pytest.fixture(scope="module")
def summ():
    return _load("summarize_sweep")


# ----------------------------------------------------------------------------- plans
def test_round1_is_the_original_run_list(sweep):
    runs = sweep.run_list("round1")
    assert len(runs) == 26 and runs == sweep.run_list()  # still the default
    assert runs[0]["name"] == "fedavg_sign_flip_0.3"  # old names: results/sweep is recognised
    assert all(r["beta"] == 0.5 and r["seed"] == 0 and r["rounds"] == 1000 and r["c_t"] is None
               for r in runs)
    args = type("A", (), dict(python="py", data_dir="data", device="auto"))()
    cmd = sweep.build_cmd(runs[0], args, Path("out"), resume=False)
    assert "--c-t" not in cmd and cmd[cmd.index("--rounds") + 1] == "1000"
    assert cmd[cmd.index("--beta") + 1] == "0.5" and cmd[cmd.index("--seed") + 1] == "0"


def test_round2_run_list(sweep):
    runs = sweep.run_list("round2")
    names = [r["name"] for r in runs]
    assert len(set(names)) == len(names)
    key = [(r["method"], r["attack"], r["malicious"], r["beta"], r["c_t"], r["seed"], r["rounds"]) for r in runs]
    expected = (
        # a) c_t ablation
        [("br_drag", "min_max", 0.6, 0.5, ct, 0, 1000) for ct in (0.25, 1.0)]
        # b) beta 0.1
        + [(m, "sign_flip", 0.3, 0.1, None, 0, 1000) for m in ("fedavg", "br_drag", "fltrust_cao")]
        # c) 60% malicious, 3 seeds
        + [(m, a, 0.6, 0.5, None, s, 2000) for m in ("br_drag", "fltrust_cao")
           for a in ("sign_flip", "min_max") for s in (0, 1, 2)]
        # d) FedAvg at 60%
        + [("fedavg", a, 0.6, 0.5, None, 0, 2000) for a in ("sign_flip", "min_max")]
    )
    assert key == expected
    assert len(runs) == 2 + 3 + 12 + 2 == 19
    assert names[0] == "br_drag_min_max_0.6_ct0.25_s0"
    assert names[2] == "fedavg_sign_flip_0.3_b0.1_s0"
    assert "br_drag_sign_flip_0.6_s2" in names

    args = type("A", (), dict(python="py", data_dir="data", device="auto"))()
    cmd = sweep.build_cmd(runs[0], args, Path("out"), resume=True)
    assert cmd[cmd.index("--c-t") + 1] == "0.25" and cmd[-1] == "--resume"
    cmd = sweep.build_cmd(runs[2], args, Path("out"), resume=False)
    assert cmd[cmd.index("--beta") + 1] == "0.1" and "--c-t" not in cmd
    assert all(r["rounds"] == 7 for r in sweep.run_list("round2", rounds=7))


def test_round2_dry_run_defaults_to_sweep2(sweep, capsys):
    sweep.main(["--plan", "round2", "--dry-run"])
    text = capsys.readouterr().out
    assert "plan round2: 19 runs" in text
    assert str(REPO / "results" / "sweep2") in text
    assert "br_drag_min_max_0.6_ct0.25_s0" in text and " 2000 rounds" in text


# ----------------------------------------------------------------------------- summary
def _fake_run(root, name, *, seed, accs, method="br_drag", attack="sign_flip", mal=0.6, c_t=0.5, beta=0.5):
    d = root / name
    d.mkdir(parents=True)
    rounds = 10 * (len(accs) - 1)
    cfg = dict(method=method, attack=attack, malicious=mal, beta=beta, c_t=c_t, seed=seed, rounds=rounds)
    (d / "config.json").write_text(json.dumps({"config": cfg}))
    with open(d / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "test_acc", "test_loss", "time_round_s"])
        for i, a in enumerate(accs):
            w.writerow([10 * i, a, 1.0 - a, 0.5])


def test_summary_reports_mean_and_std_over_seeds(summ, tmp_path):
    _fake_run(tmp_path, "br_s0", seed=0, accs=[0.1, 0.4, 0.60])
    _fake_run(tmp_path, "br_s1", seed=1, accs=[0.1, 0.5, 0.70])
    _fake_run(tmp_path, "br_s2", seed=2, accs=[0.1, 0.6, 0.80])
    _fake_run(tmp_path, "fa_s0", seed=0, accs=[0.1, 0.1, 0.10], method="fedavg")
    _fake_run(tmp_path, "br_ct", seed=0, accs=[0.1, 0.2, 0.30], c_t=0.25)

    rows = summ.aggregate_seeds(summ.summarize(tmp_path, verbose=False))
    by = {(r["method"], r["c_t"]): r for r in rows}
    br = by[("br_drag", 0.5)]
    assert br["n_seeds"] == 3 and br["seeds"] == "0 1 2"
    assert br["final_test_acc_mean"] == pytest.approx(0.7)
    assert br["final_test_acc_std"] == pytest.approx(0.1)  # sample std of 0.6, 0.7, 0.8
    assert br["rounds_to_50pct_mean"] == pytest.approx(40 / 3)  # reached at rounds 20, 10, 10
    assert br["rounds_to_50pct_n_reached"] == 3
    assert by[("fedavg", 0.5)]["n_seeds"] == 1 and by[("fedavg", 0.5)]["final_test_acc_std"] is None
    assert by[("br_drag", 0.25)]["n_seeds"] == 1  # different c_t: its own setting

    with open(tmp_path / "summary_seeds.csv", newline="") as f:
        seeds_csv = list(csv.DictReader(f))
    assert len(seeds_csv) == 3
    with open(tmp_path / "summary.csv", newline="") as f:
        per_run = list(csv.DictReader(f))
    assert len(per_run) == 5 and {"beta", "c_t", "seed"} <= set(per_run[0])
    assert (tmp_path / "plots" / "sign_flip_0.6.png").exists()


def test_summary_cli_takes_the_folder_as_argument(tmp_path):
    _fake_run(tmp_path, "a", seed=0, accs=[0.1, 0.5])
    _fake_run(tmp_path, "b", seed=1, accs=[0.1, 0.7])
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "summarize_sweep.py"), str(tmp_path)],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "(n=2)" in r.stdout and ("±" in r.stdout or "+-" in r.stdout)
    assert (tmp_path / "summary.csv").exists() and (tmp_path / "summary_seeds.csv").exists()
