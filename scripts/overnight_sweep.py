"""Overnight sweep: run a prioritised list of experiments one after another.

Run from the repo root:
    python scripts/overnight_sweep.py                       # plan round1 -> results/sweep
    python scripts/overnight_sweep.py --plan round2         # plan round2 -> results/sweep2
    python scripts/overnight_sweep.py --plan round2 --dry-run   # print the run list, run nothing
    python scripts/overnight_sweep.py --rounds 3 --limit 2 --out results/sweep_test
    python scripts/overnight_sweep.py --plan round2 --out /kaggle/working/results/sweep2 \
        --data-dir /tmp/data --device cuda                  # Kaggle

Plans (--plan, see PLANS below), all emnist_balanced, eval every 10:
- round1 (default, the original sweep): beta 0.5, seed 0, c_t 0.5, 1000 rounds; runs are named
  <method>_<attack>_<malicious>.
- round2: c_t ablation, beta 0.1, three seeds and 2000-round runs; runs are named
  <method>_<attack>_<malicious>[_b<beta>][_ct<c_t>]_s<seed> (_b / _ct only when not 0.5).
Each run goes through ``python -m fedrag.runner.run`` with output in <out>/<run name> (--out,
alias --out-root; default results/sweep for round1, results/sweep2 for round2; relative paths
are under the repo, absolute ones such as /kaggle/working/... are used as given). --rounds
overrides every run's rounds (for quick tests). Runs go in priority order so the most important
ones finish even if the night is cut short.

- <out>/plan.json, written at the start: the sweep settings and the full run list with each
  run's command (runs beyond --limit are listed with "selected": false). A different earlier
  plan.json is kept as plan.<time>.json.

- Finished runs (metrics.csv reaches the final round with an evaluation) are skipped, so the
  script can simply be started again after an interruption.
- A run folder that already holds runner files (a checkpoint, or a run that crashed before its
  first checkpoint) is continued with --resume; the runner appends to its console.log and
  run_info.json and never overwrites a finished run.
- A crashed run is logged to <out-root>/errors.log (exit code + output tail) and the sweep moves on.
- <out-root>/sweep.log gets start/end times, final accuracy and an ETA after every run (from
  the measured seconds per round of the runs finished so far x the rounds still to run).
- Each run writes console.log, run_info.json, data_info.json, final_model.pt and accuracy.png
  itself (see fedrag.runner.run). Its output is also appended to <run dir>/runner.log, which
  additionally catches failures before the runner starts (e.g. an import error).
- At the end, scripts/summarize_sweep.py writes summary.csv and plots/.
Ctrl+C stops the sweep (the interrupted run resumes from its last checkpoint next time).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_sweep import summarize  # noqa: E402

DATASET, EVAL_EVERY = "emnist_balanced", 10
DEFAULT_BETA, DEFAULT_CT = 0.5, 0.5

# round1: (attack, malicious fraction, methods) in priority order; beta 0.5, seed 0, 1000
# rounds. "none" has no attackers, so its fraction is recorded as 0.0.
ALL4 = ["fedavg", "br_drag", "fltrust_paper", "fltrust_cao"]
MAIN3 = ["fedavg", "br_drag", "fltrust_paper"]
PLAN = [
    ("sign_flip", 0.3, ALL4),
    ("none", 0.0, MAIN3),
    ("min_max", 0.3, ALL4),
    ("noise", 0.3, MAIN3),
    ("label_flip", 0.3, MAIN3),
    ("min_sum", 0.3, MAIN3),
    ("sign_flip", 0.6, MAIN3),
    ("min_max", 0.6, MAIN3),
]


def _run(method, attack, malicious, *, beta=DEFAULT_BETA, seed=0, c_t=None, rounds=1000,
         legacy_name=False) -> dict:
    """One run. ``c_t=None`` leaves --c-t out of the command (runner default 0.5)."""
    name = f"{method}_{attack}_{malicious}"
    if not legacy_name:
        name += "" if beta == DEFAULT_BETA else f"_b{beta}"
        name += "" if c_t is None or c_t == DEFAULT_CT else f"_ct{c_t}"
        name += f"_s{seed}"
    return {"name": name, "method": method, "attack": attack, "malicious": malicious,
            "beta": beta, "seed": seed, "c_t": c_t, "rounds": rounds}


def plan_round1() -> list[dict]:
    """The original sweep (names without seed / beta, so results/sweep is skipped as finished)."""
    return [_run(m, a, mal, legacy_name=True) for a, mal, methods in PLAN for m in methods]


def plan_round2() -> list[dict]:
    runs = []
    # a) BR-DRAG c_t ablation under the strongest attack
    runs += [_run("br_drag", "min_max", 0.6, c_t=ct, rounds=1000) for ct in (0.25, 1.0)]
    # b) more heterogeneous data (beta 0.1)
    runs += [_run(m, "sign_flip", 0.3, beta=0.1, rounds=1000) for m in ("fedavg", "br_drag", "fltrust_cao")]
    # c) 60% malicious, three seeds, 2000 rounds
    runs += [_run(m, a, 0.6, seed=seed, rounds=2000)
             for m in ("br_drag", "fltrust_cao") for a in ("sign_flip", "min_max") for seed in (0, 1, 2)]
    # d) FedAvg reference for c)
    runs += [_run("fedavg", a, 0.6, rounds=2000) for a in ("sign_flip", "min_max")]
    return runs


PLANS = {"round1": (plan_round1, "results/sweep"), "round2": (plan_round2, "results/sweep2")}


def run_list(plan: str = "round1", rounds: int | None = None) -> list[dict]:
    """The plan's runs in order; ``rounds`` (if given) replaces every run's rounds."""
    runs = PLANS[plan][0]()
    if rounds is not None:
        runs = [{**r, "rounds": rounds} for r in runs]
    return runs


def final_round_done(run_dir: Path, rounds: int) -> bool:
    """True if metrics.csv has an evaluated row at round >= ``rounds``."""
    path = run_dir / "metrics.csv"
    if not path.exists():
        return False
    with open(path, newline="") as f:
        return any(int(r["round"]) >= rounds and r.get("test_acc") not in ("", None)
                   for r in csv.DictReader(f))


def final_acc(run_dir: Path) -> str:
    try:
        with open(run_dir / "metrics.csv", newline="") as f:
            evals = [r for r in csv.DictReader(f) if r.get("test_acc") not in ("", None)]
        return f"{float(evals[-1]['test_acc']):.2%} at round {evals[-1]['round']}"
    except (OSError, IndexError, KeyError, ValueError):
        return "n/a"


class Logger:
    """Print and append to sweep.log with a timestamp."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def fmt_dur(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


def build_cmd(run: dict, args, out: Path, resume: bool) -> list[str]:
    cmd = [args.python, "-m", "fedrag.runner.run",
           "--dataset", DATASET, "--beta", str(run["beta"]), "--seed", str(run["seed"]),
           "--method", run["method"], "--attack", run["attack"], "--malicious", str(run["malicious"]),
           "--rounds", str(run["rounds"]), "--eval-every", str(EVAL_EVERY),
           "--out", str(out), "--data-dir", args.data_dir, "--device", args.device]
    if run.get("c_t") is not None:
        cmd += ["--c-t", str(run["c_t"])]
    return cmd + (["--resume"] if resume else [])


def has_run_files(run_dir: Path) -> bool:
    """True if the runner already wrote something in ``run_dir`` (runner.log is the sweep's own)."""
    return run_dir.is_dir() and any(p.name != "runner.log" for p in run_dir.iterdir())


def write_plan(root: Path, all_runs: list[dict], selected: int, args) -> Path:
    """Write <root>/plan.json (settings + every run with its command); keep a different old one."""
    plan = {
        "plan": getattr(args, "plan", "round1"),
        "settings": {"dataset": DATASET, "eval_every": EVAL_EVERY, "rounds_override": args.rounds,
                     "device": args.device, "data_dir": args.data_dir, "out": str(root),
                     "limit": args.limit},
        "total_rounds": sum(r["rounds"] for r in all_runs[:selected]),
        "num_runs": len(all_runs),
        "num_selected": selected,
        "runs": [{**run, "order": i, "selected": i <= selected, "out": str(root / run["name"]),
                  "command": build_cmd(run, args, root / run["name"], resume=False)}
                 for i, run in enumerate(all_runs, 1)],
    }
    path = root / "plan.json"
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            same = {k: v for k, v in old.items() if k != "created"} == plan
        except (OSError, ValueError):
            same = False
        if not same:
            os.replace(path, root / f"plan.{datetime.fromtimestamp(path.stat().st_mtime):%Y%m%d-%H%M%S}.json")
    plan = {"created": datetime.now().astimezone().isoformat(timespec="seconds"), **plan}
    tmp = path.with_name("plan.json.tmp")
    tmp.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def execute(cmd: list[str], log_path: Path) -> tuple[int, list[str]]:
    """Run ``cmd``, streaming its output to ``log_path``; return (exit code, last 40 lines)."""
    env = dict(os.environ, PYTHONPATH=str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONUNBUFFERED="1")
    tail: deque[str] = deque(maxlen=40)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in proc.stdout:
                log.write(line)
                tail.append(line.rstrip("\n"))
            return proc.wait(), list(tail)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--plan", choices=sorted(PLANS), default="round1",
                   help="which run list: round1 (original sweep) or round2")
    p.add_argument("--rounds", type=int, default=None,
                   help="override every run's rounds (default: the plan's, 1000 or 2000)")
    p.add_argument("--out", "--out-root", dest="out_root", default=None,
                   help="sweep folder (default results/sweep for round1, results/sweep2 for round2; "
                        "relative: under the repo; absolute paths are used as given)")
    p.add_argument("--limit", type=int, default=None, help="only the first N runs of the list")
    p.add_argument("--dry-run", action="store_true", help="print the run list and planned actions, run nothing")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--python", default=sys.executable, help="interpreter for the runner")
    args = p.parse_args(argv)

    root = Path(args.out_root or PLANS[args.plan][1])
    if not root.is_absolute():
        root = REPO / root
    all_runs = run_list(args.plan, args.rounds)
    runs = all_runs[: args.limit]

    if args.dry_run:
        print(f"plan {args.plan}: {len(runs)} runs, {sum(r['rounds'] for r in runs)} rounds in total, "
              f"output under {root}")
        for i, run in enumerate(runs, 1):
            out = root / run["name"]
            if final_round_done(out, run["rounds"]):
                action = "skip (finished)"
            elif has_run_files(out):
                action = "resume"
            else:
                action = "run"
            print(f"  {i:2d}. {run['name']:38s} {run['rounds']:>5d} rounds  {action}")
        print("example command:", " ".join(build_cmd(runs[0], args, root / runs[0]["name"], False)))
        return

    root.mkdir(parents=True, exist_ok=True)
    write_plan(root, all_runs, len(runs), args)
    log = Logger(root / "sweep.log")
    log(f"sweep start: plan {args.plan}, {len(runs)} runs, {sum(r['rounds'] for r in runs)} rounds "
        f"in total, device={args.device}, out={root}")
    timed: list[tuple[float, int]] = []  # (seconds, rounds) of runs trained from scratch
    n_ok = n_skip = n_fail = 0

    for i, run in enumerate(runs, 1):
        out = root / run["name"]
        tag = f"[{i}/{len(runs)}] {run['name']}"
        if final_round_done(out, run["rounds"]):
            log(f"{tag}: already finished ({final_acc(out)}), skipped")
            n_skip += 1
            continue
        resume = has_run_files(out)
        out.mkdir(parents=True, exist_ok=True)
        log(f"{tag}: start{' (resuming from checkpoint)' if resume else ''}")
        t0 = time.time()
        try:
            code, tail = execute(build_cmd(run, args, out, resume), out / "runner.log")
        except KeyboardInterrupt:
            log(f"{tag}: interrupted by user after {fmt_dur(time.time() - t0)}; stopping the sweep")
            break
        except Exception as e:  # failed to even launch the process
            code, tail = -1, [repr(e)]
        dur = time.time() - t0

        if code == 0 and final_round_done(out, run["rounds"]):
            n_ok += 1
            log(f"{tag}: done in {fmt_dur(dur)}, final acc {final_acc(out)}")
            # only full (non-resumed) runs are representative of a run's length
            if not resume:
                timed.append((dur, run["rounds"]))
        else:
            n_fail += 1
            log(f"{tag}: FAILED (exit {code}) after {fmt_dur(dur)}; see errors.log")
            with open(root / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"===== {datetime.now():%Y-%m-%d %H:%M:%S} {run['name']} exit={code}\n")
                f.write("command: " + " ".join(build_cmd(run, args, out, resume)) + "\n")
                f.write("\n".join(tail) + "\n\n")

        remaining = len(runs) - i
        if remaining and (timed or n_ok):
            # seconds per round, so 1000- and 2000-round runs are estimated correctly
            per_round = (sum(sec for sec, _ in timed) / sum(n for _, n in timed) if timed
                         else dur / run["rounds"])
            left = per_round * sum(r["rounds"] for r in runs[i:])
            eta = datetime.now() + timedelta(seconds=left)
            log(f"ETA: {remaining} run(s) left, ~{per_round:.2f} s/round -> {fmt_dur(left)} "
                f"-> about {eta:%Y-%m-%d %H:%M} (upper bound: skipped/resumed runs are faster)")

    log(f"sweep end: {n_ok} finished, {n_skip} skipped, {n_fail} failed")
    summarize(root)


if __name__ == "__main__":
    main()
