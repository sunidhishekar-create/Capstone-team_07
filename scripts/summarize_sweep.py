"""Summarise a sweep of runner outputs: CSV tables and one accuracy plot per attack setting.

Run from the repo root (also called automatically at the end of overnight_sweep.py):
    python scripts/summarize_sweep.py results/sweep2      # or --root results/sweep2
    python scripts/summarize_sweep.py                     # default results/sweep

Reads every <root>/<run>/metrics.csv (+ config.json for method / attack / malicious / beta /
c_t / seed) and writes
- <root>/summary.csv: one row per run with final and best test accuracy, rounds to 50% accuracy
  (first evaluated round with test_acc >= 0.5; blank if never reached) and run status.
- <root>/summary_seeds.csv: one row per setting (everything except the seed), with the mean and
  sample standard deviation (ddof=1) over its seeds of final / best accuracy and rounds to 50%.
  A setting with one seed has a blank std.
- <root>/plots/<attack>_<malicious>[_b<beta>].png: test accuracy vs round, one line per method
  (and c_t variant). Settings with several seeds show the mean over seeds with a +-1 std band.
The printed table shows "mean ± std (n seeds)" where a setting has several seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

METHOD_ORDER = ["fedavg", "br_drag", "fltrust_paper", "fltrust_cao"]
# Fixed categorical slots: a series keeps its color in every plot, whatever else is present.
METHOD_COLORS = {"fedavg": "#2a78d6", "br_drag": "#eb6834", "fltrust_paper": "#1baf7a",
                 "fltrust_cao": "#eda100"}
# c_t variants of BR-DRAG take the next fixed slots (magenta, violet), never generated hues.
VARIANT_COLORS = {("br_drag", 0.25): "#e87ba4", ("br_drag", 1.0): "#6250d6"}
ATTACK_ORDER = ["none", "sign_flip", "min_max", "noise", "label_flip", "min_sum"]
TARGET_ACC = 0.5
DEFAULT_BETA, DEFAULT_CT = 0.5, 0.5
REACH = f"rounds_to_{int(TARGET_ACC * 100)}pct"


def _float(x: str):
    return float(x) if x not in ("", None) else None


def load_run(run_dir: Path) -> dict | None:
    """Parse one run directory; None if it has no metrics yet."""
    mpath, cpath = run_dir / "metrics.csv", run_dir / "config.json"
    if not mpath.exists() or not cpath.exists():
        return None
    cfg = json.loads(cpath.read_text())["config"]
    with open(mpath, newline="") as f:
        rows = list(csv.DictReader(f))
    evals = [(int(r["round"]), _float(r["test_acc"]), _float(r["test_loss"]))
             for r in rows if r.get("test_acc") not in ("", None)]
    if not evals:
        return None
    last_round = max(int(r["round"]) for r in rows)
    best = max(evals, key=lambda e: e[1])
    reach = next((rnd for rnd, acc, _ in evals if acc >= TARGET_ACC), None)
    secs = [_float(r.get("time_round_s")) for r in rows if _float(r.get("time_round_s")) is not None]
    return {
        "run": run_dir.name,
        "method": cfg["method"],
        # no attackers exist without an attack, whatever --malicious said
        "attack": cfg["attack"],
        "malicious": 0.0 if cfg["attack"] == "none" else cfg["malicious"],
        "beta": cfg.get("beta", DEFAULT_BETA),
        "c_t": cfg.get("c_t", DEFAULT_CT),
        "seed": cfg.get("seed", 0),
        "rounds_planned": cfg["rounds"],
        "rounds_done": last_round,
        "status": "finished" if last_round >= cfg["rounds"] else "partial",
        "final_round": evals[-1][0],
        "final_test_acc": evals[-1][1],
        "final_test_loss": evals[-1][2],
        "best_test_acc": best[1],
        "best_round": best[0],
        REACH: reach,
        "train_s_per_round": sum(secs) / len(secs) if secs else None,
        "curve": [(rnd, acc) for rnd, acc, _ in evals],
    }


def setting_key(r: dict) -> tuple:
    """Everything that defines a setting except the seed."""
    return (r["method"], r["attack"], r["malicious"], r["beta"], r["c_t"], r["rounds_planned"])


def _sort_key(r: dict):
    a = ATTACK_ORDER.index(r["attack"]) if r["attack"] in ATTACK_ORDER else len(ATTACK_ORDER)
    m = METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else len(METHOD_ORDER)
    return (r["malicious"], a, -r["beta"], m, r["c_t"], r["rounds_planned"], r.get("seed", 0))


def _cell(v):
    return "" if v is None else round(v, 4) if isinstance(v, float) else v


def write_summary(runs: list[dict], path: Path) -> None:
    cols = [k for k in runs[0] if k != "curve"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({k: _cell(r[k]) for k in cols})


def _mean_std(xs: list) -> tuple:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else None)


def aggregate_seeds(runs: list[dict]) -> list[dict]:
    """One row per setting: mean and sample std (ddof=1) over its seeds."""
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault(setting_key(r), []).append(r)
    out = []
    for key, group in groups.items():
        first = group[0]
        row = {k: first[k] for k in ("method", "attack", "malicious", "beta", "c_t", "rounds_planned")}
        row.update(n_seeds=len(group), seeds=" ".join(str(r["seed"]) for r in sorted(group, key=lambda r: r["seed"])),
                   all_finished=all(r["status"] == "finished" for r in group))
        for metric in ("final_test_acc", "best_test_acc", REACH):
            mean, std = _mean_std([r[metric] for r in group])
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean, std
        row[f"{REACH}_n_reached"] = sum(r[REACH] is not None for r in group)
        row["runs"] = " ".join(r["run"] for r in group)
        out.append(row)
    return out


def write_seed_summary(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(v) for k, v in r.items()})


def series_label(r: dict) -> str:
    return r["method"] + ("" if r["c_t"] == DEFAULT_CT else f" c_t={r['c_t']}")


def series_color(r: dict) -> str:
    if r["c_t"] != DEFAULT_CT:
        return VARIANT_COLORS.get((r["method"], r["c_t"]), "#8a8984")
    return METHOD_COLORS.get(r["method"], "#8a8984")


def _mean_curve(group: list[dict]) -> tuple[list, list, list]:
    """Rounds evaluated in every seed, with the mean and std (0 for one seed) of test_acc there."""
    per_seed = [dict(r["curve"]) for r in group]
    rounds = sorted(set.intersection(*(set(c) for c in per_seed)))
    means, stds = [], []
    for rnd in rounds:
        vals = [c[rnd] for c in per_seed]
        means.append(statistics.fmean(vals))
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return rounds, means, stds


def plot_settings(runs: list[dict], out_dir: Path) -> list[Path]:
    """One PNG per (attack, malicious, beta): test accuracy vs round, one line per series
    (method + c_t + rounds); several seeds -> mean line with a +-1 std band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings: dict[tuple, list[dict]] = {}
    for r in runs:
        settings.setdefault((r["attack"], r["malicious"], r["beta"]), []).append(r)

    paths = []
    for (attack, mal, beta), group in settings.items():
        series: dict[tuple, list[dict]] = {}
        for r in sorted(group, key=_sort_key):
            series.setdefault((r["method"], r["c_t"], r["rounds_planned"]), []).append(r)
        labels = [series_label(g[0]) for g in series.values()]

        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=130)
        ends = []
        x_max = 0
        for (method, c_t, rounds), seeds in series.items():
            xs, ys, sd = _mean_curve(seeds)
            if not xs:
                continue
            label = series_label(seeds[0])
            if labels.count(label) > 1:  # same method + c_t at different lengths
                label += f" ({rounds} r)"
            if len(seeds) > 1:
                label += f" (mean of {len(seeds)} seeds)"
            color = series_color(seeds[0])
            if len(seeds) > 1:
                ax.fill_between(xs, [m - s for m, s in zip(ys, sd)], [m + s for m, s in zip(ys, sd)],
                                color=color, alpha=0.15, lw=0)
            ax.plot(xs, ys, color=color, lw=2, label=label, solid_capstyle="round")
            ax.plot(xs[-1], ys[-1], "o", ms=5, color=color, mec="white", mew=1.5, zorder=3)
            ends.append((ys[-1], xs[-1], label.split(" (")[0]))
            x_max = max(x_max, xs[-1])
        # direct labels at the line ends (text in ink; the colored dot carries identity),
        # spread vertically so series that end close together do not overlap
        ends.sort()
        gap, label_y = 0.055, []
        for y, _, _ in ends:
            label_y.append(max(y, label_y[-1] + gap) if label_y else y)
        shift = max(0.0, label_y[-1] - 0.98) if label_y else 0.0
        for (y, x, name), ly in zip(ends, label_y):
            ax.annotate(f"{name} {y:.1%}", xy=(x, y), xytext=(x + 0.02 * x_max, ly - shift),
                        textcoords="data", va="center", fontsize=8, color=ink)
        title = "no attack" if attack == "none" else f"{attack}, {mal:.0%} malicious"
        if beta != DEFAULT_BETA:
            title += f", beta {beta}"
        ax.set_title(f"EMNIST test accuracy: {title}", loc="left", fontsize=11, color=ink)
        ax.set_xlabel("round", color=muted)
        ax.set_ylabel("test accuracy", color=muted)
        ax.set_xlim(0, x_max * 1.22 if x_max else 1)  # room for the end labels
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color=grid, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=ink)
        fig.tight_layout()
        name = f"{attack}_{mal}" + ("" if beta == DEFAULT_BETA else f"_b{beta}")
        path = out_dir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def _plus_minus() -> str:
    """"±" unless the console cannot encode it (then "+-")."""
    try:
        "±".encode(sys.stdout.encoding or "ascii")
        return "±"
    except (UnicodeEncodeError, LookupError):
        return "+-"


def _pm(mean, std, n, pct=True) -> str:
    if mean is None:
        return "-"
    fmt = (lambda v: f"{v:.2%}") if pct else (lambda v: f"{v:.0f}")
    return fmt(mean) if n == 1 or std is None else f"{fmt(mean)} {_plus_minus()} {fmt(std)} (n={n})"


def summarize(root: Path | str = "results/sweep", verbose: bool = True) -> list[dict]:
    root = Path(root)
    runs = [r for d in sorted(root.iterdir()) if d.is_dir() and (r := load_run(d))] if root.exists() else []
    if not runs:
        if verbose:
            print(f"no finished evaluations under {root}")
        return []
    runs.sort(key=_sort_key)
    write_summary(runs, root / "summary.csv")
    seeds = aggregate_seeds(runs)
    write_seed_summary(seeds, root / "summary_seeds.csv")
    plots = plot_settings(runs, root / "plots")
    if verbose:
        print(f"{'method':14s} {'attack':11s} {'mal':>4s} {'beta':>4s} {'c_t':>4s} {'rounds':>6s} "
              f"{'final acc':>24s} {'best acc':>24s} {'to50%':>14s}")
        for s in seeds:
            n = s["n_seeds"]
            status = "" if s["all_finished"] else "  (partial)"
            print(f"{s['method']:14s} {s['attack']:11s} {s['malicious']:>4.1f} {s['beta']:>4.1f} "
                  f"{s['c_t']:>4.2f} {s['rounds_planned']:>6d} "
                  f"{_pm(s['final_test_acc_mean'], s['final_test_acc_std'], n):>24s} "
                  f"{_pm(s['best_test_acc_mean'], s['best_test_acc_std'], n):>24s} "
                  f"{_pm(s[REACH + '_mean'], s[REACH + '_std'], s[REACH + '_n_reached'], pct=False):>14s}"
                  f"{status}")
        print(f"wrote {root / 'summary.csv'}, {root / 'summary_seeds.csv'} and {len(plots)} plot(s) "
              f"in {root / 'plots'}")
    return runs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root", nargs="?", default=None, help="sweep results folder (default results/sweep)")
    p.add_argument("--root", dest="root_opt", default=None, help="same as the positional argument")
    a = p.parse_args()
    summarize(a.root or a.root_opt or "results/sweep")
