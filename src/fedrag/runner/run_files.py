"""Per-run output files besides metrics.csv / checkpoint.pt: console.log, run_info.json,
data_info.json, final_model.pt and accuracy.png.

``RunRecorder`` is a context manager around one session of a run (a fresh start or one
``--resume``). On entry it tees stdout / stderr (and logging handlers that write to them) into
``console.log``, opened in append mode and flushed on every write so it survives a crash, and
writes ``run_info.json`` with status "running" (fresh) or "resumed". During training it is
updated on every evaluation. On exit it records the end time, timings, final / best accuracy
and the status: "finished", "crashed" (with the error; the traceback goes to console.log) or
"interrupted" (Ctrl+C). It also redraws accuracy.png from the metrics so far.

A process killed from outside (SIGKILL, a Kaggle session timeout) cannot run any of this; its
run_info.json then stays at "running" / "resumed" with the progress of its last evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

CONSOLE_LOG = "console.log"
RUN_INFO = "run_info.json"
DATA_INFO = "data_info.json"
FINAL_MODEL = "final_model.pt"
PLOT = "accuracy.png"
# Files the runner itself writes into --out. If any exists, a run without --resume refuses to
# start (it would overwrite them). The sweep's runner.log is not one of them.
RUN_FILES = ("checkpoint.pt", FINAL_MODEL, "metrics.csv", RUN_INFO, CONSOLE_LOG, DATA_INFO,
             "config.json", "partition.csv", PLOT)

# Same fixed colors as scripts/summarize_sweep.py, so a method looks the same in every plot.
METHOD_COLORS = {"fedavg": "#2a78d6", "br_drag": "#eb6834", "fltrust_paper": "#1baf7a",
                 "fltrust_cao": "#eda100"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (a crash mid-write never leaves a half-written file)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def save_torch(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- environment
def _git(args: list[str], cwd: Path) -> Optional[str]:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_info(start: Path = Path(__file__).resolve().parent) -> dict:
    """Branch, commit and uncommitted changes of the repo holding this code (nulls without git).

    ``uncommitted_changes`` counts modified, staged and untracked files, except under
    ``results/`` (run outputs, not code). ``changed_files`` lists up to 50 of them.
    """
    top = _git(["rev-parse", "--show-toplevel"], start)
    if top is None:
        return {"branch": None, "commit": None, "uncommitted_changes": None, "changed_files": None}
    top_path = Path(top)
    status = _git(["status", "--porcelain", "--", ".", ":(exclude)results"], top_path)
    changed = [line[3:] for line in status.splitlines()] if status else []
    return {
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], top_path),
        "commit": _git(["rev-parse", "HEAD"], top_path),
        "uncommitted_changes": None if status is None else bool(changed),
        "changed_files": changed[:50] + ([f"... and {len(changed) - 50} more"] if len(changed) > 50 else []),
    }


def cpu_name() -> Optional[str]:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        elif sys.platform == "darwin":
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                               text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
    except Exception:  # noqa: BLE001 - best effort, never fail a run over a CPU name
        pass
    return platform.processor() or None


def environment_info(device: torch.device) -> dict:
    cuda_ok = torch.cuda.is_available()
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_ok else []
    cpu = cpu_name()
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(idx)
    else:
        device_name = cpu
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,  # CUDA version torch was built with (None for CPU builds)
        "cudnn": torch.backends.cudnn.version() if cuda_ok else None,
        "cuda_available": cuda_ok,
        "device": str(device),
        "device_name": device_name,
        "cpu": cpu,
        "gpus": gpus,
        "platform": platform.platform(),
    }


# ----------------------------------------------------------------------------- metrics
def eval_summary(metrics: list[dict]) -> dict:
    """Final and best test accuracy from the evaluated rows of ``metrics``."""
    evals = [r for r in metrics if r.get("test_acc") not in (None, "")]
    out = {"rounds_done": max((int(r["round"]) for r in metrics), default=None),
           "final_round": None, "final_test_acc": None, "final_test_loss": None,
           "best_round": None, "best_test_acc": None}
    if evals:
        best = max(evals, key=lambda r: float(r["test_acc"]))
        out.update(final_round=int(evals[-1]["round"]), final_test_acc=float(evals[-1]["test_acc"]),
                   final_test_loss=float(evals[-1]["test_loss"]), best_round=int(best["round"]),
                   best_test_acc=float(best["test_acc"]))
    return out


def plot_run(metrics: list[dict], path: Path, title: str, method: str = "") -> Optional[Path]:
    """accuracy.png: test accuracy (top) and test loss (bottom) vs round. None if nothing to plot."""
    evals = [r for r in metrics if r.get("test_acc") not in (None, "")]
    if not evals:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
    color = METHOD_COLORS.get(method, "#2a78d6")
    xs = [int(r["round"]) for r in evals]
    acc = [float(r["test_acc"]) for r in evals]
    loss = [float(r["test_loss"]) for r in evals]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.6), dpi=130, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for ax, ys, label in ((ax1, acc, "test accuracy"), (ax2, loss, "test loss")):
        ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round")
        ax.plot(xs[-1], ys[-1], "o", ms=5, color=color, mec="white", mew=1.5, zorder=3)
        ax.set_ylabel(label, color=muted)
        ax.grid(axis="y", color=grid, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylim(0, max(loss) * 1.1 if max(loss) > 0 else 1)
    ax2.set_xlabel("round", color=muted)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    best = max(range(len(acc)), key=acc.__getitem__)
    # run name on its own line; summaries as panel titles, so they never collide with the lines
    fig.suptitle(title, x=0.02, ha="left", fontsize=10, color=ink)
    ax1.set_title(f"test accuracy: final {acc[-1]:.1%}, best {acc[best]:.1%} (round {xs[best]})",
                  loc="left", fontsize=8, color=muted)
    ax2.set_title(f"test loss: final {loss[-1]:.3f}", loc="left", fontsize=8, color=muted)
    fig.tight_layout()
    tmp = path.with_name(path.name + ".tmp")
    fig.savefig(tmp, format="png")
    plt.close(fig)
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------------- console tee
class _Tee:
    """Write-through to the original stream and to the log file (flushed on every write)."""

    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log = log_file

    def write(self, s: str) -> int:
        try:
            self._stream.write(s)
        except (OSError, ValueError):  # terminal gone (closed pipe): keep logging to the file
            pass
        self._log.write(s)
        self._log.flush()
        return len(s)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except (OSError, ValueError):
            pass
        self._log.flush()

    def __getattr__(self, name: str):  # encoding, isatty, fileno, ... of the real stream
        return getattr(self._stream, name)


class RunRecorder:
    """One session of a run: console.log tee + run_info.json bookkeeping (see module doc)."""

    def __init__(self, out: Path, config: dict, device: torch.device, argv: Optional[list[str]],
                 resume_requested: bool, resumed_from_round: Optional[int]) -> None:
        self.out = Path(out)
        self.config = config
        self.device = device
        self.argv = argv
        self.resume_requested = resume_requested
        self.resumed_from_round = resumed_from_round
        self.metrics: list[dict] = []
        self.setup_s = self.train_s = self.eval_s = 0.0
        self.finished = False
        self._t0 = 0.0
        self._log = None
        self._saved: list = []
        self.info: dict = {}

    # -- run_info.json ---------------------------------------------------------------
    def _session(self) -> dict:
        return self.info["sessions"][-1]

    def _write_info(self) -> None:
        s = self._session()
        s["end_time"] = now_iso()
        s["total_time_s"] = round(time.perf_counter() - self._t0, 3)
        s["setup_time_s"] = round(self.setup_s, 3)
        s["train_time_s"] = round(self.train_s, 3)
        s["eval_time_s"] = round(self.eval_s, 3)
        sessions = self.info["sessions"]
        self.info.update(
            start_time=sessions[0]["start_time"],
            end_time=s["end_time"] if s["status"] not in ("running", "resumed") else None,
            last_update=s["end_time"],
            total_time_s=round(sum(x.get("total_time_s") or 0 for x in sessions), 3),
            setup_time_s=round(sum(x.get("setup_time_s") or 0 for x in sessions), 3),
            train_time_s=round(sum(x.get("train_time_s") or 0 for x in sessions), 3),
            eval_time_s=round(sum(x.get("eval_time_s") or 0 for x in sessions), 3),
            **eval_summary(self.metrics),
        )
        self.info["status"] = s["status"]
        write_json(self.out / RUN_INFO, self.info)

    def _start_info(self) -> None:
        path = self.out / RUN_INFO
        old = None
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                path.replace(path.with_name(f"run_info.unreadable_{int(time.time())}.json"))
        resumed = self.resumed_from_round is not None
        command = ([sys.executable, "-m", "fedrag.runner.run", *self.argv] if self.argv is not None
                   else None)  # None: called from Python (tests / notebook), see "config"
        session = {
            "kind": "resume" if self.resumed_from_round is not None else "fresh",
            "resumed_from_round": self.resumed_from_round,
            "status": "resumed" if resumed else "running",
            "start_time": now_iso(),
            "end_time": None,
            "command": command,
            "git": git_info(),
            "environment": environment_info(self.device),
        }
        if self.resume_requested and self.resumed_from_round is None:
            session["note"] = "--resume given but no checkpoint found: started from round 0"
        sessions = (old or {}).get("sessions", [])
        if old is None and self.resumed_from_round is not None:
            session["note"] = ("run_info.json did not exist (run started before it was added); "
                               "earlier sessions are not recorded")
        self.info = {
            "status": session["status"],
            "resumed": bool(resumed or any(x.get("kind") == "resume" for x in sessions)),
            "num_sessions": len(sessions) + 1,
            "command": command,
            "config": self.config,
            "git": session["git"],
            "environment": session["environment"],
            "start_time": None, "end_time": None, "last_update": None,
            "total_time_s": None, "setup_time_s": None, "train_time_s": None, "eval_time_s": None,
            "rounds_done": None, "final_round": None, "final_test_acc": None, "final_test_loss": None,
            "best_round": None, "best_test_acc": None,
            "error": None,
            "sessions": sessions + [session],
        }
        self._write_info()

    # -- context manager -------------------------------------------------------------
    def __enter__(self) -> "RunRecorder":
        self._t0 = time.perf_counter()
        self.out.mkdir(parents=True, exist_ok=True)
        self._log = open(self.out / CONSOLE_LOG, "a", encoding="utf-8", errors="replace")
        kind = ("resume from round %d" % self.resumed_from_round if self.resumed_from_round is not None
                else "fresh start" + (" (--resume, no checkpoint)" if self.resume_requested else ""))
        cmd = " ".join([sys.executable, "-m", "fedrag.runner.run", *self.argv]) if self.argv is not None \
            else "(called from Python)"
        self._log.write(f"===== {now_iso()} session start: {kind}\n===== command: {cmd}\n")
        self._log.flush()
        # Tee stdout / stderr, and point logging handlers that wrote to them at the tees.
        old_out, old_err = sys.stdout, sys.stderr
        tee_out, tee_err = _Tee(old_out, self._log), _Tee(old_err, self._log)
        handlers = []
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.StreamHandler) and h.stream in (old_out, old_err):
                handlers.append((h, h.stream))
                h.setStream(tee_out if h.stream is old_out else tee_err)
        sys.stdout, sys.stderr = tee_out, tee_err
        self._saved = [old_out, old_err, handlers]
        try:
            self._start_info()
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        old_out, old_err, handlers = self._saved
        sys.stdout, sys.stderr = old_out, old_err
        for h, stream in handlers:
            h.setStream(stream)
        self._log.close()

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is not None:
                status = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "crashed"
                self._session()["status"] = status
                self.info["error"] = self._session()["error"] = f"{exc_type.__name__}: {exc}"
                # traceback only into the file: Python prints it to the terminal itself
                self._log.write(f"\n[{status}] {now_iso()}\n")
                self._log.write("".join(traceback.format_exception(exc_type, exc, tb)))
            else:
                self._session()["status"] = "finished" if self.finished else "stopped"
                self.info["error"] = None
            try:
                self.plot()
            except Exception as e:  # noqa: BLE001 - a plotting problem must not hide the result
                self._log.write(f"[warning] could not draw {PLOT}: {e!r}\n")
            self._write_info()
            self._log.write(f"===== {now_iso()} session end: {self._session()['status']}\n")
        finally:
            self._restore()
        return False

    # -- during the run --------------------------------------------------------------
    def update(self, metrics: list[dict], train_s: float, eval_s: float) -> None:
        """Record progress (called on every evaluation); rewrites run_info.json."""
        self.metrics, self.train_s, self.eval_s = metrics, train_s, eval_s
        self._write_info()

    def plot(self) -> Optional[Path]:
        cfg = self.config
        mal = "" if cfg["attack"] == "none" else f", {cfg['malicious']:.0%} malicious"
        title = f"{cfg['dataset']}: {cfg['method']} vs {cfg['attack']}{mal}, beta {cfg['beta']}, seed {cfg['seed']}"
        return plot_run(self.metrics, self.out / PLOT, title, cfg["method"])
