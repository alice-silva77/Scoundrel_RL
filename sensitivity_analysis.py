import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent

CONFIG_ORDER = [
    "baseline", "ent_low", "ent_med", "ent_high",
    "lr_low", "lr_high", "no_kwb", "no_cc",
]

COLORS = {
    "baseline": "black",
    "ent_low":  "royalblue",
    "ent_med":  "steelblue",
    "ent_high": "deepskyblue",
    "lr_low":   "darkorange",
    "lr_high":  "tomato",
    "no_kwb":   "crimson",
    "no_cc":    "mediumpurple",
}

LABELS = {
    "baseline": "Baseline (ent=0.10, lr=5e-5)",
    "ent_low":  "ent_coef=0.01",
    "ent_med":  "ent_coef=0.05",
    "ent_high": "ent_coef=0.20",
    "lr_low":   "lr=1e-5",
    "lr_high":  "lr=2e-4",
    "no_kwb":   "No king weapon bonus",
    "no_cc":    "No card counting",
}

CHECKPOINT_STEPS = [10_000_000, 25_000_000, 40_000_000, 55_000_000]


def load_logs() -> dict[str, dict]:
    logs = {}
    for name in CONFIG_ORDER:
        path = HERE / f"training_log_{name}.json"
        if path.exists():
            with open(path) as f:
                logs[name] = json.load(f)
            print(f"Loaded: {path.name}")
        else:
            print(f"Missing (skipping): {path.name}")
    return logs


def _interp_at(ts: np.ndarray, vals: np.ndarray, target: int) -> float:
    valid = ~np.isnan(vals)
    ts_v, vals_v = ts[valid], vals[valid]
    if len(ts_v) == 0 or target < ts_v[0]:
        return float("nan")
    if target >= ts_v[-1]:
        return float(vals_v[-1])
    return float(np.interp(target, ts_v, vals_v))


def _steps_to_phase(log: dict, phase: int) -> float:
    phases = np.array(log["curriculum_phase"])
    times  = np.array(log["timesteps"])
    idxs   = np.where(phases >= phase)[0]
    if len(idxs) == 0:
        return float("nan")
    return float(times[idxs[0]])


def _peak_win_rate_in_phase(log: dict, phase: int) -> float:
    phases = np.array(log["curriculum_phase"])
    wrs    = np.array(log["win_rate"], dtype=float)
    mask   = phases == phase
    if not mask.any():
        return float("nan")
    return float(np.nanmax(wrs[mask]))


# ── Plot 1: Win rate over time ────────────────────────────────────────────────

def plot_win_rate(logs: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, log in logs.items():
        ts  = np.array(log["timesteps"])
        wrs = np.array(log["win_rate"], dtype=float)
        valid = ~np.isnan(wrs)
        ax.plot(ts[valid] / 1e6, wrs[valid],
                color=COLORS[name], label=LABELS[name], linewidth=1.5)

    if "baseline" in logs:
        bl = logs["baseline"]
        phases = np.array(bl["curriculum_phase"])
        times  = np.array(bl["timesteps"])
        prev = -1
        for i, p in enumerate(phases):
            if p != prev and prev >= 0:
                ax.axvline(times[i] / 1e6, color="gray", linestyle="--",
                           linewidth=0.8, alpha=0.5)
            prev = p

    ax.set_xlabel("Timesteps (millions)")
    ax.set_ylabel("Win Rate (last 200 episodes)")
    ax.set_title("Win Rate Over Time — Sensitivity Analysis")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = HERE / "sensitivity_win_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Plot 2: Steps to reach each phase ────────────────────────────────────────

def plot_steps_to_phase(logs: dict[str, dict]):
    phases = [1, 2, 3, 4]
    names  = list(logs.keys())
    x      = np.arange(len(phases))
    width  = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        vals = [_steps_to_phase(logs[name], p) / 1e6 for p in phases]
        offset = (i - len(names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=LABELS[name],
                      color=COLORS[name], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Phase {p}" for p in phases])
    ax.set_ylabel("Timesteps to reach phase (millions)")
    ax.set_title("Curriculum Progression Speed")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = HERE / "sensitivity_phase_speed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Plot 3: Peak win rate per phase ──────────────────────────────────────────

def plot_peak_win_rate_per_phase(logs: dict[str, dict]):
    phases = [0, 1, 2, 3, 4]
    names  = list(logs.keys())
    x      = np.arange(len(phases))
    width  = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(names):
        vals = [_peak_win_rate_in_phase(logs[name], p) for p in phases]
        offset = (i - len(names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=LABELS[name],
               color=COLORS[name], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Phase {p}" for p in phases])
    ax.set_ylabel("Peak Win Rate")
    ax.set_title("Peak Win Rate per Curriculum Phase")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = HERE / "sensitivity_peak_win_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Plot 4: Win rate at fixed checkpoints ────────────────────────────────────

def plot_win_rate_at_checkpoints(logs: dict[str, dict]):
    names = list(logs.keys())
    x     = np.arange(len(CHECKPOINT_STEPS))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(names):
        log = logs[name]
        ts  = np.array(log["timesteps"])
        wrs = np.array(log["win_rate"], dtype=float)
        vals = [_interp_at(ts, wrs, t) for t in CHECKPOINT_STEPS]
        offset = (i - len(names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=LABELS[name],
               color=COLORS[name], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t // 1_000_000}M" for t in CHECKPOINT_STEPS])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate at Fixed Timestep Checkpoints")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = HERE / "sensitivity_checkpoints.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(logs: dict[str, dict]):
    print("\n=== Summary ===")
    header = f"{'Config':<12} {'Ph.reached':>10} {'Peak WR ph4':>12} {'WR@55M':>10}"
    print(header)
    print("-" * len(header))
    for name, log in logs.items():
        phases = np.array(log["curriculum_phase"])
        max_phase = int(np.max(phases))
        peak_ph4  = _peak_win_rate_in_phase(log, 4)
        ts  = np.array(log["timesteps"])
        wrs = np.array(log["win_rate"], dtype=float)
        wr55 = _interp_at(ts, wrs, 55_000_000)
        peak_str = f"{peak_ph4:.1%}" if not np.isnan(peak_ph4) else "N/A"
        wr55_str = f"{wr55:.1%}" if not np.isnan(wr55) else "N/A"
        print(f"{name:<12} {max_phase:>10} {peak_str:>12} {wr55_str:>10}")


if __name__ == "__main__":
    logs = load_logs()
    if not logs:
        print("No log files found. Download training_log_*.json files first.")
    else:
        plot_win_rate(logs)
        plot_steps_to_phase(logs)
        plot_peak_win_rate_per_phase(logs)
        plot_win_rate_at_checkpoints(logs)
        print_summary(logs)
        print("\nDone. Open sensitivity_*.png to view results.")
