"""
Figure: best rho=0.5 policy vs MPC vs SAC rho=0 (needs results/comparison/compare.npz
from experiments.compare).

    python -m experiments.plot_comparison        # -> results/comparison/comparison.png
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.compare import RHO0, RHO5, wilson

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
# colour follows the method, in every panel
WHO = {"SAC + MPC injection (rho=0.5)": ("rho0.5 step 2610000", "#2a78d6"),
       "SAC only (rho=0)": ("rho0 final.pt (5M)", "#eb6834"),
       "MPC": ("MPC", "#1baf7a")}


def curve(run, window=50):
    rows = list(csv.DictReader(open(f"{run}/train_log.csv")))
    step = np.array([float(r["step"]) for r in rows])
    succ = np.array([float(r["train_success_last20"]) for r in rows])
    return step[window - 1:], np.convolve(succ, np.ones(window) / window, mode="valid")


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")


def main():
    z = np.load("results/comparison/compare.npz")
    res = {n: z[f"r{i}"] for i, n in enumerate(z["names"])}
    data = {label: (res[key], color) for label, (key, color) in WHO.items()}
    mpc_success = res["MPC"][:, 0].mean()

    fig = plt.figure(figsize=(12.5, 8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.5, wspace=0.42)

    # ---- training curves ----
    for ax, xmax, title in ((fig.add_subplot(gs[0, :2]), 5e6, "Training success over 5M steps"),
                            (fig.add_subplot(gs[0, 2:]), 5e5, "First 500k steps: injection learns sooner")):
        style(ax, title)
        for label, run in (("SAC + MPC injection (rho=0.5)", RHO5), ("SAC only (rho=0)", RHO0)):
            x, y = curve(run)
            ax.plot(x[x <= xmax] / 1e6, y[x <= xmax], color=WHO[label][1], linewidth=2, label=label)
        ax.axhline(mpc_success, color=WHO["MPC"][1], linewidth=2, linestyle=(0, (4, 3)), label="MPC (no training)")
        ax.set_xlim(0, xmax / 1e6)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("environment steps (millions)", color=MUTED, fontsize=9)
        ax.set_ylabel("success rate, training episodes\n(rolling mean)", color=MUTED, fontsize=9)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.axes[0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)

    # ---- final metrics on the shared held-out layouts ----
    labels = ["SAC + MPC\nrho=0.5", "SAC\nrho=0", "MPC"]
    xs = np.arange(3)
    colors = [c for _, c in data.values()]

    ax = fig.add_subplot(gs[1, 0])
    style(ax, "Success rate (95% interval)")
    for x, (r, c) in zip(xs, data.values()):
        p = r[:, 0].mean()
        lo, hi = wilson(int(r[:, 0].sum()), len(r))
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="o", color=c, markersize=8, linewidth=2, capsize=4)
        ax.annotate(f"{p:.2%}", (x, p), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_ylim(0.98, 1.0005)
    ax.set_yticks(np.arange(0.98, 1.0001, 0.005))
    ax.set_xlim(-0.5, 3.1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")

    panels = [("Collision rate (%)", lambda r: r[:, 1].mean() * 100, "{:.2f}%"),
              ("Timeout rate (%)", lambda r: (1 - r[:, 0].mean() - r[:, 1].mean()) * 100, "{:.2f}%"),
              ("Steps to reach the goal", lambda r: r[r[:, 0] > 0, 3].mean(), "{:.1f}")]
    for k, (title, f, fmt) in enumerate(panels, 1):
        ax = fig.add_subplot(gs[1, k])
        style(ax, title)
        vals = [max(f(r), 0.0) for r, _ in data.values()]
        ax.bar(xs, vals, width=0.5, color=colors)
        for x, v in zip(xs, vals):
            ax.annotate(fmt.format(v), (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.set_ylim(0, max(vals) * 1.18)
    for ax in fig.axes[2:]:
        ax.set_xticks(xs, labels, color=INK, fontsize=8.5)

    n_pol, n_mpc = len(data["SAC only (rho=0)"][0]), len(res["MPC"])
    fig.suptitle("MPC-Injection SAC vs SAC vs MPC: random 4-obstacle layouts", x=0.065, ha="left", fontsize=13, color=INK)
    fig.text(0.065, 0.02, f"Bottom row: deterministic policies on {n_pol:,} unseen layouts (MPC: the first {n_mpc:,} of them). "
             "rho=0.5 is the step-2,610,000 checkpoint; rho=0 is the 5M-step final checkpoint.", fontsize=8.5, color=MUTED)
    fig.savefig("results/comparison/comparison.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print("saved results/comparison/comparison.png")


if __name__ == "__main__":
    main()
