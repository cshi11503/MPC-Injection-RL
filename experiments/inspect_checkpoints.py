"""
Roll out saved policies and plot how far they get.

For each run folder, every available checkpoint (best.pt, latest.pt, final.pt) is
loaded and evaluated on that run's own environment:
    - 1 deterministic episode (policy mean)  -> thick line
    - N stochastic episodes (sampled actions) -> thin lines
Plots per checkpoint: trajectories on the map, and distance-to-goal over time.
Plus the run's evaluation curve from eval.csv, marking where 'best' came from.

Usage (repo root):
    python -m experiments.inspect_checkpoints results/runs/sac_rho0_fixed_seed0 results/runs/sac_rho0_fixed_r1_seed1
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import SAC  # noqa: E402
from environment import NavigationEnv, NavConfig  # noqa: E402

CKPTS = ["best.pt", "latest.pt", "final.pt"]


def load_env(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)["env"]
    return NavigationEnv(NavConfig(**cfg))


def rollout(agent, env, deterministic):
    obs, info = env.reset()
    dists, done = [info["distance_to_goal"]], False
    ret = 0.0
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs, deterministic=deterministic))
        ret += r
        dists.append(info["distance_to_goal"])
        done = term or trunc
    return {"traj": np.array(env.trajectory), "dists": np.array(dists), "return": ret,
            "event": info["event"] or "timeout", "min_dist": float(min(dists))}


def read_eval(run_dir):
    path = os.path.join(run_dir, "eval.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if k != "det_event" else r[k] for r in rows]) for k in rows[0]}


def best_step(ev):
    """Same rule train.py uses to decide when to overwrite best.pt."""
    key = list(zip(ev["det_success"] + ev["sto_success_rate"], ev["sto_return_mean"]))
    best, idx = None, 0
    for i, k in enumerate(key):
        if best is None or k > best:
            best, idx = k, i
    return int(ev["step"][idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--episodes", type=int, default=10, help="stochastic episodes per checkpoint")
    p.add_argument("--out", default="results/fixed_obstacle/checkpoint_inspection.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    panels = []   # (run_name, ckpt_name, env, results)
    for run_dir in args.runs:
        env = load_env(run_dir)
        for ck in CKPTS:
            path = os.path.join(run_dir, ck)
            if not os.path.exists(path):
                continue
            agent = SAC(env.obs_dim, env.action_dim)
            agent.load(path)
            det = rollout(agent, env, True)
            sto = [rollout(agent, env, False) for _ in range(args.episodes)]
            panels.append((os.path.basename(run_dir), ck, env, det, sto, run_dir))
            n_goal = sum(e["event"] == "goal" for e in sto)
            print(f"{os.path.basename(run_dir):28s} {ck:9s} det: {det['event']:>13s} R={det['return']:6.2f} "
                  f"closest={det['min_dist']:.2f} | sto: goal {n_goal}/{len(sto)}, "
                  f"closest(mean)={np.mean([e['min_dist'] for e in sto]):.2f}")

    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(16, 5 * n), squeeze=False)
    colors = {"goal": "tab:green", "timeout": "tab:orange", "collision": "tab:red", "out_of_bounds": "tab:purple"}
    for row, (run, ck, env, det, sto, run_dir) in enumerate(panels):
        # (a) map
        ax = axes[row, 0]
        env.reset()
        env.render(ax=ax, show_trajectory=False)
        for e in sto:
            ax.plot(e["traj"][:, 0], e["traj"][:, 1], color=colors[e["event"]], lw=0.8, alpha=0.6)
        ax.plot(det["traj"][:, 0], det["traj"][:, 1], color="k", lw=2.5, label=f"deterministic: {det['event']}")
        ax.plot(*det["traj"][-1], "kX", ms=10)
        handles = [plt.Line2D([], [], color=c, label=f"stochastic: {k}") for k, c in colors.items()
                   if any(e["event"] == k for e in sto)]
        ax.legend(handles=[ax.get_lines()[-2]] + handles, fontsize=7, loc="upper left")
        ax.set_title(f"{run}\n{ck}")

        # (b) distance to goal over time
        ax = axes[row, 1]
        for e in sto:
            ax.plot(e["dists"], color=colors[e["event"]], lw=0.8, alpha=0.6)
        ax.plot(det["dists"], color="k", lw=2.5)
        ax.axhline(env.cfg.goal_radius, color="tab:green", ls="--", lw=1, label="goal radius")
        ax.set(xlabel="time step", ylabel="distance to goal", ylim=(0, None),
               title=f"How close it gets (det closest = {det['min_dist']:.2f})")
        ax.legend(fontsize=8)

        # (c) evaluation history (latest state of the running job)
        ax = axes[row, 2]
        ev = read_eval(run_dir)
        if ev is not None:
            ax.plot(ev["step"], ev["det_return"], "k-o", ms=3, label="det return")
            ax.plot(ev["step"], ev["sto_return_mean"], "-o", ms=3, color="tab:blue", label="sto return (mean)")
            ax2 = ax.twinx()
            ax2.plot(ev["step"], ev["sto_success_rate"], "-", color="tab:green", lw=1.5, label="sto success rate")
            ax2.set(ylim=(-0.05, 1.05), ylabel="success rate")
            bs = best_step(ev)
            ax.axvline(bs, color="tab:red", ls=":", lw=1.5, label=f"best.pt from step {bs}")
            ax.set(xlabel="environment steps", ylabel="return",
                   title=f"Eval history (latest: step {int(ev['step'][-1])}, {ev['det_event'][-1]})")
            lines = ax.get_legend_handles_labels()
            lines2 = ax2.get_legend_handles_labels()
            ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=7, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=90)
    print("saved", args.out)


if __name__ == "__main__":
    main()
