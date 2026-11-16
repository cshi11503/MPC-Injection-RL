"""
Real-time GIF: SAC rho=0, SAC rho=0.5 and the MPC driving through the same 4 random layouts.

    python -m experiments.make_comparison_gif                       # layouts 50000..50003
    python -m experiments.make_comparison_gif --seeds 50010 50011 50012 50013

One frame per env step at fps = 1 / dt, so the GIF plays in real time. Policies are
deterministic. Layout seeds default to the first 4 of experiments/compare.py's test set.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from agents.sac import SAC
from environment.navigation_env import NavConfig, NavigationEnv

RUNS = "results/runs"
RHO5_DIR, RHO0_DIR = f"{RUNS}/sac_rho0.5_random_fixedends_seed0", f"{RUNS}/sac_rho0_random_fixedends_cudagraph_seed0"
# best checkpoints on the 10k-layout test in results/comparison/compare.txt
CONTROLLERS = [  # (label, checkpoint or None for MPC, color)  -- Okabe-Ito palette
    ("SAC ρ=0", f"{RHO0_DIR}/final.pt", "#0072B2"),
    ("SAC ρ=0.5", f"{RHO5_DIR}/ckpt/step_02610000.pt", "#E69F00"),
    ("MPC", None, "#CC79A7"),
]


def run_policy(path, env_cfg, hidden, seed):
    env = NavigationEnv(NavConfig(**env_cfg))
    agent = SAC(env.obs_dim, env.action_dim, hidden=tuple(hidden))
    agent.load(path)
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, term, trunc, info = env.step(agent.act(obs, deterministic=True))
        if term or trunc:
            return np.array(env.trajectory), info["event"] or "timeout", env


def run_mpc(seed):
    from experiments import generate_mpc_data as g
    seed, event, _, _ = g._episode(seed)
    return np.array(g._env.trajectory), event, g._env


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs=4, default=[50_000, 50_001, 50_002, 50_003])
    p.add_argument("--hold", type=float, default=2.0, help="seconds to hold the last frame")
    p.add_argument("--out", default="results/comparison/navigation_comparison.gif")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # labels contain "ρ"; Windows consoles default to cp1252

    torch.set_num_threads(1)
    cfg = json.load(open(f"{RHO5_DIR}/config.json"))
    env_cfg, hidden = cfg["env"], cfg["args"]["hidden"]
    from experiments import generate_mpc_data as g
    g._init(len(env_cfg["obstacles"]) if env_cfg["layout"] == "fixed" else env_cfg["n_obstacles"])

    # rollouts[i][j] = (trajectory, event) for layout i, controller j
    layouts, rollouts = [], []
    for seed in args.seeds:
        row, envs = [], []
        for label, path, _ in CONTROLLERS:
            traj, event, env = run_mpc(seed) if path is None else run_policy(path, env_cfg, hidden, seed)
            row.append((traj, event))
            envs.append(env)
            print(f"seed {seed} {label:<10} {event:<14} {len(traj) - 1} steps")
        for e in envs[1:]:   # all three must have driven the same layout
            assert np.allclose(e.obstacles, envs[0].obstacles) and np.allclose(e.goal, envs[0].goal)
        layouts.append(envs[0])
        rollouts.append(row)

    dt = env_cfg["dt"]
    fps = round(1 / dt)
    n_steps = max(len(t) for row in rollouts for t, _ in row)
    n_frames = n_steps + int(args.hold * fps)
    L, R = env_cfg["arena_size"], env_cfg["robot_radius"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 10.4))
    fig.patch.set_facecolor("white")
    artists = []   # per panel: list of (trail line, robot patch, status text) per controller
    for ax, env, seed in zip(axes.ravel(), layouts, args.seeds):
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Layout {seed}", fontsize=11)
        for cx, cy, r in env.obstacles:
            ax.add_patch(Circle((cx, cy), r, color="#6b6b6b", alpha=0.85, lw=0))
        ax.add_patch(Circle(env.goal, env_cfg["goal_radius"], color="#009E73", alpha=0.35, lw=0))
        ax.plot(*env.goal, "*", color="#00704f", ms=13, zorder=3)
        ax.plot(*env.start, "s", color="black", ms=6, zorder=3)
        panel = []
        for j, (_, _, color) in enumerate(CONTROLLERS):
            line, = ax.plot([], [], "-", color=color, lw=2, alpha=0.9, zorder=4)
            dot = Circle(env.start, max(R, 0.13), color=color, ec="black", lw=0.8, zorder=5)
            ax.add_patch(dot)
            txt = ax.text(0.5, -0.025 - 0.06 * j, "", transform=ax.transAxes, fontsize=9, color=color,
                          ha="center", va="top", fontweight="bold")
            panel.append((line, dot, txt))
        artists.append(panel)

    handles = [plt.Line2D([], [], color=c, lw=3, label=l) for l, _, c in CONTROLLERS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.975))
    clock = fig.text(0.5, 0.015, "", ha="center", fontsize=12, family="monospace")
    # manual spacing: tight_layout ignores the status lines drawn below each panel
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.13, wspace=0.06, hspace=0.30)

    def draw(k):
        clock.set_text(f"t = {min(k, n_steps - 1) * dt:4.1f} s   (real time)")
        for panel, row in zip(artists, rollouts):
            for j, ((line, dot, txt), (traj, event)) in enumerate(zip(panel, row)):
                i = min(k, len(traj) - 1)
                line.set_data(traj[:i + 1, 0], traj[:i + 1, 1])
                dot.center = traj[i]
                done = k >= len(traj) - 1
                status = {"goal": "goal", "collision": "COLLISION", "out_of_bounds": "OUT OF BOUNDS"}.get(event, event)
                txt.set_text(f"{CONTROLLERS[j][0]}: {status} {(len(traj) - 1) * dt:.1f} s" if done
                             else f"{CONTROLLERS[j][0]}: …")
        return []

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    FuncAnimation(fig, draw, frames=n_frames).save(args.out, writer=PillowWriter(fps=fps), dpi=80)
    print(f"saved {args.out}: {n_frames} frames @ {fps} fps ({n_steps * dt:.1f} s of motion)")


if __name__ == "__main__":
    main()
