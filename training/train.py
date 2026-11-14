"""
Train SAC on the navigation env, optionally with MPC-Injection.

    rho = 0      -> standard SAC baseline
    rho > 0      -> each minibatch has round(rho * B) transitions from the MPC dataset

Examples (run from repo root):
    python -m training.train                                  # baseline, seed 0
    python -m training.train --seed 1 --total-steps 200000
    python -m training.train --rho 0.5 --mpc-data data/mpc_dataset.npz

Outputs in results/runs/<run_name>/:
    config.json      every setting used (reproducibility)
    eval.csv         one row per evaluation (success / collision / timeout rates, return, ...)
    train_log.csv    averaged losses, alpha, Q, entropy every --log-every steps
    episodes.csv     one row per TRAINING episode (return, length, outcome)
    best.pt, final.pt   model checkpoints
    learning_curves.png, losses.png, trajectories.png
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import SAC, ReplayBuffer, sample_mixed  # noqa: E402
from environment import NavigationEnv  # noqa: E402


# =========================================================================
# Arguments
# =========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # experiment
    p.add_argument("--rho", type=float, default=0.0, help="MPC injection ratio (fraction of each minibatch)")
    p.add_argument("--mpc-data", type=str, default=None, help=".npz MPC dataset (required if rho > 0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layout", type=str, default="fixed", choices=["fixed", "random"])
    p.add_argument("--obstacle-radius", type=float, default=None,
                   help="fixed layout only: radius of the center obstacle (default: NavConfig value, 2.0)")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="results/runs")
    # training length / schedule
    p.add_argument("--total-steps", type=int, default=100_000, help="agent environment steps")
    p.add_argument("--warmup", type=int, default=5_000, help="uniform-random action steps before learning")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    # SAC hyperparameters
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--init-alpha", type=float, default=1.0)
    # evaluation / logging
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-episodes", type=int, default=10, help="stochastic-policy eval episodes")
    p.add_argument("--success-threshold", type=float, default=0.8, help="for the sample-efficiency metric")
    p.add_argument("--log-every", type=int, default=1_000)
    # compute
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (keep low when running in parallel)")
    args = p.parse_args(argv)

    if args.rho > 0 and not args.mpc_data:
        p.error("--mpc-data is required when --rho > 0")
    if not 0.0 <= args.rho <= 1.0:
        p.error("--rho must be in [0, 1]")
    if args.obstacle_radius is not None and args.layout != "fixed":
        p.error("--obstacle-radius only applies to --layout fixed")
    if args.run_name is None:
        r_tag = f"_r{args.obstacle_radius:g}" if args.obstacle_radius is not None else ""
        args.run_name = f"sac_rho{args.rho:g}_{args.layout}{r_tag}_seed{args.seed}"
    return args


def make_env(args):
    kwargs = {"layout": args.layout}
    if args.obstacle_radius is not None:
        kwargs["obstacles"] = ((5.0, 5.0, args.obstacle_radius),)
    return NavigationEnv(**kwargs)


class _Tee:
    """Duplicate console output into a UTF-8 log file inside the run folder."""

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


# =========================================================================
# Evaluation
# =========================================================================
def run_episode(agent, env, deterministic, seed=None):
    """Roll out one episode; returns a dict of episode statistics + the trajectory."""
    obs, info = env.reset(seed=seed)
    ret, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs, deterministic=deterministic))
        ret += r
        done = term or trunc
    traj = np.array(env.trajectory)
    return {
        "return": ret,
        "steps": info["t"],
        "event": info["event"] or "timeout",
        "success": float(info["is_success"]),
        "collision": float(info["is_collision"]),
        "timeout": float(info["event"] is None),
        "path_length": float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()),
        "final_dist": info["distance_to_goal"],
        "trajectory": traj,
    }


def evaluate(agent, env, n_stochastic, seed_base):
    """
    Deterministic episode (policy mean): what the learned policy 'really' does.
    Stochastic episodes (sampled actions): how robust it is to its own action noise.
    Fixed seeds -> identical evaluation conditions at every evaluation point.
    """
    det = run_episode(agent, env, deterministic=True, seed=seed_base)
    sto = [run_episode(agent, env, deterministic=False, seed=seed_base + 1 + i) for i in range(n_stochastic)]
    succ_sto = [e for e in sto if e["success"]]
    row = {
        "det_success": det["success"],
        "det_event": det["event"],
        "det_return": det["return"],
        "det_steps": det["steps"],
        "det_path_length": det["path_length"],
        "sto_success_rate": np.mean([e["success"] for e in sto]),
        "sto_collision_rate": np.mean([e["collision"] for e in sto]),
        "sto_timeout_rate": np.mean([e["timeout"] for e in sto]),
        "sto_return_mean": np.mean([e["return"] for e in sto]),
        "sto_return_std": np.std([e["return"] for e in sto]),
        # navigation efficiency: only meaningful over successful episodes
        "sto_steps_success": np.mean([e["steps"] for e in succ_sto]) if succ_sto else np.nan,
        "sto_path_length_success": np.mean([e["path_length"] for e in succ_sto]) if succ_sto else np.nan,
    }
    return row, det["trajectory"], [e["trajectory"] for e in sto]


# =========================================================================
# CSV helper
# =========================================================================
class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.file = None
        self.writer = None

    def log(self, row: dict):
        if self.writer is None:
            self.file = open(self.path, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


# =========================================================================
# Training
# =========================================================================
def train(args):
    torch.set_num_threads(args.threads)
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = _Tee(sys.stdout, os.path.join(run_dir, "stdout.log"))

    env = make_env(args)
    eval_env = make_env(args)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "env": env.config_dict()}, f, indent=2)

    agent = SAC(env.obs_dim, env.action_dim, hidden=tuple(args.hidden), lr=args.lr, gamma=args.gamma,
                tau=args.tau, init_alpha=args.init_alpha, device=args.device, seed=args.seed)
    rl_buffer = ReplayBuffer(env.obs_dim, env.action_dim, capacity=args.buffer_size, seed=args.seed)

    mpc_buffer = None
    if args.rho > 0:
        mpc_buffer = ReplayBuffer.from_npz(args.mpc_data, seed=args.seed + 1)
        if mpc_buffer.obs.shape[1] != env.obs_dim:
            raise ValueError(f"MPC data obs_dim {mpc_buffer.obs.shape[1]} != env obs_dim {env.obs_dim}")
        print(f"Loaded MPC dataset: {len(mpc_buffer)} transitions from {args.mpc_data}")

    eval_log = CSVLogger(os.path.join(run_dir, "eval.csv"))
    train_log = CSVLogger(os.path.join(run_dir, "train_log.csv"))
    ep_log = CSVLogger(os.path.join(run_dir, "episodes.csv"))

    print(f"Run: {args.run_name} | rho={args.rho} | {args.total_steps} steps | device={args.device} "
          f"| obs_dim={env.obs_dim}\nOutput: {run_dir}", flush=True)

    obs, _ = env.reset(seed=args.seed)
    ep_return, ep_len, n_episodes = 0.0, 0, 0
    recent = deque(maxlen=20)                    # outcomes of the last 20 training episodes
    loss_acc = defaultdict(list)
    best_key, first_hit_step = None, None
    eval_trajs = []                              # (step, det trajectory) for the trajectory plot
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        # ---- 1) act ----
        if step <= args.warmup:
            action = env.sample_action()         # uniform random exploration
        else:
            action = agent.act(obs)              # stochastic SAC policy

        # ---- 2) environment step + store transition ----
        next_obs, reward, terminated, truncated, info = env.step(action)
        # terminated (goal / collision) stops bootstrapping; truncation (time limit) does NOT
        rl_buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_return += reward
        ep_len += 1

        if terminated or truncated:
            n_episodes += 1
            outcome = info["event"] or "timeout"
            recent.append(outcome)
            ep_log.log({"step": step, "episode": n_episodes, "return": ep_return,
                        "length": ep_len, "outcome": outcome})
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        # ---- 3) learn ----
        if step > args.warmup:
            for _ in range(args.updates_per_step):
                batch = sample_mixed(rl_buffer, mpc_buffer, args.batch_size, args.rho)
                for k, v in agent.update(batch).items():
                    loss_acc[k].append(v)

        # ---- 4) log training statistics ----
        if step % args.log_every == 0 and loss_acc:
            row = {"step": step, **{k: float(np.mean(v)) for k, v in loss_acc.items()},
                   "train_success_last20": recent.count("goal") / max(len(recent), 1),
                   "episodes": n_episodes}
            train_log.log(row)
            loss_acc.clear()
            print(f"  [train] step {step:7d} | episodes {n_episodes:5d} | last20: "
                  f"goal {recent.count('goal'):2d} coll {recent.count('collision') + recent.count('out_of_bounds'):2d} "
                  f"timeout {recent.count('timeout'):2d} | critic {row['critic_loss']:.3f} | "
                  f"Q {row['q_mean']:6.2f} | alpha {row['alpha']:.3f} | entropy {row['entropy']:5.2f}", flush=True)

        # ---- 5) evaluate ----
        if step % args.eval_every == 0:
            row, det_traj, _ = evaluate(agent, eval_env, args.eval_episodes, seed_base=10_000)
            row = {"step": step, "episodes": n_episodes, "time_s": time.time() - t0, **row}
            eval_log.log(row)
            eval_trajs.append((step, det_traj, row["det_event"]))

            # sample efficiency: first eval where stochastic success rate >= threshold
            if first_hit_step is None and row["sto_success_rate"] >= args.success_threshold:
                first_hit_step = step
            # best checkpoint: success first, then return
            key = (row["det_success"] + row["sto_success_rate"], row["sto_return_mean"])
            if best_key is None or key > best_key:
                best_key = key
                agent.save(os.path.join(run_dir, "best.pt"))

            sps = step / (time.time() - t0)
            alpha = agent.alpha
            print(f"step {step:7d} | det: {row['det_event']:>13s} R={row['det_return']:7.2f} "
                  f"| sto: succ {row['sto_success_rate']:.2f} coll {row['sto_collision_rate']:.2f} "
                  f"R={row['sto_return_mean']:7.2f} | train succ(last20) "
                  f"{recent.count('goal') / max(len(recent), 1):.2f} | alpha {alpha:.3f} "
                  f"| {sps:5.1f} steps/s | ETA {(args.total_steps - step) / sps / 60:5.1f} min", flush=True)

    agent.save(os.path.join(run_dir, "final.pt"))
    for lg in (eval_log, train_log, ep_log):
        lg.close()

    summary = {
        "run_name": args.run_name,
        "rho": args.rho,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "episodes": n_episodes,
        "steps_to_success_threshold": first_hit_step,
        "success_threshold": args.success_threshold,
        "wall_time_min": (time.time() - t0) / 60,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:", json.dumps(summary, indent=2))

    make_plots(run_dir, eval_env, eval_trajs, args)
    return summary


# =========================================================================
# Plots
# =========================================================================
def make_plots(run_dir, env, eval_trajs, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def read_csv(name):
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            rows = list(csv.DictReader(f))
        out = defaultdict(list)
        for r in rows:
            for k, v in r.items():
                try:
                    out[k].append(float(v))
                except ValueError:
                    out[k].append(v)
        return {k: np.array(v) for k, v in out.items()}

    ev = read_csv("eval.csv")
    tl = read_csv("train_log.csv")

    # --- learning curves ---
    if ev:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        ax = axes[0]
        ax.plot(ev["step"], ev["sto_success_rate"], "-o", ms=3, label="success (stochastic)")
        ax.plot(ev["step"], ev["sto_collision_rate"], "-o", ms=3, label="collision (stochastic)")
        ax.plot(ev["step"], ev["sto_timeout_rate"], "-o", ms=3, label="timeout (stochastic)")
        ax.plot(ev["step"], ev["det_success"], "k--", lw=1, label="success (deterministic)")
        ax.axvline(args.warmup, color="gray", ls=":", lw=1)
        ax.set(xlabel="environment steps", ylabel="rate", ylim=(-0.05, 1.05), title="Outcome rates")
        ax.legend(fontsize=8)
        ax = axes[1]
        ax.plot(ev["step"], ev["sto_return_mean"], "-o", ms=3, label="stochastic (mean)")
        ax.fill_between(ev["step"], ev["sto_return_mean"] - ev["sto_return_std"],
                        ev["sto_return_mean"] + ev["sto_return_std"], alpha=0.2)
        ax.plot(ev["step"], ev["det_return"], "k--", lw=1, label="deterministic")
        ax.axvline(args.warmup, color="gray", ls=":", lw=1)
        ax.set(xlabel="environment steps", ylabel="episode return", title="Evaluation return")
        ax.legend(fontsize=8)
        fig.suptitle(args.run_name)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "learning_curves.png"), dpi=110)
        plt.close(fig)

    # --- losses ---
    if tl:
        keys = ["critic_loss", "actor_loss", "alpha", "q_mean", "entropy", "train_success_last20"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
        for ax, k in zip(axes.ravel(), keys):
            ax.plot(tl["step"], tl[k])
            ax.set(title=k, xlabel="environment steps")
        fig.suptitle(args.run_name + " - training diagnostics")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "losses.png"), dpi=110)
        plt.close(fig)

    # --- deterministic-policy trajectories at several points in training ---
    if eval_trajs:
        env.reset()
        idx = np.unique(np.linspace(0, len(eval_trajs) - 1, min(6, len(eval_trajs))).astype(int))
        fig, ax = plt.subplots(figsize=(6, 6))
        env.render(ax=ax, show_trajectory=False)
        cmap = plt.get_cmap("viridis")
        for j, i in enumerate(idx):
            step, traj, event = eval_trajs[i]
            ax.plot(traj[:, 0], traj[:, 1], color=cmap(j / max(len(idx) - 1, 1)), lw=1.8,
                    label=f"step {step}: {event}")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title("Deterministic policy during training")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "trajectories.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    train(parse_args())
