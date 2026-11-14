"""
Diagnose why a SAC policy stops short of the goal.

For each checkpoint:
  1. Roll out the deterministic policy; if it times out, take its final (hover) state s_h.
  2. Exploration: policy std at s_h -> how far can action noise push the robot?
  3. Critic belief: Q(s_h, a) over a grid of actions. Is "toward goal" valued below "stay"?
  4. Critic accuracy: Q(s_t, pi(s_t)) vs the ACTUAL discounted return-to-go along the rollout.
  5. Ground truth from s_h: return of (a) hovering for the rest of the budget,
     (b) a simple PD controller driving to the goal. Is the last stretch dangerous?
  6. Exploration test: from s_h, run the policy with its noise scaled x1, x3, x10.
     If larger noise finds the goal, the failure is an exploration problem.

Usage (repo root):
  python -m experiments.diagnose results/runs/sac_rho0_fixed_seed0/best.pt results/runs/sac_rho0_fixed_seed0/final.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import SAC  # noqa: E402
from environment import NavigationEnv, NavConfig  # noqa: E402

GAMMA = 0.99
BUDGET = 200          # steps allowed when restarting from the hover state


def load(ckpt):
    run_dir = os.path.dirname(ckpt)
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    env = NavigationEnv(NavConfig(**cfg["env"]))
    agent = SAC(env.obs_dim, env.action_dim, gamma=cfg["args"]["gamma"])
    agent.load(ckpt)
    return env, agent


def q_value(agent, obs, action):
    o = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)
    a = torch.as_tensor(np.atleast_2d(action), dtype=torch.float32)
    if o.shape[0] == 1 and a.shape[0] > 1:
        o = o.expand(a.shape[0], -1)
    with torch.no_grad():
        return torch.min(agent.q1(o, a), agent.q2(o, a)).numpy()


def policy_stats(agent, obs):
    with torch.no_grad():
        mu, log_std = agent.actor.net(torch.as_tensor(obs, dtype=torch.float32)).chunk(2, -1)
    return mu.numpy(), log_std.clamp(-20, 2).exp().numpy()


def start_from(env, state, budget=BUDGET):
    """Put the env in a given physical state with a fresh step budget."""
    env.reset()
    env.state = np.array(state, dtype=float)
    env.trajectory = [env.state[:2].copy()]
    env.t = env.cfg.max_steps - budget
    return env._obs()


def run(env, obs, policy):
    rews, done, info = [], False, {}
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs, env))
        rews.append(r)
        done = term or trunc
    disc = sum(GAMMA**k * r for k, r in enumerate(rews))
    return {"return": sum(rews), "disc_return": disc, "event": info["event"] or "timeout",
            "steps": len(rews), "traj": np.array(env.trajectory)}


def diagnose(ckpt, n_explore=20):
    env, agent = load(ckpt)
    name = f"{os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}"
    out = {"name": name}

    # ---- 1) deterministic rollout, Q along the way ----
    obs, info = env.reset()
    obs_list, rews = [obs], []
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs, True))
        obs_list.append(obs)
        rews.append(r)
        done = term or trunc
    out["event"] = info["event"] or "timeout"
    out["traj"] = np.array(env.trajectory)
    obs_arr = np.array(obs_list[:-1])
    acts = np.array([agent.act(o, True) for o in obs_arr])
    out["q_pred"] = q_value(agent, obs_arr, acts)
    # actual discounted return-to-go; at a time-limit truncation the true future
    # (hovering forever) is worth about -0.01/(1-gamma) = -1, add it as the tail value
    tail = -env.cfg.w_time / (1 - GAMMA) if out["event"] == "timeout" else 0.0
    rtg, g = np.zeros(len(rews)), tail
    for t in reversed(range(len(rews))):
        g = rews[t] + GAMMA * g
        rtg[t] = g
    out["q_true"] = rtg

    if out["event"] != "timeout":
        out["hover"] = None
        return out, env, agent

    # ---- 2) the hover state ----
    s_h = info["state"].copy()
    o_h = obs_list[-1]
    mu, std = policy_stats(agent, o_h)
    a_mean = np.tanh(mu)
    h = {"state": s_h, "dist_goal": info["distance_to_goal"], "mu": mu, "std": std, "a_mean": a_mean}
    # per-step velocity noise from action noise (approx, tanh slope ~1 near 0)
    h["vel_noise_per_step"] = float(np.linalg.norm(std) * env.cfg.a_max * env.cfg.dt)

    # ---- 3) Q over an action grid ----
    g1 = np.linspace(-1, 1, 41)
    AX, AY = np.meshgrid(g1, g1)
    grid = np.column_stack([AX.ravel(), AY.ravel()])
    h["q_grid"] = q_value(agent, o_h, grid).reshape(AX.shape)
    to_goal = env.goal - s_h[:2]
    a_goal = 0.5 * to_goal / np.linalg.norm(to_goal)
    h["a_goal"] = a_goal
    h["q_mean_action"] = float(q_value(agent, o_h, a_mean)[0])
    h["q_toward_goal"] = float(q_value(agent, o_h, a_goal)[0])

    # ---- 4) ground truth from the hover state ----
    obs0 = start_from(env, s_h)
    h["true_hover"] = run(env, obs0, lambda o, e: agent.act(o, True))
    obs0 = start_from(env, s_h)

    def pd(o, e, kp=1.0, kd=2.0):
        s = e.state
        return np.clip((kp * (e.goal - s[:2]) - kd * s[2:]) / e.cfg.a_max, -1, 1)
    h["true_pd"] = run(env, obs0, pd)

    # ---- 5) exploration test: scaled policy noise from the hover state ----
    h["explore"] = {}
    for k in (1, 3, 10):
        res = []
        for i in range(n_explore):
            rng = np.random.default_rng(i)

            def noisy(o, e, k=k, rng=rng):
                m, s = policy_stats(agent, o)
                return np.tanh(m + k * s * rng.standard_normal(m.shape))
            obs0 = start_from(env, s_h)
            res.append(run(env, obs0, noisy)["event"])
        h["explore"][k] = {ev: res.count(ev) / n_explore for ev in set(res)}
    out["hover"] = h
    return out, env, agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--out", default="results/diagnosis.png")
    args = p.parse_args()
    torch.set_num_threads(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = [diagnose(c) for c in args.ckpts]
    for out, env, _ in results:
        print(f"\n=== {out['name']} ===  deterministic outcome: {out['event']}")
        qp, qt = out["q_pred"], out["q_true"]
        print(f"Q at start: predicted {qp[0]:.2f} vs actual {qt[0]:.2f}  |  "
              f"mean overestimate along path: {np.mean(qp - qt):+.2f}")
        h = out["hover"]
        if h is None:
            continue
        print(f"Hover state: pos ({h['state'][0]:.2f}, {h['state'][1]:.2f}), vel ({h['state'][2]:.2f}, "
              f"{h['state'][3]:.2f}), {h['dist_goal']:.2f} from goal")
        print(f"Policy at hover: mean action {h['a_mean'].round(3)}, pre-tanh std {h['std'].round(3)} "
              f"-> velocity noise ~{h['vel_noise_per_step']:.3f} per step")
        print(f"Critic: Q(mean action) = {h['q_mean_action']:.2f} | Q(toward goal) = {h['q_toward_goal']:.2f} "
              f"| grid max {h['q_grid'].max():.2f}, min {h['q_grid'].min():.2f}")
        th, tp = h["true_hover"], h["true_pd"]
        print(f"Truth from hover ({BUDGET} steps): keep hovering -> {th['event']}, disc. return {th['disc_return']:.2f}"
              f" | PD to goal -> {tp['event']} in {tp['steps']} steps, disc. return {tp['disc_return']:.2f}")
        for k, d in h["explore"].items():
            print(f"Exploration from hover, noise x{k:2d}: " + ", ".join(f"{e} {v:.0%}" for e, v in sorted(d.items())))

    # ---------------- figure ----------------
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4.8 * n), squeeze=False)
    for row, (out, env, agent) in enumerate(results):
        h = out["hover"]
        ax = axes[row, 0]
        env.reset()
        env.render(ax=ax, show_trajectory=False, title=f"{out['name']}\n(deterministic: {out['event']})")
        ax.plot(out["traj"][:, 0], out["traj"][:, 1], "k-", lw=2, label="policy")
        if h is not None:
            tp = h["true_pd"]["traj"]
            ax.plot(tp[:, 0], tp[:, 1], "g--", lw=2, label=f"PD from hover: {h['true_pd']['event']}")
        ax.legend(fontsize=8, loc="upper left")

        ax = axes[row, 1]
        if h is not None:
            im = ax.imshow(h["q_grid"], origin="lower", extent=[-1, 1, -1, 1], cmap="viridis")
            plt.colorbar(im, ax=ax, label="Q(s_hover, a)")
            ax.plot(*h["a_mean"], "rx", ms=12, mew=3, label=f"policy mean (Q={h['q_mean_action']:.1f})")
            ax.plot(*h["a_goal"], "w^", ms=10, label=f"toward goal (Q={h['q_toward_goal']:.1f})")
            ax.set(xlabel="a_x", ylabel="a_y", title="Critic's value of each action at the hover state")
            ax.legend(fontsize=8, loc="lower left")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "policy reaches goal:\nno hover state", ha="center", va="center")

        ax = axes[row, 2]
        ax.plot(out["q_pred"], label="Q predicted (critic)")
        ax.plot(out["q_true"], label="actual discounted return-to-go")
        ax.set(xlabel="time step along deterministic rollout", ylabel="value",
               title=f"Critic accuracy (mean error {np.mean(out['q_pred'] - out['q_true']):+.1f})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=90)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
