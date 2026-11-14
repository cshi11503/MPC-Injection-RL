"""
SAC sanity check on the classic Pendulum swing-up task.

Why: before using SAC on our navigation env, confirm the implementation learns a
standard benchmark. A correct SAC reaches an average return of roughly -150 to -200
within ~10-20k steps (random policy: about -1200 to -1500).

The pendulum below reimplements the standard Pendulum dynamics (same equations and
constants as Gymnasium's Pendulum-v1) so no extra dependency is needed.

Run from repo root:  python -m experiments.sanity_pendulum
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import SAC, ReplayBuffer  # noqa: E402


class Pendulum:
    """Swing a pendulum upright. obs = [cos th, sin th, th_dot], action in [-1, 1] (x2 torque)."""
    g, m, l, dt = 10.0, 1.0, 1.0, 0.05
    max_speed, max_torque, max_steps = 8.0, 2.0, 200
    obs_dim, action_dim = 3, 1

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.th = self.rng.uniform(-np.pi, np.pi)
        self.th_dot = self.rng.uniform(-1.0, 1.0)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        u = float(np.clip(action[0], -1, 1)) * self.max_torque
        th_norm = ((self.th + np.pi) % (2 * np.pi)) - np.pi
        cost = th_norm**2 + 0.1 * self.th_dot**2 + 0.001 * u**2
        self.th_dot += (3 * self.g / (2 * self.l) * np.sin(self.th) + 3.0 / (self.m * self.l**2) * u) * self.dt
        self.th_dot = np.clip(self.th_dot, -self.max_speed, self.max_speed)
        self.th += self.th_dot * self.dt
        self.t += 1
        truncated = self.t >= self.max_steps
        return self._obs(), -cost, False, truncated, {}   # never terminates, only truncates

    def _obs(self):
        return np.array([np.cos(self.th), np.sin(self.th), self.th_dot], dtype=np.float32)


def evaluate(agent, env, n_episodes=5):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(agent.act(obs, deterministic=True))
            ret += r
            done = term or trunc
        returns.append(ret)
    return float(np.mean(returns))


def main(total_steps=15_000, warmup=1_000, batch_size=256, eval_every=1_000, seed=0):
    env, eval_env = Pendulum(seed), Pendulum(seed + 1000)
    agent = SAC(env.obs_dim, env.action_dim, seed=seed)
    buf = ReplayBuffer(env.obs_dim, env.action_dim, capacity=total_steps, seed=seed)

    obs, _ = env.reset()
    t0 = time.time()
    history = []
    for step in range(1, total_steps + 1):
        # 1) act: uniform random during warm-up, then the stochastic policy
        a = env.rng.uniform(-1, 1, size=1) if step <= warmup else agent.act(obs)
        next_obs, r, term, trunc, _ = env.step(a)
        # 2) store; only TRUE terminals stop bootstrapping (Pendulum has none)
        buf.add(obs, a, r, next_obs, term)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
        # 3) learn: one gradient update per env step after warm-up
        if step > warmup:
            m = agent.update(buf.sample(batch_size))
        if step % eval_every == 0:
            ret = evaluate(agent, eval_env)
            history.append((step, ret))
            extra = f"alpha={m['alpha']:.3f} q_mean={m['q_mean']:.1f}" if step > warmup else "(warm-up)"
            print(f"step {step:6d} | eval return {ret:8.1f} | {extra} | {time.time() - t0:5.0f}s", flush=True)

    final = np.mean([r for _, r in history[-3:]])
    print(f"\nFinal (mean of last 3 evals): {final:.1f}  ->  {'PASS' if final > -250 else 'CHECK'}"
          f" (expect > -250)")
    return history


if __name__ == "__main__":
    main()
