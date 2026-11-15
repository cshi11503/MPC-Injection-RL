"""
Roll out the MPC on random layouts: verifies it solves them AND records the
transitions as the MPC-Injection dataset (same .npz format as ReplayBuffer).

    python -m experiments.generate_mpc_data --episodes 2000 --out data/mpc_dataset.npz

Every transition the MPC actually executed is kept (collisions included: they are
real env transitions and SAC is off-policy). Episodes where the planner / QP raised
are dropped and counted as failures.
"""
import argparse
import contextlib
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_env = _mpc = None


def _init(n_obstacles):
    global _env, _mpc
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from controller.mpc import MPC
    from controller.ocp import OCPSolver
    from controller.solvers import PDIPSolver
    from environment import NavigationEnv

    # the task SAC trains on: random obstacles, fixed start (1,1) / goal (9,9)
    _env = NavigationEnv(layout="random", n_obstacles=n_obstacles, random_endpoints=False)
    L, v, a = _env.cfg.arena_size, _env.cfg.v_max, _env.cfg.a_max
    _mpc = MPC(OCPSolver(), PDIPSolver())
    _mpc.init_mpc(Q=jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0])), R=jnp.diag(jnp.array([0.1, 0.1])), N=10,
                  Ak=jnp.asarray(_env.A), Bk=jnp.asarray(_env.B),
                  x_min=jnp.array([0, 0, -v, -v]), x_max=jnp.array([L, L, v, v]),
                  u_min=jnp.array([-a, -a]), u_max=jnp.array([a, a]))


def _episode(seed):
    obs, info = _env.reset(seed=seed)
    _mpc.reset()
    rows, ret = [], 0.0
    try:
        with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):   # solver is verbose
            while True:
                action = _mpc.step(_env)
                next_obs, r, term, trunc, info = _env.step(action)
                rows.append((obs, action, r, next_obs, float(term)))
                obs, ret = next_obs, ret + r
                if term or trunc:
                    break
        event = info["event"] or "timeout"
    except RuntimeError as e:
        return seed, f"error: {e}", 0.0, []
    return seed, event, ret, rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed-base", type=int, default=1_000_000, help="layout seeds (disjoint from eval seeds)")
    p.add_argument("--n-obstacles", type=int, default=4)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", type=str, default="data/mpc_dataset.npz")
    args = p.parse_args()

    # one thread per worker: the QPs are tiny, parallelism comes from the pool
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
                      JAX_PLATFORMS="cpu")
    t0 = time.time()
    events, returns, steps, rows = {}, [], [], []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.n_obstacles,)) as pool:
        seeds = range(args.seed_base, args.seed_base + args.episodes)
        for i, (seed, event, ret, ep_rows) in enumerate(pool.imap_unordered(_episode, seeds), 1):
            key = "error" if event.startswith("error") else event
            events[key] = events.get(key, 0) + 1
            if key != "goal":
                print(f"  seed {seed}: {event}", flush=True)
            if key == "goal":
                returns.append(ret)
                steps.append(len(ep_rows))
            rows += ep_rows
            if i % 100 == 0:
                print(f"{i}/{args.episodes} episodes | {events} | {time.time() - t0:.0f}s", flush=True)

    n = args.episodes
    print(f"\nMPC on {n} random layouts: " + ", ".join(f"{k} {v} ({v / n:.1%})" for k, v in sorted(events.items())))
    if returns:
        print(f"successful episodes: return {np.mean(returns):.2f} +- {np.std(returns):.2f}, "
              f"steps {np.mean(steps):.1f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = list(zip(*rows))
    np.savez_compressed(args.out, obs=np.array(cols[0], np.float32), action=np.array(cols[1], np.float32),
                        reward=np.array(cols[2], np.float32), next_obs=np.array(cols[3], np.float32),
                        terminated=np.array(cols[4], np.float32))
    print(f"saved {len(rows)} transitions -> {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
