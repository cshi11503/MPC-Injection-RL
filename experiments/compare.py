"""
Head-to-head comparison on the SAME fresh random layouts (deterministic policies):

    python -m experiments.compare                       # top-5 of the rho=0.5 run, rho=0 SAC, MPC

Policies run on --episodes layouts; the MPC (slow: ~9 s / episode) on the first --mpc-episodes
of them. Per-episode results are saved to --out, so paired statistics can be recomputed.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNS = "results/runs"
RHO5, RHO0 = f"{RUNS}/sac_rho0.5_random_fixedends_seed0", f"{RUNS}/sac_rho0_random_fixedends_cudagraph_seed0"


def _policy(job):
    from training.train import test_checkpoint
    path, env_cfg, hidden, seeds = job
    return test_checkpoint(path, env_cfg, hidden, seeds)


def _mpc(seed):
    from experiments import generate_mpc_data as g
    seed, event, ret, rows = g._episode(seed)
    return [event == "goal", event == "collision", ret, len(rows)]


def _mpc_init(n_obstacles):
    from experiments import generate_mpc_data as g
    g._init(n_obstacles)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def row(name, r):
    n, k = len(r), int(r[:, 0].sum())
    lo, hi = wilson(k, n)
    ok = r[:, 0] > 0
    return (f"{name:<28} n={n:<6} success {k / n:7.2%} [{lo:.2%}, {hi:.2%}]  collision {r[:, 1].mean():6.2%}  "
            f"timeout {1 - r[:, 0].mean() - r[:, 1].mean():6.2%}  return {r[:, 2].mean():6.2f}  "
            f"steps(success) {r[ok, 3].mean():5.1f}")


def paired(a, b, name_a, name_b):
    """Exact McNemar test on the layouts where exactly one of the two succeeds."""
    n = min(len(a), len(b))
    sa, sb = a[:n, 0] > 0, b[:n, 0] > 0
    only_a, only_b = int((sa & ~sb).sum()), int((~sa & sb).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    both = sa & sb
    return (f"{name_a} vs {name_b} (n={n}): only {name_a} solves {only_a}, only {name_b} solves {only_b}, "
            f"p={p:.3g}; steps where both succeed {a[:n][both, 3].mean():.1f} vs {b[:n][both, 3].mean():.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--mpc-episodes", type=int, default=2_000)
    p.add_argument("--seed-base", type=int, default=50_000, help="disjoint from eval/val/test/MPC-data seeds")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default="results/comparison/compare.npz")
    args = p.parse_args()

    cfg = json.load(open(f"{RHO5}/config.json"))
    env_cfg, hidden = cfg["env"], cfg["args"]["hidden"]
    assert env_cfg == json.load(open(f"{RHO0}/config.json"))["env"], "runs were trained on different tasks"
    policies = {f"rho0.5 step {t['step']}": f"{RHO5}/{t['file']}" for t in json.load(open(f"{RHO5}/top5.json"))}
    policies |= {"rho0 best.pt": f"{RHO0}/best.pt", "rho0 final.pt (5M)": f"{RHO0}/final.pt"}
    seeds = list(range(args.seed_base, args.seed_base + args.episodes))
    chunks = [seeds[i:i + 250] for i in range(0, len(seeds), 250)]

    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JAX_PLATFORMS="cpu",
                      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    res = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for name, path in policies.items():
            res[name] = np.concatenate(pool.map(_policy, [(path, env_cfg, hidden, c) for c in chunks]))
            print(row(name, res[name]), flush=True)
    n_obs = len(env_cfg["obstacles"])
    with ctx.Pool(args.workers, initializer=_mpc_init, initargs=(n_obs,)) as pool:
        res["MPC"] = np.array(pool.map(_mpc, seeds[:args.mpc_episodes], chunksize=4), dtype=float)
    print(row("MPC", res["MPC"]), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, seeds=seeds, names=list(res), **{f"r{i}": r for i, r in enumerate(res.values())})

    top = [n for n in res if n.startswith("rho0.5")]
    best = max(top, key=lambda n: (res[n][:, 0].mean(), res[n][:, 2].mean()))
    print(f"\nbest rho=0.5 policy: {best}\n\n-- top-5 against each other --")
    for n in top:
        if n != best:
            print(paired(res[best], res[n], best, n))
    print("\n-- against the MPC (same layouts) --")
    print(row(best + " (MPC subset)", res[best][:args.mpc_episodes]))
    print(paired(res[best], res["MPC"], best, "MPC"))
    print("\n-- against SAC rho=0 --")
    for n in ("rho0 best.pt", "rho0 final.pt (5M)"):
        print(paired(res[best], res[n], best, n))


if __name__ == "__main__":
    main()
