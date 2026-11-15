
"""
Run pure MPC in the 2D navigation environment.

From the repository root:
    python -m experiments.run_mpc
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from environment.navigation_env import NavigationEnv, NavConfig
from controller.mpc import MPC
from controller.ocp import OCPSolver
from controller.solvers import PDIPSolver


def main():

    # 1. Initialize the same environment used by SAC
    env = NavigationEnv(NavConfig(layout="fixed"))
    obs, info = env.reset(seed=42)

    # 2. Get the exact dynamics from the environment
    A = jnp.asarray(env.A)
    B = jnp.asarray(env.B)

    # 3. MPC parameters
    N = 10  # prediction horizon (10 steps = 1 second)

    Q = jnp.diag(jnp.array([
        10.0, 10.0,  # position tracking
        1.0, 1.0     # velocity tracking
    ]))

    R = jnp.diag(jnp.array([
        0.1, 0.1     # acceleration penalty
    ]))

    # 4. State and control constraints
    L = env.cfg.arena_size
    v_max = env.cfg.v_max
    a_max = env.cfg.a_max

    x_min = jnp.array([0.0, 0.0, -v_max, -v_max])
    x_max = jnp.array([L, L, v_max, v_max])

    u_min = jnp.array([-a_max, -a_max])
    u_max = jnp.array([a_max, a_max])

    # 5. Initialize MPC controller
    ocp_solver = OCPSolver()
    qp_solver = PDIPSolver()

    mpc = MPC(ocp_solver, qp_solver)

    mpc.init_mpc(
        Q=Q,
        R=R,
        N=N,
        Ak=A,
        Bk=B,
        x_min=x_min,
        x_max=x_max,
        u_min=u_min,
        u_max=u_max,
    )

    mpc.reset()

    # 6. Run MPC in the environment
    total_reward = 0.0
    terminated = False
    truncated = False

    for step in range(env.cfg.max_steps):

        # MPC produces an action in [-1, 1]^2
        action = mpc.step(env)

        # Apply action to the same environment used by SAC
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    # 7. Print episode results
    print("\n========== MPC Results ==========")
    print(f"Steps: {env.t}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Final position: {env.state[:2]}")
    print(f"Goal position: {env.goal}")
    print(f"Distance to goal: {info['distance_to_goal']:.3f}")
    print(f"Event: {info['event']}")
    print(f"Success: {info['is_success']}")
    print(f"Collision: {info['is_collision']}")
    print(f"Time limit reached: {truncated}")

    # 8. Compute navigation path length
    trajectory = np.asarray(env.trajectory)
    path_length = np.linalg.norm(
        np.diff(trajectory, axis=0), axis=1
    ).sum()

    print(f"Path length: {path_length:.3f}")

    # 9. Plot the actual MPC trajectory
    fig, ax = plt.subplots(figsize=(7, 7))

    env.render(
        ax=ax,
        show_trajectory=True,
        title="Pure MPC Navigation"
    )
    waypoints = np.array(mpc.waypoints)

    ax.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        "bo--",
        label="MPC waypoints",
    )

    ax.legend()

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
