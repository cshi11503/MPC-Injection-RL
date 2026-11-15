
from controller import mpc
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from environment.navigation_env import NavigationEnv, NavConfig
from controller.mpc import MPC
from controller.ocp import OCPSolver
from controller.solvers import PDIPSolver


def main():
    num_episodes = 10

    # Start with one obstacle, but randomize each layout
    env = NavigationEnv(
        NavConfig(layout="random", n_obstacles=1)
    )

    # Use the SAME MPC parameters as your successful run
    N = 10
    Q = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))
    R = jnp.diag(jnp.array([0.1, 0.1]))

    L = env.cfg.arena_size
    v_max = env.cfg.v_max
    a_max = env.cfg.a_max

    mpc = MPC(OCPSolver(), PDIPSolver())

    mpc.init_mpc(
        Q=Q,
        R=R,
        N=N,
        Ak=jnp.asarray(env.A),
        Bk=jnp.asarray(env.B),
        x_min=jnp.array([0, 0, -v_max, -v_max]),
        x_max=jnp.array([L, L, v_max, v_max]),
        u_min=jnp.array([-a_max, -a_max]),
        u_max=jnp.array([a_max, a_max]),
    )

    successes = 0
    collisions = 0
    successful_steps = []
    successful_lengths = []

    for episode in range(num_episodes):
    #for debug
    #for episode in [9]:
        obs, info = env.reset(seed=42 + episode)
        mpc.reset()
        print(f"Episode {episode + 1}, seed={42 + episode}")
        print("Start:", env.state[:2])
        print("Goal:", env.goal)
        print("Obstacles:", env.obstacles)
        print("Margin:", mpc.obstacle_margin)

        event = None

        try:
            for step in range(env.cfg.max_steps):
                action = mpc.step(env)
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

            event = info["event"]

        except RuntimeError as error:
            event = "solver/planner failure"
            print(f"Episode {episode + 1}: {error}")

        success = event == "goal"
        collision = event == "collision"

        successes += int(success)
        collisions += int(collision)

        trajectory = np.asarray(env.trajectory)
        path_length = np.linalg.norm(
            np.diff(trajectory, axis=0), axis=1
        ).sum()

        if success:
            successful_steps.append(env.t)
            successful_lengths.append(path_length)

        print(
            f"Episode {episode + 1}: "
            f"Event={event}, "
            f"Steps={env.t}, "
            f"Path={path_length:.2f}"
        )

    print("\n========== MPC Evaluation ==========")
    print(f"Episodes: {num_episodes}")
    print(f"Success rate: {successes / num_episodes:.1%}")
    print(f"Collision rate: {collisions / num_episodes:.1%}")
    print(f"Other failures: {num_episodes - successes - collisions}")

    if successes > 0:
        print(f"Average successful steps: {np.mean(successful_steps):.1f}")
        print(f"Average successful path length: {np.mean(successful_lengths):.2f}")


if __name__ == "__main__":
    main()
