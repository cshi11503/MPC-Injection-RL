"""
Sanity tests for the navigation environment.
Run from repo root:   python -m pytest tests -q      (or)   python tests/test_navigation_env.py

Baseline layout (default): start (1,1), goal (9,9), one obstacle (5,5) radius 2.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import NavigationEnv  # noqa: E402


# ------------------------------------------------------------ baseline layout
def test_baseline_layout_is_default():
    env = NavigationEnv()
    obs, info = env.reset()
    assert np.allclose(info["state"], [1.0, 1.0, 0.0, 0.0])
    assert np.allclose(info["goal"], [9.0, 9.0])
    assert np.allclose(info["obstacles"], [[5.0, 5.0, 2.0]])
    assert env.obs_dim == 9 and obs.shape == (9,)
    # Same layout every episode, regardless of seed
    _, info2 = env.reset(seed=123)
    assert np.allclose(info2["obstacles"], info["obstacles"])
    # The straight start->goal line is blocked by the obstacle
    assert env._line_blocked(info["state"][:2], info["goal"], info["obstacles"])


def test_dynamics_match_double_integrator():
    env = NavigationEnv()
    env.reset()
    env.state = np.array([3.0, 3.0, 0.5, -0.2])
    a = np.array([0.5, -1.0])                      # normalized
    accel = a * env.cfg.a_max
    dt = env.cfg.dt
    expected_p = env.state[:2] + dt * env.state[2:] + 0.5 * dt**2 * accel
    expected_v = env.state[2:] + dt * accel
    env.step(a)
    assert np.allclose(env.state[:2], expected_p)
    assert np.allclose(env.state[2:], expected_v)


def test_velocity_is_clipped():
    env = NavigationEnv()
    env.reset()
    env.state = np.array([1.0, 8.0, env.cfg.v_max, 0.0])
    env.step([1.0, 0.0])
    assert env.state[2] <= env.cfg.v_max + 1e-12


def test_observation_values_at_start():
    env = NavigationEnv()
    obs, _ = env.reset()
    expected = [
        -0.8, -0.8,      # (p - 5)/5      with p = (1, 1)
        0.0, 0.0,        # v / v_max      (at rest)
        0.8, 0.8,        # (g - p)/10     = (8, 8)/10
        0.4, 0.4,        # (c - p)/10     = (4, 4)/10
        0.2,             # r / 10         = 2/10
    ]
    assert np.allclose(obs, expected, atol=1e-6)


def test_collision_terminates_with_penalty():
    env = NavigationEnv()
    env.reset()
    env.state = np.array([2.85, 5.0, 1.0, 0.0])    # contact at x = 5 - (2 + 0.1) = 2.9
    _, r, term, trunc, info = env.step([1.0, 0.0])
    assert term and not trunc
    assert info["event"] == "collision" and info["is_collision"]
    assert r < -env.cfg.w_collision + 1.0


def test_swept_collision_no_tunneling():
    """Robot jumps over a thin obstacle within one step -> must still collide."""
    env = NavigationEnv(dt=1.0)                    # huge step to force the jump
    env.reset(options={"obstacles": [[5.0, 5.0, 0.2]]})
    env.state = np.array([4.0, 5.0, 2.0, 0.0])     # moves ~2 units, passing x = 5
    _, _, term, _, info = env.step([0.0, 0.0])
    assert term and info["event"] == "collision"


def test_goal_reached():
    env = NavigationEnv()
    env.reset()
    env.state = np.array([8.65, 9.0, 1.0, 0.0])    # goal edge at x = 8.7, moves 0.1
    _, r, term, _, info = env.step([0.0, 0.0])
    assert term and info["is_success"]
    assert r > env.cfg.w_goal - 1.0


def test_out_of_bounds():
    env = NavigationEnv()
    env.reset()
    env.state = np.array([0.12, 2.0, -1.0, 0.0])
    _, _, term, _, info = env.step([0.0, 0.0])
    assert term and info["event"] == "out_of_bounds" and info["is_collision"]


def test_time_limit_is_truncation_not_termination():
    env = NavigationEnv(max_steps=5)
    env.reset()
    for _ in range(5):
        _, _, term, trunc, _ = env.step([0.0, 0.0])  # robot stays still
    assert trunc and not term


def test_progress_reward_sign():
    env = NavigationEnv()
    env.reset()
    _, _, _, _, info = env.step([1.0, 1.0])        # accelerate toward the goal
    assert info["reward_terms"]["progress"] > 0


# ------------------------------------------- random layouts (future experiments)
def test_random_same_seed_same_layout():
    env = NavigationEnv(layout="random")
    _, i1 = env.reset(seed=7)
    _, i2 = env.reset(seed=7)
    assert np.allclose(i1["obstacles"], i2["obstacles"]) and np.allclose(i1["goal"], i2["goal"])


def test_random_layouts_are_valid():
    env = NavigationEnv(layout="random")
    cfg = env.cfg
    for seed in range(200):
        _, info = env.reset(seed=seed)
        obs_ = info["obstacles"]
        start, goal = info["state"][:2], info["goal"]
        assert np.linalg.norm(goal - start) >= cfg.min_start_goal_dist
        for i in range(len(obs_)):
            c, r = obs_[i, :2], obs_[i, 2]
            assert np.all(c - r >= cfg.passage_width - 1e-9)
            assert np.all(c + r <= cfg.arena_size - cfg.passage_width + 1e-9)
            for j in range(i + 1, len(obs_)):
                gap = np.linalg.norm(c - obs_[j, :2]) - r - obs_[j, 2]
                assert gap >= cfg.passage_width - 1e-9
        assert env._clear_of_obstacles(start, obs_, cfg.robot_radius)
        assert env._line_blocked(start, goal, obs_)


def test_random_obstacles_fixed_endpoints():
    env = NavigationEnv(layout="random", random_endpoints=False)
    layouts = set()
    for seed in range(20):
        _, info = env.reset(seed=seed)
        assert np.allclose(info["state"][:2], env.cfg.start) and np.allclose(info["goal"], env.cfg.goal)
        assert env._line_blocked(info["state"][:2], info["goal"], info["obstacles"])
        layouts.add(info["obstacles"].round(6).tobytes())
    assert len(layouts) == 20


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
