"""
2D point-robot navigation with circular obstacles (double-integrator dynamics).

The API follows Gymnasium (but does not depend on it):
    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)

----------------------------------------------------------------------------
Baseline task (default NavConfig, layout="fixed")
----------------------------------------------------------------------------
10 x 10 arena, start (1, 1) lower-left, goal (9, 9) upper-right, one obstacle of
radius 2 at the center (5, 5). The straight start->goal line goes through the
obstacle, so the agent must learn to go around it.

----------------------------------------------------------------------------
MDP summary (matches the project proposal)
----------------------------------------------------------------------------
Physical state   z = [x, y, vx, vy]
Action           a in [-1, 1]^2  (normalized; env multiplies by a_max)
Dynamics         z_{t+1} = A z_t + B (a_max * a_t)          (exact ZOH discretization)
                 then each velocity component is clipped to [-v_max, v_max]
Reward           r = w_p (d_t - d_{t+1}) + w_g 1[goal] - w_c 1[collision] - w_t
Terminated       goal reached, obstacle hit, or robot leaves the arena
Truncated        max_steps reached (time limit -> the agent SHOULD still bootstrap)

Observation (all relative / normalized so the policy generalizes across layouts):
    [ (p - L/2)/(L/2)          (2)  absolute position in [-1, 1]
      v / v_max                (2)  velocity
      (g - p) / L              (2)  vector to goal
      for each obstacle, sorted nearest-surface first:
          (c_i - p) / L        (2)  vector to obstacle center
          r_i / L              (1)  obstacle radius ]
    dim = 6 + 3 * n_obstacles        (= 9 for the baseline)

Collision / goal / wall checks are done along the straight segment travelled
during the step (swept check), so a fast robot cannot "tunnel" through a thin
obstacle or fly through the goal between two samples.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class NavConfig:
    # --- world ---
    arena_size: float = 10.0          # square arena [0, L] x [0, L]
    dt: float = 0.1                   # time step [s]
    a_max: float = 2.0                # max |acceleration| per axis
    v_max: float = 3.0                # max |velocity| per axis (clipped)
    max_steps: int = 300              # episode time limit
    robot_radius: float = 0.1
    goal_radius: float = 0.3

    # --- layout ---
    # "fixed"  : the same start, goal and obstacles every episode (values below).
    #            DEFAULT = main baseline: one large circle in the center,
    #            start in the lower-left corner, goal in the upper-right corner.
    # "random" : rejection-sampled layouts (for later experiments, NOT the baseline).
    layout: str = "fixed"
    start: tuple = (1.0, 1.0)
    goal: tuple = (9.0, 9.0)
    obstacles: tuple = ((5.0, 5.0, 2.0),)   # rows of (cx, cy, r)

    # --- random layout generation (only used when layout == "random") ---
    n_obstacles: int = 4
    obstacle_radius_range: tuple = (0.5, 1.2)
    passage_width: float = 0.6        # min free gap between obstacles and between obstacle & wall
    start_goal_clearance: float = 0.3 # min gap between start/goal and any obstacle surface
    min_start_goal_dist: float = 6.0
    p_blocking: float = 1.0           # prob. that an obstacle must block the straight start->goal line
    max_layout_tries: int = 2000

    # --- reward weights ---
    w_progress: float = 1.0
    w_goal: float = 10.0
    w_collision: float = 10.0
    w_time: float = 0.01


class NavigationEnv:
    """Double-integrator navigation task. See module docstring for the MDP."""

    def __init__(self, config: NavConfig | None = None, **overrides):
        cfg = config if config is not None else NavConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown NavConfig field: {k}")
            setattr(cfg, k, v)
        self.cfg = cfg

        # Discrete-time double integrator (exact zero-order hold).
        # The MPC must use these SAME matrices (unless studying model mismatch).
        dt = cfg.dt
        I2 = np.eye(2)
        Z2 = np.zeros((2, 2))
        self.A = np.block([[I2, dt * I2], [Z2, I2]])
        self.B = np.vstack([0.5 * dt**2 * I2, dt * I2])

        if cfg.layout == "fixed":
            self.n_obstacles = len(cfg.obstacles)
        elif cfg.layout == "random":
            self.n_obstacles = cfg.n_obstacles
        else:
            raise ValueError(f"layout must be 'fixed' or 'random', got {cfg.layout!r}")
        self.obs_dim = 6 + 3 * self.n_obstacles
        self.action_dim = 2

        self.rng = np.random.default_rng()
        # Episode variables (set in reset)
        self.state = None        # (4,) [x, y, vx, vy]
        self.goal = None         # (2,)
        self.obstacles = None    # (K, 3) rows = [cx, cy, r]
        self.start = None        # (2,)
        self.t = 0
        self.trajectory = []     # list of (2,) positions for rendering

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Start a new episode.

        seed:    re-seeds the RNG (random actions; and the layout if layout="random").
        options: optionally override the layout for this episode, e.g. for tests:
                 {"start": (x, y), "goal": (x, y), "obstacles": [[cx, cy, r], ...]}
                 Fixed layout: keys not given use the NavConfig values.
                 Random layout: keys not given are sampled.
                 The number of obstacles must equal self.n_obstacles.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        cfg = self.cfg

        if cfg.layout == "fixed" or {"start", "goal", "obstacles"} <= options.keys():
            start = np.asarray(options.get("start", cfg.start), dtype=float)
            goal = np.asarray(options.get("goal", cfg.goal), dtype=float)
            obstacles = np.asarray(options.get("obstacles", cfg.obstacles), dtype=float).reshape(-1, 3)
        else:
            start, goal, obstacles = self._sample_layout(options)

        if obstacles.shape[0] != self.n_obstacles:
            raise ValueError(
                f"Expected {self.n_obstacles} obstacles, got {obstacles.shape[0]}"
            )

        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = np.array([start[0], start[1], 0.0, 0.0])
        self.t = 0
        self.trajectory = [start.copy()]

        info = self._info(event=None, reward_terms=None)
        return self._obs(), info

    def step(self, action):
        cfg = self.cfg
        a_norm = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        accel = cfg.a_max * a_norm

        p0 = self.state[:2].copy()
        z1 = self.A @ self.state + self.B @ accel
        z1[2:] = np.clip(z1[2:], -cfg.v_max, cfg.v_max)
        p1 = z1[:2]

        self.state = z1
        self.t += 1
        self.trajectory.append(p1.copy())

        # --- which event (if any) happened first along the segment p0 -> p1 ---
        event = self._first_event(p0, p1)

        # --- reward ---
        d0 = np.linalg.norm(self.goal - p0)
        d1 = np.linalg.norm(self.goal - p1)
        terms = {
            "progress": float(cfg.w_progress * (d0 - d1)),
            "goal": cfg.w_goal if event == "goal" else 0.0,
            "collision": -cfg.w_collision if event in ("collision", "out_of_bounds") else 0.0,
            "time": -cfg.w_time,
        }
        reward = float(sum(terms.values()))

        terminated = event is not None
        truncated = (not terminated) and self.t >= cfg.max_steps

        info = self._info(event=event, reward_terms=terms)
        return self._obs(), reward, terminated, truncated, info

    def sample_action(self):
        """Uniform random action in [-1, 1]^2 (used for warm-up exploration)."""
        return self.rng.uniform(-1.0, 1.0, size=2)

    # ------------------------------------------------------------ internals
    def _obs(self):
        cfg = self.cfg
        L = cfg.arena_size
        p, v = self.state[:2], self.state[2:]

        rel_c = self.obstacles[:, :2] - p                     # (K, 2)
        radii = self.obstacles[:, 2]
        surface_dist = np.linalg.norm(rel_c, axis=1) - radii  # distance to obstacle edge
        order = np.argsort(surface_dist)                      # nearest first

        obs_parts = [
            (p - L / 2) / (L / 2),
            v / cfg.v_max,
            (self.goal - p) / L,
            np.column_stack([rel_c[order] / L, radii[order, None] / L]).ravel(),
        ]
        return np.concatenate(obs_parts).astype(np.float32)

    def _info(self, event, reward_terms):
        """Raw (un-normalized) quantities. The MPC reads its inputs from here."""
        return {
            "state": self.state.copy(),
            "goal": self.goal.copy(),
            "obstacles": self.obstacles.copy(),
            "event": event,                        # None | "goal" | "collision" | "out_of_bounds"
            "is_success": event == "goal",
            "is_collision": event in ("collision", "out_of_bounds"),
            "distance_to_goal": float(np.linalg.norm(self.goal - self.state[:2])),
            "reward_terms": reward_terms,
            "t": self.t,
        }

    def get_model(self):
        """Static quantities the MPC needs (constant for the env's lifetime)."""
        cfg = self.cfg
        return {
            "A": self.A.copy(),
            "B": self.B.copy(),
            "dt": cfg.dt,
            "a_max": cfg.a_max,
            "v_max": cfg.v_max,
            "arena_size": cfg.arena_size,
            "robot_radius": cfg.robot_radius,
            "goal_radius": cfg.goal_radius,
        }

    def _first_event(self, p0, p1):
        """
        Sweep the robot along p0 -> p1 and return the event that happens first,
        or None. Each check returns the fraction t in [0, 1] of the step at which
        it first occurs.
        """
        cfg = self.cfg
        R = cfg.robot_radius
        candidates = []

        t_goal = _segment_circle_entry(p0, p1, self.goal, cfg.goal_radius)
        if t_goal is not None:
            candidates.append((t_goal, "goal"))

        for cx, cy, r in self.obstacles:
            t_hit = _segment_circle_entry(p0, p1, np.array([cx, cy]), r + R)
            if t_hit is not None:
                candidates.append((t_hit, "collision"))

        t_wall = _segment_box_exit(p0, p1, lo=R, hi=cfg.arena_size - R)
        if t_wall is not None:
            candidates.append((t_wall, "out_of_bounds"))

        if not candidates:
            return None
        return min(candidates, key=lambda c: c[0])[1]

    def _sample_layout(self, options):
        """
        Rejection sampling of (obstacles, start, goal).

        Solvability guarantee: obstacles are pairwise separated, and separated
        from the walls, by at least `passage_width` (> robot diameter). Disjoint
        disks strictly inside a square cannot enclose a region, so the free space
        is connected and the goal is always reachable from the start.
        """
        cfg = self.cfg
        L = cfg.arena_size
        r_lo, r_hi = cfg.obstacle_radius_range
        gap = cfg.passage_width
        assert gap > 2 * cfg.robot_radius, "passage_width must exceed robot diameter"

        for _ in range(cfg.max_layout_tries):
            # 1) obstacles
            if "obstacles" in options:
                obstacles = np.asarray(options["obstacles"], dtype=float).reshape(-1, 3)
            else:
                obstacles = []
                for _k in range(cfg.n_obstacles):
                    placed = False
                    for _try in range(100):
                        r = self.rng.uniform(r_lo, r_hi)
                        c = self.rng.uniform(r + gap, L - r - gap, size=2)
                        if all(np.linalg.norm(c - o[:2]) >= r + o[2] + gap for o in obstacles):
                            obstacles.append(np.array([c[0], c[1], r]))
                            placed = True
                            break
                    if not placed:
                        break
                if len(obstacles) < cfg.n_obstacles:
                    continue
                obstacles = np.array(obstacles)

            # 2) start and goal (several tries per obstacle layout)
            margin = cfg.robot_radius + cfg.start_goal_clearance
            need_block = self.rng.random() < cfg.p_blocking
            for _sg in range(50):
                start = np.asarray(options["start"], float) if "start" in options \
                    else self.rng.uniform(margin, L - margin, size=2)
                goal = np.asarray(options["goal"], float) if "goal" in options \
                    else self.rng.uniform(margin, L - margin, size=2)

                if np.linalg.norm(goal - start) < cfg.min_start_goal_dist:
                    continue
                if not self._clear_of_obstacles(start, obstacles, margin):
                    continue
                if not self._clear_of_obstacles(goal, obstacles, cfg.goal_radius + cfg.start_goal_clearance):
                    continue
                if need_block and not self._line_blocked(start, goal, obstacles):
                    continue
                return start, goal, obstacles

        raise RuntimeError("Could not sample a valid layout; relax NavConfig constraints.")

    @staticmethod
    def _clear_of_obstacles(point, obstacles, margin):
        d = np.linalg.norm(obstacles[:, :2] - point, axis=1) - obstacles[:, 2]
        return bool(np.all(d >= margin))

    def _line_blocked(self, start, goal, obstacles):
        """True if the straight start->goal path would hit at least one obstacle."""
        R = self.cfg.robot_radius
        return any(
            _segment_circle_entry(start, goal, o[:2], o[2] + R) is not None for o in obstacles
        )

    # ------------------------------------------------------------ rendering
    def render(self, ax=None, show_trajectory=True, title=None):
        """Draw arena, obstacles, start, goal and the trajectory so far. Returns the Axes."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        cfg = self.cfg
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        L = cfg.arena_size
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, lw=2, color="k"))

        for cx, cy, r in self.obstacles:
            ax.add_patch(Circle((cx, cy), r, color="dimgray", alpha=0.8))
        ax.add_patch(Circle(self.goal, cfg.goal_radius, color="tab:green", alpha=0.6))
        ax.plot(*self.start, "o", color="tab:blue", ms=7, label="start")
        ax.plot(*self.goal, "*", color="darkgreen", ms=12, label="goal")

        if show_trajectory and len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], "-", color="tab:orange", lw=1.5, label="path")
            ax.add_patch(Circle(traj[-1], cfg.robot_radius, color="tab:red"))
        if title:
            ax.set_title(title)
        return ax

    def config_dict(self):
        return asdict(self.cfg)


# ---------------------------------------------------------------- geometry
def _segment_circle_entry(p0, p1, center, radius):
    """
    First fraction t in [0, 1] at which the point p(t) = p0 + t (p1 - p0) is
    inside the circle (|p(t) - c| <= radius), or None if it never is.

    Solve |f + t d|^2 = R^2 with f = p0 - c, d = p1 - p0:
        (d.d) t^2 + 2 (f.d) t + (f.f - R^2) = 0  -> take the smaller root.
    """
    f = np.asarray(p0, float) - np.asarray(center, float)
    d = np.asarray(p1, float) - np.asarray(p0, float)
    c = f @ f - radius**2
    if c <= 0.0:                 # already inside at t = 0
        return 0.0
    a = d @ d
    if a == 0.0:                 # not moving
        return None
    b = 2.0 * (f @ d)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:               # line misses the circle
        return None
    t = (-b - np.sqrt(disc)) / (2.0 * a)
    return t if 0.0 <= t <= 1.0 else None


def _segment_box_exit(p0, p1, lo, hi):
    """First fraction t in [0, 1] at which p(t) leaves [lo, hi]^2, or None."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    if np.any(p0 < lo) or np.any(p0 > hi):
        return 0.0
    ts = []
    for i in range(2):
        if p1[i] < lo:
            ts.append((lo - p0[i]) / (p1[i] - p0[i]))
        elif p1[i] > hi:
            ts.append((hi - p0[i]) / (p1[i] - p0[i]))
    return min(ts) if ts else None


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    # python -m environment.navigation_env   (run from the repo root)
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results", exist_ok=True)
    env = NavigationEnv()   # baseline: fixed center-circle layout

    # 1) Six random-policy rollouts on the baseline layout
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for i, ax in enumerate(axes.ravel()):
        obs, info = env.reset(seed=i)
        ret, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(env.sample_action())
            ret += r
            done = term or trunc
        env.render(ax=ax, title=f"seed {i}: {info['event'] or 'timeout'}, "
                                f"T={info['t']}, R={ret:.1f}")
    fig.suptitle("Random policy on the baseline layout")
    fig.tight_layout()
    fig.savefig("results/env_demo_random.png", dpi=110)

    # 2) A simple proportional-derivative "go to goal" controller (ignores obstacles)
    fig, ax = plt.subplots(figsize=(5, 5))
    obs, info = env.reset()
    ret, done = 0.0, False
    while not done:
        s = info["state"]
        a = (1.5 * (info["goal"] - s[:2]) - 2.0 * s[2:]) / env.cfg.a_max
        obs, r, term, trunc, info = env.step(a)
        ret += r
        done = term or trunc
    env.render(ax=ax, title=f"PD (ignores obstacle): {info['event'] or 'timeout'}, R={ret:.1f}")
    fig.tight_layout()
    fig.savefig("results/env_demo_pd.png", dpi=110)

    print("obs_dim =", env.obs_dim, "| saved results/env_demo_random.png, results/env_demo_pd.png")
