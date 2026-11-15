
import heapq
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .ocp import OCPSolver
from .solvers import PDIPSolver


class MPC:
    def __init__(self, ocp_solver: OCPSolver, solver: PDIPSolver):
        self.ocp_solver = ocp_solver
        self.solver = solver

        # Dynamics and problem dimensions
        self.Ak = None
        self.Bk = None
        self.N = None
        self.nx = None
        self.nu = None

        # MPC quadratic program matrices
        self.big_P = None
        self.big_p = None
        self.A_eq_con = None
        self.bc_rhs = None
        self.G_ineq = None
        self.h_ineq = None

        # Navigation parameters
        self.a_max = 2.0
        self.waypoint_tolerance = 0.35
        self.obstacle_margin = 0.7
        self.waypoints = []
        self.waypoint_index = 0

    def init_mpc(
        self,
        Q: jax.Array,
        R: jax.Array,
        N: int,
        Ak: jax.Array,
        Bk: jax.Array,
        x_min: jax.Array,
        x_max: jax.Array,
        u_min: jax.Array,
        u_max: jax.Array,
    ):
        """Initialize the finite-horizon MPC quadratic program."""

        self.Ak = jnp.asarray(Ak)
        self.Bk = jnp.asarray(Bk)
        self.N = N
        self.nx = Ak.shape[0]
        self.nu = Bk.shape[1]

        self.big_P = self.ocp_solver.compute_cost_term(Q, R, N)
        self.big_p = jnp.zeros(self.big_P.shape[0])

        self.A_eq_con = self.ocp_solver.compute_equality_matrix(
            self.Ak, self.Bk, N
        )

        self.bc_rhs = self.ocp_solver.compute_equality_bc(
            jnp.zeros(self.nx), self.Ak, N
        )

        self.G_ineq, self.h_ineq = (
            self.ocp_solver.compute_ineq_con(
                x_min, x_max, u_min, u_max, N
            )
        )

        return (
            self.big_P,
            self.big_p,
            self.A_eq_con,
            self.bc_rhs,
            self.G_ineq,
            self.h_ineq,
        )

    def solve_mpc(
        self, x0: jax.Array, x_goal: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Solve MPC and return the first optimal acceleration."""

        if self.big_P is None:
            raise RuntimeError("Call init_mpc() before solve_mpc().")

        x0 = jnp.asarray(x0)
        x_goal = jnp.asarray(x_goal)

        # Update dynamics constraints using current state
        self.bc_rhs = self.ocp_solver.compute_equality_bc(
            x0, self.Ak, self.N
        )

        # Reference trajectory: constant waypoint target
        goal_blocks = []

        for _ in range(self.N):
            goal_blocks.append(x_goal)
            goal_blocks.append(jnp.zeros(self.nu))

        goal_blocks.append(x_goal)

        z_goal = jnp.concatenate(goal_blocks)

        # Linear term of the quadratic tracking cost
        p_goal = -self.big_P @ z_goal

        # Solve constrained quadratic program
        self.solver.init_problem(
            self.big_P,
            p_goal,
            self.A_eq_con,
            self.bc_rhs,
            self.G_ineq,
            self.h_ineq,
        )

        #self.solver.solve_qp()
        self.solver.solve_qp(verbose=True)

        z_opt = self.solver.x0

        # Decision vector: [x0, u0, x1, u1, ...]
        u_cmd = z_opt[self.nx:self.nx + self.nu]

        return u_cmd, z_opt

    @staticmethod
    def _segment_clear(p1, p2, obstacles, margin):
        """Check whether a line segment avoids all obstacles."""

        direction = p2 - p1
        length_sq = float(direction @ direction)

        for cx, cy, radius in obstacles:
            center = np.array([cx, cy], dtype=float)
            safe_radius = radius + margin

            if length_sq < 1e-12:
                closest = p1
            else:
                t = np.clip(
                    np.dot(center - p1, direction) / length_sq,
                    0.0, 1.0
                )
                closest = p1 + t * direction

            if np.linalg.norm(closest - center) < safe_radius:
                return False

        return True

    def plan_waypoints(self, start, goal, obstacles, arena_size):
        """Find a collision-free waypoint route using a visibility graph."""

        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 3)

        nodes = [start, goal]

        # Sample candidate waypoints around each circular obstacle
        for cx, cy, radius in obstacles:
            safe_radius = radius + self.obstacle_margin + 0.05

            for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                point = np.array([
                    cx + safe_radius * np.cos(angle),
                    cy + safe_radius * np.sin(angle),
                ])

                if not np.all(
                    (point >= self.obstacle_margin)
                    & (point <= arena_size - self.obstacle_margin)
                ):
                    continue

                if self._segment_clear(
                    point, point, obstacles, self.obstacle_margin
                ):
                    nodes.append(point)

        # Construct visibility graph
        graph = [[] for _ in nodes]

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self._segment_clear(
                    nodes[i], nodes[j],
                    obstacles, self.obstacle_margin
                ):
                    distance = np.linalg.norm(nodes[i] - nodes[j])
                    graph[i].append((j, distance))
                    graph[j].append((i, distance))

        # Dijkstra shortest-path search
        distances = [float("inf")] * len(nodes)
        previous = [-1] * len(nodes)
        distances[0] = 0.0

        queue = [(0.0, 0)]

        while queue:
            cost, current = heapq.heappop(queue)

            if cost > distances[current]:
                continue

            if current == 1:
                break

            for neighbor, weight in graph[current]:
                new_cost = cost + weight

                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    previous[neighbor] = current
                    heapq.heappush(queue, (new_cost, neighbor))

        if not np.isfinite(distances[1]):
            raise RuntimeError("No collision-free waypoint path found.")

        # Reconstruct path from goal to start
        path = []
        current = 1

        while current != -1:
            path.append(nodes[current])
            current = previous[current]

        path.reverse()

        self.waypoints = path[1:]
        self.waypoint_index = 0

        return self.waypoints

    def step(self, env):
        """Compute a normalized MPC action for NavigationEnv."""

        if env.state is None:
            raise RuntimeError("Reset the environment first.")

        x0 = np.asarray(env.state, dtype=float)
        position = x0[:2]

        # Plan a route when starting a new episode
        if not self.waypoints:
            self.plan_waypoints(
                position,
                env.goal,
                env.obstacles,
                env.cfg.arena_size,
            )

        # Move to the next waypoint when sufficiently close
        while self.waypoint_index < len(self.waypoints) - 1:
            target = self.waypoints[self.waypoint_index]

            if np.linalg.norm(position - target) > self.waypoint_tolerance:
                break

            self.waypoint_index += 1

        waypoint = self.waypoints[self.waypoint_index]

        # Target position with zero desired velocity
        x_goal = jnp.array([
            waypoint[0], waypoint[1], 0.0, 0.0
        ])

        u_cmd, _ = self.solve_mpc(jnp.asarray(x0), x_goal)

        # Convert physical acceleration to normalized RL action
        action = np.asarray(u_cmd) / env.cfg.a_max

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def reset(self):
        """Clear waypoint information for a new episode."""

        self.waypoints = []
        self.waypoint_index = 0
