import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from typing import Tuple

from ocp import OCPSolver
from solvers import PDIPSolver



class MPC():

    def __init__(self, ocp_solver: OCPSolver, solver: PDIPSolver):
        self.ocp_solver = ocp_solver
        self.solver = solver

        # Store dynamics matrices
        self.Ak = None
        self.Bk = None
        self.N = None
        self.nx = None
        self.nu = None

        # Store problem matrices for MPC
        self.big_P = None
        self.big_p = None
        self.A_eq_con = None
        self.bc_rhs = None
        self.G_ineq = None
        self.h_ineq = None


    def init_mpc(self, 
                 Q: jax.Array, 
                 R: jax.Array, 
                 N: int, 
                 Ak: jax.Array, 
                 Bk: jax.Array, 
                 x_min: jax.Array, 
                 x_max: jax.Array, 
                 u_min: jax.Array, 
                 u_max: jax.Array):
        """
        Initializes the MPC problem by computing the cost matrix and constraint matrices.

        Parameters:
            Q (jax.Array): (nx, nx) matrix for a single time step (i.e., x_k.T @ Q @ x_k)
            R (jax.Array): (nu, nu) matrix for a single time step (i.e., u_k.T @ R @ u_k)
            N (int): The number of time steps.
            Ak (jax.Array): The state transition matrix for each time step.
            Bk (jax.Array): The input matrix for each time step.
            x_min (jax.Array): The minimum state values.
            x_max (jax.Array): The maximum state values.
            u_min (jax.Array): The minimum input values.
            u_max (jax.Array): The maximum input values.


        Returns:
            big_P (jax.Array): Big cost matrix for the entire horizon.
            big_p (jax.Array): Big linear cost vector for the entire horizon.
            A_eq_con (jax.Array): Equality constraint matrix.
            bc_rhs (jax.Array): Right-hand side of the equality constraints.
            G_ineq (jax.Array): Inequality constraint matrix.
            h_ineq (jax.Array): Right-hand side of the inequality constraints.


        Instructions: 
            Use the methods implemented in ocp.py to compute the cost matrices and 
            constraint terms, and store them as attributes of the MPC class.

            These will later be reused to solve the MPC problem at each time step.
        """
        
        self.Ak = Ak
        self.Bk = Bk
        self.N = N
        self.nx = Ak.shape[0]
        self.nu = Bk.shape[1]

        self.big_P = self.ocp_solver.compute_cost_term(Q, R, N)
        self.big_p = jnp.zeros(self.big_P.shape[0])
        self.A_eq_con = self.ocp_solver.compute_equality_matrix(Ak, Bk, N)
        self.bc_rhs = self.ocp_solver.compute_equality_bc(jnp.zeros(self.nx), Ak, N)
        self.G_ineq, self.h_ineq = self.ocp_solver.compute_ineq_con(x_min, x_max, u_min, u_max, N)
        return (
            self.big_P,
            self.big_p,
            self.A_eq_con,
            self.bc_rhs,
            self.G_ineq,
            self.h_ineq
        )
       


        # raise NotImplementedError("MPC.init_mpc has yet to be implemented")


    def solve_mpc(self, x0: jax.Array, x_goal: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Solves the MPC problem for the current state and goal.

        Parameters:
            x0 (jax.Array): The current state.
            x_goal (jax.Array): The desired goal state.

        Returns:

            u_cmd (jax.Array): The optimal control input for the current step.

        Instructions:
            Use the current and goal state to compute the right hand side of the equality constraints.Then, 
            solve the QP problem using PDIP solver and return the appropriate control input from its solution.

        """
        self.bc_rhs = self.ocp_solver.compute_equality_bc(x0, self.Ak, self.N)

        # Construct goal-dependent linear cost term
        goal_blocks = []

        for k in range(self.N):
            goal_blocks.append(x_goal)
            goal_blocks.append(jnp.zeros(self.nu))
        
        goal_blocks.append(x_goal)

        z_goal = jnp.concatenate(goal_blocks)
        p_goal = -self.big_P @ z_goal

        # initialize and solve the QP
        self.solver.init_problem(
            self.big_P,
            p_goal,
            self.A_eq_con,
            self.bc_rhs,
            self.G_ineq,
            self.h_ineq
        )

        costs = self.solver.solve_qp()
        # extract first optimal control input
        z_opt = self.solver.x0
        u_cmd = z_opt[self.nx:self.nx + self.nu]

        return u_cmd, z_opt



        # raise NotImplementedError("MPC.solve_mpc has yet to be implemented")


    def step_mujoco(self, mj_data, mass, gravity, x_goal: jax.Array) -> jax.Array:
        """
        Performs one MPC step and applies the computed control input to the MuJoCo simulation.

        Parameters:
            mj_data: The MuJoCo simulation data object.
            mass (float): The mass of the helicopter supplied from simulation file.
            gravity (float): The gravitational acceleration.
            x_goal (jax.Array): The desired goal state.

        Returns: 

            force (jax.Array): The 3D force applied to the helicopter based.

        Instructions: 
            Extract the current state containing 2D position and velocity from MuJoCo simulation data.
            Use it to find the optimal force to apply to the helicopter. 
            Also account for the gravitational force in the vertical direction to ensure 
            that the helicopter is able to hover at the desired altitude.
            Return the 3D force to be applied to the helicopter.
            
            Hint: Use mj_data.qpos and mj_data.qvel to get the 3D position and velocity respectively.

        """
        # 1. Extract 2D position and velocity
        pos = jnp.array(mj_data.qpos[:2])
        vel = jnp.array(mj_data.qvel[:2])
        # Construct current state [px, py, vx, vy]
        x0 = jnp.concatenate((pos, vel))

        # 2. Solve MPC for optimal acceleration, use above function
        u_cmd, z_opt = self.solve_mpc(x0, x_goal)
        # 3. Convert acceleration to horizontal force
        force_xy = mass * u_cmd
        # 4. Compute vertical force to counteract gravity
        force_z = -mass * gravity

        # 5. Combine horizontal and vertical forces into 3D force vector
        force = jnp.array([force_xy[0], force_xy[1], force_z])

        return force



        #raise NotImplementedError("MPC.step_mujoco has yet to be implemented")

