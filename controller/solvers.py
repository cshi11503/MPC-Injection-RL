import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from typing import Tuple

class PDIPSolver():
    """
    Follows the implementation of the primal-dual interior point algorithm as found in
        "CVXGEN: a code generator for embedded convex optimization", J. Mattingley and S. Boyd (2012).
    """
    def __init__(self, max_qp_iter: int = 100, tol: float = 1e-2):
        """
        Attributes:
            max_qp_iter (int): maximum number of QP iterations
            tol (float): positive value for measuring QP residual convergence

            nx (int): number of primal variables
            n_eq (int): number of equality constraints
            n_ineq (int): number of inequality constraints

            x0 (jnp.array): primal solution x
            s0 (jnp.array): slack variable s
            z0 (jnp.array): dual variables for inequality constraints z
            y0 (jnp.array): dual variables for equality constraints y
        """
        self.max_qp_iter = max_qp_iter
        self.tol = tol

        self.nx = None
        self.n_eq = None
        self.n_ineq = None

        self.x0 = None
        self.s0 = None
        self.z0 = None
        self.y0 = None

    def init_problem(self, P: jax.Array, p: jax.Array, A: jax.Array, b: jax.Array, G: jax.Array, h: jax.Array) -> None:
        self.P = P
        self.p = p
        self.A = A
        self.b = b
        self.G = G
        self.h = h

        self.nx = P.shape[0]
        self.n_eq = A.shape[0]
        self.n_ineq = G.shape[0]

        # raise NotImplementedError("PDIPSolver.init_problem has yet to be implemented")

    def init_soln(self):
        
        """
        Follows initialization procedure from Section 5.2 in CVXGEN paper

        Returns:
            x0 (jnp.array): init. soln. for primary variable x
            s0 (jnp.array): init. soln. for slack variable s
            z0 (jnp.array): init. soln. for dual variables associated with inequality constraints
            y0 (jnp.array): init. soln. for dual variables associated with equality constraints
        """
        I = jnp.eye(self.G.shape[0])

        # KKT Matrix
        zero_ineq_eq = jnp.zeros((self.n_ineq, self.n_eq))
        zero_eq_ineq = jnp.zeros((self.n_eq, self.n_ineq))
        zero_eq_eq = jnp.zeros((self.n_eq, self.n_eq))
        K_init = jnp.block([
            [self.P, self.G.T, self.A.T],
            [self.G, -I,       zero_ineq_eq],
            [self.A, zero_eq_ineq, zero_eq_eq]
            ])

        rhs = jnp.concatenate([
            -self.p,
            self.h,
            self.b
            ])
        
        sol = jnp.linalg.solve(K_init, rhs)

        x = sol[:self.nx]
        z = sol[self.nx : self.nx + self.n_ineq]
        y = sol[self.nx + self.n_ineq :]

        # compute s0
        z = self.G @ x - self.h
        alpha_p = jnp.max(z)

        s0 = jnp.where(alpha_p < 0, -z, -z + (1 + alpha_p) * jnp.ones_like(z))

        alpha_d = -jnp.min(z)

        z0 = jnp.where(alpha_d < 0, z, z + (1 + alpha_d) * jnp.ones_like(z))

        # in conclusion
        x0 = x
        y0 = y
        return x0, s0, z0, y0

        # raise NotImplementedError("PDIPSolver.init_soln has yet to be implemented")

    def compute_residuals(self, xbar: jax.Array, sbar: jax.Array, zbar: jax.Array, ybar: jax.Array):
        """
        Implements the residuals associated with the right hand side of the KKT system.

        Parameters:
            xbar (jnp.array): Current primal variable x
            sbar (jnp.array): Current slack variable s
            zbar: (jnp.array): Current dual variable z for inequality constraints
            sbar: (jnp.array): Current dual variable s for equality constraints

        Returns:
            r1 (jnp.array): Residual for KKT derivative
            r2 (jnp.array): Residual for complementary slackness
            r3 (jnp.array): Residual for inequality constraint
            r4 (jnp.array): Residual for equality constraint
        """
        r1 = -(self.A.T @ ybar + self.G.T @ zbar + self.P @ xbar + self.p)
        r2 = -(sbar * zbar)
        r3 = -(self.G @ xbar + sbar - self.h)
        r4 = -(self.A @ xbar - self.b)

        return r1, r2, r3, r4
        # raise NotImplementedError("PDIPSolver.compute_residuals has yet to be implemented")

    def compute_centering_plus_corrector(self, s0: jax.Array, ds: jax.Array, z0: jax.Array, dz: jax.Array) -> None:
        """
        Computes the sigma and mu terms used for the centering-plus-corrector step from Section 5.2 in CVXGEN paper.
        Hint: you should be able to reuse self.compute_line_search here.

        Parameters:
            s0 (jnp.array): current iterate for slack variable s.
            ds (jnp.array): descent direction for slack variable s.
            z0 (jnp.array): current iterate for dual variable z.
            dz (jnp.array): descent direction for dual variable s.

        Returns:
            mu (float): mu term for centering-corrector step
            sigma (float): sigma term for centering-corrector step
        """
        mu = (s0 @ z0) / self.n_ineq
        alpha, line_search_successful = self.compute_line_search(s0, ds, z0, dz)
        sigma = (((s0 + alpha * ds) @ (z0 + alpha * dz))/ (s0 @ z0)) ** 3

        return mu, sigma
        # raise NotImplementedError("PDIPSolver.compute_centering_plus_corrector has yet to be implemented")

    def compute_line_search(self, s0: jax.Array, ds: jax.Array, z0: jax.Array, dz: jax.Array, n_steps: int = 500, tol: float = 1e-4):
        """
        Parameters:
            s0 (jnp.array): current iterate for slack variable s.
            ds (jnp.array): descent direction for slack variable s.
            z0 (jnp.array): current iterate for dual variable z.
            dz (jnp.array): descent direction for dual variable s.
            n_steps (int): max number of iterations for line search.
            tol (float): positive number tolerance for passing line search (i.e., s+ds >= tol)

        Returns:
            alpha (float): step size value
            line_search_successful (bool): indicates if line search found valid solution
        """

        # Start with the largest possible step
        alpha = 1.0
        line_search_successful = False
        
        # Search from alpha = 1 toward alpha = 0
        
        for _ in range(n_steps):
            s_new = s0 + alpha * ds
            z_new = z0 + alpha * dz
            # Check whether both s and z remain positive
            if jnp.all(s_new >= tol) and jnp.all(z_new >= tol):
                line_search_successful = True
                break

            # Reduce the step size
            alpha -= 1.0 / n_steps
            alpha = max(alpha, 0.0)

        return alpha, line_search_successful


        # raise NotImplementedError("PDIPSolver.compute_line_search has yet to be implemented")

    def has_converged(self) -> bool:
        """
        Given current solution iterate (x0, s0, z0, y0), computes whether the
        current QP residuals have been sufficiently minimized.

        Returns:
            converged (bool): true if QP solver has minimized residuals.

        """
        r1, r2, r3, r4 = self.compute_residuals(
            self.x0, self.s0, self.z0, self.y0
            )
        converged = (
            jnp.linalg.norm(r1) <= self.tol
            and jnp.linalg.norm(r2) <= self.tol
            and jnp.linalg.norm(r3) <= self.tol
            and jnp.linalg.norm(r4) <= self.tol
            )
        return bool(converged)
        # raise NotImplementedError("PDIPSolver.has_converged has yet to be implemented")

    def factorize_kkt_system(self, sbar: jax.Array, zbar: jax.Array):

        raise NotImplementedError("PDIPSolver.factorize_kkt_system has yet to be implemented")  

    def solve_kkt_system(self, r1: jax.Array, r2: jax.Array, r3: jax.Array, r4: jax.Array, sbar: jax.Array, zbar: jax.Array):
        """
        Solves the KKT system at the current iteration using the residual values.

        Parameters:
            r1 (jnp.array): Residual for KKT derivative
            r2 (jnp.array): Residual for complementary slackness
            r3 (jnp.array): Residual for inequality constraint
            r4 (jnp.array): Residual for equality constraint
            sbar (jnp.array)
            zbar (jnp.array)

        Returns:
            dx (jnp.array): descent direction for primal x
            ds (jnp.array): descent direction for slack variable s
            dz (jnp.array): descent direction for dual variable z
            dy (jnp.array): descent direction for dual variable y
        """
        S = jnp.diag(sbar)
        Z = jnp.diag(zbar)

        zero_x_ineq = jnp.zeros((self.nx, self.n_ineq))
        zero_ineq_x = jnp.zeros((self.n_ineq, self.nx))
        zero_ineq_eq = jnp.zeros((self.n_ineq, self.n_eq))
        zero_eq_ineq = jnp.zeros((self.n_eq, self.n_ineq))
        zero_eq_eq = jnp.zeros((self.n_eq, self.n_eq))

        I = jnp.eye(self.n_ineq)

        KKT = jnp.block([
            [self.P,       zero_x_ineq, self.G.T,       self.A.T],
            [zero_ineq_x,  Z,           S,              zero_ineq_eq],
            [self.G,       I,           jnp.zeros((self.n_ineq, self.n_ineq)), zero_ineq_eq],
            [self.A,       zero_eq_ineq, zero_eq_ineq,  zero_eq_eq]
            ])

        rhs = jnp.concatenate([r1, r2, r3, r4])

        delta = jnp.linalg.solve(KKT, rhs)

        dx = delta[:self.nx]
        ds = delta[self.nx:self.nx+self.n_ineq]
        dz = delta[self.nx+self.n_ineq:self.nx+2*self.n_ineq]
        dy = delta[self.nx+2*self.n_ineq:]

        return dx, ds, dz, dy
        # raise NotImplementedError("PDIPSolver.solve_kkt_system has yet to be implemented")

    def solve_qp(self, verbose: bool = False):
        """
        Solves the QP by:
        1. Initialize the primal, slack, and dual variables using self.init_soln
        2. Carry out max_qp_iter number of iterations of solving the KKT system
            2a. At each iteration, use self.compute_residuals and self.solve_kkt_system
            2b. Use self.compute_centering_plus_corrector for correcting dz and ds corrections
            2c. Use self.compute_line_search to compute the step size
            2d. Check for convergence

        Returns
            costs (list): a list of the objective function across iterations
        """
        # 1. Initialize the primal, slack, and dual variables using self.init_soln
        self.x0, self.s0, self.z0, self.y0 = self.init_soln()

        costs = []

        # 2. Carry out max_qp_iter number of iterations of solving the KKT system
        for k in range(self.max_qp_iter):
            # Store current objective value
            cost = (0.5 * self.x0.T @ self.P @ self.x0+ self.p.T @ self.x0)
            costs.append(cost)

            # 2d. Check for convergence
            # put it here instead of the end of the loop
            # for the case if there is a convergence at the initial guess 
            if self.has_converged():
                if verbose:
                    print(f"Converged after {k} iterations")
                break

            # 2a. At each iteration, use self.compute_residuals and self.solve_kkt_system
            r1, r2, r3, r4 = self.compute_residuals(self.x0, self.s0, self.z0, self.y0)

            dx_aff, ds_aff, dz_aff, dy_aff = self.solve_kkt_system(r1, r2, r3, r4, self.s0, self.z0)

            #2b. Use self.compute_centering_plus_corrector for correcting dz and ds corrections
            mu, sigma = self.compute_centering_plus_corrector(self.s0,ds_aff,self.z0,dz_aff)
    
            # Centering-plus-corrector RHS
            u1 = jnp.zeros(self.nx)
            u2 = (
                sigma * mu * jnp.ones(self.n_ineq)
                - ds_aff * dz_aff
            )
            u3 = jnp.zeros(self.n_ineq)
            u4 = jnp.zeros(self.n_eq)
            # solve second KKT system for cc
            dx_cc, ds_cc, dz_cc, dy_cc = self.solve_kkt_system(
                u1,
                u2,
                u3,
                u4,
                self.s0,
                self.z0
            )
            dx = dx_aff + dx_cc
            ds = ds_aff + ds_cc
            dz = dz_aff + dz_cc
            dy = dy_aff + dy_cc
            #  Find maximum feasible step
            alpha_max, line_search_successful = self.compute_line_search(
                self.s0,
                ds,
                self.z0,
                dz
            )

            if not line_search_successful:
                if verbose:
                    print("Line search failed.")
                break
            # based on the seond equation CVXGEN gievs 
            alpha = min(1.0, 0.99 * alpha_max)

            self.x0 = self.x0 + alpha * dx
            self.s0 = self.s0 + alpha * ds
            self.z0 = self.z0 + alpha * dz
            self.y0 = self.y0 + alpha * dy
            if verbose:
                print(
                    f"Iteration {k}: "
                    f"cost = {cost}, alpha = {alpha}"
                )
        return costs

      

        # raise NotImplementedError("PDIPSolver.solve_qp has yet to be implemented")
