import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

class OCPSolver():
    def compute_cost_term(self, Q: jax.Array, R: jax.Array, N: int) -> jax.Array:
        """
        Parameters:
            Q (jax.Array): (nx, nx) matrix for a single time step (i.e., x_k.T @ Q @ x_k)
            R (jax.Array): (nu, nu) matrix for a single time step (i.e., u_k.T @ R @ u_k)
            N (int): number of time steps for optimal control problem.

        Returns:
            P (jax.Array): large matrix for evaluating quadratic (i.e. x @ P @ x)
        """
        stage_cost = jsp.linalg.block_diag(Q, R)
        P = jnp.kron(jnp.eye(N), stage_cost)
         # Add terminal state cost
        P = jsp.linalg.block_diag(P, Q)

        return P
        # raise NotImplementedError("OCPSolver.compute_cost_term has yet to be implemented")

    def compute_equality_matrix(self, Ak: jax.Array, Bk: jax.Array, N: int) -> jax.Array:
        """
        Computes Abar for the consolidated linear equality constraint Abar @ x = bvec.

        Parameters:
            Ak (jax.Array): discretized double integrator Ak matrix.
            Bk (jax.Array): discretized double integrator Bk matrix.
            N (int): number of time steps for optimal control problem.

        Returns:
            Amat (jax.Array): consolidated equality constraint matrix.
        """
        nx = Ak.shape[0]
        nu = Bk.shape[1]

        n = (N + 1) * nx + N * nu

        Abar = jnp.zeros(((N + 1) * nx, n))

        # Initial state constraint: q0 = q_init
        Abar = Abar.at[:nx, :nx].set(jnp.eye(nx))

        for k in range(N):
            row = (k + 1) * nx
            col = k * (nx + nu)

            # -Ak @ q_k
            Abar = Abar.at[row:row+nx,col:col+nx].set(-Ak)

            # -Bk @ u_k
            Abar = Abar.at[row:row+nx,col+nx:col+nx+nu].set(-Bk)
            # +q_(k+1)
            next_col = col + nx + nu
            Abar = Abar.at[row:row+nx,next_col:next_col+nx].set(jnp.eye(nx))
        return Abar




        # raise NotImplementedError("OCPSolver.compute_equality_matrix has yet to be implemented")

    def compute_equality_bc(self, x0: jax.Array, Ak: jax.Array, N:int) -> jax.Array:
        """
        Computes bvec for the consolidated linear equality constraint Abar @ x = bvec.

        Parameters:
            x0 (jax.Array): initial state for problem.
            Ak (jax.Array): discretized double integrator Ak matrix.
            N (int): number of time steps for optimal control problem.

        Returns:
            Amat (jax.Array): consolidated equality constraint matrix.
        """
        nx = Ak.shape[0]
        bvec = jnp.zeros((N + 1) * nx)
        bvec = bvec.at[:nx].set(x0)
        
        return bvec
    

        # raise NotImplementedError("OCPSolver.compute_equality_bc has yet to be implemented")

    # Casey's added code to avoid confusion of x and q change the input name
    # def compute_ineq_con(self, x_min: jax.Array, x_max: jax.Array, u_min: jax.Array, u_max: jax.Array, N: int) -> tuple[jax.Array, jax.Array]:
    def compute_ineq_con(self, q_min: jax.Array, q_max: jax.Array, u_min: jax.Array, u_max: jax.Array, N: int) -> tuple[jax.Array, jax.Array]:
        """
        Computes Gmat and hvec for the consolidated linear inequality constraint Gmat @ x <= hvec.

        Parameters:
            q_min (jax.Array): lower bound for state vector.
            q_max (jax.Array): upper bound for state vector.
            u_min (jax.Array): lower bound for control vector.
            u_max (jax.Array): upper bound for control vector.
            N (int): number of time steps for optimal control problem.

        Returns:
            Gmat (jax.Array): consolidated inequality constraint matrix.
            hvec (jax.Array): consolidated inequality constraint vector.
        """
        # Number of state and control variables
        nx = q_min.shape[0]
        nu = u_min.shape[0]

        # Total number of optimization variables
        n = (N + 1) * nx + N * nu
        m = N * (nx + nu)

        # Selection matrix for bounded variables
        C = jnp.zeros((m, n))

        lower = []
        upper = []
        row = 0

        for k in range(N):
            # Control u_k
            col_u = k * (nx + nu) + nx
            C = C.at[
                row:row+nu,
                col_u:col_u+nu
            ].set(jnp.eye(nu))

            lower.append(u_min)
            upper.append(u_max)
            row += nu
            # State q_(k+1)
            col_q = (k + 1) * (nx + nu)
            C = C.at[
                row:row+nx,
                col_q:col_q+nx
            ].set(jnp.eye(nx))
            
            lower.append(q_min)
            upper.append(q_max)
            row += nx

        lower = jnp.concatenate(lower)
        upper = jnp.concatenate(upper)
        Gmat = jnp.vstack((C, -C))
        hvec = jnp.concatenate((upper, -lower))

        return Gmat, hvec
        # raise NotImplementedError("OCPSolver.compute_ineq_con has yet to be implemented")
