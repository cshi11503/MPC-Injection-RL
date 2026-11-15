
import jax
import jax.numpy as jnp
import jax.scipy as jsp


class OCPSolver:
    def compute_cost_term(
        self,
        Q: jax.Array,
        R: jax.Array,
        N: int
    ) -> jax.Array:
        """
        Construct the quadratic cost matrix for MPC.

        Decision vector:
        [x0, u0, x1, u1, ..., xN]

        Q: state cost matrix (4 x 4)
        R: control cost matrix (2 x 2)
        N: prediction horizon
        """

        # Cost for each state-control pair
        stage_cost = jsp.linalg.block_diag(Q, R)

        # Repeat stage cost over the horizon
        P = jnp.kron(jnp.eye(N), stage_cost)

        # Add terminal state cost
        P = jsp.linalg.block_diag(P, Q)

        return P

    def compute_equality_matrix(
        self,
        Ak: jax.Array,
        Bk: jax.Array,
        N: int
    ) -> jax.Array:
        """
        Construct the equality constraint matrix.

        Dynamics:
            x_(k+1) = Ak @ x_k + Bk @ u_k

        Equality constraints:
            Abar @ z = bvec
        """

        nx = Ak.shape[0]
        nu = Bk.shape[1]

        n = (N + 1) * nx + N * nu

        Abar = jnp.zeros(((N + 1) * nx, n))

        # Initial state constraint
        Abar = Abar.at[:nx, :nx].set(jnp.eye(nx))

        for k in range(N):
            row = (k + 1) * nx
            col = k * (nx + nu)

            # -Ak @ x_k
            Abar = Abar.at[
                row:row + nx,
                col:col + nx
            ].set(-Ak)

            # -Bk @ u_k
            Abar = Abar.at[
                row:row + nx,
                col + nx:col + nx + nu
            ].set(-Bk)

            # +x_(k+1)
            next_col = col + nx + nu

            Abar = Abar.at[
                row:row + nx,
                next_col:next_col + nx
            ].set(jnp.eye(nx))

        return Abar

    def compute_equality_bc(
        self,
        x0: jax.Array,
        Ak: jax.Array,
        N: int
    ) -> jax.Array:
        """
        Construct the equality constraint right-hand side.

        The first constraint fixes the initial state.
        All remaining constraints enforce dynamics.
        """

        nx = Ak.shape[0]

        bvec = jnp.zeros((N + 1) * nx)

        bvec = bvec.at[:nx].set(x0)

        return bvec

    def compute_ineq_con(
        self,
        x_min: jax.Array,
        x_max: jax.Array,
        u_min: jax.Array,
        u_max: jax.Array,
        N: int
    ) -> tuple[jax.Array, jax.Array]:
        """
        Construct inequality constraints:

            Gmat @ z <= hvec

        State constraints:
            x_min <= x_k <= x_max

        Control constraints:
            u_min <= u_k <= u_max
        """

        nx = x_min.shape[0]
        nu = u_min.shape[0]

        n = (N + 1) * nx + N * nu
        m = N * (nx + nu)

        # Matrix selecting bounded variables
        C = jnp.zeros((m, n))

        lower = []
        upper = []
        row = 0

        for k in range(N):

            # Control bounds for u_k
            col_u = k * (nx + nu) + nx

            C = C.at[
                row:row + nu,
                col_u:col_u + nu
            ].set(jnp.eye(nu))

            lower.append(u_min)
            upper.append(u_max)

            row += nu

            # State bounds for x_(k+1)
            col_x = (k + 1) * (nx + nu)

            C = C.at[
                row:row + nx,
                col_x:col_x + nx
            ].set(jnp.eye(nx))

            lower.append(x_min)
            upper.append(x_max)

            row += nx

        lower = jnp.concatenate(lower)
        upper = jnp.concatenate(upper)

        # Convert lower and upper bounds into Gz <= h
        Gmat = jnp.vstack((C, -C))
        hvec = jnp.concatenate((upper, -lower))

        return Gmat, hvec
