import abc
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.nn.initializers import normal
import equinox as eqx
import sympy as sp
from jaxtyping import PRNGKeyArray

from damped_linoss.models.common import GLU, simple_uniform_init
from damped_linoss.models.block_deer import block_deer_rollout_from_linearizer

# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence.
    Assumes a diagonal matrix A.

    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N : 1 * N]
    iB_ = A_i[1 * N : 2 * N]
    iC_ = A_i[2 * N : 3 * N]
    iD_ = A_i[3 * N : 4 * N]
    jA_ = A_j[0 * N : 1 * N]
    jB_ = A_j[1 * N : 2 * N]
    jC_ = A_j[2 * N : 3 * N]
    jD_ = A_j[3 * N : 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j


class _AbstractLinOSSLayer(eqx.Module):
    @abc.abstractmethod
    def _recurrence(self):
        raise NotImplementedError
    

class IMLayer(_AbstractLinOSSLayer):
    A_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        A_max: float,
        dt_std: float,
        key: PRNGKeyArray,
        **kwargs, 
    ):
        A_key, B_key, C_key, D_key, dt_key, key = jr.split(key, 6) # converts a single PRNG key into a specified number of new indep keys by adding a leading axis
        self.dt = normal(stddev=dt_std)(dt_key, (state_dim,))
        self.A_diag = jr.uniform(A_key, shape=(state_dim,)) * A_max
        self.B = simple_uniform_init(B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim))
        self.C = simple_uniform_init(C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim))
        # initializer that returns arrays whose values are normally distributed with mean 0 and stdev=sd
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))

    def _recurrence(self, A_diag, dt, Bu_elements):
        """Compute the LxP output of LinOSS-IM given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            dt              (float32):    discretization time-step  (P,)
            Bu_elements     (complex64):  B @ u                     (L, P)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        sql = Bu_elements.shape[0] #sq length

        S = 1.0 + dt**2.0 * A_diag # this is acc the inverse of S from the linOSS paper
        M_11 = 1.0 - dt**2.0 * A_diag / S
        M_12 = -1.0 * dt * A_diag / S
        M_21 = dt / S
        M_22 = 1 / S

        M = jnp.concatenate([M_11, M_12, M_21, M_22])
        M_elements = M * jnp.ones((sql, 4 * A_diag.shape[0]))

        F1 = M_11 * Bu_elements * dt
        F2 = M_21 * Bu_elements * dt
        F = jnp.hstack((F1, F2))

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, A_diag.shape[0] :] #xs = [zs, ys]^T in the paper 

        return ys
    
    def __call__(self, input_sequence):
        # Materialize parameters
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        # Project
        dt = nn.sigmoid(self.dt)
        A_diag = nn.relu(self.A_diag)

        # Apply SSM
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence) #automatically vectorizes a function designed for sngl data points
        ys = self._recurrence(A_diag, dt, Bu_elements)
        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)

        return xs


class IMEXLayer(_AbstractLinOSSLayer):
    A_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        A_max: float, 
        dt_std: float,
        key: PRNGKeyArray,
        **kwargs,
    ):
        A_key, B_key, C_key, D_key, dt_key, key = jr.split(key, 6)
        self.dt = normal(stddev=dt_std)(dt_key, (state_dim,))
        self.A_diag = jr.uniform(A_key, shape=(state_dim,)) * A_max
        self.B = simple_uniform_init(B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim))
        self.C = simple_uniform_init(C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim))
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))
    
    def _recurrence(self, A_diag, dt, Bu_elements):
        """Compute the LxP output of LinOSS-IMEX given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            dt              (float32):    discretization time-step  (P,)
            Bu_elements     (complex64):  B @ u                     (L, P)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        sql = Bu_elements.shape[0]

        A_ = jnp.ones_like(A_diag)
        B_ = -1.0 * dt * A_diag
        C_ = dt
        D_ = 1.0 - (dt**2.0) * A_diag

        M = jnp.concatenate([A_, B_, C_, D_])
        M_elements = M * jnp.ones((sql, 4 * A_diag.shape[0]))

        F1 = Bu_elements * dt
        F2 = Bu_elements * (dt**2.0)
        F = jnp.hstack((F1, F2))

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, A_diag.shape[0] :]

        return ys

    def __call__(self, input_sequence):
        # Materialize parameters
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        # Project
        dt = nn.sigmoid(self.dt)
        A_diag = nn.relu(self.A_diag)

        # Apply SSM
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)
        ys = self._recurrence(A_diag, dt, Bu_elements)
        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)

        return xs
    


class DampedIMEX1Layer(_AbstractLinOSSLayer):
    """
    IMEX damped oscillatory layer with bounded per-mode damping ratio gating.

    Main supported variants:
      - damping_mode = "constant"
      - damping_mode = "input",       gate_variant = "simple"
      - damping_mode = "input",       gate_variant = "energy"
      - damping_mode = "state_input", gate_variant = "simple"
      - damping_mode = "state_input", gate_variant = "energy"

    Here
        G_{i,k} = 2 * omega_i * zeta_{i,k},
        omega_i = sqrt(A_i),
        zeta_{i,k} in [zeta_min, zeta_max].
    """

    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array
    state_dim: int

    damping_mode: str
    gate_variant: str
    zeta_min: float
    zeta_max: float

    use_block_deer: bool
    deer_num_iters: int
    deer_damping: float

    # input gate
    u_to_zeta: eqx.nn.Linear | None
    input_rho: jax.Array | None
    input_bias: jax.Array | None

    # state-input gate
    si_u_to_zeta: eqx.nn.Linear | None
    si_w_z_re: jax.Array | None
    si_w_z_im: jax.Array | None
    si_w_x_re: jax.Array | None
    si_w_x_im: jax.Array | None
    si_bias: jax.Array | None
    si_rho: jax.Array | None

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        G_min: float,
        G_max: float,
        A_min: float,
        A_max: float,
        dt_std: float,
        damping_mode: str = "constant",
        gate_variant: str = "simple",
        gate_type: str = "linear",          # kept for backward compatibility
        mult_min: float = 0.25,             # unused, kept for backward compatibility
        mult_max: float = 4.0,              # unused, kept for backward compatibility
        freq_aware_damping: bool = False,   # unused, kept for backward compatibility
        zeta_min: float = 0.0,
        zeta_max: float = 4.0,
        use_block_deer: bool = False,
        deer_num_iters: int = 4,
        deer_damping: float = 0.0,
        key: PRNGKeyArray = None,
        **kwargs,
    ):
        self.state_dim = state_dim
        self.damping_mode = damping_mode
        self.gate_variant = gate_variant
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max

        self.use_block_deer = use_block_deer
        self.deer_num_iters = deer_num_iters
        self.deer_damping = deer_damping

        init_key, B_key, C_key, D_key, gate_key, key = jr.split(key, 6)

        if initialization == "uniform":
            self.A_diag, self.G_diag, self.dt = self._uniform_init_AGdt(
                A_min, A_max, G_min, G_max, dt_std, init_key
            )
        elif initialization == "ring":
            self.A_diag, self.G_diag, self.dt = self._ring_init_AGdt(
                r_min, r_max, theta_min, theta_max, dt_std, init_key
            )
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

        self.B = simple_uniform_init(
            B_key,
            shape=(state_dim, hidden_dim, 2),
            std=1.0 / jnp.sqrt(hidden_dim),
        )
        self.C = simple_uniform_init(
            C_key,
            shape=(hidden_dim, state_dim, 2),
            std=1.0 / jnp.sqrt(state_dim),
        )
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))

        # defaults
        self.u_to_zeta = None
        self.input_rho = None
        self.input_bias = None

        self.si_u_to_zeta = None
        self.si_w_z_re = None
        self.si_w_z_im = None
        self.si_w_x_re = None
        self.si_w_x_im = None
        self.si_bias = None
        self.si_rho = None

        if gate_type != "linear":
            raise NotImplementedError(
                "This refactor currently supports only linear pre-activations."
            )

        if damping_mode == "input":
            k_u, k_rho = jr.split(gate_key, 2)
            self.u_to_zeta = eqx.nn.Linear(hidden_dim, state_dim, key=k_u)
            self.input_bias = jnp.zeros((state_dim,))
            if gate_variant == "energy":
                # scan-compatible energy from current forcing magnitude |Bu_k|^2
                self.input_rho = jnp.zeros((state_dim,))
            elif gate_variant != "simple":
                raise ValueError(f"Unknown gate_variant={gate_variant}")

        elif damping_mode == "state_input":
            ku, k1, k2, k3, k4, k5 = jr.split(gate_key, 6)
            scale = 1.0 / jnp.sqrt(4.0)
            self.si_u_to_zeta = eqx.nn.Linear(hidden_dim, state_dim, key=ku)
            self.si_w_z_re = normal(stddev=scale)(k1, (state_dim,))
            self.si_w_z_im = normal(stddev=scale)(k2, (state_dim,))
            self.si_w_x_re = normal(stddev=scale)(k3, (state_dim,))
            self.si_w_x_im = normal(stddev=scale)(k4, (state_dim,))
            self.si_bias = jnp.zeros((state_dim,))
            if gate_variant == "energy":
                self.si_rho = jnp.zeros((state_dim,))
            elif gate_variant != "simple":
                raise ValueError(f"Unknown gate_variant={gate_variant}")

        elif damping_mode != "constant":
            raise ValueError(f"Unknown damping_mode={damping_mode}")

    # ------------------------------------------------------------------
    # initialization / projection
    # ------------------------------------------------------------------

    def _is_valid_AGdt(self, A_diag, G_diag, dt):
        dt = nn.sigmoid(dt)
        return (G_diag >= 0) & (((G_diag - dt * A_diag) ** 2 - 4 * A_diag) < 0)

    def _ring_init_AGdt(self, r_min, r_max, theta_min, theta_max, dt_std, key):
        a, g, dt, lam1, lam2 = sp.symbols("a g dt lam1 lam2")

        M_i = sp.Matrix([
            [1 / (1 + dt * g), -a * dt / (1 + dt * g)],
            [dt / (1 + dt * g), 1 - a * dt**2 / (1 + dt * g)],
        ])
        eigs = list(M_i.eigenvals().keys())
        eqs = [sp.Eq(eigs[0], lam1), sp.Eq(eigs[1], lam2)]
        sol = sp.solve(eqs, (a, g))[0]
        f = sp.lambdify((lam1, lam2, dt), sol, "numpy")

        mag_key, arg_key, dt_key = jr.split(key, 3)
        dt_vals = normal(stddev=dt_std)(dt_key, (self.state_dim,))
        dt_sigmoid = nn.sigmoid(dt_vals)

        mag = jnp.sqrt(
            jr.uniform(mag_key, shape=(self.state_dim,)) * (r_max**2 - r_min**2) + r_min**2
        )
        arg = jr.uniform(arg_key, shape=(self.state_dim,)) * (theta_max - theta_min) + theta_min
        lam1_vals = mag * jnp.cos(arg) + 1j * mag * jnp.sin(arg)
        lam2_vals = mag * jnp.cos(arg) - 1j * mag * jnp.sin(arg)

        a_vals, g_vals = f(lam1_vals, lam2_vals, dt_sigmoid)

        h1 = sp.lambdify((a, g, dt), eigs[0], "numpy")
        h2 = sp.lambdify((a, g, dt), eigs[1], "numpy")
        lam1_out_vals = h1(a_vals, g_vals, dt_sigmoid)
        lam2_out_vals = h2(a_vals, g_vals, dt_sigmoid)

        invertible = jnp.all(
            jnp.isclose(lam1_out_vals, lam1_vals) |
            jnp.isclose(jnp.conjugate(lam1_out_vals), lam1_vals)
        ) & jnp.all(
            jnp.isclose(lam2_out_vals, lam2_vals) |
            jnp.isclose(jnp.conjugate(lam2_out_vals), lam2_vals)
        )
        stable = jnp.all(jnp.abs(lam1_out_vals) < 1.0) & jnp.all(jnp.abs(lam2_out_vals) < 1.0)
        valid = jnp.all(self._is_valid_AGdt(a_vals, g_vals, dt_vals))

        print(f"Invertibility check: {invertible}")
        print(f"Stability check: {stable}")
        print(f"Validity check: {valid}")

        return a_vals.real, g_vals.real, dt_vals

    def _uniform_init_AGdt(self, A_min, A_max, G_min, G_max, dt_std, key):
        bsz = 512
        done = False
        A_vals, G_vals, dt_vals = [], [], []

        while not done:
            A_key, G_key, dt_key, key = jr.split(key, 4)
            A_diag = jr.uniform(A_key, shape=(bsz,)) * (A_max - A_min) + A_min
            G_diag = jr.uniform(G_key, shape=(bsz,)) * (G_max - G_min) + G_min
            dt = normal(stddev=dt_std)(dt_key, (bsz,))

            mask = self._is_valid_AGdt(A_diag, G_diag, dt)
            A_vals.extend(list(A_diag[mask]))
            G_vals.extend(list(G_diag[mask]))
            dt_vals.extend(list(dt[mask]))

            if len(A_vals) >= self.state_dim:
                done = True

        return (
            jnp.array(A_vals[:self.state_dim]),
            jnp.array(G_vals[:self.state_dim]),
            jnp.array(dt_vals[:self.state_dim]),
        )

    def _project_A_dt(self, A_diag, dt):
        dt = nn.sigmoid(dt)
        A_diag = jnp.maximum(A_diag, 0.0)
        return A_diag, dt

    def _project_G(self, G_diag, A_diag, dt):
        G_diag = jnp.maximum(G_diag, 0.0)

        A_low = (2 + dt * G_diag - 2 * jnp.sqrt(1 + dt * G_diag)) / jnp.maximum(dt**2, 1e-6)
        A_high = (2 + dt * G_diag + 2 * jnp.sqrt(1 + dt * G_diag)) / jnp.maximum(dt**2, 1e-6)

        A_proj = A_low + nn.relu(A_diag - A_low) - nn.relu(A_diag - A_high)
        return A_proj, G_diag

    # ------------------------------------------------------------------
    # gate helpers
    # ------------------------------------------------------------------

    def _omega_from_A(self, A_diag):
        return jnp.sqrt(jnp.maximum(A_diag, 1e-8))

    def _bounded_zeta(self, raw):
        return self.zeta_min + (self.zeta_max - self.zeta_min) * nn.sigmoid(raw)

    def _sigmoid_prime(self, x):
        s = nn.sigmoid(x)
        return s * (1.0 - s)

    def _state_features(self, z_prev, x_prev):
        return (
            jnp.real(z_prev),
            jnp.imag(z_prev),
            jnp.real(x_prev),
            jnp.imag(x_prev),
        )

    def _local_energy(self, z_prev, x_prev, A_diag):
        return A_diag * (jnp.abs(x_prev) ** 2) + (jnp.abs(z_prev) ** 2)

    def _input_energy(self, Bu_elements):
        # scan-compatible per-mode current forcing energy
        return jnp.abs(Bu_elements) ** 2

    def _compute_zeta_input_seq(self, input_sequence, Bu_elements):
        raw = jax.vmap(self.u_to_zeta)(input_sequence) + self.input_bias[None, :]
        if self.gate_variant == "energy":
            raw = raw + self.input_rho[None, :] * self._input_energy(Bu_elements)
        return self._bounded_zeta(raw)

    def _compute_zeta_state_input(self, u_k, z_prev, x_prev, A_diag):
        z_re, z_im, x_re, x_im = self._state_features(z_prev, x_prev)

        raw = (
            self.si_u_to_zeta(u_k)
            + self.si_w_z_re * z_re
            + self.si_w_z_im * z_im
            + self.si_w_x_re * x_re
            + self.si_w_x_im * x_im
            + self.si_bias
        )

        if self.gate_variant == "energy":
            energy = self._local_energy(z_prev, x_prev, A_diag)
            raw = raw + self.si_rho * energy

        return self._bounded_zeta(raw)

    # ------------------------------------------------------------------
    # state packing helpers
    # ------------------------------------------------------------------

    def _pack_state_real(self, z, x):
        return jnp.stack(
            [jnp.real(z), jnp.imag(z), jnp.real(x), jnp.imag(x)],
            axis=1,
        )

    def _unpack_state_real(self, packed):
        z = packed[:, 0] + 1j * packed[:, 1]
        x = packed[:, 2] + 1j * packed[:, 3]
        return z, x

    # ------------------------------------------------------------------
    # scan-compatible input path
    # ------------------------------------------------------------------

    def _compute_G_seq_input(self, input_sequence, Bu_elements, A_diag, dt):
        omega = self._omega_from_A(A_diag)
        zeta_seq = self._compute_zeta_input_seq(input_sequence, Bu_elements)
        G_seq = 2.0 * omega[None, :] * zeta_seq

        def project_one_g(g):
            A_proj, g_proj = self._project_G(g, A_diag, dt)
            return A_proj, g_proj

        A_seq, G_seq = jax.vmap(project_one_g)(G_seq)
        A_diag_final = jnp.min(A_seq, axis=0)
        return A_diag_final, G_seq

    def _recurrence(self, A_diag, G_seq, dt, Bu_elements):
        I = jnp.ones_like(A_diag)
        dt_row = dt[None, :]
        A_row = A_diag[None, :]

        S = I[None, :] + dt_row * G_seq
        M_11 = 1.0 / S
        M_12 = -(dt_row / S) * A_row
        M_21 = dt_row / S
        M_22 = 1.0 - (dt_row**2 / S) * A_row

        M_elements = jnp.concatenate([M_11, M_12, M_21, M_22], axis=1)

        F1 = dt_row * (1.0 / S) * Bu_elements
        F2 = (dt_row**2) * (1.0 / S) * Bu_elements
        F = jnp.concatenate([F1, F2], axis=1)

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, self.state_dim:]
        return ys

    # ------------------------------------------------------------------
    # sequential state-input path
    # ------------------------------------------------------------------

    def _recurrence_state_input(self, A_diag, dt, Bu_elements, input_sequence):
        omega = self._omega_from_A(A_diag)

        def step_fn(carry, xs):
            z_prev, x_prev = carry
            bu_k, u_k = xs

            zeta_k = self._compute_zeta_state_input(u_k, z_prev, x_prev, A_diag)
            G_k_raw = 2.0 * omega * zeta_k

            A_k, G_k = self._project_G(G_k_raw, A_diag, dt)

            S = 1.0 + dt * G_k
            z_next = (z_prev - dt * A_k * x_prev + dt * bu_k) / S
            x_next = x_prev + dt * z_next

            return (z_next, x_next), x_next

        z0 = jnp.zeros((self.state_dim,), dtype=Bu_elements.dtype)
        x0 = jnp.zeros((self.state_dim,), dtype=Bu_elements.dtype)
        (_, _), ys = jax.lax.scan(step_fn, (z0, x0), (Bu_elements, input_sequence))
        return ys

    # ------------------------------------------------------------------
    # unified local linearizer for DEER
    # ------------------------------------------------------------------

    def _local_linearization_state_input(self, local_state, local_driver):
        """
        local_state: (4,) = [zr, zi, xr, xi]

        local_driver:
          [a_i, dt_i, bu_re, bu_im, u_term_i,
           w_z_re_i, w_z_im_i, w_x_re_i, w_x_im_i, bias_i, rho_i]
        """
        (
            a_i, dt_i, bu_re, bu_im, u_term_i,
            w_z_re_i, w_z_im_i, w_x_re_i, w_x_im_i, bias_i, rho_i
        ) = local_driver

        zr, zi, xr, xi = local_state

        raw = (
            u_term_i
            + w_z_re_i * zr
            + w_z_im_i * zi
            + w_x_re_i * xr
            + w_x_im_i * xi
            + bias_i
        )

        if self.gate_variant == "energy":
            energy_i = a_i * (xr**2 + xi**2) + (zr**2 + zi**2)
            raw = raw + rho_i * energy_i

            draw_dzr = w_z_re_i + 2.0 * rho_i * zr
            draw_dzi = w_z_im_i + 2.0 * rho_i * zi
            draw_dxr = w_x_re_i + 2.0 * rho_i * a_i * xr
            draw_dxi = w_x_im_i + 2.0 * rho_i * a_i * xi
        else:
            draw_dzr = w_z_re_i
            draw_dzi = w_z_im_i
            draw_dxr = w_x_re_i
            draw_dxi = w_x_im_i

        sig = nn.sigmoid(raw)
        sigp = self._sigmoid_prime(raw)

        zeta_i = self.zeta_min + (self.zeta_max - self.zeta_min) * sig
        dzeta_draw = (self.zeta_max - self.zeta_min) * sigp

        omega_i = jnp.sqrt(jnp.maximum(a_i, 1e-8))
        g_raw = 2.0 * omega_i * zeta_i

        A_i_arr, G_i_arr = self._project_G(
            jnp.array([g_raw]),
            jnp.array([a_i]),
            jnp.array([dt_i]),
        )
        A_i = A_i_arr[0]
        G_i = G_i_arr[0]

        dg_dzr = 2.0 * omega_i * dzeta_draw * draw_dzr
        dg_dzi = 2.0 * omega_i * dzeta_draw * draw_dzi
        dg_dxr = 2.0 * omega_i * dzeta_draw * draw_dxr
        dg_dxi = 2.0 * omega_i * dzeta_draw * draw_dxi

        S = 1.0 + dt_i * G_i

        N_re = zr + dt_i * (-A_i * xr + bu_re)
        N_im = zi + dt_i * (-A_i * xi + bu_im)

        d_invS_dzr = -(dt_i / (S**2)) * dg_dzr
        d_invS_dzi = -(dt_i / (S**2)) * dg_dzi
        d_invS_dxr = -(dt_i / (S**2)) * dg_dxr
        d_invS_dxi = -(dt_i / (S**2)) * dg_dxi

        dzr_dzr = 1.0 / S + N_re * d_invS_dzr
        dzr_dzi = N_re * d_invS_dzi
        dzr_dxr = (-dt_i * A_i) / S + N_re * d_invS_dxr
        dzr_dxi = N_re * d_invS_dxi

        dzi_dzr = N_im * d_invS_dzr
        dzi_dzi = 1.0 / S + N_im * d_invS_dzi
        dzi_dxr = N_im * d_invS_dxr
        dzi_dxi = (-dt_i * A_i) / S + N_im * d_invS_dxi

        dxr_dzr = dt_i * dzr_dzr
        dxr_dzi = dt_i * dzr_dzi
        dxr_dxr = 1.0 + dt_i * dzr_dxr
        dxr_dxi = dt_i * dzr_dxi

        dxi_dzr = dt_i * dzi_dzr
        dxi_dzi = dt_i * dzi_dzi
        dxi_dxr = dt_i * dzi_dxr
        dxi_dxi = 1.0 + dt_i * dzi_dxi

        A_block = jnp.array([
            [dzr_dzr, dzr_dzi, dzr_dxr, dzr_dxi],
            [dzi_dzr, dzi_dzi, dzi_dxr, dzi_dxi],
            [dxr_dzr, dxr_dzi, dxr_dxr, dxr_dxi],
            [dxi_dzr, dxi_dzi, dxi_dxr, dxi_dxi],
        ])

        next_state = jnp.array([
            N_re / S,
            N_im / S,
            xr + dt_i * (N_re / S),
            xi + dt_i * (N_im / S),
        ])

        b_vec = next_state - A_block @ local_state
        return A_block, b_vec

    def _run_block_deer(self, input_sequence, A_diag, dt, B_complex):
        P = self.state_dim
        T = input_sequence.shape[0]

        initial_packed = jnp.zeros((P, 4), dtype=jnp.float32)
        states_guess = jnp.zeros((T, P, 4), dtype=jnp.float32)

        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)
        u_terms = jax.vmap(self.si_u_to_zeta)(input_sequence)

        if self.gate_variant == "energy":
            rho = self.si_rho
        else:
            rho = jnp.zeros((P,))

        local_drivers = jnp.stack([
            jnp.broadcast_to(A_diag[None, :], (T, P)),
            jnp.broadcast_to(dt[None, :], (T, P)),
            jnp.real(Bu_elements),
            jnp.imag(Bu_elements),
            u_terms,
            jnp.broadcast_to(self.si_w_z_re[None, :], (T, P)),
            jnp.broadcast_to(self.si_w_z_im[None, :], (T, P)),
            jnp.broadcast_to(self.si_w_x_re[None, :], (T, P)),
            jnp.broadcast_to(self.si_w_x_im[None, :], (T, P)),
            jnp.broadcast_to(self.si_bias[None, :], (T, P)),
            jnp.broadcast_to(rho[None, :], (T, P)),
        ], axis=-1)

        def build_linearization(states):
            if T == 1:
                A0, b0 = jax.vmap(self._local_linearization_state_input)(
                    initial_packed, local_drivers[0]
                )
                return A0[None, ...], b0[None, ...]

            A_rest, b_rest = jax.vmap(
                lambda states_t, drivers_t: jax.vmap(self._local_linearization_state_input)(
                    states_t, drivers_t
                )
            )(states[:-1], local_drivers[1:])

            A0, b0 = jax.vmap(self._local_linearization_state_input)(
                initial_packed, local_drivers[0]
            )

            A = jnp.concatenate([A0[None, ...], A_rest], axis=0)
            b = jnp.concatenate([b0[None, ...], b_rest], axis=0)
            return A, b


        final_states, _ = block_deer_rollout_from_linearizer(
            build_linearization=build_linearization,
            states_guess=states_guess,
            num_iters=self.deer_num_iters,
            mix=self.deer_damping,
        )

        _, x_states = jax.vmap(self._unpack_state_real)(final_states)
        return x_states

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def __call__(self, input_sequence):
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        A_diag, dt = self._project_A_dt(self.A_diag, self.dt)
        _, G_base = self._project_G(self.G_diag, A_diag, dt)  # kept for constant baseline

        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)

        if self.damping_mode == "constant":
            G_seq = jnp.broadcast_to(G_base[None, :], (input_sequence.shape[0], self.state_dim))
            ys = self._recurrence(A_diag, G_seq, dt, Bu_elements)

        elif self.damping_mode == "input":
            A_diag_eff, G_seq = self._compute_G_seq_input(
                input_sequence, Bu_elements, A_diag, dt
            )
            ys = self._recurrence(A_diag_eff, G_seq, dt, Bu_elements)

        elif self.damping_mode == "state_input":
            if self.use_block_deer:
                ys = self._run_block_deer(input_sequence, A_diag, dt, B_complex)
            else:
                ys = self._recurrence_state_input(A_diag, dt, Bu_elements, input_sequence)

        else:
            raise NotImplementedError(
                f"damping_mode={self.damping_mode} not implemented in DampedIMEX1Layer."
            )

        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)
        return xs




class DampedIMEX2Layer(_AbstractLinOSSLayer):
    """
    Based on the characteristic recurrence
    z_k+1 = z_k + dt * (-Ax_k - Gz_k + Bu_k+1)
    x_k+1 = x_k + dt * (z_k+1)
    """
    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array
    state_dim: int

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        G_min: float, 
        G_max: float, 
        A_min: float, 
        A_max: float, 
        dt_std: float, 
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.state_dim = state_dim
        init_key, B_key, C_key, D_key, key = jr.split(key, 5)
        if initialization == "uniform":
            self.A_diag, self.G_diag, self.dt = self._uniform_init_AGdt(A_min, A_max, G_min, G_max, dt_std, init_key)
        elif initialization == "ring":
            self.A_diag, self.G_diag, self.dt = self._ring_init_AGdt(r_min, r_max, theta_min, theta_max, dt_std, init_key)
        self.B = simple_uniform_init(B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim))
        self.C = simple_uniform_init(C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim))
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))

    def _is_valid_AGdt(self, A_diag, G_diag, dt):
        """Boolean check if (A,G,dt) in valid region"""
        dt = nn.sigmoid(dt)
        return (G_diag >= 0) & (((G_diag + dt*A_diag)**2 - 4*A_diag) < 0)

    def _ring_init_AGdt(self, r_min, r_max, theta_min, theta_max, dt_std, key):
        # Solve symbolically
        a, g, dt, lam1, lam2 = sp.symbols('a g dt lam1 lam2')

        # Characteristic recurrence for 1 decoupled 2x2 system
        M_i = sp.Matrix([[1-dt*g, -a*dt], [dt*(1-dt*g), 1 - dt**2*a]])
        # Eigenvalue pair expressions
        eigs = list(M_i.eigenvals().keys())
        eqs = [sp.Eq(eigs[0], lam1), sp.Eq(eigs[1], lam2)]
        sol = sp.solve(eqs, (a, g))[0]
        f = sp.lambdify((lam1, lam2, dt), sol, "numpy")

        # Sample timesteps
        mag_key, arg_key, dt_key = jr.split(key, 3)
        dt_vals = normal(stddev=dt_std)(dt_key, (self.state_dim,))
        dt_sigmoid = nn.sigmoid(dt_vals)

        # Sample eigenvalues in ring 
        mag = jnp.sqrt(jr.uniform(mag_key, shape=(self.state_dim,)) * (r_max**2 - r_min**2) + r_min**2)
        arg = jr.uniform(arg_key, shape=(self.state_dim,)) * (theta_max - theta_min) + theta_min
        lam1_vals = mag * jnp.cos(arg) + 1j * mag * jnp.sin(arg)
        lam2_vals = mag * jnp.cos(arg) - 1j * mag * jnp.sin(arg)

        # Convert to (A, G) representation
        a_vals, g_vals = f(lam1_vals, lam2_vals, dt_sigmoid)

        # Invertibility, stability, and validity checks
        h1 = sp.lambdify((a, g, dt), eigs[0], "numpy")
        h2 = sp.lambdify((a, g, dt), eigs[1], "numpy")
        lam1_out_vals = h1(a_vals, g_vals, dt_sigmoid)
        lam2_out_vals = h2(a_vals, g_vals, dt_sigmoid)
        invertible = jnp.all(jnp.isclose(lam1_out_vals, lam1_vals) | jnp.isclose(jnp.conjugate(lam1_out_vals), lam1_vals)) \
                   & jnp.all(jnp.isclose(lam2_out_vals, lam2_vals) | jnp.isclose(jnp.conjugate(lam2_out_vals), lam2_vals))
        stable = jnp.all(jnp.abs(lam1_out_vals) < 1.0) & jnp.all(jnp.abs(lam2_out_vals) < 1.0)
        valid = jnp.all(self._is_valid_AGdt(a_vals, g_vals, dt_vals))
        print(f"Invertibility check: {invertible}")
        print(f"Stability check: {stable}")
        print(f"Validity check: {valid}")

        # Cast to real (imag part is nonzero, ~machine precision)
        a_vals = a_vals.real
        g_vals = g_vals.real

        return a_vals, g_vals, dt_vals

    def _uniform_init_AGdt(self, A_min, A_max, G_min, G_max, dt_std, key):
        """Uniform sampling over valid (A,G,dt) region"""
        bsz = 512
        done = False 
        A_vals = []
        G_vals = []
        dt_vals = []

        while not done:
            A_key, G_key, dt_key, key = jr.split(key, 4)
            A_diag = jr.uniform(A_key, shape=(bsz,)) * (A_max - A_min) + A_min
            G_diag = jr.uniform(G_key, shape=(bsz,)) * (G_max - G_min) + G_min
            dt = normal(stddev=dt_std)(dt_key, (bsz,))

            mask = self._is_valid_AGdt(A_diag, G_diag, dt)
            A_vals.extend(list(A_diag[mask]))
            G_vals.extend(list(G_diag[mask]))
            dt_vals.extend(list(dt[mask]))

            if len(A_vals) >= self.state_dim and len(G_vals) >= self.state_dim and len(dt_vals) >= self.state_dim:
                done = True

        A_diag = jnp.array(A_vals[:self.state_dim])
        G_diag = jnp.array(G_vals[:self.state_dim])
        dt = jnp.array(dt_vals[:self.state_dim])

        return A_diag, G_diag, dt
    
    def _soft_project_AGdt(self, A_diag, G_diag, dt):
        """soft projection to the _is_valid_AGdt region"""
        dt = nn.sigmoid(dt)

        G_diag = nn.relu(G_diag)

        A_low = (2 - dt * G_diag - 2 * jnp.sqrt(1 - dt * G_diag)) / jnp.maximum(dt**2, 1e-6)
        A_high = (2 - dt * G_diag + 2 * jnp.sqrt(1 - dt * G_diag)) / jnp.maximum(dt**2, 1e-6)
        A_diag = A_low + nn.relu(A_diag - A_low) - nn.relu(A_diag - A_high)
        
        return A_diag, G_diag, dt

    def _recurrence(self, A_diag, G_diag, dt, Bu_elements):
        """Compute the LxP output of Damped-LinOSS given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            G_diag          (float32):    diagonal damping matrix   (P,)
            dt              (float32):    discretization time-step  (P,)
            Bu_elements     (complex64):  B @ u                     (L, P)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        sql = Bu_elements.shape[0]

        I = jnp.ones_like(A_diag)
        M_11 = I - dt * G_diag
        M_12 = -dt * A_diag
        M_21 = dt * (I - dt * G_diag)
        M_22 = I - dt**2 * A_diag

        M = jnp.concatenate([M_11, M_12, M_21, M_22])
        M_elements = M * jnp.ones((sql, 4 * self.state_dim))

        F1 = dt * Bu_elements
        F2 = dt**2 * Bu_elements
        F = jnp.hstack((F1, F2))

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, self.state_dim:]  # Position component

        return ys

    def __call__(self, input_sequence):
        # Materialize parameters
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        # Project
        A_diag, G_diag, dt = self._soft_project_AGdt(self.A_diag, self.G_diag, self.dt)

        # Apply SSM
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)
        ys = self._recurrence(A_diag, G_diag, dt, Bu_elements)
        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)

        return xs
    

class DampedIMLayer(_AbstractLinOSSLayer):
    """
    Based on the characteristic recurrence
    z_k+1 = z_k + dt * (-Ax_k+1 - Gz_k+1 + Bu_k+1)
    x_k+1 = x_k + dt * (z_k+1)
    """
    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array
    state_dim: int

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        G_min: float, 
        G_max: float, 
        A_min: float, 
        A_max: float, 
        dt_std: float, 
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.state_dim = state_dim
        init_key, B_key, C_key, D_key, key = jr.split(key, 5)
        if initialization == "uniform":
            self.A_diag, self.G_diag, self.dt = self._uniform_init_AGdt(A_min, A_max, G_min, G_max, dt_std, init_key)
        elif initialization == "ring":
            self.A_diag, self.G_diag, self.dt = self._ring_init_AGdt(r_min, r_max, theta_min, theta_max, dt_std, init_key)
        self.B = simple_uniform_init(B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim))
        self.C = simple_uniform_init(C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim))
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))

    def _is_valid_AGdt(self, A_diag, G_diag, dt):
        """Boolean check if (A,G,dt) in valid region"""
        dt = nn.sigmoid(dt)
        return (G_diag + dt*A_diag >= 0) & ((G_diag**2 - 4*A_diag) < 0)

    def _ring_init_AGdt(self, r_min, r_max, theta_min, theta_max, dt_std, key):
        # Solve symbolically
        a, g, dt, lam1, lam2 = sp.symbols('a g dt lam1 lam2')

        # Characteristic recurrence for 1 decoupled 2x2 system
        M_i = sp.Matrix([[1/(1 + dt*g + dt**2*a), -a*dt/(1 + dt*g + dt**2*a)], [dt/(1 + dt*g + dt**2*a), (1 + dt*g)/(1 + dt*g + dt**2*a)]])
        # Eigenvalue pair expressions
        eigs = list(M_i.eigenvals().keys())
        eqs = [sp.Eq(eigs[0], lam1), sp.Eq(eigs[1], lam2)]
        sol = sp.solve(eqs, (a, g))[0]
        f = sp.lambdify((lam1, lam2, dt), sol, "numpy")

        # Sample timesteps
        mag_key, arg_key, dt_key = jr.split(key, 3)
        dt_vals = normal(stddev=dt_std)(dt_key, (self.state_dim,))
        dt_sigmoid = nn.sigmoid(dt_vals)

        # Sample eigenvalues in ring 
        mag = jnp.sqrt(jr.uniform(mag_key, shape=(self.state_dim,)) * (r_max**2 - r_min**2) + r_min**2)
        arg = jr.uniform(arg_key, shape=(self.state_dim,)) * (theta_max - theta_min) + theta_min
        lam1_vals = mag * jnp.cos(arg) + 1j * mag * jnp.sin(arg)
        lam2_vals = mag * jnp.cos(arg) - 1j * mag * jnp.sin(arg)

        # Convert to (A, G) representation
        a_vals, g_vals = f(lam1_vals, lam2_vals, dt_sigmoid)

        # Invertibility, stability, and validity checks
        h1 = sp.lambdify((a, g, dt), eigs[0], "numpy")
        h2 = sp.lambdify((a, g, dt), eigs[1], "numpy")
        lam1_out_vals = h1(a_vals, g_vals, dt_sigmoid)
        lam2_out_vals = h2(a_vals, g_vals, dt_sigmoid)
        invertible = jnp.all(jnp.isclose(lam1_out_vals, lam1_vals) | jnp.isclose(jnp.conjugate(lam1_out_vals), lam1_vals)) \
                   & jnp.all(jnp.isclose(lam2_out_vals, lam2_vals) | jnp.isclose(jnp.conjugate(lam2_out_vals), lam2_vals))
        stable = jnp.all(jnp.abs(lam1_out_vals) < 1.0) & jnp.all(jnp.abs(lam2_out_vals) < 1.0)
        valid = jnp.all(self._is_valid_AGdt(a_vals, g_vals, dt_vals))
        print(f"Invertibility check: {invertible}")
        print(f"Stability check: {stable}")
        print(f"Validity check: {valid}")

        # Cast to real (imag part is nonzero, ~machine precision)
        a_vals = a_vals.real
        g_vals = g_vals.real

        return a_vals, g_vals, dt_vals

    def _uniform_init_AGdt(self, A_min, A_max, G_min, G_max, dt_std, key):
        """Uniform sampling over valid (A,G,dt) region"""
        bsz = 512
        done = False 
        A_vals = []
        G_vals = []
        dt_vals = []

        while not done:
            A_key, G_key, dt_key, key = jr.split(key, 4)
            A_diag = jr.uniform(A_key, shape=(bsz,)) * (A_max - A_min) + A_min
            G_diag = jr.uniform(G_key, shape=(bsz,)) * (G_max - G_min) + G_min
            dt = normal(stddev=dt_std)(dt_key, (bsz,))

            mask = self._is_valid_AGdt(A_diag, G_diag, dt)
            A_vals.extend(list(A_diag[mask]))
            G_vals.extend(list(G_diag[mask]))
            dt_vals.extend(list(dt[mask]))

            if len(A_vals) >= self.state_dim and len(G_vals) >= self.state_dim and len(dt_vals) >= self.state_dim:
                done = True

        A_diag = jnp.array(A_vals[:self.state_dim])
        G_diag = jnp.array(G_vals[:self.state_dim])
        dt = jnp.array(dt_vals[:self.state_dim])

        return A_diag, G_diag, dt
    
    def _soft_project_AGdt(self, A_diag, G_diag, dt):
        """soft projection to the _is_valid_AGdt region"""
        dt = nn.sigmoid(dt)

        G_low = -dt * A_diag
        G_diag = G_low + nn.relu(G_diag - G_low)

        A_low = 1/4*G_diag**2
        A_diag = A_low + nn.relu(A_diag - A_low)

        return A_diag, G_diag, dt

    def _recurrence(self, A_diag, G_diag, dt, Bu_elements):
        """Compute the LxP output of Damped-LinOSS given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            G_diag          (float32):    diagonal damping matrix   (P,)
            dt              (float32):    discretization time-step  (P,)
            Bu_elements     (complex64):  B @ u                     (L, P)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        sql = Bu_elements.shape[0]

        I = jnp.ones_like(A_diag)
        S = I + dt * G_diag + dt**2 * A_diag
        M_11 = 1 / S
        M_12 = -dt * A_diag / S
        M_21 = dt / S
        M_22 = (I + dt * G_diag) / S

        M = jnp.concatenate([M_11, M_12, M_21, M_22])
        M_elements = M * jnp.ones((sql, 4 * self.state_dim))

        F1 = dt * (1.0 / S) * Bu_elements
        F2 = dt**2 * (1.0 / S) * Bu_elements
        F = jnp.hstack((F1, F2))

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, self.state_dim:]  # Position component

        return ys

    def __call__(self, input_sequence):
        # Materialize parameters
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        # Project
        A_diag, G_diag, dt = self._soft_project_AGdt(self.A_diag, self.G_diag, self.dt)

        # Apply SSM
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)
        ys = self._recurrence(A_diag, G_diag, dt, Bu_elements)
        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)

        return xs
    

class DampedEXLayer(_AbstractLinOSSLayer):
    """
    Based on the characteristic recurrence
    z_k+1 = z_k + dt * (-Ax_k - Gz_k + Bu_k+1)
    x_k+1 = x_k + dt * (z_k)
    """
    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array
    state_dim: int

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        G_min: float, 
        G_max: float, 
        A_min: float, 
        A_max: float, 
        dt_std: float, 
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.state_dim = state_dim
        init_key, B_key, C_key, D_key, key = jr.split(key, 5)
        if initialization == "uniform":
            self.A_diag, self.G_diag, self.dt = self._uniform_init_AGdt(A_min, A_max, G_min, G_max, dt_std, init_key)
        elif initialization == "ring":
            self.A_diag, self.G_diag, self.dt = self._ring_init_AGdt(r_min, r_max, theta_min, theta_max, dt_std, init_key)
        self.B = simple_uniform_init(B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim))
        self.C = simple_uniform_init(C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim))
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))

    def _is_valid_AGdt(self, A_diag, G_diag, dt):
        """Boolean check if (A,G,dt) in valid region"""
        dt = nn.sigmoid(dt)
        return (G_diag - dt*A_diag >= 0) & ((G_diag**2 - 4*A_diag) < 0)

    def _ring_init_AGdt(self, r_min, r_max, theta_min, theta_max, dt_std, key):
        # Solve symbolically
        a, g, dt, lam1, lam2 = sp.symbols('a g dt lam1 lam2')

        # Characteristic recurrence for 1 decoupled 2x2 system
        M_i = sp.Matrix([[1-dt*g, -dt*a], [dt, 1]])
        # Eigenvalue pair expressions
        eigs = list(M_i.eigenvals().keys())
        eqs = [sp.Eq(eigs[0], lam1), sp.Eq(eigs[1], lam2)]
        sol = sp.solve(eqs, (a, g))[0]
        f = sp.lambdify((lam1, lam2, dt), sol, "numpy")

        # Sample timesteps
        mag_key, arg_key, dt_key = jr.split(key, 3)
        dt_vals = normal(stddev=dt_std)(dt_key, (self.state_dim,))
        dt_sigmoid = nn.sigmoid(dt_vals)

        # Sample eigenvalues in ring 
        mag = jnp.sqrt(jr.uniform(mag_key, shape=(self.state_dim,)) * (r_max**2 - r_min**2) + r_min**2)
        arg = jr.uniform(arg_key, shape=(self.state_dim,)) * (theta_max - theta_min) + theta_min
        lam1_vals = mag * jnp.cos(arg) + 1j * mag * jnp.sin(arg)
        lam2_vals = mag * jnp.cos(arg) - 1j * mag * jnp.sin(arg)

        # Convert to (A, G) representation
        a_vals, g_vals = f(lam1_vals, lam2_vals, dt_sigmoid)

        # Invertibility, stability, and validity checks
        h1 = sp.lambdify((a, g, dt), eigs[0], "numpy")
        h2 = sp.lambdify((a, g, dt), eigs[1], "numpy")
        lam1_out_vals = h1(a_vals, g_vals, dt_sigmoid)
        lam2_out_vals = h2(a_vals, g_vals, dt_sigmoid)
        invertible = jnp.all(jnp.isclose(lam1_out_vals, lam1_vals) | jnp.isclose(jnp.conjugate(lam1_out_vals), lam1_vals)) \
                   & jnp.all(jnp.isclose(lam2_out_vals, lam2_vals) | jnp.isclose(jnp.conjugate(lam2_out_vals), lam2_vals))
        stable = jnp.all(jnp.abs(lam1_out_vals) < 1.0) & jnp.all(jnp.abs(lam2_out_vals) < 1.0)
        valid = jnp.all(self._is_valid_AGdt(a_vals, g_vals, dt_vals))
        print(f"Invertibility check: {invertible}")
        print(f"Stability check: {stable}")
        print(f"Validity check: {valid}")

        # Cast to real (imag part is nonzero, ~machine precision)
        a_vals = a_vals.real
        g_vals = g_vals.real

        return a_vals, g_vals, dt_vals

    def _uniform_init_AGdt(self, A_min, A_max, G_min, G_max, dt_std, key):
        """Uniform sampling over valid (A,G,dt) region"""
        bsz = 512
        done = False 
        A_vals = []
        G_vals = []
        dt_vals = []

        while not done:
            A_key, G_key, dt_key, key = jr.split(key, 4)
            A_diag = jr.uniform(A_key, shape=(bsz,)) * (A_max - A_min) + A_min
            G_diag = jr.uniform(G_key, shape=(bsz,)) * (G_max - G_min) + G_min
            dt = normal(stddev=dt_std)(dt_key, (bsz,))

            mask = self._is_valid_AGdt(A_diag, G_diag, dt)
            A_vals.extend(list(A_diag[mask]))
            G_vals.extend(list(G_diag[mask]))
            dt_vals.extend(list(dt[mask]))

            if len(A_vals) >= self.state_dim and len(G_vals) >= self.state_dim and len(dt_vals) >= self.state_dim:
                done = True

        A_diag = jnp.array(A_vals[:self.state_dim])
        G_diag = jnp.array(G_vals[:self.state_dim])
        dt = jnp.array(dt_vals[:self.state_dim])

        return A_diag, G_diag, dt
    
    def _soft_project_AGdt(self, A_diag, G_diag, dt):
        """soft projection to the _is_valid_AGdt region"""
        dt = nn.sigmoid(dt)

        G_low = dt * A_diag
        G_diag = G_low + nn.relu(G_diag - G_low)

        A_low = 1/4*G_diag**2
        A_diag = A_low + nn.relu(A_diag - A_low)

        return A_diag, G_diag, dt

    def _recurrence(self, A_diag, G_diag, dt, Bu_elements):
        """Compute the LxP output of Damped-LinOSS given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            G_diag          (float32):    diagonal damping matrix   (P,)
            dt              (float32):    discretization time-step  (P,)
            Bu_elements     (complex64):  B @ u                     (L, P)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        sql = Bu_elements.shape[0]

        I = jnp.ones_like(A_diag)
        M_11 = I - dt * G_diag
        M_12 = -dt * A_diag
        M_21 = dt
        M_22 = I

        M = jnp.concatenate([M_11, M_12, M_21, M_22])
        M_elements = M * jnp.ones((sql, 4 * self.state_dim))

        F1 = dt * Bu_elements
        F2 = jnp.zeros_like(F1)
        F = jnp.hstack((F1, F2))

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, self.state_dim:]  # Position component

        return ys

    def __call__(self, input_sequence):
        # Materialize parameters
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        # Project
        A_diag, G_diag, dt = self._soft_project_AGdt(self.A_diag, self.G_diag, self.dt)

        # Apply SSM
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)
        ys = self._recurrence(A_diag, G_diag, dt, Bu_elements)
        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)

        return xs
    

class LinOSSBlock(eqx.Module):
    norm: eqx.nn.BatchNorm
    layer: _AbstractLinOSSLayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        layer_name: str,
        state_dim: int,
        hidden_dim: int,
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        A_min: float, 
        A_max: float, 
        G_min: float, 
        G_max: float, 
        dt_std: float, 
        drop_rate: float,
        key: PRNGKeyArray,
        **kwargs,
    ):
        ssmkey, glukey = jr.split(key, 2)
        layer_map = {
            "IM": IMLayer,
            "IMEX": IMEXLayer,
            "DampedIMEX1": DampedIMEX1Layer,
            "DampedIMEX2": DampedIMEX2Layer,
            "DampedIM": DampedIMLayer,
            "DampedEX": DampedEXLayer,
        }
        if layer_name not in layer_map.keys():
            raise KeyError(f"Layer name {layer_name} not defined.")

        self.norm = eqx.nn.BatchNorm(
            input_size=hidden_dim, axis_name="batch", channelwise_affine=False, mode="batch"
        )
        self.layer = layer_map[layer_name](
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            initialization=initialization,
            r_min=r_min,
            r_max=r_max,
            theta_min=theta_min,
            theta_max=theta_max,
            A_min=A_min,
            A_max=A_max,
            G_min=G_min,
            G_max=G_max,
            dt_std=dt_std,
            damping_mode=kwargs.get("damping_mode", "constant"),

            gate_variant=kwargs.get("gate_variant", "simple"),

            gate_type=kwargs.get("gate_type", "linear"),
            mult_min=kwargs.get("mult_min", 0.25),
            mult_max=kwargs.get("mult_max", 4.0),
            freq_aware_damping=kwargs.get("freq_aware_damping", False),
            zeta_min=kwargs.get("zeta_min", 0.0),
            zeta_max=kwargs.get("zeta_max", 4.0),

            use_block_deer=kwargs.get("use_block_deer", False),
            deer_num_iters=kwargs.get("deer_num_iters", 4),
            deer_damping=kwargs.get("deer_damping", 0.0),
            key=ssmkey,
        )
        self.glu = GLU(hidden_dim, hidden_dim, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.layer(x)
        x = jax.nn.gelu(x)
        x = self.drop(x, key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x
        return x, state


class LinOSS(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: list[LinOSSBlock]
    linear_decoder: eqx.nn.Linear
    classification: bool
    tanh_output: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True

    def __init__(
        self,
        layer_name: str,
        input_dim: int,
        state_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int,
        classification: bool,
        tanh_output: bool,
        output_step: int,
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        A_min: float, 
        A_max: float, 
        G_min: float, 
        G_max: float, 
        dt_std: float, 
        drop_rate: float,
        gate_variant: str = "simple",
        damping_mode: str = "constant",
        gate_type: str = "linear",
        mult_min: float = 0.25,
        mult_max: float = 4.0,
        freq_aware_damping: bool = False,
        zeta_min: float = 0.0,
        zeta_max: float = 4.0,
        use_block_deer: bool = False,
        deer_num_iters: int = 4,
        deer_damping: float = 0.0,
        key: PRNGKeyArray=None,
        **kwargs,
    ):
        linear_encoder_key, *block_keys, linear_decoder_key = jr.split(
            key, num_blocks + 2
        )
        self.linear_encoder = eqx.nn.Linear(input_dim, hidden_dim, key=linear_encoder_key)
        self.blocks = [
            LinOSSBlock(
                layer_name=layer_name,
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                initialization=initialization,
                r_min=r_min,
                r_max=r_max,
                theta_min=theta_min,
                theta_max=theta_max,
                A_min=A_min, 
                A_max=A_max, 
                G_min=G_min, 
                G_max=G_max, 
                dt_std=dt_std, 
                drop_rate=drop_rate,
                gate_variant=gate_variant,
                damping_mode=damping_mode,
                gate_type=gate_type,
                mult_min=mult_min,
                mult_max=mult_max,
                freq_aware_damping=freq_aware_damping,
                zeta_min=zeta_min,
                zeta_max=zeta_max,
                use_block_deer=use_block_deer,
                deer_num_iters=deer_num_iters,
                deer_damping=deer_damping,
                key=key,
            )
            for key in block_keys
        ]
        self.linear_decoder = eqx.nn.Linear(hidden_dim, output_dim, key=linear_decoder_key)

        self.classification = classification
        self.tanh_output = tanh_output
        self.output_step = output_step

    def __call__(self, x, state, key):
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)

        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)

        if self.classification:
            x = jnp.mean(x, axis=0)
            x = self.linear_decoder(x)
            x = jax.nn.softmax(x, axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.vmap(self.linear_decoder)(x)
            if self.tanh_output:
                x = jax.nn.tanh(x)

        return x, state
    
