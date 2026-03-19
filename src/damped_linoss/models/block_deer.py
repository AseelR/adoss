import jax
import jax.numpy as jnp


def block_operator(q_i, q_j):
    """
    Compose per-mode affine maps with block matrices.

    Inputs at one scan element:
      A_i: (P, B, B)
      b_i: (P, B)
      A_j: (P, B, B)
      b_j: (P, B)

    Returns:
      A_new: (P, B, B)
      b_new: (P, B)
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    # A_new = jnp.einsum("pab,pbc->pac", A_j, A_i)
    # b_new = jnp.einsum("pab,pb->pa", A_j, b_i) + b_j

    A_new = jnp.einsum("...pab,...pbc->...pac", A_j, A_i)
    b_new = jnp.einsum("...pab,...pb->...pa", A_j, b_i) + b_j
    return A_new, b_new


def linearized_rollout(local_step_fn, initial_state, local_drivers, states_guess, damping=0.0):
    """
    One DEER linearized rollout for a mode-local nonlinear recurrence.

    Args:
      local_step_fn:
        function mapping
          local_state_t:  (B,)
          local_driver_t: (D,)
        -> local_state_{t+1}: (B,)

      initial_state: (P, B)
        fixed starting state before the first driver

      local_drivers: (T, P, D)
        per-time, per-mode local driver vectors

      states_guess: (T, P, B)
        current trajectory estimate for states 1..T

      damping: float
        Jacobian damping factor in [0,1).
        A <- (1 - damping) * J

    Returns:
      new_states: (T, P, B)
    """
    T = states_guess.shape[0]
    P = states_guess.shape[1]
    B = states_guess.shape[2]

    if T == 1:
        b0 = jax.vmap(local_step_fn)(initial_state, local_drivers[0])   # (P, B)
        return b0[None, ...]

    # f(x_t, d_{t+1}) for t = 1..T-1
    fs = jax.vmap(
        lambda states_t, drivers_t: jax.vmap(local_step_fn)(states_t, drivers_t)
    )(states_guess[:-1], local_drivers[1:])   # (T-1, P, B)

    # local Jacobian blocks directly: (T-1, P, B, B)
    J_blocks = jax.vmap(
        lambda states_t, drivers_t: jax.vmap(jax.jacrev(local_step_fn, argnums=0))(states_t, drivers_t)
    )(states_guess[:-1], local_drivers[1:])

    A_rest = (1.0 - damping) * J_blocks
    b_rest = fs - jnp.einsum("tpab,tpb->tpa", A_rest, states_guess[:-1])

    b0 = jax.vmap(local_step_fn)(initial_state, local_drivers[0])      # (P, B)
    A0 = jnp.zeros((P, B, B), dtype=states_guess.dtype)

    A = jnp.concatenate([A0[None, ...], A_rest], axis=0)   # (T, P, B, B)
    b = jnp.concatenate([b0[None, ...], b_rest], axis=0)   # (T, P, B)

    _, new_states = jax.lax.associative_scan(block_operator, (A, b))
    return new_states


def block_deer_rollout(local_step_fn, initial_state, local_drivers, states_guess, num_iters=4, damping=0.0):
    """
    Run several DEER iterations.

    Args:
      local_step_fn:
        local packed-state transition map
      initial_state: (P, B)
      local_drivers: (T, P, D)
      states_guess: (T, P, B)
      num_iters: number of DEER iterations
      damping: Jacobian damping factor

    Returns:
      final_states: (T, P, B)
      states_trace: (num_iters, T, P, B)
    """
    def body_fn(states, _):
        new_states = linearized_rollout(
            local_step_fn=local_step_fn,
            initial_state=initial_state,
            local_drivers=local_drivers,
            states_guess=states,
            damping=damping,
        )
        return new_states, new_states

    final_states, states_trace = jax.lax.scan(body_fn, states_guess, None, length=num_iters)
    return final_states, states_trace