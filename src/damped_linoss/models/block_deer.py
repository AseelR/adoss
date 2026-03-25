import jax
import jax.numpy as jnp


def block_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j

    A_new = jnp.einsum("...pab,...pbc->...pac", A_j, A_i)
    b_new = jnp.einsum("...pab,...pb->...pa", A_j, b_i) + b_j
    return A_new, b_new


def linearized_rollout_from_blocks(A, b):
    """
    Args:
      A: (T, P, B, B)
      b: (T, P, B)

    Returns:
      new_states: (T, P, B)
    """
    _, new_states = jax.lax.associative_scan(block_operator, (A, b))
    return new_states


def block_deer_rollout_from_linearizer(
    build_linearization,
    states_guess,
    num_iters=4,
):
    """
    Repeated DEER updates using user-supplied affine linearizations.

    build_linearization(states_guess) should return:
      A: (T, P, B, B)
      b: (T, P, B)
    """
    def body_fn(states, _):
        A, b = build_linearization(states)
        new_states = linearized_rollout_from_blocks(A, b)
        return new_states, new_states

    final_states, states_trace = jax.lax.scan(body_fn, states_guess, None, length=num_iters)
    return final_states, states_trace


    