import jax
import jax.numpy as jnp


def list_of_pytree_to_pytree_of_list(list_of_pytrees):
    """
    Convert a list of pytrees to a pytree of lists.
    Example:
        input: [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        output: {'a': [1, 3], 'b': [2, 4]}
    """
    return jax.tree_map(lambda *x: jnp.stack(x), *list_of_pytrees)

@jax.jit
def soft_update_params(tau, params, target_params):
    """
    Soft update of the target network parameters.
    """
    return jax.tree_map(lambda p, tp: tau * p + (1 - tau) * tp, params, target_params)


def get_shapes(t):
    """
    Get the shapes of a pytree.
    """
    return jax.tree_map(lambda x: x.shape, t)


def concatenate_leaves(pytrees):
    """
    Concatenate the leaves of all pytrees in a list of pytrees.

    Example (simple):
        input: [{'a': np.array([1, 2]), 'b': np.array([3, 4])}, {'a': np.array([5, 6]), 'b': np.array([7, 8])}]
        output: {'a': np.array([1, 2, 5, 6]), 'b': np.array([3, 4, 7, 8])}

    Example (batching over the first dimension):
        input: [
            [array(shape=(1, 2)), array(shape=(1, 2))],
            [array(shape=(1, 2)), array(shape=(1, 2))],
            [array(shape=(1, 2)), array(shape=(1, 2))],
        ]
        output: [array(shape=(3, 2)), array(shape=(3, 2))]

    
    """
    return jax.tree_map(lambda *leaves: jnp.concatenate(leaves, axis=0), *pytrees)