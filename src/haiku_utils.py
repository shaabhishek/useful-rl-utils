from functools import partial
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from src.types import Model


def make_mlp(hidden_dims, output_dim):
    """
    Make a MLP with ReLU activations.

    inputs:
        hidden_dims (list): list of hidden dimensions
        output_dim (int): output dimension

    outputs:
        mlp (haiku.Sequential): MLP with ReLU activations
    """
    layers = []
    
    for hidden_dim in hidden_dims:
        layers.append(hk.Linear(hidden_dim))
        layers.append(jax.nn.relu)
    
    layers.append(hk.Linear(output_dim))
    return hk.Sequential(layers)

def Double_Q_Net(dims, x):
    """
    Create two MLPs for the Double Q-learning algorithm.

    inputs:
        dims (tuple): tuple of (n_actions, hidden_dims)
    """
    n_actions, hidden_dims = dims
    mlp1 = make_mlp(hidden_dims, n_actions)
    mlp2 = make_mlp(hidden_dims, n_actions)
    return mlp1(x), mlp2(x)


def init_Q_net(dims: tuple, lr: float, max_grad_norm: float, decay: bool, total_steps: int):
    """
    Initialize the Q network.
    """
    net = hk.without_apply_rng(hk.transform(partial(Double_Q_Net, dims)))

    if decay:
        schedule = optax.warmup_cosine_decay_schedule(init_value=lr, peak_value=lr, warmup_steps=0, decay_steps=total_steps, end_value=0.0)
        optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adamw(schedule))
    else:
        optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adamw(lr))

    return Model(net, optimizer)


