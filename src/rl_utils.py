from collections import namedtuple
from functools import partial
import jax
import jax.numpy as jnp

from src.types import Env, Trajectory


############################################################################
################ Useful Utilities for Return Computation ###################
############################################################################

def return_episode(rewards_episode: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """
    Compute the return of an episode.
    inputs:
        rewards_episode (jnp.ndarray): rewards of the episode
        gamma (float): discount factor
    outputs:
        G (float): return of the episode

    Example:
        rewards_episode = [1, 1, 1]
        gamma = 0.9
        return_episode(rewards_episode, gamma) = 2.71
    """
    G = jnp.array(0.)
    for t in range(len(rewards_episode)):
        G += gamma**t * rewards_episode[t]
    return G

# Batched version of return_episode
return_episode_vmap = jax.vmap(return_episode, in_axes=(0, None), out_axes=0)


######################################################################
################ Useful Utilities for Sampling from Env ##############
######################################################################

env = Env()

@partial(jax.jit, static_argnames=("policy", "steps_in_episode"))
def rollout(params_Q, policy, params_true, steps_in_episode, key, temp) -> Trajectory:
    """
    Rollout a jitted episode with lax.scan.

    inputs:
        params_Q (pytree): parameters of the Q function
        policy (function): policy function
        params_true (pytree): true parameters of the environment
        steps_in_episode (int): number of steps in the episode
        key (jax.random.PRNGKey): random key

    outputs:
        trajectory (Trajectory): trajectory of the episode


    """
    # Reset the environment
    state = env.reset_env(params_true)

    Carry = namedtuple('Carry', ['state', 'params_Q', 'temp', 'key', 'cum_reward', 'valid_mask'])

    def policy_step(carry: Carry, tmp):
        """lax.scan compatible step transition in jax env."""
        state, params_Q, temp, key, cum_reward, valid_mask = carry
        key, key = jax.random.split(key)

        action = policy(params_Q=params_Q, state=state, key=key, temp=temp)
        s_next, reward = env.transition(params_true, state, action, True)
        done = env.is_done(params_true, s_next)

        new_cum_reward = cum_reward + reward * valid_mask
        new_valid_mask = valid_mask * (1 - done)
        carry = Carry(s_next, params_Q, temp, key, new_cum_reward, new_valid_mask)  # This is the carry for the next step
        out = [state, action, reward, s_next, done, new_valid_mask]  # This is the output of the scan step
        return carry, out

    # Scan over episode step loop
    carry_out, out = jax.lax.scan(
      policy_step,
      init=Carry(state, params_Q, temp, key, jnp.array([0.]), jnp.array([1.])),
      xs=None,
      length=steps_in_episode
    )
    
    # Return masked sum of rewards accumulated by agent in episode
    states, actions, rewards, next_states, dones, valid_masks = out
    cum_return = carry_out.cum_reward
    
    return Trajectory(states, actions, rewards, next_states, dones, valid_masks, cum_return)


@partial(jax.jit, static_argnames=("policy", "steps_in_episode"))
def batch_rollout(params_Q, policy, params_true, steps_in_episode, rng_batch, temp) -> Trajectory:
    """Evaluate over a batch of rng keys."""
    # vmap over different rng keys
    batch_rollout = jax.vmap(rollout, in_axes=(None, None, None, None, 0, None, None))
    return batch_rollout(params_Q, policy, params_true, steps_in_episode, rng_batch, temp)
