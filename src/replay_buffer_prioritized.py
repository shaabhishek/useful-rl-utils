"""
Author: Abhishek Sharma

Based on code from:
https://github.com/coax-dev/coax/tree/main/coax/experience_replay

"""
import jax
import jax.numpy as jnp
import numpy as np

from src.types import *

class SegmentTree:
    r"""

    A `segment tree <https://en.wikipedia.org/wiki/Segment_tree>`_ data structure that allows
    for batched updating and batched partial-range (segment) reductions.

    Parameters
    ----------
    capacity : positive int

        Number of values to accommodate.

    reducer : function

        The reducer function: :code:`(float, float) -> float`.

    init_value : float

        The unit element relative to the reducer function. Some typical examples are: 0 if reducer
        is :data:`add <numpy.add>`, 1 for :data:`multiply <numpy.multiply>`, :math:`-\infty` for
        :data:`maximum <numpy.maximum>`, :math:`\infty` for :data:`minimum <numpy.minimum>`.

    Warning
    -------

    The :attr:`values` attribute and square-bracket lookups (:code:`tree[level, index]`) return
    references of the underlying storage array. Therefore, make sure that downstream code doesn't
    update these values in-place, which would corrupt the segment tree structure.

    """
    def __init__(self, capacity, reducer, init_value):
        self.capacity = capacity
        self.reducer = reducer
        self.init_value = float(init_value)
        self._height = int(np.ceil(np.log2(capacity))) + 1  # the +1 is for the values themselves
        self._arr = np.full(shape=(2 ** self.height - 1), fill_value=self.init_value)

    @property
    def height(self):
        r""" The height of the tree :math:`h\sim\log(\text{capacity})`. """
        return self._height

    @property
    def root_value(self):
        r"""

        The aggregated value, equivalent to
        :func:`reduce(reducer, values, init_value) <functools.reduce>`.

        """
        return self._arr[0]

    @property
    def values(self):
        r""" The values stored at the leaves of the tree. """
        start = 2 ** (self.height - 1) - 1
        stop = start + self.capacity
        return self._arr[start:stop]

    def __getitem__(self, lookup):
        if isinstance(lookup, int):
            level_offset, level_size = self._check_level_lookup(lookup)
            return self._arr[level_offset:(level_offset + level_size)]

        if isinstance(lookup, tuple) and len(lookup) == 1:
            level, = lookup
            return self[level]

        if isinstance(lookup, tuple) and len(lookup) == 2:
            level, index = lookup
            return self[level][index]

        raise IndexError(
            "tree lookup must be of the form: tree[level] or tree[level, index], "
            "where 'level' is an int and 'index' is a 1d array lookup")

    def set_values(self, idx, values):
        r"""

        Set or update the :attr:`values`.

        Parameters
        ----------
        idx : 1d array of ints

            The indices of the values to be updated. If you wish to update all values use ellipses
            instead, e.g. :code:`tree.set_values(..., values)`.

        values : 1d array of floats

            The new values.

        """
        idx, level_offset, level_size = self._check_idx(idx)

        # update leaf-node values
        self._arr[level_offset + (idx % level_size)] = values

        for level in range(self.height - 2, -1, -1):
            idx = np.unique(idx // 2)
            left_child = level_offset + 2 * idx
            right_child = left_child + 1

            level_offset = 2 ** level - 1
            parent = level_offset + idx
            self._arr[parent] = self.reducer(self._arr[left_child], self._arr[right_child])

    def partial_reduce(self, start=0, stop=None):
        r"""

        Reduce values over a partial range of indices. This is an efficient, batched implementation
        of :func:`reduce(reducer, values[state:stop], init_value) <functools.reduce>`.

        Parameters
        ----------
        start : int or array of ints

            The lower bound of the range (inclusive).

        stop : int or array of ints, optional

            The lower bound of the range (exclusive). If left unspecified, this defaults to
            :attr:`height`.

        Returns
        -------
        value : float

            The result of the partial reduction.

        """
        # NOTE: This is an iterative implementation, which is a lot uglier than a recursive one.
        # The reason why we use an iterative approach is that it's easier for batch-processing.

        # i and j are 1d arrays (indices for self._arr)
        i, j = self._check_start_stop_to_i_j(start, stop)

        # trivial case
        done = (i == j)
        if done.all():
            return self._arr[i]

        # left/right accumulators (mask one of them to avoid over-counting if i == j)
        a, b = self._arr[i], np.where(done, self.init_value, self._arr[j])

        # number of nodes in higher levels
        level_offset = 2 ** (self.height - 1) - 1

        # we start from the leaves and work up towards the root
        for level in range(self.height - 2, -1, -1):

            # get parent indices
            level_offset_parent = 2 ** level - 1
            i_parent = (i - level_offset) // 2 + level_offset_parent
            j_parent = (j - level_offset) // 2 + level_offset_parent

            # stop when we have a shared parent (possibly the root node, but not necessarily)
            done |= (i_parent == j_parent)
            if done.all():
                return self.reducer(a, b)

            # only accumulate right-child value if 'i' was a left child of 'i_parent'
            a = np.where((i % 2 == 1) & ~done, self.reducer(a, self._arr[i + 1]), a)

            # only accumulate left-child value if 'j' was a right child of 'j_parent'
            b = np.where((j % 2 == 0) & ~done, self.reducer(b, self._arr[j - 1]), b)

            # prepare for next loop
            i, j, level_offset = i_parent, j_parent, level_offset_parent

        assert False, 'this point should not be reached'

    def __repr__(self):
        s = ""
        for level in range(self.height):
            s += f"\n  level={level} : {repr(self[level])}"
        return f"{type(self).__name__}({s})"

    def _check_level_lookup(self, level):
        if not isinstance(level, int):
            raise IndexError(f"level lookup must be an int, got: {type(level)}")

        if not (-self.height <= level < self.height):
            raise IndexError(f"level index {level} is out of bounds; tree height: {self.height}")

        level %= self.height
        level_offset = 2 ** level - 1
        level_size = min(2 ** level, self.capacity)
        return level_offset, level_size

    def _check_level(self, level):
        if level < -self.height or level >= self.height:
            raise IndexError(f"tree level index {level} out of range; tree height: {self.height}")
        return level % self.height

    def _check_idx(self, idx):
        """ some boiler plate to turn any compatible idx into a 1d integer array """
        level_offset, level_size = self._check_level_lookup(self.height - 1)

        if isinstance(idx, int):
            idx = np.asarray([idx], dtype='int32')
        if idx is None or idx is Ellipsis:
            idx = np.arange(level_size, dtype='int32')
        elif isinstance(idx, list) and all(isinstance(x, int) for x in idx):
            idx = np.asarray(idx, dtype='int32')
        elif (isinstance(idx, np.ndarray) and np.issubdtype(idx.dtype, np.integer) and idx.ndim <= 1):
            idx = idx.reshape(-1)
        elif (isinstance(idx, jnp.ndarray) and jnp.issubdtype(idx.dtype, jnp.integer) and idx.ndim <= 1):
            idx = idx.reshape(-1)
        else:
            raise IndexError("idx must be an int or a 1d integer array")

        if not np.all((idx < level_size) & (idx >= -level_size)):
            raise IndexError("one of more entries in idx are out or range")

        return idx % level_size, level_offset, level_size

    def _check_start_stop_to_i_j(self, start, stop):
        """ some boiler plate to turn (start, stop) into left/right index arrays (i, j) """
        start_orig, stop_orig = start, stop

        # convert 'start' index to 1d array
        if isinstance(start, int):
            start = np.array([start])
        # TODO: Add support for jnp.ndarray
        if not (isinstance(start, np.ndarray)
                and start.ndim == 1
                and np.issubdtype(start.dtype, np.integer)):
            raise TypeError("'start' must be an int or a 1d integer array")

        # convert 'stop' index to 1d array
        if stop is None:
            stop = np.full_like(start, self.capacity)
        if isinstance(stop, int):
            stop = np.full_like(start, stop)
        if not (isinstance(stop, np.ndarray)
                and stop.ndim == 1
                and np.issubdtype(stop.dtype, np.integer)):
            raise TypeError("'stop' must be an int or a 1d integer array")

        # ensure that 'start' is the same size as 'stop'
        if start.size == 1 and stop.size > 1:
            start = np.full_like(stop, start[0])

        # check compatible shapes
        if start.shape != stop.shape:
            raise ValueError(
                f"shapes must be equal, got: start.shape: {start.shape}, stop.shape: {stop.shape}")

        # convert to (i, j), where j is the *inclusive* version of 'stop' (which is exclusive)
        level_offset, level_size = self._check_level_lookup(self.height - 1)
        i = level_offset + start % level_size
        j = level_offset + (stop - 1) % level_size

        # check consistency of ranges
        if not np.all((i >= level_offset) & (j < level_offset + level_size) & (i <= j)):
            raise IndexError(
                f"inconsistent ranges detected from (start, stop) = ({start_orig}, {stop_orig})")

        return i, j

class SumTree(SegmentTree):
    r"""

    A sum-tree data structure that allows for batched updating and batched weighted sampling.

    Both update and sampling operations have a time complexity of :math:`\mathcal{O}(\log N)` and a
    memory footprint of :math:`\mathcal{O}(N)`, where :math:`N` is the length of the underlying
    :attr:`values`.

    Parameters
    ----------
    capacity : positive int

        Number of values to accommodate.

    reducer : function

        The reducer function: :code:`(float, float) -> float`.

    init_value : float

        The unit element relative to the reducer function. Some typical examples are: 0 if
        reducer is :func:`operator.add`, 1 for :func:`operator.mul`, :math:`-\infty` for
        :func:`max`, :math:`\infty` for :func:`min`.

    """
    def __init__(self, capacity, random_seed=None):
        super().__init__(capacity=capacity, reducer=np.add, init_value=0)
        self.random_seed = random_seed

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, new_random_seed):
        self._rnd = np.random.RandomState(new_random_seed)
        self._random_seed = new_random_seed

    def sample(self, n):
        r"""

        Sample array indices using weighted sampling, where the sample weights are proprotional to
        the values stored in :attr:`values`.

        Parameters
        ----------
        n : positive int

            The number of samples to return.

        Returns
        -------
        idx : array of ints

            The sampled indices, shape: (n,)

        Warning
        -------

        This method presumes (but doesn't check) that all :attr:`values` stored in the tree are
        non-negative.

        """
        if not (isinstance(n, int) and n > 0):
            raise TypeError("n must be a positive integer")

        return self.inverse_cdf(self._rnd.rand(n))

    def inverse_cdf(self, u):
        r"""

        Inverse of the cumulative distribution function (CDF) of the categorical distribution
        :math:`\text{Cat}(p)`, where :math:`p` are the normalized values :math:`p_i=`
        :attr:`values[i] / sum(values) <values>`.

        This function provides the machinery for the :attr:`sample` method.

        Parameters
        ----------
        u : float or 1d array of floats

            One of more numbers :math:`u\in[0,1]`. These are typically sampled from
            :math:`\text{Unif([0, 1])}`.

        Returns
        -------
        idx : array of ints

            The indices associated with :math:`u`, shape: (n,)

        Warning
        -------

        This method presumes (but doesn't check) that all :attr:`values` stored in the tree are
        non-negative.

        """
        # NOTE: This is an iterative implementation, which is a lot uglier than a recursive one.
        # The reason why we use an iterative approach is that it's easier for batch-processing.
        if self.root_value <= 0:
            raise RuntimeError("the root_value must be positive")

        # init (will be updated in loop)
        u, isscalar = self._check_u(u)
        values = u * self.root_value
        idx = np.zeros_like(values, dtype='int32')  # this is ultimately what we'll returned
        level_offset_parent = 0                      # number of nodes in levels above parent

        # iterate down, from the root to the leaves
        for level in range(1, self.height):

            # get child indices
            level_offset = 2 ** level - 1
            left_child_idx = (idx - level_offset_parent) * 2 + level_offset
            right_child_idx = left_child_idx + 1

            # update (idx, values, level_offset_parent)
            left_child_values = self._arr[left_child_idx]
            pick_left_child = left_child_values > values
            idx = np.where(pick_left_child, left_child_idx, right_child_idx)
            values = np.where(pick_left_child, values, values - left_child_values)
            level_offset_parent = level_offset

        idx = idx - level_offset_parent
        return idx[0] if isscalar else idx

    def _check_u(self, u):
        """ some boilerplate to check validity of 'u' array """
        isscalar = False
        if isinstance(u, (float, int)):
            u = np.array([u], dtype='float')
            isscalar = True
        if isinstance(u, list) and all(isinstance(x, (float, int)) for x in u):
            u = np.asarray(u, dtype='float')
        if not (isinstance(u, np.ndarray)
                and u.ndim == 1 and np.issubdtype(u.dtype, np.floating)):
            raise TypeError("'u' must be a float or a 1d array of floats")
        if np.any(u > 1) or np.any(u < 0):
            raise ValueError("all values in 'u' must lie in the unit interval [0, 1]")
        return u, isscalar

class JNPStorageAdv:
    """
    A simple storage class.
    """
    def __init__(self, capacity, obs_shape, n_rewards):
        self.capacity = capacity
        self.obses = jnp.empty((capacity, *obs_shape), dtype=jnp.float32)
        self.next_obses = jnp.empty((capacity, *obs_shape), dtype=jnp.float32)
        self.actions = jnp.empty((capacity, 1), dtype=jnp.float32)
        self.rewards = jnp.empty((capacity, n_rewards), dtype=jnp.float32)
        self.not_dones = jnp.empty((capacity, 1), dtype=jnp.float32)
        self.idxs = jnp.arange(capacity, dtype=jnp.int32)
        self.advantages = jnp.empty((capacity,), dtype=jnp.float32)

    def __len__(self):
        return len(self.obses)

    def __getitem__(self, idx):
        return TransitionsAdv(self.obses[idx], self.actions[idx], self.rewards[idx], self.next_obses[idx], self.not_dones[idx], self.idxs[idx], self.advantages[idx])
    
    def __setitem__(self, idx, value: TransitionsAdv):
        s, a, r, s_next, d, _, Adv = value
        self.obses = self.obses.at[idx].set(s)
        self.actions = self.actions.at[idx].set(a)
        self.rewards = self.rewards.at[idx].set(r)
        self.next_obses = self.next_obses.at[idx].set(s_next)
        self.not_dones = self.not_dones.at[idx].set(not d)
        self.advantages = self.advantages.at[idx].set(Adv)

    def __repr__(self):
        return f"{type(self).__name__}(capacity={self.capacity})"

class PrioritizedReplayBuffer:
    def __init__(self, obs_shape, n_rewards, capacity, alpha=1.0, beta=1.0, epsilon=1e-4, seed=0):
        self.capacity = capacity
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._epsilon = float(epsilon)
        self.idx = 0
        self.full = False

        self._random_seed = seed
        self._rnd = np.random.RandomState(seed)

        self._storage = JNPStorageAdv(capacity=self.capacity, obs_shape=obs_shape, n_rewards=n_rewards)
        self._sumtree = SumTree(capacity=self.capacity, random_seed=self._random_seed)

    def __len__(self):
        return min(self.capacity, self.idx)

    @property
    def alpha(self):
        return self._alpha
    
    @property
    def beta(self):
        return self._beta
    
    @property
    def epsilon(self):
        return self._epsilon
    
    @alpha.setter
    def alpha(self, new_alpha):
        raise NotImplementedError("Verify that this works as expected")
        if not (isinstance(new_alpha, (float, int)) and new_alpha > 0):
            raise TypeError(f"alpha must be a positive float, got: {new_alpha}")
        if onp.isclose(new_alpha, self._alpha, rtol=0.01):
            return  # noop if new value is too close to old value (not worth the computation cost)
        new_values = onp.where(
            self._sumtree.values <= 0, 0.,  # only change exponents for positive values
            onp.exp(onp.log(onp.maximum(self._sumtree.values, 1e-15)) * (new_alpha / self._alpha)))
        self._sumtree.set_values(..., new_values)
        self._alpha = float(new_alpha)

    @beta.setter
    def beta(self, new_beta):
        raise NotImplementedError("Verify that this works as expected")
        if not (isinstance(new_beta, (float, int)) and new_beta > 0):
            raise TypeError(f"beta must be a positive float, got: {new_beta}")
        self._beta = float(new_beta)


    @epsilon.setter
    def epsilon(self, new_epsilon):
        raise NotImplementedError("Verify that this works as expected")
        if not (isinstance(new_epsilon, (float, int)) and new_epsilon > 0):
            raise TypeError(f"epsilon must be a positive float, got: {new_epsilon}")
        self._epsilon = float(new_epsilon)
    

    def add(self, transition: Transitions, Adv):
        """
        Add a transition to the replay buffer.
        
        inputs:
            transition (Transitions): transition(s, a, r, s', not_done)
            Adv (jnp.ndarray): advantage. This is used to compute the priority of the transition.
        """
        self._storage[self.idx] = TransitionsAdv(*transition, self.idx, Adv)
        self._sumtree.set_values(self.idx, jnp.power(jnp.abs(Adv) + self.epsilon, self.alpha))
        self.idx = (self.idx + 1) % self.capacity

    def add_batch(self, transitions: Transitions, Advs):
        # TODO: https://github.com/coax-dev/coax/blob/cf855957b4d7f5cda731e0873b098afe0b069011/coax/experience_replay/_prioritized.py#L146
        raise NotImplementedError

    def update(self, idx, Adv):
        idx = jnp.asarray(idx, dtype='int32')
        Adv = jnp.asarray(Adv, dtype='float32')
        # chex.assert_equal_shape([idx, Adv])
        # chex.assert_rank([idx, Adv], 1)

        idx_lookup = idx % self.capacity  # wrap around
        new_values = jnp.where(
            self._storage[idx_lookup].idxs == idx,  # only update if ids match
            jnp.power(jnp.abs(Adv) + self.epsilon, self.alpha),
            self._sumtree.values[idx_lookup])
        
        self._sumtree.set_values(idx_lookup, new_values)

    def sample(self, batch_size):
        idx = self._sumtree.sample(n=batch_size)
        P = self._sumtree.values[idx] / self._sumtree.root_value  # prioritized, biased propensities
        W = jnp.power(P * len(self), -self.beta)                  # inverse propensity weights (β≈1)
        W /= W.max()  # for stability, ensure only down-weighting (see sec. 3.4 of arxiv:1511.05952)

        transition_batch = self._storage[idx]
        return transition_batch, W
    
    def __iter__(self):
        return iter(self._storage[:len(self)])
    
    def __repr__(self):
        return f"{type(self).__name__}(capacity={self.capacity}, idx={self.idx}, full={self.full})"
    
    def save(self, path):
        raise NotImplementedError
    
    def load(self, path, reset_capacity=True):
        raise NotImplementedError
    



if __name__ == "__main__":
    buffer = PrioritizedReplayBuffer(obs_shape=(4,), n_rewards=1, capacity=1000, alpha=0.6, beta=0.4, epsilon=1e-4, seed=0)

    dummy_transition = Transitions(jnp.zeros((4,)), jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((4,)), jnp.zeros((1,)))
    
    
    for i in range(10):
        dummy_Adv = jnp.array(np.random.rand())
        buffer.add(dummy_transition, dummy_Adv)

    # Test update
    batch = buffer.sample(4)
    idxs = batch[0].idxs
    print(f"Previous priorities for idxs {idxs}: \n{buffer._sumtree.values[idxs]}")
    new_td_error = jnp.array(np.random.rand(4))
    buffer.update(idxs, new_td_error)
    print(f"New priorities for idxs {idxs}: \n{buffer._sumtree.values[idxs]}")
