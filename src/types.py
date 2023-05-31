from collections import namedtuple

Model = namedtuple('Model', ['net', 'opt'])

Transitions = namedtuple('Transitions', ['states', 'actions', 'rewards', 'next_states', 'not_dones'])
TransitionsAdv = namedtuple('Transitions', ['states', 'actions', 'rewards', 'next_states', 'not_dones', 'idxs', 'advantages'])
Trajectory = namedtuple('Trajectory', ['states', 'actions', 'rewards', 'next_states', 'dones', 'valid_masks', 'cum_return'])

class Env:
    def reset_env(self, params_true):
        raise NotImplementedError
    
    def transition(self, params_true, state, action, aux_info):
        raise NotImplementedError

    def is_done(self, params_true, state):
        raise NotImplementedError
    
