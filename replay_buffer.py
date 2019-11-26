from collections import OrderedDict
from utils.env_utils import get_dim
from gym.spaces import Discrete

import numpy as np


class ReplayBuffer(object):

    def __init__(
        self,
        max_replay_buffer_size,
        ob_space,
        action_space,
    ):
        """
        The class state which should not mutate
        """

        self._ob_space = ob_space
        self._action_space = action_space

        ob_dim = get_dim(self._ob_space)
        ac_dim = get_dim(self._action_space)

        self._max_replay_buffer_size = max_replay_buffer_size

        """
        The class mutable state
        """

        self._observations = np.zeros((max_replay_buffer_size, ob_dim))

        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, ob_dim))
        self._actions = np.zeros((max_replay_buffer_size, ac_dim))

        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))

        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        self._top = 0
        self._size = 0

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):

        assert not isinstance(self._action_space, Discrete)

        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        self._advance()

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        return batch

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    def end_epoch(self, epoch):
        return

    def get_snapshot(self):
        return dict(
            _observations=self._observations,
            _next_obs=self._next_obs,
            _actions=self._actions,
            _rewards=self._rewards,
            _terminals=self._terminals,
            _top=self._top,
            _size=self._size,
        )

    def restore_from_snapshot(self, ss):

        for key in ss.keys():
            assert hasattr(self, key)
            setattr(self, key, ss[key])
