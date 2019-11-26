from collections import deque, OrderedDict
import torch

from utils.env_utils import env_producer
from utils.eval_util import create_stats_ordered_dict
from utils.rng import get_global_pkg_rng_state, set_global_pkg_rng_state
import numpy as np
import ray
from optimistic_exploration import get_optimistic_exploration_action


class MdpPathCollector(object):
    def __init__(
            self,
            env,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):

        # The class state which we do not expect to mutate
        if render_kwargs is None:
            render_kwargs = {}
        self._render = render
        self._render_kwargs = render_kwargs
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved

        # The class mutable internal state
        self._env = env
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            policy,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            optimistic_exploration=False,
            optimistic_exploration_kwargs={}
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = rollout(
                self._env,
                policy,
                max_path_length=max_path_length_this_loop,
                optimistic_exploration=optimistic_exploration,
                optimistic_exploration_kwargs=optimistic_exploration_kwargs
            )
            path_len = len(path['actions'])
            if (
                    # incomplete path
                    path_len != max_path_length and

                    # that did not end in a terminal state
                    not path['terminals'][-1] and

                    # and we should discard such path
                    discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env_mj_state=self._env.sim.get_state(),
            env_rng=self._env.np_random.get_state(),

            _epoch_paths=self._epoch_paths,
            _num_steps_total=self._num_steps_total,
            _num_paths_total=self._num_paths_total
        )

    def restore_from_snapshot(self, ss):

        self._env.sim.set_state(ss['env_mj_state'])
        self._env.np_random.set_state(ss['env_rng'])

        self._epoch_paths = ss['_epoch_paths']
        self._num_steps_total = ss['_num_steps_total']
        self._num_paths_total = ss['_num_paths_total']


@ray.remote(num_cpus=1)
class RemoteMdpPathCollector(MdpPathCollector):

    def __init__(self,
                 domain_name, env_seed, policy_producer,
                 max_num_epoch_paths_saved=None,
                 render=False,
                 render_kwargs=None,
                 ):

        torch.set_num_threads(1)

        env = env_producer(domain_name, env_seed)

        self._policy_producer = policy_producer

        super().__init__(env,
                         max_num_epoch_paths_saved=max_num_epoch_paths_saved,
                         render=render,
                         render_kwargs=render_kwargs,
                         )

    def async_collect_new_paths(self,
                                max_path_length,
                                num_steps,
                                discard_incomplete_paths,

                                deterministic_pol,
                                pol_state_dict):

        if deterministic_pol:
            policy = self._policy_producer(deterministic=True)
            policy.stochastic_policy.load_state_dict(pol_state_dict)

        else:
            policy = self._policy_producer()
            policy.load_state_dict(pol_state_dict)

        self.collect_new_paths(policy,
                               max_path_length, num_steps,
                               discard_incomplete_paths)

    def get_global_pkg_rng_state(self):
        return get_global_pkg_rng_state()

    def set_global_pkg_rng_state(self, state):
        set_global_pkg_rng_state(state)


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        optimistic_exploration=False,
        optimistic_exploration_kwargs={},
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:

        if not optimistic_exploration:
            a, agent_info = agent.get_action(o)
        else:
            a, agent_info = get_optimistic_exploration_action(
                o, **optimistic_exploration_kwargs)

        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
