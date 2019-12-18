import abc
from collections import OrderedDict

from torch import nn as nn
from utils.logging import logger
import utils.eval_util as eval_util
from utils.rng import get_global_pkg_rng_state
import utils.pytorch_util as ptu

import gtimer as gt
from replay_buffer import ReplayBuffer
from path_collector import MdpPathCollector, RemoteMdpPathCollector
from tqdm import trange

import ray
import torch
import numpy as np
import random


class BatchRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_data_collector: MdpPathCollector,
            remote_eval_data_collector: RemoteMdpPathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            optimistic_exp_hp=None,
    ):
        super().__init__()

        """
        The class state which should not mutate
        """
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.optimistic_exp_hp = optimistic_exp_hp

        """
        The class mutable state
        """
        self._start_epoch = 0

        """
        This class sets up the main training loop, so it needs reference to other
        high level objects in the algorithm

        But these high level object maintains their own states
        and has their own responsibilities in saving and restoring their state for checkpointing
        """
        self.trainer = trainer

        self.expl_data_collector = exploration_data_collector
        self.remote_eval_data_collector = remote_eval_data_collector

        self.replay_buffer = replay_buffer

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):

        # Fill the replay buffer to a minimum before training starts
        if self.min_num_steps_before_training > self.replay_buffer.num_steps_can_sample():
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.trainer.policy,
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                trange(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):

            # To evaluate the policy remotely,
            # we're shipping the policy params to the remote evaluator
            # This can be made more efficient
            # But this is currently extremely cheap due to small network size
            pol_state_dict = ptu.state_dict_cpu(self.trainer.policy)

            remote_eval_obj_id = self.remote_eval_data_collector.async_collect_new_paths.remote(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,

                deterministic_pol=True,
                pol_state_dict=pol_state_dict)

            gt.stamp('remote evaluation submit')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.trainer.policy,
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,

                    optimistic_exploration=self.optimistic_exp_hp['should_use'],
                    optimistic_exploration_kwargs=dict(
                        policy=self.trainer.policy,
                        qfs=[self.trainer.qf1, self.trainer.qf2],
                        hyper_params=self.optimistic_exp_hp
                    )
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)

            # Wait for eval to finish
            ray.get([remote_eval_obj_id])
            gt.stamp('remote evaluation wait')

            self._end_epoch(epoch)

    def _end_epoch(self, epoch):
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        ray.get([self.remote_eval_data_collector.end_epoch.remote(epoch)])

        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        # We can only save the state of the program
        # after we call end epoch on all objects with internal state.
        # This is so that restoring from the saved state will
        # lead to identical result as if the program was left running.
        if epoch > 0:
            snapshot = self._get_snapshot(epoch)
            logger.save_itr_params(epoch, snapshot)
            gt.stamp('saving')

        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)

        write_header = True if epoch == 0 else False
        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=write_header)

    def _get_snapshot(self, epoch):
        snapshot = dict(
            trainer=self.trainer.get_snapshot(),
            exploration=self.expl_data_collector.get_snapshot(),
            evaluation_remote=ray.get(
                self.remote_eval_data_collector.get_snapshot.remote()),
            evaluation_remote_rng_state=ray.get(
                self.remote_eval_data_collector.get_global_pkg_rng_state.remote()
            ),
            replay_buffer=self.replay_buffer.get_snapshot()
        )

        # What epoch indicates is that at the end of this epoch,
        # The state of the program is snapshot
        # Not to be consfused with at the beginning of the epoch
        snapshot['epoch'] = epoch

        # Save the state of various rng
        snapshot['global_pkg_rng_state'] = get_global_pkg_rng_state()

        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Remote Evaluation
        """
        logger.record_dict(
            ray.get(self.remote_eval_data_collector.get_diagnostics.remote()),
            prefix='remote_evaluation/',
        )
        remote_eval_paths = ray.get(
            self.remote_eval_data_collector.get_epoch_paths.remote())
        logger.record_dict(
            eval_util.get_generic_path_information(remote_eval_paths),
            prefix="remote_evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging')

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times
