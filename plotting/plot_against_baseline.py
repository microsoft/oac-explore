# %%
import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import copy
import csv

from main import get_cmd_args, get_log_dir
from utils.env_utils import domain_to_epoch


plt.rcParams['font.size'] = '12'


def get_one_domain_one_run_res(domain, seed, hyper_params):

    args = get_cmd_args()

    args.base_log_dir = RLKIT_BASE_LOG_DIR
    args.domain = domain
    args.seed = seed

    for k, v in hyper_params.items():
        setattr(args, k, v)

    res_path = get_log_dir(args)

    csv_path = osp.join(
        res_path, 'progress.csv'
    )

    values = []

    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)

        col_names = next(reader)

        # Assume that the index of epoch is the last one
        # Not sure why the csv file is missing one col header
        # epoch_col_idx = col_names.index('Epoch')
        epoch_col_idx = -1
        val_col_idx = col_names.index('remote_evaluation/Average Returns')

        for row in reader:

            # If this equals Epoch, it means the header
            # was written to the csv file again
            # and we reset everything
            if row[epoch_col_idx] == 'Epoch':
                values = []
                continue

            epoch = int(row[epoch_col_idx])
            val = float(row[val_col_idx])

            # We need to check if the row contains the values
            # of the correct epoch
            # because after reloading from checkpoint,
            # we are writing the result to the same csv file
            if epoch == len(values):
                values.append(val)
            else:
                # print(
                    # f'Reloaded row found at epoch {len(values), epoch} found for', domain, seed, hyper_params)
                pass

    # Reshape the return value
    # to accomodate downstream api
    values = np.array(values)
    values = np.expand_dims(values, axis=-1)

    return values


def get_one_domain_all_run_res(domain, run_idxes, hyper_params):

    results = []

    for idx in run_idxes:
        res = get_one_domain_one_run_res(domain, idx, hyper_params)
        results.append(res)

    min_rows = min([len(col) for col in results])
    results = [col[0:min_rows] for col in results]

    results = np.hstack(results)

    return results


def smooth_results(results, smoothing_window=100):
    smoothed = np.zeros((results.shape[0], results.shape[1]))

    for idx in range(len(smoothed)):

        if idx == 0:
            smoothed[idx] = results[idx]
            continue

        start_idx = max(0, idx - smoothing_window)

        smoothed[idx] = np.mean(results[start_idx:idx], axis=0)

    return smoothed


def plot(values, label, color=[0, 0, 1, 1]):
    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1)

    x_vals = np.arange(len(mean))

    blur = copy.deepcopy(color)
    blur[-1] = 0.1

    plt.plot(x_vals, mean, label=label, color=color)
    plt.fill_between(x_vals, mean - std, mean + std, color=blur)

    plt.legend()


# DOMAINS = ['humanoid', 'halfcheetah', 'hopper', 'ant', 'walker2d']
DOMAINS = ['humanoid']

RLKIT_BASE_LOG_DIR_BASELINE = RLKIT_BASE_LOG_DIR_ALGO = './data'

RUN_IDXES = list([i for i in range(5)])
NUM_RUN = len(RUN_IDXES)


SAC_ONE_RETRAINING_PARAMS = dict(
    delta=0.0,
    beta_UB=0.0,
    num_expl_steps_per_train_loop=1000,
    num_trains_per_train_loop=1000
)


SAVE_FIG = True

print('SAVE_FIG', SAVE_FIG)

FORMAL_FIG = True

print('FORMAL_FIG', FORMAL_FIG)


def sac_get_one_domain_one_run_res(path, domain, seed):

    csv_path = osp.join(
        path, domain, f'seed_{seed}', 'progress.csv'
    )

    result = pd.read_csv(csv_path, usecols=[
        'remote_evaluation/Average Returns'])

    return result.values


def sac_plot_one_retraining_step(domain):

    args = get_cmd_args()

    set_attr_with_dict(args, SAC_ONE_RETRAINING_PARAMS)

    results = get_one_domain_all_run_res(
        domain, RUN_IDXES, SAC_ONE_RETRAINING_PARAMS)
    results = smooth_results(results)

    if FORMAL_FIG:
        label = 'Soft Actor Critic'
    else:
        label = 'SAC'

    plot(results, label=label)


def sac_plot(domain, num_trains_per_train_loop):

    if num_trains_per_train_loop == 1000:
        sac_plot_one_retraining_step(domain)

    elif num_trains_per_train_loop == 4000:
        sac_plot_four_retraining_step(domain)

    else:
        exit('Unrecognized environment setting')


def get_plot_title(args):

    if FORMAL_FIG:
        title = args.env

    else:

        title = '\n'.join([
            args.env,
            f'num_run: {NUM_RUN}', '---'

            f'beta_UB: {args.beta_UB}',
            f'delta: {args.delta}',
            f'train/env step ratio: {int(args.num_trains_per_train_loop / args.num_expl_steps_per_train_loop)}'
        ])

    return title


all_hyper_params_dict = [
    dict(
        delta=23.53,
        beta_UB=4.66,
        num_expl_steps_per_train_loop=1000,
        num_trains_per_train_loop=1000
    ),
]


def set_attr_with_dict(target, source_dict):

    for k, v in source_dict.items():
        setattr(target, k, v)

    return target


def get_tick_space(domain):

    if domain == 'Hopper':
        return 200

    if domain == 'humanoid':
        return 1000

    return 500


for hyper_params in all_hyper_params_dict:

    for domain in DOMAINS:

        plt.clf()

        """
        Set up
        """
        # We need to do this so that jupyter notebook
        # works with argparse
        import sys
        sys.argv = ['']
        del sys

        args = get_cmd_args()

        set_attr_with_dict(args, hyper_params)

        args.env = f'{domain}-v2'

        relative_log_dir = get_log_dir(
            args, should_include_base_log_dir=False, should_include_seed=False, should_include_domain=False)

        graph_base_path = osp.join(
            RLKIT_BASE_LOG_DIR_ALGO, 'plot', relative_log_dir)

        os.makedirs(graph_base_path, exist_ok=True)

        """
        Obtain Result
        """
        RLKIT_BASE_LOG_DIR = RLKIT_BASE_LOG_DIR_ALGO

        results = get_one_domain_all_run_res(domain, RUN_IDXES, hyper_params)
        results = smooth_results(results)

        if domain == 'humanoid' and FORMAL_FIG:
            mean = np.mean(results, axis=1)
            x_vals = np.arange(len(mean))

            # This is the index where OAC has
            # the same performance as SAC with 10 million steps
            # Plus 200 so that we are not overstating our claim
            magic_idx = np.argmax(mean > 8000) + 300

            plt.plot(8000 * np.ones(magic_idx), linestyle='--',
                     color=[0, 0, 1, 1], linewidth=3, label='Soft Actor Critic 10 million steps performance')
            plt.vlines(x=magic_idx,
                       ymin=0, ymax=8000, linestyle='--',
                       color=[0, 0, 1, 1],)

        """
        Plot result
        """

        plot(results, label='Optimistic Actor Critic',
             color=[1.0, 0.0, 0.0, 1.0])

        RLKIT_BASE_LOG_DIR = RLKIT_BASE_LOG_DIR_BASELINE

        sac_plot(domain, args.num_trains_per_train_loop)

        plt.title(get_plot_title(args))

        plt.ylabel('Average Episode Return')

        xticks = np.arange(0, domain_to_epoch(
            domain) + 1, get_tick_space(domain))

        plt.xticks(xticks, xticks / 1000.0)

        plt.xlabel('Number of environment steps in millions')
        plt.legend()

        if not SAVE_FIG:
            plt.show()

        else:

            fig_path = osp.join(
                graph_base_path, f'{args.env}_formal_fig_{FORMAL_FIG}.png')

            plt.savefig(fig_path, bbox_inches='tight')

            print(f'Saved fig at {fig_path}')

    print('Finish plotting for: ', hyper_params)


# %%
