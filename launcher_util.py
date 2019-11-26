import datetime
import json
import os
import os.path as osp
import pickle
import random
import sys
import time
from collections import namedtuple

import dateutil.tz
import numpy as np

from utils.logging import logger
from utils.pytorch_util import set_gpu_mode
from utils.rng import set_seed
from utils.pythonplusplus import load_gzip_pickle, load_pkl, dump_pkl

import torch
import gzip

GitInfo = namedtuple(
    'GitInfo',
    [
        'directory',
        'code_diff',
        'code_diff_staged',
        'commit_hash',
        'branch_name',
    ],
)


def get_git_infos(dirs):
    try:
        import git
        git_infos = []
        for directory in dirs:
            # Idk how to query these things, so I'm just doing try-catch
            try:
                repo = git.Repo(directory)
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = '[DETACHED]'
                git_infos.append(GitInfo(
                    directory=directory,
                    code_diff=repo.git.diff(None),
                    code_diff_staged=repo.git.diff('--staged'),
                    commit_hash=repo.head.commit.hexsha,
                    branch_name=branch_name,
                ))
            except git.exc.InvalidGitRepositoryError as e:
                print("Not a valid git repo: {}".format(directory))
    except ImportError:
        git_infos = None
    return git_infos


def run_experiment_here(
        experiment_function,
        variant,
        seed=None,
        use_gpu=True,
        gpu_id=0,

        # Logger params:
        snapshot_mode='last',
        snapshot_gap=1,

        force_randomize_seed=False,
        log_dir=None,
):
    """
    Run an experiment locally without any serialization.

    :param experiment_function: Function. `variant` will be passed in as its
    only argument.
    :param exp_prefix: Experiment prefix for the save file.
    :param variant: Dictionary passed in to `experiment_function`.
    :param exp_id: Experiment ID. Should be unique across all
    experiments. Note that one experiment may correspond to multiple seeds,.
    :param seed: Seed used for this experiment.
    :param use_gpu: Run with GPU. By default False.
    :param script_name: Name of the running script
    :param log_dir: If set, set the log directory to this. Otherwise,
    the directory will be auto-generated based on the exp_prefix.
    :return:
    """
    torch.set_num_threads(1)

    if force_randomize_seed or seed is None:
        seed = random.randint(0, 100000)
        variant['seed'] = str(seed)

    log_dir = variant['log_dir']

    # The logger's default mode is to
    # append to the text file if the file already exists
    # So this would not override and erase any existing
    # log file in the same log dir.
    logger.reset()
    setup_logger(
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        log_dir=log_dir,
    )

    # Assume this file is at the top level of the repo
    git_infos = get_git_infos([osp.dirname(__file__)])

    run_experiment_here_kwargs = dict(
        variant=variant,
        seed=seed,
        use_gpu=use_gpu,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        git_infos=git_infos,
    )

    exp_setting = dict(
        run_experiment_here_kwargs=run_experiment_here_kwargs
    )

    exp_setting_pkl_path = osp.join(log_dir, 'experiment.pkl')

    # Check if existing result exists
    prev_exp_state = None

    if osp.isfile(exp_setting_pkl_path):
        # Sanity check to make sure the experimental setting
        # of the saved data and the current experiment run is the same
        prev_exp_setting = load_pkl(exp_setting_pkl_path)

        logger.log(f'Log dir is not empty: {os.listdir(log_dir)}')

        if prev_exp_setting != exp_setting:
            logger.log("""Previous experimental setting is not
                        the same as the current experimental setting.
                        Very risky to try to reload the previous state.
                        Exitting""")
            logger.log(f'Previous: {prev_exp_setting}')
            logger.log(f'Current: {exp_setting}')
            exit(1)

        try:
            prev_exp_state = load_gzip_pickle(
                osp.join(log_dir, 'params.zip_pkl'))

            logger.log('Trying to restore the state of the experiment program')

        except FileNotFoundError:
            logger.log("""There is no previous experiment state available.
                            Do not try to restore.""")

            prev_exp_state = None

    # Log the variant
    logger.log("Variant:")
    logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
    variant_log_path = osp.join(log_dir, 'variant.json')
    logger.log_variant(variant_log_path, variant)

    # Save the current experimental setting
    dump_pkl(exp_setting_pkl_path, exp_setting)
    log_git_infos(git_infos, log_dir)

    logger.log(f'Seed: {seed}')
    set_seed(seed)

    logger.log(f'Using GPU: {use_gpu}')
    set_gpu_mode(use_gpu, gpu_id)

    return experiment_function(variant, prev_exp_state)


def log_git_infos(git_infos, log_dir):

    for (
        directory, code_diff, code_diff_staged, commit_hash, branch_name
    ) in git_infos:
        if directory[-1] == '/':
            directory = directory[:-1]
        diff_file_name = directory[1:].replace("/", "-") + ".patch"
        diff_staged_file_name = (
            directory[1:].replace("/", "-") + "_staged.patch"
        )
        if code_diff is not None and len(code_diff) > 0:
            with open(osp.join(log_dir, diff_file_name), "w") as f:
                f.write(code_diff + '\n')
        if code_diff_staged is not None and len(code_diff_staged) > 0:
            with open(osp.join(log_dir, diff_staged_file_name), "w") as f:
                f.write(code_diff_staged + '\n')
        with open(osp.join(log_dir, "git_infos.txt"), "a") as f:
            f.write("directory: {}\n".format(directory))
            f.write("git hash: {}\n".format(commit_hash))
            f.write("git branch name: {}\n\n".format(branch_name))


def setup_logger(
    log_dir,
    text_log_file="debug.log",
    tabular_log_file="progress.csv",
    log_tabular_only=False,

    snapshot_mode="last",
    snapshot_gap=1,
):

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    logger.add_tabular_output(tabular_log_path)

    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)

    logger.log(f'Logging to: {log_dir}')


def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for
                   k, v in data.items())
    return False
