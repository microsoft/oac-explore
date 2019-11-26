import random
import numpy as np
import torch


def get_global_pkg_rng_state():

    rng = dict()

    rng['py_rng_state'] = random.getstate()
    rng['np_rng_state'] = np.random.get_state()
    rng['t_cpu_rng_state'] = torch.get_rng_state()

    if torch.cuda.is_available():
        rng['t_gpu_rng_state'] = torch.cuda.get_rng_state_all()

    return rng


def set_global_pkg_rng_state(rng_states):

    random.setstate(rng_states['py_rng_state'])

    np.random.set_state(rng_states['np_rng_state'])

    torch.set_rng_state(rng_states['t_cpu_rng_state'])

    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_states['t_gpu_rng_state'])


def set_seed(seed):
    """
    Set the seed for all the possible random number generators
    for global packages.

    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
