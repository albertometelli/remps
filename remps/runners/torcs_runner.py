from multiprocessing import Queue

import numpy as np

from remps.envs.confmdp import ConfMDP
from remps.policy.policy import Policy
from remps.sampler.torcs_sampler import TorcsSampler


def collect_data(
    env: ConfMDP,
    policy: Policy,
    total_n_samples: int,
    n_params: int,
    initial_port: int,
    num_processes: int = 20,
):
    nb_samples_per_worker = total_n_samples // num_processes

    inputQs = [Queue() for _ in range(num_processes)]
    outputQ = Queue()
    workers = [
        TorcsSampler(
            policy,
            inputQs[i],
            outputQ,
            env.action_space_size,
            env.observation_space_size,
            i + initial_port,
            nb_samples_per_worker,
            n_params,
        )
        for i in range(num_processes)
    ]
    # Run the Workers
    for w in workers:
        w.start()

    X = None
    Y = None
    # Collect results when ready
    avg_rews_all = []
    returns_all = []
    for i in range(num_processes):
        _, pair = outputQ.get()
        x, y, avg_rew, returns = pair
        avg_rews_all.extend(avg_rew)
        returns_all.extend(returns)
        if X is None:
            X = x
            Y = y
        else:
            X = np.vstack((X, x))
            Y = np.vstack((Y, y))

    return X, Y, avg_rews_all, returns_all
