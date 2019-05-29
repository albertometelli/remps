import argparse
import os.path
from datetime import datetime

import numpy as np

# log
from baselines import logger
from baselines.common.misc_util import set_global_seeds

import remps.runners.remps_runner as remps_runner
from remps.envs.cartpole import CartPole
from remps.envs.chain import NChainEnv
from remps.envs.torcs.torcs import Torcs
from remps.model_approx.cartpole_model_action_noise import (
    CartPoleModel as CartPoleActionNoise,
)
from remps.model_approx.chain_model import ChainModel
from remps.model_approx.nn_model import NNModel
from remps.model_approx.torcs_model_simple import TorcsModel
from remps.policy.discrete import Discrete
from remps.policy.gaussian import Gaussian
from remps.policy.one_parameter_policy import OneParameterPolicy
from remps.utils.utils import boolean_flag

# Default Arguments
EVALUATION_STEPS = 10
HIDDEN_LAYER_SIZE = 3
EVAL_FREQ = 50
N_TRAJECTORIES = 100
ITERATION_NUMBER = 500
MAX_STEPS = 500


def runExp(
    checkpoint_file,
    logdir,
    omega,
    random_init,
    max_steps,
    hidden_layer_size,
    n_trajectories,
    file_suffix,
    restore_variables,
    overwrite_log,
    env_id,
    seed,
    exact,
    **kwargs,
):

    set_global_seeds(seed)

    # setup environments and policy
    if env_id == 1:
        env = CartPole(max_steps=max_steps)
        env_name = "cartPole"
        policy = Discrete(
            env.observation_space.shape[0], env.action_space.n, hidden_layer_size
        )
        model_approx = CartPoleActionNoise()
        if not exact:
            model_approx = NNModel(env.observation_space_size, 1, name=env_name)
    elif env_id == 2:
        env = Torcs(visual=False, port=kwargs["initial_port"])
        policy = Gaussian(
            env.observation_space_size,
            env.action_space_size,
            hidden_layer_size=hidden_layer_size,
        )
        env_name = "TORCS"
        model_approx = TorcsModel(
            env.observation_space_size,
            env.action_space_size,
            name=env_name + "2_actions",
        )

    elif env_id == 3:
        env = NChainEnv(max_steps=max_steps)
        env_name = "chain"

        # initialize policy parameter
        if not random_init:
            init_theta = 0.2
        else:
            init_theta = np.random.rand()
        policy = OneParameterPolicy(init_theta=init_theta)

        model_approx = ChainModel()
    else:
        raise ValueError("Wrong environment index")

    # initialize environment parameters
    if random_init:
        omega_bounds = env.get_params_bounds()
        omega = np.random.uniform(low=omega_bounds[:, 0], high=omega_bounds[:, 1])
    env.set_params(omega)

    algo_name = "REMPS"
    experiment_name = (
        algo_name
        + "/"
        + env_name
        + "-omega"
        + str(omega)
        + "-traj"
        + str(n_trajectories)
        + "-DualReg"
        + str(kwargs["dual_reg"])
        + "PolReg-"
        + str(kwargs["policy_reg"])
        + "TrainingSet"
        + str(kwargs["training_set_size"])
    )

    if exact:
        experiment_name = experiment_name + "exact"
    experiment_name += str(seed)

    if file_suffix is not None:
        experiment_name = experiment_name + "-" + file_suffix

    if logdir is None:
        logdir = (
            "tf_logs/model_policy_logs/"
            + experiment_name
            + "eps-"
            + str(kwargs["epsilon"])
        )

    now = datetime.now()

    # do not overwrite log files
    if os.path.isdir(logdir) and (not overwrite_log):
        logdir = logdir + "-" + now.strftime("%Y%m%d-%H%M%S") + "/"

    if checkpoint_file is None:
        experiment_name = (
            "model-policy/" + experiment_name + "eps-" + str(kwargs["epsilon"])
        )
        checkpoint_file = "tf_checkpoint/" + experiment_name + "/"

    if not restore_variables:
        # do not overwrite checkpoint files
        if os.path.isdir(checkpoint_file):
            checkpoint_file = checkpoint_file[:-1] + now.strftime("%Y%m%d-%H%M%S") + "/"
        else:
            os.makedirs(checkpoint_file)

    checkpoint_file += "model.ckpt"

    print("Logs will be saved into: " + logdir)
    print("Checkpoints will be saved into: " + checkpoint_file)

    remps_runner.train(
        env=env,
        policy=policy,
        model_approximator=model_approx,
        n_trajectories=n_trajectories,
        checkpoint_file=checkpoint_file,
        logdir=logdir,
        omega=omega,
        restore_variables=restore_variables,
        exact=exact,
        **kwargs,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    boolean_flag(parser, "restore-variables", default=False)
    boolean_flag(parser, "save-variables", default=True)
    parser.add_argument("--checkpoint-file", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eval-steps", type=int, default=EVALUATION_STEPS)
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--hidden-layer-size", type=int, default=HIDDEN_LAYER_SIZE)
    parser.add_argument(
        "--n-trajectories",
        help="Number of trajectories to collect before performing the training step",
        type=int,
        default=N_TRAJECTORIES,
    )
    parser.add_argument(
        "--iteration-number",
        help="Number of training steps",
        type=int,
        default=ITERATION_NUMBER,
    )
    parser.add_argument(
        "--max-steps",
        help="Max number of steps of a trajectory",
        type=int,
        default=MAX_STEPS,
    )
    parser.add_argument(
        "--omega", help="Initial configurable parameters", type=float, default=9
    )
    boolean_flag(
        parser, "random-init", help="If true randomly initialize omega", default=False
    )
    parser.add_argument(
        "--start-from-iteration",
        help="Iteration number from which to start",
        type=int,
        default=0,
    )
    boolean_flag(
        parser,
        "overwrite-log",
        help="If true overwrite logs in the folder",
        default=False,
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        help="suffix to add at the end of the file name",
        default=None,
    )
    parser.add_argument(
        "--env-id", type=int, help="1 cartpole, 2 torcs, 3 chain", default=0
    )
    parser.add_argument("--title", type=str, default="policy")
    parser.add_argument("--seed", type=int, default=1000)
    boolean_flag(
        parser,
        "exact",
        help="whether the environment model is exact or approximated",
        default=True,
    )
    # -----------------------------------------------------
    # --------------- REMPS PARAMETER ---------------------
    # -----------------------------------------------------
    parser.add_argument(
        "--kappa", type=float, default=1e-3, help="Constraint on KL divergence"
    )
    parser.add_argument(
        "--dual-reg", type=float, default=0.0, help="Dual Regularization"
    )
    parser.add_argument(
        "--policy-reg", type=float, default=0.0, help="Policy Regularization"
    )
    parser.add_argument(
        "--training-set-size", type=int, default=500, help="Training set size"
    )
    boolean_flag(parser, "normalize-data", default=True)

    args = parser.parse_args()
    dict_args = vars(args)

    print(f"Running experiment with settings: {dict_args}")
    return dict_args


if __name__ == "__main__":
    args = parse_args()
    logger.configure()
    # Run actual script.
    runExp(**args)
